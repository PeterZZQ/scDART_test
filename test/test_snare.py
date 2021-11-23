# In[]
import sys, os
sys.path.append('../')
sys.path.append('../src/')


import numpy as np
import pandas as pd
from scipy import sparse
import networkx as nx

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time

from sklearn.decomposition import PCA
from sklearn.manifold import MDS

import diffusion_dist as diff
import dataset as dataset
import model as model
import loss as loss
import train
import TI as ti
import benchmark as bmk
import de_analy as de

from umap import UMAP

import utils as utils

import post_align as palign
from scipy.sparse import load_npz
from scipy.stats import pearsonr, spearmanr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams["font.size"] = 20

# In[] scan and find the one with the highest neighborhood overlap score
seeds = [0, 1, 2]
latent_dims = [4, 8, 32]
reg_ds = [1, 10]
reg_gs = [0.01, 1, 10]
reg_mmds = [1, 10, 15, 20, 30]
'''
latent_dim = latent_dims[eval(sys.argv[1])]
reg_d = reg_ds[eval(sys.argv[2])]
reg_g = reg_gs[eval(sys.argv[3])]
# harder to merge, need to make mmd loss larger
reg_mmd = reg_mmds[eval(sys.argv[4])]
seed = seeds[eval(sys.argv[5])]
'''
learning_rate = 3e-4
n_epochs = 500
use_anchor = False
ts = [30, 50, 70]
use_potential = True

rna_dataset = dataset.braincortex_rna(counts_dir = "../data/snare-seq/counts_rna.csv", 
                                    anno_dir = "../data/snare-seq/anno.txt",anchor = None)
atac_dataset = dataset.braincortex_atac(counts_dir = "../data/snare-seq/counts_atac.npz", 
                                        anno_dir = "../data/snare-seq/anno.txt",anchor = None)
coarse_reg = torch.FloatTensor(load_npz("../data/snare-seq/gact.npz").T.todense()).to(device)  

batch_size = int(max([len(rna_dataset),len(atac_dataset)])/4)
libsize = rna_dataset.get_libsize()

scores = pd.DataFrame(columns = ["model", "latent_dim", "reg_d", "reg_g", "reg_mmd", "seed", "neigh_overlap", "mse", "mse_norm", "pearson"])

# baseline methods
pseudo_rna = (atac_dataset.counts.to(device) @ coarse_reg).detach().cpu().numpy()
real_rna = rna_dataset.counts.detach().cpu().numpy()
mse = np.sum((pseudo_rna - real_rna) ** 2)/pseudo_rna.shape[0]
mse_norm = np.sum((pseudo_rna/np.sum(pseudo_rna, axis = 1, keepdims = True) - real_rna/np.sum(real_rna, axis = 1, keepdims = True)) ** 2)/pseudo_rna.shape[0]
pearson = sum([pearsonr(pseudo_rna[i,:], real_rna[i,:])[0] for i in range(pseudo_rna.shape[0])])/pseudo_rna.shape[0]
spearman = sum([spearmanr(pseudo_rna[i,:], real_rna[i,:])[0] for i in range(pseudo_rna.shape[0])])/pseudo_rna.shape[0]
scores = scores.append({
            "model": "linear", 
            "latent_dim": None,
            "reg_d": None,
            "reg_g": None,
            "reg_mmd": None,
            "seed": None,
            "neigh_overlap": None,
            "mse": mse,
            "mse_norm": mse_norm,
            "pearson": pearson,
            "spearman": spearman
        }, ignore_index = True)

for latent_dim in [latent_dims[0]]:
    for reg_d in [reg_ds[0]]:
        for reg_g in reg_gs:
            for reg_mmd in reg_mmds:
                for seed in seeds:
                    print("Random seed: " + str(seed))
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    np.random.seed(seed)

                    train_rna_loader = DataLoader(rna_dataset, batch_size = batch_size, shuffle = True)
                    train_atac_loader = DataLoader(atac_dataset, batch_size = batch_size, shuffle = True)

                    EMBED_CONFIG = {
                        'gact_layers': [atac_dataset.counts.shape[1], 1024, 512, rna_dataset.counts.shape[1]], 
                        'proj_layers': [rna_dataset.counts.shape[1], 512, 128, latent_dim], # number of nodes in each 
                        'learning_rate': learning_rate,
                        'n_epochs': n_epochs + 1,
                        'use_anchor': use_anchor,
                        'reg_d': reg_d,
                        'reg_g': reg_g,
                        'reg_mmd': reg_mmd,
                        'l_dist_type': 'kl',
                        'device': device
                    }
                    '''
                    # calculate the diffusion distance
                    dist_rna = diff.diffu_distance(rna_dataset.counts.numpy(), ts = ts,
                                                    use_potential = use_potential, dr = "pca", n_components = 30)

                    dist_atac = diff.diffu_distance(atac_dataset.counts.numpy(), ts = ts,
                                                    use_potential = use_potential, dr = "lsi", n_components = 30)

                    dist_rna = dist_rna/np.linalg.norm(dist_rna)
                    dist_atac = dist_atac/np.linalg.norm(dist_atac)
                    dist_rna = torch.FloatTensor(dist_rna).to(device)
                    dist_atac = torch.FloatTensor(dist_atac).to(device)
                    '''
                    # initialize the model
                    gene_act = model.gene_act(features = EMBED_CONFIG["gact_layers"], dropout_rate = 0.0, negative_slope = 0.2).to(device)
                    encoder = model.Encoder(features = EMBED_CONFIG["proj_layers"], dropout_rate = 0.0, negative_slope = 0.2).to(device)
                    model_dict = {"gene_act": gene_act, "encoder": encoder}
                    '''
                    opt_genact = torch.optim.Adam(gene_act.parameters(), lr = learning_rate)
                    opt_encoder = torch.optim.Adam(encoder.parameters(), lr = learning_rate)
                    opt_dict = {"gene_act": opt_genact, "encoder": opt_encoder}

                    # training models
                    train.match_latent(model = model_dict, opts = opt_dict, dist_atac = dist_atac, dist_rna = dist_rna, 
                                    data_loader_rna = train_rna_loader, data_loader_atac = train_atac_loader, n_epochs = EMBED_CONFIG["n_epochs"], 
                                    reg_mtx = coarse_reg, reg_d = EMBED_CONFIG["reg_d"], reg_g = EMBED_CONFIG["reg_g"], reg_mmd = EMBED_CONFIG["reg_mmd"], use_anchor = EMBED_CONFIG["use_anchor"], norm = "l2", 
                                    mode = EMBED_CONFIG["l_dist_type"])

                    with torch.no_grad():
                        z_rna = model_dict["encoder"](rna_dataset.counts.to(device)).cpu().detach()
                        z_atac = model_dict["encoder"](model_dict["gene_act"](atac_dataset.counts.to(device))).cpu().detach()

                    np.save(file = "../test/results_snare/z_rna_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + ".npy", arr = z_rna.numpy())
                    np.save(file = "../test/results_snare/z_atac_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + ".npy", arr = z_atac.numpy())
                    torch.save(model_dict, "../test/results_snare/model_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + ".pth")
                    '''

                    model_dict = torch.load("../test/results_snare/model_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_l1_quant.pth", map_location = device)
                    with torch.no_grad():
                        z_rna = model_dict["encoder"](rna_dataset.counts.to(device)).cpu().detach()
                        z_atac = model_dict["encoder"](model_dict["gene_act"](atac_dataset.counts.to(device))).cpu().detach()

                    # calculate the neighborhood overlap score
                    n_overlap = bmk.neigh_overlap(z_rna, z_atac, k = 50)
                    print("latent_dim: {:d}, reg_d: {:.4f}, reg_g: {:.4f}, reg_mmd: {:.4f}, seed: {:d}".format(latent_dim, reg_d, reg_g, reg_mmd, seed))
                    # print("\tneighborhood overlap score: {:.4f}".format(n_overlap))

                    # prediction accuracy
                    pseudo_rna = model_dict["gene_act"](atac_dataset.counts.to(device)).detach().cpu().numpy()
                    mse = np.sum((pseudo_rna - real_rna) ** 2)/pseudo_rna.shape[0]
                    mse_norm = np.sum((pseudo_rna/np.sum(pseudo_rna, axis = 1, keepdims = True) - real_rna/np.sum(real_rna, axis = 1, keepdims = True)) ** 2)/pseudo_rna.shape[0]
                    pearson = sum([pearsonr(pseudo_rna[i,:], real_rna[i,:])[0] for i in range(pseudo_rna.shape[0])])/pseudo_rna.shape[0]
                    spearman = sum([spearmanr(pseudo_rna[i,:], real_rna[i,:])[0] for i in range(pseudo_rna.shape[0])])/pseudo_rna.shape[0]
                    
                    # print("\tMean square error: {:.4f}, normalized MSE: {:.4f}".format(mse, mse_norm))
                    
                    # # plot histogram
                    # fig = plt.figure(figsize = (10,7))
                    # ax = fig.add_subplot()
                    # ax.hist(pseudo_rna.detach().cpu().numpy().reshape(-1))
                    # ax.hist(rna_dataset.counts.numpy().reshape(-1))
                    # fig.savefig("results_snare/hist.png", bbox_inches = "tight")

                    scores = scores.append({
                                "model": "scDART", 
                                "latent_dim": latent_dim,
                                "reg_d": reg_d,
                                "reg_g": reg_g,
                                "reg_mmd": reg_mmd,
                                "seed": seed,
                                "neigh_overlap": n_overlap,
                                "mse": mse,
                                "mse_norm": mse_norm,
                                "pearson": pearson,
                                "spearman": spearman
                            }, ignore_index = True)

scores.to_csv("results_snare/scores_l1_quantile.csv")
# In[]
scores = pd.read_csv("results_snare/scores_l1.csv")
import seaborn as sns
fig = plt.figure(figsize = (30,7))
axs = fig.subplots(nrows = 1, ncols = 3)
sns.boxplot(data = scores, x = "reg_g", y = "mse", hue = "reg_mmd", ax = axs[0])
sns.boxplot(data = scores, x = "reg_g", y = "mse_norm", hue = "reg_mmd", ax = axs[1])
sns.boxplot(data = scores, x = "reg_g", y = "pearson", hue = "reg_mmd", ax = axs[2])
plt.tight_layout()


scores = pd.read_csv("results_snare/scores_l2.csv")
import seaborn as sns
fig = plt.figure(figsize = (30,7))
axs = fig.subplots(nrows = 1, ncols = 3)
sns.boxplot(data = scores, x = "reg_g", y = "mse", hue = "reg_mmd", ax = axs[0])
sns.boxplot(data = scores, x = "reg_g", y = "mse_norm", hue = "reg_mmd", ax = axs[1])
sns.boxplot(data = scores, x = "reg_g", y = "pearson", hue = "reg_mmd", ax = axs[2])
plt.tight_layout()

scores = pd.read_csv("results_snare/scores_l1_quantile.csv")
import seaborn as sns
fig = plt.figure(figsize = (30,7))
axs = fig.subplots(nrows = 1, ncols = 3)
sns.boxplot(data = scores, x = "reg_g", y = "mse", hue = "reg_mmd", ax = axs[0])
sns.boxplot(data = scores, x = "reg_g", y = "mse_norm", hue = "reg_mmd", ax = axs[1])
sns.boxplot(data = scores, x = "reg_g", y = "pearson", hue = "reg_mmd", ax = axs[2])
plt.tight_layout()


fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot()
sns.boxplot(data = scores, x = "reg_g", y = "neigh_overlap", hue = "reg_mmd", ax = ax)
plt.tight_layout()

'''
# extract gene activity matrix (region, gene)
GACT = train.infer_gact(model_dict["gene_act"], mask = (coarse_reg != 0), thresh = 1e-6).cpu().numpy()
# transform into (motif, gene)
region2motif = pd.read_csv("../data/snare-seq/chromVAR/region2motif.csv", sep = ",", index_col = 0).values
motif2gene = region2motif.T @ GACT
# check which motif is regulating which gene
'''

# %%
