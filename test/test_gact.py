# In[]
import sys, os
sys.path.append('../')
sys.path.append('../src/')


import numpy as np
import pandas as pd
import networkx as nx
import time

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from umap import UMAP
from scipy.stats import pearsonr, spearmanr

import TI as ti
import benchmark as bmk
import utils as utils
import dataset
import model as model
import post_align as palign
import diffusion_dist as diff
import train


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams["font.size"] = 20

# In[] Calculate the scores for gene activity module
scores = pd.DataFrame(columns = ["dataset"])

seeds = [0, 1, 2]
latent_dims = [4, 8, 32]
reg_ds = [1, 10]
reg_gs = [0.01, 1, 10]
reg_mmds = [1, 10, 20, 30]

latent_dim = latent_dims[eval(sys.argv[1])]
reg_d = reg_ds[eval(sys.argv[2])]
reg_g = reg_gs[eval(sys.argv[3])]
# harder to merge, need to make mmd loss larger
reg_mmd = reg_mmds[eval(sys.argv[4])]

learning_rate = 3e-4
n_epochs = 500
ts = [30, 50, 70]
use_potential = True
norm = "l1"

for data_name in ["paired_lin1", "paired_lin2", "paired_lin3", "paired_lin4", "paired_lin5", "paired_lin6"]: 
    print(data_name)

    # load data
    counts_rna = pd.read_csv("../data/all_simulations/" + data_name + "/GxC1.txt", sep = "\t", header = None).values.T
    counts_rna2 = pd.read_csv("../data/all_simulations/" + data_name + "/GxC2.txt", sep = "\t", header = None).values.T
    counts_atac = pd.read_csv("../data/all_simulations/" + data_name + "/RxC2.txt", sep = "\t", header = None).values.T
    label_rna = pd.read_csv("../data/all_simulations/" + data_name + "/cell_label1.txt", sep = "\t")["pop"].values.squeeze()
    label_atac = pd.read_csv("../data/all_simulations/" + data_name + "/cell_label2.txt", sep = "\t")["pop"].values.squeeze()
    pt_rna = pd.read_csv("../data/all_simulations/" + data_name + "/pseudotime1.txt", header = None).values.squeeze()
    pt_atac = pd.read_csv("../data/all_simulations/" + data_name + "/pseudotime2.txt", header = None).values.squeeze()

    # preprocessing
    libsize = 100
    counts_rna = counts_rna/np.sum(counts_rna, axis = 1)[:, None] * libsize
    counts_rna = np.log1p(counts_rna)
    counts_rna2 = counts_rna2/np.sum(counts_rna2, axis = 1)[:, None] * libsize
    counts_rna2 = np.log1p(counts_rna2)

    rna_dataset = dataset.dataset(counts = counts_rna, anchor = np.argsort(pt_rna)[:10])
    atac_dataset = dataset.dataset(counts = counts_atac, anchor = np.argsort(pt_atac)[:10])
    coarse_reg = torch.FloatTensor(pd.read_csv("../data/all_simulations/" + data_name + "/region2gene.txt", sep = "\t", header = None).values).to(device)

    batch_size = int(max([len(rna_dataset),len(atac_dataset)])/4)

    train_rna_loader = DataLoader(rna_dataset, batch_size = batch_size, shuffle = True)
    train_atac_loader = DataLoader(atac_dataset, batch_size = batch_size, shuffle = True)

    # baseline methods
    pseudo_rna = (atac_dataset.counts.to(device) @ coarse_reg).detach().cpu().numpy()
    real_rna = counts_rna2
    mse = np.sum((pseudo_rna - real_rna) ** 2)/pseudo_rna.shape[0]
    mse_norm = np.sum((pseudo_rna/np.sum(pseudo_rna, axis = 1, keepdims = True) - real_rna/np.sum(real_rna, axis = 1, keepdims = True)) ** 2)/pseudo_rna.shape[0]
    pearson = sum([pearsonr(pseudo_rna[i,:], real_rna[i,:])[0] for i in range(pseudo_rna.shape[0])])/pseudo_rna.shape[0]
    spearman = sum([spearmanr(pseudo_rna[i,:], real_rna[i,:])[0] for i in range(pseudo_rna.shape[0])])/pseudo_rna.shape[0]
    
    scores = scores.append({
                "dataset": data_name,
                "model": "linear", 
                "latent_dim": 0,
                "reg_d": 0,
                "reg_g": 0,
                "reg_mmd": 0,
                "norm": 0,
                "seed": 0,      
                "mse": mse,
                "mse_norm": mse_norm,
                "pearson": pearson,
                "spearman": spearman
            }, ignore_index = True)
            
    for seed in seeds: 
        # not using anchor
        results_dir = "results_acc/" 
        if not os.path.exists(results_dir + data_name + "/"):
            print("make directory")
            os.makedirs(results_dir + data_name + "/")

        EMBED_CONFIG = {
            "gact_layers": [atac_dataset.counts.shape[1], 1024, 512, rna_dataset.counts.shape[1]], 
            "proj_layers": [rna_dataset.counts.shape[1], 512, 128, latent_dim], # number of nodes in each 
            "learning_rate": learning_rate,
            "n_epochs": n_epochs + 1,
            "use_anchor": False,
            "reg_d": reg_d,
            "reg_g": reg_g,
            "reg_mmd": reg_mmd,
            "l_dist_type": "kl",
            "device": device
        }

        start_time = time.time()
        # calculate the diffusion distance
        dist_rna = diff.diffu_distance(rna_dataset.counts.numpy(), ts = ts,
                                    use_potential = use_potential, dr = "pca", n_components = 30)

        dist_atac = diff.diffu_distance(atac_dataset.counts.numpy(), ts = ts,
                                        use_potential = use_potential, dr = "lsi", n_components = 30)

        # quantile normalization
        # dist_atac = diff.quantile_norm(dist_atac, reference = dist_rna.reshape(-1), replace = True)

        dist_rna = dist_rna/np.linalg.norm(dist_rna)
        dist_atac = dist_atac/np.linalg.norm(dist_atac)
        dist_rna = torch.FloatTensor(dist_rna).to(device)
        dist_atac = torch.FloatTensor(dist_atac).to(device)

        # initialize the model
        gene_act = model.gene_act(features = EMBED_CONFIG["gact_layers"], dropout_rate = 0.0, negative_slope = 0.2).to(device)
        encoder = model.Encoder(features = EMBED_CONFIG["proj_layers"], dropout_rate = 0.0, negative_slope = 0.2).to(device)
        model_dict = {"gene_act": gene_act, "encoder": encoder}

        opt_genact = torch.optim.Adam(gene_act.parameters(), lr = learning_rate)
        opt_encoder = torch.optim.Adam(encoder.parameters(), lr = learning_rate)
        opt_dict = {"gene_act": opt_genact, "encoder": opt_encoder}
        
        # training models
        train.match_latent(model = model_dict, opts = opt_dict, dist_atac = dist_atac, dist_rna = dist_rna, 
                        data_loader_rna = train_rna_loader, data_loader_atac = train_atac_loader, n_epochs = EMBED_CONFIG["n_epochs"], 
                        reg_mtx = coarse_reg, reg_d = EMBED_CONFIG["reg_d"], reg_g = EMBED_CONFIG["reg_g"], reg_mmd = EMBED_CONFIG["reg_mmd"], use_anchor = EMBED_CONFIG["use_anchor"], norm = norm, 
                        mode = EMBED_CONFIG["l_dist_type"])

        torch.save(model_dict, results_dir + data_name + "/model_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + ".pth")

        # prediction accuracy
        pseudo_rna = model_dict["gene_act"](atac_dataset.counts.to(device)).detach().cpu().numpy()
        mse = np.sum((pseudo_rna - real_rna) ** 2)/pseudo_rna.shape[0]
        mse_norm = np.sum((pseudo_rna/np.sum(pseudo_rna, axis = 1, keepdims = True) - real_rna/np.sum(real_rna, axis = 1, keepdims = True)) ** 2)/pseudo_rna.shape[0]
        pearson = sum([pearsonr(pseudo_rna[i,:], real_rna[i,:])[0] for i in range(pseudo_rna.shape[0])])/pseudo_rna.shape[0]
        spearman = sum([spearmanr(pseudo_rna[i,:], real_rna[i,:])[0] for i in range(pseudo_rna.shape[0])])/pseudo_rna.shape[0]

        scores = scores.append({
            "dataset": data_name,
            "model": "scDART", 
            "latent_dim": latent_dim,
            "reg_d": reg_d,
            "reg_g": reg_g,
            "reg_mmd": reg_mmd,
            "norm": norm,
            "seed": seed,            
            "mse": mse,
            "mse_norm": mse_norm,
            "pearson": pearson,
            "spearman": spearman
        }, ignore_index = True)

        # using anchor        
        results_dir = "results_acc_anchor/"
        if not os.path.exists(results_dir + data_name + "/"):
            print("make directory")
            os.makedirs(results_dir + data_name + "/")

        EMBED_CONFIG["use_anchor"] = True
        gene_act = model.gene_act(features = EMBED_CONFIG["gact_layers"], dropout_rate = 0.0, negative_slope = 0.2).to(device)
        encoder = model.Encoder(features = EMBED_CONFIG["proj_layers"], dropout_rate = 0.0, negative_slope = 0.2).to(device)
        model_dict = {"gene_act": gene_act, "encoder": encoder}

        opt_genact = torch.optim.Adam(gene_act.parameters(), lr = learning_rate)
        opt_encoder = torch.optim.Adam(encoder.parameters(), lr = learning_rate)
        opt_dict = {"gene_act": opt_genact, "encoder": opt_encoder}
        
        # training models
        train.match_latent(model = model_dict, opts = opt_dict, dist_atac = dist_atac, dist_rna = dist_rna, 
                        data_loader_rna = train_rna_loader, data_loader_atac = train_atac_loader, n_epochs = EMBED_CONFIG["n_epochs"], 
                        reg_mtx = coarse_reg, reg_d = EMBED_CONFIG["reg_d"], reg_g = EMBED_CONFIG["reg_g"], reg_mmd = EMBED_CONFIG["reg_mmd"], use_anchor = EMBED_CONFIG["use_anchor"], norm = norm, 
                        mode = EMBED_CONFIG["l_dist_type"])

        torch.save(model_dict, results_dir + data_name + "/model_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + ".pth")

        # prediction accuracy
        pseudo_rna = model_dict["gene_act"](atac_dataset.counts.to(device)).detach().cpu().numpy()
        mse = np.sum((pseudo_rna - real_rna) ** 2)/pseudo_rna.shape[0]
        mse_norm = np.sum((pseudo_rna/np.sum(pseudo_rna, axis = 1, keepdims = True) - real_rna/np.sum(real_rna, axis = 1, keepdims = True)) ** 2)/pseudo_rna.shape[0]
        pearson = sum([pearsonr(pseudo_rna[i,:], real_rna[i,:])[0] for i in range(pseudo_rna.shape[0])])/pseudo_rna.shape[0]
        spearman = sum([spearmanr(pseudo_rna[i,:], real_rna[i,:])[0] for i in range(pseudo_rna.shape[0])])/pseudo_rna.shape[0]

        scores = scores.append({
            "dataset": data_name,
            "model": "scDART-anchor", 
            "latent_dim": latent_dim,
            "reg_d": reg_d,
            "reg_g": reg_g,
            "reg_mmd": reg_mmd,
            "norm": norm,
            "seed": seed,            
            "mse": mse,
            "mse_norm": mse_norm,
            "pearson": pearson,
            "spearman": spearman
        }, ignore_index = True)

scores.to_csv("results_acc/geneact_score.csv")

# In[]
import seaborn as sns
fig = plt.figure(figsize = (7, 5))
ax = fig.subplots(nrows = 1, ncols = 1)
scores = pd.read_csv("results_acc/geneact_score.csv", index_col = 0)
sns.violinplot(data = scores, x = "model", y = "mse_norm", ax = ax)
ax.legend(loc='upper left', frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
ax.set_ylabel("MSE (normalized)")
ax.set_xticklabels(["Linear", "scDART", "scDART-anchor"])
fig.savefig("results_acc/geact.png", bbox_inches = "tight")

# %%
