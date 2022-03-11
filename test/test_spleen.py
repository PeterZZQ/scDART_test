# In[]
from re import S
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
import utils as utils
import post_align as palign

from umap import UMAP
from scipy.sparse import load_npz
from scipy.stats import pearsonr, spearmanr
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams["font.size"] = 20

# In[] scan and find the one with the highest neighborhood overlap score
seeds = [0, 1, 2]
latent_dims = [4, 8, 32]
reg_ds = [1, 10]
reg_gs = [0.01, 1, 10]
reg_mmds = [1, 10, 20, 30]

latent_dim = latent_dims[2]
reg_d = reg_ds[0]
reg_g = reg_gs[1]
# harder to merge, need to make mmd loss larger
reg_mmd = reg_mmds[0]
seed = seeds[0]

learning_rate = 3e-4
n_epochs = 500
use_anchor = True
ts = [10]
# ts = [30, 50, 70]
use_potential = True
norm = "l1"

path = "../../../CFRM/data/real/diag/Xichen/"
counts_rna = np.array(load_npz(path + "small_ver/GxC1_small.npz").T.todense())# [::10,:]
counts_atac = np.array(load_npz(path + "small_ver/RxC2_small.npz").T.todense())
# preprocessing
counts_rna = counts_rna/np.sum(counts_rna, axis = 1, keepdims = True) * 100
counts_atac = (counts_atac>0).astype(int)
counts_rna = np.log1p(counts_rna)

label_rna = pd.read_csv(path + "meta_c1.csv", index_col = 0)["cell_type"].values.squeeze()#[::10]
label_atac = pd.read_csv(path + "meta_c2.csv", index_col = 0)["cell_type"].values.squeeze()

# rna_dataset = dataset.dataset(counts = counts_rna, anchor = np.where(label_rna == "B_follicular"))
# atac_dataset = dataset.dataset(counts = counts_atac, anchor = np.where(label_atac == "B_follicular"))
rna_dataset = dataset.dataset(counts = counts_rna, anchor = np.where(label_rna == "T_CD4_naive"))
atac_dataset = dataset.dataset(counts = counts_atac, anchor = np.where(label_atac == "T_CD4_naive"))
coarse_reg = torch.FloatTensor(load_npz(path + "small_ver/GxR_small.npz").todense()).T.to(device)

batch_size = int(max([len(rna_dataset),len(atac_dataset)])/4)

print("Random seed: " + str(seed))
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

train_rna_loader = DataLoader(rna_dataset, batch_size = batch_size, shuffle = True)
train_atac_loader = DataLoader(atac_dataset, batch_size = batch_size, shuffle = True)

EMBED_CONFIG = {
    'gact_layers': [atac_dataset.counts.shape[1], 512, 256, rna_dataset.counts.shape[1]], 
    'proj_layers': [rna_dataset.counts.shape[1], 128, 32, latent_dim], # number of nodes in each 
    'learning_rate': learning_rate,
    'n_epochs': n_epochs + 1,
    'use_anchor': use_anchor,
    'reg_d': reg_d,
    'reg_g': reg_g,
    'reg_mmd': reg_mmd,
    'l_dist_type': 'kl',
    'device': device
}
print(rna_dataset.counts.shape)
print(atac_dataset.counts.shape)
# calculate the diffusion distance
dist_rna = diff.diffu_distance(rna_dataset.counts.numpy(), ts = ts,
                                use_potential = use_potential, dr = "pca", n_components = 30)

dist_atac = diff.diffu_distance(atac_dataset.counts.numpy(), ts = ts,
                                use_potential = use_potential, dr = "lsi", n_components = 30)

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

with torch.no_grad():
    z_rna = model_dict["encoder"](rna_dataset.counts.to(device))
    z_atac = model_dict["encoder"](model_dict["gene_act"](atac_dataset.counts.to(device)))

np.save(file = "../test/results_spleen/z_rna.npy", arr = z_rna.cpu().numpy())
np.save(file = "../test/results_spleen/z_atac.npy", arr = z_atac.cpu().numpy())

z_rna = np.load(file = "./results_spleen/z_rna.npy")
z_atac = np.load(file = "./results_spleen/z_atac.npy")
z_rna = torch.FloatTensor(z_rna)
z_atac = torch.FloatTensor(z_atac)


# model_dict = torch.load("../test/results_snare/models_1000/model_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + ".pth", map_location = device)

pca_op = PCA(n_components = 2)
z = pca_op.fit_transform(np.concatenate((z_rna.cpu().numpy(), z_atac.cpu().numpy()), axis = 0))
z_rna_pca = z[:z_rna.shape[0],:]
z_atac_pca = z[z_rna.shape[0]:,:]

umap_op = UMAP(n_components = 2, min_dist = 0.1, random_state = 0)
z = umap_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
z_rna_umap = z[:z_rna.shape[0],:]
z_atac_umap = z[z_rna.shape[0]:,:]

utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_umap, anno1 = label_rna, 
                    anno2 = label_atac, mode = "separate", save = "results_spleen/z_joint.png", 
                    figsize = (25,7), axis_label = "PCA")
utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_umap, anno1 = label_rna, 
                    anno2 = label_atac, mode = "modality", save = "results_spleen/z_mod.png", 
                    figsize = (15,7), axis_label = "PCA")

utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = label_rna, 
                    anno2 = label_atac, mode = "separate", save = "results_spleen/z_joint_umap.png", 
                    figsize = (25,7), axis_label = "UMAP")
utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = label_rna, 
                    anno2 = label_atac, mode = "modality", save = "results_spleen/z_mod_umap.png", 
                    figsize = (15,7), axis_label = "UMAP")


z_atac, z_rna = palign.match_alignment(z_rna = z_atac.cpu(), z_atac = z_rna.cpu(), k = 10)
z_rna, z_atac = palign.match_alignment(z_rna = z_rna.cpu(), z_atac = z_atac.cpu(), k = 10)
z_rna = z_rna.cpu().numpy()
z_atac = z_atac.cpu().numpy()

# post-maching
pca_op = PCA(n_components = 2)
z = pca_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
z_rna_pca = z[:z_rna.shape[0],:]
z_atac_pca = z[z_rna.shape[0]:,:]
utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = label_rna, 
                    anno2 = label_atac, mode = "separate", save = "results_spleen/z_joint_post.png", 
                    figsize = (25,7), axis_label = "PCA")
utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = label_rna, 
                    anno2 = label_atac, mode = "modality", save = "results_spleen/z_mod_post.png", 
                    figsize = (15,7), axis_label = "PCA")

umap_op = UMAP(n_components = 2, min_dist = 0.1, random_state = 0)
z = umap_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
z_rna_umap = z[:z_rna.shape[0],:]
z_atac_umap = z[z_rna.shape[0]:,:]
utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = label_rna, 
                    anno2 = label_atac, mode = "separate", save = "results_spleen/z_joint_post_umap.png", 
                    figsize = (25,7), axis_label = "UMAP")
utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = label_rna, 
                    anno2 = label_atac, mode = "modality", save = "results_spleen/z_mod_post_umap.png", 
                    figsize = (15,7), axis_label = "UMAP")

# %%
