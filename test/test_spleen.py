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
import bmk_discrete as bmk
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

# In[] Run UnionCom
from unioncom import UnionCom
uc = UnionCom.UnionCom(epoch_pd = 10000)
integrated_data = uc.fit_transform([counts_rna, counts_atac])
z_rna = integrated_data[0]
z_atac = integrated_data[1]

np.save(file = "results_spleen/unioncom/unioncom_rna.npy", arr = z_rna)
np.save(file = "results_spleen/unioncom/unioncom_atac.npy", arr = z_atac)

pca_op = PCA(n_components = 2)
z = pca_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
z_rna_pca = z[:z_rna.shape[0],:]
z_atac_pca = z[z_rna.shape[0]:,:]

utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = label_rna, anno2 = label_atac, 
                  mode = "separate", save = "results_spleen/unioncom/unioncom_pca.png", 
                  figsize = (25,7), axis_label = "PCA")

utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = label_rna, anno2 = label_atac, 
                  mode = "modality", save = "results_spleen/unioncom/unioncom_mod_pca.png", 
                  figsize = (15,7), axis_label = "PCA")

umap_op = UMAP(n_components = 2, min_dist = 0.1, random_state = 0)
z = umap_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
z_rna_umap = z[:z_rna.shape[0],:]
z_atac_umap = z[z_rna.shape[0]:,:]

utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = label_rna, anno2 = label_atac, 
                  mode = "separate", save = "results_spleen/unioncom/unioncom_umap.png", 
                  figsize = (25,7), axis_label = "UMAP")


utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = label_rna, anno2 = label_atac, 
                  mode = "modality", save = "results_spleen/unioncom/unioncom_mod_umap.png", 
                  figsize = (15,7), axis_label = "UMAP")

# In[]
# scDART
z_rna_scdart = np.load(file = "./results_spleen/z_rna.npy")
z_atac_scdart = np.load(file = "./results_spleen/z_atac.npy")
# unioncom
z_rna_unioncom = np.load(file = "./results_spleen/unioncom/unioncom_rna.npy")
z_atac_unioncom = np.load(file = "./results_spleen/unioncom/unioncom_atac.npy")
# seurat
z_rna_seurat = pd.read_csv("./results_spleen/seurat/seurat_pca_c1.txt", sep = "\t", index_col = 0)
z_atac_seurat = pd.read_csv("./results_spleen/seurat/seurat_pca_c2.txt", sep = "\t", index_col = 0)
# liger
z_rna_liger = pd.read_csv("./results_spleen/liger/liger_c1_norm.csv", sep = ",", index_col = 0)
z_atac_liger = pd.read_csv("./results_spleen/liger/liger_c2_norm.csv", sep = ",", index_col = 0)

umap_op = UMAP(n_components = 2, min_dist = 0.1, random_state = 0)
z = umap_op.fit_transform(np.concatenate((z_rna_unioncom, z_atac_unioncom), axis = 0))
z_rna_umap = z[:z_rna_scdart.shape[0],:]
z_atac_umap = z[z_rna_scdart.shape[0]:,:]

utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = label_rna, anno2 = label_atac, 
                  mode = "joint", save = "results_spleen/unioncom/joint_umap.png", 
                  figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = label_rna, anno2 = label_atac, 
                  mode = "modality", save = "results_spleen/unioncom/mod_umap.png", 
                  figsize = (15,7), axis_label = "UMAP")

z = umap_op.fit_transform(np.concatenate((z_rna_seurat, z_atac_seurat), axis = 0))
z_rna_umap = z[:z_rna_scdart.shape[0],:]
z_atac_umap = z[z_rna_scdart.shape[0]:,:]

utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = label_rna, anno2 = label_atac, 
                  mode = "joint", save = "results_spleen/seurat/joint_umap.png", 
                  figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = label_rna, anno2 = label_atac, 
                  mode = "modality", save = "results_spleen/seurat/mod_umap.png", 
                  figsize = (15,7), axis_label = "UMAP")

z = umap_op.fit_transform(np.concatenate((z_rna_liger, z_atac_liger), axis = 0))
z_rna_umap = z[:z_rna_scdart.shape[0],:]
z_atac_umap = z[z_rna_scdart.shape[0]:,:]

utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = label_rna, anno2 = label_atac, 
                  mode = "joint", save = "results_spleen/liger/joint_umap.png", 
                  figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = label_rna, anno2 = label_atac, 
                  mode = "modality", save = "results_spleen/liger/mod_umap.png", 
                  figsize = (15,7), axis_label = "UMAP")

# In[]
# silhouette score
silhouette_scdart = bmk.silhouette_batch(X = np.concatenate((z_rna_scdart, z_atac_scdart), axis = 0), group_gt = np.concatenate([label_rna, label_atac], axis = 0), batch_gt = np.array([0] * label_rna.shape[0] + [1] * label_atac.shape[0]))
print('silhouette (scDART): {:.3f}'.format(silhouette_scdart))

silhouette_seurat = bmk.silhouette_batch(X = np.concatenate((z_rna_seurat, z_atac_seurat), axis = 0), group_gt = np.concatenate([label_rna, label_atac], axis = 0), batch_gt = np.array([0] * label_rna.shape[0] + [1] * label_atac.shape[0]))
print('silhouette (Seurat): {:.3f}'.format(silhouette_seurat))

silhouette_liger = bmk.silhouette_batch(X = np.concatenate((z_rna_liger, z_atac_liger), axis = 0), group_gt = np.concatenate([label_rna, label_atac], axis = 0), batch_gt = np.array([0] * label_rna.shape[0] + [1] * label_atac.shape[0]))
print('silhouette (Liger): {:.3f}'.format(silhouette_liger))

silhouette_unioncom = bmk.silhouette_batch(X = np.concatenate((z_rna_unioncom, z_atac_unioncom), axis = 0), group_gt = np.concatenate([label_rna, label_atac], axis = 0), batch_gt = np.array([0] * label_rna.shape[0] + [1] * label_atac.shape[0]))
print('silhouette (UNIONCOM): {:.3f}'.format(silhouette_unioncom))

n_neighbors = 50
# 1. scdart
gc_scdart = bmk.graph_connectivity(X = np.concatenate((z_rna_scdart, z_atac_scdart), axis = 0), groups = np.concatenate([label_rna, label_atac], axis = 0), k = n_neighbors)
print('GC (scDART): {:.3f}'.format(gc_scdart))

# 2. Seurat
gc_seurat = bmk.graph_connectivity(X = np.concatenate((z_rna_seurat, z_atac_seurat), axis = 0), groups = np.concatenate([label_rna, label_atac], axis = 0), k = n_neighbors)
print('GC (Seurat): {:.3f}'.format(gc_seurat))

# 3. Liger
gc_liger = bmk.graph_connectivity(X = np.concatenate((z_rna_liger, z_atac_liger), axis = 0), groups = np.concatenate([label_rna, label_atac], axis = 0), k = n_neighbors)
print('GC (Liger): {:.3f}'.format(gc_liger))

# 4. unioncom
gc_unioncom = bmk.graph_connectivity(X = np.concatenate((z_rna_unioncom, z_atac_unioncom), axis = 0), groups = np.concatenate([label_rna, label_atac], axis = 0), k = n_neighbors)
print('GC (UNIONCOM): {:.3f}'.format(gc_unioncom))

# Conservation of biological identity
# NMI and ARI
# 1. scdart
nmi_scdart = []
ari_scdart = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_scdart = bmk.leiden_cluster(X = np.concatenate((z_rna_scdart, z_atac_scdart), axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_scdart.append(bmk.nmi(group1 = np.concatenate([label_rna, label_atac]), group2 = leiden_labels_scdart))
    ari_scdart.append(bmk.ari(group1 = np.concatenate([label_rna, label_atac]), group2 = leiden_labels_scdart))
print('NMI (scDART): {:.3f}'.format(max(nmi_scdart)))
print('ARI (scDART): {:.3f}'.format(max(ari_scdart)))

# 2. Seurat
nmi_seurat = []
ari_seurat = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_seurat = bmk.leiden_cluster(X = np.concatenate((z_rna_seurat, z_atac_seurat), axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_seurat.append(bmk.nmi(group1 = np.concatenate([label_rna, label_atac]), group2 = leiden_labels_seurat))
    ari_seurat.append(bmk.ari(group1 = np.concatenate([label_rna, label_atac]), group2 = leiden_labels_seurat))
print('NMI (Seurat): {:.3f}'.format(max(nmi_seurat)))
print('ARI (Seurat): {:.3f}'.format(max(ari_seurat)))

# 3. Liger
nmi_liger = []
ari_liger = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_liger = bmk.leiden_cluster(X = np.concatenate((z_rna_liger, z_atac_liger), axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_liger.append(bmk.nmi(group1 = np.concatenate([label_rna, label_atac]), group2 = leiden_labels_liger))
    ari_liger.append(bmk.ari(group1 = np.concatenate([label_rna, label_atac]), group2 = leiden_labels_liger))
print('NMI (Liger): {:.3f}'.format(max(nmi_liger)))
print('ARI (Liger): {:.3f}'.format(max(ari_liger)))

# 4. UnionCom
nmi_unioncom = []
ari_unioncom = []
for resolution in np.arange(0.1, 10, 0.5):
    leiden_labels_unioncom = bmk.leiden_cluster(X = np.concatenate((z_rna_unioncom, z_atac_unioncom), axis = 0), knn_indices = None, knn_dists = None, resolution = resolution)
    nmi_unioncom.append(bmk.nmi(group1 = np.concatenate([label_rna, label_atac]), group2 = leiden_labels_unioncom))
    ari_unioncom.append(bmk.ari(group1 = np.concatenate([label_rna, label_atac]), group2 = leiden_labels_unioncom))
print('NMI (UNIONCOM): {:.3f}'.format(max(nmi_unioncom)))
print('ARI (UNIONCOM): {:.3f}'.format(max(ari_unioncom)))


scores = pd.DataFrame(columns = ["methods", "resolution", "NMI", "ARI", "GC", "Silhouette"])
scores["NMI"] = np.array(nmi_scdart + nmi_seurat + nmi_liger + nmi_unioncom)
scores["ARI"] = np.array(ari_scdart + ari_seurat + ari_liger + ari_unioncom)
scores["GC"] = np.array([gc_scdart] * len(nmi_scdart) + [gc_seurat] * len(nmi_seurat) + [gc_liger] * len(nmi_liger) + [gc_unioncom] * len(nmi_unioncom))
scores["Silhouette"] = np.array([silhouette_scdart] * len(nmi_scdart) + [silhouette_seurat] * len(nmi_seurat) + [silhouette_liger] * len(nmi_liger) + [silhouette_unioncom] * len(nmi_unioncom))
scores["resolution"] = np.array([x for x in np.arange(0.1, 10, 0.5)] * 4)
scores["methods"] = np.array(["scDART"] * len(nmi_scdart) + ["Seurat"] * len(nmi_seurat) + ["Liger"] * len(nmi_liger) + ["UnionCom"] * len(nmi_unioncom))
scores.to_csv("results_spleen/scores.csv")

# In[]
# score for post_nn_distance2
scores = pd.read_csv("results_spleen/scores.csv")

print("GC (scDART): {:.4f}".format(np.max(scores.loc[scores["methods"] == "scDART", "GC"].values)))
print("NMI (scDART): {:.4f}".format(np.max(scores.loc[scores["methods"] == "scDART", "NMI"].values)))
print("ARI (scDART): {:.4f}".format(np.max(scores.loc[scores["methods"] == "scDART", "ARI"].values)))

print("GC (Seurat): {:.4f}".format(np.max(scores.loc[scores["methods"] == "Seurat", "GC"].values)))
print("NMI (Seurat): {:.4f}".format(np.max(scores.loc[scores["methods"] == "Seurat", "NMI"].values)))
print("ARI (Seurat): {:.4f}".format(np.max(scores.loc[scores["methods"] == "Seurat", "ARI"].values)))

print("GC (LIGER): {:.4f}".format(np.max(scores.loc[scores["methods"] == "Liger", "GC"].values)))
print("NMI (LIGER): {:.4f}".format(np.max(scores.loc[scores["methods"] == "Liger", "NMI"].values)))
print("ARI (LIGER): {:.4f}".format(np.max(scores.loc[scores["methods"] == "Liger", "ARI"].values)))

print("GC (UNIONCOM): {:.4f}".format(np.max(scores.loc[scores["methods"] == "UnionCom", "GC"].values)))
print("NMI (UNIONCOM): {:.4f}".format(np.max(scores.loc[scores["methods"] == "UnionCom", "NMI"].values)))
print("ARI (UNIONCOM): {:.4f}".format(np.max(scores.loc[scores["methods"] == "UnionCom", "ARI"].values)))


# In[]
# GC
plt.rcParams["font.size"] = 15
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.4f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

score = pd.read_csv("results_spleen/scores.csv")
gc_scdart = np.max(score.loc[score["methods"] == "scDART", "GC"].values)
gc_seurat = np.max(score.loc[score["methods"] == "Seurat", "GC"].values)
gc_liger = np.max(score.loc[score["methods"] == "Liger", "GC"].values)
gc_unioncom = np.max(score.loc[score["methods"] == "UnionCom", "GC"].values)

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
barlist = ax.bar([1,2,3,4], [gc_scdart, gc_seurat, gc_liger, gc_unioncom], width = 0.4)
barlist[0].set_color('r')
fig.savefig("results_spleen/GC.pdf", bbox_inches = "tight")    

ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_title("graph connectivity", fontsize = 20)
_ = ax.set_xticks([1,2,3,4])
_ = ax.set_xticklabels(["scDART", "Seurat", "Liger", "UnionCom"])
# _ = ax.set_xlabel("cluster", fontsize = 20)
_ = ax.set_ylabel("GC", fontsize = 20)
show_values_on_bars(ax)
fig.savefig("results_spleen/GC.png", bbox_inches = "tight")    

# NMI
nmi_scdart = np.max(score.loc[score["methods"] == "scDART", "NMI"].values)
nmi_seurat = np.max(score.loc[score["methods"] == "Seurat", "NMI"].values)
nmi_liger = np.max(score.loc[score["methods"] == "Liger", "NMI"].values)
nmi_unioncom = np.max(score.loc[score["methods"] == "UnionCom", "NMI"].values)

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
barlist = ax.bar([1,2,3,4], [nmi_scdart, nmi_seurat, nmi_liger, nmi_unioncom], width = 0.4)
barlist[0].set_color('r')    

ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_title("NMI", fontsize = 20)
_ = ax.set_xticks([1,2,3,4])
_ = ax.set_xticklabels(["scDART", "Seurat", "Liger", "UnionCom"])
# _ = ax.set_xlabel("cluster", fontsize = 20)
_ = ax.set_ylabel("NMI", fontsize = 20)
show_values_on_bars(ax)
fig.savefig("results_spleen/NMI.png", bbox_inches = "tight")    

# ARI
ari_scdart = np.max(score.loc[score["methods"] == "scDART", "ARI"].values)
ari_seurat = np.max(score.loc[score["methods"] == "Seurat", "ARI"].values)
ari_liger = np.max(score.loc[score["methods"] == "Liger", "ARI"].values)
ari_unioncom = np.max(score.loc[score["methods"] == "UnionCom", "ARI"].values)

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
barlist = ax.bar([1,2,3,4], [ari_scdart, ari_seurat, ari_liger, ari_unioncom], width = 0.4)
barlist[0].set_color('r')    

ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_title("ARI", fontsize = 20)
_ = ax.set_xticks([1,2,3,4])
_ = ax.set_xticklabels(["scDART", "Seurat", "Liger", "UnionCom"])
# _ = ax.set_xlabel("cluster", fontsize = 20)
_ = ax.set_ylabel("ARI", fontsize = 20)
show_values_on_bars(ax)
fig.savefig("results_spleen/ARI.png", bbox_inches = "tight")     

# Silhouette
silhouette_scdart = np.max(score.loc[score["methods"] == "scDART", "Silhouette"].values)
silhouette_seurat = np.max(score.loc[score["methods"] == "Seurat", "Silhouette"].values)
silhouette_liger = np.max(score.loc[score["methods"] == "Liger", "Silhouette"].values)
silhouette_unioncom = np.max(score.loc[score["methods"] == "UnionCom", "Silhouette"].values)

fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
barlist = ax.bar([1,2,3,4], [silhouette_scdart, silhouette_seurat, silhouette_liger, silhouette_unioncom], width = 0.4)
barlist[0].set_color('r')
fig.savefig("results_spleen/Silhouette.pdf", bbox_inches = "tight")    

ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_title("Average Silhouette Width (batches)", fontsize = 20)
_ = ax.set_xticks([1,2,3,4])
_ = ax.set_xticklabels(["scDART", "Seurat", "Liger", "UnionCom"])
# _ = ax.set_xlabel("cluster", fontsize = 20)
_ = ax.set_ylabel("ASW", fontsize = 20)
show_values_on_bars(ax)
fig.savefig("results_spleen/ASW_batch.png", bbox_inches = "tight")  
# %%
