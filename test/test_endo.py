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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams["font.size"] = 20

# In[]
seeds = [0]
latent_dim = 4
learning_rate = 3e-4
n_epochs = 500
use_anchor = False
reg_d = 1
reg_g = 1
reg_mmd = 1
ts = [30, 50, 70]
use_potential = True

seed = seeds[0]
print("Random seed: " + str(seed))
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

rna_dataset = dataset.endo_rna(counts_dir = "../data/Endo/counts_rna.csv", anno_dir = "../data/Endo/anno_rna.txt", anchor = "Endo (other)")
atac_dataset = dataset.endo_atac(counts_dir = "../data/Endo/counts_atac.csv", anno_dir = "../data/Endo/anno_atac.txt", anchor = "Endo (other)")
coarse_reg = torch.FloatTensor(pd.read_csv("../data/Endo/region2gene.csv", sep = ",", index_col = 0).values).to(device)

batch_size = int(max([len(rna_dataset),len(atac_dataset)])/4)
libsize = rna_dataset.get_libsize()

train_rna_loader = DataLoader(rna_dataset, batch_size = batch_size, shuffle = True)
train_atac_loader = DataLoader(atac_dataset, batch_size = batch_size, shuffle = True)
'''
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


# calculate the diffusion distance
dist_rna = diff.diffu_distance(rna_dataset.counts.numpy(), ts = ts,
                                use_potential = use_potential, dr = "pca", n_components = 30)

dist_atac = diff.diffu_distance(atac_dataset.counts.numpy(), ts = ts,
                                use_potential = use_potential, dr = "lsi", n_components = 30)

# quantile normalization
# if EMBED_CONFIG["use_quantile"]:
#     dist_atac = diff.quantile_norm(dist_atac, reference = dist_rna.reshape(-1), replace = True)

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
                reg_mtx = coarse_reg, reg_d = EMBED_CONFIG["reg_d"], reg_g = EMBED_CONFIG["reg_g"], reg_mmd = EMBED_CONFIG["reg_mmd"], use_anchor = EMBED_CONFIG["use_anchor"], norm = "l1", 
                mode = EMBED_CONFIG["l_dist_type"])

with torch.no_grad():
    z_rna = model_dict["encoder"](rna_dataset.counts.to(device)).cpu().detach()
    z_atac = model_dict["encoder"](model_dict["gene_act"](atac_dataset.counts.to(device))).cpu().detach()

np.save(file = "../test/results_endo/z_rna.npy", arr = z_rna.numpy())
np.save(file = "../test/results_endo/z_atac.npy", arr = z_atac.numpy())
'''

z_rna = np.load(file = "./results_endo/z_rna.npy")
z_atac = np.load(file = "./results_endo/z_atac.npy")
z_rna = torch.FloatTensor(z_rna)
z_atac = torch.FloatTensor(z_atac)

# # post-maching
# pca_op = PCA(n_components = 2)
# z = pca_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
# z_rna_pca = z[:z_rna.shape[0],:]
# z_atac_pca = z[z_rna.shape[0]:,:]
# utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = rna_dataset.cell_labels, 
#                     anno2 = atac_dataset.cell_labels, mode = "separate", save = None, 
#                     figsize = (30,10), axis_label = "PCA")

# umap_op = UMAP(n_components = 2, min_dist = 0.8, random_state = 0)
# z = umap_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
# z_rna_umap = z[:z_rna.shape[0],:]
# z_atac_umap = z[z_rna.shape[0]:,:]
# utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = rna_dataset.cell_labels, 
#                     anno2 = atac_dataset.cell_labels, mode = "separate", save = None, 
#                     figsize = (30,10), axis_label = "UMAP")

# np.save(file = "../test/results_endo/z_rna.npy", arr = z_rna)
# np.save(file = "../test/results_endo/z_atac.npy", arr = z_atac)

import importlib 
importlib.reload(utils)
# post-maching
pca_op = PCA(n_components = 2)
z = pca_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
z_rna_pca = z[:z_rna.shape[0],:]
z_atac_pca = z[z_rna.shape[0]:,:]
utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = rna_dataset.cell_labels, 
                    anno2 = atac_dataset.cell_labels, mode = "joint", save = "results_endo/z_joint.png", 
                    figsize = (10,7), axis_label = "PCA")
utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = rna_dataset.cell_labels, 
                    anno2 = atac_dataset.cell_labels, mode = "modality", save = "results_endo/z_mod.png", 
                    figsize = (10,7), axis_label = "PCA")

# post-maching
# with torch.no_grad():
#     z_rna, z_atac = palign.match_alignment(z_rna = z_rna, z_atac = z_atac, k = 10)
#     z_atac, z_rna = palign.match_alignment(z_rna = z_atac, z_atac = z_rna, k = 10)


# pca_op = PCA(n_components = 2)
# z = pca_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
# z_rna_pca = z[:z_rna.shape[0],:]
# z_atac_pca = z[z_rna.shape[0]:,:]

# utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = rna_dataset.cell_labels, 
#                     anno2 = atac_dataset.cell_labels, mode = "separate", save = None, 
#                     figsize = (30,10), axis_label = "PCA")

# umap_op = UMAP(n_components = 2, min_dist = 0.8, random_state = 0)
# z = umap_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
# z_rna_umap = z[:z_rna.shape[0],:]
# z_atac_umap = z[z_rna.shape[0]:,:]
# utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = rna_dataset.cell_labels, 
#                     anno2 = atac_dataset.cell_labels, mode = "separate", save = None, 
#                     figsize = (30,10), axis_label = "UMAP")


z_rna = np.load(file = "./results_endo/z_rna.npy")
z_atac = np.load(file = "./results_endo/z_atac.npy")
z_rna = torch.FloatTensor(z_rna)
z_atac = torch.FloatTensor(z_atac)


with torch.no_grad():
    z_rna, z_atac = palign.match_alignment(z_rna = z_rna, z_atac = z_atac, k = 10)
    z_atac, z_rna = palign.match_alignment(z_rna = z_atac, z_atac = z_rna, k = 10)

# post-maching
pca_op = PCA(n_components = 2)
z = pca_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
z_rna_pca = z[:z_rna.shape[0],:]
z_atac_pca = z[z_rna.shape[0]:,:]
utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = rna_dataset.cell_labels, 
                    anno2 = atac_dataset.cell_labels, mode = "joint", save = "results_endo/z_joint_post.png", 
                    figsize = (10,7), axis_label = "PCA")
utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = rna_dataset.cell_labels, 
                    anno2 = atac_dataset.cell_labels, mode = "modality", save = "results_endo/z_mod_post.png", 
                    figsize = (10,7), axis_label = "PCA")

umap_op = UMAP(n_components = 2, min_dist = 0.8, random_state = 0)
z = umap_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
z_rna_umap = z[:z_rna.shape[0],:]
z_atac_umap = z[z_rna.shape[0]:,:]
utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = rna_dataset.cell_labels, 
                    anno2 = atac_dataset.cell_labels, mode = "joint", save = "results_endo/z_joint_post_umap.png", 
                    figsize = (10,7), axis_label = "UMAP")
utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = rna_dataset.cell_labels, 
                    anno2 = atac_dataset.cell_labels, mode = "modality", save = "results_endo/z_mod_post_umap.png", 
                    figsize = (10,7), axis_label = "UMAP")

# In[]
def plot_backbone(z1, z2, T, mean_cluster, groups, anno, mode = "joint", save = None, figsize = (20,10), axis_label = "Latent", **kwargs):
    _kwargs = {
        "s": 10,
        "alpha": 0.4,
        "markerscale": 6,
        "fontsize": 20
    }
    _kwargs.update(kwargs)

    fig = plt.figure(figsize = figsize)

    if mode == "joint":
        ax = fig.add_subplot()
        cluster_types = np.sort(np.unique(groups))
        cmap = plt.cm.get_cmap("tab20", len(np.unique(anno)))
        z = np.concatenate((z1, z2), axis = 0)

        for i, cat in enumerate(np.sort(np.unique(anno))):
            idx = np.where(anno == cat)[0]
            ax.scatter(z[idx,0], z[idx,1], color = cmap(i), cmap = 'tab20', label = cat, alpha = _kwargs["alpha"], s = _kwargs["s"])

        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                if T[i,j] != 0:
                    ax.plot([mean_cluster[i, 0], mean_cluster[j, 0]], [mean_cluster[i, 1], mean_cluster[j, 1]], 'r-')
        
        ax.scatter(mean_cluster[:,0], mean_cluster[:,1], color = "red", s = 30)
        
        for i in range(mean_cluster.shape[0]):
            if (cluster_types[i] == "HE") or (cluster_types[i] == "IAC"):
                ax.plot(mean_cluster[i,0] - 0.07, mean_cluster[i,1] + 0.01, color = "blue", marker=  "$" + cluster_types[i] + "$", markersize = 20)
            # elif (cluster_types[i] == "Arterial endo 1") or (cluster_types[i] == "Arterial endo 2") or (cluster_types[i] == "Endo (other)"):
            #     ax.plot(mean_cluster[i,0] - 0.07, mean_cluster[i,1] + 0.01, color = "blue", marker=  "$" + cluster_types[i] + "$", markersize = 70)
            elif (cluster_types[i] == "Arterial endo 1&2"):
                ax.plot(mean_cluster[i,0] - 0.07, mean_cluster[i,1] + 0.01, color = "blue", marker=  "$" + "Arterial\,endo\,1&2" + "$", markersize = 90)
            else:
                ax.plot(mean_cluster[i,0] - 0.07, mean_cluster[i,1] + 0.01, color = "blue", marker=  "$" + cluster_types[i] + "$", markersize = 60)

        ax.tick_params(axis = "both", which = "major", labelsize = 15)

        ax.set_xlabel(axis_label + " 1", fontsize = 19)
        ax.set_ylabel(axis_label + " 2", fontsize = 19)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.legend(bbox_to_anchor=(0.9,1), loc="upper left", fontsize = _kwargs["fontsize"], frameon=False, markerscale = _kwargs["markerscale"])

    if save:
        fig.savefig(save, bbox_inches = "tight")
    
    print(save)

def backbone_inf(z_rna, z_atac, groups):
    import networkx as nx
    X = np.concatenate((z_rna, z_atac), axis = 0)
    n_clust = np.unique(groups).shape[0]

    mean_cluster = [[] for x in range(n_clust)]

    for i, cat in enumerate(np.sort(np.unique(groups))):
        idx = np.where(groups == cat)[0]
        mean_cluster[i] = np.mean(X[idx,:], axis = 0)

    mst = np.zeros((n_clust,n_clust))

    for i in range(n_clust):
        for j in range(n_clust):
            mst[i,j] = np.linalg.norm(np.array(mean_cluster[i]) - np.array(mean_cluster[j]), ord = 2)

    G = nx.from_numpy_matrix(-mst)
    T = nx.maximum_spanning_tree(G, weight = 'weight', algorithm = 'kruskal')
    T = nx.to_numpy_matrix(T)
    # conn is the adj of the MST.

    return groups, mean_cluster, T

z_rna = np.load(file = "./results_endo/z_rna.npy")
z_atac = np.load(file = "./results_endo/z_atac.npy")
z_rna = torch.FloatTensor(z_rna)
z_atac = torch.FloatTensor(z_atac)

z_rna, z_atac = palign.match_alignment(z_rna = z_rna, z_atac = z_atac, k = 10)
z_atac, z_rna = palign.match_alignment(z_rna = z_atac, z_atac = z_rna, k = 10)
z_rna = z_rna.numpy()
z_atac = z_atac.numpy()

# root, manually found
root_cell = 35
dpt_mtx = ti.dpt(np.concatenate((z_rna, z_atac), axis = 0), n_neigh = 10)
pt_infer = dpt_mtx[root_cell, :]
pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
pt_infer = pt_infer/np.max(pt_infer)


pca_op = PCA(n_components = 2)
z = pca_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
z_rna_pca = z[:z_rna.shape[0],:]
z_atac_pca = z[z_rna.shape[0]:,:]

cell_labels = np.concatenate((rna_dataset.cell_labels, atac_dataset.cell_labels), axis = 0).squeeze()
cell_labels2 = np.where((cell_labels == "Arterial endo 1") | (cell_labels == "Arterial endo 2"), "Arterial endo 1&2", cell_labels)
groups, mean_cluster, T = backbone_inf(z_rna, z_atac, cell_labels2)
mean_cluster = pca_op.transform(np.array(mean_cluster))

plot_backbone(z_rna_pca, z_atac_pca, mode = "joint", mean_cluster = mean_cluster, groups = groups, T = T, figsize=(10,7), save = "results_endo/backbone.png", anno = cell_labels, axis_label = "PCA")

# In[] Find de genes
counts_rna = pd.read_csv("../data/Endo/counts_rna.csv", index_col = 0)
z_rna = np.load(file = "./results_endo/z_rna.npy")
z_atac = np.load(file = "./results_endo/z_atac.npy")
z_rna = torch.FloatTensor(z_rna)
z_atac = torch.FloatTensor(z_atac)
z_rna, z_atac = palign.match_alignment(z_rna = z_rna, z_atac = z_atac, k = 10)
z_atac, z_rna = palign.match_alignment(z_rna = z_atac, z_atac = z_rna, k = 10)
z_rna = z_rna.numpy()
z_atac = z_atac.numpy()

# infer pseudotime
root_cell = 35
dpt_mtx = ti.dpt(np.concatenate((z_rna, z_atac), axis = 0), n_neigh = 10)
pt_infer = dpt_mtx[root_cell, :]
pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
pt_infer = pt_infer/np.max(pt_infer)
# for scRNA-Seq batch
pt_infer_rna = pt_infer[:counts_rna.shape[0]]
# for scATAC-Seq batch
pt_infer_atac = pt_infer[counts_rna.shape[0]:]

# infer backbone
groups, mean_cluster, T = ti.backbone_inf(z_rna, z_atac, resolution = 0.02)
groups_rna = groups[:counts_rna.shape[0]]
groups_atac = groups[counts_rna.shape[0]:]
root_clust = groups[root_cell]

# infer all trajectories
G = nx.from_numpy_matrix(T, create_using=nx.DiGraph)
G = nx.dfs_tree(G, source = root_clust)
paths = []
for node in G:
    if G.out_degree(node)==0: #it's a leaf
        paths.append(nx.shortest_path(G, root_clust, node))
        
pca_op = PCA(n_components = 2)
z = pca_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
z_rna_pca = z[:z_rna.shape[0],:]
z_atac_pca = z[z_rna.shape[0]:,:]    
mean_cluster = pca_op.transform(np.array(mean_cluster))
utils.plot_backbone(z_rna_pca, z_atac_pca, groups = groups, T = T, mean_cluster = mean_cluster, mode = "joint", figsize=(10,7), save = None, axis_label = "PCA")


pseudo_order_rna = np.empty((groups_rna.shape[0], len(paths)))
pseudo_order_rna[:] = np.nan
pseudo_order_rna = pd.DataFrame(data = pseudo_order_rna, index = counts_rna.index.values, columns = np.array(["traj_" + str(x) for x in range(len(paths))]))

for i, path in enumerate(paths):
    selected_cells = np.concatenate([np.where(groups_rna == x)[0] for x in path], axis = 0)
    pseudo_order_rna.iloc[selected_cells, i] = pt_infer_rna[selected_cells]

# Find de genes, normal distribution after log-transform
de_genes = de.de_analy(X = counts_rna, pseudo_order = pseudo_order_rna, p_val_t = 0.05, verbose = False, distri = "normal", fdr_correct = True)
for traj in de_genes.keys():
    genes = np.array([x["feature"] for x in de_genes[traj]])
    p_val = np.array([x["p_val"] for x in de_genes[traj]])
    genes= genes[np.argsort(p_val)]
    p_val = p_val[np.argsort(p_val)]
    de_list = pd.DataFrame.from_dict({"feature": genes, "p-val": p_val})
    de_list.to_csv("./results_endo/de_endo/de_gene_" + str(traj) + ".csv")

de.de_plot(X = counts_rna, pseudo_order = pseudo_order_rna, de_feats = de_genes, figsize = (20,50), n_feats = 20)

utils.plot_latent_pt(z1 = z_rna_pca, z2 = z_atac_pca, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = "./results_endo/z_pt.png", figsize = (10,7), axis_label = "PCA")

# In[] Find de regions
# infer backbone
counts_atac = pd.read_csv("../data/Endo/counts_atac.csv", index_col = 0)

pseudo_order_atac = np.empty((groups_atac.shape[0], len(paths)))
pseudo_order_atac[:] = np.nan
pseudo_order_atac = pd.DataFrame(data = pseudo_order_atac, index = counts_atac.index.values, columns = np.array(["traj_" + str(x) for x in range(len(paths))]))

for i, path in enumerate(paths):
    # find all the cells in the path
    selected_cells = np.concatenate([np.where(groups_atac == x)[0] for x in path], axis = 0)
    # insert pseudotime of the found cells into pseudo_order
    pseudo_order_atac.iloc[selected_cells, i] = pt_infer_atac[selected_cells]

# Find de regions, binomial distribution
de_regions = de.de_analy(X = counts_atac, pseudo_order = pseudo_order_atac, p_val_t = 0.05, verbose = False, distri = "binomial", fdr_correct = True)
for traj in de_regions.keys():
    regions = np.array([x["feature"] for x in de_regions[traj]])
    p_val = np.array([x["p_val"] for x in de_regions[traj]])
    regions= regions[np.argsort(p_val)]
    p_val = p_val[np.argsort(p_val)]
    de_list = pd.DataFrame.from_dict({"feature": regions, "p-val": p_val})
    de_list.to_csv("./results_endo/de_endo/de_region_" + str(traj) + ".csv")

de.de_plot(X = counts_atac, pseudo_order = pseudo_order_atac, de_feats = de_regions, figsize = (20,50), n_feats = 20)


# In[] Find de motif
counts_motif = pd.read_csv("../data/Endo/chromVAR/motif_z.csv", index_col = 0)

# Find de regions, binomial distribution
de_motifs = de.de_analy(X = counts_motif, pseudo_order = pseudo_order_atac, p_val_t = 0.05, verbose = False, distri = "normal", fdr_correct = True)
for traj in de_motifs.keys():
    motifs = np.array([x["feature"] for x in de_motifs[traj]])
    p_val = np.array([x["p_val"] for x in de_motifs[traj]])
    motifs= motifs[np.argsort(p_val)]
    p_val = p_val[np.argsort(p_val)]
    de_list = pd.DataFrame.from_dict({"feature": motifs, "p-val": p_val})
    de_list.to_csv("./results_endo/de_endo/de_motif_" + str(traj) + ".csv")

de.de_plot(X = counts_motif, pseudo_order = pseudo_order_atac, de_feats = de_motifs, figsize = (20,50), n_feats = 20)



# In[] Other baseline methods
# 1. Liger
path = "./results_endo/"
z_rna = pd.read_csv("results_endo/liger/H1.csv", index_col = 0)
z_atac = pd.read_csv("results_endo/liger/H2.csv", index_col = 0)
integrated_data = (z_rna.values, z_atac.values)

pca_op = PCA(n_components = 2)
umap_op = UMAP(n_components = 2)

counts_rna = pd.read_csv("../data/Endo/counts_rna.csv", index_col = 0)
counts_atac = pd.read_csv("../data/Endo/counts_atac.csv", index_col = 0)
anno_rna = pd.read_csv("../data/Endo/anno_rna.txt", header = None)
anno_rna.index = counts_rna.index.values
anno_atac = pd.read_csv("../data/Endo/anno_atac.txt", header = None)
anno_atac.index = counts_atac.index.values
anno_rna = anno_rna.loc[z_rna.index.values,:]
anno_atac = anno_atac.loc[z_atac.index.values,:]

pca_latent = pca_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))
umap_latent = umap_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))

utils.plot_latent(umap_latent[:z_rna.shape[0],:], umap_latent[z_rna.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "joint", save = "results_endo/liger/liger_umap.png", figsize = (10,7), axis_label = "UMAP")
utils.plot_latent(umap_latent[:z_rna.shape[0],:], umap_latent[z_rna.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "modality", save = "results_endo/liger/liger_batches_umap.png", figsize = (10,7), axis_label = "UMAP")
utils.plot_latent(pca_latent[:z_rna.shape[0],:], pca_latent[z_rna.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "joint", save = "results_endo/liger/liger_pca.png", figsize = (10,7), axis_label = "PCA")
utils.plot_latent(pca_latent[:z_rna.shape[0],:], pca_latent[z_rna.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "modality", save = "results_endo/liger/liger_batches_pca.png", figsize = (10,7), axis_label = "PCA")


z_rna = pd.read_csv("results_endo/liger/H1_full.csv", index_col = 0)
z_atac = pd.read_csv("results_endo/liger/H2_full.csv", index_col = 0)
integrated_data = (z_rna.values, z_atac.values)

pca_latent = pca_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))
umap_latent = umap_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))

utils.plot_latent(umap_latent[:z_rna.shape[0],:], umap_latent[z_rna.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "joint", save = "results_endo/liger/liger_umap_full.png", figsize = (10,7), axis_label = "UMAP")
utils.plot_latent(umap_latent[:z_rna.shape[0],:], umap_latent[z_rna.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "modality", save = "results_endo/liger/liger_batches_umap_full.png", figsize = (10,7), axis_label = "UMAP")
utils.plot_latent(pca_latent[:z_rna.shape[0],:], pca_latent[z_rna.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "joint", save = "results_endo/liger/liger_pca_full.png", figsize = (10,7), axis_label = "PCA")
utils.plot_latent(pca_latent[:z_rna.shape[0],:], pca_latent[z_rna.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "modality", save = "results_endo/liger/liger_batches_pca_full.png", figsize = (10,7), axis_label = "PCA")



# In[] 2. Seurat
# anno_rna = pd.read_csv("../data/Endo/anno_rna.txt", header = None).values
# anno_atac = pd.read_csv("../data/Endo/anno_atac.txt", header = None).values
coembed = pd.read_csv("results_endo/seurat/pca_embedding.txt", sep = "\t").values
coembed_full = pd.read_csv("results_endo/seurat/pca_embedding_full.txt", sep = "\t").values
rna_embed = coembed[:anno_rna.shape[0],:]
atac_embed = coembed[anno_rna.shape[0]:,:]
rna_embed_full = coembed_full[:anno_rna.shape[0],:]
atac_embed_full = coembed_full[anno_rna.shape[0]:,:]

utils.plot_latent(rna_embed, atac_embed, anno_rna, anno_atac, mode = "modality", figsize = (10,7), axis_label = "PCA", save = "results_endo/seurat/pca.png")
utils.plot_latent(rna_embed, atac_embed, anno_rna, anno_atac, mode = "joint", figsize = (10,7), axis_label = "PCA", save = "results_endo/seurat/pca_joint.png")


utils.plot_latent(rna_embed_full, atac_embed_full, anno_rna, anno_atac, mode = "modality", figsize = (10,7), axis_label = "PCA", save = "results_endo/seurat/pca_full.png")
utils.plot_latent(rna_embed_full, atac_embed_full, anno_rna, anno_atac, mode = "joint", figsize = (10,7), axis_label = "PCA", save = "results_endo/seurat/pca_joint_full.png")

# In[] 3. UnionCom
z_rna = np.load("results_endo/unioncom/lat_rna.npy")
z_atac = np.load("results_endo/unioncom/lat_atac.npy")
integrated_data = (z_rna, z_atac)

pca_op = PCA(n_components = 2)
umap_op = UMAP(n_components = 2)

pca_latent = pca_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))
umap_latent = umap_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))

utils.plot_latent(umap_latent[:z_rna.shape[0],:], umap_latent[z_rna.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "joint", save = "results_endo/unioncom/unioncom_umap.png", figsize = (10,7), axis_label = "UMAP")
utils.plot_latent(umap_latent[:z_rna.shape[0],:], umap_latent[z_rna.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "modality", save = "results_endo/unioncom/unioncom_batches_umap.png", figsize = (10,7), axis_label = "UMAP")
utils.plot_latent(pca_latent[:z_rna.shape[0],:], pca_latent[z_rna.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "joint", save = "results_endo/unioncom/unioncom_pca.png", figsize = (10,7), axis_label = "PCA")
utils.plot_latent(pca_latent[:z_rna.shape[0],:], pca_latent[z_rna.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "modality", save = "results_endo/unioncom/unioncom_batches_pca.png", figsize = (10,7), axis_label = "PCA")

# %%
