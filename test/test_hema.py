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

from adjustText import adjust_text
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams["font.size"] = 20

import scanpy as sc 
import anndata 

def plot_pt(z, pseudo_order, ncols = 1,  basis = "PCA", figsize= (20,20), axis = True):
    """\
    Description        
        Plot the first order approximation estimation of pseudo-time

    Parameters
    ----------
    cellpath_obj
        Cellpath object
    basis
        the basis used for visualization
    trajs
        The number of trajectories plotted, note that trajs cannot be too much, or there may exists trajs contains too few cells, may even less than the neighs parameter
    figsize
        Figure size, tuple
    save_as
        Name of the saved file
    title
        The title name of the plot
    axis
        Show axis or not, boolean value
    """      
    num_traj = pseudo_order.shape[1]
    nrows = np.ceil(num_traj/ncols).astype(int)
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)
    pseudo_order.index = np.arange(z.shape[0])
    for i, traj in enumerate(pseudo_order.columns.values):
        sorted_pt = pseudo_order[traj].dropna(axis = 0).sort_values()
        z_traj = z[sorted_pt.index,:]
 
        if nrows != 1:
            # multiple >2 plots
            if not axis:
                axs[i%nrows, i//nrows].axis("off")
            axs[i%nrows, i//nrows].scatter(z[:,0],z[:,1], color = 'gray', alpha = 0.1)
            

            pseudo_visual = axs[i%nrows, i//nrows].scatter(z_traj[:,0],z_traj[:,1],c = np.arange(z_traj.shape[0]), cmap=plt.get_cmap('gnuplot'), alpha = 0.7)

            axs[i%nrows, i//nrows].set_title("Trajectory " + str(i), fontsize = 25)
            axs[i%nrows, i//nrows].set_xlabel(basis + "1", fontsize = 19)
            axs[i%nrows, i//nrows].set_ylabel(basis + "2", fontsize = 19)
            axs[i%nrows, i//nrows].set_xticks([])
            axs[i%nrows, i//nrows].set_yticks([])
            axs[i%nrows, i//nrows].spines['right'].set_visible(False)
            axs[i%nrows, i//nrows].spines['top'].set_visible(False)

            cbar = fig.colorbar(pseudo_visual,fraction=0.046, pad=0.04, ax = axs[i%nrows, i//nrows])
            cbar.ax.tick_params(labelsize = 20)

        elif nrows == 1 and ncols == 1:
            # one plot
            if not axis:
                axs.axis("off")            
            axs.scatter(z[:,0],z[:,1], color = 'gray', alpha = 0.1)

            pseudo_visual = axs.scatter(z_traj[:,0],z_traj[:,1],c = np.arange(z_traj.shape[0]), cmap=plt.get_cmap('gnuplot'),alpha = 0.7)

            axs.set_title("Trajectory " + str(i), fontsize = 25)
            axs.set_xlabel(basis + " 1", fontsize = 19)
            axs.set_ylabel(basis + " 2", fontsize = 19)
            axs.set_xticks([])
            axs.set_yticks([])
            axs.spines['right'].set_visible(False)
            axs.spines['top'].set_visible(False)   

            cbar = fig.colorbar(pseudo_visual,fraction=0.046, pad=0.04, ax = axs)
            cbar.ax.tick_params(labelsize = 20)

        else:
            # two plots
            if not axis:
                axs[i].axis("off")
            axs[i].scatter(z[:,0],z[:,1], color = 'gray', alpha = 0.1)
            pseudo_visual = axs[i].scatter(z_traj[:,0],z_traj[:,1],c = np.arange(z_traj.shape[0]), cmap=plt.get_cmap('gnuplot'),alpha = 0.7)
            
            axs[i].set_title("Trajectory " + str(i), fontsize = 25)
            axs[i].set_xlabel(basis + " 1", fontsize = 19)
            axs[i].set_ylabel(basis + " 2", fontsize = 19)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['top'].set_visible(False)

            cbar = fig.colorbar(pseudo_visual,fraction=0.046, pad=0.04, ax = axs[i])
            cbar.ax.tick_params(labelsize = 20)
    plt.show()     

    return fig, axs
# In[]

seeds = [0] # 2
latent_dim = 4
learning_rate = 3e-4
n_epochs = 500
use_anchor = False
reg_d = 1
reg_g = 1
reg_mmd = 1
ts = [30, 50, 70]
use_potential = True
norm = "l1"

seed = seeds[0]
print("Random seed: " + str(seed))
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

rna_dataset = dataset.hema_rna(counts_dir = "../data/hema/counts_rna.csv", anno_dir = "../data/hema/anno_rna.txt", anchor = "MEP")
atac_dataset = dataset.hema_atac(counts_dir = "../data/hema/counts_atac.csv", anno_dir = "../data/hema/anno_atac.txt", anchor = "MEP")
coarse_reg = torch.FloatTensor(pd.read_csv("../data/hema/region2gene.csv", sep = ",", index_col = 0).values).to(device)

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
                reg_mtx = coarse_reg, reg_d = EMBED_CONFIG["reg_d"], reg_g = EMBED_CONFIG["reg_g"], reg_mmd = EMBED_CONFIG["reg_mmd"], use_anchor = EMBED_CONFIG["use_anchor"], norm = norm, 
                mode = EMBED_CONFIG["l_dist_type"])

with torch.no_grad():
    z_rna = model_dict["encoder"](rna_dataset.counts.to(device)).cpu().detach()
    z_atac = model_dict["encoder"](model_dict["gene_act"](atac_dataset.counts.to(device))).cpu().detach()

np.save(file = "../test/results_hema/z_rna.npy", arr = z_rna.numpy())
np.save(file = "../test/results_hema/z_atac.npy", arr = z_atac.numpy())
'''
z_rna = np.load(file = "./results_hema/z_rna.npy")
z_atac = np.load(file = "./results_hema/z_atac.npy")
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

# np.save(file = "../test/results_hema/z_rna.npy", arr = z_rna)
# np.save(file = "../test/results_hema/z_atac.npy", arr = z_atac)

import importlib 
importlib.reload(utils)
# post-maching
pca_op = PCA(n_components = 2)
z = pca_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
z_rna_pca = z[:z_rna.shape[0],:]
z_atac_pca = z[z_rna.shape[0]:,:]
utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = rna_dataset.cell_labels, 
                    anno2 = atac_dataset.cell_labels, mode = "joint", save = "results_hema/z_joint.png", 
                    figsize = (15,7), axis_label = "PCA")
utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = rna_dataset.cell_labels, 
                    anno2 = atac_dataset.cell_labels, mode = "modality", save = "results_hema/z_mod.png", 
                    figsize = (15,7), axis_label = "PCA")

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


z_rna = np.load(file = "./results_hema/z_rna.npy")
z_atac = np.load(file = "./results_hema/z_atac.npy")
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
                    anno2 = atac_dataset.cell_labels, mode = "joint", save = "results_hema/z_joint_post.png", 
                    figsize = (15,7), axis_label = "PCA")
utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = rna_dataset.cell_labels, 
                    anno2 = atac_dataset.cell_labels, mode = "modality", save = "results_hema/z_mod_post.png", 
                    figsize = (15,7), axis_label = "PCA")

umap_op = UMAP(n_components = 2, min_dist = 0.8, random_state = 0)
z = umap_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
z_rna_umap = z[:z_rna.shape[0],:]
z_atac_umap = z[z_rna.shape[0]:,:]
utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = rna_dataset.cell_labels, 
                    anno2 = atac_dataset.cell_labels, mode = "joint", save = "results_hema/z_joint_post_umap.png", 
                    figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = rna_dataset.cell_labels, 
                    anno2 = atac_dataset.cell_labels, mode = "modality", save = "results_hema/z_mod_post_umap.png", 
                    figsize = (15,7), axis_label = "UMAP")

# run diffusion map
adata_scdart = anndata.AnnData(X = np.concatenate((z_rna,z_atac), axis = 0))
# adata_scdart = anndata.AnnData(X = np.concatenate((z_rna_pca,z_atac_pca), axis = 0))

sc.pp.neighbors(adata_scdart, use_rep = 'X', n_neighbors = 30, random_state = 0)
sc.tl.diffmap(adata_scdart, random_state = 0)
diffmap_latent = adata_scdart.obsm["X_diffmap"]
utils.plot_latent(diffmap_latent[:z_rna.shape[0],:], diffmap_latent[z_rna.shape[0]:,:], anno1 = rna_dataset.cell_labels, anno2 = atac_dataset.cell_labels, 
mode = "joint", save = "results_hema/z_joint_post_diffmap.png", figsize = (15,7), axis_label = "Diffmap")
utils.plot_latent(diffmap_latent[:z_rna.shape[0],:], diffmap_latent[z_rna.shape[0]:,:], anno1 = rna_dataset.cell_labels, anno2 = atac_dataset.cell_labels, 
mode = "modality", save = "results_hema/z_mod_post_diffmap.png", figsize = (15,7), axis_label = "Diffmap")

z_destiny = np.load("results_hema/z_diffmap.npy")
utils.plot_latent(z_destiny[:z_rna.shape[0],:], z_destiny[z_rna.shape[0]:,:], anno1 = rna_dataset.cell_labels, anno2 = atac_dataset.cell_labels, 
mode = "joint", save = "results_hema/z_joint_post_destiny.png", figsize = (15,7), axis_label = "Diffmap")
utils.plot_latent(z_destiny[:z_rna.shape[0],:], z_destiny[z_rna.shape[0]:,:], anno1 = rna_dataset.cell_labels, anno2 = atac_dataset.cell_labels, 
mode = "modality", save = "results_hema/z_mod_post_destiny.png", figsize = (15,7), axis_label = "Diffmap")


# In[]
def plot_backbone(z1, z2, T, mean_cluster, groups, anno, mode = "joint", save = None, figsize = (20,10), axis_label = "Latent", **kwargs):
    _kwargs = {
        "s": 10,
        "alpha": 0.7,
        "markerscale": 6,
        "fontsize": 20
    }
    _kwargs.update(kwargs)

    fig = plt.figure(figsize = figsize)

    if mode == "joint":
        ax = fig.add_subplot()
        cluster_types = np.sort(np.unique(groups))
        cmap = plt.cm.get_cmap("Paired", len(np.unique(anno)))
        z = np.concatenate((z1, z2), axis = 0)

        for i, cat in enumerate(np.sort(np.unique(anno))):
            idx = np.where(anno == cat)[0]
            ax.scatter(z[idx,0], z[idx,1], color = cmap(i), label = cat, alpha = _kwargs["alpha"], s = _kwargs["s"])

        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                if T[i,j] != 0:
                    ax.plot([mean_cluster[i, 0], mean_cluster[j, 0]], [mean_cluster[i, 1], mean_cluster[j, 1]], 'r-')
        
        ax.scatter(mean_cluster[:,0], mean_cluster[:,1], color = "red", s = 30)
        
        # for i in range(mean_cluster.shape[0]):
        #     if (cluster_types[i] == "HSC&MPP"):
        #         ax.plot(mean_cluster[i,0] - 0.07, mean_cluster[i,1] + 0.01, color = "blue", marker=  "$HSC\,&\,MPP$", markersize = 60)
        #     else:
        #         ax.plot(mean_cluster[i,0] - 0.07, mean_cluster[i,1] + 0.01, color = "blue", marker=  "$" + cluster_types[i] + "$", markersize = 30)
        texts = []
        for i in range(mean_cluster.shape[0]):
            # marker = cluster_types[i].split("_")[0] + "\_" + cluster_types[i].split("_")[1] 
            # ax.plot(mean_cluster[i,0] - 0.007, mean_cluster[i,1] + 0.001, color = "black", marker=  "$" + marker + "$", markersize = 70)
            texts.append(ax.text(mean_cluster[i,0] - 0.007, mean_cluster[i,1] + 0.001, color = "black", s = cluster_types[i], size = 'small', weight = 'bold', in_layout = True))


        ax.tick_params(axis = "both", which = "major", labelsize = 15)

        ax.set_xlabel(axis_label + " 1", fontsize = 19)
        ax.set_ylabel(axis_label + " 2", fontsize = 19)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize = _kwargs["fontsize"], frameon=False, markerscale = _kwargs["markerscale"])

    adjust_text(texts, only_move={'points':'y', 'texts':'y'})
    plt.tight_layout()
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

z_rna = np.load(file = "./results_hema/z_rna.npy")
z_atac = np.load(file = "./results_hema/z_atac.npy")
z_rna = torch.FloatTensor(z_rna)
z_atac = torch.FloatTensor(z_atac)

z_rna, z_atac = palign.match_alignment(z_rna = z_rna, z_atac = z_atac, k = 10)
z_atac, z_rna = palign.match_alignment(z_rna = z_atac, z_atac = z_rna, k = 10)
z_rna = z_rna.numpy()
z_atac = z_atac.numpy()

# root, manually found
root_cell = 188
dpt_mtx = ti.dpt(np.concatenate((z_rna, z_atac), axis = 0), n_neigh = 10)
pt_infer = dpt_mtx[root_cell, :]
pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
pt_infer = pt_infer/np.max(pt_infer)
# for scRNA-Seq batch
pt_infer_rna = pt_infer[:z_rna.shape[0]]
# for scATAC-Seq batch
pt_infer_atac = pt_infer[z_rna.shape[0]:]


pca_op = PCA(n_components = 2)
z = pca_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
z_rna_pca = z[:z_rna.shape[0],:]
z_atac_pca = z[z_rna.shape[0]:,:]

cell_labels = np.concatenate((rna_dataset.cell_labels, atac_dataset.cell_labels), axis = 0).squeeze()
cell_labels2 = np.where((cell_labels == "HSC") | (cell_labels == "MPP"), "HSC&MPP", cell_labels)
groups, mean_cluster, T = backbone_inf(z_rna, z_atac, cell_labels2)
mean_cluster = pca_op.transform(np.array(mean_cluster))

plot_backbone(z_rna_pca, z_atac_pca, mode = "joint", mean_cluster = mean_cluster, groups = groups, T = T, figsize=(15,7), save = "results_hema/backbone.png", anno = cell_labels, axis_label = "PCA")
utils.plot_latent_pt(z1 = z_rna_pca, z2 = z_atac_pca, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = "./results_hema/z_pt.png", figsize = (15,7), axis_label = "PCA")


# In[] Other baseline methods
# 1. Liger
path = "results_hema/liger/"
z_rna_liger = pd.read_csv(path + "H1.csv", index_col = 0)
z_atac_liger = pd.read_csv(path + "H2.csv", index_col = 0)
integrated_data = (z_rna_liger.values, z_atac_liger.values)

pca_op = PCA(n_components = 2)
umap_op = UMAP(n_components = 2)

counts_rna = pd.read_csv("../data/hema/counts_rna.csv", index_col = 0)
counts_atac = pd.read_csv("../data/hema/counts_atac.csv", index_col = 0)
anno_rna = pd.read_csv("../data/hema/anno_rna.txt", header = None)
anno_rna.index = counts_rna.index.values
anno_atac = pd.read_csv("../data/hema/anno_atac.txt", header = None)
anno_atac.index = counts_atac.index.values
anno_rna = anno_rna.loc[z_rna_liger.index.values,:]
anno_atac = anno_atac.loc[z_atac_liger.index.values,:]

pca_latent = pca_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))
umap_latent = umap_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))

utils.plot_latent(umap_latent[:z_rna_liger.shape[0],:], umap_latent[z_rna_liger.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "joint", save = path + "liger_umap.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(umap_latent[:z_rna_liger.shape[0],:], umap_latent[z_rna_liger.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "modality", save = path + "liger_batches_umap.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(pca_latent[:z_rna_liger.shape[0],:], pca_latent[z_rna_liger.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "joint", save = path + "liger_pca.png", figsize = (15,7), axis_label = "PCA")
utils.plot_latent(pca_latent[:z_rna_liger.shape[0],:], pca_latent[z_rna_liger.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "modality", save = path + "liger_batches_pca.png", figsize = (15,7), axis_label = "PCA")

z_destiny = np.load(path + "z_diffmap.npy")
utils.plot_latent(z_destiny[:z_rna.shape[0],:], z_destiny[z_rna.shape[0]:,:], anno1 = rna_dataset.cell_labels, anno2 = atac_dataset.cell_labels, 
mode = "joint", save = path + "liger_destiny.png", figsize = (15,7), axis_label = "Diffmap")
utils.plot_latent(z_destiny[:z_rna.shape[0],:], z_destiny[z_rna.shape[0]:,:], anno1 = rna_dataset.cell_labels, anno2 = atac_dataset.cell_labels, 
mode = "modality", save = path + "liger_batches_destiny.png", figsize = (15,7), axis_label = "Diffmap")

# Infer backbone
root_cell = 188
dpt_mtx = ti.dpt(np.concatenate((z_rna_liger, z_atac_liger), axis = 0), n_neigh = 10)
pt_infer = dpt_mtx[root_cell, :]
pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
pt_infer = pt_infer/np.max(pt_infer)
# for scRNA-Seq batch
pt_infer_rna = pt_infer[:z_rna_liger.shape[0]]
# for scATAC-Seq batch
pt_infer_atac = pt_infer[z_rna_liger.shape[0]:]


pca_op = PCA(n_components = 2)
z = pca_op.fit_transform(np.concatenate((z_rna_liger, z_atac_liger), axis = 0))
z_rna_pca = z[:z_rna_liger.shape[0],:]
z_atac_pca = z[z_rna_liger.shape[0]:,:]

cell_labels = np.concatenate((rna_dataset.cell_labels, atac_dataset.cell_labels), axis = 0).squeeze()
cell_labels2 = np.where((cell_labels == "HSC") | (cell_labels == "MPP"), "HSC&MPP", cell_labels)
groups, mean_cluster, T = backbone_inf(z_rna_liger, z_atac_liger, cell_labels2)
mean_cluster = pca_op.transform(np.array(mean_cluster))

plot_backbone(z_rna_pca, z_atac_pca, mode = "joint", mean_cluster = mean_cluster, groups = groups, T = T, figsize=(15,7), save = path + "backbone.png", anno = cell_labels, axis_label = "PCA")
utils.plot_latent_pt(z1 = z_rna_pca, z2 = z_atac_pca, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = path + "z_pt.png", figsize = (15,7), axis_label = "PCA")



# z_rna_liger = pd.read_csv("results_hema/liger/H1_full.csv", index_col = 0)
# z_atac_liger = pd.read_csv("results_hema/liger/H1_full.csv", index_col = 0)
# integrated_data = (z_rna_liger.values, z_atac_liger.values)

# pca_latent = pca_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))
# umap_latent = umap_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))

# utils.plot_latent(umap_latent[:z_rna_liger.shape[0],:], umap_latent[z_rna_liger.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
# mode = "joint", save = path + "liger_umap_full.png", figsize = (15,7), axis_label = "UMAP")
# utils.plot_latent(umap_latent[:z_rna_liger.shape[0],:], umap_latent[z_rna_liger.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
# mode = "modality", save = path + "liger_batches_umap_full.png", figsize = (15,7), axis_label = "UMAP")
# utils.plot_latent(pca_latent[:z_rna_liger.shape[0],:], pca_latent[z_rna_liger.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
# mode = "joint", save = path + "liger_pca_full.png", figsize = (15,7), axis_label = "PCA")
# utils.plot_latent(pca_latent[:z_rna_liger.shape[0],:], pca_latent[z_rna_liger.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
# mode = "modality", save = path + "liger_batches_pca_full.png", figsize = (15,7), axis_label = "PCA")

# # Infer backbone
# root_cell = 188
# dpt_mtx = ti.dpt(np.concatenate((z_rna_liger, z_atac_liger), axis = 0), n_neigh = 10)
# pt_infer = dpt_mtx[root_cell, :]
# pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
# pt_infer = pt_infer/np.max(pt_infer)
# # for scRNA-Seq batch
# pt_infer_rna = pt_infer[:z_rna_liger.shape[0]]
# # for scATAC-Seq batch
# pt_infer_atac = pt_infer[z_rna_liger.shape[0]:]


# pca_op = PCA(n_components = 2)
# z = pca_op.fit_transform(np.concatenate((z_rna_liger, z_atac_liger), axis = 0))
# z_rna_pca = z[:z_rna_liger.shape[0],:]
# z_atac_pca = z[z_rna_liger.shape[0]:,:]

# cell_labels = np.concatenate((rna_dataset.cell_labels, atac_dataset.cell_labels), axis = 0).squeeze()
# cell_labels2 = np.where((cell_labels == "HSC") | (cell_labels == "MPP"), "HSC&MPP", cell_labels)
# groups, mean_cluster, T = backbone_inf(z_rna_liger, z_atac_liger, cell_labels2)
# mean_cluster = pca_op.transform(np.array(mean_cluster))

# plot_backbone(z_rna_pca, z_atac_pca, mode = "joint", mean_cluster = mean_cluster, groups = groups, T = T, figsize=(15,7), save = path + "backbone_full.png", anno = cell_labels, axis_label = "PCA")
# utils.plot_latent_pt(z1 = z_rna_pca, z2 = z_atac_pca, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = path + "z_pt_full.png", figsize = (15,7), axis_label = "PCA")



# In[] 2. Seurat
# anno_rna = pd.read_csv("../data/hema/anno_rna.txt", header = None).values
# anno_atac = pd.read_csv("../data/hema/anno_atac.txt", header = None).values
path = "results_hema/seurat/"
coembed = pd.read_csv(path + "umap_embedding.txt", sep = "\t").values
z_rna_seurat = coembed[:anno_rna.shape[0],:]
z_atac_seurat = coembed[anno_rna.shape[0]:,:]
utils.plot_latent(z_rna_seurat, z_atac_seurat, anno_rna, anno_atac, mode = "modality", figsize = (15,7), axis_label = "UMAP", save = path + "umap.png")
utils.plot_latent(z_rna_seurat, z_atac_seurat, anno_rna, anno_atac, mode = "joint", figsize = (15,7), axis_label = "UMAP", save = path + "umap_joint.png")

coembed = pd.read_csv(path + "pca_embedding.txt", sep = "\t").values
z_rna_seurat = coembed[:anno_rna.shape[0],:]
z_atac_seurat = coembed[anno_rna.shape[0]:,:]
utils.plot_latent(z_rna_seurat, z_atac_seurat, anno_rna, anno_atac, mode = "modality", figsize = (15,7), axis_label = "PCA", save = path + "pca.png")
utils.plot_latent(z_rna_seurat, z_atac_seurat, anno_rna, anno_atac, mode = "joint", figsize = (15,7), axis_label = "PCA", save = path + "pca_joint.png")

z_destiny = np.load(path + "z_diffmap.npy")
utils.plot_latent(z_destiny[:z_rna.shape[0],:], z_destiny[z_rna.shape[0]:,:], anno1 = rna_dataset.cell_labels, anno2 = atac_dataset.cell_labels, 
mode = "joint", save = path + "destiny_joint.png", figsize = (15,7), axis_label = "Diffmap")
utils.plot_latent(z_destiny[:z_rna.shape[0],:], z_destiny[z_rna.shape[0]:,:], anno1 = rna_dataset.cell_labels, anno2 = atac_dataset.cell_labels, 
mode = "modality", save = path + "destiny.png", figsize = (15,7), axis_label = "Diffmap")

# Infer backbone
root_cell = 188
dpt_mtx = ti.dpt(np.concatenate((z_rna_seurat, z_atac_seurat), axis = 0), n_neigh = 10)
pt_infer = dpt_mtx[root_cell, :]
pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
pt_infer = pt_infer/np.max(pt_infer)
# for scRNA-Seq batch
pt_infer_rna = pt_infer[:z_rna_seurat.shape[0]]
# for scATAC-Seq batch
pt_infer_atac = pt_infer[z_rna_seurat.shape[0]:]

cell_labels = np.concatenate((rna_dataset.cell_labels, atac_dataset.cell_labels), axis = 0).squeeze()
cell_labels2 = np.where((cell_labels == "HSC") | (cell_labels == "MPP"), "HSC&MPP", cell_labels)
groups, mean_cluster, T = backbone_inf(z_rna_seurat, z_atac_seurat, cell_labels2)

plot_backbone(z_rna_seurat, z_atac_seurat, mode = "joint", mean_cluster = np.array(mean_cluster), groups = groups, T = T, figsize=(15,7), save = path + "backbone.png", anno = cell_labels, axis_label = "PCA")
utils.plot_latent_pt(z1 = z_rna_seurat, z2 = z_atac_seurat, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = path + "z_pt.png", figsize = (15,7), axis_label = "PCA")

# Include post-processing
z_rna_seurat_post = torch.FloatTensor(z_rna_seurat)
z_atac_seurat_post = torch.FloatTensor(z_atac_seurat)
z_rna_seurat_post, z_atac_seurat_post = palign.match_alignment(z_rna = z_rna_seurat_post, z_atac = z_atac_seurat_post, k = 10)
z_atac_seurat_post, z_rna_seurat_post = palign.match_alignment(z_rna = z_atac_seurat_post, z_atac = z_rna_seurat_post, k = 10)
z_rna_seurat_post = z_rna_seurat_post.numpy()
z_atac_seurat_post = z_atac_seurat_post.numpy()
utils.plot_latent(z_rna_seurat_post, z_atac_seurat_post, anno_rna, anno_atac, mode = "modality", figsize = (15,7), axis_label = "PCA", save = path + "pca_post.png")
utils.plot_latent(z_rna_seurat_post, z_atac_seurat_post, anno_rna, anno_atac, mode = "joint", figsize = (15,7), axis_label = "PCA", save = path + "pca_joint_post.png")

# Infer backbone
root_cell = 188
dpt_mtx = ti.dpt(np.concatenate((z_rna_seurat_post, z_atac_seurat_post), axis = 0), n_neigh = 10)
pt_infer = dpt_mtx[root_cell, :]
pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
pt_infer = pt_infer/np.max(pt_infer)
# for scRNA-Seq batch
pt_infer_rna = pt_infer[:z_rna_seurat_post.shape[0]]
# for scATAC-Seq batch
pt_infer_atac = pt_infer[z_rna_seurat_post.shape[0]:]

cell_labels = np.concatenate((rna_dataset.cell_labels, atac_dataset.cell_labels), axis = 0).squeeze()
cell_labels2 = np.where((cell_labels == "HSC") | (cell_labels == "MPP"), "HSC&MPP", cell_labels)
groups, mean_cluster, T = backbone_inf(z_rna_seurat_post, z_atac_seurat_post, cell_labels2)

plot_backbone(z_rna_seurat_post, z_atac_seurat_post, mode = "joint", mean_cluster = np.array(mean_cluster), groups = groups, T = T, figsize=(15,7), save = path + "backbone_post.png", anno = cell_labels, axis_label = "PCA")
utils.plot_latent_pt(z1 = z_rna_seurat_post, z2 = z_atac_seurat_post, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = path + "z_pt_post.png", figsize = (15,7), axis_label = "PCA")


# coembed_full = pd.read_csv(path + "pca_embedding_full.txt", sep = "\t").values
# z_rna_seurat = coembed_full[:anno_rna.shape[0],:]
# z_atac_seurat = coembed_full[anno_rna.shape[0]:,:]
# utils.plot_latent(z_rna_seurat, z_atac_seurat, anno_rna, anno_atac, mode = "modality", figsize = (15,7), axis_label = "PCA", save = path + "pca_full.png")
# utils.plot_latent(z_rna_seurat, z_atac_seurat, anno_rna, anno_atac, mode = "joint", figsize = (15,7), axis_label = "PCA", save = path + "pca_joint_full.png")
# # Infer backbone
# root_cell = 188
# dpt_mtx = ti.dpt(np.concatenate((z_rna_seurat, z_atac_seurat), axis = 0), n_neigh = 10)
# pt_infer = dpt_mtx[root_cell, :]
# pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
# pt_infer = pt_infer/np.max(pt_infer)
# # for scRNA-Seq batch
# pt_infer_rna = pt_infer[:z_rna_seurat.shape[0]]
# # for scATAC-Seq batch
# pt_infer_atac = pt_infer[z_rna_seurat.shape[0]:]

# cell_labels = np.concatenate((rna_dataset.cell_labels, atac_dataset.cell_labels), axis = 0).squeeze()
# cell_labels2 = np.where((cell_labels == "HSC") | (cell_labels == "MPP"), "HSC&MPP", cell_labels)
# groups, mean_cluster, T = backbone_inf(z_rna_seurat, z_atac_seurat, cell_labels2)

# plot_backbone(z_rna_seurat, z_atac_seurat, mode = "joint", mean_cluster = np.array(mean_cluster), groups = groups, T = T, figsize=(15,7), save = path + "backbone_full.png", anno = cell_labels, axis_label = "PCA")
# utils.plot_latent_pt(z1 = z_rna_seurat, z2 = z_atac_seurat, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = path + "z_pt_full.png", figsize = (15,7), axis_label = "PCA")


# In[] 3. UnionCom
path = "results_hema/unioncom/"
z_rna_unioncom = np.load(path + "lat_rna.npy")
z_atac_unioncom = np.load(path + "lat_atac.npy")
integrated_data = (z_rna_unioncom, z_atac_unioncom)

pca_op = PCA(n_components = 2)
umap_op = UMAP(n_components = 2)

pca_latent = pca_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))
umap_latent = umap_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))

utils.plot_latent(umap_latent[:z_rna_unioncom.shape[0],:], umap_latent[z_rna_unioncom.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "joint", save = path + "unioncom_umap.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(umap_latent[:z_rna_unioncom.shape[0],:], umap_latent[z_rna_unioncom.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "modality", save = path + "unioncom_batches_umap.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(pca_latent[:z_rna_unioncom.shape[0],:], pca_latent[z_rna_unioncom.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "joint", save = path + "unioncom_pca.png", figsize = (15,7), axis_label = "PCA")
utils.plot_latent(pca_latent[:z_rna_unioncom.shape[0],:], pca_latent[z_rna_unioncom.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "modality", save = path + "unioncom_batches_pca.png", figsize = (15,7), axis_label = "PCA")

z_destiny = np.load(path + "z_diffmap.npy")
utils.plot_latent(z_destiny[:z_rna_unioncom.shape[0],:], z_destiny[z_rna_unioncom.shape[0]:,:], anno1 = rna_dataset.cell_labels, anno2 = atac_dataset.cell_labels, 
mode = "joint", save = path + "unioncom_destiny.png", figsize = (15,7), axis_label = "Diffmap")
utils.plot_latent(z_destiny[:z_rna_unioncom.shape[0],:], z_destiny[z_rna_unioncom.shape[0]:,:], anno1 = rna_dataset.cell_labels, anno2 = atac_dataset.cell_labels, 
mode = "modality", save = path + "unioncom_batches_destiny.png", figsize = (15,7), axis_label = "Diffmap")

# Infer backbone
root_cell = 188
dpt_mtx = ti.dpt(np.concatenate((z_rna_unioncom, z_atac_unioncom), axis = 0), n_neigh = 10)
pt_infer = dpt_mtx[root_cell, :]
pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
pt_infer = pt_infer/np.max(pt_infer)
# for scRNA-Seq batch
pt_infer_rna = pt_infer[:z_rna_unioncom.shape[0]]
# for scATAC-Seq batch
pt_infer_atac = pt_infer[z_rna_unioncom.shape[0]:]

pca_op = PCA(n_components = 2)
z = pca_op.fit_transform(np.concatenate((z_rna_unioncom, z_atac_unioncom), axis = 0))
z_rna_pca = z[:z_rna_unioncom.shape[0],:]
z_atac_pca = z[z_rna_unioncom.shape[0]:,:]

cell_labels = np.concatenate((rna_dataset.cell_labels, atac_dataset.cell_labels), axis = 0).squeeze()
cell_labels2 = np.where((cell_labels == "HSC") | (cell_labels == "MPP"), "HSC&MPP", cell_labels)
groups, mean_cluster, T = backbone_inf(z_rna_unioncom, z_atac_unioncom, cell_labels2)
mean_cluster = pca_op.transform(np.array(mean_cluster))

plot_backbone(z_rna_pca, z_atac_pca, mode = "joint", mean_cluster = mean_cluster, groups = groups, T = T, figsize=(15,7), save = path + "backbone.png", anno = cell_labels, axis_label = "PCA")
utils.plot_latent_pt(z1 = z_rna_pca, z2 = z_atac_pca, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = path + "z_pt.png", figsize = (15,7), axis_label = "PCA")

# Include post-processing
z_rna_unioncom_post = torch.FloatTensor(z_rna_unioncom)
z_atac_unioncom_post = torch.FloatTensor(z_atac_unioncom)
z_rna_unioncom_post, z_atac_unioncom_post = palign.match_alignment(z_rna = z_rna_unioncom_post, z_atac = z_atac_unioncom_post, k = 10)
z_atac_unioncom_post, z_rna_unioncom_post = palign.match_alignment(z_rna = z_atac_unioncom_post, z_atac = z_rna_unioncom_post, k = 10)
z_rna_unioncom_post = z_rna_unioncom_post.numpy()
z_atac_unioncom_post = z_atac_unioncom_post.numpy()

integrated_data = (z_rna_unioncom_post, z_atac_unioncom_post)
pca_op = PCA(n_components = 2)
umap_op = UMAP(n_components = 2)
pca_latent = pca_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))
umap_latent = umap_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))


utils.plot_latent(umap_latent[:z_rna_unioncom.shape[0],:], umap_latent[z_rna_unioncom.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "joint", save = path + "unioncom_umap_post.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(umap_latent[:z_rna_unioncom.shape[0],:], umap_latent[z_rna_unioncom.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "modality", save = path + "unioncom_batches_umap_post.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(pca_latent[:z_rna_unioncom.shape[0],:], pca_latent[z_rna_unioncom.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "joint", save = path + "unioncom_pca_post.png", figsize = (15,7), axis_label = "PCA")
utils.plot_latent(pca_latent[:z_rna_unioncom.shape[0],:], pca_latent[z_rna_unioncom.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "modality", save = path + "unioncom_batches_pca_post.png", figsize = (15,7), axis_label = "PCA")

# Infer backbone
root_cell = 188
dpt_mtx = ti.dpt(np.concatenate((z_rna_unioncom_post, z_atac_unioncom_post), axis = 0), n_neigh = 10)
pt_infer = dpt_mtx[root_cell, :]
pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
pt_infer = pt_infer/np.max(pt_infer)
# for scRNA-Seq batch
pt_infer_rna = pt_infer[:z_rna_unioncom_post.shape[0]]
# for scATAC-Seq batch
pt_infer_atac = pt_infer[z_rna_unioncom_post.shape[0]:]
pca_op = PCA(n_components = 2)
z = pca_op.fit_transform(np.concatenate((z_rna_unioncom_post, z_atac_unioncom_post), axis = 0))
z_rna_pca = z[:z_rna_unioncom_post.shape[0],:]
z_atac_pca = z[z_rna_unioncom_post.shape[0]:,:]
cell_labels = np.concatenate((rna_dataset.cell_labels, atac_dataset.cell_labels), axis = 0).squeeze()
cell_labels2 = np.where((cell_labels == "HSC") | (cell_labels == "MPP"), "HSC&MPP", cell_labels)
groups, mean_cluster, T = backbone_inf(z_rna_unioncom_post, z_atac_unioncom_post, cell_labels2)
mean_cluster = pca_op.transform(np.array(mean_cluster))

plot_backbone(z_rna_pca, z_atac_pca, mode = "joint", mean_cluster = mean_cluster, groups = groups, T = T, figsize=(15,7), save = path + "backbone_post.png", anno = cell_labels, axis_label = "PCA")
utils.plot_latent_pt(z1 = z_rna_pca, z2 = z_atac_pca, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = path + "z_pt_post.png", figsize = (15,7), axis_label = "PCA")

# In[] 4. scJoint
path = "results_hema/scJoint_hema_raw_traj/"
counts_rna = pd.read_csv("../data/hema/counts_rna.csv", index_col = 0)
counts_atac = pd.read_csv("../data/hema/counts_atac.csv", index_col = 0)
anno_rna = pd.read_csv("../data/hema/anno_rna.txt", header = None)
anno_rna.index = counts_rna.index.values
anno_atac = pd.read_csv("../data/hema/anno_atac.txt", header = None)
anno_atac.index = counts_atac.index.values

z_atac_scJoint = pd.read_csv(path + "counts_atac_embeddings.txt", sep = " ", header = None).values
z_rna_scJoint = pd.read_csv(path + "counts_rna_embeddings.txt", sep = " ", header = None).values

integrated_data = [z_rna_scJoint, z_atac_scJoint]
pca_op = PCA(n_components = 2)
umap_op = UMAP(n_components = 2, random_state = 0)

pca_latent = pca_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))
umap_latent = umap_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))


utils.plot_latent(umap_latent[:z_rna_scJoint.shape[0],:], umap_latent[z_rna_scJoint.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "joint", save = path + "scjoint_umap.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(umap_latent[:z_rna_scJoint.shape[0],:], umap_latent[z_rna_scJoint.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "modality", save = path + "scjoint_batches_umap.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(pca_latent[:z_rna_scJoint.shape[0],:], pca_latent[z_rna_scJoint.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "joint", save = path + "scjoint_pca.png", figsize = (15,7), axis_label = "PCA")
utils.plot_latent(pca_latent[:z_rna_scJoint.shape[0],:], pca_latent[z_rna_scJoint.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "modality", save = path + "scjoint_batches_pca.png", figsize = (15,7), axis_label = "PCA")

z_destiny = np.load(path + "z_diffmap.npy")
utils.plot_latent(z_destiny[:z_rna_unioncom.shape[0],:], z_destiny[z_rna_unioncom.shape[0]:,:], anno1 = rna_dataset.cell_labels, anno2 = atac_dataset.cell_labels, 
mode = "joint", save = path + "scjoint_destiny.png", figsize = (15,7), axis_label = "Diffmap")
utils.plot_latent(z_destiny[:z_rna_unioncom.shape[0],:], z_destiny[z_rna_unioncom.shape[0]:,:], anno1 = rna_dataset.cell_labels, anno2 = atac_dataset.cell_labels, 
mode = "modality", save = path + "scjoint_batches_destiny.png", figsize = (15,7), axis_label = "Diffmap")

# Infer backbone
root_cell = 188
dpt_mtx = ti.dpt(np.concatenate((z_rna_scJoint, z_atac_scJoint), axis = 0), n_neigh = 10)
pt_infer = dpt_mtx[root_cell, :]
pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
pt_infer = pt_infer/np.max(pt_infer)
# for scRNA-Seq batch
pt_infer_rna = pt_infer[:z_rna_scJoint.shape[0]]
# for scATAC-Seq batch
pt_infer_atac = pt_infer[z_rna_scJoint.shape[0]:]

pca_op = PCA(n_components = 2)
z = pca_op.fit_transform(np.concatenate((z_rna_scJoint, z_atac_scJoint), axis = 0))
z_rna_pca = z[:z_rna_scJoint.shape[0],:]
z_atac_pca = z[z_rna_scJoint.shape[0]:,:]

cell_labels = np.concatenate((rna_dataset.cell_labels, atac_dataset.cell_labels), axis = 0).squeeze()
cell_labels2 = np.where((cell_labels == "HSC") | (cell_labels == "MPP"), "HSC&MPP", cell_labels)
groups, mean_cluster, T = backbone_inf(z_rna_scJoint, z_atac_scJoint, cell_labels2)
mean_cluster_pca = pca_op.transform(np.array(mean_cluster))

plot_backbone(z_rna_pca, z_atac_pca, mode = "joint", mean_cluster = mean_cluster_pca, groups = groups, T = T, figsize=(15,7), save = path + "backbone_full.png", anno = cell_labels, axis_label = "PCA")
utils.plot_latent_pt(z1 = z_rna_pca, z2 = z_atac_pca, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = path + "z_pt_full.png", figsize = (15,7), axis_label = "PCA")

# z = umap_op.fit_transform(np.concatenate((z_rna_scJoint, z_atac_scJoint), axis = 0))
z_rna_umap = umap_latent[:z_rna_scJoint.shape[0],:]
z_atac_umap = umap_latent[z_rna_scJoint.shape[0]:,:]

mean_cluster_umap = umap_op.transform(np.array(mean_cluster))

plot_backbone(z_rna_umap, z_atac_umap, mode = "joint", mean_cluster = mean_cluster_umap, groups = groups, T = T, figsize=(15,7), save = path + "backbone_full_umap.png", anno = cell_labels, axis_label = "UMAP")
utils.plot_latent_pt(z1 = z_rna_umap, z2 = z_atac_umap, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = path + "z_pt_full_umap.png", figsize = (15,7), axis_label = "UMAP")

# In[]
# Include post-processing
z_rna_scJoint_post = torch.FloatTensor(z_rna_scJoint)
z_atac_scJoint_post = torch.FloatTensor(z_atac_scJoint)
z_rna_scJoint_post, z_atac_scJoint_post = palign.match_alignment(z_rna = z_rna_scJoint_post, z_atac = z_atac_scJoint_post, k = 10)
z_atac_scJoint_post, z_rna_scJoint_post = palign.match_alignment(z_rna = z_atac_scJoint_post, z_atac = z_rna_scJoint_post, k = 10)
z_rna_scJoint_post = z_rna_scJoint_post.numpy()
z_atac_scJoint_post = z_atac_scJoint_post.numpy()

integrated_data = (z_rna_scJoint_post, z_atac_scJoint_post)
pca_op = PCA(n_components = 2)
umap_op = UMAP(n_components = 2)
pca_latent = pca_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))
umap_latent = umap_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))


utils.plot_latent(umap_latent[:z_rna_scJoint_post.shape[0],:], umap_latent[z_rna_scJoint_post.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "joint", save = path + "scjoint_umap_post.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(umap_latent[:z_rna_scJoint_post.shape[0],:], umap_latent[z_rna_scJoint_post.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "modality", save = path + "scjoint_batches_umap_post.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(pca_latent[:z_rna_scJoint_post.shape[0],:], pca_latent[z_rna_scJoint_post.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "joint", save = path + "scjoint_pca_post.png", figsize = (15,7), axis_label = "PCA")
utils.plot_latent(pca_latent[:z_rna_scJoint_post.shape[0],:], pca_latent[z_rna_scJoint_post.shape[0]:,:], anno1 = anno_rna.values, anno2 = anno_atac.values, 
mode = "modality", save = path + "scjoint_batches_pca_post.png", figsize = (15,7), axis_label = "PCA")

# Infer backbone
root_cell = 188
dpt_mtx = ti.dpt(np.concatenate((z_rna_scJoint_post, z_atac_scJoint_post), axis = 0), n_neigh = 10)
pt_infer = dpt_mtx[root_cell, :]
pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
pt_infer = pt_infer/np.max(pt_infer)
# for scRNA-Seq batch
pt_infer_rna = pt_infer[:z_rna_scJoint_post.shape[0]]
# for scATAC-Seq batch
pt_infer_atac = pt_infer[z_rna_scJoint_post.shape[0]:]
pca_op = PCA(n_components = 2)
z = pca_op.fit_transform(np.concatenate((z_rna_scJoint_post, z_atac_scJoint_post), axis = 0))
z_rna_pca = z[:z_rna_scJoint_post.shape[0],:]
z_atac_pca = z[z_rna_scJoint_post.shape[0]:,:]
cell_labels = np.concatenate((rna_dataset.cell_labels, atac_dataset.cell_labels), axis = 0).squeeze()
cell_labels2 = np.where((cell_labels == "HSC") | (cell_labels == "MPP"), "HSC&MPP", cell_labels)
groups, mean_cluster, T = backbone_inf(z_rna_scJoint_post, z_atac_scJoint_post, cell_labels2)
mean_cluster = pca_op.transform(np.array(mean_cluster))

plot_backbone(z_rna_pca, z_atac_pca, mode = "joint", mean_cluster = mean_cluster, groups = groups, T = T, figsize=(15,7), save = path + "backbone_post.png", anno = cell_labels, axis_label = "PCA")
utils.plot_latent_pt(z1 = z_rna_pca, z2 = z_atac_pca, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = path + "z_pt_post.png", figsize = (15,7), axis_label = "PCA")


# In[] Infer pseudotime
counts_rna = pd.read_csv("../data/hema/counts_rna.csv", index_col = 0)
counts_atac = pd.read_csv("../data/hema/counts_atac.csv", index_col = 0)

# infer backbone with leiden clustering
groups, mean_cluster, T = ti.backbone_inf(np.concatenate((z_rna, z_atac), axis = 0), resolution = 0.05)
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
utils.plot_backbone(z_rna_pca, z_atac_pca, groups = groups, T = T, mean_cluster = mean_cluster, mode = "joint", figsize=(15,7), save = None, axis_label = "PCA")

pseudo_order_rna = np.empty((groups_rna.shape[0], len(paths)))
pseudo_order_rna[:] = np.nan
pseudo_order_rna = pd.DataFrame(data = pseudo_order_rna, index = counts_rna.index.values, columns = np.array(["traj_" + str(x) for x in range(len(paths))]))
for i, path in enumerate(paths):
    selected_cells = np.concatenate([np.where(groups_rna == x)[0] for x in path], axis = 0)
    pseudo_order_rna.iloc[selected_cells, i] = pt_infer_rna[selected_cells]

pseudo_order_atac = np.empty((groups_atac.shape[0], len(paths)))
pseudo_order_atac[:] = np.nan
pseudo_order_atac = pd.DataFrame(data = pseudo_order_atac, index = counts_atac.index.values, columns = np.array(["traj_" + str(x) for x in range(len(paths))]))
for i, path in enumerate(paths):
    selected_cells = np.concatenate([np.where(groups_atac == x)[0] for x in path], axis = 0)
    pseudo_order_atac.iloc[selected_cells, i] = pt_infer_atac[selected_cells]

# Overall pseudo-order
pseudo_order = pd.concat((pseudo_order_rna, pseudo_order_atac), axis = 0, ignore_index = False)
pseudo_order.to_csv("results_hema/pseudo_order.csv")

fig, axs = plot_pt(z = z, pseudo_order = pseudo_order, ncols = 2, figsize = (20,7), basis = "PCA")
fig.savefig("results_hema/z_pt_sep.png", bbox_inches = "tight")



# In[] Find de genes, normal distribution after log-transform
# MEP: DOWN CD99, IGLL1, GAPDH, STMN1, SPINK2; UP: GATA2, 
# CLP: SATB1, RUNX2, 
# pseudo_order = pd.read_csv("results_hema/pseudo_order.csv", index_col = 0)
# pseudo_order_rna = pseudo_order.iloc[:z_rna.shape[0],:]
# pseudo_order_atac = pseudo_order.iloc[z_rna.shape[0]:,:]

de_genes = de.de_analy_para(X = counts_rna, pseudo_order = pseudo_order_rna, p_val_t = 0.05, verbose = False, distri = "normal", fdr_correct = True, n_jobs = 4)
for traj in de_genes.keys():
    genes = np.array([x["feature"] for x in de_genes[traj]])
    p_val = np.array([x["p_val"] for x in de_genes[traj]])
    genes= genes[np.argsort(p_val)]
    p_val = p_val[np.argsort(p_val)]
    de_list = pd.DataFrame.from_dict({"feature": genes, "p-val": p_val})
    de_list.to_csv("./results_hema/de_hema/de_gene_" + str(traj) + ".csv")

# de.de_plot(X = counts_rna, pseudo_order = pseudo_order_rna, de_feats = de_genes, figsize = (20,50), n_feats = 20)

ncols = 2
nrows = 4
genes = ["KLF", "CD99", "IGLL1", "GAPDH", "STMN1", "SPINK", "GATA2", "SATB1", "RUNX2"]
for traj in pseudo_order_rna.columns.values:
    idxs = []
    genes_traj = []
    for idx, feat in enumerate(de_genes[traj]):
        if feat["feature"] in genes:
            idxs.append(idx)

    # ordering of genes
    sorted_pt = pseudo_order_rna[traj].dropna(axis = 0).sort_values()
    # ordering = [int(x.split("_")[1]) for x in sorted_pt.index]
    ordering = sorted_pt.index.values.squeeze()
    X_traj = counts_rna.loc[ordering, :]

    # make plot
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (15, 15))
    colormap = plt.cm.get_cmap('tab20b', nrows * ncols)
    for idx, gene_id in enumerate(idxs):
        feat = de_genes[traj][gene_id]
        # plot log transformed version
        y = np.squeeze(X_traj.loc[:,feat["feature"]].values)
        y_null = feat['null']
        y_pred = feat['regression']

        axs[idx%nrows, idx//nrows].scatter(np.arange(y.shape[0]), y, color = colormap(1), alpha = 0.5)
        axs[idx%nrows, idx//nrows].plot(np.arange(y.shape[0]), y_pred, color = "black", alpha = 1, linewidth = 4)
        axs[idx%nrows, idx//nrows].plot(np.arange(y.shape[0]), y_null, color = "red", alpha = 1, linewidth = 4)
        axs[idx%nrows, idx//nrows].set_title(feat["feature"])   
    
    plt.tight_layout()
    fig.savefig("results_hema/de_hema/de_genes_" + str(traj[-1]) + ".png", bbox_inches = "tight")



# In[] Find de regions, binomial distribution
'''
de_regions = de.de_analy_para(X = counts_atac, pseudo_order = pseudo_order_atac, p_val_t = 0.05, verbose = False, distri = "binomial", fdr_correct = True)
for traj in de_regions.keys():
    regions = np.array([x["feature"] for x in de_regions[traj]])
    p_val = np.array([x["p_val"] for x in de_regions[traj]])
    regions= regions[np.argsort(p_val)]
    p_val = p_val[np.argsort(p_val)]
    de_list = pd.DataFrame.from_dict({"feature": regions, "p-val": p_val})
    de_list.to_csv("./results_hema/de_hema/de_region_" + str(traj) + ".csv")

de.de_plot(X = counts_atac, pseudo_order = pseudo_order_atac, de_feats = de_regions, figsize = (20,50), n_feats = 20)
'''

# In[] Find de motif
# HOX, ERG, MAFF for HSC cell. HOX: regulate stem cell activity. Enriched in HSC cells, but not in other differentiated cells.
# GATA1: erythroid (MEP)
# CEBPD: myeloid
# EBF1: lymphoid (CLP)
# ID3: pDC, CLP
# JUN, FOX: patient specific batch effect

counts_motif = pd.read_csv("../data/hema/chromVAR/motif_z.csv", index_col = 0)

# Find de regions, binomial distribution
de_motifs = de.de_analy_para(X = counts_motif, pseudo_order = pseudo_order_atac, p_val_t = 0.05, verbose = False, distri = "normal", fdr_correct = True)
for traj in de_motifs.keys():
    motifs = np.array([x["feature"] for x in de_motifs[traj]])
    p_val = np.array([x["p_val"] for x in de_motifs[traj]])
    motifs= motifs[np.argsort(p_val)]
    p_val = p_val[np.argsort(p_val)]
    de_list = pd.DataFrame.from_dict({"feature": motifs, "p-val": p_val})
    de_list.to_csv("./results_hema/de_hema/de_motif_" + str(traj) + ".csv")

figs = de.de_plot(X = counts_motif, pseudo_order = pseudo_order_atac, de_feats = de_motifs, figsize = (20,50), n_feats = 20)


# # In[] Key motif in the paper

# import seaborn as sns

# counts_motif = pd.read_csv("../data/hema/chromVAR/motif_z.csv", index_col = 0)
# z_rna = np.load(file = "./results_hema/z_rna.npy")
# z_atac = np.load(file = "./results_hema/z_atac.npy")
# z_rna = torch.FloatTensor(z_rna)
# z_atac = torch.FloatTensor(z_atac)

# z_rna, z_atac = palign.match_alignment(z_rna = z_rna, z_atac = z_atac, k = 10)
# z_atac, z_rna = palign.match_alignment(z_rna = z_atac, z_atac = z_rna, k = 10)
# z_rna = z_rna.numpy()
# z_atac = z_atac.numpy()

# pca_op = PCA(n_components = 2)
# z = pca_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
# z_rna_pca = z[:z_rna.shape[0],:]
# z_atac_pca = z[z_rna.shape[0]:,:]

# # 1st, HOX
# motifs = counts_motif.columns.values
# TFs = np.array([x.split("_")[1] for x in motifs])

# fig = plt.figure(figsize = (20, 10))
# ax = fig.add_subplot()
# EBF1 = ax.scatter(z_atac_pca[:, 0], z_atac_pca[:, 1], c = counts_motif['MA0901.1_HOXB13'].values.squeeze())
# cbar = fig.colorbar(EBF1, fraction=0.046, pad=0.04, ax = ax)

# idices = ["MA0154.3_EBF1"]
# for traj in pseudo_order_atac.columns.values:
#     sorted_pt = pseudo_order_atac[traj].dropna(axis = 0).sort_values()
#     ordering = sorted_pt.index.values.squeeze()

#     counts_traj = counts_motif.loc[ordering, :]
#     heatmap_data = pd.DataFrame(data = counts_traj[idices], index = counts_traj.index.values)
#     fig = plt.figure(figsize = (20,10))
#     ax = fig.add_subplot()
#     sns.heatmap(heatmap_data, ax = ax)




# %%
