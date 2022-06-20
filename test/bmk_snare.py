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
import utils as utils
import post_align as palign

from umap import UMAP
from scipy.sparse import load_npz
from scipy.stats import pearsonr, spearmanr
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

from adjustText import adjust_text
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import scanpy as sc 
import anndata 

plt.rcParams["font.size"] = 20

# In[] scan and find the one with the highest neighborhood overlap score
seeds = [0, 1, 2]
latent_dims = [4, 8, 32]
reg_ds = [1, 10]
reg_gs = [0.01, 1, 10]
reg_mmds = [1, 10, 20, 30]

latent_dim = latent_dims[0]
reg_d = reg_ds[0]
reg_g = reg_gs[1]
# harder to merge, need to make mmd loss larger
reg_mmd = reg_mmds[1]
seed = seeds[0]

learning_rate = 3e-4
n_epochs = 500
use_anchor = False
ts = [30, 50, 70]
use_potential = True
norm = "l1"

counts_rna = pd.read_csv("../data/snare-seq-1000/counts_rna.csv", index_col = 0)
counts_atac = pd.read_csv("../data/snare-seq-1000/counts_atac.csv", index_col = 0)
label_rna = pd.read_csv("../data/snare-seq-1000/anno.txt", header = None)
label_atac = pd.read_csv("../data/snare-seq-1000/anno.txt", header = None)
rna_dataset = dataset.dataset(counts = counts_rna.values, anchor = None)
atac_dataset = dataset.dataset(counts = counts_atac.values, anchor = None)
coarse_reg = torch.FloatTensor(pd.read_csv("../data/snare-seq-1000/region2gene.csv", sep = ",", index_col = 0).values).to(device)

batch_size = int(max([len(rna_dataset),len(atac_dataset)])/4)


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

# initialize the model
gene_act = model.gene_act(features = EMBED_CONFIG["gact_layers"], dropout_rate = 0.0, negative_slope = 0.2).to(device)
encoder = model.Encoder(features = EMBED_CONFIG["proj_layers"], dropout_rate = 0.0, negative_slope = 0.2).to(device)
model_dict = {"gene_act": gene_act, "encoder": encoder}

model_dict = torch.load("../test/results_snare/models_1000/model_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + ".pth", map_location = device)

with torch.no_grad():
    z_rna = model_dict["encoder"](rna_dataset.counts.to(device))
    z_atac = model_dict["encoder"](model_dict["gene_act"](atac_dataset.counts.to(device)))

pca_op = PCA(n_components = 2)
z = pca_op.fit_transform(np.concatenate((z_rna.cpu().numpy(), z_atac.cpu().numpy()), axis = 0))
z_rna_pca = z[:z_rna.shape[0],:]
z_atac_pca = z[z_rna.shape[0]:,:]
utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = label_rna, 
                    anno2 = label_atac, mode = "joint", save = "results_snare/z_joint.png", 
                    figsize = (15,7), axis_label = "PCA")
utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = label_rna, 
                    anno2 = label_atac, mode = "modality", save = "results_snare/z_mod.png", 
                    figsize = (15,7), axis_label = "PCA")


z_rna, z_atac = palign.match_alignment(z_rna = z_rna.cpu(), z_atac = z_atac.cpu(), k = 10)
z_atac, z_rna = palign.match_alignment(z_rna = z_atac.cpu(), z_atac = z_rna.cpu(), k = 10)
z_rna = z_rna.cpu().numpy()
z_atac = z_atac.cpu().numpy()

# post-maching
pca_op = PCA(n_components = 2)
z = pca_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
z_rna_pca = z[:z_rna.shape[0],:]
z_atac_pca = z[z_rna.shape[0]:,:]
utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = label_rna, 
                    anno2 = label_atac, mode = "joint", save = "results_snare/z_joint_post.png", 
                    figsize = (15,7), axis_label = "PCA")
utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = label_rna, 
                    anno2 = label_atac, mode = "modality", save = "results_snare/z_mod_post.png", 
                    figsize = (15,7), axis_label = "PCA")

umap_op = UMAP(n_components = 2, min_dist = 0.8, random_state = 0)
z = umap_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
z_rna_umap = z[:z_rna.shape[0],:]
z_atac_umap = z[z_rna.shape[0]:,:]
utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = label_rna, 
                    anno2 = label_atac, mode = "joint", save = "results_snare/z_joint_post_umap.png", 
                    figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = label_rna, 
                    anno2 = label_atac, mode = "modality", save = "results_snare/z_mod_post_umap.png", 
                    figsize = (15,7), axis_label = "UMAP")


# # run diffusion map
# adata_scdart = anndata.AnnData(X = np.concatenate((z_rna,z_atac), axis = 0))
# # adata_scdart = anndata.AnnData(X = np.concatenate((z_rna_pca,z_atac_pca), axis = 0))

# sc.pp.neighbors(adata_scdart, use_rep = 'X', n_neighbors = 30, random_state = 0)
# sc.tl.diffmap(adata_scdart, random_state = 0)
# diffmap_latent = adata_scdart.obsm["X_diffmap"]
# utils.plot_latent(diffmap_latent[:z_rna.shape[0],:], diffmap_latent[z_rna.shape[0]:,:], anno1 = rna_dataset.cell_labels, anno2 = atac_dataset.cell_labels, 
# mode = "joint", save = "results_snare/z_joint_post_diffmap.png", figsize = (15,7), axis_label = "Diffmap")
# utils.plot_latent(diffmap_latent[:z_rna.shape[0],:], diffmap_latent[z_rna.shape[0]:,:], anno1 = rna_dataset.cell_labels, anno2 = atac_dataset.cell_labels, 
# mode = "modality", save = "results_snare/z_mod_post_diffmap.png", figsize = (15,7), axis_label = "Diffmap")

z_destiny = np.load("results_snare/models_1000/z_diffmap.npy")
utils.plot_latent(z_destiny[:z_rna.shape[0],:], z_destiny[z_rna.shape[0]:,:], anno1 = label_rna, anno2 = label_atac, 
mode = "joint", save = "results_snare/z_joint_post_destiny.png", figsize = (15,7), axis_label = "Diffmap")
utils.plot_latent(z_destiny[:z_rna.shape[0],:], z_destiny[z_rna.shape[0]:,:], anno1 = label_rna, anno2 = label_atac, 
mode = "modality", save = "results_snare/z_mod_post_destiny.png", figsize = (15,7), axis_label = "Diffmap")

# In[] Plot backbone
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


# root, manually found
root_cell = 450
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

cell_labels = np.concatenate((label_rna, label_atac), axis = 0).squeeze()
groups, mean_cluster, T = backbone_inf(z_rna, z_atac, cell_labels)
mean_cluster_pca = pca_op.transform(np.array(mean_cluster))
# # mean_cluster_pca
# mean_cluster_pca = [[] for x in range(len(mean_cluster))]
# for i, cat in enumerate(np.sort(np.unique(cell_labels))):
#     idx = np.where(cell_labels == cat)[0]
#     mean_cluster_pca[i] = np.mean(z[idx,:], axis = 0)
# mean_cluster_pca = np.array(mean_cluster_pca)


plot_backbone(z_rna_pca, z_atac_pca, mode = "joint", mean_cluster = mean_cluster_pca, groups = groups, T = T, figsize=(15,7), save = "results_snare/backbone.png", anno = cell_labels, axis_label = "PCA")
utils.plot_latent_pt(z1 = z_rna_pca, z2 = z_atac_pca, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = "./results_snare/z_pt.png", figsize = (15,7), axis_label = "PCA")


# In[] Infer pseudotime
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
utils.plot_backbone(z_rna_pca, z_atac_pca, groups = groups, T = T, mean_cluster = mean_cluster, mode = "joint", figsize=(10,7), save = None, axis_label = "PCA")

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
pseudo_order.to_csv("results_snare/pseudo_order.csv")

# In[] Find de genes
de_genes = de.de_analy_para(X = counts_rna, pseudo_order = pseudo_order_rna, p_val_t = 0.05, verbose = False, distri = "normal", fdr_correct = True, n_jobs = 4)
for traj in de_genes.keys():
    genes = np.array([x["feature"] for x in de_genes[traj]])
    p_val = np.array([x["p_val"] for x in de_genes[traj]])
    genes= genes[np.argsort(p_val)]
    p_val = p_val[np.argsort(p_val)]
    de_list = pd.DataFrame.from_dict({"feature": genes, "p-val": p_val})
    de_list.to_csv("./results_snare/de_snare/de_gene_" + str(traj) + ".csv")

genes = ["Mki67", "Fabp7", "Eomes", "Unc5d", "Cux1", "Foxp1"]
ncols = 2
nrows = np.ceil(len(genes)/2).astype('int32')
figsize = (20,15)
X = counts_rna
de_feats = de_genes
pseudo_order = pseudo_order_rna

figs = []
for traj_i in de_feats.keys():
    # ordering of genes
    sorted_pt = pseudo_order[traj_i].dropna(axis = 0).sort_values()
    # ordering = [int(x.split("_")[1]) for x in sorted_pt.index]
    ordering = sorted_pt.index.values.squeeze()
    X_traj = X.loc[ordering, :]

    # make plot
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)
    colormap = plt.cm.get_cmap('tab20b', 20)
    idx = 0
    for feat in de_feats[traj_i]:
        if feat["feature"] in genes:
            # plot log transformed version
            y = np.squeeze(X_traj.loc[:,feat["feature"]].values)
            y_null = feat['null']
            y_pred = feat['regression']

            axs[idx%nrows, idx//nrows].scatter(np.arange(y.shape[0]), y, color = colormap(1), alpha = 0.5)
            axs[idx%nrows, idx//nrows].plot(np.arange(y.shape[0]), y_pred, color = "black", alpha = 1, linewidth = 4)
            axs[idx%nrows, idx//nrows].plot(np.arange(y.shape[0]), y_null, color = "red", alpha = 1, linewidth = 4)
            axs[idx%nrows, idx//nrows].set_title(feat["feature"]) 
            idx += 1               
    
    plt.tight_layout()
    figs.append(fig)
    fig.savefig("results_snare/de_snare/de_snare_" + str(traj_i) + ".png", bbox_inches = "tight")

def plot_gene(z, counts, gene, save = None, figsize = (20,10), axis_label = "Latent", **kwargs):
    _kwargs = {
        "s": 10,
        "alpha": 0.9,
    }
    _kwargs.update(kwargs)

    fig = plt.figure(figsize = figsize)

    ax = fig.add_subplot()
    sct = ax.scatter(z[:,0], z[:,1], c = counts.loc[:, gene].values.squeeze(), cmap = plt.get_cmap('gnuplot'), **_kwargs)
    

    ax.tick_params(axis = "both", which = "major", labelsize = 15)

    ax.set_xlabel(axis_label + " 1", fontsize = 19)
    ax.set_ylabel(axis_label + " 2", fontsize = 19)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  

    cbar = fig.colorbar(sct,fraction=0.046, pad=0.04, ax = ax)
    cbar.ax.tick_params(labelsize = 20)

    if save:
        fig.savefig(save, bbox_inches = "tight")
    
    print(save)

for gene in genes:
    plot_gene(z_rna_pca, counts_rna, gene, save = None, figsize = (20,10), axis_label = "Latent")


# In[] Find de motif
counts_motif = pd.read_csv("../data/snare-seq-1000/chromVAR/motif_z.csv", index_col = 0)

# Find de regions, binomial distribution
de_motifs = de.de_analy_para(X = counts_motif, pseudo_order = pseudo_order_atac, p_val_t = 0.05, verbose = False, distri = "normal", fdr_correct = True)
for traj in de_motifs.keys():
    motifs = np.array([x["feature"] for x in de_motifs[traj]])
    p_val = np.array([x["p_val"] for x in de_motifs[traj]])
    motifs= motifs[np.argsort(p_val)]
    p_val = p_val[np.argsort(p_val)]
    de_list = pd.DataFrame.from_dict({"feature": motifs, "p-val": p_val})
    de_list.to_csv("./results_snare/de_snare/de_motif_" + str(traj) + ".csv")

figs = de.de_plot(X = counts_motif, pseudo_order = pseudo_order_atac, de_feats = de_motifs, figsize = (20,50), n_feats = 20)

# In[] Motif by gene matrix
# extract gene activity matrix (region, gene)
GACT = train.infer_gact(model_dict["gene_act"], mask = (coarse_reg != 0), thresh = 1e-6).cpu().numpy()
# transform into (motif, gene)
# read in the region2motif matrix, fill in the empty regions
region2motif = pd.read_csv("../data/snare-seq-1000/chromVAR/region2motif.csv", sep = ",", index_col = 0)
region2motif_full = pd.DataFrame(index = counts_atac.columns.values, columns = region2motif.columns.values, data = 0)
region2motif_full.loc[region2motif.index.values, region2motif.columns.values] = region2motif.values


motif2gene = region2motif_full.values.T @ GACT
# check which motif is regulating which gene
motif2gene = pd.DataFrame(data = motif2gene, index = region2motif.columns.values, columns = counts_rna.columns.values)
motif2gene.to_csv("results_snare/de_snare/motif2gene_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + ".csv")

# # In[]
# # First check the Sox6
# gene = "Sox6"
# ordering = np.argsort(motif2gene[gene].values.squeeze())[::-1]
# motif2gene_ordered = motif2gene.iloc[ordering, :]
# motif2gene_ordered = motif2gene_ordered.loc[:, [gene]]
# motif2gene_ordered.to_csv("results_snare/de_snare/1000/motifs_" + gene + ".csv")

# In[]
from scipy.stats import zscore
pseudo_rna = model_dict["gene_act"](atac_dataset.counts.to(device)).detach().cpu().numpy()
pseudo_rna = zscore(pseudo_rna, axis = 0)
genes = ["Ptprd", "Mki67", "Fabp7", "Top2a", "Mef2c", "Macrod2", "Tenm2", "Dab1", "Tnc", "Frmd4a", "Celf2"]
genes = pd.read_csv("results_snare/de_snare/1000/de_gene_traj_0.csv", index_col = 0)
genes = genes.iloc[:50,0].values.squeeze()
pseudo_rna = pd.DataFrame(data = pseudo_rna, index = counts_rna.index.values, columns = counts_rna.columns.values)
pseudo_rna_sorted = pseudo_rna.iloc[np.argsort(pt_infer_rna), :]
rna_sorted = counts_rna.iloc[np.argsort(pt_infer_rna), :]
pseudo_rna_sorted = pseudo_rna_sorted.loc[:, genes]
rna_sorted = rna_sorted.loc[:, genes]
rna_sorted = zscore(rna_sorted, axis = 0)

fig = plt.figure(figsize = (20, 7))
axs = fig.subplots(1, 2)
sns.heatmap(pseudo_rna_sorted.T, ax = axs[0])
sns.heatmap(rna_sorted.T, ax = axs[1])
# fig.savefig("results_snare/de_snare/predict_rna.png", bbox_inches = "tight")

score_gact = pd.DataFrame(columns = ["Method", "Gene", "Spearman", "Pearson"])
for i, gene in enumerate(genes):
    spearman,_ = spearmanr(pseudo_rna_sorted.T.loc[gene,:], rna_sorted.T.loc[gene,:])
    if np.isnan(spearman):
        spearman = 0
    pearson,_ = pearsonr(pseudo_rna_sorted.T.loc[gene,:], rna_sorted.T.loc[gene,:])
    if np.isnan(pearson):
        pearson = 0
    # print("gene: {:s}, spearman: {:.4f}, pearson: {:.4f}".format(gene, spearman, pearson))
    score_gact = score_gact.append({"Method": "scDART", "Gene": gene, "Spearman": spearman, "Pearson": pearson}, ignore_index = True)

# linear method
pseudo_rna2 = (atac_dataset.counts.to(device) @ coarse_reg).detach().cpu().numpy()
pseudo_rna2 = zscore(pseudo_rna2, axis = 0)
pseudo_rna2 = pd.DataFrame(data = pseudo_rna2, index = counts_rna.index.values, columns = counts_rna.columns.values)
pseudo_rna2_sorted = pseudo_rna2.iloc[np.argsort(pt_infer_rna), :]
pseudo_rna2_sorted = pseudo_rna2_sorted.loc[:, genes]

for i, gene in enumerate(genes):
    spearman,_ = spearmanr(pseudo_rna2_sorted.T.loc[gene,:], rna_sorted.T.loc[gene,:])
    if np.isnan(spearman):
        spearman = 0
    pearson,_ = pearsonr(pseudo_rna2_sorted.T.loc[gene,:], rna_sorted.T.loc[gene,:])
    if np.isnan(pearson):
        pearson = 0
    # print("gene: {:s}, spearman: {:.4f}, pearson: {:.4f}".format(gene, spearman, pearson))
    score_gact = score_gact.append({"Method": "Linear", "Gene": gene, "Spearman": spearman, "Pearson": pearson}, ignore_index = True)

# Signac method
pseudo_rna_signac = pd.read_csv("results_snare/pseudoRNA/counts_rna_signac.csv", index_col = 0)
pseudo_rna_signac_sorted = pseudo_rna_signac.loc[rna_sorted.index.values,:]
pseudo_rna_signac_sorted = zscore(pseudo_rna_signac_sorted, axis = 0)
pseudo_rna_signac_sorted = pseudo_rna_signac_sorted.loc[:, genes]
pseudo_rna_signac_sorted = pseudo_rna_signac_sorted.fillna(0)

for i, gene in enumerate(genes):
    spearman,_ = spearmanr(pseudo_rna_signac_sorted.T.loc[gene,:], rna_sorted.T.loc[gene,:])
    if np.isnan(spearman):
        spearman = 0
    pearson,_ = pearsonr(pseudo_rna_signac_sorted.T.loc[gene,:], rna_sorted.T.loc[gene,:])
    if np.isnan(pearson):
        pearson = 0
    # print("gene: {:s}, spearman: {:.4f}, pearson: {:.4f}".format(gene, spearman, pearson))
    score_gact = score_gact.append({"Method": "Signac", "Gene": gene, "Spearman": spearman, "Pearson": pearson}, ignore_index = True)

# Cicero method
pseudo_rna_cicero = pd.read_csv("results_snare/pseudoRNA/counts_rna_cicero.csv", index_col = 0)
pseudo_rna_cicero = zscore(pseudo_rna_cicero, axis = 0)
pseudo_rna_signac = pd.DataFrame(data = pseudo_rna_cicero, index = counts_rna.index.values, columns = counts_rna.columns.values)
pseudo_rna_cicero_sorted = pseudo_rna_cicero.iloc[np.argsort(pt_infer_rna), :]
pseudo_rna_cicero_sorted = pseudo_rna_cicero_sorted.loc[:, genes]
pseudo_rna_cicero_sorted = pseudo_rna_cicero_sorted.fillna(0)

for i, gene in enumerate(genes):
    spearman,_ = spearmanr(pseudo_rna_cicero_sorted.T.loc[gene,:], rna_sorted.T.loc[gene,:])
    if np.isnan(spearman):
        spearman = 0
    pearson,_ = pearsonr(pseudo_rna_cicero_sorted.T.loc[gene,:], rna_sorted.T.loc[gene,:])
    if np.isnan(pearson):
        pearson = 0
    # print("gene: {:s}, spearman: {:.4f}, pearson: {:.4f}".format(gene, spearman, pearson))
    score_gact = score_gact.append({"Method": "Cicero", "Gene": gene, "Spearman": spearman, "Pearson": pearson}, ignore_index = True)


score_gact.to_csv("results_snare/de_snare/gact_acc.csv")

fig = plt.figure(figsize = (7, 7))
ax = fig.add_subplot()
x = score_gact.loc[score_gact["Method"] == "Linear", "Pearson"].values
y = score_gact.loc[score_gact["Method"] == "scDART", "Pearson"].values
ax.scatter(x, y)

print("proportion above: {:.2f}".format(np.sum((x < y).astype(int))/x.shape[0]) )
# for i in range(x.shape[0]):
#     marker = genes[i]
#     if len(marker) <= 3:
#         ax.plot(x[i] + 0.02, y[i] + 0.001, color = "black", marker=  "$" + marker + "$", markersize = 20)
#     elif len(marker) <= 5:
#         ax.plot(x[i] + 0.02, y[i] + 0.001, color = "black", marker=  "$" + marker + "$", markersize = 30)
#     else:
#         ax.plot(x[i] + 0.02, y[i] + 0.001, color = "black", marker=  "$" + marker + "$", markersize = 45)

# ax.set_yscale('log')
# ax.set_xscale('log')
ax.set_xlim([0.001, 0.4])
ax.set_ylim([0.001, 0.4])
# ax.set_xticks([0.01, 0.1])
# ax.set_yticks([0.01, 0.1])
ax.plot([0, 0.4], [0, 0.4], "r:")
ax.set_xlabel("Linear")
ax.set_ylabel("scDART")

fig.savefig("results_snare/de_snare/Pearson_linear.png", bbox_inches = "tight")


fig = plt.figure(figsize = (7, 7))
ax = fig.add_subplot()
x = score_gact.loc[score_gact["Method"] == "Signac", "Pearson"].values
y = score_gact.loc[score_gact["Method"] == "scDART", "Pearson"].values
ax.scatter(x, y)

print("proportion above: {:.2f}".format(np.sum((x < y).astype(int))/x.shape[0]) )
# for i in range(x.shape[0]):
#     marker = genes[i]
#     if len(marker) <= 3:
#         ax.plot(x[i] + 0.02, y[i] + 0.001, color = "black", marker=  "$" + marker + "$", markersize = 20)
#     elif len(marker) <= 5:
#         ax.plot(x[i] + 0.02, y[i] + 0.001, color = "black", marker=  "$" + marker + "$", markersize = 30)
#     else:
#         ax.plot(x[i] + 0.02, y[i] + 0.001, color = "black", marker=  "$" + marker + "$", markersize = 45)

# ax.set_yscale('log')
# ax.set_xscale('log')
ax.set_xlim([0.001, 0.4])
ax.set_ylim([0.001, 0.4])
# ax.set_xticks([0.01, 0.1])
# ax.set_yticks([0.01, 0.1])
ax.plot([0, 0.4], [0, 0.4], "r:")
ax.set_xlabel("Signac")
ax.set_ylabel("scDART")

fig.savefig("results_snare/de_snare/Pearson_signac.png", bbox_inches = "tight")


fig = plt.figure(figsize = (7, 7))
ax = fig.add_subplot()
x = score_gact.loc[score_gact["Method"] == "Cicero", "Pearson"].values
y = score_gact.loc[score_gact["Method"] == "scDART", "Pearson"].values
ax.scatter(x, y)

print("proportion above: {:.2f}".format(np.sum((x < y).astype(int))/x.shape[0]) )
# for i in range(x.shape[0]):
#     marker = genes[i]
#     if len(marker) <= 3:
#         ax.plot(x[i] + 0.02, y[i] + 0.001, color = "black", marker=  "$" + marker + "$", markersize = 20)
#     elif len(marker) <= 5:
#         ax.plot(x[i] + 0.02, y[i] + 0.001, color = "black", marker=  "$" + marker + "$", markersize = 30)
#     else:
#         ax.plot(x[i] + 0.02, y[i] + 0.001, color = "black", marker=  "$" + marker + "$", markersize = 45)

# ax.set_yscale('log')
# ax.set_xscale('log')
ax.set_xlim([0.001, 0.4])
ax.set_ylim([0.001, 0.4])
# ax.set_xticks([0.01, 0.1])
# ax.set_yticks([0.01, 0.1])
ax.plot([0, 0.4], [0, 0.4], "r:")
ax.set_xlabel("Cicero")
ax.set_ylabel("scDART")

fig.savefig("results_snare/de_snare/Pearson_cicero.png", bbox_inches = "tight")

# # average barplot
# spearman_scdart = np.mean(score_gact.loc[score_gact["Method"] == "scDART", "Spearman"].values)
# pearson_scdart = np.mean(score_gact.loc[score_gact["Method"] == "scDART", "Pearson"].values)
# spearman_linear = np.mean(score_gact.loc[score_gact["Method"] == "Linear", "Spearman"].values)
# pearson_linear = np.mean(score_gact.loc[score_gact["Method"] == "Linear", "Pearson"].values)
# spearman_signac = np.mean(score_gact.loc[score_gact["Method"] == "Signac", "Spearman"].values)
# pearson_signac = np.mean(score_gact.loc[score_gact["Method"] == "Signac", "Pearson"].values)
# spearman_cicero = np.mean(score_gact.loc[score_gact["Method"] == "Cicero", "Spearman"].values)
# pearson_cicero = np.mean(score_gact.loc[score_gact["Method"] == "Cicero", "Pearson"].values)

fig = plt.figure(figsize = (7,5))
ax = fig.subplots(nrows = 1, ncols = 1)
# ax = sns.barplot(data = score_gact, x = "Method", y = "Pearson", ax = ax, color = "blue", alpha = 0.7, estimator=np.median, ci='sd', capsize=.2)
ax = sns.boxplot(data = score_gact, x = "Method", y = "Pearson", ax = ax)
plt.tight_layout()
ax.set_xticklabels(labels = ["scDART", "Linear", "Signac", "Cicero"], rotation = 45)
ax.set_ylabel("Pearson")
newwidth = 0.5
for bar1 in ax.patches:
    x = bar1.get_x()
    width = bar1.get_width()
    centre = x+width/2.

    bar1.set_x(centre-newwidth/2.)
    bar1.set_width(newwidth)

show_values_on_bars(ax)
fig.savefig("results_snare/de_snare/Pearson.png", bbox_inches = "tight")

# # In[]
# pseudo_rna = model_dict["gene_act"](atac_dataset.counts.to(device)).detach().cpu().numpy()
# pseudo_rna = zscore(pseudo_rna, axis = 0)
# genes = ["Ptprd", "Mki67", "Fabp7", "Top2a", "Mef2c", "Macrod2", "Tenm2", "Dab1", "Tnc", "Frmd4a", "Celf2"]
# # genes = pd.read_csv("results_snare/de_snare/1000/de_gene_traj_0.csv", index_col = 0)
# # genes = genes.iloc[:50,0].values.squeeze()
# pseudo_rna = pd.DataFrame(data = pseudo_rna, index = counts_rna.index.values, columns = counts_rna.columns.values)
# pseudo_rna_sorted = pseudo_rna.iloc[np.argsort(pt_infer_rna), :]
# rna_sorted = counts_rna.iloc[np.argsort(pt_infer_rna), :]
# pseudo_rna_sorted = pseudo_rna_sorted.loc[:, genes]
# rna_sorted = rna_sorted.loc[:, genes]
# # rna_sorted = zscore(rna_sorted, axis = 0)


# score_gact = pd.DataFrame(columns = ["Method", "Gene", "Spearman", "Pearson"])
# # loop through cells
# for i, barcode in enumerate(pseudo_rna_sorted.index.values.squeeze()):
#     spearman,_ = spearmanr(pseudo_rna_sorted.loc[barcode, :], rna_sorted.loc[barcode, :])
#     if np.isnan(spearman):
#         spearman = 0
#     pearson,_ = pearsonr(pseudo_rna_sorted.loc[barcode, :], rna_sorted.loc[barcode, :])
#     if np.isnan(pearson):
#         pearson = 0
#     # print("gene: {:s}, spearman: {:.4f}, pearson: {:.4f}".format(gene, spearman, pearson))
#     score_gact = score_gact.append({"Method": "scDART", "Cell": barcode, "Spearman": spearman, "Pearson": pearson}, ignore_index = True)

# # linear method
# pseudo_rna2 = (atac_dataset.counts.to(device) @ coarse_reg).detach().cpu().numpy()
# pseudo_rna2 = zscore(pseudo_rna2, axis = 0)
# pseudo_rna2 = pd.DataFrame(data = pseudo_rna2, index = counts_rna.index.values, columns = counts_rna.columns.values)
# pseudo_rna2_sorted = pseudo_rna2.iloc[np.argsort(pt_infer_rna), :]
# pseudo_rna2_sorted = pseudo_rna2_sorted.loc[:, genes]

# for i, barcode in enumerate(pseudo_rna_sorted.index.values.squeeze()):
#     spearman,_ = spearmanr(pseudo_rna2_sorted.loc[barcode,:], rna_sorted.loc[barcode,:])
#     if np.isnan(spearman):
#         spearman = 0
#     pearson,_ = pearsonr(pseudo_rna2_sorted.loc[barcode,:], rna_sorted.loc[barcode,:])
#     if np.isnan(pearson):
#         pearson = 0
#     # print("gene: {:s}, spearman: {:.4f}, pearson: {:.4f}".format(gene, spearman, pearson))
#     score_gact = score_gact.append({"Method": "Linear", "Cell": barcode, "Spearman": spearman, "Pearson": pearson}, ignore_index = True)

# # Signac method
# pseudo_rna_signac = pd.read_csv("results_snare/pseudoRNA/counts_rna_signac.csv", index_col = 0)
# pseudo_rna_signac_sorted = pseudo_rna_signac.loc[rna_sorted.index.values,:]
# pseudo_rna_signac_sorted = zscore(pseudo_rna_signac_sorted, axis = 0)
# pseudo_rna_signac_sorted = pseudo_rna_signac_sorted.loc[:, genes]
# pseudo_rna_signac_sorted = pseudo_rna_signac_sorted.fillna(0)

# for i, barcode in enumerate(pseudo_rna_sorted.index.values.squeeze()):
#     spearman,_ = spearmanr(pseudo_rna_signac_sorted.loc[barcode,:], rna_sorted.loc[barcode,:])
#     if np.isnan(spearman):
#         spearman = 0
#     pearson,_ = pearsonr(pseudo_rna_signac_sorted.loc[barcode,:], rna_sorted.loc[barcode,:])
#     if np.isnan(pearson):
#         pearson = 0
#     # print("gene: {:s}, spearman: {:.4f}, pearson: {:.4f}".format(gene, spearman, pearson))
#     score_gact = score_gact.append({"Method": "Signac", "Cell": barcode, "Spearman": spearman, "Pearson": pearson}, ignore_index = True)

# # Cicero method
# pseudo_rna_cicero = pd.read_csv("results_snare/pseudoRNA/counts_rna_cicero.csv", index_col = 0)
# pseudo_rna_cicero = zscore(pseudo_rna_cicero, axis = 0)
# pseudo_rna_signac = pd.DataFrame(data = pseudo_rna_cicero, index = counts_rna.index.values, columns = counts_rna.columns.values)
# pseudo_rna_cicero_sorted = pseudo_rna_cicero.iloc[np.argsort(pt_infer_rna), :]
# pseudo_rna_cicero_sorted = pseudo_rna_cicero_sorted.loc[:, genes]
# pseudo_rna_cicero_sorted = pseudo_rna_cicero_sorted.fillna(0)

# for i, barcode in enumerate(pseudo_rna_sorted.index.values.squeeze()):
#     spearman,_ = spearmanr(pseudo_rna_cicero_sorted.loc[barcode,:], rna_sorted.loc[barcode,:])
#     if np.isnan(spearman):
#         spearman = 0
#     pearson,_ = pearsonr(pseudo_rna_cicero_sorted.loc[barcode,:], rna_sorted.loc[barcode,:])
#     if np.isnan(pearson):
#         pearson = 0
#     # print("gene: {:s}, spearman: {:.4f}, pearson: {:.4f}".format(gene, spearman, pearson))
#     score_gact = score_gact.append({"Method": "Cicero", "Cell": barcode, "Spearman": spearman, "Pearson": pearson}, ignore_index = True)


# fig = plt.figure(figsize = (7,5))
# ax = fig.subplots(nrows = 1, ncols = 1)
# # ax = sns.barplot(data = score_gact, x = "Method", y = "Pearson", ax = ax, color = "blue", alpha = 0.7, estimator=np.median, ci='sd', capsize=.2)
# ax = sns.boxplot(data = score_gact, x = "Method", y = "Pearson", ax = ax)
# plt.tight_layout()
# ax.set_xticklabels(labels = ["scDART", "Linear", "Signac", "Cicero"], rotation = 45)
# ax.set_ylabel("Pearson")
# newwidth = 0.5
# for bar1 in ax.patches:
#     x = bar1.get_x()
#     width = bar1.get_width()
#     centre = x+width/2.

#     bar1.set_x(centre-newwidth/2.)
#     bar1.set_width(newwidth)

# show_values_on_bars(ax)
# fig.savefig("results_snare/de_snare/Pearson.png", bbox_inches = "tight")


# In[] Other baseline methods
# 1. Liger
path = "results_snare/liger/"
z_rna_liger = pd.read_csv(path + "H1_full.csv", index_col = 0)
z_atac_liger = pd.read_csv(path + "H2_full.csv", index_col = 0)
integrated_data = (z_rna_liger.values, z_atac_liger.values)

pca_op = PCA(n_components = 2)
umap_op = UMAP(n_components = 2)

pca_latent = pca_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))
umap_latent = umap_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))

fig, axs = utils.plot_latent(umap_latent[:z_rna.shape[0],:], umap_latent[z_rna.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "joint", save = path + "liger_umap.png", figsize = (15,7), axis_label = "UMAP")
fig, axs = utils.plot_latent(umap_latent[:z_rna.shape[0],:], umap_latent[z_rna.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "modality", save = path + "liger_batches_umap.png", figsize = (15,7), axis_label = "UMAP")
fig, axs = utils.plot_latent(pca_latent[:z_rna.shape[0],:], pca_latent[z_rna.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "joint", save = None, figsize = (15,7), axis_label = "PCA")
axs.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig.savefig(path + "liger_pca.png", bbox_inches = "tight")
fig, axs = utils.plot_latent(pca_latent[:z_rna.shape[0],:], pca_latent[z_rna.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "modality", save = path + "liger_batches_pca.png", figsize = (15,7), axis_label = "PCA")
axs.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig.savefig(path + "liger_batches_pca.png", bbox_inches = "tight")

z_destiny = np.load(path + "z_diffmap.npy")
utils.plot_latent(z_destiny[:z_rna.shape[0],:], z_destiny[z_rna.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "joint", save = path + "liger_destiny.png", figsize = (15,7), axis_label = "Diffmap")
utils.plot_latent(z_destiny[:z_rna.shape[0],:], z_destiny[z_rna.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "modality", save = path + "liger_batches_destiny.png", figsize = (15,7), axis_label = "Diffmap")

# Infer backbone
root_cell = 450
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

cell_labels = np.concatenate((label_rna, label_atac), axis = 0).squeeze()
groups, mean_cluster, T = backbone_inf(z_rna_liger, z_atac_liger, cell_labels)
mean_cluster = pca_op.transform(np.array(mean_cluster))

plot_backbone(z_rna_pca, z_atac_pca, mode = "joint", mean_cluster = mean_cluster, groups = groups, T = T, figsize=(15,7), save = path + "backbone.png", anno = cell_labels, axis_label = "PCA")
utils.plot_latent_pt(z1 = z_rna_pca, z2 = z_atac_pca, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = path + "z_pt.png", figsize = (15,7), axis_label = "PCA")

# In[]
# 2. Seurat
path = "results_snare/seurat/"

coembed = pd.read_csv(path + "umap_embedding.txt", sep = "\t").values
z_rna_seurat = coembed[:label_rna.values.shape[0],:]
z_atac_seurat = coembed[label_rna.values.shape[0]:,:]
utils.plot_latent(z_rna_seurat, z_atac_seurat, label_rna.values, label_atac.values, mode = "modality", figsize = (15,7), axis_label = "PCA", save = path + "seurat_batches_umap.png")
utils.plot_latent(z_rna_seurat, z_atac_seurat, label_rna.values, label_atac.values, mode = "joint", figsize = (15,7), axis_label = "PCA", save = path + "seurat_umap.png")

coembed = pd.read_csv(path + "pca_embedding.txt", sep = "\t").values
z_rna_seurat = coembed[:label_rna.values.shape[0],:]
z_atac_seurat = coembed[label_rna.values.shape[0]:,:]
utils.plot_latent(z_rna_seurat, z_atac_seurat, label_rna.values, label_atac.values, mode = "modality", figsize = (15,7), axis_label = "PCA", save = path + "seurat_batches_pca.png")
utils.plot_latent(z_rna_seurat, z_atac_seurat, label_rna.values, label_atac.values, mode = "joint", figsize = (15,7), axis_label = "PCA", save = path + "seurat_pca.png")

z_destiny = np.load(path + "z_diffmap.npy")
utils.plot_latent(z_destiny[:z_rna.shape[0],:], z_destiny[z_rna.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "joint", save = path + "destiny_joint.png", figsize = (15,7), axis_label = "Diffmap")
utils.plot_latent(z_destiny[:z_rna.shape[0],:], z_destiny[z_rna.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "modality", save = path + "destiny.png", figsize = (15,7), axis_label = "Diffmap")

# Infer backbone
root_cell = 450
dpt_mtx = ti.dpt(np.concatenate((z_rna_seurat, z_atac_seurat), axis = 0), n_neigh = 10)
pt_infer = dpt_mtx[root_cell, :]
pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
pt_infer = pt_infer/np.max(pt_infer)
# for scRNA-Seq batch
pt_infer_rna = pt_infer[:z_rna_seurat.shape[0]]
# for scATAC-Seq batch
pt_infer_atac = pt_infer[z_rna_seurat.shape[0]:]


cell_labels = np.concatenate((label_rna, label_atac), axis = 0).squeeze()
groups, mean_cluster, T = backbone_inf(z_rna_seurat, z_atac_seurat, cell_labels)

plot_backbone(z_rna_seurat, z_atac_seurat, mode = "joint", mean_cluster = np.array(mean_cluster), groups = groups, T = T, figsize=(15,7), save = path + "backbone.png", anno = cell_labels, axis_label = "PCA")
utils.plot_latent_pt(z1 = z_rna_seurat, z2 = z_atac_seurat, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = path + "z_pt.png", figsize = (15,7), axis_label = "PCA")


path = "results_snare/seurat/"

coembed = pd.read_csv(path + "umap_embedding_full.txt", sep = "\t").values
z_rna_seurat = coembed[:label_rna.values.shape[0],:]
z_atac_seurat = coembed[label_rna.values.shape[0]:,:]
utils.plot_latent(z_rna_seurat, z_atac_seurat, label_rna.values, label_atac.values, mode = "modality", figsize = (15,7), axis_label = "PCA", save = path + "seurat_batches_umap_full.png")
utils.plot_latent(z_rna_seurat, z_atac_seurat, label_rna.values, label_atac.values, mode = "joint", figsize = (15,7), axis_label = "PCA", save = path + "seurat_umap_full.png")

coembed = pd.read_csv(path + "pca_embedding_full.txt", sep = "\t").values
z_rna_seurat = coembed[:label_rna.values.shape[0],:]
z_atac_seurat = coembed[label_rna.values.shape[0]:,:]

utils.plot_latent(z_rna_seurat, z_atac_seurat, label_rna.values, label_atac.values, mode = "modality", figsize = (15,7), axis_label = "PCA", save = path + "seurat_batches_pca_full.png")
utils.plot_latent(z_rna_seurat, z_atac_seurat, label_rna.values, label_atac.values, mode = "joint", figsize = (15,7), axis_label = "PCA", save = path + "seurat_pca_full.png")

# Infer backbone
root_cell = 450
dpt_mtx = ti.dpt(np.concatenate((z_rna_seurat, z_atac_seurat), axis = 0), n_neigh = 10)
pt_infer = dpt_mtx[root_cell, :]
pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
pt_infer = pt_infer/np.max(pt_infer)
# for scRNA-Seq batch
pt_infer_rna = pt_infer[:z_rna_seurat.shape[0]]
# for scATAC-Seq batch
pt_infer_atac = pt_infer[z_rna_seurat.shape[0]:]


cell_labels = np.concatenate((label_rna, label_atac), axis = 0).squeeze()
groups, mean_cluster, T = backbone_inf(z_rna_seurat, z_atac_seurat, cell_labels)

plot_backbone(z_rna_seurat, z_atac_seurat, mode = "joint", mean_cluster = np.array(mean_cluster), groups = groups, T = T, figsize=(15,7), save = path + "backbone_full.png", anno = cell_labels, axis_label = "PCA")
utils.plot_latent_pt(z1 = z_rna_seurat, z2 = z_atac_seurat, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = path + "z_pt_full.png", figsize = (15,7), axis_label = "PCA")


# Include post-processing
z_rna_seurat_post = torch.FloatTensor(z_rna_seurat)
z_atac_seurat_post = torch.FloatTensor(z_atac_seurat)
z_rna_seurat_post, z_atac_seurat_post = palign.match_alignment(z_rna = z_rna_seurat_post, z_atac = z_atac_seurat_post, k = 10)
z_atac_seurat_post, z_rna_seurat_post = palign.match_alignment(z_rna = z_atac_seurat_post, z_atac = z_rna_seurat_post, k = 10)
z_rna_seurat_post = z_rna_seurat_post.numpy()
z_atac_seurat_post = z_atac_seurat_post.numpy()
utils.plot_latent(z_rna_seurat_post, z_atac_seurat_post, label_rna.values, label_atac.values, mode = "modality", figsize = (15,7), axis_label = "PCA", save = path + "pca_post.png")
utils.plot_latent(z_rna_seurat_post, z_atac_seurat_post, label_rna.values, label_atac.values, mode = "joint", figsize = (15,7), axis_label = "PCA", save = path + "pca_joint_post.png")

# Infer backbone
root_cell = 450
dpt_mtx = ti.dpt(np.concatenate((z_rna_seurat_post, z_atac_seurat_post), axis = 0), n_neigh = 10)
pt_infer = dpt_mtx[root_cell, :]
pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
pt_infer = pt_infer/np.max(pt_infer)
# for scRNA-Seq batch
pt_infer_rna = pt_infer[:z_rna_seurat_post.shape[0]]
# for scATAC-Seq batch
pt_infer_atac = pt_infer[z_rna_seurat_post.shape[0]:]

cell_labels = np.concatenate((label_rna, label_atac), axis = 0).squeeze()
groups, mean_cluster, T = backbone_inf(z_rna_seurat, z_atac_seurat, cell_labels)

plot_backbone(z_rna_seurat_post, z_atac_seurat_post, mode = "joint", mean_cluster = np.array(mean_cluster), groups = groups, T = T, figsize=(15,7), save = path + "backbone_post.png", anno = cell_labels, axis_label = "PCA")
utils.plot_latent_pt(z1 = z_rna_seurat_post, z2 = z_atac_seurat_post, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = path + "z_pt_post.png", figsize = (15,7), axis_label = "PCA")


# In[]
# 3. unioncom
path = "results_snare/unioncom/"
z_rna_unioncom = np.load(path + "unioncom_rna_32.npy")
z_atac_unioncom = np.load(path + "unioncom_atac_32.npy")
integrated_data = (z_rna_unioncom, z_atac_unioncom)

pca_op = PCA(n_components = 2)
umap_op = UMAP(n_components = 2)

pca_latent = pca_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))
umap_latent = umap_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))

utils.plot_latent(umap_latent[:z_rna.shape[0],:], umap_latent[z_rna.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "joint", save = path + "unioncom_umap.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(umap_latent[:z_rna.shape[0],:], umap_latent[z_rna.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "modality", save = path + "unioncom_batches_umap.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(pca_latent[:z_rna.shape[0],:], pca_latent[z_rna.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "joint", save = path + "unioncom_pca.png", figsize = (15,7), axis_label = "PCA")
utils.plot_latent(pca_latent[:z_rna.shape[0],:], pca_latent[z_rna.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "modality", save = path + "unioncom_batches_pca.png", figsize = (15,7), axis_label = "PCA")

z_destiny = np.load(path + "z_diffmap.npy")
utils.plot_latent(z_destiny[:z_rna_unioncom.shape[0],:], z_destiny[z_rna_unioncom.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "joint", save = path + "unioncom_destiny.png", figsize = (15,7), axis_label = "Diffmap")
utils.plot_latent(z_destiny[:z_rna_unioncom.shape[0],:], z_destiny[z_rna_unioncom.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "modality", save = path + "unioncom_batches_destiny.png", figsize = (15,7), axis_label = "Diffmap")

# Infer backbone
root_cell = 450
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

cell_labels = np.concatenate((label_rna, label_atac), axis = 0).squeeze()
groups, mean_cluster, T = backbone_inf(z_rna_unioncom, z_atac_unioncom, cell_labels)
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


utils.plot_latent(umap_latent[:z_rna_unioncom.shape[0],:], umap_latent[z_rna_unioncom.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values,
mode = "joint", save = path + "unioncom_umap_post.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(umap_latent[:z_rna_unioncom.shape[0],:], umap_latent[z_rna_unioncom.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "modality", save = path + "unioncom_batches_umap_post.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(pca_latent[:z_rna_unioncom.shape[0],:], pca_latent[z_rna_unioncom.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values,
mode = "joint", save = path + "unioncom_pca_post.png", figsize = (15,7), axis_label = "PCA")
utils.plot_latent(pca_latent[:z_rna_unioncom.shape[0],:], pca_latent[z_rna_unioncom.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values,
mode = "modality", save = path + "unioncom_batches_pca_post.png", figsize = (15,7), axis_label = "PCA")

# Infer backbone
root_cell = 450
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
cell_labels = np.concatenate((label_rna, label_atac), axis = 0).squeeze()
groups, mean_cluster, T = backbone_inf(z_rna_unioncom_post, z_atac_unioncom_post, cell_labels)
mean_cluster = pca_op.transform(np.array(mean_cluster))

plot_backbone(z_rna_pca, z_atac_pca, mode = "joint", mean_cluster = mean_cluster, groups = groups, T = T, figsize=(15,7), save = path + "backbone_post.png", anno = cell_labels, axis_label = "PCA")
utils.plot_latent_pt(z1 = z_rna_pca, z2 = z_atac_pca, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = path + "z_pt_post.png", figsize = (15,7), axis_label = "PCA")

# In[]
# 4. scJoint
path = "results_snare/scJoint_snare_traj/"
z_atac_scJoint = pd.read_csv(path + "counts_atac_embeddings.txt", sep = " ", header = None).values
z_rna_scJoint = pd.read_csv(path + "counts_rna_embeddings.txt", sep = " ", header = None).values

integrated_data = [z_rna_scJoint, z_atac_scJoint]
pca_op = PCA(n_components = 2)
umap_op = UMAP(n_components = 2)

pca_latent = pca_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))
umap_latent = umap_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))


utils.plot_latent(umap_latent[:z_rna_scJoint.shape[0],:], umap_latent[z_rna_scJoint.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "joint", save = path + "scjoint_umap.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(umap_latent[:z_rna_scJoint.shape[0],:], umap_latent[z_rna_scJoint.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "modality", save = path + "scjoint_batches_umap.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(pca_latent[:z_rna_scJoint.shape[0],:], pca_latent[z_rna_scJoint.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "joint", save = path + "scjoint_pca.png", figsize = (15,7), axis_label = "PCA")
utils.plot_latent(pca_latent[:z_rna_scJoint.shape[0],:], pca_latent[z_rna_scJoint.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "modality", save = path + "scjoint_batches_pca.png", figsize = (15,7), axis_label = "PCA")

z_destiny = np.load(path + "z_diffmap.npy")
utils.plot_latent(z_destiny[:z_rna_unioncom.shape[0],:], z_destiny[z_rna_unioncom.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "joint", save = path + "scjoint_destiny.png", figsize = (15,7), axis_label = "Diffmap")
utils.plot_latent(z_destiny[:z_rna_unioncom.shape[0],:], z_destiny[z_rna_unioncom.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "modality", save = path + "scjoint_batches_destiny.png", figsize = (15,7), axis_label = "Diffmap")

# Infer backbone
root_cell = 450
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

cell_labels = np.concatenate((label_rna, label_atac), axis = 0).squeeze()
groups, mean_cluster, T = backbone_inf(z_rna_scJoint, z_atac_scJoint, cell_labels)
mean_cluster = pca_op.transform(np.array(mean_cluster))

plot_backbone(z_rna_pca, z_atac_pca, mode = "joint", mean_cluster = mean_cluster, groups = groups, T = T, figsize=(15,7), save = path + "backbone_full.png", anno = cell_labels, axis_label = "PCA")
utils.plot_latent_pt(z1 = z_rna_pca, z2 = z_atac_pca, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = path + "z_pt_full.png", figsize = (15,7), axis_label = "PCA")

# In[]
path = "results_snare/scJoint_snare_raw_traj/"
z_atac_scJoint = pd.read_csv(path + "counts_atac_embeddings.txt", sep = " ", header = None).values
z_rna_scJoint = pd.read_csv(path + "counts_rna_embeddings.txt", sep = " ", header = None).values

integrated_data = [z_rna_scJoint, z_atac_scJoint]
pca_op = PCA(n_components = 2)
umap_op = UMAP(n_components = 2)

pca_latent = pca_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))
umap_latent = umap_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))
print(pca_op.explained_variance_ratio_)

utils.plot_latent(umap_latent[:z_rna_scJoint.shape[0],:], umap_latent[z_rna_scJoint.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "joint", save = path + "scjoint_umap.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(umap_latent[:z_rna_scJoint.shape[0],:], umap_latent[z_rna_scJoint.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "modality", save = path + "scjoint_batches_umap.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(pca_latent[:z_rna_scJoint.shape[0],:], pca_latent[z_rna_scJoint.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "joint", save = path + "scjoint_pca.png", figsize = (15,7), axis_label = "PCA")
utils.plot_latent(pca_latent[:z_rna_scJoint.shape[0],:], pca_latent[z_rna_scJoint.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "modality", save = path + "scjoint_batches_pca.png", figsize = (15,7), axis_label = "PCA")

# Infer backbone
root_cell = 450
dpt_mtx = ti.dpt(np.concatenate((z_rna_scJoint, z_atac_scJoint), axis = 0), n_neigh = 10)
pt_infer = dpt_mtx[root_cell, :]
pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
pt_infer = pt_infer/np.max(pt_infer)
# for scRNA-Seq batch
pt_infer_rna = pt_infer[:z_rna_scJoint.shape[0]]
# for scATAC-Seq batch
pt_infer_atac = pt_infer[z_rna_scJoint.shape[0]:]

z = pca_op.fit_transform(np.concatenate((z_rna_scJoint, z_atac_scJoint), axis = 0))
z_rna_pca = z[:z_rna_scJoint.shape[0],:]
z_atac_pca = z[z_rna_scJoint.shape[0]:,:]

cell_labels = np.concatenate((label_rna, label_atac), axis = 0).squeeze()
groups, mean_cluster, T = backbone_inf(z_rna_scJoint, z_atac_scJoint, cell_labels)
mean_cluster_pca = pca_op.transform(np.array(mean_cluster))

plot_backbone(z_rna_pca, z_atac_pca, mode = "joint", mean_cluster = mean_cluster_pca, groups = groups, T = T, figsize=(15,7), save = path + "backbone_full.png", anno = cell_labels, axis_label = "PCA")
utils.plot_latent_pt(z1 = z_rna_pca, z2 = z_atac_pca, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = path + "z_pt_full.png", figsize = (15,7), axis_label = "PCA")

z = umap_op.fit_transform(np.concatenate((z_rna_scJoint, z_atac_scJoint), axis = 0))
z_rna_umap = z[:z_rna_scJoint.shape[0],:]
z_atac_umap = z[z_rna_scJoint.shape[0]:,:]

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


utils.plot_latent(umap_latent[:z_rna_scJoint_post.shape[0],:], umap_latent[z_rna_scJoint_post.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values,
mode = "joint", save = path + "scjoint_umap_post.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(umap_latent[:z_rna_scJoint_post.shape[0],:], umap_latent[z_rna_scJoint_post.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values,
mode = "modality", save = path + "scjoint_batches_umap_post.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(pca_latent[:z_rna_scJoint_post.shape[0],:], pca_latent[z_rna_scJoint_post.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values,
mode = "joint", save = path + "scjoint_pca_post.png", figsize = (15,7), axis_label = "PCA")
utils.plot_latent(pca_latent[:z_rna_scJoint_post.shape[0],:], pca_latent[z_rna_scJoint_post.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values,
mode = "modality", save = path + "scjoint_batches_pca_post.png", figsize = (15,7), axis_label = "PCA")

# Infer backbone
root_cell = 450
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
cell_labels = np.concatenate((label_rna, label_atac), axis = 0).squeeze()
groups, mean_cluster, T = backbone_inf(z_rna_scJoint_post, z_rna_scJoint_post, cell_labels)
mean_cluster = pca_op.transform(np.array(mean_cluster))

plot_backbone(z_rna_pca, z_atac_pca, mode = "joint", mean_cluster = mean_cluster, groups = groups, T = T, figsize=(15,7), save = path + "backbone_full_post.png", anno = cell_labels, axis_label = "PCA")
utils.plot_latent_pt(z1 = z_rna_pca, z2 = z_atac_pca, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = path + "z_pt_full_post.png", figsize = (15,7), axis_label = "PCA")


# In[] MMD-MA
path = "results_snare/mmd_ma/"
z_rna_mmdma = np.load(path + "mmd_ma_rna.npy")
z_atac_mmdma = np.load(path + "mmd_ma_atac.npy")

integrated_data = [z_rna_mmdma, z_atac_mmdma]
pca_op = PCA(n_components = 2)
umap_op = UMAP(n_components = 2)

pca_latent = pca_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))
umap_latent = umap_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))
print(pca_op.explained_variance_ratio_)

utils.plot_latent(umap_latent[:z_rna_mmdma.shape[0],:], umap_latent[z_rna_mmdma.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "joint", save = path + "mmdma_umap.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(umap_latent[:z_rna_mmdma.shape[0],:], umap_latent[z_rna_mmdma.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "modality", save = path + "mmdma_batches_umap.png", figsize = (15,7), axis_label = "UMAP")
utils.plot_latent(pca_latent[:z_rna_mmdma.shape[0],:], pca_latent[z_rna_mmdma.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "joint", save = path + "mmdma_pca.png", figsize = (15,7), axis_label = "PCA")
utils.plot_latent(pca_latent[:z_rna_mmdma.shape[0],:], pca_latent[z_rna_mmdma.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "modality", save = path + "mmdma_batches_pca.png", figsize = (15,7), axis_label = "PCA")

# Infer backbone
root_cell = 450
dpt_mtx = ti.dpt(np.concatenate((z_rna_mmdma, z_atac_mmdma), axis = 0), n_neigh = 10)
pt_infer = dpt_mtx[root_cell, :]
pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
pt_infer = pt_infer/np.max(pt_infer)
# for scRNA-Seq batch
pt_infer_rna = pt_infer[:z_rna_mmdma.shape[0]]
# for scATAC-Seq batch
pt_infer_atac = pt_infer[z_rna_mmdma.shape[0]:]

pca_op = PCA(n_components = 2)
z = pca_op.fit_transform(np.concatenate((z_rna_mmdma, z_atac_mmdma), axis = 0))
z_rna_pca = z[:z_rna_mmdma.shape[0],:]
z_atac_pca = z[z_rna_mmdma.shape[0]:,:]

cell_labels = np.concatenate((label_rna, label_atac), axis = 0).squeeze()
groups, mean_cluster, T = backbone_inf(z_rna_mmdma, z_atac_mmdma, cell_labels)
mean_cluster = pca_op.transform(np.array(mean_cluster))

plot_backbone(z_rna_pca, z_atac_pca, mode = "joint", mean_cluster = mean_cluster, groups = groups, T = T, figsize=(15,7), save = path + "backbone_full.png", anno = cell_labels, axis_label = "PCA")
utils.plot_latent_pt(z1 = z_rna_pca, z2 = z_atac_pca, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = path + "z_pt_full.png", figsize = (15,7), axis_label = "PCA")


# In[] Neighborhood overlap score, use also the unionCom, LIGER and Seurat
# Both the neighborhood overlap and pseudotime alignment are higher for scDART when the number of genes increase
path = "results_snare/liger/"
z_rna_liger = pd.read_csv(path + "H1_full.csv", index_col = 0).values
z_atac_liger = pd.read_csv(path + "H2_full.csv", index_col = 0).values

path = "results_snare/seurat/"
coembed = pd.read_csv(path + "pca_embedding.txt", sep = "\t").values
z_rna_seurat = coembed[:label_rna.values.shape[0],:]
z_atac_seurat = coembed[label_rna.values.shape[0]:,:]

path = "results_snare/unioncom/"
z_rna_unioncom = np.load(path + "unioncom_rna_32.npy")
z_atac_unioncom = np.load(path + "unioncom_atac_32.npy")

path = "results_snare/mmd_ma/"
z_rna_mmdma = np.load(path + "mmd_ma_rna.npy")
z_atac_mmdma = np.load(path + "mmd_ma_atac.npy")

path = "results_snare/scJoint_snare_raw_traj/"
z_rna_scJoint = np.loadtxt(path + "counts_rna_embeddings.txt")
z_atac_scJoint = np.loadtxt(path + "counts_atac_embeddings.txt")

path = "results_snare/models_1000/"
z_rna_scdart = np.load(file = path + "z_rna_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + ".npy")
z_atac_scdart = np.load(file = path + "z_atac_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + ".npy")
z_rna_scdart = torch.FloatTensor(z_rna_scdart)
z_atac_scdart = torch.FloatTensor(z_atac_scdart)
z_rna_scdart, z_atac_scdart = palign.match_alignment(z_rna = z_rna_scdart, z_atac = z_atac_scdart, k = 10)
z_atac_scdart, z_rna_scdart = palign.match_alignment(z_rna = z_atac_scdart, z_atac = z_rna_scdart, k = 10)
z_rna_scdart = z_rna_scdart.numpy()
z_atac_scdart = z_atac_scdart.numpy()

score_liger = []
score_seurat = []
score_unioncom = []
score_scdart = []
score_mmdma = []
score_scJoint = []

for k in range(10, 1000, 10):
    score_liger.append(bmk.neigh_overlap(z_rna_liger, z_atac_liger, k = k))
    score_unioncom.append(bmk.neigh_overlap(z_rna_unioncom, z_atac_unioncom, k = k))
    score_seurat.append(bmk.neigh_overlap(z_rna_seurat, z_atac_seurat, k = k))
    score_scdart.append(bmk.neigh_overlap(z_rna_scdart, z_atac_scdart, k = k))
    score_mmdma.append(bmk.neigh_overlap(z_rna_mmdma, z_atac_mmdma, k = k))
    score_scJoint.append(bmk.neigh_overlap(z_rna_scJoint, z_atac_scJoint, k = k))

score_liger = np.array(score_liger)
score_seurat = np.array(score_seurat)
score_unioncom = np.array(score_unioncom)
score_scdart = np.array(score_scdart)
score_mmdma = np.array(score_mmdma)
score_scJoint = np.array(score_scJoint)

fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot()
ax.plot(np.arange(10, 1000, 10), score_liger, label = "LIGER")
ax.plot(np.arange(10, 1000, 10), score_unioncom, label = "UnionCom")
ax.plot(np.arange(10, 1000, 10), score_seurat, label = "Seurat")
ax.plot(np.arange(10, 1000, 10), score_scdart, label = "scDART")
ax.plot(np.arange(10, 1000, 10), score_mmdma, label = "MMD-MA")
ax.plot(np.arange(10, 1000, 10), score_scJoint, label = "scJoint")
ax.legend()
ax.set_xlabel("Neighborhood size")
ax.set_ylabel("Neighborhood overlap")
ax.set_xticks([0, 200, 400, 600, 800, 1000])
fig.savefig("results_snare/neigh_ov.png", bbox_inches = "tight")

# In[] check pseudotime correlation

# root, manually found
root_cell = 450
# scdart
dpt_mtx = ti.dpt(np.concatenate((z_rna_scdart, z_atac_scdart), axis = 0), n_neigh = 10)
pt_infer_scdart = dpt_mtx[root_cell, :]
pt_infer_scdart[pt_infer_scdart.argsort()] = np.arange(len(pt_infer_scdart))
pt_infer_scdart = pt_infer_scdart/np.max(pt_infer_scdart)
spearman_scdart, _ = spearmanr(pt_infer_scdart[:z_rna_scdart.shape[0]], pt_infer_scdart[z_rna_scdart.shape[0]:])
pearson_scdart, _ = pearsonr(pt_infer_scdart[:z_rna_scdart.shape[0]], pt_infer_scdart[z_rna_scdart.shape[0]:])

# liger
dpt_mtx = ti.dpt(np.concatenate((z_rna_liger, z_atac_liger), axis = 0), n_neigh = 10)
pt_infer_liger = dpt_mtx[root_cell, :]
pt_infer_liger[pt_infer_liger.argsort()] = np.arange(len(pt_infer_liger))
pt_infer_liger = pt_infer_liger/np.max(pt_infer_liger)
spearman_liger, _ = spearmanr(pt_infer_liger[:z_rna_liger.shape[0]], pt_infer_liger[z_rna_liger.shape[0]:])
pearson_liger, _ = pearsonr(pt_infer_liger[:z_rna_liger.shape[0]], pt_infer_liger[z_rna_liger.shape[0]:])

# unioncom
dpt_mtx = ti.dpt(np.concatenate((z_rna_unioncom, z_atac_unioncom), axis = 0), n_neigh = 10)
pt_infer_unioncom = dpt_mtx[root_cell, :]
pt_infer_unioncom[pt_infer_unioncom.argsort()] = np.arange(len(pt_infer_unioncom))
pt_infer_unioncom = pt_infer_unioncom/np.max(pt_infer_unioncom)
spearman_unioncom, _ = spearmanr(pt_infer_unioncom[:z_rna_unioncom.shape[0]], pt_infer_unioncom[z_rna_unioncom.shape[0]:])
pearson_unioncom, _ = pearsonr(pt_infer_unioncom[:z_rna_unioncom.shape[0]], pt_infer_unioncom[z_rna_unioncom.shape[0]:])

# seurat
dpt_mtx = ti.dpt(np.concatenate((z_rna_seurat, z_atac_seurat), axis = 0), n_neigh = 10)
pt_infer_seurat = dpt_mtx[root_cell, :]
pt_infer_seurat[pt_infer_seurat.argsort()] = np.arange(len(pt_infer_seurat))
pt_infer_seurat = pt_infer_seurat/np.max(pt_infer_seurat)
spearman_seurat, _ = spearmanr(pt_infer_seurat[:z_rna_seurat.shape[0]], pt_infer_seurat[z_rna_seurat.shape[0]:])
pearson_seurat, _ = pearsonr(pt_infer_seurat[:z_rna_seurat.shape[0]], pt_infer_seurat[z_rna_seurat.shape[0]:])

# mmd-ma
dpt_mtx = ti.dpt(np.concatenate((z_rna_mmdma, z_atac_mmdma), axis = 0), n_neigh = 10)
pt_infer_mmdma = dpt_mtx[root_cell, :]
pt_infer_mmdma[pt_infer_mmdma.argsort()] = np.arange(len(pt_infer_mmdma))
pt_infer_mmdma = pt_infer_mmdma/np.max(pt_infer_mmdma)
spearman_mmdma, _ = spearmanr(pt_infer_mmdma[:z_rna_mmdma.shape[0]], pt_infer_mmdma[z_rna_mmdma.shape[0]:])
pearson_mmdma, _ = pearsonr(pt_infer_mmdma[:z_rna_mmdma.shape[0]], pt_infer_mmdma[z_rna_mmdma.shape[0]:])

# scJoint
dpt_mtx = ti.dpt(np.concatenate((z_rna_scJoint, z_atac_scJoint), axis = 0), n_neigh = 10)
pt_infer_scjoint = dpt_mtx[root_cell, :]
pt_infer_scjoint[pt_infer_scjoint.argsort()] = np.arange(len(pt_infer_scjoint))
pt_infer_scjoint = pt_infer_scjoint/np.max(pt_infer_scjoint)
spearman_scjoint, _ = spearmanr(pt_infer_scjoint[:z_rna_scJoint.shape[0]], pt_infer_scjoint[z_rna_scJoint.shape[0]:])
pearson_scjoint, _ = pearsonr(pt_infer_scjoint[:z_rna_scJoint.shape[0]], pt_infer_scjoint[z_rna_scJoint.shape[0]:])


# correlation smaller than 0.87, the one reported in the paper.
print("scDART: spearman: {:.4f}, pearson: {:.4f}".format(spearman_scdart, pearson_scdart))
print("LIGER: spearman: {:.4f}, pearson: {:.4f}".format(spearman_liger, pearson_liger))
print("Seurat: spearman: {:.4f}, pearson: {:.4f}".format(spearman_seurat, pearson_seurat))
print("UnionCom: spearman: {:.4f}, pearson: {:.4f}".format(spearman_unioncom, pearson_unioncom))
print("MMD-MA: spearman: {:.4f}, pearson: {:.4f}".format(spearman_mmdma, pearson_mmdma))
print("scJoint: spearman: {:.4f}, pearson: {:.4f}".format(spearman_scjoint, pearson_scjoint))

# plot barplot

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

scores = pd.DataFrame(columns = ["Method", "Spearman", "Pearson"])
scores["Method"] = np.array(["scDART", "LIGER", "Seurat", "UnionCom", "MMD-MA", "scJoint"])
scores["Spearman"] = np.array([spearman_scdart, spearman_liger, spearman_seurat, spearman_unioncom, spearman_mmdma, spearman_scjoint])
scores["Pearson"] = np.array([pearson_scdart, pearson_liger, pearson_seurat, pearson_unioncom, pearson_mmdma, pearson_scjoint])
import seaborn as sns
fig = plt.figure(figsize = (15,7))
ax = fig.subplots(nrows = 1, ncols = 2)
ax[0] = sns.barplot(data = scores, x = "Method", y = "Spearman", ax = ax[0], color = "blue", alpha = 0.7)
ax[1] = sns.barplot(data = scores, x = "Method", y = "Pearson", ax = ax[1], color = "blue", alpha = 0.7)
plt.tight_layout()
ax[0].set_xticklabels(labels = ["scDART", "LIGER", "Seurat", "UnionCom", "MMD-MA", "scJoint"], rotation = 45)
ax[1].set_xticklabels(labels = ["scDART", "LIGER", "Seurat", "UnionCom", "MMD-MA", "scJoint"], rotation = 45)
newwidth = 0.5
for bar1, bar2 in zip(ax[0].patches, ax[1].patches):
    x = bar1.get_x()
    width = bar1.get_width()
    centre = x+width/2.

    bar1.set_x(centre-newwidth/2.)
    bar1.set_width(newwidth)

    x = bar2.get_x()
    width = bar2.get_width()
    centre = x+width/2.

    bar2.set_x(centre-newwidth/2.)
    bar2.set_width(newwidth)

show_values_on_bars(ax)
fig.savefig("results_snare/correlation.png", bbox_inches = "tight")

# In[] Clustering label consistency
def clust_consist(z_rna, z_atac, nclust = 3):
    # leiden, cannot fix cluster numbers
    # conn, _ = ti.nearest_neighbor(z_rna, k = k)
    # groups_rna, _ = ti.leiden(conn, resolution = resolution)
    # conn, _ = ti.nearest_neighbor(z_atac, k = k)
    # groups_atac, _ = ti.leiden(conn, resolution = resolution)

    # k-means
    from sklearn.cluster import KMeans
    groups_rna = KMeans(n_clusters = nclust, random_state = 0).fit(z_rna).labels_
    groups_atac = KMeans(n_clusters = nclust, random_state = 0).fit(z_atac).labels_

    # TODO: measuring the alignment of clustering, including ARI, NMI, Silhouette Score
    ari_score = bmk.ari(group1 = groups_rna, group2 = groups_atac)
    nmi_score = bmk.nmi(group1 = groups_rna, group2 = groups_atac)
    # print("number of clusters in RNA: {:d}".format(np.max(groups_rna)))
    # print("number of clusters in ATAC: {:d}".format(np.max(groups_atac)))
    # print("ARI: {:.3f}, NMI: {:.3f}".format(ari_score, nmi_score))
    # Silhouette Score cannot be used for cluster label alignment
    return ari_score, nmi_score

k = 15
nclusts = [5]
print("method: scDART")
ari_scdarts = []
nmi_scdarts = []
for nclust in nclusts:
    ari_scdart, nmi_scdart = clust_consist(z_rna_scdart, z_atac_scdart, nclust = nclust)
    ari_scdarts.append(ari_scdart)
    nmi_scdarts.append(nmi_scdart)

print("method: LIGER")
ari_ligers = []
nmi_ligers = []
for nclust in nclusts:
    ari_liger, nmi_liger = clust_consist(z_rna_liger, z_atac_liger, nclust = nclust)
    ari_ligers.append(ari_liger)
    nmi_ligers.append(nmi_liger)

print("method: Seurat")
ari_seurats = []
nmi_seurats = []
for nclust in nclusts:
    ari_seurat, nmi_seurat = clust_consist(z_rna_seurat, z_atac_seurat, nclust = nclust)
    ari_seurats.append(ari_seurat)
    nmi_seurats.append(nmi_seurat)

print("method: UnionCom")
ari_unioncoms = []
nmi_unioncoms = []
for nclust in nclusts:
    ari_unioncom, nmi_unioncom = clust_consist(z_rna_unioncom, z_atac_unioncom, nclust = nclust)
    ari_unioncoms.append(ari_unioncom)
    nmi_unioncoms.append(nmi_unioncom)

print("method: MMD-MA")
ari_mmdmas = []
nmi_mmdmas = []
for nclust in nclusts:
    ari_mmdma, nmi_mmdma = clust_consist(z_rna_mmdma, z_atac_mmdma, nclust = nclust)
    ari_mmdmas.append(ari_mmdma)
    nmi_mmdmas.append(nmi_mmdma)

print("method: scJoint")
ari_scjoints = []
nmi_scjoints = []
for nclust in nclusts:
    ari_scjoint, nmi_scjoint = clust_consist(z_rna_scJoint, z_atac_scJoint, nclust = nclust)
    ari_scjoints.append(ari_scjoint)
    nmi_scjoints.append(nmi_scjoint)

ari_scdarts = np.array(ari_scdarts)
ari_ligers = np.array(ari_ligers)
ari_seurats = np.array(ari_seurats)
ari_unioncoms = np.array(ari_unioncoms)
ari_mmdmas = np.array(ari_mmdmas)
ari_scjoints = np.array(ari_scjoints)

nmi_scdarts = np.array(nmi_scdarts)
nmi_ligers = np.array(nmi_ligers)
nmi_seurats = np.array(nmi_seurats)
nmi_unioncoms = np.array(nmi_unioncoms)
nmi_mmdmas = np.array(nmi_mmdmas)
nmi_scjoints = np.array(nmi_scjoints)

ari_scdart = np.nanmax(ari_scdarts)
ari_liger = np.nanmax(ari_ligers)
ari_seurat = np.nanmax(ari_seurats)
ari_unioncom = np.nanmax(ari_unioncoms)
ari_mmdma = np.nanmax(ari_mmdmas)
ari_scjoint = np.nanmax(ari_scjoints)

nmi_scdarts = np.nanmax(nmi_scdarts)
nmi_ligers = np.nanmax(nmi_ligers)
nmi_seurats = np.nanmax(nmi_seurats)
nmi_unioncoms = np.nanmax(nmi_unioncoms)
nmi_mmdmas = np.nanmax(nmi_mmdmas)
nmi_scjoints = np.nanmax(nmi_scjoints)

scores = pd.DataFrame(columns = ["Method", "ARI", "NMI"])
scores["Method"] = np.array(["scDART", "LIGER", "Seurat", "UnionCom", "MMD-MA", "scJoint"])
scores["ARI"] = np.array([ari_scdart, ari_liger, ari_seurat, ari_unioncom, ari_mmdma, ari_scjoint])
scores["NMI"] = np.array([nmi_scdart, nmi_liger, nmi_seurat, nmi_unioncom, nmi_mmdma, nmi_scjoint])
import seaborn as sns
fig = plt.figure(figsize = (15,7))
ax = fig.subplots(nrows = 1, ncols = 2)
ax[0] = sns.barplot(data = scores, x = "Method", y = "ARI", ax = ax[0], color = "blue", alpha = 0.7)
ax[1] = sns.barplot(data = scores, x = "Method", y = "NMI", ax = ax[1], color = "blue", alpha = 0.7)
plt.tight_layout()
ax[0].set_xticklabels(labels = ["scDART", "LIGER", "Seurat", "UnionCom", "MMD-MA", "scJoint"], rotation = 45)
ax[1].set_xticklabels(labels = ["scDART", "LIGER", "Seurat", "UnionCom", "MMD-MA", "scJoint"], rotation = 45)
newwidth = 0.5
for bar1, bar2 in zip(ax[0].patches, ax[1].patches):
    x = bar1.get_x()
    width = bar1.get_width()
    centre = x+width/2.

    bar1.set_x(centre-newwidth/2.)
    bar1.set_width(newwidth)

    x = bar2.get_x()
    width = bar2.get_width()
    centre = x+width/2.

    bar2.set_x(centre-newwidth/2.)
    bar2.set_width(newwidth)

show_values_on_bars(ax)
fig.savefig("results_snare/cluster_consistency.png", bbox_inches = "tight")

# In[] Difference of distances
from scipy.spatial.distance import pdist, squareform, cosine
def dist_diff(z_rna, z_atac):
    mse = 1/z_rna.shape[0] * np.sum(np.sqrt(np.sum((z_rna - z_atac) ** 2, axis = 1)))
    cos = 0
    for i in range(z_rna.shape[0]):
        cos +=  1 - cosine(z_rna[i, :], z_atac[i, :])
    cos /= z_rna.shape[0]
    return mse, cos

mse_scdart, cos_scdart = dist_diff(z_rna = z_rna_scdart, z_atac = z_atac_scdart)
mse_seurat, cos_seurat = dist_diff(z_rna = z_rna_seurat, z_atac = z_atac_seurat)
mse_liger, cos_liger = dist_diff(z_rna = z_rna_liger, z_atac = z_atac_liger)
mse_mmdma, cos_mmdma = dist_diff(z_rna = z_rna_mmdma, z_atac = z_atac_mmdma)
mse_unioncom, cos_unioncom = dist_diff(z_rna = z_rna_unioncom, z_atac = z_atac_unioncom)
mse_scjoint, cos_scjoint = dist_diff(z_rna = z_rna_scJoint, z_atac = z_atac_scJoint)

scores = pd.DataFrame(columns = ["Method", "MSE", "cos_sim"])
scores["Method"] = np.array(["scDART", "LIGER", "Seurat", "UnionCom", "MMD-MA", "scJoint"])
scores["MSE"] = np.array([mse_scdart, mse_liger, mse_seurat, mse_unioncom, mse_mmdma, mse_scjoint])
scores["cos_sim"] = np.array([cos_scdart, cos_liger, cos_seurat, cos_unioncom, cos_mmdma, cos_scjoint])
import seaborn as sns
fig = plt.figure(figsize = (15,7))
axs = fig.subplots(nrows = 1, ncols = 2)
axs[0] = sns.barplot(data = scores, x = "Method", y = "MSE", ax = axs[0], color = "blue", alpha = 0.7)
plt.tight_layout()
axs[0].set_xticklabels(labels = ["scDART", "LIGER", "Seurat", "UnionCom", "MMD-MA", "scJoint"], rotation = 45)
axs[0].set_ylabel("MSE")
newwidth = 0.5
for bar1 in axs[0].patches:
    x = bar1.get_x()
    width = bar1.get_width()
    centre = x+width/2.

    bar1.set_x(centre-newwidth/2.)
    bar1.set_width(newwidth)

show_values_on_bars(axs[0])

axs[1] = sns.barplot(data = scores, x = "Method", y = "cos_sim", ax = axs[1], color = "blue", alpha = 0.7)
plt.tight_layout()
axs[1].set_xticklabels(labels = ["scDART", "LIGER", "Seurat", "UnionCom", "MMD-MA", "scJoint"], rotation = 45)
axs[1].set_ylabel("cosine")
newwidth = 0.5
for bar1 in axs[1].patches:
    x = bar1.get_x()
    width = bar1.get_width()
    centre = x+width/2.

    bar1.set_x(centre-newwidth/2.)
    bar1.set_width(newwidth)

show_values_on_bars(axs[1])
fig.savefig("results_snare/pdist_consistency.png", bbox_inches = "tight")


# In[] good neighborhood overlap with mmd larger than 15
'''
scores = pd.read_csv("results_snare/scores_l1.csv")
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


scores = pd.read_csv("results_snare/scores_l2.csv")
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


# %%
