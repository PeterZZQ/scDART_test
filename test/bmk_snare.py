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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
reg_mmd = reg_mmds[2]
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
                    figsize = (10,7), axis_label = "PCA")
utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = label_rna, 
                    anno2 = label_atac, mode = "modality", save = "results_snare/z_mod.png", 
                    figsize = (10,7), axis_label = "PCA")


z_rna, z_atac = palign.match_alignment(z_rna = z_rna, z_atac = z_atac, k = 10)
z_atac, z_rna = palign.match_alignment(z_rna = z_atac, z_atac = z_rna, k = 10)
z_rna = z_rna.numpy()
z_atac = z_atac.numpy()

# post-maching
pca_op = PCA(n_components = 2)
z = pca_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
z_rna_pca = z[:z_rna.shape[0],:]
z_atac_pca = z[z_rna.shape[0]:,:]
utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = label_rna, 
                    anno2 = label_atac, mode = "joint", save = "results_snare/z_joint_post.png", 
                    figsize = (10,7), axis_label = "PCA")
utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = label_rna, 
                    anno2 = label_atac, mode = "modality", save = "results_snare/z_mod_post.png", 
                    figsize = (10,7), axis_label = "PCA")

umap_op = UMAP(n_components = 2, min_dist = 0.8, random_state = 0)
z = umap_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
z_rna_umap = z[:z_rna.shape[0],:]
z_atac_umap = z[z_rna.shape[0]:,:]
utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = label_rna, 
                    anno2 = label_atac, mode = "joint", save = "results_snare/z_joint_post_umap.png", 
                    figsize = (10,7), axis_label = "UMAP")
utils.plot_latent(z1 = z_rna_umap, z2 = z_atac_umap, anno1 = label_rna, 
                    anno2 = label_atac, mode = "modality", save = "results_snare/z_mod_post_umap.png", 
                    figsize = (10,7), axis_label = "UMAP")


# In[] Plot backbone
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
            marker = cluster_types[i].split("_")[0] + "\_" + cluster_types[i].split("_")[1] 
            ax.plot(mean_cluster[i,0] - 0.007, mean_cluster[i,1] + 0.001, color = "blue", marker=  "$" + marker + "$", markersize = 70)

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
mean_cluster = pca_op.transform(np.array(mean_cluster))

plot_backbone(z_rna_pca, z_atac_pca, mode = "joint", mean_cluster = mean_cluster, groups = groups, T = T, figsize=(10,7), save = "results_snare/backbone.png", anno = cell_labels, axis_label = "PCA")
utils.plot_latent_pt(z1 = z_rna_pca, z2 = z_atac_pca, pt1 = pt_infer_rna, pt2 = pt_infer_atac, mode = "joint", save = "./results_snare/z_pt.png", figsize = (10,7), axis_label = "PCA")


# In[] Infer pseudotime
# infer backbone with leiden clustering
groups, mean_cluster, T = ti.backbone_inf(z_rna, z_atac, resolution = 0.05)
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

# In[]
# First check the Sox6
gene = "Sox6"
ordering = np.argsort(motif2gene[gene].values.squeeze())[::-1]
motif2gene_ordered = motif2gene.iloc[ordering, :]
motif2gene_ordered = motif2gene_ordered.loc[:, [gene]]
motif2gene_ordered.to_csv("results_snare/de_snare/1000/motifs_" + gene + ".csv")

pseudo_rna = model_dict["gene_act"](atac_dataset.counts.to(device)).detach().cpu().numpy()
genes = ["Ptprd", "Mki67", "Fabp7", "Top2a", "Mef2c", "Macrod2", "Tenm2", "Dab1", "Tnc", "Frmd4a", "Celf2"]
pseudo_rna = pd.DataFrame(data = pseudo_rna, index = counts_rna.index.values, columns = counts_rna.columns.values)
pseudo_rna_sorted = pseudo_rna.iloc[np.argsort(pt_infer_rna), :]
rna_sorted = counts_rna.iloc[np.argsort(pt_infer_rna), :]
pseudo_rna_sorted = pseudo_rna_sorted.loc[:, genes]
rna_sorted = rna_sorted.loc[:, genes]
fig = plt.figure(figsize = (20, 7))
axs = fig.subplots(1, 2)
sns.heatmap(pseudo_rna_sorted.T, ax = axs[0])
sns.heatmap(rna_sorted.T, ax = axs[1])
fig.savefig("results_snare/de_snare/predict_rna.png", bbox_inches = "tight")

for i, gene in enumerate(genes):
    spearman,_ = spearmanr(pseudo_rna_sorted.T.loc[gene,:], rna_sorted.T.loc[gene,:])
    pearson,_ = pearsonr(pseudo_rna_sorted.T.loc[gene,:], rna_sorted.T.loc[gene,:])
    print("gene: {:s}, spearman: {:.4f}, pearson: {:.4f}".format(gene, spearman, pearson))

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
mode = "joint", save = path + "liger_umap.png", figsize = (10,7), axis_label = "UMAP")
fig, axs = utils.plot_latent(umap_latent[:z_rna.shape[0],:], umap_latent[z_rna.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "modality", save = path + "liger_batches_umap.png", figsize = (10,7), axis_label = "UMAP")
fig, axs = utils.plot_latent(pca_latent[:z_rna.shape[0],:], pca_latent[z_rna.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "joint", save = None, figsize = (10,7), axis_label = "PCA")
axs.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig.savefig(path + "liger_pca.png", bbox_inches = "tight")
fig, axs = utils.plot_latent(pca_latent[:z_rna.shape[0],:], pca_latent[z_rna.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "modality", save = path + "liger_batches_pca.png", figsize = (10,7), axis_label = "PCA")
axs.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig.savefig(path + "liger_batches_pca.png", bbox_inches = "tight")

# 2. Seurat
path = "results_snare/seurat/"

coembed = pd.read_csv(path + "pca_embedding.txt", sep = "\t").values
z_rna_seurat = coembed[:label_rna.values.shape[0],:]
z_atac_seurat = coembed[label_rna.values.shape[0]:,:]

utils.plot_latent(z_rna_seurat, z_atac_seurat, label_rna.values, label_atac.values, mode = "modality", figsize = (10,7), axis_label = "PCA", save = path + "seurat_batches_pca.png")
utils.plot_latent(z_rna_seurat, z_atac_seurat, label_rna.values, label_atac.values, mode = "joint", figsize = (10,7), axis_label = "PCA", save = path + "seurat_pca.png")

coembed = pd.read_csv(path + "umap_embedding.txt", sep = "\t").values
z_rna_seurat = coembed[:label_rna.values.shape[0],:]
z_atac_seurat = coembed[label_rna.values.shape[0]:,:]
utils.plot_latent(z_rna_seurat, z_atac_seurat, label_rna.values, label_atac.values, mode = "modality", figsize = (10,7), axis_label = "PCA", save = path + "seurat_batches_umap.png")
utils.plot_latent(z_rna_seurat, z_atac_seurat, label_rna.values, label_atac.values, mode = "joint", figsize = (10,7), axis_label = "PCA", save = path + "seurat_umap.png")

# 3. unioncom
path = "results_snare/unioncom/"
z_rna_unioncom = np.load(path + "unioncom_rna_32.npy")
z_atac_unioncom = np.load(path + "unioncom_atac_32.npy")
integrated_data = (z_rna, z_atac)

pca_op = PCA(n_components = 2)
umap_op = UMAP(n_components = 2)

pca_latent = pca_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))
umap_latent = umap_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))

utils.plot_latent(umap_latent[:z_rna.shape[0],:], umap_latent[z_rna.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "joint", save = path + "unioncom_umap.png", figsize = (10,7), axis_label = "UMAP")
utils.plot_latent(umap_latent[:z_rna.shape[0],:], umap_latent[z_rna.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "modality", save = path + "unioncom_batches_umap.png", figsize = (10,7), axis_label = "UMAP")
utils.plot_latent(pca_latent[:z_rna.shape[0],:], pca_latent[z_rna.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "joint", save = path + "unioncom_pca.png", figsize = (10,7), axis_label = "PCA")
utils.plot_latent(pca_latent[:z_rna.shape[0],:], pca_latent[z_rna.shape[0]:,:], anno1 = label_rna.values, anno2 = label_atac.values, 
mode = "modality", save = path + "unioncom_batches_pca.png", figsize = (10,7), axis_label = "PCA")


# In[] Neighborhood overlap score, use also the unionCom, LIGER and Seurat
# Both the neighborhood overlap and pseudotime alignment are higher for scDART when the number of genes increase
path = "results_snare/liger/"
z_rna_liger = pd.read_csv(path + "H1_full.csv", index_col = 0)
z_atac_liger = pd.read_csv(path + "H2_full.csv", index_col = 0)

path = "results_snare/seurat/"
coembed = pd.read_csv(path + "pca_embedding.txt", sep = "\t").values
z_rna_seurat = coembed[:label_rna.values.shape[0],:]
z_atac_seurat = coembed[label_rna.values.shape[0]:,:]

path = "results_snare/unioncom/"
z_rna_unioncom = np.load(path + "unioncom_rna_32.npy")
z_atac_unioncom = np.load(path + "unioncom_atac_32.npy")

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

for k in range(10, 1000, 10):
    score_liger.append(bmk.neigh_overlap(z_rna_liger, z_atac_liger, k = k))
    score_unioncom.append(bmk.neigh_overlap(z_rna_unioncom, z_atac_unioncom, k = k))
    score_seurat.append(bmk.neigh_overlap(z_rna_seurat, z_atac_seurat, k = k))
    score_scdart.append(bmk.neigh_overlap(z_rna_scdart, z_atac_scdart, k = k))

score_liger = np.array(score_liger)
score_seurat = np.array(score_seurat)
score_unioncom = np.array(score_unioncom)
score_scdart = np.array(score_scdart)

fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot()
ax.plot(np.arange(10, 1000, 10), score_liger, label = "LIGER")
ax.plot(np.arange(10, 1000, 10), score_unioncom, label = "UnionCom")
ax.plot(np.arange(10, 1000, 10), score_seurat, label = "Seurat")
ax.plot(np.arange(10, 1000, 10), score_scdart, label = "scDART")
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


# correlation smaller than 0.87, the one reported in the paper.
print("scDART: spearman: {:.4f}, pearson: {:.4f}".format(spearman_scdart, pearson_scdart))
print("LIGER: spearman: {:.4f}, pearson: {:.4f}".format(spearman_liger, pearson_scdart))
print("Seurat: spearman: {:.4f}, pearson: {:.4f}".format(spearman_seurat, pearson_seurat))
print("UnionCom: spearman: {:.4f}, pearson: {:.4f}".format(spearman_unioncom, pearson_unioncom))

# In[] good neighborhood overlap with mmd larger than 15
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



# In[]
'''

'''


# %%
