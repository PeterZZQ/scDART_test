import sys, os
sys.path.append('../../')
sys.path.append('../../scDART/')

from unioncom import UnionCom
import numpy as np
import pandas as pd
from umap import UMAP
from scipy.sparse import load_npz
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
import scDART.diffusion_dist as diff
import scDART.dataset as dataset
import scDART.utils as utils
import scDART.benchmark as bmk
   
def plot_latent(z1, z2, anno1 = None, anno2 = None, mode = "joint", save = None, figsize = (20,10), axis_label = "Latent"):
    fig = plt.figure(figsize = figsize)
    if mode == "modality":
        colormap = plt.cm.get_cmap("Paired")
        ax = fig.add_subplot()
        ax.scatter(z1[:,0], z1[:,1], color = colormap(1), label = "RNA", alpha = 1)
        ax.scatter(z2[:,0], z2[:,1], color = colormap(2), label = "ATAC", alpha = 1)
        ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(0.9, 1))
        
        ax.tick_params(axis = "both", which = "major", labelsize = 15)

        ax.set_xlabel(axis_label + " 1", fontsize = 19)
        ax.set_ylabel(axis_label + " 2", fontsize = 19)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  

    elif mode == "joint":
        ax = fig.add_subplot()
        cluster_types = np.unique(anno1)
        colormap = plt.cm.get_cmap("tab20", cluster_types.shape[0])

        for i, cluster_type in enumerate(cluster_types):
            index = np.where(anno1 == cluster_type)[0]
            ax.scatter(z1[index,0], z1[index,1], color = colormap(i), label = cluster_type, alpha = 1)
        
        cluster_types = np.unique(anno2)
        colormap = plt.cm.get_cmap("tab20", cluster_types.shape[0])
        for i, cluster_type in enumerate(cluster_types):
            index = np.where(anno2 == cluster_type)[0]
            ax.scatter(z2[index,0], z2[index,1], color = colormap(i), label = cluster_type, alpha = 1)

        ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(0.9, 1))
        
        ax.tick_params(axis = "both", which = "major", labelsize = 15)

        ax.set_xlabel(axis_label + " 1", fontsize = 19)
        ax.set_ylabel(axis_label + " 2", fontsize = 19)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)  


    elif mode == "separate":
        axs = fig.subplots(1,2)
        cluster_types = np.unique(anno1)
        colormap = plt.cm.get_cmap("tab20", cluster_types.shape[0])

        for i, cluster_type in enumerate(cluster_types):
            index = np.where(anno1 == cluster_type)[0]
            axs[0].scatter(z1[index,0], z1[index,1], color = colormap(i), label = cluster_type, alpha = 1)
        # axs[0].legend(fontsize = font_size)
        axs[0].legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(0.9, 1))
        axs[0].set_title("Dataset 1", fontsize = 25)

        axs[0].tick_params(axis = "both", which = "major", labelsize = 15)

        axs[0].set_xlabel(axis_label + " 1", fontsize = 19)
        axs[0].set_ylabel(axis_label + " 2", fontsize = 19)
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['top'].set_visible(False)  

        cluster_types = np.unique(anno2)
        colormap = plt.cm.get_cmap("tab20",  cluster_types.shape[0])

        for i, cluster_type in enumerate(cluster_types):
            index = np.where(anno2 == cluster_type)[0]
            axs[1].scatter(z2[index,0], z2[index,1], color = colormap(i), label = cluster_type, alpha = 1)
        # axs[1].axis("off")
        axs[1].legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(0.9, 1))
        axs[1].set_title("Dataset 2", fontsize = 25)

        axs[1].tick_params(axis = "both", which = "major", labelsize = 15)

        axs[1].set_xlabel(axis_label + " 1", fontsize = 19)
        axs[1].set_ylabel(axis_label + " 2", fontsize = 19)
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)  

    if save:
        fig.savefig(save, bbox_inches = "tight")


def lsi_ATAC(X, k = 100, use_first = False):
    """\
    Description:
    ------------
        Compute LSI with TF-IDF transform, i.e. SVD on document matrix, can do tsne on the reduced dimension

    Parameters:
    ------------
        X: cell by feature(region) count matrix
        k: number of latent dimensions
        use_first: since we know that the first LSI dimension is related to sequencing depth, we just ignore the first dimension since, and only pass the 2nd dimension and onwards for t-SNE
    
    Returns:
    -----------
        latent: cell latent matrix
    """    
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.decomposition import TruncatedSVD

    # binarize the scATAC-Seq count matrix
    bin_X = np.where(X < 1, 0, 1)
    
    # perform Latent Semantic Indexing Analysis
    # get TF-IDF matrix
    tfidf = TfidfTransformer(norm='l2', sublinear_tf=True)
    normed_count = tfidf.fit_transform(bin_X)

    # perform SVD on the sparse matrix
    lsi = TruncatedSVD(n_components = k, random_state=42)
    lsi_r = lsi.fit_transform(normed_count)
    
    # use the first component or not
    if use_first:
        return lsi_r
    else:
        return lsi_r[:, 1:]


libsize = 100

pca_op = PCA(n_components = 2)
umap_op = MDS(n_components = 2)
tsne_op = TSNE(n_components = 2)


# hema
counts_rna = pd.read_csv("../../data/hema/counts_rna.csv", index_col=0).values
anno_rna = pd.read_csv("../../data/hema/anno_rna.txt", header = None).values
if libsize is None:
    libsize = np.median(np.sum(counts_rna, axis = 1))

counts_rna = counts_rna/np.sum(counts_rna, axis = 1)[:, None] * libsize 
counts_rna = np.log1p(counts_rna)

counts_atac = pd.read_csv("../../data/hema/counts_atac.csv", index_col=0).values
counts_atac = np.where(counts_atac < 1, 0, 1)
anno_atac = pd.read_csv("../../data/hema/anno_atac.txt", header = None).values
counts_atac = lsi_ATAC(counts_atac, k = 100)

print("start hema")
uc = UnionCom.UnionCom(epoch_pd = 10000)
integrated_data = uc.fit_transform([counts_rna, counts_atac])
np.save(file = "./results_hema/lat_rna.npy", arr = integrated_data[0])
np.save(file = "./results_hema/lat_atac.npy", arr = integrated_data[1])

pca_latent = pca_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))
umap_latent = umap_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))
tsne_latent = tsne_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))

plot_latent(pca_latent[:counts_rna.shape[0],:], pca_latent[counts_rna.shape[0]:,:], anno1 = anno_rna, anno2 = anno_atac, 
mode = "separate", save = "./results_hema/pca.pdf", figsize = (35,10), axis_label = "Latent_pca")
plot_latent(umap_latent[:counts_rna.shape[0],:], umap_latent[counts_rna.shape[0]:,:], anno1 = anno_rna, anno2 = anno_atac, 
mode = "separate", save = "./results_hema/mds.pdf", figsize = (35,10), axis_label = "Latent_mds")
plot_latent(tsne_latent[:counts_rna.shape[0],:], tsne_latent[counts_rna.shape[0]:,:], anno1 = anno_rna, anno2 = anno_atac, 
mode = "separate", save = "./results_hema/tsne.pdf", figsize = (35,10), axis_label = "Latent_tsne")

# Endo
counts_rna = pd.read_csv("../../data/Endo/counts_rna.csv", index_col=0).values
anno_rna = pd.read_csv("../../data/Endo/anno_rna.txt", header = None).values
if libsize is None:
    libsize = np.median(np.sum(counts_rna, axis = 1))

counts_rna = counts_rna/np.sum(counts_rna, axis = 1)[:, None] * libsize 
counts_rna = np.log1p(counts_rna)

counts_atac = pd.read_csv("../../data/Endo/counts_atac.csv", index_col=0).values
counts_atac = np.where(counts_atac < 1, 0, 1)
anno_atac = pd.read_csv("../../data/Endo/anno_atac.txt", header = None).values
counts_atac = lsi_ATAC(counts_atac, k = 100)

print("start endo")
uc = UnionCom.UnionCom(epoch_pd = 10000)
integrated_data = uc.fit_transform([counts_rna, counts_atac])
np.save(file = "./results_endo/lat_rna.npy", arr = integrated_data[0])
np.save(file = "./results_endo/lat_atac.npy", arr = integrated_data[1])

pca_latent = pca_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))
umap_latent = umap_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))
tsne_latent = tsne_op.fit_transform(np.concatenate((integrated_data[0],integrated_data[1]), axis = 0))

plot_latent(pca_latent[:counts_rna.shape[0],:], pca_latent[counts_rna.shape[0]:,:], anno1 = anno_rna, anno2 = anno_atac, 
mode = "separate", save = "./results_endo/pca.pdf", figsize = (35,10), axis_label = "Latent_pca")
plot_latent(umap_latent[:counts_rna.shape[0],:], umap_latent[counts_rna.shape[0]:,:], anno1 = anno_rna, anno2 = anno_atac, 
mode = "separate", save = "./results_endo/mds.pdf", figsize = (35,10), axis_label = "Latent_mds")
plot_latent(tsne_latent[:counts_rna.shape[0],:], tsne_latent[counts_rna.shape[0]:,:], anno1 = anno_rna, anno2 = anno_atac, 
mode = "separate", save = "./results_endo/tsne.pdf", figsize = (35,10), axis_label = "Latent_tsne")


# snare
print("start snare")
counts_rna = pd.read_csv("../../data/snare-seq/counts_rna.csv", index_col=0).values
counts_atac = load_npz("../../data/snare-seq/counts_atac.npz").todense()
counts_atac = diff.lsi_ATAC(counts_atac, k = 100, use_first = False)
counts_rna = PCA(n_components = 100).fit_transform(counts_rna)

uc = UnionCom.UnionCom(epoch_pd = 10000, output_dim = 8)
integrated_data = uc.fit_transform([counts_rna, counts_atac])
z_rna = integrated_data[0]
z_atac = integrated_data[1]

neighov_unioncom = bmk.neigh_overlap(z_rna = z_rna, z_atac = z_atac, k = 30)
print("neighborhood overlap: " + str(neighov_unioncom))

rna_dataset = dataset.braincortex_rna(counts_dir = "../data/snare-seq/counts_rna.csv", anno_dir = "../data/snare-seq/anno.txt",anchor = None)
atac_dataset = dataset.braincortex_atac(counts_dir = "../data/snare-seq/counts_atac.npz", anno_dir = "../data/snare-seq/anno.txt",anchor = None)

pca_op = PCA(n_components = 2)
ae_coord = pca_op.fit_transform(np.concatenate((z_rna, z_atac), axis = 0))
ae_coordinates1 = ae_coord[:z_rna.shape[0],:]
ae_coordinates2 = ae_coord[z_rna.shape[0]:,:]
utils.plot_latent(z1 = ae_coordinates1, z2 = ae_coordinates2, anno1 = rna_dataset.cell_labels,
                  anno2 = atac_dataset.cell_labels, mode = "joint",save = "../results_snare/UnionCom/unioncom_paired8.png", figsize = (10,7), axis_label = "PCA")

