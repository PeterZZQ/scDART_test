# In[]
import sys, os 
sys.path.append('../../../src/')

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

import utils

plt.rcParams["font.size"] = 20
# In[]
anno_rna = pd.read_csv("../../../data/Endo/anno_rna.txt", header = None).values
anno_atac = pd.read_csv("../../../data/Endo/anno_atac.txt", header = None).values
coembed = pd.read_csv("pca_embedding.txt", sep = "\t").values
coembed_full = pd.read_csv("pca_embedding_full.txt", sep = "\t").values
rna_embed = coembed[:anno_rna.shape[0],:]
atac_embed = coembed[anno_rna.shape[0]:,:]
rna_embed_full = coembed_full[:anno_rna.shape[0],:]
atac_embed_full = coembed_full[anno_rna.shape[0]:,:]

utils.plot_latent(rna_embed, atac_embed, anno_rna, anno_atac, mode = "modality", figsize = (10,7), axis_label = "PCA", save = "pca.png")
utils.plot_latent(rna_embed, atac_embed, anno_rna, anno_atac, mode = "joint", figsize = (10,7), axis_label = "PCA", save = "pca_joint.png")


utils.plot_latent(rna_embed_full, atac_embed_full, anno_rna, anno_atac, mode = "modality", figsize = (10,7), axis_label = "PCA", save = "pca_full.png")
utils.plot_latent(rna_embed_full, atac_embed_full, anno_rna, anno_atac, mode = "joint", figsize = (10,7), axis_label = "PCA", save = "pca_joint_full.png")

# %%
