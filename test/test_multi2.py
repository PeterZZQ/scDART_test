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

from umap import UMAP

import utils as utils
from unioncom import UnionCom

import post_align as palign
from scipy.sparse import load_npz

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams["font.size"] = 20

results_dir = "results_multi2/"

# In[]
seeds = [1]
latent_dims = [4, 8, 32]
reg_ds = [1, 10]
reg_gs = [0.01, 1, 10]
reg_mmds = [1, 10, 20, 30]

latent_dim = latent_dims[1]
reg_d = reg_ds[0]
reg_g = reg_gs[1]
# harder to merge, need to make mmd loss larger
reg_mmd = reg_mmds[0]

learning_rate = 3e-4
n_epochs = 500
use_anchor = True
ts = [30, 50, 70]
use_potential = True
norm = "l1"


# In[]
scores = pd.DataFrame(columns = ["dataset", "kendall-tau", "F1-score"])

for data_name in ["multi2_1"]: 
    if not os.path.exists(results_dir + data_name + "/"):
        print("make directory")
        os.makedirs(results_dir + data_name + "/")

    for seed in seeds:
        print("Random seed: " + str(seed))
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        counts_rna1 = pd.read_csv("../data/simulated/" + data_name + "/GxC1.txt", sep = "\t", header = None).values.T
        counts_rna2 = pd.read_csv("../data/simulated/" + data_name + "/GxC2.txt", sep = "\t", header = None).values.T
        counts_atac1 = pd.read_csv("../data/simulated/" + data_name + "/RxC3.txt", sep = "\t", header = None).values.T
        counts_atac2 = pd.read_csv("../data/simulated/" + data_name + "/RxC4.txt", sep = "\t", header = None).values.T
        label_rna1 = pd.read_csv("../data/simulated/" + data_name + "/cell_label1.txt", sep = "\t")["pop"].values.squeeze()
        label_rna2 = pd.read_csv("../data/simulated/" + data_name + "/cell_label2.txt", sep = "\t")["pop"].values.squeeze()
        label_atac1 = pd.read_csv("../data/simulated/" + data_name + "/cell_label3.txt", sep = "\t")["pop"].values.squeeze()
        label_atac2 = pd.read_csv("../data/simulated/" + data_name + "/cell_label4.txt", sep = "\t")["pop"].values.squeeze()
        pt_rna1 = pd.read_csv("../data/simulated/" + data_name + "/pseudotime1.txt", header = None).values.squeeze()
        pt_rna2 = pd.read_csv("../data/simulated/" + data_name + "/pseudotime2.txt", header = None).values.squeeze()
        pt_atac1 = pd.read_csv("../data/simulated/" + data_name + "/pseudotime3.txt", header = None).values.squeeze()
        pt_atac2 = pd.read_csv("../data/simulated/" + data_name + "/pseudotime4.txt", header = None).values.squeeze()

        # preprocessing
        libsize = 100
        counts_rna1 = counts_rna1/np.sum(counts_rna1, axis = 1)[:, None] * libsize
        counts_rna1 = np.log1p(counts_rna1)
        counts_rna2 = counts_rna2/np.sum(counts_rna2, axis = 1)[:, None] * libsize
        counts_rna2 = np.log1p(counts_rna2)



        pca_op = PCA(n_components = 5)
        z_rna1_pca = pca_op.fit_transform(counts_rna1)
        z_rna2_pca = pca_op.fit_transform(counts_rna2)
        z_atac1_pca = diff.lsi_ATAC(counts_atac1, k = 5)
        z_atac2_pca = diff.lsi_ATAC(counts_atac2, k = 5)

        utils.plot_latent_ext(zs = [z_rna1_pca, z_rna2_pca, z_atac1_pca, z_atac2_pca], annos = [label_rna1, label_rna2, label_atac1, label_atac2], 
                        mode = "separate", save = results_dir + data_name + "/plot_ori.png", 
                        figsize = (10,20), axis_label = "PCA")
                        

        rna_dataset1 = dataset.dataset(counts = counts_rna1, anchor = np.argsort(pt_rna1)[:10])
        rna_dataset2 = dataset.dataset(counts = counts_rna2, anchor = np.argsort(pt_rna2)[:10])
        atac_dataset1 = dataset.dataset(counts = counts_atac1, anchor = np.argsort(pt_atac1)[:10])
        atac_dataset2 = dataset.dataset(counts = counts_atac2, anchor = np.argsort(pt_atac2)[:10])
        coarse_reg = torch.FloatTensor(pd.read_csv("../data/simulated/" + data_name + "/region2gene.txt", sep = "\t", header = None).values).to(device)
        
        batch_size = int(max([len(rna_dataset1),len(rna_dataset2),len(atac_dataset1),len(atac_dataset2)])/4)

        train_rna_loader1 = DataLoader(rna_dataset1, batch_size = batch_size, shuffle = True)
        train_rna_loader2 = DataLoader(rna_dataset2, batch_size = batch_size, shuffle = True)
        train_atac_loader1 = DataLoader(atac_dataset1, batch_size = batch_size, shuffle = True)
        train_atac_loader2 = DataLoader(atac_dataset2, batch_size = batch_size, shuffle = True)

        EMBED_CONFIG = {
            "gact_layers": [atac_dataset1.counts.shape[1], 1024, 512, rna_dataset1.counts.shape[1]], 
            "proj_layers": [rna_dataset1.counts.shape[1], 512, 128, latent_dim], # number of nodes in each 
            "learning_rate": learning_rate,
            "n_epochs": n_epochs + 1,
            "use_anchor": use_anchor,
            "reg_d": reg_d,
            "reg_g": reg_g,
            "reg_mmd": reg_mmd,
            "l_dist_type": "kl",
            "device": device
        }

        start_time = time.time()
        # calculate the diffusion distance
        dist_rna1 = diff.diffu_distance(rna_dataset1.counts.numpy(), ts = ts,
                                    use_potential = use_potential, dr = "pca", n_components = 30)
        dist_rna2 = diff.diffu_distance(rna_dataset2.counts.numpy(), ts = ts,
                                    use_potential = use_potential, dr = "pca", n_components = 30)
        dist_atac1 = diff.diffu_distance(atac_dataset1.counts.numpy(), ts = ts,
                                        use_potential = use_potential, dr = "lsi", n_components = 30)
        dist_atac2 = diff.diffu_distance(atac_dataset2.counts.numpy(), ts = ts,
                                        use_potential = use_potential, dr = "lsi", n_components = 30)

        # quantile normalization
        dist_rna1 = diff.quantile_norm(dist_rna1, reference = dist_rna2.reshape(-1), replace = True)
        dist_atac1 = diff.quantile_norm(dist_atac1, reference = dist_rna2.reshape(-1), replace = True)
        dist_atac2 = diff.quantile_norm(dist_atac2, reference = dist_rna2.reshape(-1), replace = True)

        dist_rna1 = dist_rna1/np.linalg.norm(dist_rna1)
        dist_rna2 = dist_rna2/np.linalg.norm(dist_rna2)
        dist_atac1 = dist_atac1/np.linalg.norm(dist_atac1)
        dist_atac2 = dist_atac2/np.linalg.norm(dist_atac2)
        dist_rna1 = torch.FloatTensor(dist_rna1).to(device)
        dist_rna2 = torch.FloatTensor(dist_rna2).to(device)
        dist_atac1 = torch.FloatTensor(dist_atac1).to(device)
        dist_atac2 = torch.FloatTensor(dist_atac2).to(device)

        # initialize the model
        gene_act = model.gene_act(features = EMBED_CONFIG["gact_layers"], dropout_rate = 0.0, negative_slope = 0.2).to(device)
        encoder = model.Encoder(features = EMBED_CONFIG["proj_layers"], dropout_rate = 0.0, negative_slope = 0.2).to(device)
        model_dict = {"gene_act": gene_act, "encoder": encoder}

        opt_genact = torch.optim.Adam(gene_act.parameters(), lr = learning_rate)
        opt_encoder = torch.optim.Adam(encoder.parameters(), lr = learning_rate)
        opt_dict = {"gene_act": opt_genact, "encoder": opt_encoder}
        
        # training models
        train.match_latent_batches(model = model_dict, opts = opt_dict, dist_atacs = [dist_atac1, dist_atac2], dist_rnas = [dist_rna1, dist_rna2], 
                        data_loaders_rna = [train_rna_loader1, train_rna_loader2], data_loaders_atac = [train_atac_loader1, train_atac_loader2], n_epochs = EMBED_CONFIG["n_epochs"], 
                        reg_mtx = coarse_reg, reg_d = EMBED_CONFIG["reg_d"], reg_g = EMBED_CONFIG["reg_g"], reg_mmd = EMBED_CONFIG["reg_mmd"], use_anchor = EMBED_CONFIG["use_anchor"], norm = norm, 
                        mode = EMBED_CONFIG["l_dist_type"])

        end_time = time.time()
        # Plot results
        with torch.no_grad():
            z_rna1 = model_dict["encoder"](rna_dataset1.counts.to(device)).cpu().detach()
            z_rna2 = model_dict["encoder"](rna_dataset2.counts.to(device)).cpu().detach()
            z_atac1 = model_dict["encoder"](model_dict["gene_act"](atac_dataset1.counts.to(device))).cpu().detach()
            z_atac2 = model_dict["encoder"](model_dict["gene_act"](atac_dataset2.counts.to(device))).cpu().detach()

        torch.save(model_dict, results_dir + data_name + "/model_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + ".pth")
        np.save(file = results_dir + data_name + "/z_rna1_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + ".npy", arr = z_rna1)
        np.save(file = results_dir + data_name + "/z_atac1_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + ".npy", arr = z_atac1)
        np.save(file = results_dir + data_name + "/z_rna2_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + ".npy", arr = z_rna2)
        np.save(file = results_dir + data_name + "/z_atac2_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + ".npy", arr = z_atac2)
        
        # post-maching
        z_rna1, z_rna2 = palign.match_alignment(z_rna = z_rna1, z_atac = z_rna2, k = 10)
        z_rna1, z_atac1 = palign.match_alignment(z_rna = z_rna1, z_atac = z_atac1, k = 10)
        z_rna1, z_atac2 = palign.match_alignment(z_rna = z_rna1, z_atac = z_atac2, k = 10)
         
        torch.cuda.empty_cache()
        del model_dict
        pca_op = PCA(n_components = 2)
        z = pca_op.fit_transform(np.concatenate((z_rna1.numpy(), z_rna2.numpy(), z_atac1.numpy(), z_atac2.numpy()), axis = 0))
        z_rna1_pca = z[:z_rna1.shape[0],:]
        z_rna2_pca = z[z_rna1.shape[0]:(z_rna1.shape[0] + z_rna2.shape[0]),:]
        z_atac1_pca = z[(z_rna1.shape[0] + z_rna2.shape[0]):(z_rna1.shape[0] + z_rna2.shape[0] + z_atac1.shape[0]),:]
        z_atac2_pca = z[(z_rna1.shape[0] + z_rna2.shape[0] + z_atac1.shape[0]):,:]

        utils.plot_latent_ext(zs = [z_rna1_pca, z_rna2_pca, z_atac1_pca, z_atac2_pca], annos = [label_rna1, label_rna2, label_atac1, label_atac2], 
                        mode = "separate", save = results_dir + data_name + "/plot_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + "_sep.png", 
                        figsize = (10,20), axis_label = "PCA", markerscale = 6)

        utils.plot_latent_ext(zs = [z_rna1_pca, z_rna2_pca, z_atac1_pca, z_atac2_pca], annos = [label_rna1, label_rna2, label_atac1, label_atac2], 
                        mode = "modality", save = results_dir + data_name + "/plot_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + "_mod.png", 
                        figsize = (20,10), axis_label = "PCA", markerscale = 6)

# In[]
seeds = [0]
for data_name in ["paired1"]: 
    if not os.path.exists(results_dir + "neighborhood_ov/"):
        print("make directory")
        os.makedirs(results_dir + "neighborhood_ov/")

    for seed in seeds:
        print("Random seed: " + str(seed))
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        counts_rna1 = pd.read_csv("../data/simulated/" + data_name + "/GxC1.txt", sep = "\t", header = None).values.T
        counts_atac1 = pd.read_csv("../data/simulated/" + data_name + "/RxC1.txt", sep = "\t", header = None).values.T
        # counts_atac1 = pd.read_csv("../data/simulated/" + data_name + "/RxC2.txt", sep = "\t", header = None).values.T
        label_rna1 = pd.read_csv("../data/simulated/" + data_name + "/cell_label1.txt", sep = "\t")["pop"].values.squeeze()
        label_atac1 = pd.read_csv("../data/simulated/" + data_name + "/cell_label1.txt", sep = "\t")["pop"].values.squeeze()
        # label_atac1 = pd.read_csv("../data/simulated/" + data_name + "/cell_label2.txt", sep = "\t")["pop"].values.squeeze()
        pt_rna1 = pd.read_csv("../data/simulated/" + data_name + "/pseudotime1.txt", header = None).values.squeeze()
        pt_atac1 = pd.read_csv("../data/simulated/" + data_name + "/pseudotime1.txt", header = None).values.squeeze()
        # pt_atac1 = pd.read_csv("../data/simulated/" + data_name + "/pseudotime2.txt", header = None).values.squeeze()

        # preprocessing
        libsize = 100
        counts_rna1 = counts_rna1/np.sum(counts_rna1, axis = 1)[:, None] * libsize
        counts_rna1 = np.log1p(counts_rna1)


        pca_op = PCA(n_components = 5)
        z_rna1_pca = pca_op.fit_transform(counts_rna1)
        z_atac1_pca = diff.lsi_ATAC(counts_atac1, k = 5)
        
        utils.plot_latent_ext(zs = [z_rna1_pca, z_atac1_pca], annos = [label_rna1, label_atac1], 
                        mode = "separate", save = None, 
                        figsize = (10,20), axis_label = "PCA")
                        

        rna_dataset1 = dataset.dataset(counts = counts_rna1, anchor = np.argsort(pt_rna1)[:30])
        atac_dataset1 = dataset.dataset(counts = counts_atac1, anchor = np.argsort(pt_atac1)[:30])
        coarse_reg = torch.FloatTensor(pd.read_csv("../data/simulated/" + data_name + "/region2gene.txt", sep = "\t", header = None).values).to(device)
        
        batch_size = int(max([len(rna_dataset1),len(atac_dataset1)])/4)

        train_rna_loader1 = DataLoader(rna_dataset1, batch_size = batch_size, shuffle = True)
        train_atac_loader1 = DataLoader(atac_dataset1, batch_size = batch_size, shuffle = True)

        EMBED_CONFIG = {
            "gact_layers": [atac_dataset1.counts.shape[1], 1024, 512, rna_dataset1.counts.shape[1]], 
            "proj_layers": [rna_dataset1.counts.shape[1], 512, 128, latent_dim], # number of nodes in each 
            "learning_rate": learning_rate,
            "n_epochs": n_epochs + 1,
            "use_anchor": use_anchor,
            "reg_d": reg_d,
            "reg_g": reg_g,
            "reg_mmd": reg_mmd,
            "l_dist_type": "kl",
            "device": device
        }

        start_time = time.time()
        # calculate the diffusion distance
        dist_rna1 = diff.diffu_distance(rna_dataset1.counts.numpy(), ts = ts,
                                    use_potential = use_potential, dr = "pca", n_components = 30)
        dist_atac1 = diff.diffu_distance(atac_dataset1.counts.numpy(), ts = ts,
                                        use_potential = use_potential, dr = "lsi", n_components = 30)

        # quantile normalization
        # dist_atac1 = diff.quantile_norm(dist_atac1, reference = dist_rna1.reshape(-1), replace = True)

        dist_rna1 = dist_rna1/np.linalg.norm(dist_rna1)
        dist_atac1 = dist_atac1/np.linalg.norm(dist_atac1)
        dist_rna1 = torch.FloatTensor(dist_rna1).to(device)
        dist_atac1 = torch.FloatTensor(dist_atac1).to(device)

        # initialize the model
        gene_act = model.gene_act(features = EMBED_CONFIG["gact_layers"], dropout_rate = 0.0, negative_slope = 0.2).to(device)
        encoder = model.Encoder(features = EMBED_CONFIG["proj_layers"], dropout_rate = 0.0, negative_slope = 0.2).to(device)
        model_dict = {"gene_act": gene_act, "encoder": encoder}

        opt_genact = torch.optim.Adam(gene_act.parameters(), lr = learning_rate)
        opt_encoder = torch.optim.Adam(encoder.parameters(), lr = learning_rate)
        opt_dict = {"gene_act": opt_genact, "encoder": opt_encoder}
        
        # training models
        train.match_latent(model = model_dict, opts = opt_dict, dist_atac = dist_atac1, dist_rna = dist_rna1, 
                        data_loader_rna = train_rna_loader1, data_loader_atac = train_atac_loader1, n_epochs = EMBED_CONFIG["n_epochs"], 
                        reg_mtx = coarse_reg, reg_d = EMBED_CONFIG["reg_d"], reg_g = EMBED_CONFIG["reg_g"], reg_mmd = EMBED_CONFIG["reg_mmd"], use_anchor = EMBED_CONFIG["use_anchor"], norm = norm, 
                        mode = EMBED_CONFIG["l_dist_type"])

        end_time = time.time()
        # Plot results
        with torch.no_grad():
            z_rna1 = model_dict["encoder"](rna_dataset1.counts.to(device)).cpu().detach()
            z_atac1 = model_dict["encoder"](model_dict["gene_act"](atac_dataset1.counts.to(device))).cpu().detach()

        # post-maching
        # z_rna1, z_atac1 = palign.match_alignment(z_rna = z_rna1, z_atac = z_atac1, k = 10)
        # z_atac1, z_rna1 = palign.match_alignment(z_rna = z_atac1, z_atac = z_rna1, k = 10)
         
        torch.cuda.empty_cache()
        del model_dict
        pca_op = PCA(n_components = 2)
        z = pca_op.fit_transform(np.concatenate((z_rna1.numpy(), z_atac1.numpy()), axis = 0))
        z_rna1_pca = z[:z_rna1.shape[0],:]
        z_atac1_pca = z[z_rna1.shape[0]:(z_rna1.shape[0] + z_atac1.shape[0]),:]

        utils.plot_latent_ext(zs = [z_rna1_pca, z_atac1_pca], annos = [label_rna1, label_atac1], 
                        mode = "separate", save = None, figsize = (10,20), axis_label = "PCA")

        utils.plot_latent_ext(zs = [z_rna1_pca, z_atac1_pca], annos = [label_rna1, label_atac1], 
                        mode = "modality", save = None, figsize = (20,10), axis_label = "PCA")




# scDART
z_rna_scdart = z_rna1.cpu().detach().numpy()
z_atac_scdart = z_atac1.cpu().detach().numpy()
# Liger
z_rna_liger = pd.read_csv(f'results_multi2/neighborhood_ov/liger/{data_name}/Liger_H1.csv', index_col = 0).values
z_atac_liger = pd.read_csv(f'results_multi2/neighborhood_ov/liger/{data_name}/Liger_H2.csv', index_col = 0).values
# Seurat
coembed = pd.read_csv(f'results_multi2/neighborhood_ov/seurat/{data_name}/Seurat_pca.txt', sep = "\t").values
z_rna_seurat = coembed[:z_rna_scdart.shape[0],:]
z_atac_seurat = coembed[z_rna_scdart.shape[0]:,:]
# check variance of Seurat
var_rna_seurat = np.var(z_rna_seurat, axis = 0)
var_rna_seurat = var_rna_seurat/np.sum(var_rna_seurat)
var_atac_seurat = np.var(z_atac_seurat, axis = 0)
var_atac_seurat = var_atac_seurat/np.sum(var_atac_seurat)
print(var_rna_seurat)
print(var_atac_seurat)
print(np.argsort(var_rna_seurat)[::-1])
print(np.argsort(var_atac_seurat)[::-1])
# z_rna_seurat = z_rna_seurat[:, np.argsort(var_rna_seurat)[::-1]]
# z_atac_seurat = z_atac_seurat[:, np.argsort(var_atac_seurat)[::-1]]
# scJoint
# z_rna_scjoint = np.loadtxt(f'results_multi2/neighborhood_ov/scJoint/{data_name}/counts_rna_embeddings.txt')
# z_atac_scjoint = np.loadtxt(f'results_multi2/neighborhood_ov/scJoint/{data_name}/counts_atac_embeddings.txt')
# UnionCom
# uc = UnionCom.UnionCom(epoch_pd = 10000)
# integrated_data = uc.fit_transform([counts_rna1, counts_atac1])
# z_rna_unioncom = integrated_data[0]
# z_atac_unioncom = integrated_data[1]
# np.save(file = "results_multi2/neighborhood_ov/unioncom/z_rna.npy", arr = z_rna_unioncom)
# np.save(file = "results_multi2/neighborhood_ov/unioncom/z_atac.npy", arr = z_atac_unioncom)
# z_rna_unioncom = np.load(file = f'results_multi2/neighborhood_ov/unioncom/{data_name}/z_rna.npy')
# z_atac_unioncom = np.load(file = f'results_multi2/neighborhood_ov/unioncom/{data_name}/z_atac.npy')
# MMD-MA
# objs = []
# for seed in range(20):
#     objs.append(pd.read_csv(f'results_multi2/neighborhood_ov/MMDMA/{data_name}/objective_'+ str(seed) + ".txt", header = None).values[0,0])
# objs = np.array(objs)
# # the objective function that minimize the value
# seed = np.argmin(objs)
# # embedding of dataset 1 at the beginning of the training
# alpha_0 = pd.read_csv(f'results_multi2/neighborhood_ov/MMDMA/{data_name}/seed_' + str(seed) + "/alpha_hat_" + str(seed) + "_0.txt", sep = " ", header=None).values
# # embedding of dataset 1 at the end of the training
# alpha_10000 = pd.read_csv(f'results_multi2/neighborhood_ov/MMDMA/{data_name}/seed_' + str(seed) + "/alpha_hat_" + str(seed) + "_10000.txt", sep = " ", header=None).values
# # embedding of dataset 2 at the beginning of the training
# beta_0 = pd.read_csv(f'results_multi2/neighborhood_ov/MMDMA/{data_name}/seed_' + str(seed) + "/beta_hat_" + str(seed) + "_0.txt", sep = " ", header=None).values
# # embedding of dataset 2 at the end of the training
# beta_10000 = pd.read_csv(f'results_multi2/neighborhood_ov/MMDMA/{data_name}/seed_' + str(seed) + "/beta_hat_" + str(seed) + "_10000.txt", sep = " ", header=None).values
# z_rna_mmdma = alpha_10000
# z_atac_mmdma = beta_10000

score_liger = []
score_seurat = []
# score_unioncom = []
score_scdart = []
# score_mmdma = []
# score_scJoint = []
for k in range(10, 1000, 10):
    score_liger.append(bmk.neigh_overlap(z_rna_liger, z_atac_liger, k = k))
    # score_unioncom.append(bmk.neigh_overlap(z_rna_unioncom, z_atac_unioncom, k = k))
    score_seurat.append(bmk.neigh_overlap(z_rna_seurat, z_atac_seurat, k = k))
    score_scdart.append(bmk.neigh_overlap(z_rna_scdart, z_atac_scdart, k = k))
    # score_mmdma.append(bmk.neigh_overlap(z_rna_mmdma, z_atac_mmdma, k = k))
    # score_scJoint.append(bmk.neigh_overlap(z_rna_scjoint, z_atac_scjoint, k = k))

score_liger = np.array(score_liger)
score_seurat = np.array(score_seurat)
# score_unioncom = np.array(score_unioncom)
score_scdart = np.array(score_scdart)
# score_mmdma = np.array(score_mmdma)
# score_scJoint = np.array(score_scJoint)

fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot()
ax.plot(np.arange(10, 1000, 10), score_liger, label = "LIGER")
# ax.plot(np.arange(10, 1000, 10), score_unioncom, label = "UnionCom")
ax.plot(np.arange(10, 1000, 10), score_seurat, label = "Seurat")
ax.plot(np.arange(10, 1000, 10), score_scdart, label = "scDART")
# ax.plot(np.arange(10, 1000, 10), score_mmdma, label = "MMD-MA")
# ax.plot(np.arange(10, 1000, 10), score_scJoint, label = "scJoint")
ax.legend()
ax.set_xlabel("Neighborhood size")
ax.set_ylabel("Neighborhood overlap")
ax.set_xticks([0, 200, 400, 600, 800, 1000])
fig.savefig("results_multi2/neighborhood_ov/neigh_ov.png", bbox_inches = "tight")

# In[]

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


def dist_diff(z_rna, z_atac):
    mse = 1/z_rna.shape[0] * np.sum(np.sqrt(np.sum((z_rna - z_atac) ** 2, axis = 1)))
    return mse

mse_scdart = dist_diff(z_rna = z_rna_scdart, z_atac = z_atac_scdart)
mse_seurat = dist_diff(z_rna = z_rna_seurat, z_atac = z_atac_seurat)
mse_liger = dist_diff(z_rna = z_rna_liger, z_atac = z_atac_liger)
# mse_mmdma = dist_diff(z_rna = z_rna_mmdma, z_atac = z_atac_mmdma)
# mse_unioncom = dist_diff(z_rna = z_rna_unioncom, z_atac = z_atac_unioncom)
# mse_scjoint = dist_diff(z_rna = z_rna_scJoint, z_atac = z_atac_scJoint)

scores = pd.DataFrame(columns = ["Method", "MSE"])
scores["Method"] = np.array(["scDART", "LIGER", "Seurat"])
scores["MSE"] = np.array([mse_scdart, mse_liger, mse_seurat])
import seaborn as sns
fig = plt.figure(figsize = (7,5))
ax = fig.subplots(nrows = 1, ncols = 1)
ax = sns.barplot(data = scores, x = "Method", y = "MSE", ax = ax, color = "blue", alpha = 0.7)
plt.tight_layout()
ax.set_xticklabels(labels = ["scDART", "LIGER", "Seurat"], rotation = 45)
ax.set_ylabel("MSE")
newwidth = 0.5
for bar1 in ax.patches:
    x = bar1.get_x()
    width = bar1.get_width()
    centre = x+width/2.

    bar1.set_x(centre-newwidth/2.)
    bar1.set_width(newwidth)

show_values_on_bars(ax)

# In[]
def neigh_overlap(z_rna, z_atac, k = 30):
    dsize = z_rna.shape[0]
    _, neigh_ind = bmk.get_k_neigh_ind(np.concatenate((z_rna, z_atac), axis = 0), k = k)
#     print(neigh_ind)
    z1_z2 = ((neigh_ind[:dsize,:] - dsize - np.arange(dsize)[:, None]) == 0)
    # print(z1_z2)
    z2_z1 = (neigh_ind[dsize:,:] - np.arange(dsize)[:, None] == 0)
#     print(z2_z1)
    return 0.5 * (np.sum(z1_z2) + np.sum(z2_z1))/dsize, z1_z2, z2_z1


k = 10
score, z1_z2, z2_z1 = neigh_overlap(z_rna_seurat, z_atac_seurat, k = k)
utils.plot_latent_ext(zs = [z_rna_seurat, z_atac_seurat], annos = [z1_z2.astype(int), z2_z1.astype(int)], mode = "separate", save = None, figsize = (10,20), axis_label = "PCA")
print(np.sum(z1_z2))
print(np.sum(z2_z1))
k = 10
score, z1_z2, z2_z1 = neigh_overlap(z_rna_scdart, z_atac_scdart, k = k)
utils.plot_latent_ext(zs = [z_rna1_pca, z_atac1_pca], annos = [z1_z2.astype(int), z2_z1.astype(int)], mode = "separate", save = None, figsize = (10,20), axis_label = "PCA")
print(np.sum(z1_z2))
print(np.sum(z2_z1))
k = 100


# In[]
coembed = pd.read_csv(f"results_multi2/neighborhood_ov/seurat/{data_name}/Seurat_pca.txt", sep = "\t").values

coembed = UMAP(n_components = 2).fit_transform(coembed)
z_rna_seurat = coembed[:z_rna_scdart.shape[0],:]
z_atac_seurat = coembed[z_rna_scdart.shape[0]:,:]
utils.plot_latent_ext(zs = [z_rna_seurat, z_atac_seurat], annos = [label_rna1, label_atac1], mode = "separate", save = None, figsize = (10,20), axis_label = "PCA")
# %%
