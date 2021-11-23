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

import post_align as palign
from scipy.sparse import load_npz

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams["font.size"] = 20

results_dir = "results_acc/"

class symsim2_rna(Dataset):
    def __init__(self, counts_dir = "./data/symsim2/rand1/GxC.txt", anno_dir = "./data/symsim2/rand1/cell_label1.txt", pt = "./data/symsim2/rand1/pseudotime1.txt", anchor = None, libsize = None):
        counts = pd.read_csv(counts_dir, sep = "\t", header = None).values.T
        cell_labels = pd.read_csv(anno_dir, sep = "\t")["pop"].values
        self.pt = pd.read_csv(pt, sep = "\t", header = None).values.squeeze()

        
        if counts_dir.split("/")[-2].split("_")[0] == "linear":
            idx = np.random.choice(counts.shape[0], size = 1000, replace = False)
        else:
            idx = np.arange(counts.shape[0])

        if libsize is None:
            self.libsize = np.median(np.sum(counts, axis = 1))
        else:
            self.libsize = libsize
        
        counts = counts/np.sum(counts, axis = 1)[:, None] * self.libsize 
        # minor difference after log
        counts = np.log1p(counts)

        # update the libsize after the log transformation
        self.libsize = np.mean(np.sum(counts, axis = 1))

        self.counts = torch.FloatTensor(counts[idx,:])
        self.cell_labels = cell_labels[idx]
        self.pt = self.pt[idx]

        if anchor is not None:
            self.is_anchor = (self.cell_labels == anchor)
        else:
            self.is_anchor = self.pt < (0.05 * np.max(self.pt))
        
        self.is_anchor = torch.tensor(self.is_anchor)
        self.use_clust = False

    def __len__(self):
        return self.counts.shape[0]
    
    def get_libsize(self):
        return self.libsize
    
    def __getitem__(self, idx):
        # data original data, index the index of cell, label, corresponding labels, batch, corresponding batch number
        sample = {"count": self.counts[idx,:], "index": idx, "is_anchor": self.is_anchor[idx]}
        return sample

class symsim2_atac(Dataset):
    def __init__(self, counts_dir = "./data/symsim2/rand2/RxC.txt", anno_dir = "./data/symsim2/rand2/cell_label2.txt", pt = "./data/symsim2/rand1/pseudotime2.txt", anchor = None):
        counts = pd.read_csv(counts_dir, sep = "\t", header = None).values.T
        counts = np.where(counts < 1, 0, 1)
        cell_labels = pd.read_csv(anno_dir, sep = "\t")["pop"].values
        self.pt = pd.read_csv(pt, sep = "\t", header = None).values.squeeze()
        
        if counts_dir.split("/")[-2].split("_")[0] == "linear":
            idx = np.random.choice(counts.shape[0], size = 1000, replace = False)
        else:
            idx = np.arange(counts.shape[0])
        
        self.counts = torch.FloatTensor(counts[idx,:])
        self.cell_labels = cell_labels[idx]
        self.pt = self.pt[idx]
        if anchor is not None:
            self.is_anchor = (self.cell_labels == anchor)
        else:
            self.is_anchor = self.pt < (0.05 * np.max(self.pt))
            
        self.is_anchor = torch.tensor(self.is_anchor)
        self.use_clust = False

    def __len__(self):
        return self.counts.shape[0]
    
    
    def __getitem__(self, idx):
        sample = {"count": self.counts[idx,:], "index": idx, "is_anchor": self.is_anchor[idx]}
        return sample


# In[]
seeds = [0,1,2,3,4]
latent_dim = 32
learning_rate = 3e-4
n_epochs = 700
use_anchor = False
reg_d = 1
reg_g = 1
reg_mmd = 1
ts = [30, 50, 70]
use_potential = True

scores = pd.DataFrame(columns = ["dataset", "kendall-tau", "F1-score"])
for data_name in ["lin_rand1", "lin_rand2", "lin_rand3", 
                "lin_new1", "lin_new2", "lin_new3",
                "lin_new4", "lin_new5", "lin_new6",
                "bifur1", "bifur2", "bifur3",
                "bifur4", "bifur5", "bifur6",
                "bifur7", "bifur8", "bifur9",
                "trifur_rand1", "trifur_rand2", "trifur_rand3",
                "trifur_new1","trifur_new2","trifur_new3", 
                "trifur_new4","trifur_new5","trifur_new6"
                ]:  
    for use_anchor in [True]:
        for seed in seeds:
            print("Random seed: " + str(seed))
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)

            rna_dataset = symsim2_rna(counts_dir = "../data/all_simulations/" + data_name + "/GxC1.txt", 
                                        anno_dir = "../data/all_simulations/" + data_name + "/cell_label1.txt",
                                        pt = "../data/all_simulations/" + data_name + "/pseudotime1.txt",        
                                        anchor = None, libsize = 100)

            atac_dataset = symsim2_atac(counts_dir = "../data/all_simulations/" + data_name + "/RxC2.txt", 
                                        anno_dir = "../data/all_simulations/" + data_name + "/cell_label2.txt",
                                        pt = "../data/all_simulations/" + data_name + "/pseudotime2.txt",
                                        anchor = None)

            ####################################################################
            # scDART
            ####################################################################       
            coarse_reg = torch.FloatTensor(pd.read_csv("../data/all_simulations/" + data_name + "/region2gene.txt", 
                                                       sep = "\t", header = None).values).to(device)

            batch_size = int(max([len(rna_dataset),len(atac_dataset)])/2)
            libsize = rna_dataset.get_libsize()

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
                            reg_mtx = coarse_reg, reg_d = EMBED_CONFIG["reg_d"], reg_g = EMBED_CONFIG["reg_g"], reg_mmd = EMBED_CONFIG["reg_mmd"], use_anchor = EMBED_CONFIG["use_anchor"], norm = "l2", 
                            mode = EMBED_CONFIG["l_dist_type"])
            
            # Plot results
            with torch.no_grad():
                z_rna = model_dict["encoder"](rna_dataset.counts.to(device)).cpu().detach()
                z_atac = model_dict["encoder"](model_dict["gene_act"](atac_dataset.counts.to(device))).cpu().detach()

            if os.path.exists(results_dir + data_name + "/"):
                os.makedirs(results_dir + data_name + "/")
            np.save(file = results_dir + data_name + "/z_rna.npy", arr = z_rna)
            np.save(file = results_dir + data_name + "/z_atac.npy", arr = z_atac)
            
            # post-maching
            z_rna, z_atac = palign.match_alignment(z_rna = z_rna, z_atac = z_atac, k = 10)
            z_atac, z_rna = palign.match_alignment(z_rna = z_atac, z_atac = z_rna, k = 10)
            
            pca_op = PCA(n_components = 2)
            z = pca_op.fit_transform(np.concatenate((z_rna.numpy(), z_atac.numpy()), axis = 0))
            z_rna_pca = z[:z_rna.shape[0],:]
            z_atac_pca = z[z_rna.shape[0]:,:]

            utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = rna_dataset.cell_labels, 
                            anno2 = atac_dataset.cell_labels, mode = "separate", save = None, 
                            figsize = (30,10), axis_label = "PCA")
            
            # calculate the diffusion distance
            dpt_mtx = ti.dpt(np.concatenate((z_rna, z_atac), axis = 0), n_neigh = 10)
            pt_infer = dpt_mtx[np.argmin(rna_dataset.pt), :]
            pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
            pt_infer = pt_infer/np.max(pt_infer)

            pt_true = np.concatenate((rna_dataset.pt, atac_dataset.pt))
            pt_true[pt_true.argsort()] = np.arange(len(pt_true))
            pt_true = pt_true/np.max(pt_true)

            pt_rna = dpt_mtx[np.argmin(rna_dataset.pt), :z_rna.shape[0]]
            pt_atac = dpt_mtx[np.argmin(rna_dataset.pt), z_rna.shape[0]:]
            
            # backbone
            z = np.concatenate((z_rna, z_atac), axis = 0)
            cell_labels = np.concatenate((rna_dataset.cell_labels, atac_dataset.cell_labels), axis = 0).squeeze()
            
            # alignment score
            groups, mean_cluster, conn = ti.backbone_inf(z_rna, z_atac, resolution = 0.1)
            mean_cluster = np.array(mean_cluster)
            root = groups[np.argmin(pt_infer)]
            G = nx.from_numpy_matrix(conn)
            T = nx.dfs_tree(G, source = root)
            
            # find all branches
            branching_nodes = [x for x,d in T.out_degree() if (d >= 2)]
            paths = [nx.shortest_path(G, source = root, target = x) for x,d in T.out_degree() if (d == 0)]
            branches = []
            for path in paths:
                last_idx = 0
                for idx, node in enumerate(path):
                    if node in branching_nodes:
                        if len(path[last_idx:idx]) > 0:
                            branches.append(path[last_idx:idx])
                            last_idx = idx
                if len(path[last_idx:]) > 0:
                    branches.append(path[last_idx:])         
            branches = sorted(list(set(map(tuple,branches))))

            # find cells for all branches
            cell_labels_predict = np.zeros(groups.shape)
            for idx, branch in enumerate(branches):
                for x in branch:
                    cell_labels_predict[groups == x] = idx
                    
            F1 = bmk.F1_branches(branches = cell_labels, branches_gt = cell_labels_predict)
            kt = bmk.kendalltau(pt_infer, pt_true)

            scores.append({
                "dataset": data_name,
                "kendall-tau": kt,
                "F1-score": F1
            })
            print("F1 score: {:.2f}".format(F1))
            print("Kendall-tau score: {:.2f}".format(kt))

scores.to_csv(results_dir + "scores.csv")
