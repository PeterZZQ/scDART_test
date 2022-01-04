# In[]
import sys, os
sys.path.append('../')
sys.path.append('../src/')


import numpy as np
import pandas as pd
import networkx as nx

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import time

from sklearn.decomposition import PCA

import diffusion_dist as diff
import dataset as dataset
import model as model
import train
import TI as ti
import benchmark as bmk

import utils as utils

import post_align as palign

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams["font.size"] = 20

results_dir = "./results_hyperparams/"

def calc_score(z_rna, z_atac, pt, cell_labels_rna, cell_labels_atac, data_name):
    """\
    Description:
    -----------
        Calculating the lineage separation score
    """
    
    z = np.concatenate((z_rna, z_atac), axis = 0)
    cell_labels = np.concatenate((cell_labels_rna, cell_labels_atac), axis = 0).squeeze()
    
    # alignment score
    groups, mean_cluster, conn = ti.backbone_inf(z, resolution = 0.1)
    mean_cluster = np.array(mean_cluster)
    root = groups[np.argmin(pt)]
    G = nx.from_numpy_matrix(conn)
    T = nx.dfs_tree(G, source = root)
    paths = [nx.shortest_path(G, source = root, target = x) for x,d in T.out_degree() if (d == 0)]
    
    cellcomps = []
    for path in paths:
        cell_comp = []
        for x in path:
            cell_comp.extend(np.where(groups == x)[0])
        cellcomps.append(cell_comp)

    if data_name[:5] == "bifur":
        scores = np.zeros((len(cellcomps), 2))
        cellcomps_gt = [[x for x in np.where(cell_labels == "3_1")[0]], [x for x in np.where(cell_labels == "3_2")[0]]]
    
    else:
        scores = np.zeros((len(cellcomps), 3))
        cellcomps_gt = [[x for x in np.where(cell_labels == "4_1")[0]], 
                        [x for x in np.where(cell_labels == "4_2")[0]],
                        [x for x in np.where(cell_labels == "4_3")[0]]]
    
    # calculate score matrix
    for i, cell_comp in enumerate(cellcomps):
        for j, cell_comp_gt in enumerate(cellcomps_gt):
            z1 = z[cell_comp_gt,:]
            z2 = z[cell_comp,:]
            scores[i,j] = bmk.alignscore(z1, z2, k = 10)
    
    # remove error detection
    if (data_name[:5] == "bifur") & (scores.shape[0] != 2):
        drop_comp = np.argsort(np.sum(scores, axis = 1))[-1:-3:-1]
        scores = scores[drop_comp,:]
    if (data_name[:6] == "trifur") & (scores.shape[0] != 3):
        print("data is:", data_name)
        print("score shape:", scores.shape[0])
        drop_comp = np.argsort(np.sum(scores, axis = 1))[-1:-4:-1]
        scores = scores[drop_comp,:]
    
    # calculate ave score
    ave_score = 0
    temp = scores.copy()
    
    for i in range(scores.shape[0]):
        ave_score += np.max(temp[:,i])
        temp = np.delete(temp, np.argmax(temp[:,i]), axis = 0)
    
    ave_score /= scores.shape[0]
    
    return scores, mean_cluster, groups, conn, ave_score


# In[1]
seeds = [0, 1, 2]
latent_dims = [4, 8, 32]
reg_ds = [1]
reg_gs = [0.1, 1, 10]
reg_mmds = [1, 10]
norms = ["l2"]

reg_d = reg_ds[0]
learning_rate = 3e-4
n_epochs = 500
ts = [30, 50, 70]
use_potential = True


# scores = pd.DataFrame(columns= ["data_name", "model", "ts", "latent_dim", "reg_g", "reg_mmd", "kendall-tau", "F1-score"])
scores = pd.read_csv(results_dir + "scores.csv", index_col = 0)

# for data_name in ["bifur1", "bifur2", "bifur3","bifur4", "bifur5", "bifur6",
#                  "trifur1", "trifur2", "trifur3","trifur4","trifur5","trifur6"]: 
for data_name in ["bifur1", "bifur2", "trifur1", "trifur2"]:

    counts_rna = pd.read_csv("../data/simulated/" + data_name + "/GxC1.txt", sep = "\t", header = None).values.T
    counts_atac = pd.read_csv("../data/simulated/" + data_name + "/RxC2.txt", sep = "\t", header = None).values.T
    label_rna = pd.read_csv("../data/simulated/" + data_name + "/cell_label1.txt", sep = "\t")["pop"].values.squeeze()
    label_atac = pd.read_csv("../data/simulated/" + data_name + "/cell_label2.txt", sep = "\t")["pop"].values.squeeze()
    pt_rna = pd.read_csv("../data/simulated/" + data_name + "/pseudotime1.txt", header = None).values.squeeze()
    pt_atac = pd.read_csv("../data/simulated/" + data_name + "/pseudotime2.txt", header = None).values.squeeze()
    
    # preprocessing
    libsize = 100
    counts_rna = counts_rna/np.sum(counts_rna, axis = 1)[:, None] * libsize
    counts_rna = np.log1p(counts_rna)

    rna_dataset = dataset.dataset(counts = counts_rna, anchor = np.argsort(pt_rna)[:10])
    atac_dataset = dataset.dataset(counts = counts_atac, anchor = np.argsort(pt_atac)[:10])
    coarse_reg = torch.FloatTensor(pd.read_csv("../data/simulated/" + data_name + "/region2gene.txt", sep = "\t", header = None).values).to(device)
    
    batch_size = int(max([len(rna_dataset),len(atac_dataset)])/4)

    if not os.path.exists(results_dir + data_name + "/"):
        print("make directory")
        os.makedirs(results_dir + data_name + "/")

    # loop through all hyper-parameters
    for norm in norms:
        for latent_dim in latent_dims:
            for reg_g in reg_gs:
                for reg_mmd in reg_mmds:
                    for seed in seeds:
                        print("Random seed: " + str(seed))
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed(seed)
                        np.random.seed(seed)

                        train_rna_loader = DataLoader(rna_dataset, batch_size = batch_size, shuffle = True)
                        train_atac_loader = DataLoader(atac_dataset, batch_size = batch_size, shuffle = True)

                        EMBED_CONFIG = {
                            "gact_layers": [atac_dataset.counts.shape[1], 1024, 512, rna_dataset.counts.shape[1]], 
                            "proj_layers": [rna_dataset.counts.shape[1], 512, 128, latent_dim], # number of nodes in each 
                            "learning_rate": learning_rate,
                            "n_epochs": n_epochs + 1,
                            "use_anchor": False,
                            "reg_d": reg_d,
                            "reg_g": reg_g,
                            "reg_mmd": reg_mmd,
                            "l_dist_type": "kl",
                            "device": device
                        }

                        start_time = time.time()
                        # calculate the diffusion distance
                        dist_rna = diff.diffu_distance(rna_dataset.counts.numpy(), ts = ts,
                                                    use_potential = use_potential, dr = "pca", n_components = 30)

                        dist_atac = diff.diffu_distance(atac_dataset.counts.numpy(), ts = ts,
                                                        use_potential = use_potential, dr = "lsi", n_components = 30)

                        # quantile normalization
                        # dist_atac = diff.quantile_norm(dist_atac, reference = dist_rna.reshape(-1), replace = True)

                        dist_rna = dist_rna/np.linalg.norm(dist_rna)
                        dist_atac = dist_atac/np.linalg.norm(dist_atac)
                        dist_rna = torch.FloatTensor(dist_rna).to(device)
                        dist_atac = torch.FloatTensor(dist_atac).to(device)

                        # ---------------------------------------------
                        #
                        # scDART
                        # 
                        # ---------------------------------------------
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

                        end_time = time.time()
                        # Plot results
                        with torch.no_grad():
                            z_rna = model_dict["encoder"](rna_dataset.counts.to(device)).cpu().detach()
                            z_atac = model_dict["encoder"](model_dict["gene_act"](atac_dataset.counts.to(device))).cpu().detach()

                        torch.save(model_dict, results_dir + data_name + "/model_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + ".pth")
                        np.save(file = results_dir + data_name + "/z_rna_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + ".npy", arr = z_rna)
                        np.save(file = results_dir + data_name + "/z_atac_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + ".npy", arr = z_atac)

                        # post-maching
                        z_rna, z_atac = palign.match_alignment(z_rna = z_rna, z_atac = z_atac, k = 10)
                        z_atac, z_rna = palign.match_alignment(z_rna = z_atac, z_atac = z_rna, k = 10)
                        
                        torch.cuda.empty_cache()
                        del model_dict
                        pca_op = PCA(n_components = 2)
                        z = pca_op.fit_transform(np.concatenate((z_rna.numpy(), z_atac.numpy()), axis = 0))
                        z_rna_pca = z[:z_rna.shape[0],:]
                        z_atac_pca = z[z_rna.shape[0]:,:]

                        utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = label_rna, 
                                        anno2 = label_atac, mode = "separate", save = results_dir + data_name + "/plot_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + ".png", 
                                        figsize = (20,10), axis_label = "PCA")
                        
                        # calculate the diffusion pseudotime
                        dpt_mtx = ti.dpt(np.concatenate((z_rna, z_atac), axis = 0), n_neigh = 10)
                        pt_infer = dpt_mtx[np.argmin(pt_rna), :]
                        pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
                        pt_infer = pt_infer/np.max(pt_infer)

                        pt_true = np.concatenate((pt_rna, pt_atac))
                        pt_true[pt_true.argsort()] = np.arange(len(pt_true))
                        pt_true = pt_true/np.max(pt_true)
                        
                        # backbone
                        z = np.concatenate((z_rna, z_atac), axis = 0)
                        cell_labels = np.concatenate((label_rna, label_atac), axis = 0).squeeze()
                        
                        groups, mean_cluster, conn = ti.backbone_inf(z, resolution = 0.01)
                        mean_cluster = np.array(mean_cluster)
                        root = groups[np.argmin(pt_infer)]
                        G = nx.from_numpy_matrix(conn)
                        T = nx.dfs_tree(G, source = root)
                        
                        # find trajectory backbone
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

                        scores = scores.append({
                            "data_name": data_name,
                            "model": "scDART", 
                            "ts": ts,
                            "latent_dim": latent_dim,
                            "reg_g": reg_g,
                            "reg_mmd": reg_mmd,
                            "norm": norm,
                            "seed": seed,            
                            "kendall-tau": kt,
                            "F1-score": F1,
                            "time": end_time - start_time
                        }, ignore_index = True)

                        # ---------------------------------------------
                        #
                        # scDART-anchor
                        # 
                        # ---------------------------------------------
                        # initialize the model
                        EMBED_CONFIG["use_anchor"] = True

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

                        end_time = time.time()
                        # Plot results
                        with torch.no_grad():
                            z_rna = model_dict["encoder"](rna_dataset.counts.to(device)).cpu().detach()
                            z_atac = model_dict["encoder"](model_dict["gene_act"](atac_dataset.counts.to(device))).cpu().detach()

                        torch.save(model_dict, results_dir + data_name + "/model_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + "_anchor.pth")
                        np.save(file = results_dir + data_name + "/z_rna_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + "_anchor.npy", arr = z_rna)
                        np.save(file = results_dir + data_name + "/z_atac_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + "_anchor.npy", arr = z_atac)

                        # post-maching
                        z_rna, z_atac = palign.match_alignment(z_rna = z_rna, z_atac = z_atac, k = 10)
                        z_atac, z_rna = palign.match_alignment(z_rna = z_atac, z_atac = z_rna, k = 10)
                        
                        torch.cuda.empty_cache()
                        del model_dict
                        pca_op = PCA(n_components = 2)
                        z = pca_op.fit_transform(np.concatenate((z_rna.numpy(), z_atac.numpy()), axis = 0))
                        z_rna_pca = z[:z_rna.shape[0],:]
                        z_atac_pca = z[z_rna.shape[0]:,:]

                        utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = label_rna, 
                                        anno2 = label_atac, mode = "separate", save = results_dir + data_name + "/plot_" + str(latent_dim) + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_" + str(seed) + "_" + norm + "_anchor.png", 
                                        figsize = (20,10), axis_label = "PCA")
                        
                        # calculate the diffusion pseudotime
                        dpt_mtx = ti.dpt(np.concatenate((z_rna, z_atac), axis = 0), n_neigh = 10)
                        pt_infer = dpt_mtx[np.argmin(pt_rna), :]
                        pt_infer[pt_infer.argsort()] = np.arange(len(pt_infer))
                        pt_infer = pt_infer/np.max(pt_infer)

                        pt_true = np.concatenate((pt_rna, pt_atac))
                        pt_true[pt_true.argsort()] = np.arange(len(pt_true))
                        pt_true = pt_true/np.max(pt_true)
                        
                        # backbone
                        z = np.concatenate((z_rna, z_atac), axis = 0)
                        cell_labels = np.concatenate((label_rna, label_atac), axis = 0).squeeze()
                        
                        groups, mean_cluster, conn = ti.backbone_inf(z, resolution = 0.01)
                        mean_cluster = np.array(mean_cluster)
                        root = groups[np.argmin(pt_infer)]
                        G = nx.from_numpy_matrix(conn)
                        T = nx.dfs_tree(G, source = root)
                        
                        # find trajectory backbone
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

                        scores = scores.append({
                            "data_name": data_name,
                            "model": "scDART-anchor", 
                            "ts": ts,
                            "latent_dim": latent_dim,
                            "reg_g": reg_g,
                            "reg_mmd": reg_mmd,
                            "norm": norm,
                            "seed": seed,            
                            "kendall-tau": kt,
                            "F1-score": F1,
                            "time": end_time - start_time
                        }, ignore_index = True)

scores.to_csv(results_dir + "scores_full.csv")
# In[2]
scores = pd.read_csv(results_dir + "scores.csv", index_col = 0)
scores = scores[scores["norm"] == "l1"]
scores = scores[scores["model"] == "scDART"]

boxprops = dict(linestyle='-', linewidth=3,alpha=.7)
medianprops = dict(linestyle='-', linewidth=3.5, color='firebrick')

fig = plt.figure(figsize = (15,5))
ax = fig.subplots(nrows = 1, ncols = 2)
scores_1 = scores[scores["reg_mmd"] == 1]
scores_10 = scores[scores["reg_mmd"] == 10]

sns.boxplot(data = scores_1, x = "latent_dim", y = "F1-score", hue = "reg_g", ax = ax[0], palette=sns.color_palette("Set2"), 
            boxprops=boxprops, medianprops = medianprops)

sns.boxplot(data = scores_10, x = "latent_dim", y = "F1-score", hue = "reg_g", ax = ax[1], palette=sns.color_palette("Set2"), 
            boxprops=boxprops, medianprops = medianprops)

# ax[0].legend(loc='upper left', frameon = True, ncol = 1, bbox_to_anchor=(1.04, 1), title = "$\lambda_{g}$")
ax[0].get_legend().remove()
ax[1].legend(loc='upper left', frameon = True, ncol = 1, bbox_to_anchor=(1.04, 1), title = "$\lambda_{g}$")
ax[0].set_xlabel("Latent dimension")
ax[1].set_xlabel("Latent dimension")
ax[0].set_ylabel("F1 score")
ax[1].set_ylabel("F1 score")
ax[0].set_title("$\lambda_{mmd} = 1$")
ax[1].set_title("$\lambda_{mmd} = 10$")
ax[0].set_ylim([0,1])
ax[1].set_ylim([0,1])
plt.tight_layout()
fig.savefig(results_dir + "F1.png", bbox_inches = "tight")

fig = plt.figure(figsize = (15,5))
ax = fig.subplots(nrows = 1, ncols = 2)
scores_1 = scores[scores["reg_mmd"] == 1]
scores_10 = scores[scores["reg_mmd"] == 10]

sns.boxplot(data = scores_1, x = "latent_dim", y = "kendall-tau", hue = "reg_g", ax = ax[0], palette=sns.color_palette("Set2"), 
            boxprops=boxprops, medianprops = medianprops)

sns.boxplot(data = scores_10, x = "latent_dim", y = "kendall-tau", hue = "reg_g", ax = ax[1], palette=sns.color_palette("Set2"), 
            boxprops=boxprops, medianprops = medianprops)

# ax[0].legend(loc='upper left', frameon = True, ncol = 1, bbox_to_anchor=(1.04, 1), title = "$\lambda_{g}$")
ax[0].get_legend().remove()
ax[1].legend(loc='upper left', frameon = True, ncol = 1, bbox_to_anchor=(1.04, 1), title = "$\lambda_{g}$")
ax[0].set_xlabel("Latent dimension")
ax[1].set_xlabel("Latent dimension")
ax[0].set_ylabel("kendall-tau score")
ax[1].set_ylabel("kendall-tau score")
ax[0].set_title("$\lambda_{mmd} = 1$")
ax[1].set_title("$\lambda_{mmd} = 10$")

plt.tight_layout()
fig.savefig(results_dir + "kt.png", bbox_inches = "tight")

# In[3]
scores = pd.read_csv(results_dir + "scores.csv", index_col = 0)
scores = scores[scores["norm"] == "l1"]
scores = scores[scores["model"] == "scDART-anchor"]

boxprops = dict(linestyle='-', linewidth=3,alpha=.7)
medianprops = dict(linestyle='-', linewidth=3.5, color='firebrick')

fig = plt.figure(figsize = (15,5))
ax = fig.subplots(nrows = 1, ncols = 2)
scores_1 = scores[scores["reg_mmd"] == 1]
scores_10 = scores[scores["reg_mmd"] == 10]

sns.boxplot(data = scores_1, x = "latent_dim", y = "F1-score", hue = "reg_g", ax = ax[0], palette=sns.color_palette("Set2"), 
            boxprops=boxprops, medianprops = medianprops)

sns.boxplot(data = scores_10, x = "latent_dim", y = "F1-score", hue = "reg_g", ax = ax[1], palette=sns.color_palette("Set2"), 
            boxprops=boxprops, medianprops = medianprops)

# ax[0].legend(loc='upper left', frameon = True, ncol = 1, bbox_to_anchor=(1.04, 1), title = "$\lambda_{g}$")
ax[0].get_legend().remove()
ax[1].legend(loc='upper left', frameon = True, ncol = 1, bbox_to_anchor=(1.04, 1), title = "$\lambda_{g}$")
ax[0].set_xlabel("Latent dimension")
ax[1].set_xlabel("Latent dimension")
ax[0].set_ylabel("F1 score")
ax[1].set_ylabel("F1 score")
ax[0].set_title("$\lambda_{mmd} = 1$")
ax[1].set_title("$\lambda_{mmd} = 10$")
ax[0].set_ylim([0,1])
ax[1].set_ylim([0,1])
plt.tight_layout()
fig.savefig(results_dir + "F1-anchor.png", bbox_inches = "tight")

fig = plt.figure(figsize = (15,5))
ax = fig.subplots(nrows = 1, ncols = 2)
scores_1 = scores[scores["reg_mmd"] == 1]
scores_10 = scores[scores["reg_mmd"] == 10]

sns.boxplot(data = scores_1, x = "latent_dim", y = "kendall-tau", hue = "reg_g", ax = ax[0], palette=sns.color_palette("Set2"), 
            boxprops=boxprops, medianprops = medianprops)

sns.boxplot(data = scores_10, x = "latent_dim", y = "kendall-tau", hue = "reg_g", ax = ax[1], palette=sns.color_palette("Set2"), 
            boxprops=boxprops, medianprops = medianprops)

# ax[0].legend(loc='upper left', frameon = True, ncol = 1, bbox_to_anchor=(1.04, 1), title = "$\lambda_{g}$")
ax[0].get_legend().remove()
ax[1].legend(loc='upper left', frameon = True, ncol = 1, bbox_to_anchor=(1.04, 1), title = "$\lambda_{g}$")
ax[0].set_xlabel("Latent dimension")
ax[1].set_xlabel("Latent dimension")
ax[0].set_ylabel("kendall-tau score")
ax[1].set_ylabel("kendall-tau score")
ax[0].set_title("$\lambda_{mmd} = 1$")
ax[1].set_title("$\lambda_{mmd} = 10$")

plt.tight_layout()
fig.savefig(results_dir + "kt-anchor.png", bbox_inches = "tight")


# %%
