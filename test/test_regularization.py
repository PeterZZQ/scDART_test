# In[0]
import sys

sys.path.append('../')
sys.path.append('../scDART/')


import numpy as np
import pandas as pd

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from umap import UMAP

import scDART.diffusion_dist as diff
import scDART.dataset as dataset
import scDART.model as model
import scDART.loss as loss
# import scDART.train_new as train_mmd
import scDART.train as train_mmd
import scDART.utils as utils
import scDART.post_align as palign
import scDART.benchmark as bmk
import scDART.TI as ti
import networkx as nx
import seaborn as sns

from matplotlib import rcParams
labelsize = 16
rcParams['xtick.labelsize'] = labelsize
rcParams['ytick.labelsize'] = labelsize 
rcParams['axes.labelsize'] = labelsize
rcParams['axes.titlesize'] = labelsize
rcParams['legend.fontsize'] = labelsize


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

result_dir = "./results_regularization/"


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

def scDART_train(EMBED_CONFIG, reg_mtx, train_rna_loader, train_atac_loader, test_rna_loader, test_atac_loader, 
                 n_epochs = 1001, use_anchor = True, reg_d = 1, reg_g = 1, reg_mmd = 1):
    seed = EMBED_CONFIG["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # calculate the distance
    for data in test_rna_loader:
        dist_rna = diff.diffu_distance(data["count"].numpy(), ts = EMBED_CONFIG["ts"], 
        use_potential = False, dr = "pca", method = "exact", n_anchor = EMBED_CONFIG["num_anchors"])

    for data in test_atac_loader:
        dist_atac = diff.diffu_distance(data["count"].numpy(), ts = EMBED_CONFIG["ts"], 
        use_potential = False, dr = "lsi", method = "exact", n_anchor = EMBED_CONFIG["num_anchors"])

    dist_rna = dist_rna/np.linalg.norm(dist_rna)
    dist_atac = dist_atac/np.linalg.norm(dist_atac)

    # quantile norm, might still be necessary
    # if dist_rna.shape[0] > dist_atac.shape[0]:
    #     dist_atac = diff.quantile_norm(dist_mtx = dist_atac, reference = dist_rna, replace = False)
    # else:
    #     dist_rna = diff.quantile_norm(dist_mtx = dist_rna, reference = dist_atac, replace = False)

    dist_rna = torch.FloatTensor(dist_rna).to(device)
    dist_atac = torch.FloatTensor(dist_atac).to(device)

    genact = model.gene_act(features = EMBED_CONFIG["gact_layers"], dropout_rate = 0.0, negative_slope = 0.2).to(device)
    encoder = model.Encoder(features = EMBED_CONFIG["proj_layers"], dropout_rate = 0.0, negative_slope = 0.2).to(device)
    decoder = model.Decoder(features = EMBED_CONFIG["proj_layers"][::-1], dropout_rate = 0.0, negative_slope = 0.2).to(device)
    genact_t = model.gene_act_t(features = EMBED_CONFIG["gact_layers"][::-1], dropout_rate = 0.0, negative_slope = 0.2).to(device)
    model_dict = {"gene_act": genact, "encoder": encoder, "decoder": decoder, "gene_act_t": genact_t}

    learning_rate = EMBED_CONFIG['learning_rate']
    opt_genact = torch.optim.Adam(genact.parameters(), lr = learning_rate)
    opt_encoder = torch.optim.Adam(encoder.parameters(), lr = learning_rate)
    opt_decoder = torch.optim.Adam(decoder.parameters(), lr = learning_rate)
    opt_genact_t = torch.optim.Adam(genact_t.parameters(), lr = learning_rate)
    opt_dict = {"gene_act": opt_genact, "encoder": opt_encoder, "decoder": opt_decoder, "gene_act_t": opt_genact_t}

    train_mmd.match_latent(model = model_dict, opts = opt_dict, dist_atac = dist_atac, dist_rna = dist_rna, 
                        data_loader_rna = train_rna_loader, data_loader_atac = train_atac_loader, n_epochs = n_epochs, 
                        reg_mtx = reg_mtx, reg_d = reg_d, reg_g = reg_g, reg_mmd = reg_mmd, use_anchor = use_anchor, norm = "l1", 
                        mode = "kl")


    with torch.no_grad():
        for data in test_rna_loader:
            z_rna = model_dict["encoder"](data['count'].to(device)).cpu().detach()

        for data in test_atac_loader:
            z_atac = model_dict["encoder"](model_dict["gene_act"](data['count'].to(device))).cpu().detach()


    # post-maching
    z_rna, z_atac = palign.match_alignment(z_rna = z_rna, z_atac = z_atac, k = 10)
    z_atac, z_rna = palign.match_alignment(z_rna = z_atac, z_atac = z_rna, k = 10)

  
    return model_dict, z_rna, z_atac

def calc_score(z_rna, z_atac, pt, cell_labels_rna, cell_labels_atac, data_name):
    """\
    Description:
    -----------
        Calculating the lineage separation score
    """
    
    z = np.concatenate((z_rna, z_atac), axis = 0)
    cell_labels = np.concatenate((cell_labels_rna, cell_labels_atac), axis = 0).squeeze()
    
    # alignment score
    groups, mean_cluster, conn = ti.backbone_inf(z_rna, z_atac, resolution = 0.1)
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
scores = pd.DataFrame(columns= ["data_name", "ts", "num_anchors", "kendall-tau", "ave_score"])
use_anchor = True

# total number of cells are 3000
for data_name in ["bifur1", "bifur2", "bifur3","bifur4", "bifur5", "bifur6","bifur7", "bifur8", "bifur9"]:
    for reg_d in [0.1, 1, 10]:
        for reg_g in [0.1, 1, 10]:
            for reg_mmd in [0.1, 1, 10]:
                # set random seed
                torch.manual_seed(0)
                np.random.seed(0)

                rna_dataset = symsim2_rna(counts_dir = "../data/all_simulations/" + data_name + "/GxC1.txt", 
                                            anno_dir = "../data/all_simulations/" + data_name + "/cell_label1.txt",
                                            pt = "../data/all_simulations/" + data_name + "/pseudotime1.txt",        
                                            anchor = None, libsize = 100)

                atac_dataset = symsim2_atac(counts_dir = "../data/all_simulations/" + data_name + "/RxC2.txt", 
                                            anno_dir = "../data/all_simulations/" + data_name + "/cell_label2.txt",
                                            pt = "../data/all_simulations/" + data_name + "/pseudotime2.txt",
                                            anchor = None)
                
                cell_labels_rna = pd.read_csv("../data/all_simulations/" + data_name + "/cell_label1.txt", sep = "\t")["pop"].values
                cell_labels_atac = pd.read_csv("../data/all_simulations/" + data_name + "/cell_label2.txt", sep = "\t")["pop"].values

                
                print("data: " + data_name + ", number of cells: " + str(len(rna_dataset) + len(atac_dataset)))
                print()

                coarse_reg = torch.FloatTensor(pd.read_csv("../data/all_simulations/" + data_name + "/region2gene.txt", 
                                                            sep = "\t", header = None).values).to(device)


                batch_size = int(max([len(rna_dataset),len(atac_dataset)])/4)
                libsize = rna_dataset.get_libsize()

                train_rna_loader = DataLoader(rna_dataset, batch_size = batch_size, shuffle = True)
                train_atac_loader = DataLoader(atac_dataset, batch_size = batch_size, shuffle = True)
                test_rna_loader = DataLoader(rna_dataset, batch_size = len(rna_dataset), shuffle = False)
                test_atac_loader = DataLoader(atac_dataset, batch_size = len(atac_dataset), shuffle = False)

                EMBED_CONFIG = {
                    'gact_layers': [atac_dataset.counts.shape[1], 512, 256, rna_dataset.counts.shape[1]], 
                    'proj_layers': [rna_dataset.counts.shape[1], 128, 8], # number of nodes in each 
                    'learning_rate': 5e-4,
                    'ts':[10, 30, 50],
                    'seed': 0,
                    'num_anchors': 500
                }

                model_dict, z_rna, z_atac = scDART_train(EMBED_CONFIG = EMBED_CONFIG, reg_mtx = coarse_reg, 
                                                                        train_rna_loader = train_rna_loader, 
                                                                        train_atac_loader = train_atac_loader, 
                                                                        test_rna_loader = test_rna_loader, 
                                                                        test_atac_loader = test_atac_loader, 
                                                                        n_epochs = 1201, use_anchor = use_anchor, 
                                                                        reg_d = reg_d, reg_g = reg_g, reg_mmd = reg_mmd)

                # visualize the pseudotime accuracy
                pca_op = PCA(n_components = 2)
                z = pca_op.fit_transform(np.concatenate((z_rna.numpy(), z_atac.numpy()), axis = 0))
                z_rna_pca = z[:z_rna.shape[0],:]
                z_atac_pca = z[z_rna.shape[0]:,:]

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

                # calculate the topology preservation
                *_, ave_score = calc_score(z_rna, z_atac, pt_infer, cell_labels_rna, cell_labels_atac, data_name)

                scores = scores.append({"data_name": data_name, "reg_d": reg_d, "reg_g": reg_g, "reg_mmd": reg_mmd, "kendall-tau": bmk.kendalltau(pt_infer, pt_true), "ave_score": ave_score}, ignore_index=True)

                # save files
                np.save(file = result_dir + "detailed/" + data_name + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_z_rna.npy", arr = z_rna)
                np.save(file = result_dir + "detailed/" + data_name + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_z_atac.npy", arr = z_atac)

                utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = rna_dataset.cell_labels, 
                                    anno2 = atac_dataset.cell_labels, mode = "separate",
                                    save = result_dir + "detailed/" + data_name + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_clust.png", 
                                    figsize = (20,7), axis_label = "PCA")

                utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = rna_dataset.is_anchor.numpy(), 
                                    anno2 = atac_dataset.is_anchor.numpy(), mode = "separate",
                                    save = result_dir + "detailed/" + data_name + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_anchor.png", 
                                    figsize = (20,7), axis_label = "PCA")

                utils.plot_latent(z1 = z_rna_pca, z2 = z_atac_pca, anno1 = rna_dataset.cell_labels, 
                                    anno2 = atac_dataset.cell_labels, mode = "modality",
                                    save = result_dir + "detailed/" + data_name + "_" + str(reg_d) + "_" + str(reg_g) + "_" + str(reg_mmd) + "_mod.png", 
                                    figsize = (10,7), axis_label = "PCA")

scores.to_csv(result_dir + "score.csv")
# In[2]

fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot()
scores = pd.read_csv(result_dir + "score.csv", index_col = 0)
for reg_g in [0.1, 1, 10]:
    scores = scores[scores["reg_g"] == reg_g]
    sns.boxplot(data = scores, x = "reg_d", y = "kendall-tau", hue = "reg_mmd", ax = ax)
    ax.set_ylim([0,1])
    fig.savefig(result_dir + "kt_"+ str(reg_g) + ".png", bbox_inches = "tight")

    fig = plt.figure(figsize = (10,7))
    ax = fig.add_subplot()
    scores = pd.read_csv(result_dir + "score.csv", index_col = 0)
    sns.boxplot(data = scores, x = "reg_d", y = "ave_score", hue = "reg_mmd", ax = ax)
    ax.set_ylim([0,0.5])
    ax.set_ylabel("Lineage separation score")
    fig.savefig(result_dir + "ave_score_" + str(reg_g) + ".png", bbox_inches = "tight")

# %%
