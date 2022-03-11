import torch
import numpy as np
import torch.nn.functional as F

##################################################
#
# Our implementation, from SAUCIE tensorflow
#
##################################################
def compute_pairwise_distances(x, y):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def _gaussian_kernel_matrix(x, y, device):    
    sigmas = torch.FloatTensor([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]).to(device)
    dist = compute_pairwise_distances(x, y)
    beta = 1. / (2. * sigmas[:,None])
    s = - beta.mm(dist.reshape((1, -1)) )
    result =  torch.sum(torch.exp(s), dim = 0)
    return result


def maximum_mean_discrepancy(x, y, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')): #Function to calculate MMD value    
    cost = torch.mean(_gaussian_kernel_matrix(x, x, device))
    cost += torch.mean(_gaussian_kernel_matrix(y, y, device))
    cost -= 2.0 * torch.mean(_gaussian_kernel_matrix(x, y, device))
    cost = torch.sqrt(cost ** 2 + 1e-9)
    if cost.data.item()<0:
        cost = torch.FloatTensor([0.0]).to(device)

    return cost

def maximum_mean_discrepancy_ext(xs, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')): #Function to calculate MMD value
    nbatches = len(xs)
    ref_batch = 0
    # assuming batch 0 is the reference batch
    x_ref = xs[ref_batch] 
    cost = 0
    # within batch
    for batch in range(nbatches):
        if batch == ref_batch:
            cost += nbatches * torch.mean(_gaussian_kernel_matrix(xs[batch], xs[batch], device))
        else:
            cost += torch.mean(_gaussian_kernel_matrix(xs[batch], xs[batch], device))
    
    # between batches
    for batch in range(1, nbatches):
        cost -= 2.0 * torch.mean(_gaussian_kernel_matrix(xs[ref_batch], xs[batch], device))
    
    cost = torch.sqrt(cost ** 2 + 1e-9)
    if cost.data.item()<0:
        cost = torch.FloatTensor([0.0]).to(device)

    return cost


##################################################
#
# SAUCIE pytorch: https://github.com/khigashi1987/SAUCIE_PyTorch/blob/master/saucie_pytorch.py
#
##################################################
def pairwise_dist(x1, x2):
    r1 = torch.sum(x1*x1, 1, keepdim=True)
    r2 = torch.sum(x2*x2, 1, keepdim=True)
    K = r1 - 2*torch.matmul(x1, torch.t(x2)) + torch.t(r2)
    return K
    
def gaussian_kernel_matrix(dist):
    # Multi-scale RBF kernel. (average of some bandwidths of gaussian kernels)
    # This must be properly set for the scale of embedding space
    sigmas = [1e-5, 1e-4, 1e-3, 1e-2]
    beta = 1. / (2. * torch.unsqueeze(torch.tensor(sigmas), 1))
    s = torch.matmul(beta, torch.reshape(dist, (1, -1)))
    return torch.reshape(torch.sum(torch.exp(-s), 0), dist.shape) / len(sigmas)


def reg_b(embed, labels):
    """
    Maximum Mean Discrepancy regularization.
    """
    labels = labels.reshape(-1)
    # number of clusters except for the reference cluster (-1)
    n_others = torch.unique(labels).size()[0] - 1
    # the embedding divided by the mean value of the embedding, consider to be normalization
    e = embed / torch.mean(embed)
    # calculate pairwise squared distance, and transform into kernel matrix
    K = pairwise_dist(e, e)
    K = K / torch.max(K)
    K = gaussian_kernel_matrix(K)
    loss = 0
    # reference batch (batch_ind == 0) vs. other batches
    # single-batch term in MMD
    for batch_ind in range(int(labels.max().detach().numpy()) + 1):
        # batch_K: subset the kernel similarity matrix to have only the cells correspond to batch_ind 
        batch_rows = K[labels == batch_ind, :]
        batch_K = torch.t(batch_rows)[labels == batch_ind, :]

        # batch_nrows: number of cells correspond to the batch
        batch_nrows = torch.sum(torch.ones_like(labels)[labels == batch_ind]).float()
        # empirical mean of the kernel similarity matrix of cells within batch_ind
        var_within_batch = torch.sum(batch_K) / (batch_nrows**2)
        if batch_ind == 0:
            # the reference batch is calculated for n_other times
            loss += var_within_batch * n_others
        else:
            # each other batch is only calculated for once
            loss += var_within_batch
    
    # between-batches (reference vs. other batch) term in MMD
    # number of cells in the reference batch
    nrows_ref = torch.sum(torch.ones_like(labels)[labels == 0]).float()
    for batch_ind_vs in range(int(labels.max().detach().numpy())+1)[1:]:
        # K_bw: subset the kernel similarity matrix (nrows: ncells in the reference batch, ncols: ncells in the other batch)
        K_bw = K[labels == 0, :]
        K_bw = torch.t(K_bw)[labels == batch_ind_vs, :]
        K_bw = torch.sum(torch.t(K_bw))
        nrows_vs = torch.sum(torch.ones_like(labels)[labels == batch_ind_vs]).float()
        loss -= (2*K_bw) / (nrows_ref * nrows_vs)
    return loss