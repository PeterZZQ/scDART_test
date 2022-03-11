rm(list = ls())
gc()
library(rliger)
# library(liger)
library(Seurat)
# library(Signac)
library(Matrix)
library(patchwork)
# library(EnsDb.Hsapiens.v86)

# setwd("~/Documents/xsede/scDART_full_test/test")
################################################
#
# Using Raw data
#
################################################
# Read in the raw data
results_dir <- "results_acc/"
for(data_name in c("lin1", "lin2", "lin3", "lin4", "lin5", "lin6", 
                   "bifur1", "bifur2", "bifur3", "bifur4", "bifur5", "bifur6", 
                   "trifur1", "trifur2", "trifur3", "trifur4", "trifur5", "trifur6")){
  counts.rna <- as.matrix(read.table(file = paste0("../data/simulated/", data_name, "/GxC1.txt"), sep = "\t", header = FALSE))
  rownames(counts.rna) <- paste("Gene_", seq(1, dim(counts.rna)[1]), sep = "")
  colnames(counts.rna) <- paste("Cell_", seq(1, dim(counts.rna)[2]), sep = "")
  counts.atac <- as.matrix(read.table(file = paste0("../data/simulated/", data_name, "/RxC2.txt"), sep = "\t", header = FALSE))
  rownames(counts.atac) <- paste("Region_", seq(1, dim(counts.atac)[1]), sep = "")
  colnames(counts.atac) <- paste("Cell_", seq(dim(counts.rna)[2] + 1, dim(counts.rna)[2] + dim(counts.atac)[2]), sep = "")
  label.rna <- read.table(file = paste0("../data/simulated/", data_name, "/cell_label1.txt"), sep = "\t", header = TRUE)[["pop"]]
  label.atac <- read.table(file = paste0("../data/simulated/", data_name, "/cell_label2.txt"), sep = "\t", header = TRUE)[["pop"]]
  pt_rna <- read.table(paste0("../data/simulated/", data_name, "/pseudotime1.txt"), header = FALSE)[[1]]
  pt_atac <- read.table(paste0("../data/simulated/", data_name, "/pseudotime2.txt"), header = FALSE)[[1]]
  region2gene <- as.matrix(read.table(file = paste0("../data/simulated/", data_name, "/region2gene.txt"), sep = "\t", header = FALSE))
  rownames(region2gene) <- paste("Region_", seq(1, dim(region2gene)[1]), sep = "")
  colnames(region2gene) <- paste("Gene_", seq(1, dim(region2gene)[2]), sep = "")
  activity.matrix <- t(region2gene) %*% counts.atac
  
  ifnb_liger <- rliger::createLiger(list(rna1 = counts.rna, atac2 = activity.matrix), remove.missing = FALSE)
  ifnb_liger <- rliger::normalize(ifnb_liger, verbose = TRUE)
  # select all the genes
  # ifnb_liger <- rliger::selectGenes(ifnb_liger, var.thresh = 0)
  ifnb_liger@var.genes <- paste("Gene_", seq(1, dim(region2gene)[2]), sep = "")
  ifnb_liger <- rliger::scaleNotCenter(ifnb_liger, remove.missing = FALSE)
  
  num_clust <- 8
  ifnb_liger <- rliger::optimizeALS(ifnb_liger, k = num_clust)
  # quantile normalization
  print("Writing results...")
  
  ifnb_liger <- rliger::quantile_norm(ifnb_liger)
  H1 <- ifnb_liger@H$rna1
  H2 <- ifnb_liger@H$atac2
  
  H.norm1 <- ifnb_liger@H.norm[1:dim(H1)[1],]
  H.norm2 <- ifnb_liger@H.norm[(dim(H1)[1] + 1):(dim(H1)[1] + dim(H2)[1]),]
  write.csv(H.norm1, paste0(results_dir, data_name, "/Liger_H1.csv"))
  write.csv(H.norm2, paste0(results_dir, data_name, "/Liger_H2.csv"))
}



