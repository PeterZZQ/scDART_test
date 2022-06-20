rm(list =ls())
gc()
library(Signac)
library(Seurat)
library(GenomeInfoDb)
library(ggplot2)
library(patchwork)
library(Matrix)
library(rliger)
library(stringr)
setwd("/localscratch/ziqi/scDART_proj/scDART_test/test/")

# Read in the raw data
path <- "../../../CFRM/data/real/diag/Xichen/"
result_path <- "results_spleen/liger/"
counts.atac <- readMM(paste0(path, "small_ver/RxC2_small.mtx"))
rownames(counts.atac) <- read.table(paste0(path, "small_ver/regions.txt"), header = F, sep = ",")[[1]]
colnames(counts.atac) <- rownames(read.csv(paste0(path, "meta_c2.csv"), header = T, row.names = 1, sep = ","))
meta.atac <- read.csv(paste0(path, "meta_c2.csv"), header = T, row.names = 1, sep = ",")

counts.rna <- readMM(paste0(path, "small_ver/GxC1_small.mtx"))
rownames(counts.rna) <- read.table(paste0(path, "small_ver/genes.txt"), header = F, sep = ",")[[1]]
colnames(counts.rna) <- rownames(read.csv(paste0(path, "meta_c1.csv"), header = T, row.names = 1, sep = ","))

gene2region <- readMM(paste0(path, "small_ver/GxR_small.mtx"))
rownames(gene2region) <- rownames(counts.rna)
colnames(gene2region) <- rownames(counts.atac)
meta.rna <- read.csv(paste0(path, "meta_c1.csv"), header = T, row.names = 1, sep = ",")


################################################
#
# Using LIGER
#
################################################
activity.matrix <- gene2region %*% counts.atac
ifnb_liger <- rliger::createLiger(list(rna1 = counts.rna, atac2 = activity.matrix), remove.missing = FALSE)
ifnb_liger <- rliger::normalize(ifnb_liger, verbose = TRUE)
ifnb_liger <- rliger::selectGenes(ifnb_liger, datasets.use = 2)
# select all the genes
# ifnb_liger@var.genes <- paste("Gene_", seq(1, dim(region2gene)[2]), sep = "")
ifnb_liger <- rliger::scaleNotCenter(ifnb_liger, remove.missing = FALSE)

num_clust <- 20
ifnb_liger <- rliger::optimizeALS(ifnb_liger, k = num_clust)
# quantile normalization
print("Writing results...")

ifnb_liger <- rliger::quantile_norm(ifnb_liger)
H1 <- ifnb_liger@H$rna1
H2 <- ifnb_liger@H$atac2

write.csv(H1, paste0(result_path, "liger_c1.csv"))
write.csv(H2, paste0(result_path, "liger_c2.csv"))

H.norm1 <- ifnb_liger@H.norm[1:dim(H1)[1],]
H.norm2 <- ifnb_liger@H.norm[(dim(H1)[1] + 1):(dim(H1)[1] + dim(H2)[1]),]
write.csv(H.norm1, paste0(result_path, "liger_c1_norm.csv"))
write.csv(H.norm2, paste0(result_path, "liger_c2_norm.csv"))
