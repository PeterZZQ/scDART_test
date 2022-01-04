rm(list = ls())
gc()
library(rliger)
library(Seurat)
# library(Signac)
library(Matrix)
library(patchwork)
# library(EnsDb.Hsapiens.v86)

setwd("~/Documents/xsede/scDART_full_test/test/results_hema/liger")
################################################
#
# Using processed data
#
################################################
counts.atac <- read.table(file = '../../../data/hema/counts_atac.csv', sep = ',', header = TRUE, row.names = 1)
counts.rna <- read.table(file = '../../../data/hema/counts_rna.csv', sep = ',', header = TRUE, row.names = 1)
labels.rna <- read.table(file = '../../../data/hema/anno_rna.txt', sep = '\t', header = FALSE)
colnames(labels.rna) <- c("celltype")
labels.atac <- read.table(file = '../../../data/hema/anno_atac.txt', sep = '\t', header = FALSE)
colnames(labels.atac) <- c("celltype")
counts.atac <- t(counts.atac)
counts.rna <- t(counts.rna)

# change row names
row.names(counts.atac) <-lapply(row.names(counts.atac), function(x){
  x <- strsplit(x, split = "_")[[1]]
  x <- paste0(x[1], ":", x[2], "-", x[3])
  return(x)})

# transform, gene by cell, using Seurat function
ref <- "~/Dropbox (GaTech)/Research/Projects/pipeline_integration/reference_genome/Homo_sapiens.GRCh37.82.gtf"
activity.matrix <- CreateGeneActivityMatrix(peak.matrix = counts.atac, annotation.file = ref,
                                                    seq.levels = c(1:22, "X", "Y"), upstream = 2000, verbose = TRUE)

num_clust <- 6

ifnb_liger <- rliger::createLiger(list(rna1 = counts.rna.sub, atac2 = activity.matrix))

# already normalized
ifnb_liger <- rliger::normalize(ifnb_liger, verbose = TRUE)

ifnb_liger <- rliger::selectGenes(ifnb_liger)
ifnb_liger <- rliger::scaleNotCenter(ifnb_liger)

ifnb_liger <- rliger::optimizeALS(ifnb_liger, k = num_clust)
# quantile normalization
print("Writing results...")

ifnb_liger <- rliger::quantile_norm(ifnb_liger)
H1 <- ifnb_liger@H$rna1
H2 <- ifnb_liger@H$atac2
# write before post-processing
# write.csv(H1, "cell_factors/k_30/H1.csv")
# write.csv(H2, "cell_factors/k_30/H2.csv")

H.norm1 <- ifnb_liger@H.norm[1:dim(H1)[1],]
H.norm2 <- ifnb_liger@H.norm[(dim(H1)[1] + 1):(dim(H1)[1] + dim(H2)[1]),]
write.csv(H.norm1, "H1.csv")
write.csv(H.norm2, "H2.csv")
# write.csv(ifnb_liger@clusters, paste0("cell_factors/k_30/clust_id.csv"))
