rm(list = ls())
gc()
library(rliger)
# library(liger)
library(Seurat)
# library(Signac)
library(Matrix)
library(patchwork)
# library(EnsDb.Hsapiens.v86)

setwd("~/Documents/xsede/scDART_full_test/test")
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
  rownames(counts.rna) <- paste("Gene-", seq(1, dim(counts.rna)[1]), sep = "")
  colnames(counts.rna) <- paste("Cell-", seq(1, dim(counts.rna)[2]), sep = "")
  counts.atac <- as.matrix(read.table(file = paste0("../data/simulated/", data_name, "/RxC2.txt"), sep = "\t", header = FALSE))
  rownames(counts.atac) <- paste("Region-", seq(1, dim(counts.atac)[1]), sep = "")
  colnames(counts.atac) <- paste("Cell-", seq(dim(counts.rna)[2] + 1, dim(counts.rna)[2] + dim(counts.atac)[2]), sep = "")
  label.rna <- read.table(file = paste0("../data/simulated/", data_name, "/cell_label1.txt"), sep = "\t", header = TRUE)[["pop"]]
  label.atac <- read.table(file = paste0("../data/simulated/", data_name, "/cell_label2.txt"), sep = "\t", header = TRUE)[["pop"]]
  pt_rna <- read.table(paste0("../data/simulated/", data_name, "/pseudotime1.txt"), header = FALSE)[[1]]
  pt_atac <- read.table(paste0("../data/simulated/", data_name, "/pseudotime2.txt"), header = FALSE)[[1]]
  region2gene <- as.matrix(read.table(file = paste0("../data/simulated/", data_name, "/region2gene.txt"), sep = "\t", header = FALSE))
  rownames(region2gene) <- paste("Region-", seq(1, dim(region2gene)[1]), sep = "")
  colnames(region2gene) <- paste("Gene-", seq(1, dim(region2gene)[2]), sep = "")
  activity.matrix <- t(region2gene) %*% counts.atac
  
  seurat.atac <- CreateSeuratObject(counts = counts.atac, assay = "ATAC", project = "10x_ATAC")
  seurat.atac[["ACTIVITY"]] <- CreateAssayObject(counts = activity.matrix)
  seurat.atac$tech <- "atac"

  # preprocess the transformed matrix
  DefaultAssay(seurat.atac) <- "ACTIVITY"
  seurat.atac <- FindVariableFeatures(seurat.atac)
  seurat.atac <- NormalizeData(seurat.atac)
  seurat.atac <- ScaleData(seurat.atac)
  # preprocess the scATAC matrix, 
  # We use all peaks that have at least 100 reads across all cells, and reduce dimensionality to 50.
  DefaultAssay(seurat.atac) <- "ATAC"
  VariableFeatures(seurat.atac) <- names(which(Matrix::rowSums(seurat.atac) > 100))
  seurat.atac <- RunLSI(seurat.atac, n = 50, scale.max = NULL)
  seurat.atac <- RunUMAP(seurat.atac, reduction = "lsi", dims = 1:50)
  
  # create seurat obj of scRNA-Seq
  seurat.rna <- CreateSeuratObject(counts=counts.rna, assay = "RNA", project = "full_matrix", min.cells = 1, min.features = 1)
  seurat.rna$tech <- "rna"
  
  # preprocessing scRNA-Seq data
  seurat.rna <- NormalizeData(seurat.rna, verbose = FALSE)
  # select highly variable features, use all features for symsim
  seurat.rna <- FindVariableFeatures(seurat.rna, selection.method = "vst", verbose = FALSE)
  
  # learn anchors
  transfer.anchors <- FindTransferAnchors(reference = seurat.rna, query = seurat.atac, features = VariableFeatures(object = seurat.rna), 
                                          reference.assay = "RNA", query.assay = "ACTIVITY", reduction = "cca")
  
  # learns coembedding
  # note that we restrict the imputation to variable genes from scRNA-seq, but could impute the
  # full transcriptome if we wanted to
  genes.use <- VariableFeatures(seurat.rna)
  refdata <- GetAssayData(seurat.rna, assay = "RNA", slot = "data")[genes.use, ]
  
  # refdata (input) contains a scRNA-seq expression matrix for the scRNA-seq cells.  imputation
  # (output) will contain an imputed scRNA-seq matrix for each of the ATAC cells
  imputation <- TransferData(anchorset = transfer.anchors, refdata = refdata, weight.reduction = seurat.atac[["lsi"]])
  
  # this line adds the imputed data matrix to the pbmc.atac object
  seurat.atac[["RNA"]] <- imputation
  coembed <- merge(x = seurat.rna, y = seurat.atac)
  
  # Finally, we run PCA and UMAP on this combined object, to visualize the co-embedding of both
  # datasets
  coembed <- ScaleData(coembed, features = genes.use, do.scale = FALSE)
  
  coembed <- RunPCA(coembed, features = genes.use, verbose = FALSE)
  coembed <- RunUMAP(coembed, dims = 1:30)

  write.table(coembed@reductions$pca@cell.embeddings, file = paste0(results_dir, data_name, "/Seurat_pca.txt"), sep = "\t")
  write.table(coembed@reductions$umap@cell.embeddings, file = paste0(results_dir, data_name, "/Seurat_umap.txt"), sep = "\t")
}



