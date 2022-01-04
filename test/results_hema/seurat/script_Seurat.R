# # genome assembly and gene annotation packages
# # Mouse mm10 
# BiocManager::install(c('BSgenome.Mmusculus.UCSC.mm10', 'EnsDb.Mmusculus.v79'))
# # Human hg19
# BiocManager::install(c('BSgenome.Hsapiens.UCSC.hg19', 'EnsDb.Hsapiens.v75'))
# # Human hg38
# BiocManager::install(c('BSgenome.Hsapiens.UCSC.hg38', 'EnsDb.Hsapiens.v86'))
# install seurat v3.2
# need to use old spatstat
# install.packages('https://cran.r-project.org/src/contrib/Archive/spatstat/spatstat_1.64-1.tar.gz', repos=NULL,type="source")
# remotes::install_version("Seurat", version = "3.2")

rm(list =ls())
gc()
library(Seurat)
library(ggplot2)
library(Matrix)

setwd("~/Documents/xsede/scDART_full_test/test/results_hema/seurat")
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

# binarize
# endo.atac <- (endo.atac > 0) + 0

row.names(counts.atac) <-lapply(row.names(counts.atac), function(x){
  x <- strsplit(x, split = "_")[[1]]
  x <- paste0(x[1], ":", x[2], "-", x[3])
  return(x)})

# transform, gene by cell, using Seurat function
ref <- "~/Dropbox (GaTech)/Research/Projects/pipeline_integration/reference_genome/Homo_sapiens.GRCh37.82.gtf"
activity.matrix <- CreateGeneActivityMatrix(peak.matrix = counts.atac, annotation.file = ref,
                                            seq.levels = c(1:22, "X", "Y"), upstream = 2000, verbose = TRUE)

# create seurat obj of scATAC-Seq
start_time <- Sys.time()
seurat.atac <- CreateSeuratObject(counts = counts.atac, assay = "ATAC", project = "10x_ATAC")
# input the transformed matrix
seurat.atac[["ACTIVITY"]] <- CreateAssayObject(counts = activity.matrix)
# add in the meta info of scATAC-Seq data
seurat.atac <- AddMetaData(seurat.atac, metadata = labels.atac)
# filtering, no need for simulated data
# seurat.atac <- subset(seurat.atac, subset = nCount_ATAC > 5000)
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
seurat.rna <- AddMetaData(seurat.rna, metadata = labels.rna)
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

end_time <- Sys.time()
coembed <- RunPCA(coembed, features = genes.use, verbose = FALSE)
coembed <- RunUMAP(coembed, dims = 1:30)
coembed$celltype <- coembed$pop

p1 <- DimPlot(coembed, group.by = "tech")
p2 <- DimPlot(coembed, group.by = "celltype", label = TRUE, repel = TRUE)
CombinePlots(list(p1, p2))

write.table(coembed@reductions$pca@cell.embeddings, file = paste0("./pca_embedding.txt"), sep = "\t")
write.table(coembed@reductions$umap@cell.embeddings, file = paste0("./umap_embedding.txt"), sep = "\t")

