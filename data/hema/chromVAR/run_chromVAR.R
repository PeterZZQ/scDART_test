rm(list = ls())
gc()
library(SummarizedExperiment)
library(chromVAR)
library(stringr)
# library(BSgenome.Hsapiens.UCSC.hg38)
# sequence alignment in hema uses hg19
library(BSgenome.Hsapiens.UCSC.hg19)
library(motifmatchr)
library(chromVARmotifs)
# library(chromVARxx)

setwd("/Users/ziqizhang/Documents/xsede/scDART_full_test/data/hema/chromVAR")

count <- read.csv(file = "../counts_atac.csv", sep = ",", header = TRUE, row.names = 1)
regions <- colnames(count)
barcodes <- rownames(count)

bedfile <- data.frame(str_split_fixed(regions, "_", 3))
colnames(bedfile) <- c("chr", "start", "end")
bedfile <- makeGRangesFromDataFrame(bedfile)

# Set up chromVAR
SE <- SummarizedExperiment(
  rowRanges = bedfile, 
  assays = list(counts = t(count)),
  colData = DataFrame(barcodes)
)
SE <- addGCBias(SE, BSgenome.Hsapiens.UCSC.hg19)
# filter peaks with 0 fragment, and only leave non-overlapping peaks
SE <- filterPeaks(SE, non_overlapping = TRUE)

# Fetch Mouse motif
mm_motifs <- getJasparMotifs(species = "Homo sapiens")
mm <- matchMotifs(mm_motifs, SE, BSgenome.Hsapiens.UCSC.hg19)

# From scAI, construct a region by motif matrix, 
motif_matrix <- motifMatches(mm)
data_m <- as.data.frame(as.matrix(motif_matrix))
write.table(data_m,file = "region2motif.csv", sep = ',', quote = FALSE)
# TFs <- colnames(data_m)
# TFs <-as.vector(TFs)
# out2 <- paste(baseName, "chromVAR_motif_names.txt", sep="/")
# write.table(TFs,file = out2,sep = '\t')
#motif_matrix_score <- motifScores(mm)  # a matrix with the score of the high motif score within each range/sequence (score only reported if match present)
#motif_matrix_count <- motifCounts(mm) # a matrix with the number of motif matches.


# two motif score matrices, deviation and deviation z-score, of the shape (motif/TFs, cells)
# motif name, of the form matrixID_TF, details see document: https://jaspar.genereg.net/docs/ 
dev <- computeDeviations(SE, mm)
dim(dev)
# obtain the motif deviation z-score
tfs <- data.matrix(assays(dev)[["z"]])
write.table(t(tfs), file = "motif_z.csv", sep = ",", quote = FALSE)

# bagged <- bagDeviations(dev, cor = 0.7, organism = "mouse")
# dim(bagged)
# tfs2 <- data.matrix(assays(bagged)[["z"]])
# rownames(tfs2) <- rowData(bagged)$name
# saveRDS(tfs2, file = "../output/TF_cell_matrix_bagged.rds")

