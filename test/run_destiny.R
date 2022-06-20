library(destiny)
library(reticulate)
np <- import("numpy")
path <- "/localscratch/ziqi/scDART_proj/scDART_test/test/results_snare/"
z_rna <- np$load(paste0(path, "models_1000/z_rna_4_1_1_10_0_l1.npy"))
z_atac = np$load(paste0(path, "models_1000/z_atac_4_1_1_10_0_l1.npy"))
z_scdart <- rbind(z_rna, z_atac)
scdart_dm <- DiffusionMap(data = z_scdart)
plot(scdart_dm)
z_dm <- scdart_dm@eigenvectors
np$save(paste0(path, "models_1000/z_diffmap.npy"), z_dm)


# liger
H1 <- read.csv(paste0(path, "liger/H1_full.csv"), row.names = 1)
H2 <- read.csv(paste0(path, "liger/H2_full.csv"), row.names = 1)
H <- rbind(H1, H2)
H["GAGTTTGGGTAG_2", 3] <- H["GAGTTTGGGTAG_2", 3] + 0.0001
LIGER_dm <- DiffusionMap(data = H)
plot(LIGER_dm)
z_dm <- LIGER_dm@eigenvectors
np$save(paste0(path, "liger/z_diffmap.npy"), z_dm)


# seurat
z_seurat <- read.csv(paste0(path, "seurat/pca_embedding.txt"), sep = "\t")
seurat_dm <- DiffusionMap(data = z_seurat)
plot(seurat_dm)
z_dm <- seurat_dm@eigenvectors
np$save(paste0(path, "seurat/z_diffmap.npy"), z_dm)

# UnionCom
z_rna <- np$load(paste0(path, "unioncom/unioncom_rna_32.npy"))
z_atac = np$load(paste0(path, "unioncom/unioncom_atac_32.npy"))
z_unioncom <- rbind(z_rna, z_atac)
unioncom_dm <- DiffusionMap(data = z_unioncom)
plot(unioncom_dm)
z_dm <- unioncom_dm@eigenvectors
np$save(paste0(path, "unioncom/z_diffmap.npy"), z_dm)

# scjoint
z_rna <- read.csv(paste0(path, "scJoint_snare_traj/counts_rna_embeddings.txt"), sep = " ", header = F)
z_atac <- read.csv(paste0(path, "scJoint_snare_traj/counts_atac_embeddings.txt"), sep = " ", header = F)
z_scjoint <- rbind(z_rna, z_atac)
scjoint_dm <- DiffusionMap(data = z_scjoint)
plot(scjoint_dm)
z_dm <- scjoint_dm@eigenvectors
np$save(paste0(path, "scJoint_snare_traj/z_diffmap.npy"), z_dm)
