# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python rpy2_3
#     language: python
#     name: rpy2_3
# ---

# %%
import scanpy as sc
import anndata as ann
import numpy as np 
import seaborn as sb
import pandas as pd
import pickle
from sklearn import preprocessing as pp
import diffxpy.api as de
import time
from scipy import sparse
import glob

import sys  
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/mm_pancreas_atlas_rep/code/')
from importlib import reload  
import helper
reload(helper)
import helper as h
from constants import SAVE
import expected_multiplet_rate as emr
reload(emr)
import expected_multiplet_rate as emr
#sc.settings.verbosity = 3

from matplotlib import rcParams
import matplotlib.pyplot as plt

#R interface
import rpy2.rinterface_lib.callbacks
import logging
from rpy2.robjects import pandas2ri
import anndata2ri

rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)
from rpy2.rinterface_lib import na_values as r_na

pandas2ri.activate()
anndata2ri.activate()
# %load_ext rpy2.ipython

# %%
#import neighbours 
#reload(neighbours )
#import neighbours as snn

# %% language="R"
# library(scran)
# library(biomaRt)
# library(BiocParallel)
# #library(Seurat)

# %%
# Path for saving results - last shared folder by all datasets
shared_folder='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/VSG_PF_WT_cohort/rev7/nonref/'
UID2='VSG_nonref_annotation'

# %% [markdown]
# Load data:

# %%
#Load data
#adata=pickle.load(  open( shared_folder+"data_normalised.pkl", "rb" ) )
adata=h.open_h5ad(shared_folder+"data_normalised.h5ad",unique_id2=UID2)

# %% [markdown]
# ## Visualisation

# %%
sc.pp.pca(adata, n_comps=30, use_highly_variable=True, svd_solver='arpack')
sc.pl.pca_variance_ratio(adata)

# %%
# Select number of PCs to use
N_PCS=15

# %% [markdown]
# Compare different embeddings based on previously defined annotation. 

# %%
#sc.pp.neighbors(adata,n_pcs = N_PCS,metric='correlation') 
#sc.tl.umap(adata)

# %%
#rcParams['figure.figsize']=(7,7)
#sc.pl.umap(adata,size=10,color=['pre_cell_type'])
#sc.pl.umap(adata,size=10,color=['file'])

# %%
sc.pp.neighbors(adata,n_pcs = N_PCS) 
sc.tl.umap(adata)

# %%
rcParams['figure.figsize']=(7,7)
#sc.pl.umap(adata,size=10,color=['pre_cell_type'])
sc.pl.umap(adata,size=10,color=['file'])

# %% [markdown]
# #### On scaled data

# %%
# Scale data and perform PCA
rcParams['figure.figsize']=(6,4)
adata_scl=adata.copy()
sc.pp.scale(adata_scl,max_value=10)
sc.pp.pca(adata_scl, n_comps=30, use_highly_variable=True, svd_solver='arpack')
sc.pl.pca_variance_ratio(adata_scl)

# %%
#C: Can stay the same as above
N_PCS

# %%
# neighbours on scaled data
sc.pp.neighbors(adata_scl,n_pcs = N_PCS) 

# %%
# Umap on scaled data
sc.tl.umap(adata_scl)

# %%
# Add scaled embedding to adata
adata.obsm['X_umap_scl']=adata_scl.obsm['X_umap']

# %%
rcParams['figure.figsize']=(7,7)
#sc.pl.umap(adata_scl,size=10,color=['pre_cell_type'])
sc.pl.umap(adata_scl,size=10,color=['file'])

# %% [markdown]
# ## Cell cycle
# Performed separately for individual batches.

# %% [markdown]
# ### Seurat/Scanpy - score by G2M and S

# %% [raw]
# # the below few cells needed to be executed only once
# # Extract human cell cycle genes 
# cell_cycle_hs=pd.read_csv('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/cell_cycle_hs_Macosko2015.csv',sep=';')
# cell_cycle_hs=cell_cycle_hs.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# %% [raw] magic_args="-i cell_cycle_hs -o cell_cycle_mm" language="R"
# # Map human to mouse cell cycle genes and order the dataframe
# # Mapping function taken from https://www.r-bloggers.com/converting-mouse-to-human-gene-names-with-biomart-package/
# genes_hs_mm <- function(human_genes){
# human = useMart("ensembl", dataset = "hsapiens_gene_ensembl")
# mouse = useMart("ensembl", dataset = "mmusculus_gene_ensembl")
#
# genesV2 = getLDS(attributes = c("hgnc_symbol"), filters = "hgnc_symbol", values = human_genes , mart = human, attributesL = c("mgi_symbol"), martL = mouse, uniqueRows=T)
# #print(genesV2)
# return(unique(genesV2[, 'MGI.symbol']))
# }
#
# # Map for each cell cycle phase
# genes<-c()
# phases<-c()
# for(col in colnames(cell_cycle_hs)){
#     human_genes<-unique(cell_cycle_hs[,col])
#     human_genes<-human_genes[!is.na(human_genes)]
#     mouse_genes <- genes_hs_mm(human_genes)
#     print(paste(col,'human:',length(human_genes),'mouse:',length(mouse_genes)))
#     genes<-c(genes,mouse_genes)
#     phases<-c(phases,rep(col,length(mouse_genes)))
# }
# cell_cycle_mm<-data.frame(Gene=genes,Phase=phases)

# %% [raw]
# # Save the mapped data
# cell_cycle_mm.to_csv('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/cell_cycle_mm_Macosko2015.tsv',sep='\t',index=False)

# %%
# Load mouse cell cycle genes
cell_cycle_mm=pd.read_table('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/cell_cycle_mm_Macosko2015.tsv',sep='\t')

# %% [markdown]
# Use cell cycle genes that overlap HVGs (from different batches). Display these genes on HVG plots (with non-phase genes being marked as .NA).
#
# Use G2/M and M gene sets for G2/M annotation and S gene set for S annotation.

# %%
# How many of the cell cycle phase genes are present in HVG and in var and how variable they are
hvg=set(adata.var_names[adata.var.highly_variable])
i=0
rcParams['figure.figsize']=(4,15)
fig,axs=plt.subplots(5)
s_hvg=[]
g2m_hvg=[]
for phase in cell_cycle_mm.Phase.unique():
    genes_phase = set(cell_cycle_mm.query('Phase =="'+phase+'"').Gene)
    overlap_var = set(adata.var_names) & genes_phase
    overlap_hvg = hvg & genes_phase
    print(phase,'N genes:',len(genes_phase),'overlap var:',len(overlap_var),'overlap hvg (all):',len(overlap_hvg))
    phase_df=pd.DataFrame([phase]*len(overlap_var),index=overlap_var,columns=['Phase']).reindex(adata.var_names).fillna('.NA').sort_values('Phase')
    phase_df.loc[overlap_hvg,'Phase']=phase+'_hvg'
    phase_df['mean']=adata.var.means
    phase_df['dispersions_norm']=adata.var.dispersions_norm
    sb.scatterplot(x="mean", y="dispersions_norm", hue="Phase",data=phase_df,ax=axs[i],palette='hls')
    i+=1
    if phase == 'S':
        s_hvg.extend(overlap_hvg)
    if phase in ['G2/M','M']:
        g2m_hvg.extend(overlap_hvg)
        
print('N genes for scoring S:',len(s_hvg),'and G2/M:',len(g2m_hvg))

# %% [markdown]
# Cell cycle annotation

# %%
# Annotated cell cycle per batch
adata.obs['S_score']= np.zeros(adata.shape[0])
adata.obs['G2M_score'] = np.zeros(adata.shape[0])
adata.obs['phase'] = np.zeros(adata.shape[0])

for batch in enumerate(adata.obs['file'].cat.categories):
    batch=batch[1]
    idx = adata.obs.query('file=="'+batch+'"').index
    adata_tmp = adata[idx,:].copy()
    sc.tl.score_genes_cell_cycle(adata_tmp, s_genes=s_hvg, g2m_genes=g2m_hvg,use_raw=False)
    adata.obs.loc[idx,'S_score'] = adata_tmp.obs['S_score']
    adata.obs.loc[idx,'G2M_score'] = adata_tmp.obs['G2M_score']
    adata.obs.loc[idx,'phase'] = adata_tmp.obs['phase']
    
del adata_tmp

# %%
# Count of cells annotated to each phase
adata.obs['phase'].value_counts()

# %% [markdown]
# Display cell cycle score distributions and annotation.

# %%
adata.uns['phase_colors']=['#ff7f0e', '#2ca02c','#46aaf0']

rcParams['figure.figsize']=(5,5)
sb.scatterplot(x='G2M_score',y='S_score',hue='phase',data=adata.obs)
sc.pl.umap(adata, color=['S_score', 'G2M_score'], size=10, use_raw=False)
sc.pl.umap(adata, color=['phase','Mki67'], size=10, use_raw=False)

# %% [markdown]
# #C: There might be some proliferating populations, but it is not very clear as many cells that probably are not proliferating (low Mki67 expression - not G2M) are annotated as cycling / G2M.

# %% [markdown]
# #### Check expression of S and G2M markers

# %%
set(s_hvg)&set(g2m_hvg)

# %%
print('S and G2M marker set overlap %i out of %i S and %i G2M HVGs'%
      (len(set(s_hvg)&set(g2m_hvg)), len(s_hvg), len(g2m_hvg)))

# %%
print('S HVG')
sc.pl.umap(adata, color=s_hvg, size=10, use_raw=False)
print('G2M HVG')
sc.pl.umap(adata, color=g2m_hvg, size=10, use_raw=False)

# %% [markdown]
# #C: The expression of G2M and S markers is not very consistent. Thus annotation that uses these markers is probably not reliable (-> use cyclone?).

# %% [markdown]
#  ### Cyclone - based on G1, S, and G2/M scores

# %% [markdown]
# Add gene Entrez IDs to adata in order to map genes to cell cycle database.

# %%
# Adata genes for R
genes=adata.var_names

# %% [markdown]
# #C: Current Ensembl release 101

# %% magic_args="-i genes -o gene_ids" language="R"
# # Extract Ensembl gene IDs
# mouse = useMart("ENSEMBL_MART_ENSEMBL", dataset = "mmusculus_gene_ensembl",version='Ensembl Genes 101',
#                 host="https://www.ensembl.org")
# gene_ids = getBM(attributes = c("mgi_symbol",'ensembl_gene_id'), 
#                  filters = "mgi_symbol", values = genes , mart = mouse, uniqueRows=FALSE)

# %%
# Add gene ids to adata, use only genes with unique mapped ensembl ids
gene_ids.drop_duplicates(subset='mgi_symbol', keep=False, inplace=True)
gene_ids.index=gene_ids.mgi_symbol
gene_ids=gene_ids.reindex(list(adata.var_names))
adata.var['EID']=gene_ids.ensembl_gene_id

# %%
# Prepare R data for cyclonee
x_mat=adata.X.T
gene_ids=adata.var.EID
batches=adata.obs.file
cells=adata.obs.index

# %% magic_args="-i x_mat -i gene_ids -i batches -i cells -o cyclone_anno" language="R"
# # Cyclone cell scores, calculated separately for each batch
# mm.pairs <- readRDS(system.file("exdata", "mouse_cycle_markers.rds", package="scran"))
# phases<-c()
# s<-c()
# g2m<-c()
# g1<-c()
# cells_order<-c()
# for(batch in unique(batches)){
#     # Select batch data
#     x_mat_batch=x_mat[,batches==batch]
#     print(batch,dim(x_mat_batch[1]))
#     # Scores
#     assignments <- cyclone(x_mat_batch, mm.pairs, gene.names=gene_ids,BPPARAM=MulticoreParam(workers = 16))
#     phases<-c(phases,assignments$phases)
#     s<-c(s,assignments$score$S)
#     g2m<-c(g2m,assignments$score$G2M)
#     g1<-c(g1,assignments$score$G1)
#     # Save cell order
#     cells_order<-c(cells_order,cells[batches==batch])
# }
# cyclone_anno<-data.frame(phase_cyclone=phases,s_cyclone=s,g2m_cyclone=g2m,g1_cyclone=g1)
# rownames(cyclone_anno)<-cells_order

# %%
# Count of cells annotated to each phase
cyclone_anno.phase_cyclone.value_counts()

# %%
# Add cyclone annotation to adata
cyclone_anno=cyclone_anno.reindex(adata.obs.index)
adata.obs=pd.concat([adata.obs,cyclone_anno],axis=1)

# %%
# replace NA annotation of cell phases
adata.obs.phase_cyclone.replace(r_na.NA_Character,'NA',inplace=True)
adata.obs.g1_cyclone.replace(r_na.NA_Character,'NA',inplace=True)
adata.obs.g2m_cyclone.replace(r_na.NA_Character,'NA',inplace=True)
adata.obs.s_cyclone.replace(r_na.NA_Character,'NA',inplace=True)

# %%
# Plot score distributions and cell assignment on UMAP
rcParams['figure.figsize']=(20,5)
fig,axs=plt.subplots(1,3)
palette=sb.color_palette(['#ff7f0e', '#2ca02c','#46aaf0','#484848'])
sb.scatterplot(x='g2m_cyclone',y='s_cyclone',hue='phase_cyclone',data=cyclone_anno,ax=axs[0],palette=palette)
sb.scatterplot(x='g1_cyclone',y='s_cyclone',hue='phase_cyclone',data=cyclone_anno,ax=axs[1],palette=palette)
sb.scatterplot(x='g2m_cyclone',y='g1_cyclone',hue='phase_cyclone',data=cyclone_anno,ax=axs[2],palette=palette)
rcParams['figure.figsize']=(5,5)
sc.pl.umap(adata, color=['s_cyclone', 'g2m_cyclone','g1_cyclone'], size=10, use_raw=False)
adata.uns['phase_cyclone_colors']=['#ff7f0e', '#2ca02c','#46aaf0','#484848']
sc.pl.umap(adata, color=['phase_cyclone'], size=10, use_raw=False)

# %% [markdown]
# #C: The cyclone may be leaving out proliferative cells.

# %% [markdown]
# #### Try on raw data without removed genes as suggested in manual.

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/VSG_PF_WT_cohort/rev7/scanpy_AnnData/MUC136*/'
metadata=pd.read_excel('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/scRNA-seq_pancreas_metadata.xlsx',
             sheet_name='VSG')
samples=metadata.query('for_reference == False & tissue == "islet"')

# List all files
file_name='filtered_feature_bc_matrix.h5ad'
files=glob.glob(path_data+file_name)
# Find which parts of file paths differ between files to later use them as file id
diff_path_idx=[]
for position in range(len(path_data.split('/'))):
    values=set([file.split('/')[position] for file in files])
    if len(values)>1:
        diff_path_idx.append(position)
# Subset to files used for reference        
files_subset=[]
for file in files:
    sample='_'.join([file.split('/')[i] for i in diff_path_idx])
    if any(sample_name in sample for sample_name in samples.sample_name.values):
        files_subset.append(file)
        
# Load files and extract parts of file path that identifies the file, compared to other loaded files
adatas=[]
file_diffs=[]
for file in files_subset:
    print('Reading file',file)
    #adatas.append(sc.read_h5ad(file))
    adatas.append(h.open_h5ad(file=file,unique_id2=UID2))
    file_diffs.append('_'.join([file.split('/')[i] for i in diff_path_idx]))
    
adata_raw = ann.AnnData.concatenate( *adatas,  batch_key = 'file', batch_categories = file_diffs).copy()    

# %%
# Subest to retained cells
adata_raw=adata_raw[adata.obs_names,:]

# %%
# Adata genes for R
genes=adata_raw.var_names

# %% [markdown]
# #C: Current Ensembl release 101

# %% magic_args="-i genes -o gene_ids" language="R"
# # Extract Ensembl gene IDs
# mouse = useMart("ENSEMBL_MART_ENSEMBL", dataset = "mmusculus_gene_ensembl",version='Ensembl Genes 101',
#                 host="https://www.ensembl.org")
# gene_ids = getBM(attributes = c("mgi_symbol",'ensembl_gene_id'), 
#                  filters = "mgi_symbol", values = genes , mart = mouse, uniqueRows=FALSE)

# %%
# Add gene ids to adata, use only genes with unique mapped ensembl ids
gene_ids.drop_duplicates(subset='mgi_symbol', keep=False, inplace=True)
gene_ids.index=gene_ids.mgi_symbol
gene_ids=gene_ids.reindex(list(adata_raw.var_names))
adata_raw.var['EID']=gene_ids.ensembl_gene_id

# %%
# Prepare R data for cyclonee
x_mat=adata_raw.X.T
gene_ids=adata_raw.var.EID
batches=adata_raw.obs.file
cells=adata_raw.obs.index

# %% magic_args="-i x_mat -i gene_ids -i batches -i cells -o cyclone_anno" language="R"
# # Cyclone cell scores, calculated separately for each batch
# mm.pairs <- readRDS(system.file("exdata", "mouse_cycle_markers.rds", package="scran"))
# phases<-c()
# s<-c()
# g2m<-c()
# g1<-c()
# cells_order<-c()
# for(batch in unique(batches)){
#     # Select batch data
#     x_mat_batch=x_mat[,batches==batch]
#     print(batch,dim(x_mat_batch[1]))
#     # Scores
#     assignments <- cyclone(x_mat_batch, mm.pairs, gene.names=gene_ids,BPPARAM=MulticoreParam(workers = 16))
#     phases<-c(phases,assignments$phases)
#     s<-c(s,assignments$score$S)
#     g2m<-c(g2m,assignments$score$G2M)
#     g1<-c(g1,assignments$score$G1)
#     # Save cell order
#     cells_order<-c(cells_order,cells[batches==batch])
# }
# cyclone_anno<-data.frame(phase_cyclone=phases,s_cyclone=s,g2m_cyclone=g2m,g1_cyclone=g1)
# rownames(cyclone_anno)<-cells_order

# %%
# Count of cells annotated to each phase
cyclone_anno.phase_cyclone.value_counts()

# %%
# Add cyclone annotation to adata
cyclone_anno=cyclone_anno.reindex(adata_raw.obs.index)
adata_raw.obs=pd.concat([adata_raw.obs,cyclone_anno],axis=1)

# %%
# replace NA annotation of cell phases
adata_raw.obs.phase_cyclone.replace(r_na.NA_Character,'NA',inplace=True)
adata_raw.obs.g1_cyclone.replace(r_na.NA_Character,'NA',inplace=True)
adata_raw.obs.g2m_cyclone.replace(r_na.NA_Character,'NA',inplace=True)
adata_raw.obs.s_cyclone.replace(r_na.NA_Character,'NA',inplace=True)

# %%
adata_raw.obsm['X_umap']=adata.obsm['X_umap'].copy()

# %%
# Plot score distributions and cell assignment on UMAP
rcParams['figure.figsize']=(20,5)
fig,axs=plt.subplots(1,3)
palette=sb.color_palette(['#ff7f0e', '#2ca02c','#46aaf0','#484848'])
sb.scatterplot(x='g2m_cyclone',y='s_cyclone',hue='phase_cyclone',data=cyclone_anno,ax=axs[0],palette=palette)
sb.scatterplot(x='g1_cyclone',y='s_cyclone',hue='phase_cyclone',data=cyclone_anno,ax=axs[1],palette=palette)
sb.scatterplot(x='g2m_cyclone',y='g1_cyclone',hue='phase_cyclone',data=cyclone_anno,ax=axs[2],palette=palette)
rcParams['figure.figsize']=(5,5)
sc.pl.umap(adata_raw, color=['s_cyclone', 'g2m_cyclone','g1_cyclone'], size=10, use_raw=False)
adata_raw.uns['phase_cyclone_colors']=['#ff7f0e', '#2ca02c','#46aaf0','#484848']
sc.pl.umap(adata_raw, color=['phase_cyclone'], size=10, use_raw=False)

# %% [markdown]
# #C: Using raw data (that has all genes (and is not normalised)) does not improve (or importantly change) the cell type annotation results.

# %% [markdown]
# #C: Decide to use cyclone (more conservative) as gene set scoring annotation is annotationg too many cells as cycling. 

# %% [markdown]
# ## Save intermediate results before cell type annotation

# %%
if SAVE:
    h.save_h5ad(adata, shared_folder+"data_annotated.h5ad",unique_id2=UID2)
