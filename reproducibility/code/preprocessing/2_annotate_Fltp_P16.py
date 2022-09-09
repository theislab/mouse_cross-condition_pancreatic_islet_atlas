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
pandas2ri.activate()
anndata2ri.activate()
# %load_ext rpy2.ipython

# %% language="R"
# library(scran)
# library(biomaRt)
# library(BiocParallel)
# #library(Seurat)

# %%
# Path for saving results - last shared folder by all datasets
shared_folder='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/salinno_project/rev4/'
UID2='Fltp16_annotation'

# %% [markdown]
# Load data:

# %%
#Load data
#adata=pickle.load(  open( shared_folder+"data_normalised.pkl", "rb" ) )
adata=h.open_h5ad(shared_folder+"data_normalised.h5ad",unique_id2=UID2)

# %% [markdown]
# ## Add previously generated annotation

# %% [markdown]
# At the time of this analysis the previously generated annotation was still internal (unpublished) and may thus differ from the finnaly published annotation. Howevre, this previous annotatiuon was used only for internal validation and non of the analyses depend on it.

# %%
# Add previously generated annotation (Maren Buttner)
#adata_preannotated=h.open_h5ad("/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/salinno_project/rev4/maren/data_endo_final.h5ad",unique_id2=UID2)
adata_preannotated=h.open_h5ad("/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/salinno_project/rev4/maren/data_annotated.h5ad",unique_id2=UID2)

# %%
adata_preannotated

# %%
# Match the cell names between preanotated and current dataset
cell_name_map={'-mGFP':'-145_mGFP','-mRFP':'-146_mRFP','-mTmG':'-147_mTmG'}
adata_preannotated.obs.index=[h.replace_all(cell,cell_name_map) for cell in adata_preannotated.obs.index]

# %%
# Add pre-prepared cell type annotation to currently used dataset
adata.obs['pre_cell_type']=adata_preannotated.obs.reindex(adata.obs.index)['cell_type']
adata.obs['pre_cell_type'] = adata.obs['pre_cell_type'].cat.add_categories('NA')
adata.obs['pre_cell_type'].fillna('NA', inplace =True) 

# %% [markdown]
# Pre-annotated cell counts

# %%
# Count of cells per annotation
adata.obs.groupby('pre_cell_type').size()

# %% [markdown]
# ## Visualisation

# %%
sc.pp.pca(adata, n_comps=30, use_highly_variable=True, svd_solver='arpack')
sc.pl.pca_variance_ratio(adata)

# %%
# Select number of PCs to use
N_PCS=16

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
sc.pl.umap(adata,size=10,color=['pre_cell_type'])
sc.pl.umap(adata,size=10,color=['file'])

# %% [markdown]
# #C: Decided that Euclidean distance (default) will be ok.

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
#  ### Cyclone - based on G1, S, and G2/M scores

# %% [markdown]
# Add gene Entrez IDs to adata in order to map genes to cell cycle database.

# %%
# Adata genes for R
genes=adata.var_names

# %% magic_args="-i genes -o gene_ids" language="R"
# # Extract Ensembl gene IDs
# mouse = useMart("ensembl", dataset = "mmusculus_gene_ensembl")
# gene_ids = getBM(attributes = c("mgi_symbol",'ensembl_gene_id'), filters = "mgi_symbol", values = genes , mart = mouse, uniqueRows=FALSE)

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
# Plot score distributions and cell assignment on UMAP
rcParams['figure.figsize']=(20,5)
fig,axs=plt.subplots(1,3)
palette=sb.color_palette(['#ff7f0e', '#2ca02c','#46aaf0'])
sb.scatterplot(x='g2m_cyclone',y='s_cyclone',hue='phase_cyclone',data=cyclone_anno,ax=axs[0],palette=palette)
sb.scatterplot(x='g1_cyclone',y='s_cyclone',hue='phase_cyclone',data=cyclone_anno,ax=axs[1],palette=palette)
sb.scatterplot(x='g2m_cyclone',y='g1_cyclone',hue='phase_cyclone',data=cyclone_anno,ax=axs[2],palette=palette)
rcParams['figure.figsize']=(5,5)
sc.pl.umap(adata, color=['s_cyclone', 'g2m_cyclone','g1_cyclone'], size=10, use_raw=False)
adata.uns['phase_cyclone_colors']=['#ff7f0e', '#2ca02c','#46aaf0']
sc.pl.umap(adata, color=['phase_cyclone'], size=10, use_raw=False)

# %% [markdown]
# #C: The cyclone results seem more reliable based on previous cell type annotation. 

# %% [markdown]
# ## Sex scores

# %% [markdown]
# Extract expressed genes from X and Y chromosome and score their expression, separately for each batch.

# %% [markdown]
# Add chromosome information:

# %%
# Gene names for R for chromosome assignment
genes=adata.var_names

# %% magic_args="-i genes -o genes_chr" language="R"
# # Chromosome assignment 
#
# mouse = useMart("ensembl", dataset = "mmusculus_gene_ensembl")
# # Return duplicated results - e.g. one gene being annotated to multiple chromosomes
# genes_chr = getBM(attributes = c("mgi_symbol",'chromosome_name'), filters = "mgi_symbol", values = genes , mart = mouse, uniqueRows=F)
#

# %%
# Remove genes annotated to non-standard chromosomes and 
# confirm that each gene is now annotated to only one chromosome
mouse_chromosomes=[str(i) for i in range(1,20)]+['X','Y','MT']
genes_chr_filtered=genes_chr.query('chromosome_name in @mouse_chromosomes')
#Remove duplicated rows (gene was returned multiple times with the same chromosome)
genes_chr_filtered=genes_chr_filtered.drop_duplicates()
print('Each gene has one annotated chromosome:',
      genes_chr_filtered.mgi_symbol.unique().shape[0]==genes_chr_filtered.shape[0])
print('N genes with chromosomes:',genes_chr_filtered.shape[0],'out of',adata.shape[1],'genes')

# %%
# Add chromosomes in adata
genes_chr_filtered.index=genes_chr_filtered.mgi_symbol
genes_chr_filtered=genes_chr_filtered.reindex(adata.var_names)
adata.var['chromosome']=genes_chr_filtered['chromosome_name']

# %% [markdown]
# Extract X and Y HVGs and compare their number to all expressed X and Y genes.

# %%
# Select X and Y genes that are HVG
x_hvg = adata.var.query('chromosome=="X" & highly_variable').index
y_hvg = adata.var.query('chromosome=="Y" & highly_variable').index
print('X HVG:',len(x_hvg),'/',adata.var.query('chromosome=="X"').shape[0],
      'Y HVG:',len(y_hvg),'/',adata.var.query('chromosome=="Y"').shape[0])

# %% [markdown]
# Show Y gene expression on UMAP.

# %%
# Show Y HVGs and Y non-HVGs
rcParams['figure.figsize']=(3,3)
print('Y specific HVG')
sc.pl.umap(adata, color=y_hvg, size=10, use_raw=False)
print('Y specific non-HVG')
sc.pl.umap(adata, color=adata.var.query('chromosome=="Y"& ~highly_variable').index, size=10, use_raw=False)

# %% [markdown]
# #C: Due to low number of expressed Y genes both HVG and non-HVG genes will be used. Genes Gm47283 (unsepcific??) and Gm29650 (low expression in most cells) will not be used for Y scoring. 
# For X chromosome only HVG will be used. 

# %% [markdown]
# Score cells:

# %%
# Prepare final scoring gene sets for X and Y
y_all=adata.var.query('chromosome=="Y"').index
y_selected=[gene for gene in y_all if gene not in ['Gm47283','Gm29650']]
x_selected=list(x_hvg)
print('N selected genes from Y:',len(y_selected),'and from X:',len(x_selected))

# %%
#Do not use the smaller gene set as controll size as Y gene set is very small
#ctrl_size = min(len(x_genes), len(y_genes))

# For performing it on all batches together
#sc.tl.score_genes(adata, x_selected, score_name = 'x_score', use_raw=False)
#sc.tl.score_genes(adata, y_selected, score_name = 'y_score',use_raw=False)

# Compute sex score per batch
adata.obs['x_score']= np.zeros(adata.shape[0])
adata.obs['y_score'] = np.zeros(adata.shape[0])

for batch in enumerate(adata.obs['file'].cat.categories):
    batch=batch[1]
    idx = adata.obs.query('file=="'+batch+'"').index
    adata_tmp = adata[idx,:].copy()
    sc.tl.score_genes(adata_tmp, x_selected, score_name = 'x_score', use_raw=False)
    sc.tl.score_genes(adata_tmp, y_selected, score_name = 'y_score',use_raw=False)
    adata.obs.loc[idx,'y_score'] = adata_tmp.obs['y_score']
    adata.obs.loc[idx,'x_score'] = adata_tmp.obs['x_score']
    
del adata_tmp

# %% [markdown]
# Plot X and Y scores

# %%
rcParams['figure.figsize']=(10,3)
fig,axs=plt.subplots(1,2)
# Scores across batches
sc.pl.violin(adata, ['x_score'], groupby='file', size=1, log=False,rotation=90,ax=axs[0],show=False)
sc.pl.violin(adata, ['y_score'], groupby='file', size=1, log=False,rotation=90,ax=axs[1],show=False)
plt.show()
# X vs Y score
rcParams['figure.figsize']=(3,3)
sb.scatterplot(x='x_score',y='y_score',data=adata.obs)
# Distribution of scores
sc.pl.umap(adata, color=['x_score','y_score'] ,size=10, use_raw=False)
rcParams['figure.figsize']=(10,3)
fig,axs=plt.subplots(1,2)
sb.distplot(adata.obs['x_score'],  kde=False,  bins=100,ax=axs[0])
sb.distplot(adata.obs['y_score'],  kde=False,  bins=100,ax=axs[1])

# %% [markdown]
# #C: It seems that Y score alone performs much better - X score depends more on the cell type. Thus only Y score will be used for cell type annotation.

# %%
# More detailed plot of y_score region of interest
rcParams['figure.figsize']=(7,3)
sb.distplot(adata.obs.query('y_score>-0.07 & y_score<0.05')['y_score'],  kde=False,  bins=50)

# %% [markdown]
# #C: Set male threshold at >= - 0.02 (Score 0 means equal expression as a reference data set.)

# %%
adata.obs['sex']=(adata.obs['y_score']>=-0.02).replace(False,'female').replace(True,'male')

rcParams['figure.figsize']=(4,4)
sc.pl.umap(adata, color=['sex'] ,size=10, use_raw=False)

# %% [markdown]
# ## Save intermediate results before cell type annotation

# %%
if SAVE:
    h.save_h5ad(adata, shared_folder+"data_annotated.h5ad",unique_id2=UID2)

# %% [markdown]
# # Cell type annotation

# %%
adata=h.open_h5ad(shared_folder+"data_annotated.h5ad",unique_id2=UID2)

# %% [markdown]
# ## Endo high annotation

# %%
# Normalise raw data for cell type scoring
adata_rawnorm=adata.raw.to_adata().copy()
adata_rawnorm.X /= adata.obs['size_factors'].values[:,None] # This reshapes the size-factors array
sc.pp.log1p(adata_rawnorm)
adata_rawnorm.X = np.asarray(adata_rawnorm.X)
adata_rawnorm.obs=adata.obs.copy()

# %% [markdown]
# #### Ins

# %%
genes=['Ins1','Ins2']
score_name='ins'

# %%
# Compute score
sc.tl.score_genes(adata_rawnorm, gene_list=genes, score_name=score_name+'_score',  use_raw=False)

# %%
scores=adata_rawnorm.obs[[score_name+'_score','file']]
scores[score_name+'_score_norm']=pp.minmax_scale(scores[score_name+'_score'])
rcParams['figure.figsize']=(12,10)
fig,axs=plt.subplots(2,1)
sb.violinplot(x='file',y=score_name+'_score',data=scores,inner=None,ax=axs[0])
axs[0].title.set_text(score_name+' scores')
axs[0].grid()
sb.violinplot(x='file',y=score_name+'_score_norm',data=scores,inner=None,ax=axs[1])
axs[1].title.set_text('Scaled '+score_name+' scores')
axs[1].grid()
#sb.violinplot(x='file',y=score_name+'_score_norm',data=scores[scores[score_name+'_score_norm']>0.15],
#             inner=None,ax=axs[2])
#axs[2].title.set_text('Scaled '+score_name+' scores without very low '+score_name+' cells')
#axs[2].grid()

# %%
# Find score high cells
# Use different thresholds across files as there might be different ambient counts (or other effects), 
# thus shifting distributions up a bit
file_thresholds={'145_mGFP':0.6,'146_mRFP':0.65,'147_mTmG':0.6}
adata_rawnorm.obs[score_name+'_high']=scores.apply(lambda x: x[score_name+'_score_norm'] > file_thresholds[x.file], 
                                                   axis=1)
print('Proportion of '+score_name+' high across samples:')
adata_rawnorm.obs[['file',score_name+'_high']].groupby('file')[score_name+'_high'].value_counts(
    normalize=True,sort=False)

# %%
# Add info about score high to main adata and save it
adata.obs[score_name+'_score']=adata_rawnorm.obs[score_name+'_score']
adata.obs[score_name+'_high']=adata_rawnorm.obs[score_name+'_high']

# %% [markdown]
# #### Gcg

# %%
genes=['Gcg']
score_name='gcg'

# %%
# Compute score
sc.tl.score_genes(adata_rawnorm, gene_list=genes, score_name=score_name+'_score',  use_raw=False)

# %%
scores=adata_rawnorm.obs[[score_name+'_score','file']]
scores[score_name+'_score_norm']=pp.minmax_scale(scores[score_name+'_score'])
rcParams['figure.figsize']=(12,10)
fig,axs=plt.subplots(2,1)
sb.violinplot(x='file',y=score_name+'_score',data=scores,inner=None,ax=axs[0])
axs[0].title.set_text(score_name+' scores')
axs[0].grid()
sb.violinplot(x='file',y=score_name+'_score_norm',data=scores,inner=None,ax=axs[1])
axs[1].title.set_text('Scaled '+score_name+' scores')
axs[1].grid()
#sb.violinplot(x='file',y=score_name+'_score_norm',data=scores[scores[score_name+'_score_norm']>0.15],
#             inner=None,ax=axs[2])
#axs[2].title.set_text('Scaled '+score_name+' scores without very low '+score_name+' cells')
#axs[2].grid()

# %%
# Find score high cells
# Use different thresholds across files as there might be different ambient counts (or other effects), 
# thus shifting distributions up a bit
file_thresholds={'145_mGFP':0.6,'146_mRFP':0.65,'147_mTmG':0.6}
adata_rawnorm.obs[score_name+'_high']=scores.apply(lambda x: x[score_name+'_score_norm'] > file_thresholds[x.file], 
                                                   axis=1)
print('Proportion of '+score_name+' high across samples:')
adata_rawnorm.obs[['file',score_name+'_high']].groupby('file')[score_name+'_high'].value_counts(
    normalize=True,sort=False)

# %%
# Add info about score high to main adata and save it
adata.obs[score_name+'_score']=adata_rawnorm.obs[score_name+'_score']
adata.obs[score_name+'_high']=adata_rawnorm.obs[score_name+'_high']

# %% [markdown]
# #### Sst

# %%
genes=['Sst']
score_name='sst'

# %%
# Compute score
sc.tl.score_genes(adata_rawnorm, gene_list=genes, score_name=score_name+'_score',  use_raw=False)

# %%
scores=adata_rawnorm.obs[[score_name+'_score','file']]
scores[score_name+'_score_norm']=pp.minmax_scale(scores[score_name+'_score'])
rcParams['figure.figsize']=(12,10)
fig,axs=plt.subplots(2,1)
sb.violinplot(x='file',y=score_name+'_score',data=scores,inner=None,ax=axs[0])
axs[0].title.set_text(score_name+' scores')
axs[0].grid()
sb.violinplot(x='file',y=score_name+'_score_norm',data=scores,inner=None,ax=axs[1])
axs[1].title.set_text('Scaled '+score_name+' scores')
axs[1].grid()
#sb.violinplot(x='file',y=score_name+'_score_norm',data=scores[scores[score_name+'_score_norm']>0.15],
#             inner=None,ax=axs[2])
#axs[2].title.set_text('Scaled '+score_name+' scores without very low '+score_name+' cells')
#axs[2].grid()

# %%
# Find score high cells
# Use different thresholds across files as there might be different ambient counts (or other effects), 
# thus shifting distributions up a bit
file_thresholds={'145_mGFP':0.6,'146_mRFP':0.6,'147_mTmG':0.5}
adata_rawnorm.obs[score_name+'_high']=scores.apply(lambda x: x[score_name+'_score_norm'] > file_thresholds[x.file], 
                                                   axis=1)
print('Proportion of '+score_name+' high across samples:')
adata_rawnorm.obs[['file',score_name+'_high']].groupby('file')[score_name+'_high'].value_counts(
    normalize=True,sort=False)

# %%
# Add info about score high to main adata and save it
adata.obs[score_name+'_score']=adata_rawnorm.obs[score_name+'_score']
adata.obs[score_name+'_high']=adata_rawnorm.obs[score_name+'_high']

# %% [markdown]
# #### Ppy

# %%
genes=['Ppy']
score_name='ppy'

# %%
# Compute score
sc.tl.score_genes(adata_rawnorm, gene_list=genes, score_name=score_name+'_score',  use_raw=False)

# %%
scores=adata_rawnorm.obs[[score_name+'_score','file']]
scores[score_name+'_score_norm']=pp.minmax_scale(scores[score_name+'_score'])
rcParams['figure.figsize']=(12,10)
fig,axs=plt.subplots(2,1)
sb.violinplot(x='file',y=score_name+'_score',data=scores,inner=None,ax=axs[0])
axs[0].title.set_text(score_name+' scores')
axs[0].grid()
sb.violinplot(x='file',y=score_name+'_score_norm',data=scores,inner=None,ax=axs[1])
axs[1].title.set_text('Scaled '+score_name+' scores')
axs[1].grid()
#sb.violinplot(x='file',y=score_name+'_score_norm',data=scores[scores[score_name+'_score_norm']>0.15],
#             inner=None,ax=axs[2])
#axs[2].title.set_text('Scaled '+score_name+' scores without very low '+score_name+' cells')
#axs[2].grid()

# %%
# Find score high cells
# Use different thresholds across files as there might be different ambient counts (or other effects), 
# thus shifting distributions up a bit
file_thresholds={'145_mGFP':0.8,'146_mRFP':0.8,'147_mTmG':0.75}
adata_rawnorm.obs[score_name+'_high']=scores.apply(lambda x: x[score_name+'_score_norm'] > file_thresholds[x.file], 
                                                   axis=1)
print('Proportion of '+score_name+' high across samples:')
adata_rawnorm.obs[['file',score_name+'_high']].groupby('file')[score_name+'_high'].value_counts(
    normalize=True,sort=False)

# %%
# Add info about score high to main adata and save it
adata.obs[score_name+'_score']=adata_rawnorm.obs[score_name+'_score']
adata.obs[score_name+'_high']=adata_rawnorm.obs[score_name+'_high']

# %% [markdown]
# #### Save

# %%
h.save_h5ad(adata, shared_folder+"data_annotated.h5ad",unique_id2=UID2)

# %% [markdown]
# ## Clustering

# %% [markdown]
# #### Leiden clustering on log transformed data.

# %%
sc.tl.leiden(adata,resolution=1.5)

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata, color=['leiden'] ,size=10, use_raw=False)

# %% [markdown]
# #### Seurat style clustering

# %%
# Preapre data for Seurat clustering
expression=pd.DataFrame(adata.X.T,index=adata.var_names,columns=adata.obs.index)
hvg=adata.var_names[adata.var.highly_variable]

# %% magic_args="-i expression -i hvg -i N_PCS -i shared_folder" language="R"
# # Seurat clustering data preparatrion - so that seurat object required for clustering can be computed once and then reused
# seurat_obj<-CreateSeuratObject(counts=expression)
# seurat_obj <- ScaleData(seurat_obj, features = hvg)
# seurat_obj <- RunPCA(seurat_obj, features = hvg,npcs=N_PCS)
# seurat_obj <- FindNeighbors(seurat_obj, dims = 1:N_PCS)
# saveRDS(seurat_obj, file = paste0(shared_folder,"data_clustering_seurat_annotemp.rds"))

# %%
res=0.3

# %% magic_args="-i res -i shared_folder -o clusters " language="R"
# #Seurat clustering
# seurat_obj <- readRDS( file = paste0(shared_folder,"data_clustering_seurat_annotemp.rds"))
# seurat_obj <- FindClusters(seurat_obj, resolution = res)
# clusters<-data.frame(unlist(Idents(seurat_obj )))
# rownames(clusters)<-names(Idents(seurat_obj ))
# colnames(clusters)<-c('cluster_seurat')

# %%
# Add Seurat clusters
clusters=clusters.reindex(adata.obs.index)
adata.obs['cluster_seurat_r'+str(res)]=clusters.cluster_seurat

# %%
# Plot Seurat clustering
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata, color=['cluster_seurat_r0.3'] ,size=10, use_raw=False)

# %% [markdown]
# #### Leiden clustering on log transformed z-scaled data.

# %%
# Scale data and perform PCA
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
# This neighbour weight computation (used in  Seurat) does not improve clustering - the main improvement is due to scaling
#snn.shared_knn_adata(adata,n_pcs=N_PCS,n_jobs=8)

# %%
# Umap on scaled data
sc.tl.umap(adata_scl)

# %%
# Add scaled embedding to adata
adata.obsm['X_umap_scl']=adata_scl.obsm['X_umap']

# %%
del adata_scl.uns['sex_colors']
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata_scl, color=['sex','file'] ,size=10, use_raw=False,wspace=0.2)

# %%
# Clustering resolution
res=0.4

# %%
# Cluster scaled data
sc.tl.leiden(adata_scl, resolution=res, key_added='leiden_scaled_r'+str(res), directed=True, use_weights=True)

# %%
# Compare UMAPs and clusters on scaled data
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata_scl, color=['leiden_scaled_r'+str(res)] ,size=10, use_raw=False)
sc.pl.umap(adata_scl, color=['pre_cell_type'] ,size=10, use_raw=False)

# %% [markdown]
# #C: Umap and clustering on scaled data performs better.

# %% [markdown]
# ## Cell type annotation

# %%
# Normalise raw data for plotting and cell type scoring
adata_raw=adata_scl.raw.to_adata().copy()
sc.pp.normalize_total(adata_raw, target_sum=1e4, exclude_highly_expressed=True,inplace=False)
sc.pp.log1p(adata_raw)
adata_rawnormalised=adata_scl.copy()
adata_rawnormalised.raw=adata_raw
del adata_raw

# %%
# Markers data
markers=pd.read_excel('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/markers.xlsx',
                          sheet_name='mice')

# %%
# Subset markers to non-immature and non-dedifferentiated for initial annotation
print('Subtypes (original):',markers.Subtype.unique())
remove_subtype=['immature','dedifferentiated']
remove_subtype=[]
markers_filter=markers.query('Subtype not in @remove_subtype')

# %%
# Plot markers for each cell type
for cell_type in sorted(list(markers_filter.Cell.unique()), key=str.lower):
    print(cell_type)
    genes=list(markers_filter.query('Cell == "'+cell_type+'"').Gene)
    # Retain genes present in raw var_names
    genes=[gene for gene in genes if gene in adata.raw.var_names]
    rcParams['figure.figsize']=(4,4)
    sc.pl.umap(adata_rawnormalised, color=['pre_cell_type'],size=5, use_raw=True)
    sc.pl.umap(adata_rawnormalised, color=genes ,size=5, use_raw=True)

# %%
# Selected markers that seem to be able to distinguish between cell types
# Use only hormone genes for endocrine cells
markers_selection={
'immune':['Cd74','Ptprc','Cd86','Adgre1','Lyz2','Cd86','Itgax','Cd52'],
'schwann':['Sox10','Cmtm5','Cryab','Ngfr'],
'stellate':['Col1a2','Pdgfra'],
'endothelial':['Plvap','Pecam1'],
'pericyte':['Ndufa4l2','Pdgfrb','Acta2','Cspg4','Des','Rgs5','Abcc9'],
'acinar':['Cpa1','Prss2'],
'ductal':['Muc1','Sox9','Spp1','Krt19'],
'alpha':['Gcg',],
'beta':['Ins1','Ins2'],
'delta':['Sst'],
'gamma':['Ppy'],
'epsilon':['Ghrl']
}

#markers_selection={
#'immune':['Cd74','Ptprc','Cd86','Adgre1','Lyz2','Cd86','Itgax','Cd52'],
#'schwann':['Sox10'],
#'stellate':['Col1a2','Pdgfra'],
#'endothelial':['Plvap','Pecam1'],
#'ductal':['Muc1','Sox9','Spp1','Krt19'],
#'acinar':['Cpa1','Prss2'],
#'beta':['Slc2a2','Ins1','Ins2'],
#'alpha':['Gcg','Irx1','Irx2'],
#'delta':['Sst','Hhex','Neurog3'],
#'gamma':['Ppy'],
#'epsilon':['Ghrl']
#}

# %% [markdown]
# ### Marker expression scores
# Score each cell for marker expression of each cell type. Cells can thus be annotated qith 0 to N cell types. The presence of annotations is then checked for each cluster.

# %%
#Score cells for each cell type

# Save score column names 
scores=[]
for cell_type,genes in markers_selection.items():
    score_name=cell_type+'_score'
    scores.append(cell_type+'_score')
    sc.tl.score_genes(adata_rawnormalised, gene_list=genes, score_name=score_name,  use_raw=True)

# %%
# Which clusters (column name) to analyse for cell scores
res=0.4
clusters_col='leiden_scaled_r'+str(res)

# %%
# Add cluster information to the used adata
adata_rawnormalised.obs[clusters_col]=adata_scl.obs[clusters_col]

# %%
# Plot scores distribution across clusters
rcParams['figure.figsize']=(10,3)
for score in scores:
    sc.pl.violin(adata_rawnormalised, keys=score, groupby=clusters_col, use_raw=True, stripplot=False)

# %%
# Scores per cluster, scaled from 0 to 1 in each cell type
#C: Not very informative!
#scores_df=adata_rawnormalised.obs[scores+[clusters_col]]
#score_means=scores_df.groupby(clusters_col).mean()
#score_means=pd.DataFrame(pp.minmax_scale(score_means),index=score_means.index,columns=score_means.columns)
#sb.clustermap(score_means,yticklabels=1)

# %%
# Scores normalised to interval [0,1] for each cell type - so that they can be more easily compared
scores_df_norm=adata_rawnormalised.obs[scores]
scores_df_norm=pd.DataFrame(pp.minmax_scale(scores_df_norm),columns=scores_df_norm.columns,index=adata_rawnormalised.obs.index)

# %%
# Plot of normalised scores distribution in whole dataset, per cluster
rcParams['figure.figsize']=(20,3)
fig,ax=plt.subplots()
sb.violinplot(data=scores_df_norm,inner=None,ax=ax)
ax.grid()

# %%
# Plot of normalised scores distribution, excluding low scores, per cluster
rcParams['figure.figsize']=(20,3)
fig,ax=plt.subplots()
sb.violinplot(data=scores_df_norm[scores_df_norm>0.1],inner=None,ax=ax)
ax.grid()

# %%
# Check gamma score in gamma cluster to reset the threshold
fig,ax1=plt.subplots()
a=ax1.hist(scores_df_norm.gamma_score[(adata_scl.obs[clusters_col]=='5').values],bins=70,alpha=0.5,color='r')
ax1.tick_params(axis='y', labelcolor='r')
ax2 = ax1.twinx()
a=ax2.hist(scores_df_norm.gamma_score[(adata_scl.obs[clusters_col]!='5').values],bins=70,alpha=0.5,color='b')
ax2.tick_params(axis='y', labelcolor='b')

# %%
# Thresholds for cell type assignemnt based on normalised scores
thresholds=[]
for col in scores_df_norm:
    threshold=0.55
    if col=='gamma_score':
        threshold=0.8
    elif col=='stellate_score':
        threshold=0.5
    elif col=='pericyte_score':
        threshold=0.25
    elif col =='immune_score' :
        threshold=0.4
    elif col =='epsilon_score' :
        threshold=0.8
    thresholds.append(threshold)


# %%
# Assign cell types based on scores to each cell
assignment_df=scores_df_norm>=thresholds
assignment_df.columns=[col.replace('_score','') for col in scores_df_norm.columns] 

# %%
# Count of cells per cell type
assignment_df[[col.replace('_score','') for col in scores_df_norm.columns]].sum()

# %%
# How many cell types were annotated to each cell
a=plt.hist(assignment_df.sum(axis=1))

# %%
# For each cell make a (standard) string of annotated cell types: 
# e.g. each annotated cell type in the same order, separated by '_' when multiple cell types were annotated
type_unions=[]
for idx,row in assignment_df.iterrows():
    type_union=''
    for col in row.index:
        if row[col]:
            type_union=type_union+col+'_'
    if type_union=='':
        type_union='NA'
    type_unions.append(type_union.rstrip('_'))

# %%
# Add cell types strings of cells to scores/assignment DF
assignment_df['type_union']=type_unions
assignment_df[clusters_col]=adata_rawnormalised.obs[clusters_col].values


# %% [markdown]
# ### Annotate clusters

# %%
def add_category(df,idxs,col,category):
    """
    Add single value to multiple rows of DF column (useful when column might be categorical). 
    If column is categorical the value is beforehand added in the list of present categories 
    (required for categorical columns). 
    :param df: DF to which to add values
    :param idxs: Index names of rows where the value should be assigned to the column.
    :param col: Column to which to add the value.
    :param category: The value to add to rows,column.
    """
    # If column is already present, is categorical and value is not in categories add the value to categories first.
    if col in df.columns and df[col].dtype.name=='category' and category not in df[col].cat.categories:
        df[col] = df[col].cat.add_categories([category])
    df.loc[idxs,col]=category


# %%
# Make DF of marker gene expressions (normalised, log transformed), scaled for each gene to [0,1]
used_markers=list(dict.fromkeys([marker for marker_list in markers_selection.values() for marker in marker_list]))
gene_idx=adata_rawnormalised.raw.var_names.isin(used_markers)
genes=adata_rawnormalised.raw.var_names[gene_idx]
scaled_expression=pd.DataFrame(pp.minmax_scale(adata_rawnormalised.raw.X.toarray()[:, gene_idx]),
                               columns=genes,index=adata_rawnormalised.obs.index
                              )[[marker for marker in used_markers if marker in genes]]


def subcluster(adata_cluster,res,original_cluster,assignment_df_temp,clusters_name_prefix='leiden_scaled_r'):
    """
    Cluster adata and add reult into cell type assignment DF. Plot the clustering on UMAP.
    Add name of the original cluster to which the adata belongs to new cluster names.
    Cluster column name: clusters_name_prefix+str(res) 
    :param adata_cluster: Adata (with already computed neighbours and UMAP) to cluster with leiden clustering. 
    :param res: Leiden resolution
    :param original_cluster: Original cluster name, new clusters are named 'original cluster'+'_'+'new cluster'
    :param assignment_df_temp: assignment_df to which to add clusters - should have same row order as adata
    :param clusters_name_prefix: Prefix for cluster column name.
    """
    clusters_col_name=clusters_name_prefix+str(res)
    sc.tl.leiden(adata_cluster, resolution=res, key_added=clusters_col_name, directed=True, use_weights=True)
    adata_cluster.obs[clusters_col_name]=[original_cluster+'_'+subcluster 
                                                 for subcluster in adata_cluster.obs[clusters_col_name]]
    sc.pl.umap(adata_cluster,color=clusters_col_name)
    time.sleep(1.0)
    
    assignment_df_temp['leiden_scaled_r'+str(res)]=adata_cluster.obs['leiden_scaled_r'+str(res)].values
    
def get_cluster_data(adata_original, cluster,cluster_col,assignment_df):
    """
    Subset adata and assignment_df to cluster.
    :param adata_original: Adata to subset.
    :param  cluster: Cluster for which to extract the data.
    :param cluster_col: Column where clusters are listed. Should be present in adata and assignment_df.
    :param assignment_df: DF with cell type assignmnets in column 'type_union' and cluster info. Should have index names
    that are present in adata.obs.index.
    :return: Subset of adata, assignment_df
    """
    adata_temp=adata_original[adata_original.obs[cluster_col]==cluster].copy()
    assignment_df_temp=pd.DataFrame(assignment_df.loc[adata_temp.obs.index,'type_union'])
    return adata_temp, assignment_df_temp

def cluster_annotate(assignment_df,cluster_col,present_threshold,nomain_threshold,
                     save,cluster_annotation,adata_main):
    """
    Loop through clusters, check present cell types and if there is no main cell type plot expression and QC metrics.
    For each cluster list proportion of present cell types.
    Expression (normalised, log transformed, [0,1] per gene scaled) is plotted for marker genes across cells (heatmap).
    QC metrics are plotted as scatterplot of N genes and N counts with cluster cells overlayed over all cels.
    :param assignment_df: DF with cell type assignment in column 'type_union' and cluster assignment.
    :param cluster_col: Cluster column name in assignment_df.
    :param present_threshold: Count among present cell types each cell type present if it represents 
    at least present_threshold proportion of cells in cluster. 
    :param nomain_threshold: If only 1 cell type was selected with present_threshold and 
    no cell type is present at proportion at least nomain_threshold add 'other' to annotated cell types list.
    :param save: Save cluster information to adata_main (column 'cluster') and cell types list to cluster_annotation.
    :param cluster_annotation: Dict where each cluster (key) has list of annotated cell types (value).
    :param adata_main: Add cluster to adata_main.obs.cluster. Use this adata for plotting of QC mettrics (for this a
    column 'temp_selected' is added.)
    """
    for group in assignment_df[['type_union',cluster_col]].groupby(cluster_col):
        types_summary=group[1].type_union.value_counts(normalize=True)
        present=list(types_summary[types_summary >= present_threshold].index)
        if len(present)==1 and types_summary.max() < nomain_threshold:
            present.append('other')
            
        if save:
            cluster_annotation[group[0]]=present
            add_category(df=adata_main.obs,idxs=group[1].index,col='cluster',category=group[0])
        
        print('\nCluster:',group[0],'\tsize:',group[1].shape[0])    
        print('Present:',present)
        print(types_summary)
        
        #present_nonna=[cell_type for cell_type in present if cell_type !='NA']
        #if len(present_nonna)!=1:
        if len(present)!=1 or present==['NA']:
            sb.clustermap(
                pd.DataFrame(scaled_expression.loc[group[1].index,:]),
                col_cluster=False,xticklabels=1,yticklabels=False,figsize=(7,5))
            plt.title('Cluster: '+group[0]+' assigned:'+str(present))
            adata_main.obs['temp_selected']=adata_main.obs.index.isin(group[1].index)
            rcParams['figure.figsize']=(5,5)
            p1 = sc.pl.scatter(adata_main, 'n_counts', 'n_genes', color='temp_selected', size=40,color_map='viridis')
            time.sleep(1.0)

def save_cluster_annotation(adata,cluster_name,cell_types:list,cell_names):
    """
    Save cluster information for a set of cells to cluster_annotation (from outer scope) dictionary and specified adata.
    :param adata: Adata to which to save cluster name for cells. 
    Cluster information is saved to adata.obs in 'cluster' column.
    :param cluster_name: Name of the cluster.
    :param cell_types: List of cell types annotated to the cluster.
    :param cell_names: Cell names (matching adata.obs.index) to which the new cluster is asigned.
    """
    cluster_annotation[cluster_name]=cell_types
    add_category(df=adata.obs,idxs=cell_names,col='cluster',category=cluster_name)


# %%
#Save cell types annotated to each cluster
cluster_annotation=dict()

# %%
# Display cell type annotation distribution of each cluster.
cluster_annotate(assignment_df=assignment_df,
                 cluster_col=clusters_col,present_threshold=0.1,nomain_threshold=0.7,
                     save=True,cluster_annotation=cluster_annotation,adata_main=adata_rawnormalised)

# %%
# Reassign cell type of a cluster
cluster_annotation['12']=['schwann']

# %%
# recluster clusters that previously had less or more than 1 annotted cell type (exclusing NA and other).
for cluster in sorted(list(adata_rawnormalised.obs[clusters_col].unique()),key=int):
    if len(cluster_annotation[cluster]) != 1 or len(
        [cell_type for cell_type in cluster_annotation[cluster] if cell_type !='NA' and cell_type !='other']) == 0:
        print('**** Original cluster',cluster)
        res=0.3
        adata_temp,assignment_df_temp=get_cluster_data(adata_original=adata_rawnormalised, cluster=cluster,
                                                       cluster_col='cluster',assignment_df=assignment_df)
        subcluster(adata_cluster=adata_temp,res=res,original_cluster=cluster,assignment_df_temp=assignment_df_temp)
        cluster_annotate(assignment_df=assignment_df_temp,
                         cluster_col='leiden_scaled_r'+str(res),
                         # Use more leanient condition as this is already subclustered - 
                        # otherwise data could be in some cases subclustered almost idefinitely
                         present_threshold=0.1,nomain_threshold=0.9,
                     save=True,cluster_annotation=cluster_annotation,adata_main= adata_rawnormalised)

# %% [markdown]
# #C: Cluster 10_2 is besides acinar and ductal cells, but it has annotated beta - analyse it.

# %%
# Expression of markers in cluster 10_2
sb.clustermap(pd.DataFrame(scaled_expression.loc[adata_rawnormalised.obs.query('cluster=="10_2"').index,:]), 
                  row_cluster=row_cluster, col_cluster=False,xticklabels=1,yticklabels=False,figsize=(7,5),vmin=0,vmax=1)

# %% [markdown]
# #C: Different clustering resolutions were used and some of the clusters were subclustered, however this could not resolve some of the clusters that are comprised of multiple cell types.

# %% [markdown]
# #C: Part of beta cells clusters with acinar and ductal, thus this is named differently.

# %%
# Correct entries in cluster_annotation dict - all clusters that do not have exactly one entry will be assigned 'NA'
# If cluster is to be removed add 'remove'

# Cluster 6 will be adjusted for cell cycle to try to resolve cell typs better
cluster_annotation['7_0']=['stellate_pericyte']
cluster_annotation['7_1']=['stellate']
cluster_annotation['7_2']=['pericyte']
cluster_annotation['9_2']=['endothelial']
cluster_annotation['10_0']=['ductal']
cluster_annotation['10_1']=['acinar']
cluster_annotation['10_2']=['acinar_low']

# %% [markdown]
# #### Annotate cells from unsolvable clusters based on cell type score threshold
# All those clusters are endocrine so strip any non-endocrine annotations from cell names.

# %%
# Get clusters that still do not have single cell type annotation
unsolvable=[]
for cluster in sorted(list(adata_rawnormalised.obs['cluster'].unique()),key=int):
    if len(cluster_annotation[cluster]) != 1:
        unsolvable.append(cluster)
print('Unsolvable clusters:',unsolvable)        

# %%
# Get all cells from the clusters 
idx_unsolvable=adata_rawnormalised.obs.query('cluster in @unsolvable').index

# %%
# Make new assignments with only endocrine cell types (excluding epsilone)
type_unions=[]
for idx in idx_unsolvable:
    row=assignment_df.loc[idx,:]
    type_union=''
    # Use list comprehension instead of just the endocrine list so that names of cell types stay in the same order as above
    for col in [col for col in row.index if col in ['alpha','beta','delta','gamma','epsilon']]:
        if row[col]:
            type_union=type_union+col+'_'
    if type_union=='':
        type_union='NA'
    type_unions.append(type_union.rstrip('_'))

#Reasign cell types in assignment_df
assignment_df.loc[idx_unsolvable,'type_union']=type_unions

# %%
# Group unsolvable cells by cell type and plot expression to check if assignment was ok
for cell_type in assignment_df.loc[idx_unsolvable,'type_union'].unique():
    type_idx=assignment_df.loc[idx_unsolvable,:].query('type_union == "'+cell_type+'"').index
    print(cell_type,'N:',len(type_idx))
    row_cluster=True
    if len(type_idx)<2:
        row_cluster=False
    sb.clustermap(pd.DataFrame(scaled_expression.loc[type_idx,:]), 
                  row_cluster=row_cluster, col_cluster=False,xticklabels=1,yticklabels=False,figsize=(7,5),vmin=0,vmax=1)
    plt.show()
    time.sleep(1.0)

# %%
# Replace cluster names in adata and cluster_annotation with threshold based annotation
for cell_type in assignment_df.loc[idx_unsolvable,'type_union'].unique():
    # Subset assignment_df to idx_unsolvable before extracting idices of each cell type so that 
    # Cells from other clusters to not get assigned new clusters because they share phenotype
    type_idx=assignment_df.loc[idx_unsolvable,:].query('type_union == "'+cell_type+'"').index
    save_cluster_annotation(adata=adata_rawnormalised,cluster_name=cell_type,cell_types=[cell_type],cell_names=type_idx)

# %% [markdown]
# #### Resolve cluster involved in cell cycle - UNUSED

# %% [raw]
# # Find cycling cluster
# rcParams['figure.figsize']=(4,4)
# sc.pl.umap(adata_rawnormalised, color=['phase_cyclone',clusters_col], size=10, use_raw=False)

# %% [raw]
# #Extract cycling cluster
# cluster='6'
# adata_temp,assignment_df_temp = get_cluster_data(adata_original=adata_rawnormalised, cluster=cluster,
#                                                 cluster_col=clusters_col,assignment_df=assignment_df)

# %% [raw]
# # Regress out cell cycle effect
# sc.pp.regress_out(adata_temp, ['s_cyclone', 'g2m_cyclone', 'g1_cyclone'])

# %% [raw]
# # Rescale data
# adata_temp.X=pp.StandardScaler().fit_transform(adata_temp.X)
# adata_temp.X[adata_temp.X>10]=10

# %% [raw]
# # Recompute PCA on corrected data
# rcParams['figure.figsize']=(10,5)
# sc.tl.pca(adata_temp,n_comps=20, use_highly_variable=True, svd_solver='arpack')
# sc.pl.pca_variance_ratio(adata_temp)

# %% [raw]
# # Recompute umpa and neighbours on corrected data
# sc.pp.neighbors(adata_temp,n_pcs = N_PCS) 
# sc.tl.umap(adata_temp) 

# %% [raw]
# # Plor corrected data UMAP
# rcParams['figure.figsize']=(4,4)
# sc.pl.umap(adata_temp, color=['phase_cyclone'], size=40, use_raw=False)

# %% [raw]
# # Cluster corrected data and analyse cell types
# res=0.4
# subcluster(adata_cluster=adata_temp,res=res,original_cluster=cluster,assignment_df_temp=assignment_df_temp)
# cluster_annotate(assignment_df=assignment_df_temp,
#                          cluster_col='leiden_scaled_r'+str(res),
#                          # Use more leanient condition as this is already subclustered - 
#                         # otherwise data could be in some cases subclustered almost idefinitely
#                          present_threshold=0.2,nomain_threshold=0.7,
#                      save=True,cluster_annotation=cluster_annotation,adata_main= adata_rawnormalised)

# %% [raw]
# # Try to recluster one of the sub clusters (but do not use it at the end)
# cluster2='6_2'
# adata_temp2,assignment_df_temp2 = get_cluster_data(adata_original=adata_temp, cluster=cluster2,
#                                                 cluster_col='leiden_scaled_r'+str(res),assignment_df=assignment_df_temp)
# res2=1
# subcluster(adata_cluster=adata_temp2,res=res2,original_cluster=cluster2,assignment_df_temp=assignment_df_temp2)
# cluster_annotate(assignment_df=assignment_df_temp2,
#                          cluster_col='leiden_scaled_r'+str(res2),
#                          # Use more leanient condition as this is already subclustered - 
#                         # otherwise data could be in some cases subclustered almost idefinitely
#                          present_threshold=0.2,nomain_threshold=0.7,
#                  # Do not use this subclustering
#                      save=False,
#                  cluster_annotation=cluster_annotation,adata_main= adata_rawnormalised)

# %% [raw]
# #C: Cluster 6_2 could not be resolved even with additional subclustering. Thus this reclustering is not used. The cluster is termed as mixed_endo.

# %% [raw]
# # manually annotated cluster that could not be annotated as mixed
# cluster_annotation['6_2']=['mixed_endo']

# %% [raw]
# # Add proliferative to the cell types names of cells located in cluster 6
# for cluster_sub,cell_types in cluster_annotation.copy().items():
#     if cluster_sub.startswith(cluster):
#         cluster_annotation[cluster_sub]=[cell_type+'_proliferative' for cell_type in cell_types]

# %% [markdown]
# #### Add cluster based annotation to adata

# %%
# Display cluster annotation of clusters that will be used
clusters=list(adata_rawnormalised.obs['cluster'].unique())
clusters.sort()
for cluster in clusters:
    print(cluster ,cluster_annotation[cluster ])

# %%
# Add cell type annotation to clusters. 
# If clusters was annotated with more than one cell type (inbcluding NA or other) set it to 'NA'.
if 'cell_type' in adata_rawnormalised.obs.columns:
    adata_rawnormalised.obs.drop('cell_type',axis=1,inplace=True)
for row_idx in adata_rawnormalised.obs.index:
    cluster=adata_rawnormalised.obs.at[row_idx,'cluster']
    cell_type=cluster_annotation[cluster]
    if len(cell_type)!=1:
        cell_type='NA'
    else:
        cell_type=cell_type[0]
        
    add_category(df=adata_rawnormalised.obs,idxs=row_idx,col='cell_type',category=cell_type)


# %% [markdown]
# #### Annotate proliferative cells

# %%
for idx in adata_rawnormalised.obs.index[adata_rawnormalised.obs[clusters_col]=='6']:
    add_category(adata_rawnormalised.obs, idxs=[idx], col='cell_type',
                 category=adata_rawnormalised.obs.at[idx,'cell_type']+'_proliferative')
    add_category(adata_rawnormalised.obs, idxs=[idx], col='cluster',
                 category=adata_rawnormalised.obs.at[idx,'cluster']+'_proliferative')

# %% [markdown]
# #### Add rare cell types that do not have clusters - UNUSED

# %% [raw]
# #Check that indices match
# (scaled_expression.index==assignment_df.index).all()

# %% [raw]
# # Change thresholds for cell type assignment
# if True:
#     thresholds=[]
#     for col in scores_df_norm:
#         threshold=0.55
#         if col=='gamma_score':
#             threshold=0.75
#         elif col=='stellate_score':
#             threshold=0.5
#         elif col=='pericyte_score':
#             threshold=0.25
#         elif col =='immune_score' :
#             threshold=0.4
#         elif col =='epsilon_score' :
#             threshold=0.8
#         thresholds.append(threshold)
#     assignment_df=scores_df_norm>=thresholds
#     assignment_df.columns=[col.replace('_score','') for col in scores_df_norm.columns] 

# %% [raw]
# # Epsilon
# expression=pd.DataFrame(adata_rawnormalised.raw.X.toarray(),
#                         index=adata_rawnormalised.obs.index,columns=adata_rawnormalised.raw.var_names)
# sb.clustermap(scaled_expression[assignment_df['epsilon']],
#                 col_cluster=False,xticklabels=1,yticklabels=False,figsize=(5,5))
# fig,ax=plt.subplots()
# ax.violinplot([expression.loc[assignment_df.query('epsilon').index,'Ghrl'].values,
#                expression.loc[assignment_df.query('~epsilon').index,'Ghrl'].values])
# adata_rawnormalised.obs['is_epsilon']=assignment_df['epsilon']
# # Keep size large to see all annotated cells
# sc.pl.umap(adata_rawnormalised, color=['is_epsilon','cell_type'], size=40, use_raw=False,color_map='viridis')


# %% [raw]
# # Add epsilon annotation
# add_category(df=adata_rawnormalised.obs, idxs=assignment_df.query('epsilon').index, col='cell_type',category='epsilon')

# %% [markdown]
# ### Cell type annotation evaluation

# %%
# Add cell type to adata
adata.obs['cell_type']=adata_rawnormalised.obs.cell_type
# Add final clusters and starting clusters to adata
adata.obs[['cluster',clusters_col]]=adata_rawnormalised.obs[['cluster',clusters_col]]
# Add cell type scores
score_cols=[col for col in scores_df_norm.columns if '_score' in col]
adata.obs[score_cols]=scores_df_norm[score_cols].reindex(adata.obs.index)

# %%
# Plot cell types on UMPA (pre-annotated and new ones)
rcParams['figure.figsize']=(8,8)
sc.pl.umap(adata_rawnormalised, color=['pre_cell_type','cell_type'], size=40, use_raw=False,wspace=0.7)

# %%
# Count of new cell types
adata_rawnormalised.obs.cell_type.value_counts()

# %%
# Plot mean marker expression in each cluster
rcParams['figure.figsize']=(10,5)
fig,ax=plt.subplots()
sb.heatmap(scaled_expression.groupby(adata_rawnormalised.obs['pre_cell_type']).mean(),yticklabels=1,xticklabels=1,
          vmin=0,vmax=1)
ax.set_title('pre_cell_type')
rcParams['figure.figsize']=(10,8)
fig,ax=plt.subplots()
sb.heatmap(scaled_expression.groupby(adata_rawnormalised.obs['cell_type']).mean(),yticklabels=1,xticklabels=1,
          vmin=0,vmax=1)
ax.set_title('cell_type')

# %% [markdown]
# #C: Annotated stellate are not neuronal but rather more fibroblast like (stellate != pancreatic stellate).

# %% [markdown]
# #### Cell type markers

# %% [markdown]
# Upregulated genes in each cell type compared to other cells on normalised log scaled data.

# %%
# Compute overexpressed genes in each cell type on normalised log scaled data
#Retain only cell types with >=10 cells and non-NA annotation
groups_counts=adata.obs.cell_type.value_counts()
groups=[cell_type for cell_type in groups_counts[groups_counts>=10].index if cell_type!='NA']
# Compute markers
sc.tl.rank_genes_groups(adata,groupby='cell_type',groups=groups, use_raw=False)
sc.tl.filter_rank_genes_groups(adata,groupby='cell_type', use_raw=False)

# %%
# Plot cell_type vs rest upregulated genes
rcParams['figure.figsize']=(4,3)
sc.pl.rank_genes_groups(adata,key='rank_genes_groups_filtered')

# %% [markdown]
# Expression of upregulated genes on normalised log transformed z-scaled data

# %%
# Plot expression of cell type upregulated genes on normalised log transformed z-scaled data
adata_scl.uns['rank_genes_groups_filtered']=adata.uns['rank_genes_groups_filtered']
adata_scl.obs['cell_type']=adata.obs['cell_type']
sc.pl.rank_genes_groups_stacked_violin(adata_scl,key='rank_genes_groups_filtered',n_genes=3,use_raw=False)

# %% [markdown]
# #### Markers of potential dublet populations
# Find genes overexpressed in potential dublet populations compared to individual cell types present in both parental cell types.
#
# Mixed cell type is set as reference - to use mean for filtering (besidel lFC and qval).

# %% [markdown]
# QC metrics in each cell type.

# %%
# QC metrics per cell type
rcParams['figure.figsize']=(10,3)
sc.pl.violin(adata, ['n_counts'], groupby='cell_type', size=1, log=True,rotation=90)
sc.pl.violin(adata, ['n_genes'], groupby='cell_type', size=1, log=False,rotation=90)
sc.pl.violin(adata, ['mt_frac'], groupby='cell_type', size=1, log=False,rotation=90)
sc.pl.violin(adata, 'doublet_score',groupby='cell_type',size=1, log=True,rotation=90)


# %%
# Doublet score per file
for file in adata.obs.file.unique():
    print(file)
    sc.pl.violin(adata[adata.obs.file==file,:], 'doublet_score',groupby='cell_type',size=1, log=True,rotation=90)
    time.sleep(1)

# %% [raw]
# alpha_delta:

# %% [raw]
# # Compare alpha_delta to individual cell types
# # To alpha
# #Comparison order of cell types
# cell_types_comparison=['alpha_delta','alpha']
# adata_sub=adata[adata.obs.cell_type.isin(cell_types_comparison)]
# # Set categories order - doublet as ref so that mean filtering can be used
# adata_sub.obs.cell_type.cat.reorder_categories(cell_types_comparison, inplace=True)
# test_1 = de.test.wald(data=adata_sub.layers['counts'],
#     formula_loc="~ 1 + cell_type",factor_loc_totest="cell_type",
#     gene_names=adata_sub.var_names,sample_description=adata_sub.obs)
#
# # To delta
# #Comparison order of cell types
# cell_types_comparison=['alpha_delta','delta']
# adata_sub=adata[adata.obs.cell_type.isin(cell_types_comparison)]
# # Set categories order
# adata_sub.obs.cell_type.cat.reorder_categories(cell_types_comparison, inplace=True)
# test_2 = de.test.wald(data=adata_sub.layers['counts'],
#     formula_loc="~ 1 + cell_type",factor_loc_totest="cell_type",
#     gene_names=adata_sub.var_names,sample_description=adata_sub.obs)

# %% [raw]
# # Find genes upregulated in potential dubled compared to each of the individual cell types
# shared_up=list(set(test_1.summary().query('log2fc <= -1 & qval <= 0.05 & mean >=0.1').gene
#    ) & set(test_2.summary().query('log2fc <= -1 & qval <= 0.05 & mean >=0.1').gene))
# print(shared_up)

# %% [raw]
# # Plot upregulated genes
# a=sc.pl.stacked_violin(adata[adata.obs.cell_type.isin(['alpha_delta','delta','alpha'])], 
#                            var_names=shared_up[:10],groupby='cell_type', use_raw=False )

# %% [raw]
# #C: Based on QC scores and upregulated genes the alpha_delta cluster might be comprised of dublets.

# %% [raw]
# beta_delta

# %% [raw]
# # Compare endothelial_perycite to individual cell types
# # To endothelial
# #Comparison order of cell types
# cell_types_comparison=['beta_delta','beta']
# adata_sub=adata[adata.obs.cell_type.isin(cell_types_comparison)]
# # Set categories order
# adata_sub.obs.cell_type.cat.reorder_categories(cell_types_comparison, inplace=True)
# test_1 = de.test.wald(data=adata_sub.layers['counts'],
#     formula_loc="~ 1 + cell_type",factor_loc_totest="cell_type",
#     gene_names=adata_sub.var_names,sample_description=adata_sub.obs)
#
# # To pericyte
# #Comparison order of cell types
# cell_types_comparison=['beta_delta','delta']
# adata_sub=adata[adata.obs.cell_type.isin(cell_types_comparison)]
# # Set categories order
# adata_sub.obs.cell_type.cat.reorder_categories(cell_types_comparison, inplace=True)
# test_2 = de.test.wald(data=adata_sub.layers['counts'],
#     formula_loc="~ 1 + cell_type",factor_loc_totest="cell_type",
#     gene_names=adata_sub.var_names,sample_description=adata_sub.obs)

# %% [raw]
# # Find genes upregulated in potential dubled compared to each of the individual cell types
# shared_up=list(set(test_1.summary().query('log2fc <= -1 & qval <= 0.05 & mean >=0.1').gene
#    ) & set(test_2.summary().query('log2fc <= -1 & qval <= 0.05 & mean >=0.1').gene))
# print(shared_up)

# %% [raw]
# # Plot upregulated genes
# a=sc.pl.stacked_violin(adata[adata.obs.cell_type.isin(['beta_delta','delta','beta'])], 
#                            var_names=shared_up[:10],groupby='cell_type', use_raw=False )

# %% [raw]
# #C: Based on QC scores and upregulated genes the beta_delta cluster might be comprised of dublets.

# %% [raw]
# endothelial_pericyte:

# %% [raw]
# # Compare endothelial_perycite to individual cell types
# # To endothelial
# #Comparison order of cell types
# cell_types_comparison=['endothelial_pericyte','endothelial']
# adata_sub=adata[adata.obs.cell_type.isin(cell_types_comparison)]
# # Set categories order
# adata_sub.obs.cell_type.cat.reorder_categories(cell_types_comparison, inplace=True)
# test_1 = de.test.wald(data=adata_sub.layers['counts'],
#     formula_loc="~ 1 + cell_type",factor_loc_totest="cell_type",
#     gene_names=adata_sub.var_names,sample_description=adata_sub.obs)
#
# # To pericyte
# #Comparison order of cell types
# cell_types_comparison=['endothelial_pericyte','pericyte']
# adata_sub=adata[adata.obs.cell_type.isin(cell_types_comparison)]
# # Set categories order
# adata_sub.obs.cell_type.cat.reorder_categories(cell_types_comparison, inplace=True)
# test_2 = de.test.wald(data=adata_sub.layers['counts'],
#     formula_loc="~ 1 + cell_type",factor_loc_totest="cell_type",
#     gene_names=adata_sub.var_names,sample_description=adata_sub.obs)

# %% [raw]
# # Find genes upregulated in potential dubled compared to each of the individual cell types
# shared_up=list(set(test_1.summary().query('log2fc <= -1 & qval <= 0.05 & mean >=0.1').gene
#    ) & set(test_2.summary().query('log2fc <= -1 & qval <= 0.05 & mean >=0.1').gene))
# print(shared_up)

# %% [raw]
# # Plot upregulated genes
# a=sc.pl.stacked_violin(adata[adata.obs.cell_type.isin(['endothelial','endothelial_pericyte','pericyte'])], 
#                            var_names=shared_up[:10],groupby='cell_type', use_raw=False )

# %% [raw]
# #C: Endothelial_pericyte cluster does not seem to have dublet-like QC metrics. However, it does not have unique upregulated genes towards both of the individual cell types.

# %% [raw]
# stellate_pericyte

# %% [raw]
# # Compare stellate_perycite to individual cell types
# # To stellate
# #Comparison order of cell types
# cell_types_comparison=['stellate_pericyte','stellate']
# adata_sub=adata[adata.obs.cell_type.isin(cell_types_comparison)]
# # Set categories order
# adata_sub.obs.cell_type.cat.reorder_categories(cell_types_comparison, inplace=True)
# test_1 = de.test.wald(data=adata_sub.layers['counts'],
#     formula_loc="~ 1 + cell_type",factor_loc_totest="cell_type",
#     gene_names=adata_sub.var_names,sample_description=adata_sub.obs)
#
# # To pericyte
# #Comparison order of cell types
# cell_types_comparison=['stellate_pericyte','pericyte']
# adata_sub=adata[adata.obs.cell_type.isin(cell_types_comparison)]
# # Set categories order
# adata_sub.obs.cell_type.cat.reorder_categories(cell_types_comparison, inplace=True)
# test_2 = de.test.wald(data=adata_sub.layers['counts'],
#     formula_loc="~ 1 + cell_type",factor_loc_totest="cell_type",
#     gene_names=adata_sub.var_names,sample_description=adata_sub.obs)

# %% [raw]
# # Find genes upregulated in potential dubled compared to each of the individual cell types
# shared_up=list(set(test_1.summary().query('log2fc <= -2 & qval <= 0.01 & mean >=0.1').gene
#    ) & set(test_2.summary().query('log2fc <= -2 & qval <= 0.01 & mean >=0.1').gene))
# print(shared_up)

# %% [raw]
# # Plot upregulated genes
# a=sc.pl.stacked_violin(adata[adata.obs.cell_type.isin(['stellate','stellate_pericyte','pericyte'])], 
#                            var_names=shared_up[:10],groupby='cell_type', use_raw=False )

# %% [raw]
# #C: Stellate_pericyte cluster does not seem to have dublet-like QC metrics and it has some genes upregulated compared to the individual cell types. Thus this might be a unique cell population.

# %% [markdown]
# ### Expected doublet rates
#
# Adapted from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6126471/#ref-7, but using 0.1 as doublet rate (described in my notes).
#
# Expected multiplet number is calculated for each file separately.

# %%
# Singlet cell types whose multiplet rates are predicted
cell_types=['alpha','beta','gamma','delta']
# DFs with expected multiplet rates
expected_dfs=[]
# Exclude some endocrine cells as they form completely separate populations 
# with no multiplets in neighbourhood)
exclude_types=[cell_type for cell_type in adata.obs.cell_type.unique() if 'proliferative'  in cell_type]
print('Excluding following cells:',exclude_types)
# Calculate expected rate for each file separately
for file in adata.obs.file.unique():
    print('\nFile:',file)
    cell_type_temp=adata.obs.query('file == "'+file+'" & cell_type not in @exclude_types').cell_type.copy()
    
    # N of droplets containing at least one cell of the relevant cell type
    Ns=dict()
    for cell_type in cell_types:
        n_celltype=cell_type_temp.str.contains(cell_type).sum()
        Ns[cell_type]=n_celltype
    print('N droplets that contain at least one:',Ns)
    
    # Calculate N (see formula in notes)
    N=emr.get_N(Ns=Ns.values())
    regex_any='|'.join(['.*'+cell_type+'.*' for cell_type in cell_types])
    print('N:',round(N,1), '(observed cell-containing N with relevant cell types:',
          cell_type_temp.str.contains(fr'{regex_any}').sum(),')')
    
    # Calculate mu for each cell type
    mus=dict()
    for cell_type in cell_types:
        mus[cell_type]=emr.mu_cell_type(N_cell_type=Ns[cell_type], N=N)
    print('mu:',{k:round(v,4) for k,v in mus.items()})
    
    # Extract multiplet types and their components
    multiplet_types=dict()
    for cell_type in cell_type_temp.unique():
        # Get components of multiplets by retaining name parts contained in original cell type dict
        components_all=cell_type.split('_')
        components_relevant=[type_singlet for type_singlet in components_all if type_singlet in cell_types]
        if len(components_relevant) > 1:
            multiplet_types[cell_type]=components_relevant
            # This does not assure that initially the cell types are not counted wrongly 
            # (e.g. including proliferative, ...)
            #if len(components_relevant) < len(components_all):
            #    warnings.warn('Multiplet type has additional unrecognised component: '+cell_type)
    #print('Relevant multiplet cell types:',multiplet_types.keys())
    
    # This also analyses O and E of singlet types, but this is not so relevant as changes in multiplets also 
    # affect it?
    #types_OE= dict(zip(cell_types,[[i] for i in cell_types]))
    #types_OE.update(multiplet_types)
    
    # Calculate O (observed) and E (expected) numbers
    expected_df=pd.DataFrame(index=multiplet_types.keys(),columns=['O_'+file,'E_'+file
                                                                  # ,'E_atleast'
                                                                  ])
    #for multiplet, components in types_OE.items():
    for multiplet, components in multiplet_types.items():
        absent=np.setdiff1d(cell_types,components)
        # N of cellls of this cell type
        expected_df.at[multiplet,'O_'+file]=(cell_type_temp==multiplet).sum()
        # Expected N of cells that have all the individual cell types of multiplet and non of the other 
        # considered possible multiplet ontributors
        expected_df.at[multiplet,'E_'+file]=emr.expected_multiplet(
            mu_present=[mus[cell_type] for cell_type in components], 
            mu_absent=[mus[cell_type] for cell_type in absent], 
            N=N)
        # E that droplet contains at least one cell of each present cell type,
        # but does not exclude other cell types from being present
        #expected_df.at[multiplet,'E_atleast']=emr.expected_multiplet(
        #    mu_present=[mus[cell_type] for cell_type in components], 
        #    mu_absent=[], 
        #    N=N)
    expected_dfs.append(expected_df)
    
# Merge O/E rates for all files into single DF
display(pd.concat(expected_dfs,axis=1))
del cell_type_temp

# %% [markdown]
# #C: Potential doublet populations alpha_delta and beta_delta are differently likely to be doublets across different samples (based on O/E and doublet scores). Thus their distribution is shown on sample-specific UMAPs.

# %%
# Display where are ambigous multiplets present in different files
rcParams['figure.figsize']=(6,6)
for file in adata.obs.file.unique():
    print(file)
    adata.obs['temp_file']=adata.obs.file==file
    adata.obs['temp_file_AD']=(adata.obs.file==file) & (adata.obs.cell_type=='alpha_delta')
    adata.obs['temp_file_BD']=(adata.obs.file==file) & (adata.obs.cell_type=='beta_delta')
    sc.pl.embedding(adata,'X_umap_scl', color=['temp_file','temp_file_AD','temp_file_BD'], 
                    size=10, use_raw=False,wspace=0.1)
    time.sleep(1)
adata.obs.drop(['temp_file','temp_file_AD','temp_file_BD'],axis=1,inplace=True)

# %% [markdown]
# #C: AD and BD have same location across files, but different frequency. AD was assigned as non-doublet in other datasets (Fltp, VSGref, STZref) and BD was assigned as doublet. However, here it seems that both populations might be non-doublets. In case of BD this could be due to poor cell dissociation in one of the samples (cells are located in the same region as doublet-like cells from other files) as the same cell trype was not observed as non-doublet in other files. 

# %% [markdown]
# Proliferative potential doublets are added to multiplets based on non-multiplet results and low N.

# %%
# Reassign cell types to multiplet
multiplets=['alpha_beta_delta','beta_delta_gamma','beta_gamma','beta_delta',
            # Add proliferative
            # beta_delta_proliferative and alpha_delta_gamma_proliferative are
            # added to multiplets as they have relatively low counts (4 cells, 1 cell) and high doublet scores
            'alpha_beta_proliferative','beta_delta_proliferative',
            'alpha_beta_delta_proliferative','alpha_delta_gamma_proliferative','beta_gamma_proliferative']

adata.obs['cell_type_multiplet']=adata.obs.cell_type.replace(multiplets, 'multiplet')

# %%
# Plot new cell types
rcParams['figure.figsize']=(6,6)
sc.pl.embedding(adata,'X_umap_scl', color=['cell_type','cell_type_multiplet'], size=40, use_raw=False,wspace=0.8)

# %% [markdown]
# ## Resolve beta subtypes

# %% [markdown]
# #### Preprocess beta cell data

# %% [markdown]
# Select beta cells only

# %%
[ct for ct in adata.obs.cell_type_multiplet.unique() if 'beta' in ct]

# %% [markdown]
# #C: No beta str containing cell types to exclude

# %%
# Subset adata
selected_beta=[ct for ct in adata.obs.cell_type_multiplet.unique() if 'beta' in ct]
adata_beta=adata[adata.obs.cell_type.isin(selected_beta),:].copy()
adata_beta.shape

# Normalise raw data for plotting and cell type scoring
adata_raw_beta=adata_beta.raw.to_adata().copy()
# Normalize and log transform
adata_raw_beta.X /= adata_raw_beta.obs['size_factors'].values[:,None] # This reshapes the size-factors array
sc.pp.log1p(adata_raw_beta)
adata_raw_beta.X = sparse.csr_matrix(np.asarray(adata_raw_beta.X))

# Scale beta adata for umap
adata_scl_beta=adata_beta
del adata_beta
sc.pp.scale(adata_scl_beta,max_value=10)
sc.pp.pca(adata_scl_beta, n_comps=10, use_highly_variable=True, svd_solver='arpack')
sc.pp.neighbors(adata_scl_beta,n_pcs=10)
sc.tl.umap(adata_scl_beta)

# Combine raw and X data
adata_rawnormalised_beta=adata_scl_beta.copy()
adata_rawnormalised_beta.raw=adata_raw_beta
del adata_raw_beta

# %%
metadata=pd.read_excel('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/scRNA-seq_pancreas_metadata.xlsx',
                      sheet_name='Fltp_P16')
# Add metadata 
samples=adata_rawnormalised_beta.obs.file.unique()
value_map={sample:metadata.query('sample_name =="'+sample+'"')['design'].values[0] for sample in samples}
adata_rawnormalised_beta.obs['design']=adata_rawnormalised_beta.obs.file.map(value_map)

# %%
rcParams['figure.figsize']=(4,4)
random_indices=np.random.permutation(list(range(adata_rawnormalised_beta.shape[0])))
sc.pl.umap(adata_rawnormalised_beta[random_indices,:], color=['design','file'],size=10, use_raw=True,wspace=0.3)

# %% [markdown]
# #### Check markers

# %%
# Select beta and endocrine markers and group by cell type and subtype
for group_name, group in markers.query('Cell in ["beta","endocrine"]').fillna('/').groupby(['Cell','Subtype']):
    print(group_name)
    genes=group.Gene
    # Retain genes present in raw var_names
    missing=[gene for gene in genes if gene not in adata.raw.var_names]
    genes=[gene for gene in genes if gene in adata.raw.var_names]
    print('Missing genes:',missing)
    rcParams['figure.figsize']=(4,4)
    sc.pl.umap(adata_rawnormalised_beta, color=['cell_type'],size=10, use_raw=True)
    sc.pl.umap(adata_rawnormalised_beta, color=genes ,size=10, use_raw=True)

# %% [markdown]
# #C: The following populations can be identified: immature (high: Rbp4, Cd81, Mafb,Pyy, low: Mafa, Ucn3), Etv1 high, Ins1,2 low. Etv1 high and Ins low might be overlapping, similarly Etv1 and Pyy high. Proliferative beta cells will not be re-annotated (excluded from further analysis). Ccl5 seems to be sample specific thus a obs could be added another column with Ccl5 high per cell - otherwise all other subtypes would be separated based on this sample specific value. Howevrer,  Ccl5 is high in Fltp- samples of P16 and adoult - thus confounding with Fltp, indicating true population. Ccl5 will be thus used as other markers. Nevertheless, this is not evident in 2y Fltp- samples.

# %%
adata_rawnormalised_beta=adata_rawnormalised_beta[adata_rawnormalised_beta.obs.cell_type!='beta_proliferative',:]

# %% [markdown]
# Try to cluster cells to see if clusters can resolve above defined populations.

# %%
res=0.7
sc.tl.leiden(adata_rawnormalised_beta, resolution=res, key_added='leiden_scaled_r'+str(res), directed=True, use_weights=True)
sc.pl.umap(adata_rawnormalised_beta, color='leiden_scaled_r'+str(res))

# %% [markdown]
# #C: Clustering is not able to well separate immature from mature cells (based on markers). This could be in part due to using HVGs defined on the whole dataset for distance calculation. It is also possible that there are other differences between cells (e.g. unknown markers). As markers will be used for annotation the annotation will be rather done based on scores (and not clusters) as this better correesponds to the known markers. Immature and Cd81 high were separted for consistency with other studies.

# %%
# Selected markers that seem to be able to distinguish between cell seubtypes
markers_selection_beta={
'immature':['Rbp4',  'Mafb', 'Pyy'],
'Cd81_high':['Cd81'],
'mature':['Mafa', 'Ucn3'],
'Etv1_high':['Etv1'],
'ins_high':['Ins1','Ins2'],
'Cxcl10_high':['Cxcl10'],
'Ccl5_high':['Ccl5']
}

# %% [markdown]
# #### Calculate scores

# %%
# Calculate scores
scores=[]
for cell_type,genes in markers_selection_beta.items():
    score_name=cell_type+'_score'
    scores.append(cell_type+'_score')
    sc.tl.score_genes(adata_rawnormalised_beta, gene_list=genes, score_name=score_name,  use_raw=True)

# %%
# Scores normalised to interval [0,1] for each cell type - so that they can be more easily compared
scores_df_norm_beta=adata_rawnormalised_beta.obs[scores]
scores_df_norm_beta=pd.DataFrame(pp.minmax_scale(scores_df_norm_beta),columns=scores_df_norm_beta.columns,index=adata_rawnormalised_beta.obs.index)

# %% [markdown]
# #### Plot score distn and score relationships

# %%
# Plot of normalised scores distribution in whole dataset, per cluster
rcParams['figure.figsize']=(20,3)
fig,ax=plt.subplots()
sb.violinplot(data=scores_df_norm_beta,inner=None,ax=ax)
ax.grid()

# %%
# Dict for score thresholds
beta_marker_thresholds=dict()

# %%
rcParams['figure.figsize']=(4,4)
plt.scatter(scores_df_norm_beta['immature_score'],scores_df_norm_beta['mature_score'],s=0.1)
plt.xlabel('immature_score')
plt.ylabel('mature_score')

# %% [markdown]
# Distribution of differences between immature and mature scores

# %%
scores_df_norm_beta['immature-mature_score']=scores_df_norm_beta['immature_score']-scores_df_norm_beta['mature_score']

# %%
a=plt.hist(scores_df_norm_beta['immature-mature_score'],bins=50)
beta_marker_thresholds['immature-mature_score']=0.15
plt.axvline(beta_marker_thresholds['immature-mature_score'],c='r')
plt.xlabel('immature-mature score')

# %% [markdown]
# #C: The mature annd immaturte populations are hard to separate. The thershold will be set at 0.15 of their difference.

# %%
plt.scatter(scores_df_norm_beta['immature_score'],scores_df_norm_beta['mature_score'],s=0.1,
            c=scores_df_norm_beta['immature-mature_score']>=beta_marker_thresholds['immature-mature_score'],
           cmap='PiYG')
plt.xlabel('immature_score')
plt.ylabel('mature_score')

# %%
rcParams['figure.figsize']=(4,4)
plt.scatter(scores_df_norm_beta['Cd81_high_score'],scores_df_norm_beta['immature-mature_score'],s=0.1)
plt.xlabel('Cd81_high_score')
plt.ylabel('iommature-mature_score')

# %% [markdown]
# #C: First large peak is often 0 expressing cells and the second and third peak then represent low (e.g. low but not zero) and high cells. 

# %%
a=plt.hist(scores_df_norm_beta['Cd81_high_score'],bins=70)
#plt.yscale('log')
plt.ylabel('Cd81_high_score')
beta_marker_thresholds['Cd81_high_score']=0.3
plt.axvline(beta_marker_thresholds['Cd81_high_score'],c='r')

# %%
# Convert ins high to ins low
scores_df_norm_beta['ins_low_score']=scores_df_norm_beta['ins_high_score']*-1+1

# %%
rcParams['figure.figsize']=(4,4)
plt.scatter(scores_df_norm_beta['Etv1_high_score'],scores_df_norm_beta['ins_low_score'],s=1)
plt.xlabel('Etv1_high_score')
plt.ylabel('ins_low_score')
beta_marker_thresholds['Etv1_high_score']=0.53
plt.axvline(beta_marker_thresholds['Etv1_high_score'],c='r')

# %%
a=plt.hist(scores_df_norm_beta['ins_low_score'],bins=50)
plt.yscale('log')
plt.ylabel('ins_low_score')
beta_marker_thresholds['ins_low_score']=0.4
plt.axvline(beta_marker_thresholds['ins_low_score'],c='r')

# %%
rcParams['figure.figsize']=(4,4)
plt.scatter(scores_df_norm_beta['immature-mature_score'],scores_df_norm_beta['ins_low_score'],s=1)
plt.xlabel('immature-mature_score')
plt.ylabel('ins_low_score')

# %%
a=plt.hist(scores_df_norm_beta['Cxcl10_high_score'],bins=70)
#plt.yscale('log')
plt.ylabel('Cxcl10_high_score')
beta_marker_thresholds['Cxcl10_high_score']=0.3
plt.axvline(beta_marker_thresholds['Cxcl10_high_score'],c='r')

# %% [markdown]
# #C: A higher Cxcl10 threshold was selected as threshold between main two peaks still includes cells all across UMAP.

# %%
rcParams['figure.figsize']=(4,4)
plt.scatter(scores_df_norm_beta['immature-mature_score'],scores_df_norm_beta['Cxcl10_high_score'],s=1)
plt.xlabel('immature-mature_score')
plt.ylabel('Cxcl10_high_score')

# %%
a=plt.hist(scores_df_norm_beta['Ccl5_high_score'],bins=70)
#plt.yscale('log')
plt.ylabel('Ccl5_high_score')
beta_marker_thresholds['Ccl5_high_score']=0.2
plt.axvline(beta_marker_thresholds['Ccl5_high_score'],c='r')

# %% [markdown]
# #C: Based on UMAPs Etv1 high and ins low mostly overlap immature cells. Check on UMAP where are each of these cells to see if all Etv1 high and ins low cells can be classified as immature.

# %%
# Add Ccl5 high to special obs column
#ccl5=pd.Series(index=scores_df_norm_beta.index)
#ccl5[scores_df_norm_beta['Ccl5_high_score']>=beta_marker_thresholds['Ccl5_high_score']]='high'
#ccl5.fillna('low',inplace=True)
#adata.obs.loc[adata_rawnormalised_beta.obs.index,'Ccl5']=ccl5

# %%
# Thresholds for cell type assignemnt based on normalised scores
scores_cols_beta=['immature-mature_score','Cd81_high_score','Etv1_high_score','ins_low_score','Cxcl10_high_score',
                  'Ccl5_high_score']
thresholds_beta=[beta_marker_thresholds[score_col] for score_col in scores_cols_beta]

# %%
# Assign cell types based on scores to each cell
assignment_df_beta=scores_df_norm_beta[scores_cols_beta]>=thresholds_beta
# replace '-mature' from immature-mature to get just immature out for the cell type name
assignment_df_beta.columns=[col.replace('_score','').replace('-mature','') for col in scores_cols_beta] 

# %%
# For each cell make a (standard) string of annotated cell types: 
# e.g. each annotated cell type in the same order, separated by '_' when multiple cell types were annotated
type_unions_beta=[]
for idx,row in assignment_df_beta.iterrows():
    type_union=''
    for col in row.index:
        if row[col] :
            type_union=type_union+col+'_'
    if type_union=='':
        type_union='NA'
    type_unions_beta.append(type_union.rstrip('_'))

# %%
# Add cell types strings of cells to scores/assignment DF
assignment_df_beta['type_union']=type_unions_beta

# %%
assignment_df_beta['type_union'].value_counts()

# %%
adata_rawnormalised_beta.obs['beta_subtype']=assignment_df_beta['type_union']

# %%
rcParams['figure.figsize']=(4,4)
sc.pl.umap(adata_rawnormalised_beta,color='beta_subtype')

# %% [markdown]
# Show rare subtypes if they cluster on UMAP with more common subtypes.

# %%
for subtype in adata_rawnormalised_beta.obs.beta_subtype.value_counts().index:
    if subtype !='NA':
        adata_rawnormalised_beta.obs['temp']=adata_rawnormalised_beta.obs['beta_subtype']==subtype
        sc.pl.umap(adata_rawnormalised_beta,color='temp',size=40,title=subtype)
adata_rawnormalised_beta.obs.drop('temp',axis=1,inplace=True)

# %% [markdown]
# #C: Set less common cell types to more common ones based on UMAP position.

# %%
# rename cell types

adata_rawnormalised_beta.obs['beta_subtype'].replace({
    'immature_Cd81_high':'immature',
'Cd81_high_Etv1_high':'immature_Etv1_high',
'immature_Cd81_high_ins_low':'immature_ins_low',
'immature_Cd81_high_ins_low_Ccl5_high':'immature_ins_low',
'immature_Cd81_high_Cxcl10_high':'immature_Cxcl10_high',
'immature_Cd81_high_Cxcl10_high_Ccl5_high':'immature_Cxcl10_high_Ccl5_high',
'immature_Cd81_high_Etv1_high':'immature_Etv1_high',
'immature_Cd81_high_Etv1_high_Ccl5_high':'immature_Etv1_high_Ccl5_high',
'immature_Cd81_high_Etv1_high_Cxcl10_high':'immature_Etv1_high', 
'Cd81_high_ins_low':'immature_ins_low',
'Etv1_high_Cxcl10_high':'NA', 
'immature_Cd81_high_ins_low_Cxcl10_high':'immature_ins_low', 
'immature_Etv1_high_ins_low':'immature_Etv1_high', 
'Etv1_high':'NA', 
'Etv1_high_Ccl5_high':'NA', 
'immature_Cd81_high_Etv1_high_ins_low':'immature_Etv1_high',
'immature_Cd81_high_Etv1_high_ins_low_Ccl5_high':'immature_Etv1_high_Ccl5_high',
'immature_Cd81_high_Etv1_high_ins_low_Cxcl10_high':'immature_Etv1_high',
'immature_Cd81_high_Etv1_high_ins_low_Cxcl10_high_Ccl5_high':'immature_Etv1_high_Ccl5_high',
'immature_Etv1_high_Cxcl10_high':'immature_Cxcl10_high',
'immature_Cd81_high_Ccl5_high':'immature_Ccl5_high'
                                                }, inplace=True)

# %%
adata_rawnormalised_beta.obs['beta_subtype'].value_counts()

# %%
for subtype in adata_rawnormalised_beta.obs.beta_subtype.value_counts().index:
    if subtype !='NA':
        adata_rawnormalised_beta.obs['temp']=adata_rawnormalised_beta.obs['beta_subtype']==subtype
        sc.pl.umap(adata_rawnormalised_beta,color='temp',size=40,title=subtype)
adata_rawnormalised_beta.obs.drop('temp',axis=1,inplace=True)

# %%
# Add cell type data to adata
adata.obs[['cell_subtype','cell_subtype_multiplet']]=adata.obs[['cell_type','cell_type_multiplet']]
for subtype in adata_rawnormalised_beta.obs['beta_subtype'].unique():
    if subtype != 'NA':
        idxs=adata_rawnormalised_beta.obs.query('beta_subtype == @subtype').index
        subtype='beta_'+subtype
        add_category(df=adata.obs,idxs=idxs,col='cell_subtype',category=subtype)
        add_category(df=adata.obs,idxs=idxs,col='cell_subtype_multiplet',category=subtype)
# reorder categories
adata.obs.cell_subtype=pd.Categorical(adata.obs.cell_subtype,
                                   categories=sorted(list(adata.obs.cell_subtype.unique())),ordered=True)
adata.obs.cell_subtype_multiplet=pd.Categorical(adata.obs.cell_subtype_multiplet,
                                  categories=sorted(list(adata.obs.cell_subtype_multiplet.unique())),ordered=True)

# %%
sc.pl.embedding(adata,'X_umap_scl',color='cell_subtype_multiplet')

# %% [markdown]
# ### Save annotation

# %%
if SAVE:
    h.save_h5ad(adata, shared_folder+"data_annotated.h5ad",unique_id2=UID2)

# %%
#adata=h.open_h5ad(shared_folder+"data_annotated.h5ad",unique_id2=UID2)

# %%
