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
import loompy as lo
import numpy as np 
import glob
import seaborn as sb
import pandas as pd
import scrublet as scr
import pickle

sc.settings.verbosity = 3

from matplotlib import rcParams
import matplotlib.pyplot as plt

import sys  
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/mm_pancreas_atlas_rep/code/')
import helper as h
from constants import SAVE

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
# library(RColorBrewer)
# library(DropletUtils)
# library(BiocParallel)

# %%
# Path for saving results - last shared folder by all datasets
shared_folder='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE142465/'
#Path for loading individual samples
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE142465/scanpy_AnnData/SRR107515*/'
#Unique ID2 for reading/writing h5ad files with helper function
UID2='spikein_drug_pp'

# %% [markdown]
# ## Load data (filtered)

# %%
# Load metadata for the project
metadata=pd.read_excel('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/scRNA-seq_pancreas_metadata.xlsx',
             sheet_name='spikein_drug')

# %%
# Find samples used for reference
samples=metadata
print('Selected samples:')
print(samples)

# %%
# List all files
file_name='filtered_feature_bc_matrix.h5ad'
files=glob.glob(path_data+file_name)
# Find which parts of file paths differ between files to later use them as file id
diff_path_idx=[]
for position in range(len(path_data.split('/'))):
    values=set([file.split('/')[position] for file in files])
    if len(values)>1:
        diff_path_idx.append(position)

# %%
# Subset to files used for reference
files_subset=[]
for file in files:
    sample='_'.join([file.split('/')[i] for i in diff_path_idx])
    if any(sample_name in sample for sample_name in samples.sample_name.values):
        files_subset.append(file)
print('Reference sample files:')
print(files_subset)

# %%
# Load files and extract parts of file path that identifies the file, compared to other loaded files
adatas=[]
file_diffs=[]
for file in files_subset:
    print('Reading file',file)
    #adatas.append(sc.read_h5ad(file))
    adatas.append(h.open_h5ad(file=file,unique_id2=UID2))
    file_diffs.append('_'.join([file.split('/')[i] for i in diff_path_idx]))
    
adata = ann.AnnData.concatenate( *adatas,  batch_key = 'file', batch_categories = file_diffs).copy()    


# %%
# Sample names
file_diffs

# %%
adata

# %%
# Add file annotation if single sample is present so that below code works
if len(file_diffs)==1:
    adata.obs['file']=file_diffs[0]

# %% [markdown]
# ## Empty droplets and ambient gene expression

# %% [markdown]
# ### Check that empty droplets were removed

# %% [markdown]
# N counts for cells that passed CellRanger filtering

# %%
# Visually check if empty cells are present
adata.obs['n_counts'] = adata.X.sum(axis = 1)


# %%
rcParams['figure.figsize']= (15,5)
t1 = sc.pl.violin(adata, 'n_counts',
                  groupby='file',
                  size=2, log=True, cut=0)

# %%
# Load raw data
# List all files
file_name='raw_feature_bc_matrix.h5ad'
files=glob.glob(path_data+file_name)
# Find which parts of file paths differ between files to later use them as file id
diff_path_idx=[]
for position in range(len(path_data.split('/'))):
    values=set([file.split('/')[position] for file in files])
    if len(values)>1:
        diff_path_idx.append(position)

# %%
# Subset to files used for reference
files_subset=[]
for file in files:
    sample='_'.join([file.split('/')[i] for i in diff_path_idx])
    if any(sample_name in sample for sample_name in samples.sample_name.values):
        files_subset.append(file)
print('Reference sample files:')
print(files_subset)

# %%
# Load files and extract parts of file path that identifies the file, compared to other loaded files
adatas_raw=[]
file_diffs=[]
for file in files_subset:
    print('Reading file',file)
    #adatas_raw.append(sc.read_h5ad(file))
    adatas_raw.append(h.open_h5ad(file=file,unique_id2=UID2))
    file_diffs.append('_'.join([file.split('/')[i] for i in diff_path_idx]))
    
adata_raw = ann.AnnData.concatenate( *adatas_raw,  batch_key = 'file', batch_categories = file_diffs).copy()    


# %%
adata_raw

# %%
# Add file annotation if single sample is present so that below code works
if len(file_diffs)==1:
    adata_raw.obs['file']=file_diffs[0]

# %%
adata_raw.obs['n_counts'] = adata_raw.X.sum(1)

# %%
# Find drops removed by CellRanger
filtered_drops=~adata_raw.obs.index.isin(adata.obs.index)
print('N drops filtered out as empty:',filtered_drops.sum(),
      'out of all drops:',adata_raw.shape[0],'-> remaining:',adata.shape[0])

# %% [markdown]
# Distribution of N counts of drops that were removed by CellRanger

# %%
# Plot n_counts of drops that were removed by CellRanger
rcParams['figure.figsize']= (15,5)
sc.pl.violin(adata_raw[filtered_drops], ['n_counts'], groupby='file', size=1, log=False,rotation=90,stripplot=False)

# %%
#print('N cells per non-filtered sample')
#adata_raw.obs['file'].value_counts()

# %%
#Remove empty genes and cells
sc.pp.filter_cells(adata_raw, min_counts=1)
sc.pp.filter_genes(adata_raw, min_cells=1)

# %%
adata_raw

# %% [markdown]
# Cell N counts sorted by cell N counts rank without all 0 cells. 

# %%
from cycler import cycler

# %%
# For each file plot N count vs cell rank by count with log axes
rcParams['figure.figsize']= (15,10)
fig, ax=plt.subplots()
ax.set_prop_cycle(cycler(color=plt.cm.tab20.colors))
for file in adata_raw.obs.file.unique():
    adata_raw_sub=adata_raw[adata_raw.obs.file==file,:].copy()
    plt.plot(list(range(1,adata_raw_sub.shape[0]+1)),adata_raw_sub.obs.n_counts.sort_values(ascending=False),
            label=file,lw=2)
del adata_raw_sub
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Droplet rank by count')
plt.ylabel('N counts')
ax.axvline(500,c='k',alpha=0.5)
ax.axvline(1000,c='k',alpha=0.5)
ax.axvline(2000,c='k',alpha=0.5)
ax.axvline(25000,c='k',alpha=0.5)
ax.axvline(8000,c='k',alpha=0.5)
ax.axhline(8,c='k',alpha=0.5)

# %% [markdown]
# ### Compute ambient genes

# %% [markdown]
# Compute ambient genes with emptyDrops and based on mean expression in low count droplets across all samples.

# %%
# Prepare data for emptyDrops
sparse_mat = adata_raw.X.T
genes = adata_raw.var_names
barcodes = adata_raw.obs_names

# %% magic_args="-i sparse_mat -i genes -i barcodes -o ambient_scores" language="R"
# # Run emptyDrops and output scores per gene
# sce <- SingleCellExperiment(assays = list(counts = sparse_mat), colData=barcodes)
# rownames(sce) <- genes 
# ambient <- emptyDrops(counts(sce),BPPARAM=MulticoreParam(workers = 16))
# #is_cell <- ambient$FDR <= 0.05
# #threshold_ambient <- 0.005
# #ambient_genes <- names(ambient@metadata$ambient[ambient@metadata$ambient> threshold_ambient,])
# ambient_scores <- as.data.frame(ambient@metadata$ambient)
# #barcodes_filtered <- barcodes[which(is_cell)]

# %% [markdown]
# emptyDrops top ambient genes with scores.

# %%
ambient_scores.columns=['ambient_score']
ambient_scores.sort_values('ambient_score',ascending=False).query('ambient_score>=0.005')

# %% [markdown]
# Genes with highest mean expression in empty droplets (n_counts < 100, similar to emptyDrops).

# %%
ambient_vars=['mean_ambient_n_counts']
adata_raw.var['mean_ambient_n_counts']=adata_raw[adata_raw.obs['n_counts']<100].X.mean(0).A1
adata_raw.var['mean_ambient_n_counts'].sort_values(ascending=False)[:20]

# %% [markdown]
# #C: The two gene lists are very similar for top genes.

# %% [markdown]
# Compute ambient genes for each sample based on mean expression in empty droplets.

# %%
# Mean empty expression in individual samples
#for file in adata_raw.obs.file.cat.categories:
for file in adata_raw.obs.file.unique():
    adata_raw.var['mean_ambient_n_counts_' + file] = adata_raw[(adata_raw.obs.file == file) & (adata_raw.obs['n_counts']<100)
                                                  ].X.mean(0).A1
    ambient_vars.append('mean_ambient_n_counts_' + file)
    
# Top genes in individual samples    
ambient_genes = set()
#for file in adata_raw.obs.file.cat.categories:
for file in adata_raw.obs.file.unique():
    ambient_file = list(adata_raw.var['mean_ambient_n_counts_' + file][np.flip(np.argsort(adata_raw.var['mean_ambient_n_counts_' + file]))].index)
    ambient_genes.update(ambient_file)

# %% [markdown]
# Mean ambient expression in individual samples divided by total mean ambient expression of individual samples. Ordered by combined smples, including top 20 ambient genes from each comparison.

# %%
# Normalise ambient
ambient_df=adata_raw.var[ambient_vars]
ambient_df=ambient_df.loc[ambient_genes,ambient_vars]/ambient_df.sum(axis=0)

# %%
# Display ambient genes across samples
rcParams['figure.figsize']= (20,10)
sb.heatmap(ambient_df.sort_values('mean_ambient_n_counts',ascending=False).iloc[:20,:],annot=True,fmt='.2e')

# %%
# Save ambient DF for top N genes
if SAVE:
    ambient_df.to_csv(shared_folder+"ambient_genes_topN_scores.tsv",sep='\t')

# %%
# Save genes with scaled mean ambient expression at least > 0.005 in any sample - 
# use the same genes for each sample so that further prrocessing (cell type annotation, embedding) can be done jointly
# !!! Check on heatmap that all such genes were included in the filtered 20 genes
ambient_genes_selection=list(ambient_df[(ambient_df>0.005).any(axis=1)].index)

print('Selected ambient genes:',ambient_genes_selection)
if SAVE:
    pickle.dump( ambient_genes_selection, open( shared_folder+"ambient_genes_selection.pkl", "wb" ) )

# %% [markdown]
# #### Distances between samples based on ambient gene scores

# %%
# Summed absolute distances between ambient scores of genes for each pair of samples
columns=[col for col in ambient_df.columns if col!='mean_ambient_n_counts']
col_names=[file+'_'+design for file,design in zip(metadata.sample_name,metadata.design)]
ambient_dist=pd.DataFrame(index=col_names,columns=col_names)
for idx1,col1 in enumerate(columns[:-1]):
    for idx2,col2 in enumerate(columns[idx1+1:]):
        diff=abs(ambient_df[col1]-ambient_df[col2]).sum()
        col_name1=col1.replace('mean_ambient_n_counts_','')
        col_name1=col_name1+'_'+metadata.query('sample_name==@col_name1')['design'].values[0]
        col_name2=col2.replace('mean_ambient_n_counts_','')
        col_name2=col_name2+'_'+metadata.query('sample_name==@col_name2')['design'].values[0]
        ambient_dist.at[col_name1,col_name2]=diff
        ambient_dist.at[col_name2,col_name1]=diff
ambient_dist.fillna(0,inplace=True)
sb.clustermap(ambient_dist)

# %%
# Summed absolute distances between top (based on avg) ambient genes ambient scores for each pair of samples
ambient_df=ambient_df.sort_values('mean_ambient_n_counts',ascending=False)
columns=[col for col in ambient_df.columns if col!='mean_ambient_n_counts']
col_names=[file+'_'+design for file,design in zip(metadata.sample_name,metadata.design)]
ambient_dist=pd.DataFrame(index=col_names,columns=col_names)
for idx1,col1 in enumerate(columns[:-1]):
    for idx2,col2 in enumerate(columns[idx1+1:]):
        diff=abs(ambient_df.iloc[:50,:][col1]-ambient_df.iloc[:50,:][col2]).sum()
        col_name1=col1.replace('mean_ambient_n_counts_','')
        col_name1=col_name1+'_'+metadata.query('sample_name==@col_name1')['design'].values[0]
        col_name2=col2.replace('mean_ambient_n_counts_','')
        col_name2=col_name2+'_'+metadata.query('sample_name==@col_name2')['design'].values[0]
        ambient_dist.at[col_name1,col_name2]=diff
        ambient_dist.at[col_name2,col_name1]=diff
ambient_dist.fillna(0,inplace=True)
sb.clustermap(ambient_dist)

# %% [markdown]
# #C: Ambient genes do not group samples from the same replicate together.

# %%
#### Proportion of ambience based on ambient threshold
#All calculations are based on removing ambient genes from single sample, except for the plot line 
#"N removed genes across samples" that represents N removed genes at threshold across all samples

# %%
# Calculate retained ambient proportion and sum(abs(ambient_mean_geneI-ambient_sample_geneI))
# for ambient gene removal thresholds. Ambient genes are removed per sample.
thresholds=list(1/np.logspace(1,18,num=300,base=2,dtype='int'))
ambient_diffs=pd.DataFrame(columns=adata_raw.obs.file.unique())
removed_genes=pd.DataFrame(columns=list(adata_raw.obs.file.unique())+['all'])
ambient_proportions=pd.DataFrame(columns=adata_raw.obs.file.unique())
for idx,threshold in enumerate(thresholds):
    ambient_df_sub=ambient_df[~(ambient_df>threshold).any(axis=1)]
    removed_genes.at[idx,'all']=adata_raw.shape[1]-ambient_df_sub.shape[0]
    for sample in adata_raw.obs.file.unique():
        ambient_df_sub=ambient_df[~(ambient_df['mean_ambient_n_counts_'+sample]>threshold)]
        removed_genes.at[idx,sample]=adata_raw.shape[1]-ambient_df_sub.shape[0]
        diff=abs(ambient_df_sub['mean_ambient_n_counts']-ambient_df_sub['mean_ambient_n_counts_'+sample]).sum()
        ambient_diffs.at[idx,sample]=diff
        ambient_proportions.at[idx,sample]=ambient_df_sub['mean_ambient_n_counts_'+sample].sum()

# %%
AMBIENT_THR=0.001

# %%
#Difference between mean and per sample ambient scores for retained genes
#and N removed genes across samples

# %%
# Difference to average ambient scores for retained genes at each threshold
rcParams['figure.figsize']= (10,5)
fig,ax=plt.subplots()
ax2 = ax.twinx()  
for sample in adata_raw.obs.file.unique():
    ax.plot(thresholds,ambient_diffs[sample])
ax.set_xscale('log')
ax.set_ylabel('Sum of abs (mean_ambient-sample_ambient)')
ax.set_xlabel('Ambient threshold')
ax2.plot(thresholds,removed_genes['all'],linestyle='dotted')
ax2.set_yscale('log')
ax2.set_ylabel('N removed genes across samples (dotted line)')
plt.axvline(AMBIENT_THR,c='k')

# %%
#Retained ambient proportion per sample
#and N removed genes across samples

# %%
# Retained ambience per sample vs threshold
rcParams['figure.figsize']= (10,5)
fig,ax=plt.subplots()
for sample in adata_raw.obs.file.unique():
    ax.plot(thresholds,ambient_proportions[sample])
ax.set_xscale('log')
ax.set_ylabel('Sum of retained ambient proportions')
ax.set_xlabel('Ambient threshold')
ax2 = ax.twinx()  
ax2.plot(thresholds,removed_genes['all'],linestyle='dotted')
ax2.set_yscale('log')
ax2.set_ylabel('N removed genes across samples (dotted line)')
plt.axvline(AMBIENT_THR,c='k')

# %%
#N removed genes per sample

# %%
# N removed genes per sample
rcParams['figure.figsize']= (10,5)
fig,ax=plt.subplots()
for sample in adata_raw.obs.file.unique():
    ax.plot(thresholds,removed_genes[sample],linestyle='dotted')
ax.set_xscale('log')
ax.set_ylabel('N removed genes')
ax.set_xlabel('Ambient threshold')
ax.set_yscale('log')
plt.axvline(AMBIENT_THR,c='k')

# %%
#Removed ambience proportion divided by removed genes per sample

# %%
# Comparison of removed ambience vs N of removed genes per sample
rcParams['figure.figsize']= (10,5)
fig,ax=plt.subplots()
for sample in adata_raw.obs.file.unique():
    any_removed=np.array(removed_genes[sample])>0
    ax.plot(np.array(thresholds)[any_removed],((1-ambient_proportions[sample])/removed_genes[sample])[any_removed])
ax.set_xscale('log')
ax.set_ylabel('Removed ambient proportion/N removed genes')
ax.set_xlabel('Ambient threshold')
plt.axvline(AMBIENT_THR,c='k')

# %%
# Save genes with scaled mean ambient expression at least > threshold in any sample - 
# use the same genes for each sample so that further prrocessing (cell type annotation, embedding) can be done jointly
# !!! Check on heatmap that all such genes were included in the filtered 20 genes
ambient_genes_selection=list(ambient_df[(ambient_df>AMBIENT_THR).any(axis=1)].index)

print('N selected ambient genes:',len(ambient_genes_selection))
if SAVE:
    pickle.dump( ambient_genes_selection, open( shared_folder+"ambient_genes_selection_extended.pkl", "wb" ) )

# %%
del adata_raw

# %% [markdown]
# ## QC (counts, genes, mt) + SPIKEIN

# %%
# Add other QC metrics

#adata.obs['n_counts'] = adata.X.sum(axis = 1)
#adata.obs['log_counts'] = np.log(adata.obs['n_counts'])
adata.obs['n_genes'] = (adata.X > 0).sum(axis = 1)

mt_gene_mask = np.flatnonzero([gene.startswith('mt-') for gene in adata.var_names])
# the `.A1` is only necessary as X is sparse (to transform to a dense array after summing)
adata.obs['mt_frac'] = np.sum(adata[:, mt_gene_mask].X, axis=1).A1/adata.obs['n_counts']

# %% [markdown]
# ### QC on UMAP 
# Data used for UMAP: Total count normalised data with log transformation and PC preprocessing. UMAP distances are based on correlation.

# %%
# Preprocess data for UMAP
adata_pp=adata.copy()
sc.pp.normalize_total(adata_pp, target_sum=1e6, exclude_highly_expressed=True)
sc.pp.log1p(adata_pp)

# %%
# Select number of PCs to use for UMAP
sc.pp.pca(adata_pp,n_comps=15,use_highly_variable =False)
sc.pl.pca_variance_ratio(adata_pp)

# %%
# Compute UMAP
sc.pp.neighbors(adata_pp, n_neighbors=15, n_pcs=7, metric='correlation')
sc.tl.umap(adata_pp)

# %%
# Plot UMAP
rcParams['figure.figsize']=(5,5)
sc.pl.umap(adata_pp, color=['n_counts','n_genes','mt_frac','file'],size=10)

# %% [markdown]
# #C: There is an evident high mt fraction subpopulation.

# %% [markdown]
# ### Remove spike in cell clusters

# %%
# Add previously generated annotation 
#adata_preannotated=h.open_h5ad("/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/salinno_project/rev4/maren/data_endo_final.h5ad",unique_id2=UID2)
preannotated=pd.read_table("/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE142465/preannotated/GSE142465_MouseLTI_CellAnnotation_final.tsv",index_col=0)

# %%
preannotated.columns

# %%
preannotated.iloc[0,:]

# %%
# Load metadata for the project
sample_dict=pd.read_excel('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/scRNA-seq_pancreas_metadata.xlsx',
             sheet_name='spikein_drug',index_col='metadata')['sample_name'].to_dict()

# %%
# Reindex previous info df
preannotated.index=[idx.split('-')[0]+'-1-'+sample_dict[preannotated.at[idx,'sample']] for idx in preannotated.index]
preannotated=preannotated.reindex( adata_pp.obs.index)

# %%
# Add pre-prepared cell type annotation to currently used dataset
# Add 'pre_' before each original annotation
for annotation in ['celltype','PredictedCell','celltype2']:
    annotation_new='pre_'+annotation
    # Data was sorted before - remove index as they do not match
    adata_pp.obs[annotation_new]=pd.Series(preannotated[annotation].values,dtype='category').values
    # Replace np.nan with na
    adata_pp.obs[annotation_new] = adata_pp.obs[annotation_new].cat.add_categories('NA')
    adata_pp.obs[annotation_new].fillna('NA', inplace =True) 
    # Remove unused categories
    adata_pp.obs[annotation_new].cat.remove_unused_categories(inplace=True)

# %% [markdown]
# Pre-annotated cell type count

# %%
# Count of cells per annotation
for annotation in ['celltype','PredictedCell','celltype2']:
    # For some reason the same line above does not remove all categories, thus it is done here again
    adata_pp.obs['pre_'+annotation].cat.remove_unused_categories(inplace=True)
    print('pre_'+annotation,':')
    print(adata_pp.obs['pre_'+annotation].value_counts())

# %%
# Set one annotation column to pre_cell_type to enable below code to work
adata_pp.obs.rename(columns={'pre_celltype':'pre_cell_type'}, inplace=True)

# %%
sc.pl.umap(adata_pp, color='pre_cell_type',size=10)

# %%
for si in ['SI_Human','SI_Mouse']:
    s=40
    adata_pp.obs['temp']=adata_pp.obs['pre_cell_type']==si
    sc.pl.umap(adata_pp,color='temp',size=40,title=si)
adata_pp.obs.drop('temp',axis=1,inplace=True)

# %%
res=0.3
sc.tl.leiden(adata_pp, resolution=res, key_added='leiden', directed=True, use_weights=True)

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata_pp, color=['leiden'] ,size=10, use_raw=False)

# %%
for cluster in adata_pp.obs['leiden'].unique():
    cluster_data=adata_pp.obs.query('leiden == @cluster')
    cluster_anno_n=cluster_data.pre_cell_type.value_counts(normalize=True)
    print('Cluster %s: human SI pratio %.2e, mouse SI ratio %.2e' %
          (cluster,cluster_anno_n.loc['SI_Human'],cluster_anno_n.loc['SI_Mouse']))

# %% [markdown]
# #C: Spike-ins are in clusters 13 and 14 and some other clusters. As this clustering does not seem to resolve spike ins well they will be removed latter.

# %% [markdown]
# ### QC - select thresholds

# %% [markdown]
# Check:
# - joint distribution of N genes, N counts, and mt fraction
# - distribution of metrics across samples

# %%
#Data quality summary plots
rcParams['figure.figsize']=(8,5)
p1 = sc.pl.scatter(adata, 'n_counts', 'n_genes', color='mt_frac', size=20)
sc.pl.violin(adata, ['n_counts'], groupby='file', size=1, log=True,rotation=90)
sc.pl.violin(adata, ['n_genes'], groupby='file', size=1, log=False,rotation=90)
sc.pl.violin(adata, ['mt_frac'], groupby='file', size=1, log=False,rotation=90)

# %% [markdown]
# #C: There are two samples with low n genes and n counts (but not high mt fraction).

# %%
p1 = sc.pl.scatter(adata[np.logical_and(adata.obs['n_genes']<4000, adata.obs['n_counts']<10000)], 'n_counts', 'n_genes', color='mt_frac', size=20,show=False)
p1.grid()
plt.show()

# %% [markdown]
# N counts:

# %%
rcParams['figure.figsize']=(20,5)
fig_ind=np.arange(131, 134)
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.6)

p3 = sb.distplot(adata.obs['n_counts'], 
                 kde=False, 
                 ax=fig.add_subplot(fig_ind[0]), bins=100)
#p3.set_xscale('log')
p4 = sb.distplot(adata.obs['n_counts'][adata.obs['n_counts']<10000], 
                 kde=False, bins=50, 
                 ax=fig.add_subplot(fig_ind[1]))
p4.set_yscale('log')
p5 = sb.distplot(adata.obs['n_counts'][adata.obs['n_counts']>50000], 
                 kde=False, bins=30, 
                 ax=fig.add_subplot(fig_ind[2]))
plt.show()

# %% [markdown]
# N genes:

# %%
rcParams['figure.figsize']=(20,5)
fig_ind=np.arange(131, 134)
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.6) #create a grid for subplots

p6 = sb.distplot(adata.obs['n_genes'], kde=False, bins=100, ax=fig.add_subplot(fig_ind[0]))

p7 = sb.distplot(adata.obs['n_genes'][adata.obs['n_genes']<500], 
                 kde=False, bins=30, ax=fig.add_subplot(fig_ind[1]))
p8 = sb.distplot(adata.obs['n_genes'][adata.obs['n_genes']>5000], 
                 kde=False, bins=30, ax=fig.add_subplot(fig_ind[2]))
plt.show()

# %% [markdown]
# MT fraction:

# %%
rcParams['figure.figsize']=(20,5)
fig_ind=np.arange(131, 133)
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.6)

p8 = sb.distplot(adata.obs['mt_frac'], kde=False, bins=60, ax=fig.add_subplot(fig_ind[0]))

p9 = sb.distplot(adata.obs['mt_frac'][adata.obs['mt_frac'].between(0.05, 0.4)], 
                 kde=False, bins=60, ax=fig.add_subplot(fig_ind[1]))
plt.show()


# %% [markdown]
# For cells that have too low/high metrics show location on UMAP.

# %%
def assign_high_low(adata,obs_col,low,high):
    """
    Add low/high annotation to each cell for a metric. 
    Add obs column specifying if cell has ok/low/high value (categories ordered in this order) and uns color map - 
    gray for ok, blue for low, red for high.
    New obs columns is named obs_col_filter and colourmap is named obs_col_filter_colors.
    :param adata: anndata object that contains column with metric to be filtered and to which filter result column
    and colur map are added
    :param obs_col: obs column on which to perform filtering
    :param low: low - cells that have obs_col value < low are assigned 'low'
    :param high: high - cells that have obs_col value > high are assigned 'high'
    """
    cell_type=[]
    for val in adata.obs[obs_col]:
        if val>high:
            cell_type.append('high')
        elif val<low:
            cell_type.append('low')
        else:
            cell_type.append('ok')
    adata.obs[obs_col+'_filter']=cell_type
    adata.obs[obs_col+'_filter']=adata.obs[obs_col+'_filter'].astype('category')
    # So that low and high are plotted on top
    adata.obs[obs_col+'_filter'].cat.reorder_categories(
        [category for category in ['ok','low','high'] if category in adata.obs[obs_col+'_filter'].cat.categories], inplace=True)
    type_col={'high':'#e62e0e','low':'#02c6ed','ok':'#a8a8a8'}
    col_list=[]
    for filter_type in adata.obs[obs_col+'_filter'].cat.categories:
        col_list.append(type_col[filter_type])
    adata.uns[obs_col+'_filter_colors']=col_list


# %%
param='n_counts'
rcParams['figure.figsize']=(5,5)
COUNTS_THR_MIN=5000
COUNTS_THR_MAX=70000
assign_high_low(adata=adata,obs_col=param,low=COUNTS_THR_MIN,high=COUNTS_THR_MAX)
print(adata.obs[param+'_filter'].value_counts())
adata_pp.obs[param+'_filter']=adata.obs[param+'_filter']
adata_pp.uns[param+'_filter_colors']=adata.uns[param+'_filter_colors']
sc.pl.umap(adata_pp, color=[param+'_filter'],size=20)

# %%
param='n_genes'
rcParams['figure.figsize']=(5,5)
GENES_THR_MIN=250
GENES_THR_MAX=7000
assign_high_low(adata=adata,obs_col=param,low=GENES_THR_MIN,high=GENES_THR_MAX)
print(adata.obs[param+'_filter'].value_counts())
adata_pp.obs[param+'_filter']=adata.obs[param+'_filter']
adata_pp.uns[param+'_filter_colors']=adata.uns[param+'_filter_colors']
sc.pl.umap(adata_pp, color=[param+'_filter'],size=20)

# %%
param='mt_frac'
MT_THR=0.2
rcParams['figure.figsize']=(5,5)
assign_high_low(adata=adata,obs_col=param,low=-1,high=MT_THR)
print(adata.obs[param+'_filter'].value_counts())
adata_pp.obs[param+'_filter']=adata.obs[param+'_filter']
adata_pp.uns[param+'_filter_colors']=adata.uns[param+'_filter_colors']
sc.pl.umap(adata_pp, color=[param+'_filter'],size=40)

# %% [markdown]
# #C: High n_counts will be filtered but high n_genes will not be as they seem to be closer to major population. Low n_counts will not be filtered as too many cells with otherwise ok metrics would be removed.

# %% [markdown]
# N cells in which a gene is expressed:

# %%
adata.var['n_cells']=(adata.X > 0).sum(axis = 0).T

# %%
rcParams['figure.figsize']=(20,5)
fig_ind=np.arange(131, 133)
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.6) #create a grid for subplots

p6 = sb.distplot(adata.var['n_cells'], kde=False, bins=60, ax=fig.add_subplot(fig_ind[0]))

p8 = sb.distplot(adata.var['n_cells'][adata.var['n_cells']<100], 
                 kde=False, bins=100, ax=fig.add_subplot(fig_ind[1]))
plt.show()
sb.violinplot(adata.var['n_cells'] )

# %% [markdown]
# Filter out genes and cells

# %%
# Filter cells according to identified QC thresholds:
print('Total number of cells: {:d}'.format(adata.n_obs))


sc.pp.filter_cells(adata, max_counts = COUNTS_THR_MAX)
print('Number of cells after max count filter: {:d}'.format(adata.n_obs))

adata = adata[adata.obs['mt_frac'] <= MT_THR]
print('Number of cells after MT filter: {:d}'.format(adata.n_obs))

sc.pp.filter_cells(adata, min_genes = GENES_THR_MIN)
print('Number of cells after min gene filter: {:d}'.format(adata.n_obs))

# %%
#Filter genes:
print('Total number of genes: {:d}'.format(adata.n_vars))

CELLS_THR_MIN=20
# Min 20 cells - filters out 0 count genes
sc.pp.filter_genes(adata, min_cells=CELLS_THR_MIN)
print('Number of genes after cell filter: {:d}'.format(adata.n_vars))

# %% [markdown]
# Subset genes so that ref genes are not removed - UNUSED - insetad of the above cell

# %% [raw]
# # Check which genes are present in reference data
# var_ref=h.open_h5ad(file='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/ref_combined/preprocessed/data_normalised.h5ad',
#                     unique_id2=UID2).raw.var_names

# %% [raw]
# #Filter genes:
# print('Total number of genes: {:d}'.format(adata.n_vars))
# CELLS_THR_MIN=20
# # Min 20 cells - filters out 0 count genes
# retain=sc.pp.filter_genes(adata, min_cells=CELLS_THR_MIN,inplace=False)[0]
# var_retained=adata.var_names[retain]
# print('Number of genes after cell filter: {:d}'.format(len(var_retained)))
#
# print('Retained in ref but not after filtering: {:d}'.format(len(set(var_ref)-set(var_retained))))
# # Remove only genes that are also absent from reference data
# var_ref_df=pd.DataFrame([True]*len(var_ref),index=var_ref).reindex(adata.var_names,fill_value=False)
# adata=adata[:,np.array([var_ref_df.values.ravel(),retain]).any(axis=0)]
# print('Number of genes after cell filter and retaining reference genes: {:d}'.format(adata.shape[1]))


# %% [markdown]
# ## Doublet filtering

# %% [markdown]
# Perform doublet filtering with Scrublet per sample.

# %%
adata.obs['doublet_score']=np.zeros(adata.shape[0])
adata.obs['doublet']=np.zeros(adata.shape[0])

# filtering/preprocessing parameters:
min_counts = 3
min_cells = 3
vscore_percentile = 85
n_pc = 30

# doublet detector parameters:
expected_doublet_rate = 0.05 
sim_doublet_ratio = 2
n_neighbors = None #If `None`, this is set to round(0.5 * sqrt(n_cells))

# Detect doublets per sample
for batch in adata.obs['file'].cat.categories:
    idx = adata.obs[adata.obs['file']==batch].index
    print(batch, 'N cells:',idx.shape[0])
    scrub = scr.Scrublet(counts_matrix = adata[idx,:].X,  
                     n_neighbors = n_neighbors,
                     sim_doublet_ratio = sim_doublet_ratio,
                     expected_doublet_rate = expected_doublet_rate)
    doublet_scores, predicted_doublets = scrub.scrub_doublets( 
                    min_counts = min_counts, 
                    min_cells = min_cells, 
                    n_prin_comps = n_pc,
                    use_approx_neighbors = True, 
                    get_doublet_neighbor_parents = False)
    adata.obs.loc[idx,'doublet_score'] = doublet_scores
    adata.obs.loc[idx,'doublet'] = predicted_doublets

# %% [markdown]
# Distribution of doublet scores

# %%
rcParams['figure.figsize']=(12,6)
fig,ax=plt.subplots(1,2)
sb.distplot(adata.obs['doublet_score'], bins=100, kde=False,ax=ax[0])
sb.distplot(adata[adata.obs.doublet_score>0.1].obs['doublet_score'], bins=60, kde=False,ax=ax[1])
plt.show()

rcParams['figure.figsize']=(15,7)
fig,ax=plt.subplots()
sc.pl.violin(adata, 'doublet_score',
                  groupby='file',
                  size=2, log=True, cut=0,ax=ax,show=False)
DOUBLET_THR=0.4
ax.axhline(DOUBLET_THR)
plt.show()

# %%
# Remove cells from adata_pp that were removed before from adata
adata_pp=adata_pp[adata.obs.index]

# Plot doublet score and filtering threshold on UMAP
rcParams['figure.figsize']=(6,6)
adata_pp.obs['doublet_score']=adata.obs['doublet_score']
sc.pl.umap(adata_pp, color=['doublet_score'],size=40)
param='doublet_score'
rcParams['figure.figsize']=(5,5)
assign_high_low(adata=adata,obs_col=param,low=-np.inf,high=DOUBLET_THR)
print(adata.obs[param+'_filter'].value_counts())
adata_pp.obs[param+'_filter']=adata.obs[param+'_filter']
adata_pp.uns[param+'_filter_colors']=adata.uns[param+'_filter_colors']
sc.pl.umap(adata_pp[adata_pp.obs[param+'_filter'].sort_values().index], color=[param+'_filter'],size=40)

# %%
print('Number of cells before doublet filter: {:d}'.format(adata.n_obs))
idx_filt = adata.obs['doublet_score']<=DOUBLET_THR

adata = adata[idx_filt].copy()
print('Number of cells after doublet filter: {:d}'.format(adata.n_obs))

# %% [markdown]
# ## Summary after QC

# %%
# Summary statistics per file/batch
df = adata.obs[['n_genes','n_counts','file']]
df_all = pd.DataFrame(index=df['file'].unique())

df_all['mean_genes']=df.groupby(by='file')['n_genes'].mean()
df_all['median_genes']=df.groupby(by='file')['n_genes'].median()
df_all['mean_counts']=df.groupby(by='file')['n_counts'].mean()
df_all['median_counts']=df.groupby(by='file')['n_counts'].median()
df_all['n_cells']=df['file'].value_counts()
df_all.astype('float').round(1)

# %% [markdown]
# #C: Two samples had low n_counts but after other filters they have comparable N cells to other samples and will thus be retained. 

# %%
# Check that all filters were used properly - the min/max values are as expected
print('N counts range:',round(adata.obs['n_counts'].min(),1),'-',round(adata.obs['n_counts'].max(),1))
print('N genes range:',adata.obs['n_genes'].min(),'-',adata.obs['n_genes'].max())
print('Mt fraction range:',"{:.3e}".format(adata.obs['mt_frac'].min()),'-',"{:.3e}".format(adata.obs['mt_frac'].max()))
print('Doublet score range:',"{:.3e}".format(adata.obs['doublet_score'].min()),'-',"{:.3e}".format(adata.obs['doublet_score'].max()))
print('N cellls expressing a gene range:',adata.var['n_cells'].min(),'-',adata.var['n_cells'].max())

# %% [markdown]
# ## Save QC data

# %%
# Save QC data
if SAVE:
    #adata.write(shared_folder+'data_QC.h5ad')
    h.save_h5ad(adata=adata,file=shared_folder+'data_QC.h5ad',unique_id2=UID2)
    #pickle.dump( adata, open( shared_folder+"data_QC.pkl", "wb" ) )

# %% [markdown]
# ## Normalisation and log-scaling

# %%
# Load QC data
#adata=sc.read_h5ad(shared_folder+'data_QC.h5ad')
#adata=pickle.load( open( shared_folder+"data_QC.pkl", "rb" ) )
adata=h.open_h5ad(file=shared_folder+'data_QC.h5ad',unique_id2=UID2)

# %%
# Remove ambient genes from analysis - required if joint normalisation is performed
print('Number of genes: {:d}'.format(adata.var.shape[0]))
ambient_genes=pickle.load( open( shared_folder+"ambient_genes_selection.pkl", "rb" ) )
# Save all genes to raw
adata.raw=adata.copy()
adata = adata[:,np.invert(np.in1d(adata.var_names, ambient_genes))].copy()
print('Number of genes after ambient removal: {:d}'.format(adata.var.shape[0]))

# %%
adata.layers['counts'] = adata.X.copy()

# %%
# Data for: clustering for scran normalization in clusters and visualisation of samples on UMAP
# Make new adata_pp object that also has removed unexpressed genes 
# The adata_pp is pre-processed with normalisation to N total counts, log transformation and PC dimeni
adata_pp=adata.copy()
sc.pp.normalize_total(adata_pp, target_sum=1e6, exclude_highly_expressed=True)
sc.pp.log1p(adata_pp)
sc.pp.pca(adata_pp, n_comps=15)
sc.pp.neighbors(adata_pp)

# %% [markdown]
# Cluster cells for scran normalisation

# %%
# Perform clustering
sc.tl.leiden(adata_pp, key_added='groups', resolution=1)
print('N clusters:',adata_pp.obs['groups'].unique().shape[0])

# %% [markdown]
# Compare samples on UMAP before scran normalisation to decide if it can be peroformed jointly or not.

# %%
# Calculate UMAP
sc.tl.umap(adata_pp)

# %%
# Plot UMAP
rcParams['figure.figsize']=(15,12)
sc.pl.umap(adata_pp, color=['file','groups'],size=60)

# %% [markdown]
# Joint Scran normalisation

# %%
#Preprocess variables for scran normalization
input_groups = adata_pp.obs['groups']
data_mat = adata.X.T

# %% magic_args="-i data_mat -i input_groups -o size_factors" language="R"
# # min.mean was increased compared to other studies not to get error in size factors (negative)
# size_factors =  calculateSumFactors(data_mat, clusters=input_groups, min.mean=0.15,BPPARAM=MulticoreParam(workers = 8))

# %% [markdown]
# Distribution of size factors

# %%
# Visualize the estimated size factors
adata.obs['size_factors'] = size_factors
adata_pp.obs['size_factors'] = size_factors

rcParams['figure.figsize']=(8,8)
sc.pl.scatter(adata, 'size_factors', 'n_counts', color='file')
sc.pl.scatter(adata, 'size_factors', 'n_genes', color='file')

#let us visualise how size factors differ across clusters
rcParams['figure.figsize']=(8,8)
#Use adata_pp here as it has obs 'group' - the n_genes and n_counts were copied from andata (counts/not normalised)
sc.pl.scatter(adata_pp, 'size_factors', 'n_counts', color='groups')
sc.pl.scatter(adata_pp, 'size_factors', 'n_genes', color='groups')

print('Distribution of size factors')
sb.distplot(size_factors, bins=50, kde=False)
plt.show()

# %%
# Scale data with size factors
adata.X /= adata.obs['size_factors'].values[:,None] # This reshapes the size-factors array
sc.pp.log1p(adata)
adata.X = np.asarray(adata.X)

# %%
del adata_pp

# %% [markdown]
# ## Highly variable genes

# %% [markdown]
# Compare Seurat and CellRanger HVGs.

# %%
##hvg_vars=['highly_variable', 'means', 'dispersions', 'dispersions_norm', 'highly_variable_nbatches', 'highly_variable_intersection']

## Seurat
#sc.pp.highly_variable_genes(adata, flavor='seurat', batch_key='file')
#n_hvg_seurat=np.sum(adata.var['highly_variable'])
#print('\n','Number of highly variable genes: {:d}'.format(n_hvg_seurat))
##hvg_seurat=adata.var[hvg_vars]
#rcParams['figure.figsize']=(10,5)
#sc.pl.highly_variable_genes(adata)

## Same number of genes in CellRanger
#sc.pp.highly_variable_genes(adata, flavor='cell_ranger', batch_key='file',n_top_genes =n_hvg_seurat)
#print('\n','Number of highly variable genes: {:d}'.format(np.sum(adata.var['highly_variable'])))
##hvg_cellranger=adata.var[hvg_vars]
#rcParams['figure.figsize']=(10,5)
#sc.pl.highly_variable_genes(adata)

# %% [markdown]
# #C: Decided for CellRanger method.

# %%
# Compute and plot HVG
sc.pp.highly_variable_genes(adata, flavor='cell_ranger', batch_key='file',n_top_genes =2000)
print('\n','Number of highly variable genes: {:d}'.format(np.sum(adata.var['highly_variable'])))
rcParams['figure.figsize']=(10,5)
sc.pl.highly_variable_genes(adata)

# %%
adata

# %% [markdown]
# ## Save normalised data

# %%
# Used due to problems with saving h5ad
#pickle.dump( adata, open( shared_folder+"data_normalised.pkl", "wb" ) )

# %%
if SAVE:
    #adata.write(shared_folder+"data_normalised.pkl")
    h.save_h5ad(adata=adata, file=shared_folder+"data_normalised.h5ad",unique_id2=UID2)

# %%
#adata_temp=adata.copy()
#adata=adata_temp.copy()

# %%
#adata_temp.write('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/salinno_project/rev4/scanpy_AnnData/data_processed_temp.h5ad')
