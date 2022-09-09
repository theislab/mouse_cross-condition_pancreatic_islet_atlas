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
import pandas as pd
import numpy as np

import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib import rcParams

import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/')
from importlib import reload  
import helper as h
reload(h)
import helper as h

# %%
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
# library(BiocParallel)

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/'

# %%
adata_rn=sc.read(path_data+'data_rawnorm_integrated_annotated.h5ad')

# %%
adata=sc.read(path_data+'data_integrated_analysed.h5ad')

# %%
# Add obs info to rn data
adata_rn.obs=adata.obs.copy()
# log-Normalised n_counts
adata_rn.obs['n_counts_log_norm']=adata_rn.X.sum(axis=1)

# %%
ct_col='cell_type_integrated_v1'

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(adata,color=[ct_col,'study'],s=20,wspace=0.3)

# %% [markdown]
# ## Counts distn across cell types and samples

# %% [markdown]
# Heatmap of n_counts (based on log-transformed normalised expression - normalisation per sample) as a mean across cell types of each sample.

# %%
# Counts DF for heatmap
counts_norm=[]
studies=[]
for study in adata.obs.study.unique():
    for sample in adata.obs.query('study==@study').file.unique():
        obs_sub=adata_rn.obs.query('study==@study & file==@sample')
        counts_norm_sample=obs_sub.groupby(ct_col)['n_counts_log_norm'].mean()
        counts_norm_sample.name=study+'_'+sample
        counts_norm.append(counts_norm_sample)
        studies.append(study)

counts_norm=pd.concat(counts_norm,axis=1)
counts_norm.fillna(0,inplace=True)

# %%
# Heatma anno
study_cmap=dict(zip(adata.obs.study.cat.categories,adata.uns['study_colors']))
ct_cmap=dict(zip(adata.obs[ct_col].cat.categories,adata.uns[ct_col+'_colors']))
row_anno=pd.Series(counts_norm.index.map(ct_cmap),index=counts_norm.index,name='cell type')
col_anno=pd.Series([study_cmap[study] for study in studies],
                   index=counts_norm.columns,name='study')

# %%
sb.clustermap(counts_norm,xticklabels=True,yticklabels=True, 
              row_cluster=False,col_cluster=False,
              row_colors=row_anno,col_colors=col_anno)

# %% [markdown]
# C: It seems that some studies/samples have much higher n_counts in log-transformed normalised data than others.

# %% [markdown]
# ## Integrated cl based size factors
# Since size factors from per-sample data are biased redo normalisation on integrated clusters.

# %%
scran_groups='leiden_r2'

# %%
sc.pl.umap(adata,color=scran_groups,s=20)

# %%
# Preprocess variables for scran normalization

# Scran groups
input_groups = adata.obs[scran_groups]

# get raw data
adata_raw=adata.raw.to_adata()
# Base normalisation of genes present in integrated datasets (e.g. expressed, filtered top ambient)
adata_raw=adata_raw[:,adata.var_names]
print('Adata raw:',adata_raw.shape)
# Transpose and convert to dense for scran
data_mat=adata_raw.X.T.todense()
del adata_raw
print('Data mat:',data_mat.shape)

# %% magic_args="-i data_mat -i input_groups -o size_factors" language="R"
# size_factors =  calculateSumFactors(data_mat, clusters=input_groups, min.mean=0.1,
#                                     BPPARAM=MulticoreParam(workers = 20))

# %%
# Add sf to adata
adata.obs['size_factors_integrated'] = size_factors

# %% [markdown]
# Compare size factors to n_genes and n_counts computed on data_mat

# %%
data_mat.sum(axis=0).reshape(-1,1).shape

# %%
# Add n counts and n genes info to adata for plotting
adata.obs['n_counts']=data_mat.sum(axis=0).reshape(-1,1)
adata.obs['n_genes']=(data_mat>0).sum(axis=0).reshape(-1,1)

# %%
# SF by sample
rcParams['figure.figsize']=(8,8)
sc.pl.scatter(adata, 'size_factors_integrated', 'n_counts', color='study_sample')
sc.pl.scatter(adata, 'size_factors_integrated', 'n_genes', color='study_sample')

# %%
# SF by study
rcParams['figure.figsize']=(8,8)
sc.pl.scatter(adata, 'size_factors_integrated', 'n_counts', color='study')
sc.pl.scatter(adata, 'size_factors_integrated', 'n_genes', color='study')

# %% [markdown]
# C: Embryo has a bit different size-factor to n_counts distn compared to other studies. This is likely due to different cell types present in embryo.

# %%
#let us visualise how size factors differ across clusters
rcParams['figure.figsize']=(8,8)
sc.pl.scatter(adata, 'size_factors_integrated', 'n_counts', color=scran_groups)
sc.pl.scatter(adata, 'size_factors_integrated', 'n_genes', color=scran_groups)

# %%
print('Distribution of size factors')
sb.distplot(size_factors, bins=500, kde=False)
plt.show()

# %% [markdown]
# #### Save size factors

# %%
h.update_adata(adata_new=adata,path=path_data+'data_integrated_analysed.h5ad',
               add=[('obs',True,'size_factors_integrated','size_factors_integrated')],
               rm=None,unique_id2=None,io_copy=False)

# %%
h.update_adata(adata_new=adata,path=path_data+'data_rawnorm_integrated_annotated.h5ad',
               add=[('obs',True,'size_factors_integrated','size_factors_integrated')],
               rm=None,unique_id2=None,io_copy=False)

# %% [markdown]
# ### New expression counts per cell type and sample

# %%
adata_rn2=h.get_rawnormalised(adata,sf_col='size_factors_integrated',use_log=True)

# %%
adata_rn2.obs['n_counts_log_norm']=adata_rn2.X.sum(axis=1)

# %%
# Counts DF for heatmap
counts_norm=[]
studies=[]
for study in adata.obs.study.unique():
    for sample in adata.obs.query('study==@study').file.unique():
        obs_sub=adata_rn2.obs.query('study==@study & file==@sample')
        counts_norm_sample=obs_sub.groupby(ct_col)['n_counts_log_norm'].mean()
        counts_norm_sample.name=study+'_'+sample
        counts_norm.append(counts_norm_sample)
        studies.append(study)

counts_norm=pd.concat(counts_norm,axis=1)
counts_norm.fillna(0,inplace=True)

# %%
# Heatma anno
study_cmap=dict(zip(adata.obs.study.cat.categories,adata.uns['study_colors']))
ct_cmap=dict(zip(adata.obs[ct_col].cat.categories,adata.uns[ct_col+'_colors']))
row_anno=pd.Series(counts_norm.index.map(ct_cmap),index=counts_norm.index,name='cell type')
col_anno=pd.Series([study_cmap[study] for study in studies],
                   index=counts_norm.columns,name='study')

# %%
sb.clustermap(counts_norm,xticklabels=True,yticklabels=True, 
              row_cluster=False,col_cluster=False,
              row_colors=row_anno,col_colors=col_anno)

# %% [markdown]
# Do the same on clusters

# %%
# Counts DF for heatmap
ct_col2='leiden_r2'
counts_norm=[]
studies=[]
for study in adata.obs.study.unique():
    for sample in adata.obs.query('study==@study').file.unique():
        obs_sub=adata_rn2.obs.query('study==@study & file==@sample')
        counts_norm_sample=obs_sub.groupby(ct_col2)['n_counts_log_norm'].mean()
        counts_norm_sample.name=study+'_'+sample
        counts_norm.append(counts_norm_sample)
        studies.append(study)

counts_norm=pd.concat(counts_norm,axis=1)
counts_norm.fillna(0,inplace=True)

# Heatma anno
study_cmap=dict(zip(adata.obs.study.cat.categories,adata.uns['study_colors']))
ct_cmap=dict(zip(adata.obs[ct_col2].cat.categories,adata.uns[ct_col2+'_colors']))
row_anno=pd.Series(counts_norm.index.map(ct_cmap),index=counts_norm.index,name='cell type')
col_anno=pd.Series([study_cmap[study] for study in studies],
                   index=counts_norm.columns,name='study')

sb.clustermap(counts_norm,xticklabels=True,yticklabels=True, 
              row_cluster=False,col_cluster=False,
              row_colors=row_anno,col_colors=col_anno)

# %% [markdown]
# C: Now there is more variation across cell types rather than samples/studies.

# %% [markdown]
# #### Save differently normalised layer in rawnormalised

# %%
adata_rn2.layers['X_sf_integrated']=adata_rn2.X
adata_rn2

# %%
h.update_adata(adata_new=adata_rn2,path=path_data+'data_rawnorm_integrated_annotated.h5ad',
               add=[('layers',True,'X_sf_integrated','X_sf_integrated')],
               rm=None,unique_id2=None,io_copy=False)

# %% [markdown]
# ## Add to beta data

# %%
# Load current rn beta data
adata_rn_b=sc.read(path_data+'data_rawnorm_integrated_analysed_beta_v1s1.h5ad')

# %%
# Load data that has size factors
adata=sc.read(path_data+'data_integrated_analysed.h5ad',backed='r')

# %%
# Make new beta data object
adata_rn_b_isf=adata_rn_b.copy()
adata_rn_b_isf.X=adata_rn[adata_rn_b_isf.obs_names,:].layers['X_sf_integrated']
adata_rn_b_isf.obs['size_factors_integrated']=adata[adata_rn_b_isf.obs_names,
                                                       :].obs['size_factors_integrated']

# %%
adata_rn_b_isf

# %%
# Save new beta data object
adata_rn_b_isf.write(path_data+'data_rawnorm_integrated_analysed_beta_v1s1_sfintegrated.h5ad')
