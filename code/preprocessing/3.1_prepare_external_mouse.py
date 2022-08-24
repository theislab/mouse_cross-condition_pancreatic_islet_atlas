# -*- coding: utf-8 -*-
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
import numpy as np
import pandas as pd
import glob
from scipy.sparse import csr_matrix
import gzip
from tempfile import TemporaryDirectory
import shutil

from scipy import sparse

import matplotlib.pyplot as plt

import mygene

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()
# # %load_ext rpy2.ipython

import rpy2.rinterface_lib.callbacks
import logging
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)

import sys  
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/')
import helper as h
from importlib import reload
reload(h)
import helper as h
from constants import SAVE

# %%
ro.r('library("scran")')
ro.r('library("BiocParallel")')

# %%
data_path='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'

# %% [markdown]
# ## GSE137909 - STRT-seq

# %%
dataset='GSE137909'
path_ds=data_path+dataset+'/GEO/'

# %%
x=pd.read_table(path_ds+'GSE137909_UMI.txt',index_col=0)

# %%
x

# %%
x_norm=pd.read_table(path_ds+'GSE137909_TM0.1M.txt',index_col=0)

# %%
x_norm

# %%
# Make sure index of x tables matches
if (x.index!=x_norm.index).any() or (x.columns!=x_norm.columns).any():
    raise ValueError('Tables not matching')

# %%
# Make expression adata
adata=sc.AnnData(csr_matrix(x.iloc[:,1:].T),var=pd.DataFrame(x.iloc[:,0]),
                obs=pd.DataFrame(index=x.iloc[:,1:].columns))
adata.var.columns=['gene_symbol']
adata.layers['raw']=adata.X.copy()
adata.layers['normalised_original']=csr_matrix(x_norm.iloc[:,1:].T)

# %% [markdown]
# Load metadata

# %%
obs1=pd.read_table(path_ds+'GSE137909_metadata.txt',index_col=1)

# %%
obs1

# %%
obs2=pd.read_table(path_ds+'GSE137909_series_matrix.txt',index_col=0,skiprows=48)

# %%
obs2

# %%
obs=pd.read_excel(path_ds+'1-s2.0-S2212877820300569-mmc1.xlsx',skiprows=1,index_col=0)

# %%
obs

# %% [markdown]
# C: The xlsx table should have all necesary metadata

# %%
# Parse obs
obs=obs[['Strain', 'Treatment', 'Insulin implants', 'Time after STZ',
       'Age at death', 'Batch', 'Putative Cell Type', 'Putative β-cell Group']].copy()
obs.columns=['strain','STZ','insulin_implant','time_after_STZ','age','batch',
             'cell_type_original','cell_subtype_original']

# %%
obs['cell_type_original'].value_counts(dropna=False)

# %% [markdown]
# C: All cells are annotated

# %%
# Parse obs values
obs['STZ']=obs['STZ'].apply(lambda x: True if x=='STZ' else False)
obs['insulin_implant']=obs['insulin_implant'].apply(lambda x: True if 'Yes' in x else False)
obs['time_after_STZ']=obs['time_after_STZ'].apply(
    lambda x: np.nan if x=='-' else x[1:]+' '+x[0].lower())
obs['age']=obs['age'].map({'2M':'2 m', '4M':'4 m','2M 12D':'2.4 m','2M 23D':'2.77 m','3M':'3 m',
                          '3M 12D':'3.4 m','2M 6D':'2.2 m','7M':'7 m','11M':'11 m',
                          'P3':'3 d','P12':'12 d','P21':'21 d'})
obs['cell_type']=obs['cell_type_original'].replace({
    'PP':'gamma','duct':'ductal','endothelium':'endothelial'})
obs['cell_subtype_original']=obs['cell_subtype_original'].apply(
    lambda x: np.nan if x=='-' else 'beta_'+str(x))
obs['donor']=pd.Series(obs.index).apply(lambda x: '_'.join(x.split('_')[:-1])).values
obs['sex']='male'
obs['disease']=obs['STZ'].map({True:'T2D',False:'healthy'})

# %%
obs

# %%
obs.drop_duplicates('donor')

# %% [markdown]
# Adata

# %%
adata.obs=obs.reindex(adata.obs_names)

# %%
# Log normalise
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# %%
adata

# %%
for col in adata.obs.columns:
    print(col)
    print(adata.obs[col].unique().tolist())

# %%
if SAVE:
    adata.write(path_ds+'adata.h5ad')

# %% [markdown]
# Prepare per-sample scran normalisation

# %%
# Load data
adata=sc.read(path_ds+'adata.h5ad')

# %%
# Use raw
adata.X=adata.layers['raw']

# %%
for sample, idx_sample in adata.obs.groupby('donor').groups.items():
    # Subset data
    adata_sub=adata[idx_sample,:].copy()
    print('Normalising:',sample,adata_sub.shape)
    # Faster on sparse matrices
    if not sparse.issparse(adata_sub.X): 
        adata_sub.X = sparse.csr_matrix(adata_sub.X)
    # Sort indices is necesary for conversion to R object 
    adata_sub.X.sort_indices()
    
    # Prepare clusters for scran
    adata_sub_pp=adata_sub.copy()
    sc.pp.normalize_total(adata_sub_pp, target_sum=1e6, exclude_highly_expressed=True)
    sc.pp.log1p(adata_sub_pp)
    sc.pp.pca(adata_sub_pp, n_comps=15)
    sc.pp.neighbors(adata_sub_pp)
    sc.tl.louvain(adata_sub_pp, key_added='groups', resolution=1)
    
    # Normalise
    ro.globalenv['data_mat'] = adata_sub.X.T.todense()
    ro.globalenv['input_groups'] = adata_sub_pp.obs['groups']
    try:
        size_factors = ro.r(f'calculateSumFactors(data_mat, clusters = input_groups, min.mean = 0.1, BPPARAM=MulticoreParam(workers = 16))')
    except:
        # Sometimes the above does not work so change parameter
        size_factors = ro.r(f'calculateSumFactors(data_mat, clusters = input_groups, min.mean = 0.2, BPPARAM=MulticoreParam(workers = 16))')
    adata.obs.loc[adata_sub.obs.index,'size_factors_sample'] = size_factors

del adata_sub
del adata_sub_pp

# %%
# Save parse anno and colors
if SAVE:
    h.update_adata(
            adata_new=adata, path=path_ds+'adata.h5ad',
            io_copy=False,
            add=[('obs',True,'size_factors_sample','size_factors_sample')],
        rm=None)

# %% [markdown]
# ## GSE83146 - SMARTer

# %%
dataset='GSE83146'
path_ds=data_path+dataset+'/GEO/'

# %%
x=pd.read_table(path_ds+'GSE83146_expr_tpm.txt',index_col=0)

# %%
x

# %% [markdown]
# C: The gene ids are entrez ids

# %%
mg = mygene.MyGeneInfo()
genemap = mg.querymany(x.index.to_list(), scopes='entrezgene', 
                       fields=['ensembl.gene','symbol'],  species='mouse')

# %%
genemap_df=[]
for g in genemap:
    g_parsed={'uid':g['query']}
    g_parsed['gene_symbol']=g['symbol'] if 'symbol' in g else np.nan
    # Genes with multiple EIDs have these as list
    if 'ensembl' in g:
        if isinstance(g['ensembl'],list):
            g_parsed['EID']=','.join([gs['gene'] for gs in g['ensembl']])
        else:
            g_parsed['EID']=g['ensembl']['gene']
    genemap_df.append(g_parsed)
genemap_df=pd.DataFrame(genemap_df)
genemap_df.index=genemap_df.uid
genemap_df.drop('uid',axis=1,inplace=True)

# %%
genemap_df

# %%
adata=sc.AnnData(X=csr_matrix(x.T),obs=pd.DataFrame(index=x.columns),
                 var=pd.DataFrame(index=x.index))

# %%
# Logtransform
adata.layers['normalised_original']=adata.X.copy()
sc.pp.log1p(adata)

# %%
# Add gene info
for col in genemap_df:
    adata.var[col]=genemap_df[col]

# %%
obs=pd.read_table(path_ds+'GSE83146_series_matrix.txt',index_col=0,skiprows=26)

# %%
obs

# %%
# Subset obs
obs=obs.T.iloc[:,[0,10,11]]

# %%
obs.columns=['geo_accession','age','sex']
obs['age']=obs['age'].apply(lambda x: x.split(': ')[1].replace('months','m'))
obs['sex']=obs['sex'].str.replace('Sex: ','')
obs['cell_type']='beta'
obs['disease']='healthy'

# %% [markdown]
# C: All cells were annotated as beta

# %%
obs

# %%
adata.obs=obs.reindex(adata.obs_names)

# %%
adata

# %%
for col in adata.obs.columns:
    if col!='geo_accession':
        print(col)
        print(adata.obs[col].unique().tolist())

# %%
if SAVE:
    adata.write(path_ds+'adata.h5ad')

# %%
