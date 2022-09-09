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

# %% [markdown]
# # Extract hyperopt data
# Extract data from the hyperopt search to be thereafeter analysed in another notebook.

# %%
from autotune_advanced_notebook import Benchmarkable
import torch
import io
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib 
matplotlib.use('TkAgg')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from scvi.dataset.anndataset import AnnDatasetFromAnnData
import scanpy as sc
import numpy as np
import anndata

# %%
# Paths to expression data and scVI hyperopt
path_adata='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/ref_combined/preprocessed/'
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/ref_combined/scVI/hyperopt/'

# %% [markdown]
# ### Load data

# %%
# Hyperopt folder for which to prepare data
folder_name='1599165129.871496_2000_mll'
# Name used for Benchmarkable
hyperopt_run_name='ref_2000_mll'
# Hvg column name or 'all' to use all genes
hvg_name='highly_variable'

# %%
# Load hyperopt result (requires GPU)
hyperopt_res=Benchmarkable(global_path=path_data+folder_name, exp_key='ref', name=hyperopt_run_name)

# %% [markdown]
# ### Get parameters search results

# %%
# Extract all tested params and corresponding metrics - save as tsv into hyperopt folder
params_res=hyperopt_res.get_param_df()
params_res.to_csv(path_data+folder_name+'/trials_ref.tsv',sep='\t',index=False)

# %% [markdown]
# ### Get latent space of best model

# %%
# Load original data to transform to latent space
adata=sc.read_h5ad(path_adata+'data_normalised.h5ad')

# %%
# Prepare scVI data object
# Replace X with counts
data=adata.copy()
data.X=data.layers['counts']
del data.layers['counts']
del data.raw
# Select genes
if hvg_name != 'all':
    data=data[:,data.var[hvg_name]]
# Add batch info
batches=data.obs.study_sample.unique()
data.obs['batch_int']=data.obs.study_sample.replace(dict(zip(batches,range(len(batches)))))
# Make scVi data object
data=AnnDatasetFromAnnData(ad = data,batch_label='batch_int')

# %%
# Get latent space
posterior = hyperopt_res.trainer.create_posterior(hyperopt_res.trainer.model, data, indices=np.arange(len(data)))
latent, batch_indices, labels = posterior.sequential().get_latent()

# %%
# Store latent space as adata 
latent_adata = anndata.AnnData(X=latent)
# Add cell metadata
latent_adata.obs=adata.obs.copy()
latent_adata.obs_names=adata.obs_names.copy()

# %%
# Compute neighbours and UMAp on latent data
sc.pp.neighbors(latent_adata,n_pcs=0)
sc.tl.umap(latent_adata)

# %%
latent_adata.write(path_data+folder_name+'/latent.h5ad')

# %%

# %%

# %%
