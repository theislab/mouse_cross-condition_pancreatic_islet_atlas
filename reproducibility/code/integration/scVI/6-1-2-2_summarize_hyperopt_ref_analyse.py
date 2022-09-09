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
# # Analyse results of hyperopt search

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import patsy
import pickle
import numpy as np
import scanpy as sc

from matplotlib import rcParams

import rpy2.rinterface_lib.callbacks
import logging
from rpy2.robjects import pandas2ri
import anndata2ri

import sys  
sys.path.insert(0, '/lustre/groups/ml01/workspace/karin.hrovatin/code/diabetes_analysis/')
import helper as h

rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)
pandas2ri.activate()
anndata2ri.activate()
# %load_ext rpy2.ipython

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/ref_combined/scVI/hyperopt/'
UID2='eval_scVI_ref'

# %%
folder_name='1599165129.871496_2000_mll/'

# %% [markdown]
# ## Run parameters analysis

# %%
# Read parameters
params_res=pd.read_table(path_data+folder_name+'trials_ref.tsv',sep='\t')

# %% [markdown]
# Top 10 runs

# %%
# Display top 10 runs - based on marginal_ll
params_res.sort_values('marginal_ll').iloc[:10,:]

# %% [markdown]
# Distribution of metric based on parameter values (not adjusted for other parameters).

# %%
# Boxplot metric vs param for each param
metric='marginal_ll'
for param in params_res.columns:
    # Remove non-parameters
    if param not in ['marginal_ll','n_params','run index']:
        plt.subplots()
        sb.boxplot(x=param,y=metric,data=params_res)

# %% [markdown]
# Note on hyperopt model selection: This relies on mll, which is most likely not of the same quality for evaluation of integration as scIB evaluation used elsewhere. However, for some of the below parameters that we decided to change from the default we also latter used scIB on other datasets and got consistently similar results (e.g. higher n_hidden).

# %% [markdown]
# Aproximately determine most important parameters with multiple regression (not very reliable due to possibly non-linear relationship). 

# %%
# Prepare data for linear regression
# Use intercept to get proper encoding of categorical (single column) but remove it by subsetting array
xs=["n_layers","n_hidden","n_latent","reconstruction_loss","dropout_rate","lr",'dispersion',"n_epochs"]
x_formula=' + '.join(xs)
y_formula = 'marginal_ll'
params_res_matrix=patsy.dmatrix(x_formula, params_res)
col_names=[col.replace('[','').replace(']','').replace('T.','_').replace('-','_') 
           for col in params_res_matrix.design_info.column_names[1:]]
x_formula=' + '.join(col_names)
params_res_matrix=pd.DataFrame(params_res_matrix[:,1:],columns=col_names)
# Add y
params_res_matrix[y_formula]=params_res[y_formula]

# %% magic_args="-i params_res_matrix -i x_formula -i y_formula" language="R"
# # Linear regression
# fit <- lm(as.formula(paste0(y_formula,' ~ ',x_formula)), data=params_res_matrix)
# print(summary(fit))
# layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page
# plot(fit)

# %% [markdown]
# #C: The relationship is not linear - questionable how reliable are the results of the regression.

# %% [markdown]
# ## Best model

# %% [markdown]
# ### Training history

# %%
trials = pickle.load( open( path_data+folder_name+'trials_ref', "rb" ) )

# %%
best_model_idx=params_res.sort_values('marginal_ll').iloc[0,:]['run index']
history=trials.trials[best_model_idx]['result']["history"]["elbo_test_set"]

# %% [markdown]
# Plot history. Set ylim top so that very very high losses are not shown.

# %%
plt.plot(range(len(history)),history)
# Use this to remove outliers
plt.ylim(bottom= min(history)-min(history)*0.001,top=np.quantile(history, 0.95))
plt.xlabel('epoch')
plt.ylabel('test loss')

# %% [markdown]
# ### Latent sapce

# %%
latent=h.open_h5ad(file=path_data+folder_name+'latent.h5ad',unique_id2=UID2)

# %%
# Compute random indices for adata cells so that they can be plotted in random order (since they are stored in batch order)
random_indices=np.random.permutation(list(range(latent.shape[0])))

# %%
rcParams['figure.figsize']= (9,8)
sc.pl.umap(latent[random_indices,:],color=['study_sample','study'],wspace=0.3,s=5)
sc.pl.umap(latent[random_indices,:],color=['cell_type','cell_type_multiplet'],wspace=0.8,s=5)

# %% [markdown]
# ### Embedding of only beta cells

# %% [markdown]
# Select beta populations

# %%
[ct for ct in latent.obs.cell_type_multiplet.unique() if 'beta' in ct]

# %%
selected_beta=['beta',
 'beta_proliferative',
 'beta_subpopulation_STZref',
 'beta_ins_low']

# %%
latent_beta=latent[latent.obs.cell_type.isin(selected_beta),:].copy()

# %%
sc.pp.neighbors(latent_beta,n_pcs=0)
sc.tl.umap(latent_beta)

# %%
random_indices_beta=np.random.permutation(list(range(latent_beta.shape[0])))

# %%
sc.pl.umap(latent_beta[random_indices_beta,:],color=['study_sample','study'],wspace=0.3,s=5)
sc.pl.umap(latent_beta[random_indices_beta,:],color=['cell_type_multiplet'],wspace=0.8,s=5)

# %%


# %%
