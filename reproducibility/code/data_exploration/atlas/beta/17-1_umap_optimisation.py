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
import pandas as pd
import numpy as np
import scanpy as sc

from matplotlib import rcParams
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/')
import helper as h
import importlib
importlib.reload(h)
import helper as h

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/'

# %%
adata_rn_b=sc.read(path_data+'data_rawnorm_integrated_analysed_beta_v1s1_sfintegrated.h5ad')


# %% [markdown]
# Test different parameters affecting UMAP embedding to find one on which cell clusters can be well distinguished by eye (e.g. not hidden behind each other, etc.).

# %%
random_indices=np.random.permutation(list(range(adata_rn_b.shape[0])))
for nn in [3,5,10,15,50,100]:
    sc.pp.neighbors(adata_rn_b, n_neighbors=nn, use_rep='X_integrated', 
                    key_added='neighbors_temp')
    for min_dist in [0,0.1,0.2,0.5,0.7]:
        for spread in [0.1,0.5,1,2,5]:
            param_str=' '.join(['nn',str(nn),'min_dist',str(min_dist),'spread',str(spread)])
            print(param_str)
            sc.tl.umap(adata_rn_b, min_dist=min_dist, spread=spread, 
                     neighbors_key='neighbors_temp')
            rcParams['figure.figsize']=(5,5)
            g=sc.pl.umap(adata_rn_b[random_indices,:],
                       color=['hc_gene_programs_parsed','leiden_r1.5_parsed'],
                       s=10, wspace=0.7,show=False, return_fig=True)
            g.suptitle(param_str)
            display(g)
            plt.close()


# %% [markdown]
# #### Selected
# Selected params: nn 15 min_dist 0 spread 1
#
# Compute selected UMAP, add to adata, and save.

# %%
# recalculate UMAp with selected params
sc.pp.neighbors(adata_rn_b, n_neighbors=15, use_rep='X_integrated', 
                    key_added='neighbors_temp')
sc.tl.umap(adata_rn_b, min_dist=0, spread=1, 
                     neighbors_key='neighbors_temp')

# %%
# replot
random_indices=np.random.permutation(list(range(adata_rn_b.shape[0])))
rcParams['figure.figsize']=(5,5)
sc.pl.umap(adata_rn_b[random_indices,:],
                       color=['hc_gene_programs_parsed','leiden_r1.5_parsed'],
                       s=10, wspace=0.7)

# %%
# Save
h.update_adata(
        adata_new=adata_rn_b, path=path_data+'data_rawnorm_integrated_analysed_beta_v1s1_sfintegrated.h5ad',
        io_copy=False,
        add=[('obsm',True,'X_umap','X_umap_opt')],
    rm=None)

# %%
