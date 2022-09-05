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
import pickle

import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import rcParams
import matplotlib.patches as mpatches

from sklearn.preprocessing import minmax_scale,scale


# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/'
path_de='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/de/'
path_de1=path_de+'de_diseased_T1_NODelim_meld/'
path_de2=path_de+'de_diseased_T2_VSGSTZ_meld_covarStudy/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'

# %%
# Saving figures
path_fig='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/figures/paper/'

# %%
# Load T1 and T2 DE results
summary_t1=pd.read_table(path_de1+'deDataClusters.tsv',index_col=0)
summary_t2=pd.read_table(path_de2+'deDataClusters.tsv',index_col=0)

# %% [markdown]
# Overlap between T1D and T2D DE clusters

# %%
# Obverlap between genes as ratio of the smaller group
overlap=pd.DataFrame(index=sorted(summary_t1.hc.dropna().unique()),
                     columns=sorted(summary_t2.hc.dropna().unique()))
overlap.index.name='NOD'
overlap.columns.name='db/db+mSTZ'
for cl1 in overlap.index:
    for cl2 in overlap.columns:
        g1=set(summary_t1.query('hc==@cl1').index)
        g2=set(summary_t2.query('hc==@cl2').index)
        g1g2=g1&g2
        o=len(g1g2)/min([len(g1),len(g2)])
        overlap.at[cl1,cl2]=o

        # Also print overlap
        if len(g1g2)>0:
            print('NOD %s and db/db+mSTZ %s'%(cl1,cl2))
            print(sorted(summary_t1.loc[g1g2,'gene_symbol'].to_list()))
overlap=overlap.astype(float)

# %% [markdown]
# Plot overlap

# %%
# heatmap size params
w_dend=1.3
nrow=overlap.shape[0]*0.3
ncol=overlap.shape[1]*0.26
w=ncol+w_dend
h=nrow+w_dend
# Heatmap
g=sb.clustermap(overlap,cmap='viridis',
                col_cluster=False,row_cluster=False,
              figsize=(h,w),
            dendrogram_ratio=(w_dend/h,w_dend/w),
                cbar_pos=(0.1,0.5,0.04,0.2))
g.ax_cbar.set_title('overlap',fontsize=10)   

#remove dendrogram
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)


# Save
plt.savefig(path_fig+'heatmap_beta_DEdiabetesNODelim-VSGSTZ_cloverlap.png',
            dpi=300,bbox_inches='tight')

# %%
