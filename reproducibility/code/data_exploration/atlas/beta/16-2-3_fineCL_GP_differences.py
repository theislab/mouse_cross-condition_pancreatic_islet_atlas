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
import anndata
import math

import pickle as pkl
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.patches import Patch
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import itertools

# %%
# Saving figures
path_fig='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/figures/paper/'


# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/'
path_gp='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/moransi/sfintegrated/'

# %%
adata_rn_b=sc.read(path_data+'data_rawnorm_integrated_analysed_beta_v1s1_sfintegrated.h5ad')
adata_rn_b.shape

# %%
# reload
genes_hc=pd.read_table(path_gp+'gene_hc_t'+str(2.4)+'.tsv',sep='\t',index_col=0)

# %% [markdown]
# ## GP scores across cells

# %%
# Compute GP scores 
gene_cl='hc'
adata_rn_b.obs.drop([col for col in adata_rn_b.obs.columns 
                     if 'gene_score_cluster_'+gene_cl in col],axis=1,inplace=True)
for ct in sorted(genes_hc[gene_cl].unique()):
    score_name='gene_score_cluster_'+gene_cl+str(ct)
    sc.tl.score_genes(adata_rn_b, 
                      gene_list=genes_hc.index[genes_hc[gene_cl]==ct], 
                     score_name=score_name, use_raw=False)
    adata_rn_b.obs[score_name+'_scaled']=minmax_scale(adata_rn_b.obs[score_name])

# %% [markdown]
# GP score distn across cells

# %%
cols=sorted(genes_hc.hc.unique())
fig,axs=plt.subplots(2,len(cols),
                    figsize=(2*len(cols),2*2),
                    sharex=True,sharey=False)
for j,gp in enumerate(cols):
    axs[0,j].hist(adata_rn_b.obs['gene_score_cluster_hc'+str(gp)+'_scaled'],
                  bins=50,density=True)
    axs[0,j].set_title(gp)
    axs[1,j].hist(adata_rn_b.obs['gene_score_cluster_hc'+str(gp)+'_scaled'],
                  bins=50,density=True)
    axs[1,j].set_yscale('log')
fig.suptitle('Distn of scaled gene program scores (columns)')
fig.tight_layout()

# %% [markdown]
# C: It seems taht a few GPs have outliers, thus redo norm by removing N cells with highest/lowest scores

# %% [markdown]
# Plot GP tails to find out N cells in high/low regions

# %%
# Plot bottom (top plot) and upper (bottom plot) tails of GP score distn for each GP
cols=sorted(genes_hc.hc.unique())
fig,axs=plt.subplots(2,len(cols),
                    figsize=(2*len(cols),2*2),
                    sharex=False,sharey=False)
for j,gp in enumerate(cols):
    axs[0,j].hist(adata_rn_b.obs['gene_score_cluster_hc'+str(gp)+'_scaled'
                                ].sort_values().head(100),
                  bins=20,density=True)
    axs[0,j].set_title(gp)
    #axs[0,j].set_yscale('log')
    axs[1,j].hist(adata_rn_b.obs['gene_score_cluster_hc'+str(gp)+'_scaled'
                                ].sort_values().tail(100),
                  bins=20,density=True)
    #axs[1,j].set_yscale('log')
fig.suptitle('Distn of scaled gene program scores (columns) for extreme cells')
fig.tight_layout()

# %% [markdown]
# C: removing top/bottom 20 cells before normalisation range calculation would  solve the issue of outlier effect

# %%
# rescale scores without outliers
# Gene cluster scores 
gene_cl='hc'
for ct in sorted(genes_hc[gene_cl].unique()):
    score_name='gene_score_cluster_'+gene_cl+str(ct)
    bottom=adata_rn_b.obs[score_name].nsmallest(20).max()
    top=adata_rn_b.obs[score_name].nlargest(20).min()
    scores_clip=adata_rn_b.obs[score_name][
        (adata_rn_b.obs[score_name]<top).values & (adata_rn_b.obs[score_name]>bottom).values]
    adata_rn_b.obs[score_name+'_scaled_out']=MinMaxScaler(clip=True).fit(
        scores_clip.values.reshape(-1,1)
        ).transform(adata_rn_b.obs[score_name].values.reshape(-1,1))


# %% [markdown]
# Replot re-normalised scores

# %%
cols=sorted(genes_hc.hc.unique())
fig,axs=plt.subplots(2,len(cols),
                    figsize=(2*len(cols),2*2),
                    sharex=True,sharey=False)
for j,gp in enumerate(cols):
    axs[0,j].hist(adata_rn_b.obs['gene_score_cluster_hc'+str(gp)+'_scaled_out'],
                  bins=50,density=True)
    axs[0,j].set_title(gp)
    axs[1,j].hist(adata_rn_b.obs['gene_score_cluster_hc'+str(gp)+'_scaled_out'],
                  bins=50,density=True)
    axs[1,j].set_yscale('log')
fig.suptitle('Distn of scaled gene program scores (columns)')
fig.tight_layout()

# %% [markdown]
# How are GP scores (cols) distn within cell clusters (rows)

# %%
rows=sorted(adata_rn_b.obs.hc_gene_programs.unique())
cols=sorted(genes_hc.hc.unique())
fig,axs=plt.subplots(len(rows),len(cols),
                    figsize=(2*len(cols),2*len(rows)),
                    sharex=True,sharey=True)
for i,cl in enumerate(rows):
    for j,gp in enumerate(cols):
        axs[i,j].hist(adata_rn_b.obs.query('hc_gene_programs==@cl'
                                          )['gene_score_cluster_hc'+str(gp)+'_scaled'],
                      bins=50,density=True)
        if i == 0:
            axs[i,j].set_title(gp)
        if j == 0:
            axs[i,j].set_ylabel(cl)
fig.suptitle('Distn of scaled gene program scores (columns) within cell subtypes (rows)')
fig.tight_layout()

# %% [markdown]
# C: Not all gp scores within clusters are normally distributed. 
#
# C: Most gene program distributions within cell clusters are unimodal (and not too strongly skewed) and can thus be well summarized by the mean. 
#

# %% [markdown]
# ## Compare means of scaled gp values
# Do not use effect size metrics (such as cllif's delta) as they look at differences between 2 groups and do not account for relative variability across all groups (e.g. how much does GP differ between two groups compared to across all groups - i.e. is variation relevant for interpretation). Thus, using such metrics, two gene programs that are relatively similar across two cell clusters compared to other cell clusters still get a high score if they are separated enough.
#
#
# Do not do comparisons with a statistical test as the p-vals are too often strongly significant due to very large number of cells in each cluster.
#
#
# Thus, we compare means of GP values scaled across cells. To make the range of GP values and thus GP differences more comparable across GPs we above excluded outliers in scaling as else they would affect the scaled GP score distn.

# %%
# Collect data for combined comparison plot
diffs_plot={}

# %% [markdown]
# ### Healthy vs diseased for T2D
# adult2 vs db/db+mSTZ for VSG and STZ datasets

# %% [markdown]
# #### Subset to non-treated
# When comparing adult2 and db/db+mSTZ from VSG and STZ subset to only healthy and diabetic samples - exclude treated ones.

# %%
# Compute GP differences between clusters
score_cols=[c for c in adata_rn_b.obs.columns if 
            c.startswith('gene_score_cluster_hc') and c.endswith('_scaled_out')]
diffs=[]
for study,(samples1,samples2) in [(
    'VSG',
    [['VSG_MUC13633_chow_WT','VSG_MUC13634_chow_WT'],
     [ 'VSG_MUC13639_sham_Lepr-/-','VSG_MUC13641_sham_Lepr-/-']]),
    ('STZ',
    [['STZ_G1_control'],
     ['STZ_G2_STZ']])]:
    cells1=adata_rn_b.obs.query('study==@study & study_sample_design in @samples1 & '+\
        'hc_gene_programs_parsed=="adult2"').index
    cells2=adata_rn_b.obs.query('study==@study & study_sample_design in @samples2 & '+\
        'hc_gene_programs_parsed=="db/db+mSTZ"').index
    print(study,'N cells cl1,2:',len(cells1),len(cells2))
    diff=pd.DataFrame(adata_rn_b.obs.loc[cells2,score_cols].mean()-\
                    adata_rn_b.obs.loc[cells1,score_cols].mean()).rename({0:'diff'},axis=1)
    diff['dataset']=adata_rn_b.obs.query('study==@study')['study_parsed'][0]
    diffs.append(diff)
diffs=pd.concat(diffs)
diffs['gp']=[i.replace('gene_score_cluster_hc','').replace('_scaled_out','') 
             for i in diffs.index]
diffs['dataset']=pd.Categorical(
    values=diffs['dataset'],ordered=True,
    categories=[c for c in adata_rn_b.uns['study_parsed_order'] 
                if c in diffs['dataset'].unique()])

# %%
# Plot GP diffs
fig,ax=plt.subplots(figsize=(6,3))
sb.barplot(x="gp", y="diff",  data=diffs,
           hue='dataset',dodge=True,ax=ax,
          palette=dict(zip(adata_rn_b.uns['study_parsed_order'],
                           adata_rn_b.uns['study_parsed_colors']))
          )
ax.set_ylabel('difference db/db+mSTZ - adult2')
ax.set_xlabel('gene program')
ax.grid(axis='y')
ax.set(facecolor = (0,0,0,0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# %% [markdown]
# #### Same as above, but use all samples from the datasets

# %%
# Compute GP differences between clusters
score_cols=[c for c in adata_rn_b.obs.columns if 
            c.startswith('gene_score_cluster_hc') and c.endswith('_scaled_out')]
diffs=[]
for study in ['VSG',"STZ"]:
    cells1=adata_rn_b.obs.query('study==@study & hc_gene_programs_parsed=="adult2"').index
    cells2=adata_rn_b.obs.query('study==@study & hc_gene_programs_parsed=="db/db+mSTZ"').index
    print(study,'N cells cl1,2:',len(cells1),len(cells2))
    diff=pd.DataFrame(adata_rn_b.obs.loc[cells2,score_cols].mean()-\
                    adata_rn_b.obs.loc[cells1,score_cols].mean()).rename({0:'diff'},axis=1)
    diff['dataset']=adata_rn_b.obs.query('study==@study')['study_parsed'][0]
    diffs.append(diff)
diffs=pd.concat(diffs)
diffs['gp']=[i.replace('gene_score_cluster_hc','').replace('_scaled_out','') 
             for i in diffs.index]
diffs['dataset']=pd.Categorical(
    values=diffs['dataset'],ordered=True,
    categories=[c for c in adata_rn_b.uns['study_parsed_order'] 
                if c in diffs['dataset'].unique()])

# %%
# plot GP diffs
fig,ax=plt.subplots(figsize=(6,3))
sb.barplot(x="gp", y="diff",  data=diffs,
           hue='dataset',dodge=True,ax=ax,
          palette=dict(zip(adata_rn_b.uns['study_parsed_order'],
                           adata_rn_b.uns['study_parsed_colors']))
          )
ax.set_ylabel('difference db/db+mSTZ - adult2')
ax.set_xlabel('gene program')
ax.grid(axis='y')
ax.set(facecolor = (0,0,0,0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(bbox_to_anchor=(0.8,0.95),title='dataset')

# %% [markdown]
# C: Excluding or including anti-diabetic treated samples that map in compared clusters as expected leads to similar results. - E.g. if cells map to the same cluster they have similar expression.

# %% [markdown]
# ### Healthy vs diseased for t1D and T2D on one plot
# adul2 vs db/db+mSTZ or NOD-D per dataset
# Similar as above, but now compare healthy cluster (adult2) to diseased cluster for all diabetes models.

# %%
# Compute GP differences between clusters
score_cols=[c for c in adata_rn_b.obs.columns if 
            c.startswith('gene_score_cluster_hc') and c.endswith('_scaled_out')]
diffs=[]
for study,cl2 in [('VSG','db/db+mSTZ'),('STZ','db/db+mSTZ'),('NOD_elimination','NOD-D')]:
    cells1=adata_rn_b.obs.query('study==@study & hc_gene_programs_parsed=="adult2"').index
    cells2=adata_rn_b.obs.query('study==@study & hc_gene_programs_parsed==@cl2').index
    print(study,'N cells cl1,2:',len(cells1),len(cells2))
    diff=pd.DataFrame(adata_rn_b.obs.loc[cells2,score_cols].mean()-\
                    adata_rn_b.obs.loc[cells1,score_cols].mean()).rename({0:'diff'},axis=1)
    diff['gp']=[i.replace('gene_score_cluster_hc','').replace('_scaled_out','') 
             for i in diff.index]
    diff['dataset']=adata_rn_b.obs.query('study==@study')['study_parsed'][0]
    
    # Compute significance
    diff['pval']=np.nan
    for gp in diff.gp:
        gp_name='gene_score_cluster_hc'+gp+'_scaled_out'
        diff.at[gp_name,'pval']=mannwhitneyu(
            adata_rn_b.obs.loc[cells2,gp_name].values,
            adata_rn_b.obs.loc[cells1,gp_name].values )[1]
    diff['padj']=multipletests(diff['pval'],method='fdr_bh')[1]
    
    diffs.append(diff)
    display(diff)
diffs=pd.concat(diffs)
diffs['dataset']=pd.Categorical(
    values=diffs['dataset'],ordered=True,
    categories=[c for c in adata_rn_b.uns['study_parsed_order'] 
                if c in diffs['dataset'].unique()])

# %%
# Save data for combined plot below
diffs_plot['db/db+mSTZ or NOD-D - adult2']=diffs

# %%
# Plot GP diffs
fig,ax=plt.subplots(figsize=(6,3))
sb.barplot(x="gp", y="diff",  data=diffs,
           hue='dataset',dodge=True,ax=ax,
          palette=dict(zip(adata_rn_b.uns['study_parsed_order'],
                           adata_rn_b.uns['study_parsed_colors']))
          )
ax.set_ylabel('difference db/db+mSTZ or NOD-D - adult2')
ax.set_xlabel('gene program')
ax.grid(axis='y')
ax.set(facecolor = (0,0,0,0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(bbox_to_anchor=(0.8,0.95),title='dataset')

# %% [markdown]
# ### Compare two diseased clusters in STZ dataset
# db/db+mSTZ vs mSTZ for STZ dataset

# %%
# Compute GP differences between clusters
score_cols=[c for c in adata_rn_b.obs.columns if 
            c.startswith('gene_score_cluster_hc') and c.endswith('_scaled_out')]
cells1=adata_rn_b.obs.query('study=="STZ" & hc_gene_programs_parsed=="db/db+mSTZ"').index
cells2=adata_rn_b.obs.query('study=="STZ" & hc_gene_programs_parsed=="mSTZ"').index
print(study,'N cells cl1,2:',len(cells1),len(cells2))
diffs=pd.DataFrame(adata_rn_b.obs.loc[cells2,score_cols].mean()-\
                adata_rn_b.obs.loc[cells1,score_cols].mean()).rename({0:'diff'},axis=1)
diffs['gp']=[i.replace('gene_score_cluster_hc','').replace('_scaled_out','') 
         for i in diffs.index]
diffs['dataset']='mSTZ'

# Compute significance
diffs['pval']=np.nan
for gp in diffs.gp:
    gp_name='gene_score_cluster_hc'+gp+'_scaled_out'
    diffs.at[gp_name,'pval']=mannwhitneyu(
        adata_rn_b.obs.loc[cells2,gp_name].values,
        adata_rn_b.obs.loc[cells1,gp_name].values )[1]
diffs['padj']=multipletests(diffs['pval'],method='fdr_bh')[1]
display(diffs)

# %%
# Save diffs for combined plot
diffs_plot['mSTZ - db/db+mSTZ']=diffs

# %%
# Plot diffs
fig,ax=plt.subplots(figsize=(6,3))
sb.barplot(x="gp", y="diff",  data=diffs,
           dodge=True,ax=ax,color='k',
          )
ax.set_ylabel('difference mSTZ - db/db+mSTZ')
ax.set_xlabel('gene program')
ax.grid(axis='y')
ax.set(facecolor = (0,0,0,0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# %% [markdown]
# ### Healthy vs intermediate cluster for all datasets with diabetes models
# adult2 vs D-inter. for VSG, STZ, and NOD_elim datasets

# %%
# Compute GP differences between clusters
score_cols=[c for c in adata_rn_b.obs.columns if 
            c.startswith('gene_score_cluster_hc') and c.endswith('_scaled_out')]
diffs=[]
for study in ['VSG',"STZ",'NOD_elimination']:
    cells1=adata_rn_b.obs.query('study==@study & hc_gene_programs_parsed=="adult2"').index
    cells2=adata_rn_b.obs.query('study==@study & hc_gene_programs_parsed=="D-inter."').index
    print(study,'N cells cl1,2:',len(cells1),len(cells2))
    diff=pd.DataFrame(adata_rn_b.obs.loc[cells2,score_cols].mean()-\
                    adata_rn_b.obs.loc[cells1,score_cols].mean()).rename({0:'diff'},axis=1)
    diff['gp']=[i.replace('gene_score_cluster_hc','').replace('_scaled_out','') 
             for i in diff.index]
    diff['dataset']=adata_rn_b.obs.query('study==@study')['study_parsed'][0]
    
    # Compute significance
    diff['pval']=np.nan
    for gp in diff.gp:
        gp_name='gene_score_cluster_hc'+gp+'_scaled_out'
        diff.at[gp_name,'pval']=mannwhitneyu(
            adata_rn_b.obs.loc[cells2,gp_name].values,
            adata_rn_b.obs.loc[cells1,gp_name].values )[1]
    diff['padj']=multipletests(diff['pval'],method='fdr_bh')[1]
    diffs.append(diff)
    display(diff)
diffs=pd.concat(diffs)
diffs['dataset']=pd.Categorical(
    values=diffs['dataset'],ordered=True,
    categories=[c for c in adata_rn_b.uns['study_parsed_order'] 
                if c in diffs['dataset'].unique()])

# %%
# Save diffs for combined plot
diffs_plot['D-inter. - adult2']=diffs

# %%
# plot diffs
fig,ax=plt.subplots(figsize=(6,3))
sb.barplot(x="gp", y="diff",  data=diffs,
           hue='dataset',dodge=True,ax=ax,
          palette=dict(zip(adata_rn_b.uns['study_parsed_order'],
                           adata_rn_b.uns['study_parsed_colors']))
          )
ax.set_ylabel('difference D-inter. - adult2')
ax.set_xlabel('gene program')
ax.grid(axis='y')
ax.set(facecolor = (0,0,0,0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(bbox_to_anchor=(0.8,0.95),title='dataset')

# %% [markdown]
# ### Combine all diffs into a single plot

# %%
nrow=len(diffs_plot)
fig,axs=plt.subplots(nrow,1,figsize=(7,2.5*nrow),
                     #sharex=True,
                     sharey=False)
palette={'STZ':'#FFC107','VSG':'#F1720A','NOD_elimination':'#13A1E8'}
palette={adata_rn_b.obs.query('study==@s')['study_parsed'][0]:c for s,c in palette.items()}
plt.subplots_adjust( hspace=0.1)
for idx,(diff_name,diff) in enumerate(diffs_plot.items()):
    ax=axs[idx]
    sb.barplot(x="gp", y="diff",  data=diff,
           hue='dataset',dodge=True,ax=ax,
          palette=palette)
    ax.set_ylabel(diff_name)
    if idx==(nrow-1):
        ax.set_xlabel('gene program')
    else:
        ax.set_xlabel('')
    ax.grid(axis='y')
    #ax.grid(axis='x')
    ax.set(facecolor = (0,0,0,0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if idx==0:
        ax.legend(bbox_to_anchor=(0.5,0.55),title='dataset')
    else:
        ax.get_legend().remove()
plt.savefig(path_fig+'barplot_gpdiff_disease_interm_stz_perDataset.png',
            dpi=300,bbox_inches='tight')
