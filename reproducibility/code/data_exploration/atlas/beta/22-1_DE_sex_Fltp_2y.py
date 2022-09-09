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

# %% tags=[]
import scanpy as sc
import pandas as pd
import diffxpy.api as de
import numpy as np
import patsy
import pickle
import matplotlib.pyplot as plt
import seaborn as sb
import upsetplot as usp
from matplotlib import rcParams
from diffxpy.testing.det import DifferentialExpressionTestWald
import matplotlib.cm as cm
import matplotlib as mpl
from sklearn.preprocessing import minmax_scale
from anndata import AnnData
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy import sparse
from diffxpy.testing.det import DifferentialExpressionTestWald

import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/data_exploration/')
import helper_diffxpy as hde
import importlib
importlib.reload(hde)
import helper_diffxpy as hde
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/')
import helper as h
import importlib
importlib.reload(h)
import helper as h


# %%
# Set Path variables to call when needed
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/'
path_full='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/'
path_de='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/de/de_sexaging_covarSample/''


# %%
# Load the needed data 
adata_b_rn=sc.read(path_data+'data_rawnorm_integrated_analysed_beta_v1s1_sfintegrated.h5ad')

# %% [markdown]
# ## Prepare data

# %% [markdown]
# Subset data


# %%
# Samples which should be analysed 
studies=['Fltp_2y']
samples=adata_b_rn.obs.query('study in @studies').study_sample.unique().tolist()
print('N samples to use',len(samples))
print(samples)

# %%
# Subset 
# Samples
adata_b_rn_sub=adata_b_rn[adata_b_rn.obs.study_sample.isin(samples),:].copy()
# Clusters
adata_b_rn_sub=adata_b_rn_sub[ ~adata_b_rn_sub.obs['leiden_r1.5'].isin(['19','20']),:].copy()
print(adata_b_rn_sub.shape)
print(adata_b_rn_sub.obs.study_sample_design.unique().tolist())
print(adata_b_rn_sub.obs['leiden_r1.5'].unique().tolist())

# %% [markdown]
# Plot data

# %%
# Compute a neighborhood graph of observations and the umap
sc.pp.neighbors(adata_b_rn_sub,n_pcs=0,use_rep='X_integrated')
sc.tl.umap(adata_b_rn_sub)

# %%
# Plot umap for adata.obs study and ins_score_scaled
rcParams['figure.figsize']=(6,6)
random_indices=np.random.permutation(list(range(adata_b_rn_sub.shape[0])))
sc.pl.umap(adata_b_rn_sub[random_indices,:],
           color=['study','ins_score_scaled'],s=20,sort_order=False,wspace=0.3)

# %%
# Plot umap for adata.obs study_sample_design, to see the different samples
rcParams['figure.figsize']=(6,6)
random_indices=np.random.permutation(list(range(adata_b_rn_sub.shape[0])))
sc.pl.umap(adata_b_rn_sub[random_indices,:],
           color=['study_sample_design'],s=20,sort_order=False,wspace=0.3)

# %% [markdown] tags=[]
# ## DE test

# %% [markdown]
# DE data

# %%
# Get raw expression data for DE testing, subsetting by cells
adata_b_raw_sub=sc.read( path_data+'data_integrated_analysed_beta_v1s1.h5ad'
    ).raw.to_adata()[ adata_b_rn_sub.obs_names,:].copy()
print(adata_b_raw_sub.shape)

# %%
# Add the new obs 'sex_female' to adata with female = 1 and male = 0
adata_b_rn_sub.obs['sex_num']=adata_b_rn_sub.obs.sex.replace(['female', 'male'], [1, 0])

# %%
# Add needed obs for DE
adata_b_raw_sub.obs=pd.concat( [adata_b_rn_sub.obs['size_factors_integrated'], 
                                adata_b_rn_sub.obs['sex'], 
                                adata_b_rn_sub.obs['age'],
                                adata_b_rn_sub.obs['sex_num'],
                                adata_b_raw_sub.obs['file']],axis=1)

# %% [markdown]
# Expression filtering


# %%
# Set the minimun cell ratio:
min_cells_ratio=0.05

# %%
# prepare adata
# Data for testing
adata=adata_b_raw_sub.copy()
# Select expressed genes in partition
min_cells=adata.shape[0]*min_cells_ratio
print('Min cells: %.1f'% min_cells)
expressed=np.array((adata.X.todense()!=0).sum(axis=0)>=min_cells)[0]
adata=adata[:,expressed]
print('Data after expression filtering:',adata.shape)


# %% [markdown]
# Ambient removal

# %%
# Remove all ambient genes (from all studies)
ambient_nobeta=pickle.load(open('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/ambient/ambient_nonbeta.pkl'
    ,'rb'))['ambient_nonbeta']
print('N ambient genes:',len(ambient_nobeta))
adata=adata[:,[g for g in adata.var_names 
                                   if g not in ambient_nobeta]]
print('Data after ambient filtering:',adata.shape)

# %% [markdown] tags=[]
# Design-Matrix

# %%
# Create a design Matrix

dmat_loc=pd.DataFrame(index=adata.obs_names)
dmat_loc['Intercept']=1

# process of interest
process='sex_num'
dmat_loc[process]=adata.obs[process]

# Add files, these are the samples
# 3 samples, one is in intercept
dmat_loc['MUC13975']=adata_b_rn_sub.obs.file.replace(['MUC13974', 'MUC13975', 'MUC13976'], [0, 1, 0])
dmat_loc['MUC13976']=adata_b_rn_sub.obs.file.replace(['MUC13974', 'MUC13975', 'MUC13976'], [0, 0, 1])
    
dmat_loc.fillna(0,inplace=True)
dmat_loc=dmat_loc.astype('float')

print('dmat_loc')
display(dmat_loc)

dmat_scale=pd.DataFrame(dmat_loc['Intercept'])
constraints_scale=None
constraints_loc=None

sex_female=process

# %%
# Create dense adata, so that the Waldtest can work
if sparse.issparse(adata.X):
        new_adata = sc.AnnData(X=adata.X.A, obs=adata.obs.copy(deep=True), var=adata.var.copy(deep=True), uns=adata.uns.copy(), obsm=adata.obsm.copy())

# %%
# Perform the Wald test
if True:
    # Compute result
    result=de.test.wald(
        data=new_adata,
        coef_to_test=sex_female,
        dmat_loc=dmat_loc,
        dmat_scale=dmat_scale,
        # Use integarted sf
        size_factors=new_adata.obs.size_factors_integrated,
        )

    # Add design info to result
    result.dmat_loc=dmat_loc
    result.dmat_scale=dmat_scale
    result.coef_loc_totest_name=sex_female

    # save result
    pickle.dump(result, 
        open(path_de+'maleFemale_Fltp_2y_'+str(min_cells_ratio)+'.pkl', 
                     "wb" ))

# %% [markdown] tags=[]
# ## Analyse DE data


# %%
# Loada data
result=pickle.load(
    open(path_de+'maleFemale_Fltp_2y_'+str(min_cells_ratio)+'.pkl','rb'))

summary=result.summary()
summary.index=summary.gene

# %% [markdown]
# ### DE result vs gene characteristics

# %%
# Calculate empirical_log2fc
c2=adata_b_rn_sub.obs_names[adata_b_rn_sub.obs['sex_num']==1]
c1=adata_b_rn_sub.obs_names[adata_b_rn_sub.obs['sex_num']==0]
result.empirical_log2fc=pd.Series(np.asarray(np.log2(
     (adata_b_rn_sub[c2,summary.gene].X/
          adata_b_rn_sub[c2,summary.gene].obs['size_factors_integrated'].values.reshape(-1,1)).mean(axis=0)/
     (adata_b_rn_sub[c1,summary.gene].X/
          adata_b_rn_sub[c1,summary.gene].obs['size_factors_integrated'].values.reshape(-1,1)).mean(axis=0)
    )).ravel(), index=summary.gene).replace(np.inf,np.nan)

# %%
# calculate n_cells and then calculate mean_expr_in_expr_cells
n_cells=pd.Series(np.asarray((adata_b_rn_sub.X.todense()!=0).sum(axis=0)).ravel(),
                  index=adata_b_rn_sub.var_names)
mean_expr_in_expr_cells=pd.Series(np.asarray(
    adata_b_rn_sub[result.dmat_loc.index].X.sum(axis=0)
          ).ravel(),index=adata_b_rn_sub.var_names)/n_cells

# %%
# Plot diagramms for coef sd over coef mle,  fitted log2fc over empirical log3fc, 
# n cells over coef mle,  mean expr in gene_epxr cells over coef mle,  
# -log10(qval+10^-20) over coef mle,
# to see if we need to filter the results/summary for to small coef sd

fig,ax=plt.subplots(1,5,figsize=(20,4))
plt.subplots_adjust(wspace=0.4)

ax[0].scatter(summary.coef_mle,summary.coef_sd,c=-np.log10(summary.qval+10**-20))
ax[0].set_xlabel('coef mle')
ax[0].set_ylabel('coef sd')
ax[0].set_yscale('log')
ax[0].set_xscale('symlog')
ax[0].set_title('Colored by -log10(qval+10^-20)')

ax[4].scatter(summary.coef_mle,-np.log10(summary.qval+10**(-20)))
ax[4].set_xlabel('coef mle')
ax[4].set_ylabel('-log10(qval+10^-20)')
ax[4].set_xscale('symlog')

if 'empirical_log2fc' in dir(result):
    ax[1].set_title('(coloured by log10(coef_sd))')
    # genes with nan empirical lfc will not be shown
    ax[1].scatter(x=result.empirical_log2fc.values,y=summary.log2fc,
                c=np.log10(summary.coef_sd))
    ax[1].set_xlabel('empirical log2fc')
    ax[1].set_ylabel('fitted log2fc')
    #ax[1].set_yscale('symlog')
    ax[1].axhline(10)
    ax[1].axhline(-10)

ax[2].scatter(summary.coef_mle,n_cells[summary.gene])
ax[2].set_xlabel('coef mle')
ax[2].set_ylabel('n cells')
ax[2].set_xscale('symlog')

ax[3].scatter(summary.coef_mle,mean_expr_in_expr_cells[summary.gene])
ax[3].set_xlabel('coef mle')
ax[3].set_ylabel('mean expression in gene-expressing cells')
ax[3].set_xscale('symlog')
ax[3].set_yscale('log')

display(fig)
plt.close()
print('Min coef_sd:',summary.coef_sd.min())

# %% [markdown]
# C: Some genes could not be fitted - seen from the extremely small coef SD (technical issue).

# %% tags=[]
# Filter result/summary to remove those with too small sd
if 'empirical_log2fc' in dir(result):
    result.empirical_log2fc=result.empirical_log2fc[
        summary.coef_sd>2.2227587494850775e-162]
summary = summary[summary.coef_sd>2.2227587494850775e-162]

# %% [markdown]
# Replot result vs gene characteristics after filtering out genes that could not be fitted.

# %% tags=[]
# Plot 5 different diagramms for, coef sd over coef mle,  fitted log2fc over empirical log3fc, n cells over coef mle,  mean expr in gene_epxr cells over coef mle,  -log10(qval+10^-20) over coef mle,
# to see if the filtering was successfull and we can advance 

fig,ax=plt.subplots(1,5,figsize=(20,4))
plt.subplots_adjust(wspace=0.4)

ax[0].scatter(summary.coef_mle,summary.coef_sd,c=-np.log10(summary.qval+10**-20))
ax[0].set_xlabel('coef mle')
ax[0].set_ylabel('coef sd')
ax[0].set_yscale('log')
ax[0].set_xscale('symlog')
ax[0].set_title('Colored by -log10(qval+10^-20)')

ax[4].scatter(summary.coef_mle,-np.log10(summary.qval+10**(-20)))
ax[4].set_xlabel('coef mle')
ax[4].set_ylabel('-log10(qval+10^-20)')
ax[4].set_xscale('symlog')

if 'empirical_log2fc' in dir(result):
    ax[1].set_title('(coloured by log10(coef_sd))')
    # genes with nan empirical lfc will not be shown
    ax[1].scatter(x=result.empirical_log2fc.values,y=summary.log2fc,
                c=np.log10(summary.coef_sd))
    ax[1].set_xlabel('empirical log2fc')
    ax[1].set_ylabel('fitted log2fc')
    #ax[1].set_yscale('symlog')
    ax[1].axhline(10)
    ax[1].axhline(-10)

ax[2].scatter(summary.coef_mle,n_cells[summary.gene])
ax[2].set_xlabel('coef mle')
ax[2].set_ylabel('n cells')
ax[2].set_xscale('symlog')

ax[3].scatter(summary.coef_mle,mean_expr_in_expr_cells[summary.gene])
ax[3].set_xlabel('coef mle')
ax[3].set_ylabel('mean expression in gene-expressing cells')
ax[3].set_xscale('symlog')
ax[3].set_yscale('log')

display(fig)
plt.close()
print('Min coef_sd:',summary.coef_sd.min())

# %% [markdown] tags=[]
# Add gene symbols as index

# %%
# give summary an index with the gene_symbols
summary['EID']=summary.index
summary.index=adata_b_rn_sub.var.loc[summary.index,'gene_symbol']

# %%
# Save summary table for sharing
if True:
    summary.to_csv(
        path_de+'maleFemale_Fltp_2y_'+str(min_cells_ratio)+'_summary_sdFiltered.tsv',
    sep='\t',index=False)

# %% [markdown] tags=[]
# ### Check top genes

# %%
# The top/bottom DE genes
nAnz = 10
# Get the top 10 Genes, and save then in summary_abs_top10
summary_abs_topnAnz = summary.coef_mle.sort_values(ascending=False)[0:nAnz]
# Get the bottom 10 Genes, and save then in summary_abs_bottom10
summary_abs_bottomnAnz = summary.coef_mle.sort_values(ascending=True)[0:nAnz]

# Get the Top/bottom 10 Genes in a list
genes_top = summary_abs_topnAnz.index.tolist()
genes_bottom = summary_abs_bottomnAnz.index.tolist()


# %%
# Print out the top DE genes to see all data for them:
summary.loc[genes_top,:]

# %%
# Plot genes with  strongest lfc
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata_b_rn_sub,
           color=summary.sort_values(
               ['coef_mle'],ascending=[False]).gene[:nAnz],
           s=20)

# %%
# Print out the bottom DE genes to see all data for them:
summary.loc[genes_bottom,:]

# %%
# Plot genes with  strongest negative lfc
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata_b_rn_sub,
           color=summary.sort_values(['coef_mle']).gene[:nAnz],
           s=20)

# %% [markdown] tags=[]
# ### N DE genes at different filtering thresholds

# %%
# Try different thresholds for FDR and ALFC to see how many genes are retained
for fdr,lfc in [(0.05,1),(0.05,0.5),(0.05,0.3),
                (0.025,1),(0.025,0.5), (0.025,0.3)]:
    print('fdr %.2e, abs(lfc) > %.1f up %i down %i'%
          (fdr,lfc,summary.query('qval<@fdr & log2fc>@lfc').shape[0],
           summary.query('qval<@fdr & log2fc<-@lfc').shape[0]))

# %%
# Plot the distribution for log2fc
plt.hist(summary.log2fc,bins=100)
plt.xlabel('log2fc')
