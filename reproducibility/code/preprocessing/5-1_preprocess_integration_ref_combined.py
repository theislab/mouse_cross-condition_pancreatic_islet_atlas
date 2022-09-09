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
# # Prepare combined ref data for integration
# Prepare a smaller data subset for integaryion in order to test out different integratiuon parameters.

# %%
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ann
import pickle
from scipy import sparse
import scIB as scib

import seaborn as sb
from matplotlib import rcParams
import matplotlib.pyplot as plt
from venn import venn

import sys  
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/mm_pancreas_atlas_rep/code/')
import helper as h
from constants import SAVE

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import anndata2ri
pandas2ri.activate()
anndata2ri.activate()
# %load_ext rpy2.ipython

import rpy2.rinterface_lib.callbacks
import logging
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)

# %% language="R"
# library(scran)
# library(BiocParallel)

# %%
#Path for saving results
shared_folder='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/ref_combined/preprocessed/'
#Unique ID2 for reading/writing h5ad files with helper function
UID2='combined_ref'

# %%
# List of datasets informations: (study_name, path)
data=[('Fltp_P16','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/salinno_project/rev4/'),
      ('Fltp_adult','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islet_fltp_headtail/rev4/'),
      ('Fltp_2y','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islets_aged_fltp_iCre/rev6/'),
      ('VSG','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/VSG_PF_WT_cohort/rev7/ref/'),
      ('STZ','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islet_glpest_lickert/rev7/ref/')]

# %% [markdown]
# ## Load data

# %%
# Load data from individual studies, using raw counts and adding additional metadata
adatas=[]
for study,path in data:
    print(study)
    #Load data
    adata=h.open_h5ad(file=path+'data_annotated.h5ad',unique_id2=UID2).raw.to_adata()
    print(adata.shape)
    metadata=pd.read_excel('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/scRNA-seq_pancreas_metadata.xlsx',
                          sheet_name=study)
    # Add metadata 
    samples=adata.obs.file.unique()
    for col in ['sex','age','design','strain','tissue','technique']:
        #Do not add metadata that is already in adata - otherwise sex would be overwritten where it was annotated
        if col in metadata.columns and col not in adata.obs.columns:
            value_map={sample:metadata.query('sample_name =="'+sample+'"')[col].values[0] for sample in samples}
            adata.obs[col]=adata.obs.file.map(value_map)
            
    adatas.append(adata)
    
# Combine datasets    
adata = ann.AnnData.concatenate( *adatas,  batch_key = 'temp_batch', batch_categories = [d[0] for d in data ]).copy()

# %%
# Rename index of studies that have single sample and do not contain sample info (STZ)
index=[]
for idx,row in adata.obs.iterrows():
    if '-'+row['file']+'-'+row['study'] in idx:
        index.append(idx)
    else:
        index.append(idx.replace('-'+row['study'],'-'+row['file']+'-'+row['study']))

adata.obs_names=index
if adata.raw is not None:
    adata_raw=adata.raw.to_adata()
    adata_raw.obs_names=index
    adata.raw=adata_raw
    del adata_raw

# %%
#Retain only relevant obs columns
adata.obs=adata.obs[['file', 'n_counts', 'n_genes', 'mt_frac', 'doublet_score', 'size_factors',
           'pre_cell_type', 'cell_type','cell_type_multiplet',
           'S_score', 'G2M_score', 'phase', 'phase_cyclone', 's_cyclone', 'g2m_cyclone', 'g1_cyclone', 
           'x_score', 'y_score', 'sex', 
           'age', 'design', 'strain', 'tissue', 'technique', 'study']]

#Rename per study size factors
adata.obs.rename(columns={'size_factors':'size_factors_study'}, inplace=True)

# Remove obsm
del adata.obsm

# %%
# Add merged column for study and sample: study_sample
adata.obs['study_sample']=[study+'_'+sample for study, sample in zip(adata.obs.study,adata.obs.file)]

# %%
adata

# %% [markdown]
# List cell types to decide if they need to be unified

# %%
# Check if cell types need to be unified
adata.obs.cell_type.unique()

# %% [markdown]
# ## Ambient genes

# %%
# Merge all ambient scores - from DFs of top ambient genes of individual studies
ambient_df=[]
for study,path in data:
    ambient_study=pd.read_table(path+"ambient_genes_topN_scores.tsv",index_col=0).drop('mean_ambient_n_counts',axis=1)
    ambient_study.columns=[col.replace('mean_ambient_n_counts',study) for col in ambient_study.columns]
    ambient_df.append(ambient_study)
ambient_df=pd.concat(ambient_df, axis=1)

# %%
# Add median ambeint score for sorting
ambient_df['median']=ambient_df.median(axis=1)
ambient_df.sort_values('median',ascending=False,inplace=True)

# %%
rcParams['figure.figsize']= (30,35)
sb.heatmap(ambient_df,annot=True,fmt='.2e')

# %%
# Save genes with scaled mean ambient expression at least > 0.004 in any sample 
# (this still includes last gene ambient across studies) - 
# use the same genes for each sample so that further prrocessing (cell type annotation, embedding) can be done jointly
ambient_genes_selection=list(ambient_df[(ambient_df>0.004).any(axis=1)].index)

print('Selected ambient genes:',ambient_genes_selection)
pickle.dump( ambient_genes_selection, open( shared_folder+"ambient_genes_selection.pkl", "wb" ) )

# %%
# Remove ambient genes from analysis 
print('Number of genes: {:d}'.format(adata.var.shape[0]))
ambient_genes=pickle.load( open( shared_folder+"ambient_genes_selection.pkl", "rb" ) )
# Save all genes to raw
adata.raw=adata.copy()
adata = adata[:,np.invert(np.in1d(adata.var_names, ambient_genes))].copy()
print('Number of genes after ambient removal: {:d}'.format(adata.var.shape[0]))

# %% [markdown]
# ### Save merged data without ambient genes

# %%
if SAVE:
    h.save_h5ad(adata=adata,file=shared_folder+'data_noambient.h5ad',unique_id2=UID2)

# %% [markdown]
# ## Normalisation and log-scaling
#
# Remove ambient genes (move counts to raw).
#
# Scale per sample.

# %%
adata=h.open_h5ad(file=shared_folder+'data_noambient.h5ad',unique_id2=UID2)

# %%
adata.layers['counts'] = adata.X.copy()

# %%
for sample, idx_sample in adata.obs.groupby(['study','file']).groups.items():
    print(sample)
    # Subset data
    adata_sub=adata[idx_sample,:].copy()
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
    ro.globalenv['data_mat'] = adata_sub.X.T
    ro.globalenv['input_groups'] = adata_sub_pp.obs['groups']
    size_factors = ro.r(f'calculateSumFactors(data_mat, clusters = input_groups, min.mean = 0.1, BPPARAM=MulticoreParam(workers = 8))')
    adata.obs.loc[adata_sub.obs.index,'size_factors_sample'] = size_factors

del adata_sub
del adata_sub_pp

# %%
rcParams['figure.figsize']=(6,5)
sc.pl.scatter(adata, 'size_factors_sample', 'n_counts', color='file',size=30)
sc.pl.scatter(adata, 'size_factors_sample', 'n_genes', color='file',size=30)
rcParams['figure.figsize']=(15,4)
print('Distribution of size factors')
sc.pl.violin(adata,keys='size_factors_sample', groupby='file',stripplot=False)

# %%
# Scale data with size factors
adata.X /= adata.obs['size_factors_sample'].values[:,None] # This reshapes the size-factors array
sc.pp.log1p(adata)
adata.X = np.asarray(adata.X)

# %% [markdown]
# ## Highly variable genes

# %%
# Compute HVG across batches (samples) using scIB function
hvgs=scib.preprocessing.hvg_batch(adata, batch_key='study_sample', target_genes=2000, flavor='cell_ranger')

# %%
# Add HVGs to adata
hvgs=pd.DataFrame([True]*len(hvgs),index=hvgs,columns=['highly_variable'])
hvgs=hvgs.reindex(adata.var_names,copy=False)
hvgs=hvgs.fillna(False)

adata.var['highly_variable']=hvgs.values
print('Number of highly variable genes: {:d}'.format(np.sum(adata.var['highly_variable'])))

# %% [markdown]
# Add also a larger set of HVGs:

# %%
# Compute HVG across batches (samples) using scIB function
hvgs=scib.preprocessing.hvg_batch(adata, batch_key='study_sample', target_genes=5000, flavor='cell_ranger')

# %%
# Add HVGs to adata
hvgs=pd.DataFrame([True]*len(hvgs),index=hvgs,columns=['highly_variable_5000'])
hvgs=hvgs.reindex(adata.var_names,copy=False)
hvgs=hvgs.fillna(False)

adata.var['highly_variable_5000']=hvgs.values
print('Number of highly variable genes: {:d}'.format(np.sum(adata.var['highly_variable_5000'])))

# %% [markdown]
# #### HVGs based on beta cells only

# %% [markdown]
# Subset adata to beta cells

# %%
[ct for ct in adata.obs.cell_type_multiplet.unique() if 'beta' in ct]

# %%
excluded_beta=['beta_delta']
selected_beta=[ct for ct in adata.obs.cell_type_multiplet.unique() if 'beta' in ct and ct not in excluded_beta]
selected_beta

# %%
# Subset adata
adata_beta=adata[adata.obs.cell_type.isin(selected_beta),:].copy()
#Filter genes:
print('Total number of genes: {:d}'.format(adata_beta.n_vars))
CELLS_THR_MIN=20
# Min 20 cells - filters out 0 count genes
sc.pp.filter_genes(adata_beta, min_cells=CELLS_THR_MIN)
print('Number of genes after beta cell filter: {:d}'.format(adata_beta.n_vars))
adata_beta.shape

# %%
# Compute HVG across batches (samples) using scIB function
hvgs=scib.preprocessing.hvg_batch(adata_beta, batch_key='study_sample', target_genes=2000, flavor='cell_ranger')
# Add HVGs to adata
hvg_col='highly_variable_2000_beta'
hvgs=pd.DataFrame([True]*len(hvgs),index=hvgs,columns=[hvg_col])
hvgs=hvgs.reindex(adata.var_names,fill_value=False)
#hvgs=hvgs.fillna(False)

adata.var[hvg_col]=hvgs.values
print('Number of highly variable genes: {:d}'.format(np.sum(adata.var[hvg_col])))

# %% [markdown]
# #### HVGs in individual studies

# %%
# Compute and store HVGs per study
for study in adata.obs.study.unique():
    adata_sub=adata[adata.obs.study==study,:].copy()
    print(study,adata_sub.shape[0])
    sc.pp.highly_variable_genes(adata_sub, flavor='cell_ranger', batch_key='file',n_top_genes =2000)
    hvg_col='highly_variable_2000_'+study
    adata.var[hvg_col]=adata_sub.var.highly_variable.reindex(adata.var_names,fill_value=False)

# %%
# Compute and store HVGs per study for beta cells
for study in adata_beta.obs.study.unique():
    adata_sub=adata_beta[adata_beta.obs.study==study,:].copy()
    print(study,adata_sub.shape[0])
    sc.pp.highly_variable_genes(adata_sub, flavor='cell_ranger', batch_key='file',n_top_genes =2000)
    hvg_col='highly_variable_2000_beta_'+study
    adata.var[hvg_col]=adata_sub.var.highly_variable.reindex(adata.var_names,fill_value=False)

# %% [markdown]
# #### HVGs union across samples instead of intersection across samples

# %%
# Extract HVGs (ordered by normalised dispersion) from each study_sample
hvgs_lists=dict()
for study_sample in adata.obs.study_sample.unique():
    adata_sub=adata[adata.obs.study_sample==study_sample,:].copy()
    print(study_sample,adata_sub.shape[0])
    sc.pp.highly_variable_genes(adata_sub, flavor='cell_ranger', n_top_genes =2000)
    hvgs_lists[study_sample]=adata_sub.var.dispersions_norm.sort_values(ascending=False).index
hvgs_lists=pd.DataFrame(hvgs_lists)
# Add next best HVG from each study_sample untill target N of HVGs is reached
hvgs=set()
for idx,row in hvgs_lists.iterrows():
    if len(hvgs) >= 2000:
        break
    hvgs.update(row.values)

# %%
# Add HVGs to adata
hvg_col='highly_variable_samples_target2000'
hvgs=pd.DataFrame([True]*len(hvgs),index=hvgs,columns=[hvg_col])
hvgs=hvgs.reindex(adata.var_names,fill_value=False)
adata.var[hvg_col]=hvgs.values
print('Number of highly variable genes: {:d}'.format(np.sum(adata.var[hvg_col])))

# %%
# Same as above for beta cells

# Extract HVGs (ordered by normalised dispersion) from each study_sample
hvgs_lists=dict()
for study_sample in adata_beta.obs.study_sample.unique():
    adata_sub=adata_beta[adata_beta.obs.study_sample==study_sample,:].copy()
    print(study_sample,adata_sub.shape[0])
    sc.pp.highly_variable_genes(adata_sub, flavor='cell_ranger', n_top_genes =2000)
    hvgs_lists[study_sample]=adata_sub.var.dispersions_norm.sort_values(ascending=False).index
hvgs_lists=pd.DataFrame(hvgs_lists)

# Add next best HVG from each study_sample untill target N of HVGs is reached
hvgs=set()
for idx,row in hvgs_lists.iterrows():
    if len(hvgs) >= 2000:
        break
    hvgs.update(row.values)
    
# Add HVGs to adata
hvg_col='highly_variable_samples_target2000_beta'
hvgs=pd.DataFrame([True]*len(hvgs),index=hvgs,columns=[hvg_col])
hvgs=hvgs.reindex(adata.var_names,fill_value=False)
adata.var[hvg_col]=hvgs.values
print('Number of highly variable genes: {:d}'.format(np.sum(adata.var[hvg_col])))

# %% [markdown]
# #### HVGs at multiple resolutions

# %%
hvgs_lists=[]
for study_sample in adata.obs.study_sample.unique():
    level='0'
    cluster='0'
    adata_sub=adata[adata.obs.study_sample==study_sample,:].copy()
    print(study_sample,adata_sub.shape)
    sc.pp.highly_variable_genes(adata_sub, flavor='cell_ranger', n_top_genes =2000)
    hvgs_lists.append(pd.Series(adata_sub.var.dispersions_norm.sort_values(ascending=False).index,
                                name=study_sample+'_l'+level+'_c'+cluster))
    adata_sub_scl=adata_sub.copy()
    sc.pp.scale(adata_sub_scl,max_value=10)
    sc.pp.pca(adata_sub_scl, n_comps=10, use_highly_variable=True, svd_solver='arpack')
    sc.pp.neighbors(adata_sub_scl,n_pcs = 10) 
    sc.tl.leiden(adata_sub_scl, resolution=1, directed=True, use_weights=True)
    adata_sub.obs['leiden_scl']=adata_sub_scl.obs.leiden
    level='1'
    for cluster in adata_sub.obs.leiden_scl.unique():
        adata_sub_sub=adata_sub[adata_sub.obs.leiden_scl==cluster,:].copy()
        # Subset genes
        min_cells=10
        if adata_sub_sub.shape[0] >= min_cells:
            sc.pp.filter_genes(adata_sub_sub, min_cells=min_cells)
            print(level,cluster,adata_sub_sub.shape)
            sc.pp.highly_variable_genes(adata_sub_sub, flavor='cell_ranger', n_top_genes =2000)
            hvgs_lists.append(pd.Series(adata_sub_sub.var.dispersions_norm.sort_values(ascending=False).index,
                             name=study_sample+'_l'+level+'_c'+cluster))
        else:
            print('Cluster %s (level %s) has %i cells (less than %i) and is thus ignored'%(
                cluster,level,adata_sub_sub.shape[0],min_cells))
hvgs_lists=pd.concat(hvgs_lists,axis=1)

# %%
# Add next best HVG from each study_sample untill target N of HVGs is reached
hvgs=set()
for idx,row in hvgs_lists.iterrows():
    if len(hvgs) >= 2000:
        break
    hvgs.update(row.dropna().values)
    
# Add HVGs to adata
hvg_col='highly_variable_multilevel_target2000'
hvgs=pd.DataFrame([True]*len(hvgs),index=hvgs,columns=[hvg_col])
hvgs=hvgs.reindex(adata.var_names,fill_value=False)
adata.var[hvg_col]=hvgs.values
print('Number of highly variable genes: {:d}'.format(np.sum(adata.var[hvg_col])))

# %% [markdown]
# ### Compare HVG sets

# %% [markdown]
# Overlap between different HVG sets

# %%
rcParams['figure.figsize']= (25,17)
fig,axs=plt.subplots(1,3)
hvgs = {
    "hvgs_union_beta": set(adata.var_names[adata.var.highly_variable_samples_target2000_beta]),
    "hvgs_intersection_beta":set(adata.var_names[adata.var.highly_variable_2000_beta]),
}
venn(hvgs,ax=axs[0])
hvgs = {
    "hvgs_union": set(adata.var_names[adata.var.highly_variable_samples_target2000]),
    "hvgs_intersection":set(adata.var_names[adata.var.highly_variable]),
}
venn(hvgs,ax=axs[1])
hvgs = {
    "hvgs_union": set(adata.var_names[adata.var.highly_variable_samples_target2000]),
    "hvgs_intersection":set(adata.var_names[adata.var.highly_variable]),
    "hvgs_union_beta": set(adata.var_names[adata.var.highly_variable_samples_target2000_beta]),
    "hvgs_intersection_beta":set(adata.var_names[adata.var.highly_variable_2000_beta]),
}
venn(hvgs,ax=axs[2])

# %%
rcParams['figure.figsize']= (8,8)
fig,ax=plt.subplots()
hvgs = {
    "hvgs_union": set(adata.var_names[adata.var.highly_variable_samples_target2000]),
    "hvgs_intersection":set(adata.var_names[adata.var.highly_variable]),
    "hvgs_multilevel":set(adata.var_names[adata.var.highly_variable_multilevel_target2000]),
}
venn(hvgs,ax=ax)


# %% [markdown]
# #C: Selecting HVGs at multiple levels has large effect on HVG composition.

# %%
# HVGs per study
rcParams['figure.figsize']= (25,17)
fig,axs=plt.subplots(1,2)
hvgs = {}
for study in adata.obs.study.unique():
    hvgs[study+'_beta']=set(adata.var_names[adata.var['highly_variable_2000_beta_'+study]])
venn(hvgs,ax=axs[0])
hvgs = {}
for study in adata.obs.study.unique():
    hvgs[study]=set(adata.var_names[adata.var['highly_variable_2000_'+study]])
venn(hvgs,ax=axs[1])

# %% [markdown]
# #C: Beta HVGs are more diverse across studies. This is probably due to cell-type HVGs being conserved across studies due to presence of similar cell types, while beta cell HVGs depend on biological perturbation present in each study.

# %% [markdown]
# Ratio overlap (based on smaller HVG set) between HVG sets

# %%
# Compute HVG set overlap as ratio of smaller HVG set size
hvg_endings=['2000','samples_target2000','multilevel_target2000','5000','2000_beta','samples_target2000_beta']
overlap=pd.DataFrame(index=hvg_endings,columns=hvg_endings)
hvgs=adata.var.rename({'highly_variable':'highly_variable_2000'},axis=1)
for i in range(len(hvg_endings)-1):
    for j in range(i+1,len(hvg_endings)):
        hvg_col1='highly_variable_'+hvg_endings[i]
        hvg_col2='highly_variable_'+hvg_endings[j]
        hvgs1=set(hvgs.index[hvgs[hvg_col1]])
        hvgs2=set(hvgs.index[hvgs[hvg_col2]])
        overlap.loc[hvg_endings[i],hvg_endings[j]]=len(hvgs1&hvgs2)/min([len(hvgs1),len(hvgs2)])

# %%
# Plot HVG overlap ratio
rcParams['figure.figsize']= (6,5)
mask = np.zeros_like(overlap.values)
mask[np.tril_indices_from(mask)] = True
sb.heatmap(overlap.fillna(0),mask=mask,annot=True, fmt=".2f")
t=plt.title('Proportion of shared HVGs (scaled by smaller HVG set)\n')

# %% [markdown]
# #C: Increasing HVG set (5000 HVGs) does not cover multilevel HVGs (it cover union/intersection HVGs). It covers beta HVGs better than multilevel HVGs, possibly indicating bias of HVGs towards beta cells. 

# %% [markdown]
# Presence of beta subtype markers (known to vary across beta cells) in different HVG sets

# %%
# beta subtype markers
beta_subtype_markers=['Ins1','Ins2','Mafa','Ucn3','Slc2a2','Rbp4','Mafb','Pyy','Cd81','Chgb','Chga',
                      'Cxcl10','Ccl5','Etv1']
# Check if marker is in var_names and if so if it is in HVG sets
marker_hvg=pd.DataFrame(columns=['highly_variable_'+hvg_ending for hvg_ending in hvg_endings])
for marker in beta_subtype_markers:
    if marker not in adata.var_names:
        print(marker,'not in var_names')
    else:
        marker_hvg.loc[marker,:]=hvgs.loc[marker,marker_hvg.columns].values
# Plot marker presence in HVG sets
marker_hvg=marker_hvg.replace({False:0,True:1})
sb.heatmap(marker_hvg)

# %% [markdown]
# #C: It is odd that Chgb is not present in beta HVGs as it varies across and within some of the samples in beta cells. It is also unexpected that some other beta cell subtype markers are missing only from beta HVGs. This might be due to them being even more lowly/highly expressed in non-beta populations, thus being marked as HVG only when other cells are included.

# %% [markdown]
# Presence of beta markers in per study beta cell based HVGs.

# %%
# beta subtype markers
beta_subtype_markers=['Ins1','Ins2','Mafa','Ucn3','Slc2a2','Rbp4','Mafb','Pyy','Cd81','Chgb','Chga',
                      'Cxcl10','Ccl5','Etv1']
# Check if marker is in var_names and if so if it is in HVG sets
marker_hvg=pd.DataFrame(columns=['highly_variable_2000_beta_'+study for study in adata.obs.study.unique()])
for marker in beta_subtype_markers:
    if marker not in adata.var_names:
        print(marker,'not in var_names')
    else:
        marker_hvg.loc[marker,:]=hvgs.loc[marker,marker_hvg.columns].values
# Plot marker presence in HVG sets
marker_hvg=marker_hvg.replace({False:0,True:1})
sb.heatmap(marker_hvg)

# %% [markdown]
# ## Visualisation

# %%
# Sort age categories
sorted_age=['16 d','3 m','182 d','20 w','2 y']
# Make sure that all categories were used for sorting
if set(sorted_age)==set(list(adata.obs.age.unique())):
    adata.obs['age']= pd.Categorical(adata.obs.age,sorted_age)

# %% [markdown]
# #### Unscaled data

# %%
sc.pp.pca(adata, n_comps=30, use_highly_variable=True, svd_solver='arpack')

# %%
rcParams['figure.figsize']= (8,6)
sc.pl.pca_variance_ratio(adata)

# %%
# Select number of PCs to use
N_PCS=13

# %%
sc.pp.neighbors(adata,n_pcs = N_PCS) 
sc.tl.umap(adata)

# %% [markdown]
# Cheeck if np.nan values are present in metadata that would not be plotted:

# %%
# Check if metadata contains nan values that would not be plotted
for col in ['study','file','sex','age','phase_cyclone','tissue','cell_type']:
    n_na=adata.obs[col].isna().sum()
    print(col,'N nan:',n_na)

# %%
adata

# %%
rcParams['figure.figsize']= (8,8)
sc.pl.umap(adata, color=['study','file','sex','age','phase_cyclone','tissue',] ,size=10, use_raw=False,wspace=0.3)
sc.pl.umap(adata, color=['cell_type'] ,size=10, use_raw=False,wspace=0.3)
sc.pl.umap(adata, color=['cell_type_multiplet'] ,size=10, use_raw=False,wspace=0.3)

# %%
adata.obs.cell_type_multiplet[adata.obs.cell_type_multiplet.str.contains('NA|multiplet')].unique()

# %% [markdown]
# #### Save normalised log scaled data with HVG annotation with UMAP

# %%
if SAVE:
    h.save_h5ad(adata=adata,file=shared_folder+'data_normalised.h5ad',unique_id2=UID2)

# %% [markdown]
# #### Z-scaled data

# %%
# Scale data and compute PCA
adata_scl=adata.copy()
sc.pp.scale(adata_scl,max_value=10)
sc.pp.pca(adata_scl, n_comps=30, use_highly_variable=True, svd_solver='arpack')

# %%
rcParams['figure.figsize']= (8,6)
sc.pl.pca_variance_ratio(adata_scl)

# %%
# Select number of PCs to use
N_PCS=13

# %%
sc.pp.neighbors(adata_scl,n_pcs = N_PCS) 
sc.tl.umap(adata_scl)

# %%
rcParams['figure.figsize']= (8,8)
sc.pl.umap(adata_scl, color=['study','file','sex','age','phase_cyclone','tissue'] ,size=10, use_raw=False,wspace=0.3)
sc.pl.umap(adata, color=['cell_type'] ,size=10, use_raw=False,wspace=0.3)
sc.pl.umap(adata, color=['cell_type_multiplet'] ,size=10, use_raw=False,wspace=0.3)

# %% [raw]
# ##### Save z-scaled data with UMAP

# %% [raw]
# h.save_h5ad(adata=adata_scl,file=shared_folder+'data_z-scaled_UMAP.h5ad',unique_id2=UID2)

# %% [raw]
# ## Add new cell type annotations
#
# This is not needed for reproducibility. - This was done only as part of the cell type annotation (other notebooks) was added after making the combined object (the annotation notebooks were extended).

# %% [raw]
# adata=h.open_h5ad(file=shared_folder+'data_normalised.h5ad',unique_id2=UID2)

# %% [raw]
# cell_type_cols=['cell_type','cell_type_multiplet','cell_subtype','cell_subtype_multiplet']
# adata.obs.drop([col for col in cell_type_cols if col in adata.obs.columns],axis=1,inplace=True)
# for study,path in data:
#     print(study)
#     obs_sub=h.open_h5ad(file=path+'data_annotated.h5ad',unique_id2=UID2).obs
#     obs_sub.index=[idx+'-'+study for idx in obs_sub.index]
#     for col in cell_type_cols:
#         adata.obs.loc[obs_sub.index,col]=obs_sub[col]

# %% [raw]
# h.save_h5ad(adata=adata,file=shared_folder+'data_normalised.h5ad',unique_id2=UID2)

# %% [markdown]
# ## Summary of metadata

# %%
adata=h.open_h5ad(file=shared_folder+'data_normalised.h5ad',unique_id2=UID2)
#adata=sc.read_h5ad(shared_folder+'data_normalised.h5ad')

# %% [markdown]
# N cells per study:

# %%
# N cells
print('N cells per sample')
adata.obs['study_sample'].value_counts()

# %%
sample_count_df=pd.DataFrame(adata.obs['study_sample'].value_counts())
sample_count_df.columns=['n_cells']
sample_count_df['study_sample']=sample_count_df.index
sample_count_df['study']=[adata.obs.query('study_sample == @sample')['study'][0] for sample in sample_count_df.index]
sample_count_df['sample']=[adata.obs.query('study_sample == @sample')['file'][0] for sample in sample_count_df.index]

# %%
rcParams['figure.figsize']= (8,6)
sb.barplot(x="sample", y="n_cells", hue="study", data=sample_count_df.sort_values('study'),dodge=False)
a=plt.xticks(rotation=90)
plt.legend( title='study', bbox_to_anchor=(1.05, 1), loc='upper left')

# %% [markdown]
# Sex distribution:

# %%
print('N male/female per sample')
adata.obs.groupby(['study_sample','sex']).size().unstack(fill_value=0)

# %%
print('\nMedian cell type counts across samples')
median_type_n=adata.obs.groupby(['study_sample','cell_type']).size().unstack(fill_value=0).median().sort_values(ascending=False)
median_type_n

# %%
# Find cells with good/bad annotation
eval_cells=~adata.obs.cell_type_multiplet.str.contains('NA|multiplet')
eval_cells.replace({True:'yes',False:'no'},inplace=True)
adata.obs['true_type']=eval_cells

# %%
anno_counts=pd.DataFrame(adata.obs.groupby(['study','true_type','cell_type']).size())
anno_counts.columns=['n_cells']
anno_counts.reset_index(inplace=True)

# %%
rcParams['figure.figsize']= (25,20)
fix,axs=plt.subplots(5,1,sharex=True,sharey=True)
plt.xticks(fontsize= 25)
for i,study in enumerate(adata.obs.study.unique()):
    anno_counts_sub=anno_counts.query('study ==@study')
    sb.barplot(x="cell_type", y="n_cells", hue="true_type", 
               data=anno_counts_sub,dodge=False, ax=axs[i],
               # Sort by median cell type count across studies
               order=median_type_n.index.values,hue_order=['yes','no'])
    a=plt.xticks(rotation=90)
    plt.yscale('log')
    axs[i].set_xlabel('')
    axs[i].set_ylabel('n_cells\n'+study,fontsize=25)
    if i>0:
        axs[i].get_legend().remove()
    else:
        l=axs[i].legend( title='true cell type', bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=25)
        l.get_title().set_fontsize(25)

# %%
print('Cell type counts per sample')
pd.set_option('display.max_columns', len(adata.obs.cell_type.unique()))
display(adata.obs.groupby(['study_sample','cell_type']).size().unstack(fill_value=0))

# %% [markdown]
# #### Beta cells

# %%
excluded_beta=['beta_delta']
selected_beta=[ct for ct in adata.obs.cell_type_multiplet.unique() if 'beta' in ct and ct not in excluded_beta]
selected_beta

# %%
# Subset adata to beta cells
adata_beta=adata[adata.obs.cell_type.isin(selected_beta),:].copy()
anno_counts_beta=pd.DataFrame(adata_beta.obs.groupby(['study','true_type','cell_subtype']).size())
anno_counts_beta.columns=['n_cells']
anno_counts_beta.reset_index(inplace=True)

# %% [markdown]
# Cell proportions for beta cells only

# %%
print('\nMedian beta cell subtype counts across samples')
median_type_n_beta=adata_beta.obs.groupby(['study_sample','cell_subtype']).size().unstack(fill_value=0).median().sort_values(ascending=False)
median_type_n_beta

# %%
rcParams['figure.figsize']= (25,20)
fix,axs=plt.subplots(5,1,sharex=True,sharey=True)
plt.xticks(fontsize= 25)
for i,study in enumerate(adata_beta.obs.study.unique()):
    anno_counts_sub=anno_counts_beta.query('study ==@study')
    sb.barplot(x="cell_subtype", y="n_cells", hue="true_type", 
               data=anno_counts_sub,dodge=False, ax=axs[i],
               # Sort by median cell type count across studies
               order=median_type_n_beta.index.values,hue_order=['yes','no'])
    a=plt.xticks(rotation=90)
    plt.yscale('log')
    axs[i].set_xlabel('')
    axs[i].set_ylabel('n_cells\n'+study,fontsize=25)
    if i>0:
        axs[i].get_legend().remove()
    else:
        l=axs[i].legend( title='true cell type', bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=25)
        l.get_title().set_fontsize(25)

# %%
