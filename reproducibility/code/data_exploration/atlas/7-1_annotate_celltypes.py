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
# # Re-annotation
# - Annotate cell clusters based on re-clustering data and using majority vote from ref annotation.
# - Use QC metrics, marker expression, and hormone scores to further annotate low quality and multiplet subclusters (focusing on endocrine cells).
# - Summary plots.

# %%
import scanpy as sc
import pandas as pd
import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/')

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib 
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sb

import h5py
import numpy as np
import anndata
from sklearn.preprocessing import minmax_scale

from importlib import reload  
import helper as h
reload(h)
import helper as h

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/'
UID2='integrated_anno'

# %%
# Saving figures
path_fig='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/figures/paper/'

# %%
adata=sc.read(path_data+'data_integrated_annotated.h5ad')

# %%
adata.raw.shape

# %% [markdown]
# ### Plot previously annotated cell types: 
# - cell_type: our per dataset annotation for ref samples
# - cell_type_multiplet: as cell_type, but marking likely multiplets as a single category
# - pre_cell_type: annotation from previous studies

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata, color=['cell_type','cell_type_multiplet','pre_cell_type'] ,size=10, use_raw=False,ncols=1)

# %% [markdown]
# UMAP with main cell types from ref (e.g. removing some subpopulations, multiplets, etc) for the purpose of easier visualisation of main cell types.

# %%
# Count of non-multiplet ref cell types
ct_counts=adata.obs.cell_type_multiplet.value_counts()
ct_counts

# %%
# Select cells common in ref and plot them over unannoated UMAP
# Select cell types
main_cell_types=[ct for ct in ct_counts[ct_counts>90].index if ct!='NA' and ct!='multiplet']
# Plot unannotated UMAP and main cell types
fig,ax=plt.subplots(figsize=(8,8))
adata.obs['constant']=['NA']*adata.shape[0]
sc.pl.umap(adata,s=10,ax=ax,show=False,color='constant',palette=['#ebebeb'])
adata.obs['cell_type_main']=[ct if ct in main_cell_types else np.nan for ct in adata.obs.cell_type_multiplet]
sc.pl.umap(adata[~pd.isna(adata.obs.cell_type_main),:],s=10,ax=ax,show=False,color='cell_type_main')
adata.obs.drop(['constant','cell_type_main'],axis=1,inplace=True)

# %%
rcParams['figure.figsize']=(10,10)
random_indices=np.random.permutation(list(range(adata.shape[0])))
sc.pl.umap(adata[random_indices,:], color=['study'] ,size=10, use_raw=False,ncols=1)

# %% [markdown]
# ### Endocrine multiplets visualisation
# Plot multiplet assignment from ref and hormone high scores to get idea on endocrine multiplets.

# %%
cell_type='multiplet'
adata.obs[cell_type]=adata.obs.cell_type_multiplet.str.contains(cell_type)
sc.pl.umap(adata,color=cell_type,s=5)
adata.obs.drop(cell_type,axis=1,inplace=True)

# %%
cell_type='ins_low'
adata.obs[cell_type]=~adata.obs.ins_high
sc.pl.umap(adata,color=cell_type,s=5)
adata.obs.drop(cell_type,axis=1,inplace=True)

# %%
sc.pl.umap(adata,color='ins_high',s=5)

# %%
sc.pl.umap(adata,color='gcg_high',s=5)

# %%
sc.pl.umap(adata,color='sst_high',s=5)

# %%
sc.pl.umap(adata,color='ppy_high',s=5)

# %%
# hormone scores scaled by sample
scores=['gcg','ins','ppy','sst']
for score in scores:
    adata.obs[score+'_score_scaled']=adata.obs.groupby('study_sample')[score+'_score'].apply(
        lambda x: pd.DataFrame(minmax_scale(x),index=x.index,columns=[score+'_socre_scaled'])
).unstack()[score+'_socre_scaled']

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(adata,color=[score+'_score_scaled' for score in scores],s=20)

# %% [markdown]
# Check how comparable are normalised hormone scores across studies

# %%
for score,thr in {'gcg':0.3,'ins':0.2,'ppy':0.6,'sst':0.25}.items():
    score=score+"_score_scaled"
    fig,ax=plt.subplots(1,2,figsize=(20,5))
    rcParams['figure.figsize']=(8,4)
    sb.violinplot(x="study", y=score, data=adata.obs,inner=None,ax=ax[0])
    sb.violinplot(x="study", y=score,
                  data=adata[adata.obs[score]>thr,:].obs,inner=None,ax=ax[1])
    ax[0].set_title('Distribution of scores')
    ax[1].set_title('Distribution of scores above '+str(thr))
    fig.suptitle(score)

# %% [markdown]
# It seems that general separation of cells in low and high score is comparable across studies. Exceptions is embryo - endocrine cells are rare there.

# %%
# Check markers from other cts
rcParams['figure.figsize']=(8,8)
markers_list=['Plvap','Pecam1','Ndufa4l2','Acta2','Pdgfrb','Col1a2','Cd86',
                            'Ptprc','Cpa1','Car2','Sox9','Cryab']
markers_id=[adata_rawnorm.var.query('gene_symbol==@g').index[0] for g in markers_list]
sc.pl.umap(adata_rawnorm,color=markers_id,s=30)

# %% [markdown]
# ### QC metrics

# %%
# Calculate QC metrics
adata.obs['n_counts']=adata.raw.X.sum(axis=1)
adata.obs['n_genes']=(adata.raw.X > 0).sum(axis = 1)
mt_gene_mask = np.flatnonzero([gene.startswith('mt-') 
                               for gene in adata_rawnorm.var.gene_symbol])
adata.obs['mt_frac'] = np.sum(
    adata.raw[:, mt_gene_mask].X, axis=1).A1/adata.obs['n_counts']

# %%
rcParams['figure.figsize']=(9,9)
sc.pl.umap(adata,color=['n_genes','n_counts','mt_frac'],s=10)

# %% [markdown]
# Add empty drops information

# %%
# Files with saved empty drops info
files=pd.read_table('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/raw_file_list.tsv')
files.index=[study+'_'+sample for study,sample in zip(files['study'],files['sample'])]

# %%
# Load empty drops info
for study_sample in adata.obs.study_sample.unique():
    file=files.at[study_sample,'dir']+'raw_'+files.at[study_sample,'ending']
    # Index used in raw and merged data
    index_raw=adata.obs_names[adata.obs.study_sample==study_sample]
    index_parsed=[
        idx.replace('-'+files.at[study_sample,'sample']+'-'+files.at[study_sample,'study'],'')
    for idx in index_raw]
    # Load ambient info
    adata.obs.loc[index_raw,'emptyDrops_LogProb'
                     ]=sc.read(file,backed='r').obs.loc[index_parsed,'emptyDrops_LogProb'].values

# %%
sc.pl.violin(adata,keys='emptyDrops_LogProb',groupby='study_sample',stripplot=False)

# %%
random_indices=np.random.permutation(list(range(adata.shape[0])))
sc.pl.umap(adata[random_indices,:],color='emptyDrops_LogProb',s=10,sort_order=False)

# %%
# Scale empty drops per sample
adata.obs['emptyDrops_LogProb_scaled']=adata.obs.groupby('study_sample')['emptyDrops_LogProb'].apply(
        lambda x: pd.DataFrame(minmax_scale(x),index=x.index,columns=['emptyDrops_LogProb_scaled'])
).unstack()['emptyDrops_LogProb_scaled']

# %%
sc.pl.violin(adata,keys='emptyDrops_LogProb_scaled',groupby='study_sample',stripplot=False)

# %%
random_indices=np.random.permutation(list(range(adata.shape[0])))
sc.pl.umap(adata[random_indices,:],color='emptyDrops_LogProb_scaled',s=10,sort_order=False)

# %% [markdown]
# C: The cluster of endocrine cells with poor QC metrics may be empty/low quality cells.

# %% [markdown]
# ## Cell type annotation

# %% [markdown]
# Subcluster - run the clustering multiple times with different resolutions, requires manually changing the res param and re-running the clustering command.

# %%
res=0.4

# %%
sc.tl.leiden(adata, resolution=res, key_added='leiden_r'+str(res), directed=True, use_weights=True)

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata, color=['leiden_r0.1'] ,size=10, use_raw=False)

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata, color=['leiden_r0.5'] ,size=10, use_raw=False)

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata, color=['leiden_r0.4'] ,size=5, use_raw=False)

# %%
CLUSTERING='leiden_r0.4'

# %% [markdown]
# Previously annotated cell type count per cluster

# %%
for group,data in adata.obs.groupby(CLUSTERING):
    print('***',group,'***')
    ct_count=data['cell_type'].value_counts()
    pct_count=data['pre_cell_type'].value_counts()
    print('** Cell type')
    print(ct_count[ct_count!=0])
    print('** Preannotated cell type')
    print(pct_count[pct_count!=0])
    print('\n')

# %%
# Annotate clusters based on previous information
cluster_annotation={'0':'endocrine','1':'endocrine','2':'endocrine',
'3':'embryo','4':'endocrine','5':'fibroblast_related','6':'endothelial',
'7':'immune','8':'immune','9':'ductal',
'10':'embryo endocrine','11':'immune',
'12':'pericyte','13':'immune','14':'endocrine proliferative','15':'acinar'}

# %%
# Add annotation to adata

# %%
# Name of CT col
#CT_COL_INTEGRATION='cell_type_integrated' # This was the old integration with less resolved beta cell subtypes
CT_COL_INTEGRATION='cell_type_integrated_v1'

# %%
adata.obs[CT_COL_INTEGRATION]=[cluster_annotation[cluster] for cluster in adata.obs[CLUSTERING]]

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata, color=[CT_COL_INTEGRATION] ,size=10, use_raw=False)

# %% [markdown]
# ### Save data

# %%
# Check if any obs columns are redundant
adata

# %%
keep_obs=['study_sample', 'study', 'file', 'reference', 'size_factors_sample', 
          'S_score', 'G2M_score', 'phase', 'phase_cyclone', 's_cyclone', 'g2m_cyclone', 'g1_cyclone', 
          'sex', 'pre_cell_type', 
          'ins_score', 'ins_high', 'gcg_score', 'gcg_high', 'sst_score', 'sst_high', 'ppy_score', 'ppy_high', 
          'cell_filtering', 'age', 'strain', 'tissue', 'technique', 'internal_id', 'batch', 
          'study_sample_design', 'cell_type', 'cell_type_multiplet', 'cell_subtype', 'cell_subtype_multiplet', 
          'leiden_r0.4', 'design','cell_type_integrated_v1']
drop=[col for col in adata.obs.columns if col not in keep_obs]
adata.obs.drop(drop,axis=1,inplace=True)

# %%
h.save_h5ad(adata=adata,file=path_data+'data_integrated_analysed.h5ad', unique_id2=UID2)

# %% [markdown]
# ## Resolve endocrine populations

# %% [markdown]
# ### Beta cells - postnatal non-proliferative

# %%
# Beta clusters
clusters=['0','2']
adata_sub=adata[adata.obs[CLUSTERING].isin(clusters)]
adata_sub.obs.drop([col for col in adata_sub.obs.columns if 'leiden_r' in col],
                   axis=1,inplace=True)

# %%
#(109294, 13786)
adata_sub.shape

# %% [markdown]
# Plot hormone and QC scores

# %%
sc.pp.neighbors(adata_sub,n_pcs=0,use_rep='X_integrated')
sc.tl.umap(adata_sub)

# %%
sc.pl.umap(adata_sub,s=10,
           color=['ins_high','gcg_high','sst_high','ppy_high'])

# %%
sc.pl.umap(adata_sub,color=[score+'_score_scaled' for score in scores],s=30)

# %%
sc.pl.violin(adata_sub,keys='emptyDrops_LogProb_scaled',groupby='study_sample',stripplot=False)

# %%
sc.pl.violin(adata_sub,keys='emptyDrops_LogProb_scaled',groupby='study',stripplot=False)

# %%
random_indices=np.random.permutation(list(range(adata_sub.shape[0])))
sc.pl.umap(adata_sub[random_indices,:]
           ,color=['n_genes','n_counts','mt_frac','emptyDrops_LogProb_scaled'],
           s=10,sort_order=False)

# %% [markdown]
# C: In the low Q population there are different types of cells: High ambient and mt - probably damaged cells. Low N counts, but not so high ambient and not so low N genes - also probably cells, but of lower quality. High ambient, low mt, low counts and genes - probably ambient cells. The latter will be annotated as low_quality.

# %%
sc.pl.umap(adata_sub,s=10,color=['study'])

# %%
rcParams['figure.figsize']=(6,6)
cell_type='ins_low'
adata_sub.obs[cell_type]=~adata_sub.obs.ins_high
sc.pl.umap(adata_sub,color=cell_type,s=10)
adata_sub.obs.drop(cell_type,axis=1,inplace=True)

# %% [markdown]
# Subcluster. Manually change res and rerun clustering multiple times to visualise different clustering resolutions.

# %%
res=1

# %%
sc.tl.leiden(adata_sub, resolution=res, key_added='leiden_r'+str(res), 
             directed=True, use_weights=True)

# %%
sc.pl.umap(adata_sub,s=10,color=['leiden_r0.7','leiden_r1','leiden_r1.5'],wspace=0.3)

# %% [markdown]
# Quality merics per cluster

# %%
sc.pl.violin(adata_sub,keys='emptyDrops_LogProb_scaled',groupby='leiden_r1',stripplot=False)

# %%
sc.pl.violin(adata_sub,keys=['n_genes','n_counts','mt_frac'],groupby='leiden_r1',stripplot=False)

# %% [markdown]
# Plot beta cell heterogeneity markers to detremine if low quality clusters are likely low quality cells or a specific subpopulation

# %%
markers=pd.read_excel('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/markers.xlsx', 
                      sheet_name='mice')

# %%
# prepare adata for plotting - gene symbols in var_names
adata_rawnorm_sub_temp=anndata.AnnData(adata_rawnorm[adata_sub.obs_names,].X.copy(),
               obs=adata_sub.obs.copy(),
               var=adata_rawnorm.var.copy(),
               uns=adata_sub.uns.copy(),
               obsm=adata_sub.obsm.copy(),obsp=adata_sub.obsp.copy())
adata_rawnorm_sub_temp.var_names=adata_rawnorm_sub_temp.var['gene_symbol'].astype('str')
adata_rawnorm_sub_temp.var_names=adata_rawnorm_sub_temp.var_names.fillna('NA')
adata_rawnorm_sub_temp.var_names_make_unique()

# %%
# Plot groups of beta cell related markers
for group_name, group in markers.query('Cell in ["beta","endocrine"]').fillna('/').groupby(['Cell','Subtype']):
    print(group_name)
    genes=group.Gene
    # Retain genes present in raw var_names
    missing=[gene for gene in genes if gene not in adata_rawnorm_sub_temp.var_names]
    genes=[gene for gene in genes if gene in adata_rawnorm_sub_temp.var_names]
    print('Missing genes:',missing)
    rcParams['figure.figsize']=(4,4)
    sc.pl.umap(adata_rawnorm_sub_temp, color=genes ,size=10, use_raw=False)

# %% [markdown]
# Some potential low Q regions have distinct markers highly expressed.

# %% [markdown]
# Markers of other cell types to find potential doublets with them

# %%
rcParams['figure.figsize']=(8,8)
markers_list=['Plvap','Pecam1','Ndufa4l2','Acta2','Pdgfrb','Col1a2','Cd86',
                            'Ptprc','Cpa1','Car2','Sox9','Cryab']
#markers_id=[adata_rawnorm_sub_temp.var.query('gene_symbol==@g').index[0] for g in markers_list]
sc.pl.umap(adata_rawnorm_sub_temp,color=markers_list,s=20)

# %% [markdown]
# C: Due to different version of scanpy the UMAP here is a bit different
#
# C: There is a small cluster of cells that seem to be potentially comming from another ct.

# %%
del adata_rawnorm_sub_temp

# %% [markdown]
# C: Clusters 11 and 13 have low quality metrics - high ambience and low genes&counts or high mt_fraction. Thus they will be marked as low_quality. Although subcluster 11 needs subclustering as part of it may be true cells. Clusters 11 had also low expression of various beta markers, while cluster 13 had higher expression and it was thus decided it will not be removed.

# %% [markdown]
# Proportions of horomone high cells in each cluster to find potential doublets.

# %%
proportions2=[]
for group,data in adata_sub.obs.groupby('leiden_r1.5'):
    #print('***',group,'***')
    #ct_count=data['cell_type'].value_counts()
    #print('** Cell type')
    #print(ct_count[ct_count!=0])
    #print('Ins_high:\n',data['ins_high'].value_counts(normalize=True,sort=False)[True])
    #print('Gcg_high:\n',data['gcg_high'].value_counts(normalize=True,sort=False)[True])
    #print('Sst_high:\n',data['sst_high'].value_counts(normalize=True,sort=False)[True])
    #print('Ppy_high:\n',data['ppy_high'].value_counts(normalize=True,sort=False)[True])
    for gene in ['ins','gcg','sst','ppy']:
        proportion=data[gene+'_high'].value_counts(normalize=True,sort=False)
        if True in proportion.index:
            proportion=proportion[True]
        else:
            proportion=0
        proportions2.append({'cluster':group,'gene':gene,
                            'proportion':proportion})
proportions2=pd.DataFrame(proportions2)
g = sb.catplot(
    data=proportions2, kind="bar",
    x="cluster", y="proportion", hue="gene", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("Cluster", "Porportion of hormone high cells")

# %%
proportions2=[]
for group,data in adata_sub.obs.groupby('leiden_r1'):
    #print('***',group,'***')
    #ct_count=data['cell_type'].value_counts()
    #print('** Cell type')
    #print(ct_count[ct_count!=0])
    #print('Ins_high:\n',data['ins_high'].value_counts(normalize=True,sort=False)[True])
    #print('Gcg_high:\n',data['gcg_high'].value_counts(normalize=True,sort=False)[True])
    #print('Sst_high:\n',data['sst_high'].value_counts(normalize=True,sort=False)[True])
    #print('Ppy_high:\n',data['ppy_high'].value_counts(normalize=True,sort=False)[True])
    for gene in ['ins','gcg','sst','ppy']:
        proportion=data[gene+'_high'].value_counts(normalize=True,sort=False)
        if True in proportion.index:
            proportion=proportion[True]
        else:
            proportion=0
        proportions2.append({'cluster':group,'gene':gene,
                            'proportion':proportion})
proportions2=pd.DataFrame(proportions2)
g = sb.catplot(
    data=proportions2, kind="bar",
    x="cluster", y="proportion", hue="gene", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("Cluster", "Porportion of hormone high cells")

# %%
adata_sub.obs['leiden_r1'].value_counts()

# %% [markdown]
# C: With resolution 1 cl 14 is beta-alpha, cl 9 is beta-delta and clusters 7, 16, and 4 will be subclustered. However, cluster 16 is very small so subclustering would not make much sense. Thus it will not be subclustered.

# %%
CLUSTERING_SUB='leiden_r0.7'

# %%
proportions=[]
for group,data in adata_sub.obs.groupby('leiden_r0.7'):
    #print('***',group,'***')
    #ct_count=data['cell_type'].value_counts()
    #print('** Cell type')
    #print(ct_count[ct_count!=0])
    #print('Ins_high:\n',data['ins_high'].value_counts(normalize=True,sort=False)[True])
    #print('Gcg_high:\n',data['gcg_high'].value_counts(normalize=True,sort=False)[True])
    #print('Sst_high:\n',data['sst_high'].value_counts(normalize=True,sort=False)[True])
    #print('Ppy_high:\n',data['ppy_high'].value_counts(normalize=True,sort=False)[True])
    for gene in ['ins','gcg','sst','ppy']:
        proportion=data[gene+'_high'].value_counts(normalize=True,sort=False)
        if True in proportion.index:
            proportion=proportion[True]
        else:
            proportion=0
        proportions.append({'cluster':group,'gene':gene,
                            'proportion':proportion})
proportions=pd.DataFrame(proportions)

# %%
g = sb.catplot(
    data=proportions, kind="bar",
    x="cluster", y="proportion", hue="gene", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("Cluster", "Porportion of hormone high cells")


# %% [markdown]
# Check what proportion of cells from each sample is in beta-delta cluster. Per sample: proportion - proportion of beta cluster cells that are in beta-delta cluster, n - n all cells in beta cluster.

# %%
# Calculate proportion of cells in doublet cluster for each sample: 
# e.g. n in doublet subcluster/n in the whole beta cluster
proportion_subcluster=[]
n_all=[]
samples=adata_sub.obs.study_sample.unique()
for sample in samples:
    n_all_sample=adata_sub[adata_sub.obs['study_sample']==sample].shape[0]
    proportion_subcluster.append(
        adata_sub[(adata_sub.obs['study_sample']==sample).values & (adata_sub.obs['leiden_r0.7']=='8').values
                 ].shape[0]/n_all_sample)
    n_all.append(n_all_sample)
proportion_subcluster=pd.DataFrame({'proportion':proportion_subcluster,'n':n_all},index=samples)
# Sort by proportion of cells in doublet cluster and display prportion and n cells in beta cluster
proportion_subcluster.sort_values('proportion',ascending=False).style.background_gradient(cmap='viridis')

# %% [markdown]
# #### Subclustering

# %% [markdown]
# Make new column for subclustering.

# %%
CL_RES_SUB1='leiden_r'+str(1)

# %%
adata_sub.obs[CL_RES_SUB1+'_subclustered']=adata_sub.obs[CL_RES_SUB1].copy().astype('str')

# %%
rcParams['figure.figsize']=(4,4)

# %% [markdown]
# ##### Subcluster cluster 11. 

# %%
CL='11'

# %%
clusters=[CL]
adata_sub1=adata_sub[adata_sub.obs[CL_RES_SUB1].isin(clusters)]
adata_sub1.obs.drop([col for col in adata_sub1.obs.columns if 'leiden_r' in col],
                   axis=1,inplace=True)
#(3078, 13999)
adata_sub1.shape

# %%
sc.pp.neighbors(adata_sub1,n_pcs=0,use_rep='X_integrated')
sc.tl.umap(adata_sub1)

# %%
random_indices=np.random.permutation(list(range(adata_sub1.shape[0])))
sc.pl.umap(adata_sub1[random_indices,:]
           ,color=['n_genes','n_counts','mt_frac','emptyDrops_LogProb_scaled'],
           s=40,sort_order=False)

# %%
# prepare adata for plotting - gene symbols in var_names
adata_rawnorm_sub1_temp=anndata.AnnData(adata_rawnorm[adata_sub1.obs_names,].X.copy(),
               obs=adata_sub1.obs.copy(),
               var=adata_rawnorm.var.copy(),
               uns=adata_sub1.uns.copy(),
               obsm=adata_sub1.obsm.copy(),obsp=adata_sub1.obsp.copy())
adata_rawnorm_sub1_temp.var_names=adata_rawnorm_sub1_temp.var['gene_symbol'].astype('str')
adata_rawnorm_sub1_temp.var_names=adata_rawnorm_sub1_temp.var_names.fillna('NA')
adata_rawnorm_sub1_temp.var_names_make_unique()

# %%
for group_name, group in markers.query('Cell in ["beta","endocrine"]').fillna('/').groupby(['Cell','Subtype']):
    print(group_name)
    genes=group.Gene
    # Retain genes present in raw var_names
    missing=[gene for gene in genes if gene not in adata_rawnorm_sub1_temp.var_names]
    genes=[gene for gene in genes if gene in adata_rawnorm_sub1_temp.var_names]
    print('Missing genes:',missing)
    rcParams['figure.figsize']=(4,4)
    sc.pl.umap(adata_rawnorm_sub1_temp, color=genes ,size=50, use_raw=False)

# %% [markdown]
# C: Also interesting is population with high Pax6 as this is higher than in most beta cells (see above), so likely not just due to ambient effects.

# %%
res=1

# %%
sc.tl.leiden(adata_sub1, resolution=res, key_added='leiden_r'+str(res), 
             directed=True, use_weights=True)

# %%
sc.pl.umap(adata_sub1,color='leiden_r'+str(res),s=100)

# %%
sc.pl.violin(adata_sub1,keys=['n_genes','n_counts','mt_frac','emptyDrops_LogProb_scaled'],
             groupby='leiden_r'+str(res))

# %%
sc.pl.umap(adata_sub1,color=[score+'_score_scaled' for score in scores],s=100)

# %%
sc.pl.violin(adata_sub1,
             keys=['n_genes','n_counts','mt_frac','emptyDrops_LogProb_scaled'],
             groupby='leiden_r1',stripplot=False)

# %% [markdown]
# C: Subclusters 0,2,8,5,6 are probably not low quality cells. Clusters 1,7,9 have higher mt and are also interesting and will not be removed as ambient.

# %%
adata_sub.obs.loc[adata_sub1.obs_names,CL_RES_SUB1+'_subclustered'
                 ]=[CL+'_'+cl for cl in adata_sub1.obs['leiden_r'+str(res)]]

# %%
adata_sub.obs[CL_RES_SUB1+'_subclustered' ].value_counts()

# %% [markdown]
# ##### Subcluster cl 13

# %%
CL='13'

# %%
clusters=[CL]
adata_sub1=adata_sub[adata_sub.obs[CL_RES_SUB1].isin(clusters)]
adata_sub1.obs.drop([col for col in adata_sub1.obs.columns if 'leiden_r' in col],
                   axis=1,inplace=True)
#(1182, 13999)
adata_sub1.shape

# %%
sc.pp.neighbors(adata_sub1,n_pcs=0,use_rep='X_integrated')
sc.tl.umap(adata_sub1)

# %%
random_indices=np.random.permutation(list(range(adata_sub1.shape[0])))
sc.pl.umap(adata_sub1[random_indices,:]
           ,color=['n_genes','n_counts','mt_frac','emptyDrops_LogProb_scaled'],
           s=100,sort_order=False)

# %% [markdown]
# C: Whole cluster 13 seems to be of low quality, but it does not seem to be ambient due to high mt fraction.

# %% [markdown]
# ##### Subcluster cluster 7

# %%
CL='7'

# %%
clusters=[CL]
adata_sub1=adata_sub[adata_sub.obs[CL_RES_SUB1].isin(clusters)]
adata_sub1.obs.drop([col for col in adata_sub1.obs.columns if 'leiden_r' in col],
                   axis=1,inplace=True)
#(7102, 13999)
adata_sub1.shape

# %%
sc.pp.neighbors(adata_sub1,n_pcs=0,use_rep='X_integrated')
sc.tl.umap(adata_sub1)

# %%
res=0.5

# %%
sc.tl.leiden(adata_sub1, resolution=res, key_added='leiden_r'+str(res), 
             directed=True, use_weights=True)

# %%
np.random.seed(0)
random_indices=np.random.permutation(list(range(adata_sub1.shape[0])))
sc.pl.umap(adata_sub1[random_indices,:],
           color=[score+'_score_scaled' for score in scores]+['leiden_r'+str(res)],
          sort_order=False)

# %%
sc.pl.umap(adata_sub1, color=[score+'_score_scaled' for score in scores]+['leiden_r'+str(res)])

# %%
proportions2=[]
for group,data in adata_sub1.obs.groupby('leiden_r'+str(res)):
    for gene in ['ins','gcg','sst','ppy']:
        proportion=data[gene+'_high'].value_counts(normalize=True,sort=False)
        if True in proportion.index:
            proportion=proportion[True]
        else:
            proportion=0
        proportions2.append({'cluster':group,'gene':gene,
                            'proportion':proportion})
proportions2=pd.DataFrame(proportions2)
g = sb.catplot(
    data=proportions2, kind="bar",
    x="cluster", y="proportion", hue="gene", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("Cluster", "Porportion of hormone high cells")

# %%
adata_sub.obs.loc[adata_sub1.obs_names,CL_RES_SUB1+'_subclustered'
                 ]=[CL+'_'+cl for cl in adata_sub1.obs['leiden_r'+str(res)]]

# %%
adata_sub.obs[CL_RES_SUB1+'_subclustered' ].value_counts()

# %% [markdown]
# C: There seem to be beta-gamma cells in subcluster 3: 7_3.

# %% [markdown]
# ##### Subcluster cluster 4

# %%
CL='4'

# %%
clusters=[CL]
adata_sub1=adata_sub[adata_sub.obs[CL_RES_SUB1].isin(clusters)]
adata_sub1.obs.drop([col for col in adata_sub1.obs.columns if 'leiden_r' in col],
                   axis=1,inplace=True)
#(9041, 13999)
adata_sub1.shape

# %%
sc.pp.neighbors(adata_sub1,n_pcs=0,use_rep='X_integrated')
sc.tl.umap(adata_sub1)

# %%
res=1.5

# %%
sc.tl.leiden(adata_sub1, resolution=res, key_added='leiden_r'+str(res), 
             directed=True, use_weights=True)

# %%
np.random.seed(0)
random_indices=np.random.permutation(list(range(adata_sub1.shape[0])))
sc.pl.umap(adata_sub1[random_indices,:],vmin=0,vmax=1,
           color=[score+'_score_scaled' for score in scores]+[
               'leiden_r'+str(res),'study','cell_type'],
          sort_order=False,wspace=0.4)

# %%
sc.pl.umap(adata_sub1, 
           color=[score+'_score_scaled' for score in scores]+['leiden_r'+str(res)],
           vmin=0,vmax=1, wspace=0.4)

# %% [markdown]
# C: Subcluster 7: 4_7 seems to be alpha_beta_gamma. Subcluster 9: 4_9 may be beta_gamma, but this is not likely - socre seems to be lower than in true gama multiplets and it also is not preseent in reference annotation of Fltp_2y. The same goes for 4_7 which is probably alpha_beta.

# %%
ct='7'
col='is_'+ct
adata_sub1.obs[col]=adata_sub1.obs['leiden_r'+str(res)]==ct
sc.pl.umap(adata_sub1,color=col,s=20)
adata_sub1.obs.drop([col],axis=1,inplace=True)

# %%
proportions2=[]
for group,data in adata_sub1.obs.groupby('leiden_r'+str(res)):
    for gene in ['ins','gcg','sst','ppy']:
        proportion=data[gene+'_high'].value_counts(normalize=True,sort=False)
        if True in proportion.index:
            proportion=proportion[True]
        else:
            proportion=0
        proportions2.append({'cluster':group,'gene':gene,
                            'proportion':proportion})
proportions2=pd.DataFrame(proportions2)
g = sb.catplot(
    data=proportions2, kind="bar",
    x="cluster", y="proportion", hue="gene", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("Cluster", "Porportion of hormone high cells")

# %% [markdown]
# Subcluster 7 needs to be further subclustered.

# %% [markdown]
# Subcluster

# %%
CL1='7'
clusters=[CL1]
adata_sub2=adata_sub1[adata_sub1.obs['leiden_r'+str(res)].isin(clusters)]
adata_sub2.obs.drop([col for col in adata_sub2.obs.columns if 'leiden_r' in col],
                   axis=1,inplace=True)
#(413, 13999)
adata_sub2.shape

# %%
sc.pp.neighbors(adata_sub2,n_pcs=0,use_rep='X_integrated')
sc.tl.umap(adata_sub2)

# %%
res=0.4

# %%
sc.tl.leiden(adata_sub2, resolution=res, key_added='leiden_r'+str(res), 
             directed=True, use_weights=True)

# %%
np.random.seed(0)
random_indices=np.random.permutation(list(range(adata_sub2.shape[0])))
sc.pl.umap(adata_sub2[random_indices,:],vmin=0,vmax=1,
           color=[score+'_score_scaled' for score in scores]+[
               'leiden_r'+str(res),'study','cell_type'],
          sort_order=False,wspace=0.4)

# %% [markdown]
# C: Here the number of cells is low so maybe less neighbours should be used. However, even when using only 5 neighbours the smaller alpha_beta population did not get separated out.

# %%
proportions2=[]
for group,data in adata_sub2.obs.groupby('leiden_r'+str(res)):
    for gene in ['ins','gcg','sst','ppy']:
        proportion=data[gene+'_high'].value_counts(normalize=True,sort=False)
        if True in proportion.index:
            proportion=proportion[True]
        else:
            proportion=0
        proportions2.append({'cluster':group,'gene':gene,
                            'proportion':proportion})
proportions2=pd.DataFrame(proportions2)
g = sb.catplot(
    data=proportions2, kind="bar",
    x="cluster", y="proportion", hue="gene", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("Cluster", "Porportion of hormone high cells")

# %%
adata_sub.obs.loc[adata_sub2.obs_names,CL_RES_SUB1+'_subclustered'
                 ]=[CL+'_'+CL1+'_'+cl for cl in adata_sub2.obs['leiden_r'+str(res)]]

# %%
adata_sub.obs[CL_RES_SUB1+'_subclustered' ].value_counts()

# %% [markdown]
# C: Cluster 2: 4_7_2 is alpha_beta.

# %% [markdown]
# ##### Subcluster cluster 16

# %%
CL='16'

# %%
clusters=[CL]
adata_sub1=adata_sub[adata_sub.obs[CL_RES_SUB1].isin(clusters)]
adata_sub1.obs.drop([col for col in adata_sub1.obs.columns if 'leiden_r' in col],
                   axis=1,inplace=True)
#(54, 13999)
adata_sub1.shape

# %%
sc.pp.neighbors(adata_sub1,n_pcs=0,use_rep='X_integrated')
sc.tl.umap(adata_sub1)

# %%
res=0.5

# %%
sc.tl.leiden(adata_sub1, resolution=res, key_added='leiden_r'+str(res), 
             directed=True, use_weights=True)

# %%
np.random.seed(0)
random_indices=np.random.permutation(list(range(adata_sub1.shape[0])))
sc.pl.umap(adata_sub1[random_indices,:],
           color=[score+'_score_scaled' for score in scores]+['leiden_r'+str(res)],
          sort_order=False)

# %% [markdown]
# C: Subclustering this cluster would not make sense.

# %% [markdown]
# #### Try to use less neighbors to see if this separates doublets better

# %%
adata_sub2=adata_sub.copy()
sc.pp.neighbors(adata_sub2,n_neighbors=5,n_pcs=0,use_rep='X_integrated')
sc.tl.umap(adata_sub2)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(adata_sub2,s=10,
           color=['ins_high','gcg_high','sst_high','ppy_high'])

# %%
res=0.5
sc.tl.leiden(adata_sub2, resolution=res, key_added='leiden_r'+str(res), directed=True, use_weights=True)

# %%
sc.pl.umap(adata_sub2,s=10,color=['leiden_r0.5','leiden_r0.7','leiden_r1','study'],wspace=0.3)

# %%
proportions2=[]
for group,data in adata_sub2.obs.groupby('leiden_r'+str(0.5)):
    for gene in ['ins','gcg','sst','ppy']:
        proportion=data[gene+'_high'].value_counts(normalize=True,sort=False)
        if True in proportion.index:
            proportion=proportion[True]
        else:
            proportion=0
        proportions2.append({'cluster':group,'gene':gene,
                            'proportion':proportion})
proportions2=pd.DataFrame(proportions2)
g = sb.catplot(
    data=proportions2, kind="bar",
    x="cluster", y="proportion", hue="gene", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("Cluster", "Porportion of hormone high cells")

# %%
proportions2=[]
for group,data in adata_sub2.obs.groupby('leiden_r'+str(0.7)):
    for gene in ['ins','gcg','sst','ppy']:
        proportion=data[gene+'_high'].value_counts(normalize=True,sort=False)
        if True in proportion.index:
            proportion=proportion[True]
        else:
            proportion=0
        proportions2.append({'cluster':group,'gene':gene,
                            'proportion':proportion})
proportions2=pd.DataFrame(proportions2)
g = sb.catplot(
    data=proportions2, kind="bar",
    x="cluster", y="proportion", hue="gene", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("Cluster", "Porportion of hormone high cells")

# %%
del adata_sub2

# %% [markdown]
# #### Add subclustering anno

# %%
CLUSTERING_SUB2=CL_RES_SUB1+'_subclustered'

# %%
adata_sub.obs[CLUSTERING_SUB2].value_counts()

# %%
low_quality_cl=[cl for cl in adata_sub.obs[CLUSTERING_SUB2].unique()
                      if cl.startswith('11_') and 
                cl not in ['11_0','11_1','11_2','11_5','11_6','11_7','11_8','11_9']]
print('Low quality',low_quality_cl)

# %%
# Add beta cell annotation to whole adata
h.add_category(df=adata.obs,idxs=adata_sub.obs.index[
    ~adata_sub.obs[CLUSTERING_SUB2].isin(['9','14','7_3','4_7_2']+low_quality_cl)],
               col='cell_type_integrated_v1',category='beta')
# Add beta-delta cell annotation to whole data
h.add_category(df=adata.obs,idxs=adata_sub.obs.index[adata_sub.obs[CLUSTERING_SUB2]=="9"],
               col='cell_type_integrated_v1',category='beta_delta')
# Add alpha_beta cell annotation to whole data
h.add_category(df=adata.obs,idxs=adata_sub.obs.index[adata_sub.obs[CLUSTERING_SUB2]=="14"],
               col='cell_type_integrated_v1',category='alpha_beta')
# Add polyhormonal cell annotation to whole data
h.add_category(df=adata.obs,idxs=adata_sub.obs.index[
    adata_sub.obs[CLUSTERING_SUB2].isin(low_quality_cl)],
               col='cell_type_integrated_v1',category='ambient')
# Add beta-gamma cell annotation to whole data
h.add_category(df=adata.obs,idxs=adata_sub.obs.index[adata_sub.obs[CLUSTERING_SUB2]=="7_3"],
               col='cell_type_integrated_v1',category='beta_gamma')
# Add alpha_beta cell annotation to whole data
h.add_category(df=adata.obs,idxs=adata_sub.obs.index[adata_sub.obs[CLUSTERING_SUB2]=="4_7_2"],
               col='cell_type_integrated_v1',category='alpha_beta')

# %%
# Add info to adata sub
adata_sub.obs['cell_type_integrated_v1']=adata[adata_sub.obs_names,:].obs['cell_type_integrated_v1']

# %% [markdown]
# C: Low_quality was renamed to ambient.

# %%
rcParams['figure.figsize']=(6,6)
np.random.seed(0)
random_indices=np.random.permutation(list(range(adata_sub.shape[0])))
sc.pl.umap(adata_sub[random_indices,:],color='cell_type_integrated_v1',s=10)

# %%
rcParams['figure.figsize']=(10,10)
np.random.seed(0)
random_indices=np.random.permutation(list(range(adata.shape[0])))
sc.pl.umap(adata[random_indices,:],color='cell_type_integrated_v1',s=10)

# %% [markdown]
# Update annotation - save

# %%
path_data+'data_integrated_analysed.h5ad'

# %%
# Save new anno
h.update_adata(
    adata_new=adata, path=path_data+'data_integrated_analysed.h5ad', io_copy=False,
    add=[('obs',True,'cell_type_integrated_v1','cell_type_integrated_v1')],rm=None)

# %% [markdown]
# ### Keep only beta

# %%
adata_beta=adata[adata.obs['cell_type_integrated_v1']=='beta']
adata_beta.obs.drop([col for col in adata_beta.obs.columns if 'leiden_r' in col],
                   axis=1,inplace=True)

# %%
adata_beta.shape

# %% [markdown]
# #### Analyse beta cells

# %%
sc.pp.neighbors(adata_beta,n_pcs=0,use_rep='X_integrated')
sc.tl.umap(adata_beta)

# %%
rcParams['figure.figsize']=(6,6)
cell_type='ins_low'
adata_beta.obs[cell_type]=~adata_beta.obs.ins_high.astype('int')
sc.pl.umap(adata_beta,color=cell_type,s=10)
adata_beta.obs.drop(cell_type,axis=1,inplace=True)

# %%
random_indices=np.random.permutation(list(range(adata_beta.shape[0])))
sc.pl.umap(adata_beta[random_indices,:], color=['study'] ,size=10, use_raw=False,ncols=1)

# %% [markdown]
# ##### Other hormones

# %%
rcParams['figure.figsize']=(7,7)
sc.pl.umap(adata_beta,color=[score+'_score_scaled' for score in scores],vmin=0,vmax=1,s=20)

# %% [markdown]
# C: There may still be some residoual doublets. However, it seems that most of them were removed.

# %% [markdown]
# ##### Markers

# %%
markers=pd.read_excel('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/markers.xlsx', 
                      sheet_name='mice')

# %%
# prepare adata for plotting - gene symbols in var_names
adata_rawnorm_sub_temp=adata_rawnorm_beta.copy()
adata_rawnorm_sub_temp.var_names=adata_rawnorm_sub_temp.var['gene_symbol'].astype('str')
adata_rawnorm_sub_temp.var_names=adata_rawnorm_sub_temp.var_names.fillna('NA')
adata_rawnorm_sub_temp.var_names_make_unique()

# %%
for group_name, group in markers.query('Cell in ["beta","endocrine"]').fillna('/').groupby(['Cell','Subtype']):
    print(group_name)
    genes=group.Gene
    # Retain genes present in raw var_names
    missing=[gene for gene in genes if gene not in adata_rawnorm_sub_temp.var_names]
    genes=[gene for gene in genes if gene in adata_rawnorm_sub_temp.var_names]
    print('Missing genes:',missing)
    rcParams['figure.figsize']=(4,4)
    sc.pl.umap(adata_rawnorm_sub_temp, color=genes ,size=10, use_raw=False)

# %%
del adata_rawnorm_sub_temp

# %% [markdown]
# C: Some cells have very low expression of most of the markers. Thus check if they may be low-quality cells.

# %% [markdown]
# ##### QC

# %% [markdown]
# Check QC metrics

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata_beta,color=['n_genes','n_counts','mt_frac'],s=20)

# %% [markdown]
# C: There are twoi subpopulations with low N genes, one of them has also more mt counts.

# %% [markdown]
# Check ambient scores of those cells.

# %%
sc.pl.violin(adata_beta,keys='emptyDrops_LogProb_scaled',groupby='study_sample')

# %%
rcParams['figure.figsize']=(7,7)
random_indices=np.random.permutation(list(range(adata_beta.shape[0])))
sc.pl.umap(adata_beta[random_indices,:],color='emptyDrops_LogProb_scaled',s=10,sort_order=False)

# %% [markdown]
# ##### Subpopulations

# %%
design_order=['mRFP','mTmG','mGFP',
              'head_Fltp-','tail_Fltp-', 'head_Fltp+', 'tail_Fltp+',
              'IRE1alphafl/fl','IRE1alphabeta-/-', 
              '8w','14w', '16w',
              'DMSO_r1','DMSO_r2', 'DMSO_r3','GABA_r1','GABA_r2', 'GABA_r3',
                   'A1_r1','A1_r2','A1_r3','A10_r1','A10_r2', 'A10_r3',  'FOXO_r1', 'FOXO_r2', 'FOXO_r3', 
              'E12.5','E13.5','E14.5', 'E15.5', 
              'chow_WT','sham_Lepr-/-','PF_Lepr-/-','VSG_Lepr-/-',   
              'control','STZ', 'STZ_GLP-1','STZ_estrogen', 'STZ_GLP-1_estrogen',
                  'STZ_insulin','STZ_GLP-1_estrogen+insulin' 
            ]

# %%
# Prepare figure grid
fig_rows=adata_beta.obs.study.unique().shape[0]
fig_cols=max(
    [adata_beta.obs.query('study ==@study').file.unique().shape[0] 
     for study in adata_beta.obs.study.unique()])
rcParams['figure.figsize']= (6*fig_cols,6*fig_rows)
fig,axs=plt.subplots(fig_rows,fig_cols)
# Calculagte density
for row_idx,study in enumerate(adata_beta.obs.study.unique()):
    designs=adata_beta.obs.query('study ==@study')[['study_sample_design','design']].drop_duplicates()
    designs.design=pd.Categorical(designs.design, 
                      categories=[design for design in design_order if design in list(designs.design.values)],
                      ordered=True)
    #designs=dict(zip(designs['study_sample_design'],designs['design']))
    #designs={k: v for k, v in sorted(designs.items(), key=lambda item: item[1])}
    for col_idx,sample in enumerate(designs.sort_values('design')['study_sample_design'].values):
        subset=adata_beta.obs.study_sample_design==sample
        adata_sub_ss=adata_beta[subset,:]
        if adata_sub_ss.shape[0]>=20:
            print(sample,adata_sub_ss.shape)
            sc.tl.embedding_density(adata_sub_ss)
            sc.pl.umap(adata_beta,ax=axs[row_idx,col_idx],s=10,show=False)
            sc.pl.embedding_density(adata_sub_ss,ax=axs[row_idx,col_idx],title=sample,show=False) 

# %%
for res in [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5]:
    print(res)
    sc.tl.leiden(adata_beta, resolution=res, key_added='leiden_r'+str(res), 
                 directed=True, use_weights=True)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(adata_beta,s=10,color=['leiden_r0.2','leiden_r0.3','leiden_r0.4','leiden_r0.5',
                                 'leiden_r0.6','leiden_r0.7','leiden_r0.8','leiden_r0.9',
                                 'leiden_r1','leiden_r1.5',
                                 'study'],wspace=0.3)

# %% [markdown]
# ### Save beta data
# Before different beta data was saved as the polyhormonal cells were not annotated with subclustering.

# %%
adata_beta.write(path_data+'data_integrated_analysed_beta_v1.h5ad')

# %% [markdown]
# Save rawnorm beta data

# %%
# Make rawnorm beta cell data
adata_rawnorm_beta=anndata.AnnData(adata_rawnorm[adata_beta.obs_names,].X.copy(),
               obs=adata_beta.obs.copy(),
               var=adata_rawnorm.var.copy(),
               uns=adata_beta.uns.copy(),
               obsm=adata_beta.obsm.copy(),obsp=adata_beta.obsp.copy())

# %%
adata_rawnorm_beta

# %%
# Save rawnorm beta cell data
adata_rawnorm_beta.write(path_data+'data_rawnorm_integrated_analysed_beta_v1.h5ad')

# %% [markdown]
# ### Reannotate part of beta cells

# %%
adata.obs.loc[adata_beta[adata_beta.obs['leiden_r1.5']=='21'].obs_names,
    'cell_type_integrated_v1']='stellate_activated'

# %%
rcParams['figure.figsize']=(10,10)
np.random.seed(0)
random_indices=np.random.permutation(list(range(adata.shape[0])))
sc.pl.umap(adata[random_indices,:],color='cell_type_integrated_v1',s=10)

# %%
# remove cells from the specified cluster
adata_beta=adata_beta[adata_beta.obs['leiden_r1.5']!='21']
# remove data not associated with new embedding
adata_beta.obs.drop([col for col in adata_beta.obs.columns if 'leiden' in col],
                    axis=1,inplace=True)
adata_beta.uns.clear()
del adata_beta.obsm['X_umap']
adata_beta.obsp.clear()

# %%
print(adata_beta.shape)

# %%
# redo UMAp and neighbours
sc.pp.neighbors(adata_beta,n_pcs=0,use_rep='X_integrated')
sc.tl.umap(adata_beta)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(adata_beta,color=['n_genes','n_counts','mt_frac','emptyDrops_LogProb_scaled'],
           s=20)

# %%
# Redo clustering
for res in [0.4,0.7,1,1.5,2]:
    print(res)
    sc.tl.leiden(adata_beta, resolution=res, key_added='leiden_r'+str(res), 
                 directed=True, use_weights=True)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(adata_beta,s=10,color=[
    'leiden_r0.4','leiden_r0.7','leiden_r1','leiden_r1.5','leiden_r2','study'],wspace=0.3)

# %%
# make new rawnorm data
adata_rawnorm_beta=adata_rawnorm_beta[adata_beta.obs_names,:]
adata_rawnorm_beta.obs=adata_beta.obs.copy()
adata_rawnorm_beta.uns=adata_beta.uns.copy()
adata_rawnorm_beta.obsm=adata_beta.obsm.copy()
adata_rawnorm_beta.obsp=adata_beta.obsp.copy()

# %%
adata_rawnorm_beta

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(adata_beta,color=  ['gcg_score_scaled', 'ins_score_scaled', 'ppy_score_scaled', 
                               'sst_score_scaled'],s=30)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(adata_beta,color=  ['gcg_score_scaled', 'ins_score_scaled', 'ppy_score_scaled', 
                               'sst_score_scaled'],s=30,sort_order=False)

# %%
rcParams['figure.figsize']=(6,6)
cell_type='ins_low'
adata_beta.obs[cell_type]=~adata_beta.obs.ins_high.astype('int')
sc.pl.umap(adata_beta,color=cell_type,s=10)
adata_beta.obs.drop(cell_type,axis=1,inplace=True)

# %%
random_indices=np.random.permutation(list(range(adata_beta.shape[0])))
sc.pl.umap(adata_beta[random_indices,:], color=['study'] ,size=10, use_raw=False,ncols=1)

# %%
# Check markers from other cts
rcParams['figure.figsize']=(8,8)
markers_list=['Plvap','Pecam1','Ndufa4l2','Acta2','Pdgfrb','Col1a2','Cd86',
                            'Ptprc','Cpa1','Car2','Sox9','Cryab']
markers_id=[adata_rawnorm_beta.var.query('gene_symbol==@g').index[0] for g in markers_list]
sc.pl.umap(adata_rawnorm_beta,color=markers_id,s=30)

# %%
# Save beta data
adata_beta.write(path_data+'data_integrated_analysed_beta_v1s1.h5ad')

# %%
# Save rawnorm beta cell data
adata_rawnorm_beta.write(path_data+'data_rawnorm_integrated_analysed_beta_v1s1.h5ad')

# %% [markdown]
# ## Alpha and gamma cells

# %%
clusters=['1']
adata_sub=adata[adata.obs[CLUSTERING].isin(clusters)]
adata_sub.obs.drop([col for col in adata_sub.obs.columns if 'leiden_r' in col],
                   axis=1,inplace=True)

# %%
#(48011, 13786)
adata_sub.shape

# %%
sc.pp.neighbors(adata_sub,n_pcs=0,use_rep='X_integrated')
sc.tl.umap(adata_sub)

# %% [markdown]
# Quality metrics

# %%
sc.pl.violin(adata_sub,keys='emptyDrops_LogProb_scaled',groupby='study_sample',stripplot=False)

# %%
rcParams['figure.figsize']=(7,5)
sc.pl.violin(adata_sub,keys='emptyDrops_LogProb_scaled',groupby='study',stripplot=False)

# %%
rcParams['figure.figsize']=(6,6)
random_indices=np.random.permutation(list(range(adata_sub.shape[0])))
sc.pl.umap(adata_sub[random_indices,:]
           ,color=['n_genes','n_counts','mt_frac','emptyDrops_LogProb_scaled'],
           s=10,sort_order=False)

# %% [markdown]
# C: There might be some low-quality alpha cells, but as long as markers are not analysed these will not be removed as they seem to mainly come from single study and represent large proportion of cells. High mt proportion indicates that these may be damaged and not just ambient cells.

# %%
random_indices=np.random.permutation(list(range(adata_sub.shape[0])))
sc.pl.umap(adata_sub[random_indices,:],color=['study'],  s=10,sort_order=False)

# %%
sc.pl.umap(adata_sub,s=10,
           color=['ins_high','gcg_high','sst_high','ppy_high'])

# %% [markdown]
# C: there seem to be some potential doublet populations, but nothing very distinctive (also see cluster composition of horomne high/low below).

# %%
# prepare adata for plotting - gene symbols in var_names
adata_rawnorm_sub_temp=anndata.AnnData(adata_rawnorm[adata_sub.obs_names,].X.copy(),
               obs=adata_sub.obs.copy(),
               var=adata_rawnorm.var.copy(),
               uns=adata_sub.uns.copy(),
               obsm=adata_sub.obsm.copy(),obsp=adata_sub.obsp.copy())
adata_rawnorm_sub_temp.var_names=adata_rawnorm_sub_temp.var['gene_symbol'].astype('str')
adata_rawnorm_sub_temp.var_names=adata_rawnorm_sub_temp.var_names.fillna('NA')
adata_rawnorm_sub_temp.var_names_make_unique()

# %%
# Plot markers of some other cell types
rcParams['figure.figsize']=(8,8)
markers_list=['Plvap','Pecam1','Ndufa4l2','Acta2','Pdgfrb','Col1a2','Cd86',
                            'Ptprc','Cpa1','Car2','Sox9','Cryab']
#markers_id=[adata_rawnorm_sub_temp.var.query('gene_symbol==@g').index[0] for g in markers_list]
sc.pl.umap(adata_rawnorm_sub_temp,color=markers_list,s=20)

# %% [markdown]
# C: there seems to be a subpopulation of activated stellate cells with hig hambience of endocrine hormones.

# %%
del adata_rawnorm_sub_temp

# %%
res=1

# %%
sc.tl.leiden(adata_sub, resolution=res, key_added='leiden_r'+str(res), directed=True, use_weights=True)

# %%
rcParams['figure.figsize']=(10,10)
sc.pl.umap(adata_sub,color=['leiden_r1'],s=30)

# %% [markdown]
# C: clustering for annotation of activated stellate (above)

# %%
adata_sub.obs['leiden_r1'].value_counts()

# %%
adata_rawnorm_sub_temp.obs=adata_sub.obs.copy()
sc.pl.violin(adata=adata_rawnorm_sub_temp, keys=['Ndufa4l2','Acta2','Pdgfrb','Col1a2'], 
             groupby='leiden_r1', log=False, use_raw=False, stripplot=False)

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata_sub,s=10,color=['leiden_r1'])

# %%
# These clusters were used for 1st version of annotation
sc.pl.umap(adata_sub,s=10,color=['leiden_r0.3','leiden_r0.5','leiden_r0.7','leiden_r1','leiden_r1.5'])

# %% [markdown]
# Check if any of the clusters is ritch in potential multiplets

# %%
CLUSTERING_SUB='leiden_r1'

# %%
proportions=[]
for group,data in adata_sub.obs.groupby(CLUSTERING_SUB):
    #print('***',group,'***')
    #ct_count=data['cell_type'].value_counts()
    #print('** Cell type')
    #print(ct_count[ct_count!=0])
    #print('Ins_high:\n',data['ins_high'].value_counts(normalize=True,sort=False)[True])
    #print('Gcg_high:\n',data['gcg_high'].value_counts(normalize=True,sort=False)[True])
    #print('Sst_high:\n',data['sst_high'].value_counts(normalize=True,sort=False)[True])
    #print('Ppy_high:\n',data['ppy_high'].value_counts(normalize=True,sort=False)[True])
    for gene in ['ins','gcg','sst','ppy']:
        proportion=data[gene+'_high'].value_counts(normalize=True,sort=False)
        if True in proportion.index:
            proportion=proportion[True]
        else:
            proportion=0
        proportions.append({'cluster':group,'gene':gene,
                            'proportion':proportion})
proportions=pd.DataFrame(proportions)

# %%
g = sb.catplot(
    data=proportions, kind="bar",
    x="cluster", y="proportion", hue="gene", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("Cluster", "Porportion of hormone high cells")


# %%
for group,data in adata_sub.obs.groupby(CLUSTERING_SUB):
    print('***',group,'***')
    ct_count=data['cell_type'].value_counts()
    print('** Cell type')
    print(ct_count[ct_count!=0])
    print('\n')

# %% [markdown]
# #C: No cluster has very high proportion of more than one hormone high cells. Thus there is likely no doublet clusters - potential doublet cells are intermixed with non-doublet cells. As additional validation gcg and ppy scores are plotted, below, also showing distribution in studies to make check if there is study-bias.

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata_sub,s=10,
           color=['ppy_score','gcg_score','study','cell_type'])

# %%
ct='alpha_gamma'
col='is_'+ct
adata_sub.obs[col]=adata_sub.obs['cell_type']==ct
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata_sub,color=col,s=20)
adata_sub.obs.drop([col],axis=1,inplace=True)

# %% [markdown]
# C: The previously annotated AG cells do not seem to cluster together and are not separable by clustering.

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(adata_sub,color=[score+'_score_scaled' for score in scores],s=20)

# %% [markdown]
# C: there are no very clear regions high in Gcg and Ppy, thus the gigh annotation done previously might have been due to specific selected thresholds, biased in cretain studies (e.g. Fltp_adult and VSG).

# %%
rcParams['figure.figsize']=(12,4)
sb.violinplot(x="study", y="ppy_score", data=adata_sub.obs,inner=None)

# %%
rcParams['figure.figsize']=(12,4)
sb.violinplot(x="study", y="gcg_score", data=adata_sub.obs,inner=None)

# %% [markdown]
# #C: There seem to be no doublet populations based on gcg and ppy scores.

# %% [markdown]
# #### Subcluster cl 11 to get activated stellate

# %%
adata_sub2=adata_sub[adata_sub.obs['leiden_r1']=='11'].copy()

# %%
sc.pp.neighbors(adata_sub2,n_pcs=0,use_rep='X_integrated')
sc.tl.umap(adata_sub2)

# %%
# Plot markers of some other cell types
rcParams['figure.figsize']=(5,5)
markers_list=['Ndufa4l2','Acta2','Pdgfrb']
markers_id=[adata_sub2.var.query('gene_symbol==@g').index[0] for g in markers_list]
sc.pl.umap(adata_sub2,color=markers_id,s=100)

# %%
res=0.2
sc.tl.leiden(adata_sub2, resolution=res, key_added='leiden_r'+str(res), 
             directed=True, use_weights=True)

# %%
sc.pl.umap(adata_sub2,color='leiden_r0.2')

# %% [markdown]
# #### Add alpha and gamma annotation to whole data

# %%
# Add alpha cell annotation to whole adata
h.add_category(df=adata.obs,idxs=adata_sub.obs.index[adata_sub.obs['leiden_r0.5']!="3"],
               col='cell_type_integrated_v1',category='alpha')
# Add gamma cell annotation to whole data
h.add_category(df=adata.obs,idxs=adata_sub.obs.index[adata_sub.obs['leiden_r0.5']=="3"],
               col='cell_type_integrated_v1',category='gamma')

# %%
# Add alpha cell annotation to whole adata
h.add_category(df=adata.obs,idxs=adata_sub2.obs.index[adata_sub2.obs['leiden_r0.2']=="1"],
               col='cell_type_integrated_v1',category='stellate_activated')

# %%
rcParams['figure.figsize']=(10,10)
sc.pl.umap(adata,color='cell_type_integrated_v1',s=10)

# %% [markdown]
# ## Delta cells

# %%
clusters=['4']
adata_sub=adata[adata.obs[CLUSTERING].isin(clusters)]
adata_sub.obs.drop([col for col in adata_sub.obs.columns if 'leiden_r' in col],
                   axis=1,inplace=True)

# %%
#(28509, 13786)
adata_sub.shape

# %%
sc.pp.neighbors(adata_sub,n_pcs=0,use_rep='X_integrated')
sc.tl.umap(adata_sub)

# %% [markdown]
# QC

# %%
rcParams['figure.figsize']=(7,5)
sc.pl.violin(adata_sub,keys='emptyDrops_LogProb_scaled',groupby='study_sample',stripplot=False)

# %%
rcParams['figure.figsize']=(7,5)
sc.pl.violin(adata_sub,keys='emptyDrops_LogProb_scaled',groupby='study',stripplot=False)

# %%
rcParams['figure.figsize']=(6,6)
random_indices=np.random.permutation(list(range(adata_sub.shape[0])))
sc.pl.umap(adata_sub[random_indices,:]
           ,color=['n_genes','n_counts','mt_frac','emptyDrops_LogProb_scaled'],
           s=10,sort_order=False)

# %%
sc.pl.umap(adata_sub,s=10, color='study')

# %% [markdown]
# C: there seem to be some low-quality populations. However, the larger population (see clusters below) will not be removed as it contains too many cells. The small population will be removed. It likely corresponds to the cluster that clusters separately on embedding of all cells as well. The larger population is likely damaged cells (high mt) while the smaller population is likely ambient cells - high ambient score and low genes, counts, and mt.

# %%
sc.pl.umap(adata_sub,s=10,
           color=['ins_high','gcg_high','sst_high','ppy_high'])

# %%
res=1

# %%
sc.tl.leiden(adata_sub, resolution=res, key_added='leiden_r'+str(res), directed=True, 
             use_weights=True)

# %%
sc.pl.umap(adata_sub,s=10,color=['leiden_r0.5','leiden_r0.7','leiden_r1','leiden_r1.5'],wspace=0.3)

# %% [markdown]
# C: It seems that leiden 1.5 could resolve more populations than leiden 1. However, these populations would mainly separate Fltp_P16 from other studies, not corresponding to previous cell type annotation (see below). Thus this might be only due to too low thershold for Ppy+ calling in this study. When plotting 0-1 normalised scores the res=1 clustering seems to suffice.

# %%
CLUSTERING_SUB='leiden_r1'

# %%
proportions=[]
for group,data in adata_sub.obs.groupby(CLUSTERING_SUB):
    #print('***',group,'***')
    #ct_count=data['cell_type'].value_counts()
    #print('** Cell type')
    #print(ct_count[ct_count!=0])
    #print('Ins_high:\n',data['ins_high'].value_counts(normalize=True,sort=False)[True])
    #print('Gcg_high:\n',data['gcg_high'].value_counts(normalize=True,sort=False)[True])
    #print('Sst_high:\n',data['sst_high'].value_counts(normalize=True,sort=False)[True])
    #print('Ppy_high:\n',data['ppy_high'].value_counts(normalize=True,sort=False)[True])
    for gene in ['ins','gcg','sst','ppy']:
        proportion=data[gene+'_high'].value_counts(normalize=True,sort=False)
        if True in proportion.index:
            proportion=proportion[True]
        else:
            proportion=0
        proportions.append({'cluster':group,'gene':gene,
                            'proportion':proportion})
proportions=pd.DataFrame(proportions)

# %%
g = sb.catplot(
    data=proportions, kind="bar",
    x="cluster", y="proportion", hue="gene", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("Cluster", "Porportion of hormone high cells")


# %% [markdown]
# #C: Clusters 12 (beta-delta) and 6 (alpha-delta) are most likely doublets. This is less clear for clusters 11, 13, and 5. As cluster 13 (beta-delta) clearly spearates from the main beta-delta cluster (in location and Ins strength) this may in fact not be a doublet population. Cell identitiy is less clear for gamma-delta clusters 11 and 5. Thus the raw ppy scores are plotted below. As they may not be comparable across studies their distribution is also plotted per study.

# %%
proportions=[]
for group,data in adata_sub.obs.groupby('leiden_r1.5'):

    for gene in ['ins','gcg','sst','ppy']:
        proportion=data[gene+'_high'].value_counts(normalize=True,sort=False)
        if True in proportion.index:
            proportion=proportion[True]
        else:
            proportion=0
        proportions.append({'cluster':group,'gene':gene,
                            'proportion':proportion})
proportions=pd.DataFrame(proportions)
g = sb.catplot(
    data=proportions, kind="bar",
    x="cluster", y="proportion", hue="gene", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("Cluster", "Porportion of hormone high cells")

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata_sub,s=20, color=['cell_type'])

# %%
for ct in ['delta_gamma','beta_delta','alpha_delta','alpha_delta_gamma']:
    col='is_'+ct
    adata_sub.obs[col]=adata_sub.obs['cell_type']==ct
    rcParams['figure.figsize']=(6,6)
    sc.pl.umap(adata_sub,color=col,s=20)
    adata_sub.obs.drop([col],axis=1,inplace=True)

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata_sub,s=10,
           color=['ppy_score','study'])

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(adata_sub,color=[score+'_score_scaled' for score in scores],s=20)

# %% [markdown]
# C: The current annotation corresponds to normalised scores. Note: The scores are not completely comparable as the distributions are still shifted across studies. But they should serve as an approximation.

# %%
rcParams['figure.figsize']=(12,6)
sb.violinplot(x="study", y="ppy_score_scaled", data=adata_sub.obs,inner=None)

# %% [markdown]
# #C: ppy scores seem to be comparable across most of the studies, separating into low, medium, and small very high regions. Based on this thereb seem to be single ppy high region (cluster 11). Main studies in clusters 11 (STZ) and 5 (FLTP_adult and Fltp_P16) seem to have similar ppy score distribution, but cluster 11 has much higher ppy score, thus cluster 5 probably is not true multiplet cluster.

# %% [markdown]
# #### Add delta and gdelta multiplet annotation to whole data

# %% [markdown]
# Old annotation

# %%
# Add alpha cell annotation to whole adata
h.add_category(df=adata.obs,idxs=adata_sub.obs.index[
    ~adata_sub.obs[CLUSTERING_SUB].isin(['12','11','6'])],
               col='cell_type_integrated_v1',category='delta')
# Add beta-delta cell annotation to whole data
h.add_category(df=adata.obs,idxs=adata_sub.obs.index[adata_sub.obs[CLUSTERING_SUB]=="12"],
               col='cell_type_integrated_v1',category='beta_delta')
# Add alpha-delta cell annotation to whole data
h.add_category(df=adata.obs,idxs=adata_sub.obs.index[adata_sub.obs[CLUSTERING_SUB]=="6"],
               col='cell_type_integrated_v1',category='alpha_delta')
# Add delta-gamma cell annotation to whole data
h.add_category(df=adata.obs,idxs=adata_sub.obs.index[adata_sub.obs[CLUSTERING_SUB]=="11"],
               col='cell_type_integrated_v1',category='delta_gamma')

# %% [markdown]
# New annotation

# %%
# Add alpha cell annotation to whole adata
h.add_category(df=adata.obs,idxs=adata_sub.obs.index[
    ~adata_sub.obs[CLUSTERING_SUB].isin(['12','11','6'])],
               col='cell_type_integrated_v1',category='delta')
# Add beta-delta cell annotation to whole data
h.add_category(df=adata.obs,idxs=adata_sub.obs.index[adata_sub.obs[CLUSTERING_SUB]=="12"],
               col='cell_type_integrated_v1',category='beta_delta')
# Add alpha-delta cell annotation to whole data
h.add_category(df=adata.obs,idxs=adata_sub.obs.index[adata_sub.obs[CLUSTERING_SUB]=="6"],
               col='cell_type_integrated_v1',category='alpha_delta')
# Add delta-gamma cell annotation to whole data
h.add_category(df=adata.obs,idxs=adata_sub.obs.index[adata_sub.obs[CLUSTERING_SUB]=="11"],
               col='cell_type_integrated_v1',category='delta_gamma')
# Add low_quality cell annotation to whole data
h.add_category(df=adata.obs,idxs=adata_sub.obs.index[adata_sub.obs[CLUSTERING_SUB]=="13"],
               col='cell_type_integrated_v1',category='ambient')

# %%
rcParams['figure.figsize']=(10,10)
sc.pl.umap(adata,color='cell_type_integrated_v1',s=10)

# %%
adata_sub.obs['cell_type_integrated_v1']=adata.obs.loc[adata_sub.obs_names,'cell_type_integrated_v1']

# %% [markdown]
# C: Low quality was renamed to ambient.

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata_sub,color='cell_type_integrated_v1',s=10)

# %%
# Save new anno
h.update_adata(
    adata_new=adata, path=path_data+'data_integrated_analysed.h5ad', io_copy=False,
    add=[('obs',True,'cell_type_integrated_v1','cell_type_integrated_v1')],rm=None)

# %% [markdown]
# ### Reannotate some other cell types

# %%
res=0.5
sc.tl.leiden(adata, resolution=res, key_added='leiden_r'+str(res), directed=True, use_weights=True)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(adata,color='leiden_r0.5',s=20)

# %%
adata.obs

# %%
#Add scwann ct anno
h.add_category(df=adata.obs,idxs=adata.obs_names[adata.obs['leiden_r0.5']=="17"],
               col='cell_type_integrated_v1',category='schwann')

# %%
# replace pericyte and fibroblast related with stellate
adata.obs['cell_type_integrated_v1']=adata.obs['cell_type_integrated_v1'].replace(
    {'pericyte':'stellate_activated','fibroblast_related':'stellate_quiescent'})

# %%
h.update_adata(
    adata_new=adata, path=path_data+'data_integrated_analysed.h5ad', io_copy=False,
    add=[('obs',True,'cell_type_integrated_v1','cell_type_integrated_v1'),
         ('obs',True,'leiden_r0.5','leiden_r0.5')
        ],rm=None)

# %% [markdown]
# ## Plot final ct

# %%
rcParams['figure.figsize']=(10,10)
sc.pl.umap(adata,color='cell_type_integrated_v1',s=5)

# %%
adata.obs['cell_type_integrated_v1'].value_counts()

# %% [markdown]
# ## Cell proportions across studies and samples

# %%
anno_ratio=pd.DataFrame(adata.obs.groupby(['study_sample'])['cell_type_integrated_v1'].value_counts(normalize=True))
anno_ratio['proportion']=anno_ratio['cell_type_integrated_v1']
index=anno_ratio.copy().index.to_frame()
anno_ratio['study_sample']=index['study_sample']
anno_ratio['cell_type_integrated_v1']=index['cell_type_integrated_v1']
del index

# %%
n_types=adata.obs['cell_type_integrated_v1'].unique().shape[0]
rcParams['figure.figsize']= (25,4*n_types)
fix,axs=plt.subplots(n_types,1,sharex=True,sharey=True)
plt.xticks(fontsize= 20)
for i,ct in enumerate(sorted(adata.obs['cell_type_integrated_v1'].unique())):
    anno_ratio_sub=anno_ratio.query('cell_type_integrated_v1 ==@ct')
    sb.barplot(x="study_sample", y="proportion", 
               data=anno_ratio_sub,ax=axs[i])
    a=plt.xticks(rotation=90)
    #plt.yscale('log')
    axs[i].set_xlabel('')
    axs[i].set_ylabel(ct+'\nproportion in sample',fontsize=20)
    axs[i].set(ylim=(0, 1))


# %% [markdown]
# ## Save data

# %%
# Check if any obs columns are redundant
adata

# %%
keep_obs=['study_sample', 'study', 'file', 'reference', 'size_factors_sample', 
          'S_score', 'G2M_score', 'phase', 'phase_cyclone', 's_cyclone', 'g2m_cyclone', 'g1_cyclone', 
          'sex', 'pre_cell_type', 
          'ins_score', 'ins_high', 'gcg_score', 'gcg_high', 'sst_score', 'sst_high', 'ppy_score', 'ppy_high', 
          'cell_filtering', 'age', 'strain', 'tissue', 'technique', 'internal_id', 'batch', 
          'study_sample_design', 'cell_type', 'cell_type_multiplet', 'cell_subtype', 'cell_subtype_multiplet', 
          'leiden_r0.4', 'design','cell_type_integrated_v1']
drop=[col for col in adata.obs.columns if col not in keep_obs]
adata.obs.drop(drop,axis=1,inplace=True)

# %%
#h.save_h5ad(adata=adata,file=path_data+'data_integrated_analysed.h5ad', unique_id2=UID2)
adata.write(path_data+'data_integrated_analysed.h5ad')

# %% [markdown]
# ## Further analysis

# %% [markdown]
# For each multiplet group plot where it is located in the integrated UMAP.

# %%
# Per-dataset ref anno
endocrine=set(['alpha','beta','gamma','delta','epsilon'])
for ct in [ct for ct in adata.obs.cell_type.value_counts().index
           if len(set(ct.split('_'))&endocrine)>1]:
    col='is_'+ct
    adata.obs[col]=adata.obs['cell_type']==ct
    rcParams['figure.figsize']=(6,6)
    sc.pl.umap(adata,color=col,s=20)
    adata.obs.drop([col],axis=1,inplace=True)

# %%
# re-annotated integrated
for ct in ['alpha_beta','alpha_delta','beta_delta','beta_gamma','delta_gamma']:
    col='is_'+ct
    adata.obs[col]=adata.obs['cell_type_integrated_v1']==ct
    rcParams['figure.figsize']=(6,6)
    sc.pl.umap(adata,color=col,s=5)
    adata.obs.drop([col],axis=1,inplace=True)

# %%
rcParams['figure.figsize']=(8,8)
sc.pl.umap(adata,color=[score+'_score_scaled' for score in scores],s=20)

# %% [markdown]
# ### Parsed anno
# Make cell type names prettier for paper

# %%
adata=sc.read(path_data+'data_integrated_analysed.h5ad')

# %%
# Change cell type names
adata.obs['cell_type_integrated_v1_parsed']=adata.obs['cell_type_integrated_v1'].replace({
    'alpha_beta':'alpha+beta',
    'alpha_delta':'alpha+delta',
    'ambient':'lowQ',
    'beta_delta':'beta+delta',
    'beta_gamma':'beta+gamma',
    'delta_gamma':'delta+gamma',
    'embryo':'E non-endo.',
    'embryo endocrine':'E endo.',
    'endocrine proliferative':'endo. prolif.',
    'stellate_activated':'stellate a.',
    'stellate_quiescent':'stellate q.'
})

# %%
# make sorted order
adata.uns['cell_type_integrated_v1_parsed_order']=[
 'E endo.',
 'E non-endo.',
 'alpha',
 'beta',
 'delta',
 'gamma',
 'endo. prolif.',
 'acinar',
 'ductal',
 'endothelial',
 'immune',
 'schwann',
 'stellate a.',
 'stellate q.',
 'alpha+beta',
 'alpha+delta',
 'beta+delta',
 'beta+gamma',
 'delta+gamma',
 'lowQ']
adata.obs['cell_type_integrated_v1_parsed']=pd.Categorical(
    values=adata.obs['cell_type_integrated_v1_parsed'],
    categories=adata.uns['cell_type_integrated_v1_parsed_order'],ordered=True)

# %%
# List cts
print(adata.obs['cell_type_integrated_v1_parsed'].cat.categories.values)
if not (adata.obs['cell_type_integrated_v1_parsed'].nunique()==
      adata.obs['cell_type_integrated_v1'].nunique()):
        raise ValueError('Not all ts succesfully renamed')

# %%
if True:
    h.update_adata(adata_new=adata,
               path=path_data+'data_integrated_analysed.h5ad',
               add=[('obs',True,
                     'cell_type_integrated_v1_parsed',
                     'cell_type_integrated_v1_parsed'),
                    ('uns',True,
                     'cell_type_integrated_v1_parsed_order',
                     'cell_type_integrated_v1_parsed_order')],
                   rm=None, unique_id2=None,io_copy=False)

# %%
del adata

# %% [markdown]
# # Summary stats
# The below plots rely on study colors and ordering added in the notebook for the sample metadata summary plot. Thus they must be run after that notebook.

# %%
obs=sc.read(path_data+'data_integrated_analysed.h5ad',backed='r').obs.copy()
uns=sc.read(path_data+'data_integrated_analysed.h5ad',backed='r').uns.copy()

# %% [markdown]
# ### Cell N per sample ct

# %%
# Cell counts
group_sample_ratios=pd.DataFrame()
obs['studyparsed_design_sample']=[
    '_'.join(i) for i in 
     zip(obs.study_parsed,obs.design,obs.file)]
for (sample,hc),ratio in obs.\
        groupby('studyparsed_design_sample',observed=True)['cell_type_integrated_v1_parsed'].\
        value_counts(normalize=False,sort=False).iteritems():
    group_sample_ratios.at[sample,hc]=ratio 
group_sample_ratios.fillna(0,inplace=True)  
group_sample_ratios.replace({0:1},inplace=True)# Set 0 to 1 as after log10 will be set to 0
group_sample_ratios=np.log10(group_sample_ratios)

# Add design and study info for sorting and anno
group_sample_ratios['design']=[
    obs.query('studyparsed_design_sample==@sample').design.values[0]  
    for sample in group_sample_ratios.index]
group_sample_ratios['design']=pd.Categorical(group_sample_ratios['design'],
              categories=[ 
              'E12.5','E13.5','E14.5', 'E15.5', 
                  'mRFP','mTmG','mGFP',
              'head_Fltp-','tail_Fltp-', 'head_Fltp+', 'tail_Fltp+',
              'DMSO_r1','DMSO_r2', 'DMSO_r3','GABA_r1','GABA_r2', 'GABA_r3',
                   'A1_r1','A1_r2','A1_r3','A10_r1','A10_r2', 'A10_r3',  'FOXO_r1', 'FOXO_r2', 'FOXO_r3',
              'IRE1alphafl/fl','IRE1alphabeta-/-', 
              '8w','14w', '16w',
              'control','STZ', 'STZ_GLP-1','STZ_estrogen', 'STZ_GLP-1_estrogen',
                  'STZ_insulin','STZ_GLP-1_estrogen+insulin' ,
                  'chow_WT','sham_Lepr-/-','PF_Lepr-/-','VSG_Lepr-/-',   
            ],
            ordered=True)
group_sample_ratios['study']=[
    obs.query('studyparsed_design_sample==@sample').study_parsed.values[0]  
    for sample in group_sample_ratios.index]
group_sample_ratios['study']=pd.Categorical(
    group_sample_ratios['study'],ordered=True,
    categories=uns['study_parsed_order']
                                           )
# Sort rows 
group_sample_ratios=group_sample_ratios.sort_values(['study','design'])
# Save study info
studies=group_sample_ratios.study.values
# Drop unused cols
group_sample_ratios.drop(['study','design'],axis=1,inplace=True)
# Sort/select columns
group_sample_ratios=group_sample_ratios[obs['cell_type_integrated_v1_parsed'].cat.categories]

# Order columns
#group_sample_ratios=group_sample_ratios.iloc[:,
#    h.opt_order(group_sample_ratios.T,metric='correlation',method='ward')]

# Add rowcolors 

# Study
adata_temp=sc.AnnData(obs=obs)
#sc.pl._utils._set_default_colors_for_categorical_obs(adata_temp, 'study_parsed')
study_cmap=dict(zip(obs.study_parsed.cat.categories,
                    adata_temp.uns['study_parsed_colors']))
study_colors=[study_cmap[s] for s in studies]

# Age map
study_parsed_map=dict(zip(obs.study_parsed,obs.study))
study_age_categ={
    'Fltp_2y':'2y', 
    'Fltp_adult':'3-7m', 
    'Fltp_P16':'0-1m', 
    'NOD':'1-1.5m', 
    'spikein_drug':'2-3m', 
    'VSG':'3-7m', 
    'STZ':'3-7m',
    'embryo':'E'}
ages_order=['0-1m','1-1.5m','1.5-2m','2-3m','3-7m','2y']
normalize = mcolors.Normalize(vmin=0,  vmax=len(ages_order)-1)
age_cmap={'E':'k'}
age_cmap.update({age:cm.viridis(normalize(idx)) for idx,age in enumerate(ages_order)})
ages_order.insert(0,'E')
ages=[]
for sample,study in zip(group_sample_ratios.index,studies):
    study=study_parsed_map[study]
    if study=='NOD_elimination':
        if '8w' in sample:
            age='1.5-2m'
        else:
            age='3-7m'
    else:
        age=study_age_categ[study]
    ages.append(age)
age_colors=[age_cmap[a] for a in ages]

# Perturbation map
perturbation_map={}
for sample in group_sample_ratios.index:
    if 'mSTZ' in sample and 'control' not in sample:
        p='mSTZ'
    elif 'db/db' in sample and 'WT' not in sample:
        p='db/db'
    elif 'chem' in sample and 'DMSO' not in sample:
        p='other chemical'
    elif 'NOD' in sample:
        p='NOD'
    else:
        p='none'
    perturbation_map[sample]=p
perturbation_cmap={'NOD':'#58017B','db/db':'#9A0A0A','mSTZ':'#E06397',
                   'other chemical':'#A2B308','none':'#C3C3C3'}
perturbation_colors=[perturbation_cmap[ perturbation_map[s]] for s in group_sample_ratios.index]

# Combine anno
blank=['#FFFFFF']*group_sample_ratios.shape[0]
row_anno=pd.DataFrame({'perturbation':perturbation_colors,
                       'age group':age_colors,
                      'study':study_colors,'':blank},
                      index=group_sample_ratios.index)

# %%
# Cmap
g=sb.clustermap(group_sample_ratios,row_colors=row_anno,
                #annot=True,fmt='.2f',
             col_cluster=False,row_cluster=False,
             xticklabels=True,yticklabels=True,figsize=(9,16),
               colors_ratio=0.03,cbar_pos=(-0.07,0.48,0.05,0.1))
g.cax.set_title("log10(N cells)\n",fontsize=11)
# Remove row colors tick
g.ax_row_colors.xaxis.set_ticks_position('none') 

# Add legend for row colors
legend_elements = [ Patch(alpha=0,label='stress')]+[
     Patch(facecolor=perturbation_cmap[s],label=s) 
    for s in ['NOD','mSTZ','db/db','other chemical','none']
]+[ Patch(alpha=0,label='\nage group')]+[
     Patch(facecolor=c,label=s) for s,c in age_cmap.items()
]+[ Patch(alpha=0,label='\ndataset')]+[
     Patch(facecolor=study_cmap[s],label=s) for s in uns['study_parsed_order']
]
ax=g.fig.add_subplot(223)
ax.axis('off')
ax.legend(handles=legend_elements, bbox_to_anchor=(0.2,0.79),frameon=False)

# Save
plt.savefig(path_fig+'heatmap_atlas_celltype_samples.png',dpi=300,bbox_inches='tight')

# %% [markdown]
# ### Cell N per sample

# %%
# N cells per sample
n=pd.DataFrame(obs.groupby('study_sample').size().rename('N cells'))
n['study']=[obs.query('study_sample==@s').study_parsed[0] for s in n.index]

# Order studies by median N cells per study
n['study']=pd.Categorical(
    n['study'],
    n.groupby('study').median().sort_values('N cells',ascending=False).index,
    ordered=True)

# Study colors
adata_temp=sc.AnnData(obs=obs)
sc.pl._utils._set_default_colors_for_categorical_obs(adata_temp, 'study_parsed')
study_cmap=dict(zip(obs.study_parsed.cat.categories,
                    adata_temp.uns['study_parsed_colors']))

# %%
# Plot swarmplot of N cells per sample per study
rcParams['figure.figsize']=(3,3)
g=sb.swarmplot(x='N cells',y='study',data=n,hue='study',palette=study_cmap,s=4)
g.set(facecolor = (0,0,0,0))
g.spines['top'].set_visible(False)
g.spines['right'].set_visible(False)
g.get_legend().remove()
plt.savefig(path_fig+'swarmplot_atlas_count_studysample.png',dpi=300,bbox_inches='tight')

# %% [markdown]
# ### N cells per ct
# The below plots rely on the color asignment saved in annotation comparison notebook and must thus be run after that notebook.

# %%
# N cells per ct
n=pd.DataFrame(obs.groupby('cell_type_integrated_v1_parsed').size().rename('N cells'))
n.index.name='cell type'
n=n.reset_index()
n.index=n['cell type'].rename(None)
n['N cells (thousands)']=n['N cells']/1000

# ct colors
ct_map=dict(zip())
ct_cmap=dict(zip(obs.cell_type_integrated_v1.cat.categories,
                uns['cell_type_integrated_v1_colors'].copy()))
ct_cmap={obs.query('cell_type_integrated_v1==@ct')['cell_type_integrated_v1_parsed'][0]:c 
        for ct,c in ct_cmap.items()}

# Order by N cells
n['cell type']=pd.Categorical(
    n['cell type'],list(n['N cells'].sort_values(ascending=False).index),ordered=True
)

# %%
# Plot n cells per ct
rcParams['figure.figsize']=(3,4)
g=sb.barplot(x='N cells (thousands)',y='cell type',data=n,palette=ct_cmap,)
# text anno
for idx,p in enumerate(g.patches):
    n_cells=n.sort_values('cell type')['N cells'][idx]
    g.annotate(n_cells, 
                   (p.get_x()+p.get_width() , p.get_y()+p.get_height()/2 ), 
                   ha = 'center', va = 'center', 
                   xytext = (25,0), 
                   textcoords = 'offset points')
# Transparent
g.set_xlim(0,n['N cells (thousands)'].max()+20)
g.set(facecolor = (0,0,0,0))
g.spines['top'].set_visible(False)
g.spines['right'].set_visible(False)
plt.savefig(path_fig+'barplot_atlas_count_celltype.png',dpi=300,bbox_inches='tight')

# %%
# Plot n cells per ct - exclude lowQ/ambient
rcParams['figure.figsize']=(3,4)
n_sub=n.query('~(`cell type`.str.contains("\+")) and ~(`cell type`.str.contains("lowQ"))',
              engine='python').copy()
# This removes size ordering
n_sub['cell type']=n_sub['cell type'].astype('str')
g=sb.barplot(x='N cells (thousands)',y='cell type',data=n_sub,palette=ct_cmap,)
# text anno
for idx,p in enumerate(g.patches):
    n_cells=n_sub.sort_values('cell type')['N cells'][idx]
    g.annotate(n_cells, 
                   (p.get_x()+p.get_width() , p.get_y()+p.get_height()/2 ), 
                   ha = 'center', va = 'center', 
                   xytext = (25,0), 
                   textcoords = 'offset points')
# Transparent
g.set_xlim(0,n['N cells (thousands)'].max()+20)
g.set(facecolor = (0,0,0,0))
g.spines['top'].set_visible(False)
g.spines['right'].set_visible(False)
plt.savefig(path_fig+'barplot_atlas_count_celltype_nolowQ.png',dpi=300,bbox_inches='tight')
