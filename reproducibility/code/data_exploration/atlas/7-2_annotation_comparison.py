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
from pathlib import Path
import math
from collections import defaultdict

from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/')
import helper as h
import importlib
importlib.reload(h)
import helper as h

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/'

# %%
# Saving figures
path_fig='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/figures/paper/'
sc._settings.ScanpyConfig.figdir=Path(path_fig)

# %%
adata=sc.read(path_data+'data_integrated_analysed.h5ad')

# %% [markdown]
# ## Get and unify previous annotation
# Annotation from previous studies and manual mapping to unified set of names.

# %%
# Object for storing anno
anno_combined=[]
anno_original=[]

# %% [markdown]
# ### STZ

# %%
# STZ
adata_preannotated=sc.read("/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islet_glpest_lickert/rev7/subarna_study/GSE128565_adata_processed.h5ad.h5",
                          backed='r')

# %%
# Rename preannotated dataset obs and order it to match current dataset
preannotated_obs=adata_preannotated.obs.copy()
preannotated_obs.index=[idx.replace('_','-')+'-STZ' for idx in preannotated_obs.index]

# %%
# Columns in obs
preannotated_obs.columns.to_list()

# %%
# Cell type annotations
for annotation in ['groups_named_broad','groups_named_fine', 
                   'groups_named_beta_dpt']:
    print(annotation)
    display(preannotated_obs[annotation].value_counts(dropna=False))

# %%
# Compare broad and fine annotation
pd.crosstab(preannotated_obs['groups_named_broad'],preannotated_obs['groups_named_fine'],
           dropna=False)

# %%
# Remap annotation
anno_map={'Ins_1':'beta',
          'Ins_2':'beta',
          'Ins_dedifferentiated':'beta',
          'Ins-Ppy':'beta_gamma',
          'Ins-Sst':'beta_delta',
          'Ins-Sst-Ppy':'beta_delta_gamma',
          'Gcg':'alpha',
          'Gcg-Ppy_low':'alpha_gamma',
          'Gcg-Ppy_high':'alpha_gamma',
          'Ppy':'gamma',
          'Sst-Ppy_low':'delta'}

# %% [markdown]
# C: Annotation is hard to remap here to my annotation as polyhormonal cells are annotated with hormones rather than cell types (as in my data).

# %%
# Save anno
anno_combined.append(preannotated_obs['groups_named_fine'].map(anno_map))
anno_original.append(preannotated_obs['groups_named_fine'])

# %% [markdown]
# ### Fltp_P16

# %%
# Fltp_P16 data - 2 adatas were in GEO
adata_preannotated=sc.read("/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/salinno_project/rev4/preannotated/GSE161966_data_annotated.h5ad",
                           backed='r')
adata_preannotated2=sc.read("/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/salinno_project/rev4/preannotated/GSE161966_data_endo_final.h5ad",
                           backed='r')

# %%
# Rename preannotated dataset obs and order it to match current dataset
# Merge both obs and rename them
obs1=adata_preannotated.obs.copy()
obs1.columns=[c+'_1' for c in obs1.columns]
obs2=adata_preannotated2.obs.copy()
obs2.columns=[c+'_2' for c in obs2.columns]
preannotated_obs=pd.concat([obs1,obs2],axis=1)
cell_name_map={'-mGFP':'-145_mGFP','-mRFP':'-146_mRFP','-mTmG':'-147_mTmG'}
preannotated_obs.index=[h.replace_all(cell,cell_name_map)+'-Fltp_P16' 
                        for cell in preannotated_obs.index]

# %%
# Columns in obs
preannotated_obs.columns.to_list()

# %%
# Cell type annotations
for annotation in ['cell_type_1','cell_type_2','louvain_R_2','cell_type_refined_2']:
    print(annotation)
    display(preannotated_obs[annotation].value_counts(dropna=False))

# %%
# Compare annotations from the two adata objects
pd.crosstab(preannotated_obs['cell_type_1'],preannotated_obs['cell_type_refined_2'],
           dropna=False)

# %%
# Remap annotation
anno_map={'alpha':'alpha',
 'beta':'beta',
 'beta-delta':'beta_delta',
 'alpha-delta':'alpha_delta',
 'gamma, proliferating':'gamma proliferative',
 'delta':'delta',
 'stellate':'stellate',
 'delta, proliferating':'delta proliferaive',
 'alpha, proliferating':'alpha proliferative',
 'beta, proliferating':'beta proliferative',
 'acinar/ductal':'acinar/ductal',
 'gamma':'gamma',
 'alpha-beta':'alpha_beta',
 'epsilon':'epsilon',
 'immune':'immune',
 'endothelial':'endothelial'}

# %%
# Save anno
anno_combined.append(preannotated_obs['cell_type_1'].map(anno_map))
anno_original.append(preannotated_obs['cell_type_1'])

# %% [markdown]
# ### Spikein_drug

# %%
# Spikein drug
preannotated_obs=pd.read_table('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE142465/preannotated/GSE142465_MouseLTI_CellAnnotation_final.tsv',
                                 index_col=0)


# %%
# Rename preannotated dataset obs and order it to match current dataset
sample_dict=pd.read_excel('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/scRNA-seq_pancreas_metadata.xlsx',
             sheet_name='spikein_drug',index_col='metadata')['sample_name'].to_dict()
preannotated_obs.index=[idx.split('-')[0]+'-1-'+\
                        sample_dict[preannotated_obs.at[idx,'sample']] +\
                        '-spikein_drug'
                        for idx in preannotated_obs.index]

# %%
# Columns in obs
preannotated_obs.columns.to_list()

# %%
# Cell type annotations
for annotation in ['celltype','celltype2','PredictedCell']:
    print(annotation)
    display(preannotated_obs[annotation].value_counts(dropna=False))

# %%
# Compare broad and fine annotation
pd.crosstab(preannotated_obs['celltype'],preannotated_obs['celltype2'],
           dropna=False)

# %% [markdown]
# C: They first annotated cells, then did prediction to re-annotate hard to annotate cells, and then re-annotated "endocrine" manually annotated cells to specific endocrine cell type based on the prediction. So the right annotation here is celltype2.

# %%
# Remap annotation
anno_map={'Beta':'beta',
 'Alpha':'alpha',
 'Acinar':'acinar',
 'Delta':'delta',
 'Gamma':'gamma',
 'Endothelial1':'endothelial',
 '3':'unknown',
 'Endothelial2':'endothelial',
 '4':'unknown',
 '10':'unknown',
 'SI_Human':'spikein_human',
 'SI_Mouse':'spikein_mouse',
 '13':'unknown',
 '11':'unknown'}

# %%
# Save anno
anno_combined.append(preannotated_obs['celltype2'].map(anno_map))
anno_original.append(preannotated_obs['celltype2'])

# %% [markdown]
# ### Embryo

# %%
# Embryo
adata_preannotated=sc.read("/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE132188/rev7/preannotated/GSE132188_adata.h5ad.h5",
                          backed='r')

# %%
# Rename preannotated dataset obs and order it to match current dataset
preannotated_obs=adata_preannotated.obs.copy()
preannotated_obs.index=['-'.join(idx.split('-')[:-1])+\
                        '-E'+preannotated_obs.at[idx,'day'].replace('.','_')+\
                        '-embryo' for idx in preannotated_obs.index]

# %%
# Columns in obs
preannotated_obs.columns.to_list()

# %%
# Cell type annotations
for annotation in [ 
     'clusters_fig3_final',
     'clusters_fig3_final_noep',
     'clusters_fig4_final',
     'clusters_fig2_final',
     'clusters_fig6_broad_final',
     'clusters_fig6_fine_final',
     'clusters_fig6_alpha_final']:
    print(annotation)
    display(preannotated_obs[annotation].value_counts(dropna=False))

# %%
# Compare broad and fine annotation
pd.crosstab(preannotated_obs['clusters_fig3_final'],preannotated_obs['clusters_fig6_broad_final'],
           dropna=False)

# %%
# Compare broad and fine annotation
pd.crosstab(preannotated_obs['clusters_fig6_fine_final'],preannotated_obs['clusters_fig6_broad_final'],
           dropna=False)

# %%
# Compare broad and fine annotation
pd.crosstab(preannotated_obs['clusters_fig6_alpha_final'],preannotated_obs['clusters_fig6_broad_final'],
           dropna=False)

# %% [markdown]
# C: Different annotations have different cell types resolved. 
# - clusters_fig6_broad_final: general resolution
# - clusters_fig6_alpha_final: resolves primary and secondary alpha cells, resolves endocrine progenitor cells
# - clusters_fig6_fine_final: resolves endocrine progenitor cells
# - clusters_fig3_final: From fig3 plot. But fig6 is higher resolution.

# %%
# Make fine cell type column from fig6 annotations
preannotated_obs['cell_type_combined']=[ct if ct not in ['Ngn3 high EP','Alpha','Fev+'] 
                                        else ct_a
                                        for ct,ct_a in zip(
                                            preannotated_obs['clusters_fig6_broad_final'],
                                            preannotated_obs['clusters_fig6_alpha_final'])]

# %%
# Compare broad and fine annotation
pd.crosstab(preannotated_obs['cell_type_combined'],preannotated_obs['clusters_fig6_broad_final'],
           dropna=False)

# %%
preannotated_obs['cell_type_combined'].unique().tolist()

# %%
# Remap annotation
anno_map={
     'Trunk':'trunk',
     'Tip':'tip',
     'Ngn3 low EP':'EP_Ngn3low',
     'Prlf. Tip':'tip proliferative',
     'Prlf. Trunk':'trunk proliferative',
     'Multipotent':'multipotent',
     'Prlf. Ductal':'ductal proliferative',
     'Ductal':'ductal',
     'Fev+ Alpha':'alpha_Fev+',
     'primary Alpha':'alpha_primary',
     'Ngn3 High early':'EP_Ngn3high_early',
     'Ngn3 High late':'EP_Ngn3high_late',
     'Fev+ Pyy':'Fev+_Pyy',
     'Beta':'beta',
     'secondary Alpha':'alpha_secondary',
     'Prlf. Acinar':'acinar proliferative',
     'Fev+ Beta':'Fev+_beta',
     'Fev+ Epsilon':'Fev+_epsilon',
     'Delta':'delta',
     'Epsilon':'epsilon',
     'Fev+ Delta':'Fev+_delta',
     'Mat. Acinar':'acinar_mature'}
anno_map={ct:'embryo '+ct_new for ct,ct_new in anno_map.items()}

# %%
# Save anno
anno_combined.append(preannotated_obs['cell_type_combined'].map(anno_map))
anno_original.append(preannotated_obs['cell_type_combined'])

# %% [markdown]
# ### VSG

# %%
# VSG
adata_preannotated=sc.read("/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/VSG_PF_WT_cohort/rev7/preannotated/adata_annotated_Lickert_10x_dbdb_VSG_2021.h5ad",
                          backed='r')

# %%
# Rename preannotated dataset obs and order it to match current dataset
preannotated_obs=adata_preannotated.obs.copy()
preannotated_obs.index=[idx.split('-')[0]+'-1-'+sample+'-VSG' 
                        for idx,sample in zip(preannotated_obs.index,preannotated_obs['sample'])]

# %%
# Columns in obs
preannotated_obs.columns.to_list()

# %%
# Cell type annotations
for annotation in ['annotations','annotations_broad']:
    print(annotation)
    display(preannotated_obs[annotation].value_counts(dropna=False))

# %%
# Compare broad and fine annotation
pd.crosstab(preannotated_obs['annotations'],preannotated_obs['annotations_broad'],
           dropna=False)

# %%
# Remap annotation
anno_map={
     'Beta':'beta',
     'Alpha':'alpha',
     'Stellate':'stellate',
     'Endothelial':'endothelial',
     'Immune':'immune',
     'Polyhormonal':'polyhormonal',
     'Delta':'delta',
     'Ductal':'ductal',
     'PP':'gamma',
     'Acinar':'acinar'}

# %% [markdown]
# C: Annotation is hard to remap here to my annotation as polyhormonal cells are annotated with hormones rather than cell types (as in my data).

# %%
# Save anno
anno_combined.append(preannotated_obs['annotations_broad'].map(anno_map))
anno_original.append(preannotated_obs['annotations_broad'])

# %% [markdown]
# ### Add anno to adata

# %%
# Combine anno and add to adata
anno_combined=pd.concat(anno_combined)
adata.obs['pre_cell_type_unified']=anno_combined.reindex(
    adata.obs_names).fillna('NA').astype('category')
anno_original=pd.concat(anno_original)
adata.obs['pre_cell_type_original']=anno_original.reindex(
    adata.obs_names).fillna('NA').astype('category')

# %%
# Also set unknown to NA
adata.obs['pre_cell_type_unified']=adata.obs['pre_cell_type_unified'].replace({'unknown':'NA'})

# %% [markdown]
# #### Parse some anno

# %%
adata.obs['cell_type_parsed']=adata.obs['cell_type'].\
       str.replace('proliferative','prolif.').\
       str.replace('_beta','+beta').\
       str.replace('_delta','+delta').\
       str.replace('_gamma','+gamma').\
       str.replace('_',' ')
sorted(adata.obs['cell_type_parsed'].unique())

# %%
adata.obs['pre_cell_type_unified']=adata.obs['pre_cell_type_unified'].\
    str.replace('embryo','E' ).\
    str.replace('proliferative','prolif.').str.replace('proliferaive','prolif.').\
    replace({
    'alpha_beta':'alpha+beta',
    'alpha_delta':'alpha+delta',
    'alpha_gamma':'alpha+gamma',
    'beta_delta':'beta+delta',
    'beta_delta_gamma':'beta+delta+gamma',
    'beta_gamma':'beta+gamma',
    }).str.replace('_',' ')
sorted(adata.obs['pre_cell_type_unified'].unique())

# %% [markdown]
# #### Save anno

# %%
# Save

h.update_adata(adata_new=adata,path=path_data+'data_integrated_analysed.h5ad',
               add=[
                   ('obs',True,'cell_type_parsed','cell_type_parsed'),
                   ('obs',True,'pre_cell_type_unified','pre_cell_type_unified'),
                   ('obs',True,'pre_cell_type_original','pre_cell_type_original')
               ],
               rm=None,unique_id2=None,io_copy=False)

# %% [markdown]
# ### Compare re-annotation and previous annotations

# %%
# Make another col without embryo for easier visualisation of other cell types
adata.obs['pre_cell_type_unified_postnatal']=adata.obs['pre_cell_type_unified'].apply(
    lambda x: 'embryo' if x.startswith('E ') else x).astype('category')

# %% [raw]
# # Used to see which hex corresponds to which color
# for i,c in enumerate([
#    '#FFAA92', '#0AA6D8', '#FAD09F', '#72418F', '#9B9700', '#006FA6',
#    '#FFB500', '#D790FF', '#FF913F', '#A4E804', '#324E72', '#D16100',
#    '#44a4b8', '#B05B6F', '#A77500', '#788D66', '#BEC459', '#FF8A9A',
#    '#7A87A1', '#D157A0']):
#     plt.scatter(i,i,c=c,label=c)
# plt.legend()

# %%
# Pre-set colors for main cts
adata.uns['cell_type_integrated_v2_colors']=[
   '#FFAA92', '#0AA6D8', '#FAD09F', '#72418F', '#9B9700', '#006FA6',
   '#FFB500', '#D790FF', '#FF913F', '#A4E804', '#324E72', '#D16100',
   '#44a4b8', '#B05B6F', '#A77500', '#788D66', '#BEC459', '#FF8A9A',
   '#7A87A1', '#D157A0']
ct_integr_cmap=dict(zip(adata.obs['cell_type_integrated_v2'].cat.categories,
                        adata.uns['cell_type_integrated_v2_colors']))
ct_integr_cmap={
    adata.obs.query('cell_type_integrated_v2==@ct')['cell_type_integrated_v2_parsed'][0]:c 
        for ct,c in ct_integr_cmap.items()}

# %%
# Custom color palete matched across columns with NA in light gary
ct_cols=['cell_type_integrated_v2_parsed',
         'pre_cell_type_unified','pre_cell_type_unified_postnatal',
         'cell_type_parsed']
for col in ct_cols:
    adata.obs[col]=adata.obs[col].astype('category')
cts_all=[ct for col in ct_cols for ct in adata.obs[col].unique().tolist()]
# Make cts unique, but so that ct integrated is first - this ensures that it gets nice colors
# For this ct integrated must also be first in cols list
# Preselect nice colors for the main figure
cts_all=[ct for idx,ct in enumerate(cts_all) if ct not in cts_all[:idx]]
colors=[ct_integr_cmap[ct] for ct in cts_all if ct in ct_integr_cmap]
colors=colors+[c for c in sc.pl.palettes.default_102 if c !='#c7c7c7' and c not in colors]
colors = dict(zip(cts_all,colors[:len(cts_all)]))
colors['NA']='#c7c7c7'
for col in ct_cols:
    adata.uns[col+'_colors']=[colors[ct] for ct in adata.obs[col].cat.categories]

# %%
# Custom palete for original ct - set NA to gray
cts_all=adata.obs['pre_cell_type_original'].unique().tolist()
colors=[c for c in sc.pl.palettes.default_102 if c !='#c7c7c7']
colors = dict(zip(cts_all,colors[:len(cts_all)]))
colors['NA']='#c7c7c7'
adata.uns['pre_cell_type_original_colors']=[
    colors[ct] for ct in adata.obs['pre_cell_type_original'].cat.categories]

# %%
# Save color palete for cts
if False:
    h.update_adata(adata_new=adata,path=path_data+'data_integrated_analysed.h5ad',
               add=[
                  ('uns',True,'cell_type_integrated_v2_colors','cell_type_integrated_v2_colors'),
               ],
               rm=None,unique_id2=None,io_copy=False)

# %%
# Plot different cell types
cols=ct_cols+['pre_cell_type_original']
fig,ax=plt.subplots(len(cols),1,figsize=(6,6*len(cols)))
random_indices=np.random.permutation(list(range(adata.shape[0])))
for idx,(col,title) in enumerate(zip(cols,
                     ['cell type','cell type original unified',
                 'cell type original unified postnatal','cell type per-study',
                      'cell type original'])):
    sc.pl.umap(adata[random_indices,:],color=col, ncols=1,
               s=5,hspace=0.7,
              title=title,
              frameon=False, ax=ax[idx],show=False)
#fig.tight_layout()
plt.savefig(path_fig+'umap_atlas_celltypes.png',dpi=300,bbox_inches='tight')

# %% [markdown]
# Plot study info

# %% [raw]
# # Parse study names - UNUSED - done elsewhere
# cluster_map={
#  'Fltp_2y':'old',
#  'Fltp_adult':'young_adult',
#  'Fltp_P16':'postnatal',
#  'NOD':'NOD_young',
#  'NOD_elimination':'NOD_progression',
#  'spikein_drug':'spikein_chem',
#  'embryo':'embryo',
#  'VSG':'dbdb',
#  'STZ':'STZ'
# }
# clustering = 'study' 
# adata.obs[clustering+'_parsed']=adata.obs[clustering].map(cluster_map)

# %%
fig,ax=plt.subplots(figsize=(5,5))
np.random.seed(0)
random_indices=np.random.permutation(list(range(adata.shape[0])))
sc.pl.umap(adata[random_indices,:],color='study_parsed',s=3,
          frameon=False,title='study',ax=ax,show=False)
#fig.tight_layout() # Set to False so that stuff is properly drawn
ax.legend_.set_title('study')
plt.savefig(path_fig+'umap_atlas_covariate_study.png',dpi=300,bbox_inches='tight')

# %% [markdown]
# #### Plots without lowQ - low quality populations on atlas and low quality cells from individual subtypes. 
# NOTE: This part needs to be done after generating beta cell, immune, and alpha anootation.

# %%
# Remove low q/doublet
subset=~(adata.obs.cell_type_integrated_v1_parsed.str.contains("\+").values | 
         adata.obs.cell_type_integrated_v1_parsed.str.contains("lowQ").values)
print(subset.sum(),subset.shape)

# %%
# Get also lowQ from alpha and beta
alpha_lowq=sc.read(path_data+'data_rawnorm_integrated_analysed_alpha.h5ad',
                   backed='r').obs.query('low_quality').index.copy()
beta_lowq=sc.read(path_data+'data_rawnorm_integrated_analysed_beta_v1s1_sfintegrated.h5ad',
                  backed='r').obs.query('`leiden_r1.5_parsed_const`.str.contains("low_quality")',
                             engine='python').index.copy()
# Update subset of cells to be used
subset=pd.Series(subset,index=adata.obs_names)
subset.loc[alpha_lowq]=False
subset.loc[beta_lowq]=False
print(subset.sum())

# %%
# Save lowq assignment
adata.obs['low_q']=subset
if True:
    h.update_adata(adata_new=adata,path=path_data+'data_integrated_analysed.h5ad',
               add=[
                  ('obs',True,'low_q','low_q'),
               ],
               rm=None,unique_id2=None,io_copy=False)

# %%
fig,ax=plt.subplots(figsize=(5,5))
np.random.seed(0)
random_indices=np.random.permutation(list(range(adata[adata.obs['low_q'],:].shape[0])))
sc.pl.umap(adata[adata.obs['low_q'],:][random_indices,:],color='study_parsed',s=3,
          frameon=False,title='study',ax=ax,show=False)
#fig.tight_layout() # Set to False so that stuff is properly drawn
ax.legend_.set_title('study')
plt.savefig(path_fig+'umap_atlas_covariate_study_nolowQ.png',dpi=300,bbox_inches='tight')

# %%
# Ct
fig,ax=plt.subplots(figsize=(5,5))
np.random.seed(0)
cl_cmap=dict(zip(adata.obs['cell_type_integrated_v2'].cat.categories,
                 adata.uns['cell_type_integrated_v2_colors']))
cl_cmap={ct:cl_cmap[adata.obs.query('cell_type_integrated_v2_parsed==@ct'
                        )['cell_type_integrated_v2'].values[0]] 
         for ct in adata.obs['cell_type_integrated_v2_parsed'].cat.categories}
random_indices=np.random.permutation(list(range(adata[adata.obs['low_q'],:].shape[0])))
sc.pl.umap(adata[adata.obs['low_q'],:][random_indices,:],
           color='cell_type_integrated_v2_parsed',s=3,
          frameon=False,title='study',ax=ax,show=False,palette=cl_cmap)
#fig.tight_layout() # Set to False so that stuff is properly drawn
ax.legend_.set_title('study')
plt.savefig(path_fig+'umap_atlas_celltype_integrated_nolowQ.png',
            dpi=300,bbox_inches='tight')

# %% [markdown]
# #### Compare cell type overlaps per study

# %%
# Per study comparison of previous and my annotation, using unified annotation
for study in adata.obs.study_parsed.unique():
    obs_sub=adata[adata.obs.study_parsed==study,:].obs
    if not (obs_sub['pre_cell_type_unified']=='NA').all():
        # Confusion matrix normalised by previous anno group
        # Do not display NA from previous anno (cells not present)
        confusion=obs_sub.groupby(['pre_cell_type_unified'],dropna=False)[
            'cell_type_integrated_v2'].value_counts(
            normalize=True,dropna=False).unstack().fillna(0).drop('NA',axis=0)
        confusion.columns.name='cell type'
        confusion.index.name='cell type original unified'
        
        # Colorscale
        # Min and max values
        confusion_unnorm=obs_sub.groupby(['pre_cell_type_unified'],dropna=False)[
            'cell_type_integrated_v2'].value_counts(
            dropna=False).unstack().fillna(0).drop('NA',axis=0)
        # Round scale to certain precision
        sums_rows=confusion_unnorm.sum(axis=1)
        sums_cols=confusion_unnorm.sum(axis=0)
        round_to=100
        min_n=math.floor(min(sums_cols.min(),sums_rows.min())/round_to)*round_to
        max_n=math.ceil(max(sums_cols.max(),sums_rows.max()) /round_to)*round_to        
        # Make colorscale colors
        normalize = mcolors.Normalize(vmin=min_n,  vmax=max_n)
        cmap=cm.summer
        col_colors=sums_cols.apply(lambda n:cmap(normalize(n))).rename(None)
        row_colors=sums_rows.apply(lambda n:cmap(normalize(n))).rename(None)

        # Heatmap
        figsize=[confusion.shape[1]/3+2,
                 confusion.shape[0]/3+2]
        g=sb.clustermap(confusion,
                        xticklabels=True,yticklabels=True,
                       figsize=figsize,
                        col_colors=col_colors,row_colors=row_colors)
        g.ax_cbar.set_title('overlap')   
        g.fig.suptitle(study)
        
        # Legend N cells
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
        ax=g.fig.add_subplot(223)
        cbar=plt.colorbar(scalarmappaple,location='left',
                     ticks=[
            min_n, math.ceil((max_n+min_n)/(round_to*2))*round_to,max_n],fraction=0.1,
                     pad=1.1,aspect=1.5)
        cbar.ax.set_title('N cells')
        ax.axis('off')
        
        #remove dendrogram
        g.ax_row_dendrogram.set_visible(False)
        g.ax_col_dendrogram.set_visible(False)
        
        display(g.fig)
        
        # Print N cells
        n_cells_previous=obs_sub['pre_cell_type_unified'].value_counts()
        display(n_cells_previous)
        
        plt.close()

# %%
# Per study comparison of previous and my annotation (not unified)
for study in adata.obs.study_parsed.unique():
    obs_sub=adata[adata.obs.study_parsed==study,:].obs
    if not (obs_sub['pre_cell_type_original']=='NA').all():
        # Confusion matrix normalised by previous anno group
        # Do not display NA from previous anno (cells not present)
        confusion=obs_sub.groupby(['pre_cell_type_original'],dropna=False)[
            'cell_type_integrated_v2_parsed'].value_counts(
            normalize=True,dropna=False).unstack().fillna(0).drop('NA',axis=0)
        confusion.columns.name='cell type'
        confusion.index.name='cell type original'
        
        # Colorscale
        # Min and max values
        confusion_unnorm=obs_sub.groupby(['pre_cell_type_original'],dropna=False)[
            'cell_type_integrated_v2_parsed'].value_counts(
            dropna=False).unstack().fillna(0).drop('NA',axis=0)
        # Round scale to certain precision
        sums_rows=confusion_unnorm.sum(axis=1)
        sums_cols=confusion_unnorm.sum(axis=0)
        round_to=100
        min_n=math.floor(min(sums_cols.min(),sums_rows.min())/round_to)*round_to
        max_n=math.ceil(max(sums_cols.max(),sums_rows.max()) /round_to)*round_to        
        # Make colorscale colors
        normalize = mcolors.Normalize(vmin=min_n,  vmax=max_n)
        cmap=cm.cividis
        col_colors=sums_cols.apply(lambda n:cmap(normalize(n))).rename(None)
        row_colors=sums_rows.apply(lambda n:cmap(normalize(n))).rename(None)

        # Heatmap
        w_dend=1.5
        w_colors=0.3
        nrow=confusion.shape[0]*0.3+1
        ncol=confusion.shape[1]*0.3
        w=ncol+w_colors+w_dend
        h=nrow+w_colors+w_dend
        g=sb.clustermap(confusion,
                        xticklabels=True,yticklabels=True,
                       figsize=(w,h),
                        colors_ratio=(w_colors/w,1.02*w_colors/h),
                        dendrogram_ratio=(w_dend/w,w_dend/h),
                        col_colors=col_colors,row_colors=row_colors,
                       vmin=0,vmax=1)
        g.ax_cbar.set_title('overlap')   
        g.fig.suptitle(study)
        
        # Legend N cells
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
        ax=g.fig.add_subplot(223)
        cbar=plt.colorbar(scalarmappaple,location='left',
                     ticks=[
            min_n, math.ceil((max_n+min_n)/(round_to*2))*round_to,max_n],fraction=0.1,
                     pad=1.1,aspect=4)
        cbar.ax.set_title('N cells')
        cbar.ax.yaxis.set_ticks_position('right')
        ax.axis('off')
        
        #remove dendrogram
        g.ax_row_dendrogram.set_visible(False)
        g.ax_col_dendrogram.set_visible(False)
        
        # remove color ticks
        g.ax_row_colors.tick_params(axis='both', which='both', length=0)
        g.ax_col_colors.tick_params(axis='both', which='both', length=0)
        
        display(g.fig)
        plt.savefig(path_fig+'heatmap_atlas_celltype_pre_comparison_'+study.replace('/','')+'.png',
                    dpi=300,bbox_inches='tight')
        
        # Print N cells
        n_cells_previous=obs_sub['pre_cell_type_unified'].value_counts()
        display(n_cells_previous)
        
        plt.close()

# %% [markdown]
# Comparison of my ref per study annotation and integrated annotation

# %%
# Per study comparison of previous and my annotation (not unified)
obs_sub=adata.obs
# Confusion matrix normalised by previous anno group
# Do not display NA from previous anno (cells not present)
# Drop NA (missing) cells in comparison ct set
confusion=obs_sub.groupby(['cell_type'],dropna=False)[
    'cell_type_integrated_v2'].value_counts(
    normalize=True,dropna=False).unstack().fillna(0).drop('NA',axis=0)
confusion.columns.name='cell type'
confusion.index.name='cell type per-study'

# Colorscale
# Min and max values
confusion_unnorm=obs_sub.groupby(['cell_type'],dropna=False)[
    'cell_type_integrated_v2'].value_counts(
    dropna=False).unstack().fillna(0).drop('NA',axis=0)
# Round scale to certain precision
sums_rows=confusion_unnorm.sum(axis=1)
sums_cols=confusion_unnorm.sum(axis=0)
round_to=100
min_n=math.floor(min(sums_cols.min(),sums_rows.min())/round_to)*round_to
max_n=math.ceil(max(sums_cols.max(),sums_rows.max()) /round_to)*round_to        
# Make colorscale colors
normalize = mcolors.Normalize(vmin=min_n,  vmax=max_n)
cmap=cm.summer
col_colors=sums_cols.apply(lambda n:cmap(normalize(n))).rename(None)
row_colors=sums_rows.apply(lambda n:cmap(normalize(n))).rename(None)

# Heatmap
figsize=[confusion.shape[1]/3+2,
         confusion.shape[0]/3.2+2]
g=sb.clustermap(confusion,
                xticklabels=True,yticklabels=True,
               figsize=figsize,
                col_colors=col_colors,row_colors=row_colors)
g.ax_cbar.set_title('overlap')   

# Legend N cells
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
ax=g.fig.add_subplot(223)
cbar=plt.colorbar(scalarmappaple,location='left',
             ticks=[
    min_n, math.ceil((max_n+min_n)/(round_to*2))*round_to,max_n],fraction=0.1,
             pad=1.1,aspect=1.5)
cbar.ax.set_title('N cells')
ax.axis('off')

#remove dendrogram
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)

display(g.fig)
plt.savefig(path_fig+'heatmap_atlas_celltype_refperstudy_comparison.png',
            dpi=300,bbox_inches='tight')

# Print N cells
n_cells_previous=obs_sub['pre_cell_type_unified'].value_counts()
display(n_cells_previous)

plt.close()

# %% [markdown]
# C: 
# - We did not resolve endocrine proliferative clusters, although they are clearly visible in the UMAP.
# - Could not resolve epsilon cells as too little of them. They did not form a separate cluster (close to gamma cells).
# - Some doublet annotations differ, but these are hard anyways. Also for gamma cells it is sometimes hard to figure out if gamma doublets or just Ppy expression as other cts also express it.
# - Annotated some non-endocrine cell types not annotated in other studies (e.g. schwann, stellate subtypes,...)
# - Spikein drug has many wrong anno from original study - some of it was corrected in paper but not GEO. 
# - Interestingly, embryo delta cells map to postnatal delta cells.

# %%
