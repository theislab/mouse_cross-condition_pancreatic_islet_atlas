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
import scanpy as sc
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb
import colorcet as cc


# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'
path_save='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/ct_composition/subtypes/'

# %%
# Get endothelial cells
obs_names=sc.read(
    path_data+'data_integrated_analysed.h5ad',backed='r'
    ).obs.query('cell_type_integrated_v1=="endothelial"').index
obs_names.shape

# %%
# Load rawnorm. Correctly norm expression is in layers X_sf_integrated
adata_rn=sc.read(path_data+'data_rawnorm_integrated_annotated.h5ad')[obs_names,:].copy()
adata_rn.X=adata_rn.layers['X_sf_integrated']

# %%
# Redo embedding
sc.pp.neighbors(adata_rn, use_rep='X_integrated'  )
sc.tl.umap(adata_rn)

# %% [markdown]
# ## Known markers

# %%
# orthologues
orthologues=pd.read_table(path_genes+'orthologues_ORGmus_musculus_ORG2homo_sapiens_V103.tsv'
                         ).rename(
    {'Gene name':'gs_mm','Human gene name':'gs_hs','Gene stable ID':'eid_mm'},axis=1)

# %%
# Human markers
markers_human={
'endothelial': ['PECAM1', 'CDH5', 'VWF','PLVAP'],
'microvascular':['PLVAP', 'RAMP2', 'RAMP3', 'VWF'],
'capillary': ['RGCC', 'CA4', 'AQP1', 'VWA1'],
'arterial capillary':['CA4', 'FCN3'],
'arterial': ['SEMA3G', 'HEY1', 'DLL4', 'GJA4',  'EFNB2'],
'venous': ['ACKR1', 'NR2F2', 'DLL1', 'PLVAP', 'PLAT', 'VWF'],
'lymphatic': ['PROX1', 'PDPN', 'MMRN1','LYVE1',  'CCL21', 'PTPRE'],
'pericytes': ['ABCC9', 'KCNJ8', 'CSPG4','RGS5', 'MCAM','COX4I2', 'HIGD1B', 'NOTCH3', 'FAM162B'],
'smooth_muscle': ['MYH11', 'ACTA2', 'TAGLN', 'CNN1'],
'mesothelial': ['TBX1', 'WT1', 'MSLN', 'CALB2', 'PRG4']
 }

# %%
# Mouse markers
markers=defaultdict(set)

# %%
# Map human to mouse. print missing orthologues
for group,gs in markers_human.items():
    for g in gs:
        mm_gss=orthologues.query('gs_hs ==@g')['gs_mm']
        if len(mm_gss)==0:
            print(g)
        else:
            markers[group].update(mm_gss)

# %% [markdown]
# C: Note that some pericyte markers are also showing up in stellate (activated)

# %%
genes_all=set(adata_rn.var.gene_symbol)

# %%
for group,genes in markers.items():
    rcParams['figure.figsize']=(5,5)
    print(group)
    sc.pl.umap(adata_rn,color=[g for g in genes if g in genes_all],
               gene_symbols='gene_symbol',s=20)

# %% [markdown]
# Markers from: https://pubmed.ncbi.nlm.nih.gov/32059779/ (S5)

# %%
markers_mm={
    'arthery':['8430408G22Rik',
         'Clu',
         'Crip1',
         'Fbln2',
         'Gja4',
         'Hey1',
         'Mecom',
         'Sat1',
         'Sema3g',
         'Sox17',
         'Tm4sf1',
         'Tsc22d1'],
    'capillary':['AW112010',
         'BC028528',
         'Car4',
         'Cd200',
         'Cd300lg',
         'Gpihbp1',
         'Kdr',
         'Rgcc',
         'Sgk1',
         'Sparc'],
    'vein':['Apoe', 'Bgn', 'Ctla2a', 'Icam1', 'Il6st', 'Ptgs1', 'Tmsb10', 'Vcam1', 'Vwf'],
    'lymphatic':['Ccl21a',
         'Cd63',
         'Cp',
         'Fgl2',
         'Flt4',
         'Fth1',
         'Fxyd6',
         'Maf',
         'Marcks',
         'Mmrn1',
         'Pard6g',
         'Pdpn',
         'Prelp',
         'Reln',
         'Rnase4',
         'Scn1b',
         'Stab1',
         'Thy1',
         'Timp2',
         'Timp3']
}

# %%
for group,genes in markers_mm.items():
    rcParams['figure.figsize']=(5,5)
    print(group)
    sc.pl.umap(adata_rn,color=[g for g in genes if g in genes_all],
               gene_symbols='gene_symbol',s=20)

# %% [markdown]
# #### Find markers of each cluster

# %%
sc.tl.leiden(adata_rn,resolution=1)

# %%
sc.pl.umap(adata_rn,color='leiden')

# %%
# Find markers OvR
sc.tl.rank_genes_groups(adata_rn, groupby='leiden')
sc.tl.filter_rank_genes_groups(adata_rn)

# %%
# Markers OVR
sc.pl.rank_genes_groups(adata_rn,gene_symbols ='gene_symbol')

# %%
# Save enrichment tables
writer = pd.ExcelWriter(path_save+'endothelial_clmarkers.xlsx',
                        engine='xlsxwriter') 
for cl in adata_rn.obs['leiden'].cat.categories:
    data=pd.DataFrame({
        'padj':adata_rn.uns['rank_genes_groups']['pvals_adj'][str(cl)],
        'lFC':adata_rn.uns['rank_genes_groups']['logfoldchanges'][str(cl)],
        'EID':adata_rn.uns['rank_genes_groups']['names'][str(cl)],
        'gene_symbol':adata_rn.var.loc[
            adata_rn.uns['rank_genes_groups']['names'][str(cl)],'gene_symbol'],
        'retained':[isinstance(n,str) 
                    for n in adata_rn.uns['rank_genes_groups_filtered']['names'][str(cl)]]
    })
    data.to_excel(writer, sheet_name=str(cl))   
writer.save()

# %%
path_save+'endothelial_clmarkers.xlsx'

# %% [markdown]
# ### Annotation

# %% [markdown]
# Plot study to help with removing batch-driven subtypes in annotation, since it is unclear if this are residual batch effects or true bio differences.

# %%
sc.pl.umap(adata_rn,color='study')

# %% [markdown]
# Add annotation names - saved as tab names in corresponding excel file

# %%
annotation = pd.ExcelFile(path_save+'endothelial_clmarkers_annotated.xlsx').sheet_names
print(annotation)

# %%
# Map clusters
annotation_map=dict(zip(adata_rn.obs['leiden'].cat.categories,annotation))
adata_rn.obs['cell_subtype_v0']=adata_rn.obs.leiden.map(annotation_map)

# %%
rcParams['figure.figsize']=(4,4)
sc.pl.umap(adata_rn,color=['cell_subtype_v0'
                           #,'cell_subtype_v0_parsed'
                          ],
           palette=sb.color_palette(cc.glasbey, 
                                    n_colors=adata_rn.obs['cell_subtype_v0'].nunique()),ncols=1)

# %%
rcParams['figure.figsize']=(10,10)
sc.pl.umap(adata_rn,color=['cell_subtype_v0'
                           #,'cell_subtype_v0_parsed'
                          ],
           palette=sb.color_palette(cc.glasbey, 
                                    n_colors=adata_rn.obs['cell_subtype_v0'].nunique()),
           ncols=1,legend_loc='on data')

# %% [markdown]
# C: These annotations are probably batch driven - need to be coarsened.

# %%
# Coarse ct
ct_coarse_map={
'Cebpb+EC':'capilary',
 'Cpe+EC':'capilary',
 'EC_Cd14+':'capilary',
 'EC_MHCI':'capilary',
 'EC_MHCII':'capilary',
 'EC_acinar':'capilary',
 'EC_art1':'lymphatic',
 'EC_cap':'capilary',
 'EC_immature':'capilary',
 'EC_preBreach':'pericyte-like',
 'EC_ven1':'capilary',
 'EC_ven2':'vein',
 'Inhbb+EC':'capilary',
 'Lrg1+EC':'vein',
 'Mgp+EC':'pericyte-like',
 'Rgs5+EC':'pericyte-like',
 'STm1+EC':'multiplet',
 'Txnip+EC':'capilary',
 'Ubd+EC':'vein',
 'id1+EC':'capilary'
}
adata_rn.obs['cell_subtype_v1_parsed_coarse']=adata_rn.obs['cell_subtype_v0'].map(
    ct_coarse_map)

# %%
rcParams['figure.figsize']=(7,7)
sc.pl.umap(adata_rn,color=['cell_subtype_v1_parsed_coarse','study'],
           wspace=0.5,s=15)

# %% [markdown]
# ## Save

# %%
adata_rn

# %%
adata_rn.write(path_data+'data_rawnorm_integrated_analysed_endothelial.h5ad')

# %% [markdown]
# ## Proportion changes in T2D

# %% [markdown]
# Plot of embedding of samples relevant for diabetes analysis (potential population shifts):
# - two studies: VSG, STZ (start of sample name)
# - healthy samples contain control/WT, others are T2D

# %%
rcParams['figure.figsize']=(4,4)
for sample in ['STZ_G2_STZ','STZ_G1_control','VSG_MUC13633_chow_WT',
 'VSG_MUC13634_chow_WT', 'VSG_MUC13641_sham_Lepr-/-', 'VSG_MUC13639_sham_Lepr-/-']:
    sc.pl.umap(adata_rn,color='study_sample_design',groups=[sample],
               palette='g' if 'control' in sample or 'WT' in sample else 'r')

# %%
obs=sc.read(path_data+'data_rawnorm_integrated_analysed_endothelial.h5ad',backed='r').obs.copy()

# %% [markdown]
# Subset to relevant T2D samples

# %%
samples=['STZ_G2_STZ','STZ_G1_control','VSG_MUC13633_chow_WT',
 'VSG_MUC13634_chow_WT', 'VSG_MUC13641_sham_Lepr-/-', 'VSG_MUC13639_sham_Lepr-/-',]
obs_sub=obs.query('study_sample_design in @samples ').copy()

obs_sub['study_sample_design'].unique()

# %%
pd.crosstab(obs_sub['study_sample_design'],obs_sub['cell_subtype_v1_parsed_coarse'])

# %%
# Proportions of cells per ct
props=pd.crosstab(obs_sub['study_sample_design'],obs_sub['cell_subtype_v1_parsed_coarse'],
                  normalize='index')

# %%
sb.clustermap(props)

# %% [markdown]
# C: Unclear if this may be technical or true bio effect. 

# %%
