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
from collections import defaultdict
import pickle as pkl

from sklearn.preprocessing import minmax_scale
from scipy.stats import ttest_ind,combine_pvalues
from statsmodels.stats.multitest import multipletests

from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/')
from importlib import reload  
import helper as h
reload(h)
import helper as h

# %%
path_rna='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/'
path_data=path_rna+'combined/'
path_clde=path_data+'beta_subtype_general/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'

# %%
# Saving figures
path_fig='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/figures/paper/'

# %%
# Orthologues
orthologues=pd.read_table(path_genes+'orthologues_ORGmus_musculus_ORG2homo_sapiens_V103.tsv'
                         ).rename(
    {'Gene name':'gs_mm','Human gene name':'gs_hs','Gene stable ID':'eid_mm'},axis=1)

# %%
# Lad DE
marker_info=pkl.load(open(path_clde+'markersCLparsed_leiden_r1.5_parsed_const_study_selected.pkl','rb'))
# Extract markers
markers_new=defaultdict(list)
for group,mm_eids in marker_info.items():
    # Match group to parsed cluster names
    #group=cl_map[group]
    markers_new[group]=set(orthologues.query('eid_mm in @mm_eids').gs_hs.to_list())
markers_new


# %%
# Known markers
markers_mm={
    'beta':['Ins1','Ins2','Nkx6-1','Pdx1'],
    'imm.':['Rbp4','Cd81','Mafb'],
    'mature':['Mafa','Ucn3','Slc2a2'],
    'aged':['Cdkn2a','Trp53bp1'],
    'T2D':['Gast','Gc','Aldh1a3','Cck','Aldob','Chgb'],
    'T1D':['B2m'],
    }
# Map to human
markers=defaultdict(set)
for group,gss_mm in markers_mm.items():
    markers[group].update(orthologues.query('gs_mm in @gss_mm')['gs_hs'])
print(markers)

# %% [markdown]
# ## Marker DE OvR
# Compute OvR DE of markers to summarise their performance across datasets. In each dataset group donors in groups that should correspond either to population expected to be high in a marker or "rest" of the donors. Only datasets that have the group expected to be high in a marker are used for that marker (i.e. computation of DE is on per-dataset level).

# %% [markdown]
# Queries for groups of cells expected to be high in above specified marker groups

# %%
group_queries={
    'T1D':'disease=="T1D"',
    'T2D':'disease=="T2D"',
    'adult':'disease=="healthy" & 19<=age_years<=64',
    'mature': 'disease=="healthy" & 19<=age_years', # Contains both adult and aged
    'agedF':'disease=="healthy" & 65<=age_years & sex=="female"',
    'agedM':'disease=="healthy" & 65<=age_years & sex=="male"',
    'aged':'disease=="healthy" & 65<=age_years',
    'imm.':'disease=="healthy" & age_years<=18',
}
# Add group queries for other groups
group_queries['NOD-D']=group_queries['T1D']
group_queries['db/db+mSTZ']=group_queries['T2D']

# %% [markdown]
# Compute DE

# %%
# Do not use groups with less than this N cells
min_cells=5

# %%
# Compute OvR DE for markers
de_results=[]
for dataset,ddir in [('GSE83139','GSE83139/GEO/'),
                    ('GSE154126','GSE154126/GEO/'),
                    ('GSE101207','GSE101207/GEO/'),
                    ('GSE124742_GSE164875_patch','GSE124742_GSE164875/GEO/patch/'),
                    ('GSE124742_GSE164875_FACS','GSE124742_GSE164875/GEO/FACS/'),
                    ('GSE86469','GSE86469/GEO/'),
                    ('GSE81547','GSE81547/GEO/'),
                    ('GSE198623','P21000/sophie/human/'),
                    ('GSE81608','GSE81608/GEO/'),
                    ('GSE148073','GSE148073/GEO/')
                    ]:
    # Load adata
    print(dataset)
    adata=sc.read(path_rna+ddir+'adata_filtered.h5ad')
    # Subset to beta cells and disease phenotypes (removing pre pehnotypes)
    adata=adata[(adata.obs.cell_type=='beta').values& 
                (adata.obs.disease.isin(['healthy','T1D','T2D'])).values].copy()
    print(adata.shape)
    adata.obs['age_years']=adata.obs.age.apply(lambda x: h.age_years(x)).astype(float)
    if 'gene_symbol' in adata.var.columns:
        adata.var_names=adata.var.gene_symbol

    de_res=[]
    for marker_collection,marker_groups in [('known',markers),('new',markers_new)]:
        for group,genes in marker_groups.items():
            if group in group_queries:
                cells_group=set(adata.obs.query(group_queries[group]).index)
                cells_rest=[ c for c in adata.obs_names if c not in cells_group]
                if len(cells_group)>=min_cells and len(cells_rest)>=min_cells:
                    for gene in genes:
                        if gene in adata.var_names:
                            # Compute OvR DE
                            x_group=adata[list(cells_group),gene].X.todense()
                            x_rest=adata[cells_rest,gene].X.todense()
                            # If both groups are all 0 set lfc to 0 and pval to 1
                            if (x_group==0).all() and (x_rest==0).all():
                                p=1
                                lfc=0
                            else:
                                p=ttest_ind(x_group,x_rest,alternative='greater').pvalue[0]
                                lfc=np.log2((x_group.mean()+ 1e-9)/(x_rest.mean()+ 1e-9))
                            ratio_group=(x_group!=0).sum()/x_group.shape[0]
                            de_res.append({
                                'dataset':dataset,
                                'marker_collection':marker_collection,
                                'group':group,
                                'gene':gene,
                                'lfc':lfc,
                                'pval':p,
                                'ratio_group':ratio_group
                            })
    de_res=pd.DataFrame(de_res)
    display(de_res)
    de_results.append(de_res)
de_results=pd.concat(de_results)

# %%
# Are any results missing/inf
display(de_results[de_results.isna().any(axis=1)])
display(de_results[np.isinf(de_results[['lfc','pval','ratio_group']]).any(axis=1)])

# %% [markdown]
# Plot DE result of each marker across datasets

# %%
# Add pval log for plotting
de_results['-log10(pval)']=-np.log10(de_results.pval)
de_results.rename({'lfc':'lFC','ratio_group':'ratio of cells\nin group'},axis=1,inplace=True)
de_results['significant']=de_results.pval<0.05
# Order genes for plotting - UNUSED - resorted below
de_results['gene']=pd.Categorical(
    de_results['gene'],  
    sorted(pd.Series([m for  mc in (markers.values(),markers_new.values()) for ms in mc for m in ms]
             ).drop_duplicates().to_list()),
    ordered=True)

# %%
# Plot genes per group
# Plot known and new markers in the same line for a smaller plot
lp_min=de_results['-log10(pval)'].min()
lp_max=de_results['-log10(pval)'].max()
rg_min=de_results['ratio of cells\nin group'].min()
rg_max=de_results['ratio of cells\nin group'].max()
# N genes= n gridspec total, allow each ax as much place in gridspec as it has genes
genes_per_group=de_results.groupby(['marker_collection','group']).gene.nunique()
fig, axs = plt.subplots(nrows=1, ncols=genes_per_group.shape[0], sharex='col', sharey=True,
                               gridspec_kw={'width_ratios':genes_per_group.values,
                                           'wspace':0.01},
                               figsize=(genes_per_group.values.sum()*0.5, 3))
for idx,(collection,group) in enumerate(genes_per_group.index):
    data=de_results.query('marker_collection==@collection & group==@group')
    # Map genes to numbers to allow scatter computation along x
    genes=sorted(data.gene.unique())
    gene_map=dict(zip(genes,range(len(genes))))
    data['gene_idx']=data.gene.map(gene_map)
    data['gene_idx']=data.gene_idx+ np.random.normal(0,0.1,data.gene_idx.shape)
    ax=axs[idx]
    # 0 lfc mark
    ax.axhline(0,lw=0.2,c='k')
    # Plot genes
    legend=False if idx<genes_per_group.shape[0]-1 else 'auto'
    sb.scatterplot(x='gene_idx',y='lFC',hue='significant',size='ratio of cells\nin group',data=data,ax=ax,
                   # Specify colormap as else  may differ across subplots
                   legend=legend,palette={False:'gray',True:'red'},
                   #hue_norm=(lp_min,lp_max),
                   size_norm=(rg_min,rg_max),alpha=0.5)
    ax.set_yscale('symlog')
    ax.set_title(group)
    ax.margins(x=1/data.gene.nunique())
    # Map x labels back to genes
    ax.set_xticks(list(range(len(genes))))
    ax.set_xticklabels(labels=genes)   
    plt.setp(ax.get_xticklabels(),rotation=90) 
    
    ax.set_xlabel('')
    # Legend only at the end
    if legend!=False:
        l=plt.legend(bbox_to_anchor=(1.05,1.01),frameon=False)
    ax.set(facecolor = (0,0,0,0))
plt.savefig(path_fig+'scatter_beta_CLcoarse_human.png',dpi=300,bbox_inches='tight')

# %% [markdown]
# Keep in mind: The above significance isnt padj based.

# %%
