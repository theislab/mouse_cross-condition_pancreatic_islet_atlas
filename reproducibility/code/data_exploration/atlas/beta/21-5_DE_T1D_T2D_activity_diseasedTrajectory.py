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
# # DE genes across diabetes trajectory clusters
# Expression of T1D and T2D DE genes across T1D and T2D cluster progression trajectories.

# %%
import pandas as pd
import scanpy as sc
import numpy as np

import pickle
from pathlib import Path

from sklearn.preprocessing import minmax_scale,maxabs_scale

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/')
import helper as h
import importlib
importlib.reload(h)
import helper as h

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()
# %load_ext rpy2.ipython

import rpy2.rinterface_lib.callbacks
import logging
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)

from rpy2.robjects.packages import importr
grdevices = importr('grDevices')

# %% language="R"
# library('ComplexHeatmap')
# library(viridis)

# %%
ro.r('library("hypeR")')
ro.r("source(paste(Sys.getenv('WSCL'),'diabetes_analysis/data_exploration/','helper_hypeR.R',sep=''))")


# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/'
path_de='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/de/'
path_de1=path_de+'de_diseased_T1_NODelim_meld/'
path_de2=path_de+'de_diseased_T2_VSGSTZ_meld_covarStudy/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'
path_save=path_de+'compare_T1D-NODelim_T2D-VSGSTZ/'

# %%
# Saving figures
path_fig='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/figures/paper/'


# %%
adata_rn_b=sc.read(path_data+'data_rawnorm_integrated_analysed_beta_v1s1_sfintegrated.h5ad')
adata_rn_b.shape

# %%
# Add full gene symbol info
var=sc.read(path_data+'data_rawnorm_integrated_analysed_beta_v1s1_sfintegrated.h5ad',
           backed='r').var.copy()
var['gene_symbol']=h.get_symbols(var.index)
# Check that gene symbols not nan -
if (var.gene_symbol=='nan').sum() :
        raise ValueError('Some gene symbols are nan - requires different parsing')

# %%
# DE results and gene groups
de1=pd.read_table(path_de1+'deDataClusters.tsv')
de2=pd.read_table(path_de2+'deDataClusters.tsv')

# %% [markdown]
# ## Cell groups

# %% [markdown]
# Cell cluster sizes

# %%
# How many cells per ssample are in each cl to be plotted?
studies=['STZ',"VSG",'NOD_elimination']
groups=['adult1','adult2','D-inter.','db/db+mSTZ','NOD-D']
pd.set_option('display.max_rows', 500)
display(adata_rn_b.obs.query('study in @studies and hc_gene_programs_parsed in @groups'
                    )[['study_sample_design','hc_gene_programs_parsed']].astype(str).\
                groupby('study_sample_design')['hc_gene_programs_parsed'].value_counts())
pd.reset_option('display.max_rows')

# %%
# How many cells per study are in each cl to be plotted?
studies=['STZ',"VSG",'NOD_elimination']
groups=['adult2','D-inter.','db/db+mSTZ','NOD-D']
adata_rn_b.obs.query('study in @studies and hc_gene_programs_parsed in @groups'
                    )[['study','hc_gene_programs_parsed']].astype(str).\
                groupby('study')['hc_gene_programs_parsed'].value_counts()

# %% [markdown]
# C: There are enough cells to keep adult2 and D-intermediate for all models and the diabetes-model specific clusters for others.

# %% [markdown]
# Make cell groups

# %%
# Make groups
# Subset data
studies=['STZ',"VSG",'NOD_elimination']
groups=['adult2','D-inter.','db/db+mSTZ','NOD-D']
adata_rn_b_sub=adata_rn_b[
    adata_rn_b.obs.query('study in @studies and hc_gene_programs_parsed in @groups').index,
    :].copy()
adata_rn_b_sub.obs['group']=adata_rn_b_sub.obs.apply(
    lambda x:x.hc_gene_programs_parsed+' ('+x.study_parsed+')',axis=1)
# Keep only relevant groups
groups=[g for g in adata_rn_b_sub.obs['group'].unique() 
        if 'adult2' in g or 'D-inter.' in g 
        or g=='NOD-D (8-16wNOD)' or g=='db/db+mSTZ (mSTZ)' or g=='db/db+mSTZ (db/db)']
adata_rn_b_sub=adata_rn_b_sub[
    adata_rn_b_sub.obs.query('group in @groups').index, :].copy()
print(adata_rn_b_sub.obs.group.unique())

# %%
# Order groups
adata_rn_b_sub.obs['group']=pd.Categorical(
    values=adata_rn_b_sub.obs['group'],ordered=True,
    categories=list(adata_rn_b_sub.obs[['hc_gene_programs_parsed','study_parsed','group']].drop_duplicates(
).sort_values(['hc_gene_programs_parsed','study_parsed']).group)
)
adata_rn_b_sub.obs.group.cat.categories

# %% [markdown]
# ## Example genes plot
# Plot just some example known genes to show how they change across the trajectory 

# %%
genes={
       'down shared':['Gck','Atf3','Atf4','Mt2','Mt1','Prss53','Slc30a8'],
    'down db/db+mSTZ':['Ins1','Kcnj11','Pdx1','Vegfa'],
    'down NOD':['Txnip','Herpud1','Gm45805'],
     'up shared':['Gc','Chgb','Vgf','Iapp','Nucb2'],
      'up NOD':['B2m','H2-K1','Stat1','Cxcl10','Gbp7','Gbp8'],
       'up db/db+mSTZ':[
           'Ptprn', 'Aldh1a3','Gast','Aldob','Neurog3','Tesc','Dct']
      }
sc.pl.dotplot(adata_rn_b_sub, var_names=genes,
              gene_symbols='gene_symbol',use_raw=False,
              groupby='group',
              standard_scale='var',
              show=False)

# %%
for group,gs in genes.items():
    print(group)
    print('T1D')
    display(de1.query('gene_symbol in @gs')[['qval','log2fc','gene_symbol']])
    print('T2D')
    display(de2.query('gene_symbol in @gs')[['qval','log2fc','gene_symbol']])

# %% [markdown]
# ## Genes shared in T1D and T2D

# %% [markdown]
# ### DE genes expression across clusters

# %%
# Order groups for heatmap
categories=list(adata_rn_b_sub.obs[['hc_gene_programs_parsed','study_parsed','group']
                                  ].drop_duplicates().sort_values(
                ['study_parsed','hc_gene_programs_parsed']).group)


# %% [markdown]
# Genes in down groups in both T1D and T2D.

# %%
# Expression data of shared DE genes
g=list(set(de1[de1.hc.fillna('nan').str.contains('down')].gene)&
       set(de2[de2.hc.fillna('nan').str.contains('down')].gene))
x=adata_rn_b_sub[:,g].to_df()
x['group']=adata_rn_b_sub.obs.group
x=x.groupby('group').mean()
# Filter 0 expr as some genes are 0 expr when using only these clusters
x=x.T[(x>0).any()].T
# Sclae per dataset
x['dataset']=pd.Series(x.index).apply(lambda x:x.split('(')[1].split(')')[0]).values
x=x.groupby('dataset').apply(
    lambda x: pd.DataFrame(minmax_scale(x.drop('dataset',axis=1)),
                           index=x.index,columns=x.drop('dataset',axis=1).columns))

# %%
# Add gene symbols to df names 
x.columns=var.loc[x.columns,'gene_symbol']

# %%
sb.clustermap(x.loc[categories,:],row_cluster=False,xticklabels=True,figsize=(25,10))
# Could maybe plot lFC on top

# %% [markdown]
# Genes in up groups in both T1D and T2D 

# %%
# Expression data of shared DE genes
# Get mean expr per group for DE genes
g=list(set(de1[de1.hc.fillna('nan').str.contains('up')].gene)&
       set(de2[de2.hc.fillna('nan').str.contains('up')].gene))
x=adata_rn_b_sub[:,g].to_df()
x['group']=adata_rn_b_sub.obs.group
x=x.groupby('group').mean()
# Filter 0 expr as some genes are 0 expr when using only these clusters
x=x.T[(x>0).any()].T
# Sclae per dataset
x['dataset']=pd.Series(x.index).apply(lambda x:x.split('(')[1].split(')')[0]).values
x=x.groupby('dataset').apply(
    lambda x: pd.DataFrame(minmax_scale(x.drop('dataset',axis=1)),
                           index=x.index,columns=x.drop('dataset',axis=1).columns))

# %%
# Add gene symbols to df names 
x.columns=var.loc[x.columns,'gene_symbol']

# %%
sb.clustermap(x.loc[categories,:],row_cluster=False,xticklabels=True,figsize=(10,10))
# Could maybe plot lFC on top

# %% [markdown]
# Shared up and down on the same plot

# %%
g_down=list(set(de1[de1.hc.fillna('nan').str.contains('down')].gene)&
       set(de2[de2.hc.fillna('nan').str.contains('down')].gene))
g_up=list(set(de1[de1.hc.fillna('nan').str.contains('up')].gene)&
       set(de2[de2.hc.fillna('nan').str.contains('up')].gene))

x=adata_rn_b_sub[:,g_down+g_up].to_df()
x['group']=adata_rn_b_sub.obs.group
x=x.groupby('group').mean()
# Filter 0 expr as some genes are 0 expr when using only these clusters
x=x.T[(x>0).any()].T
# Sclae per dataset
x['dataset']=pd.Series(x.index).apply(lambda x:x.split('(')[1].split(')')[0]).values
x=x.groupby('dataset').apply(
    lambda x: pd.DataFrame(minmax_scale(x.drop('dataset',axis=1)),
                           index=x.index,columns=x.drop('dataset',axis=1).columns))

# Cluster genes per down/up
# Direction (also serves for annotation)
directions= pd.concat([pd.Series(['down']*len(g_down),index=g_down),
           pd.Series(['up']*len(g_up),index=g_up)])
gene_list= h.opt_order_withincl(x.T,directions, cl_order=['down','up'])

# %%
# Name axes
x.columns.name='genes'
x.index.name='cell clusters (per dataset)'

# %%
# Make columns anno
direction_cmap={'down':'#8a9e59','up':'#c97fac'}
col_anno=directions.map(direction_cmap)
col_anno.name='direction'

# %%
# Heatmap
w_dend=1
w_colors=0.3
nrow=x.shape[0]*0.24
ncol=x.shape[1]*0.05
w=ncol+w_dend
h=nrow+w_colors+w_dend
g=sb.clustermap(x.loc[categories,gene_list],
              row_cluster=False,col_cluster=False,xticklabels=False,
              col_colors=col_anno,
             cbar_pos=(0,0.46,0.03,0.1),
                figsize=(w,h),
                colors_ratio=(w_colors/w,0.8*w_colors/h),
                dendrogram_ratio=(w_dend/w,w_dend/h),
             )
# legends
# Cbar title
g.cax.set_title("relative\nexpression\nper dataset",fontsize=11)
# Add legend for col colors
legend_elements = [ Patch(alpha=0,label='direction')]+[
     Patch(facecolor=c,label=s) for s,c in direction_cmap.items()]
ax=g.fig.add_subplot(223)
ax.axis('off')
ax.legend(handles=legend_elements, bbox_to_anchor=(0.3,0.79),frameon=False)

#remove dendrogram
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)

# Remove colors tick
g.ax_col_colors.yaxis.set_ticks_position('none') 

# Save fig
plt.savefig(path_fig+'heatmap_beta_diabetesSharedDE_clFineDInterDataset.png',
            dpi=300,bbox_inches='tight')

# %% [markdown]
# Same as above, but with marked gene names

# %%
# Add to R expression 
ro.globalenv['x_ordered']=x.loc[categories,gene_list]

# %%
# Gene name anno for R
down_mark=['Siah2', 'Herc4', 'Hspa1a', 'Cbx4', 'Ddit3', 'Dnajb1', 'Usp27x', 'Fos', 'Jun', 
      'Gadd45b', 'Btg2', 'Prkab2', 'Junb', 'Dusp10', 'Npy', 'Dusp1', 'Gipr', 'Dusp18', 
      'Irf2bp2', 'Usp2', 'Atf3', 'Tnfaip3', 'Nnat', 'Hspa1b', 'Socs3']
up_mark=['Plaat3', 'Gc', 'Fabp5', 'Dapl1', 'Phlda3', 'Spp1', 'Vgf']
genes_show=up_mark+down_mark
gene_list_symbols=var.loc[gene_list,'gene_symbol'].values.ravel()
genes_show_idx=[np.argwhere(gene_list_symbols==g)[0][0] for g in genes_show]
ro.globalenv['genes_show']=genes_show
ro.globalenv['genes_show_idx']=genes_show_idx

# %%
# Add DE directions in R
ro.globalenv['directions_anno']=pd.DataFrame(directions).T[gene_list]

# %% language="R"
# # Gene annotation - names and direction
# # Prepare gene name data type
# genes_show<-unlist(genes_show)
# genes_show_idx<-unlist(genes_show_idx)
# # Prepare direction data type
# directions_anno<-factor(directions_anno,levels = c('down','up'))
# ha_col = columnAnnotation(
#     genes=anno_mark(at = genes_show_idx, labels = genes_show),
#     direction=directions_anno,
#     col = list(direction = setNames( c("#8a9e59", "#c97fac"),c('down','up'))),
#     annotation_name_side = "left"
# )
#

# %% magic_args="-w 800 -h 300 " language="R"
# h<-Heatmap(x_ordered,col=viridis(256),
#        cluster_columns = FALSE, cluster_rows = FALSE,
#        show_column_names = FALSE, show_row_names = TRUE,
#        row_title ="cell clusters\n(per dataset)",
#        top_annotation=ha_col,
#        row_names_side = "left",
#        heatmap_legend_param = list( title = "relative\nmean\nexpression\nper dataset\n"),
#        row_gap = unit(0, "mm"),
#        width= unit(15, "cm"), height= unit(4.5, "cm"),
#        show_row_dend = FALSE, 
#        )
# draw(h)

# %% magic_args="-i path_fig" language="R"
# pdf(file=paste0(path_fig,"heatmap_beta_diabetesSharedDE_clFineDInterDataset_annotated.pdf"), 
#     width=9.7, height=3)
# h<-Heatmap(x_ordered,col=viridis(256),
#        cluster_columns = FALSE, cluster_rows = FALSE,
#        show_column_names = FALSE, show_row_names = TRUE,
#        row_title ="cell clusters\n(per dataset)",
#        top_annotation=ha_col,
#        row_names_side = "left",
#        heatmap_legend_param = list( title = "relative\nmean\nexpression\nper dataset\n"),
#        row_gap = unit(0, "mm"),
#        width= unit(15, "cm"), height= unit(4.5, "cm"),
#        show_row_dend = FALSE, 
#        )
# draw(h)

# %% [markdown]
# ### Enrichment of shared up/down group genes

# %%
# Ref genes - present in both T1D and T2D summaries
ref=set.intersection(*[set(d.gene) for d in [de1,de2]])
ref=var.loc[ref,'gene_symbol'].to_list()
ro.globalenv['ref']=ref

# %%
# Get gene sets: GO+KEGG+Reactome
print('MSIGdb version:',ro.r(f'msigdb_version()'))
gene_sets_go=ro.r(f"msigdb_gsets_custom(species='Mus musculus',category='C5',subcategories=c('GO:BP','GO:CC','GO:MF'),size_range=c(5,500),filter_gene_sets=NULL,background=ref)")
gene_sets_kegg=ro.r(f"msigdb_gsets_custom(species='Mus musculus',category='C2',subcategories=c('KEGG'),size_range=c(5,500),filter_gene_sets=NULL,background=ref)")
gene_sets_reactome=ro.r(f"msigdb_gsets_custom(species='Mus musculus',category='C2',subcategories=c('REACTOME'),size_range=c(5,500),filter_gene_sets=NULL,background=ref)")
# %R -i gene_sets_go -i gene_sets_kegg -i gene_sets_reactome -o gene_sets gene_sets=c(gene_sets_go,gene_sets_kegg,gene_sets_reactome)
print('N gene sets:',len(gene_sets))
ro.globalenv['gene_sets']=gene_sets

# %%
# Perform enrichment per cl
enrich_datas={}
for direction in ['up','down']:
    print('Direction',direction)    
    # Query genes
    genes=list(set(de1[de1.hc.fillna('nan').str.contains(direction)].gene)&
               set(de2[de2.hc.fillna('nan').str.contains(direction)].gene))

    genes=var.loc[genes,'gene_symbol'].to_list()
    print('N genes %i'%len(genes))

    # Calculate enrichment
    enrich_fdr=0.25
    ro.globalenv['gs_fdr']=enrich_fdr
    ro.globalenv['genes']=genes
    res=ro.r(f'hypeR(signature=as.vector(unlist(genes)),genesets=gene_sets,test = "hypergeometric",background =  as.vector(unlist(ref)),pval = 1,fdr = gs_fdr,plotting = FALSE,quiet = TRUE)')
    ro.globalenv['res']=res
    data=ro.r(f'res$data')
    enrich_datas[direction]=data
    print('N enriched gene sets:',data.shape[0])

    if data.shape[0]>0:
        # Plot top enriched gene sets
        print('Top enriched gene sets')
        data['recall']=data['overlap']/data['geneset']
        data['query_size']=len(genes)
        h.plot_enrich(data=data.rename(
            {'label':'name','fdr':'p_value','overlap':'intersection_size'},axis=1),
            n_terms=20, save=False,min_pval=10**-30, max_pval=enrich_fdr,percent_size=True,
               recall_lim=(0,1))
        h.plot_enr_heatmap(data=data,n_gs=None,xticklabels=False,yticklabels=True)

# Save enrichment tables
writer = pd.ExcelWriter(path_save+'sharedClUpDown_enrichment.xlsx',
                        engine='xlsxwriter') 
for sheet,data in enrich_datas.items():
    data.to_excel(writer, sheet_name=str(sheet))   
writer.save()

# %%
