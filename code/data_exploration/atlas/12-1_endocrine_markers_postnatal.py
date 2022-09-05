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
import pickle
import gc

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb
import upsetplot as usp
from matplotlib.patches import Patch
import venn 
from adjustText import adjust_text
from matplotlib.lines import Line2D

from sklearn.preprocessing import minmax_scale,maxabs_scale

import diffxpy.api as de
from diffxpy.testing.det import DifferentialExpressionTestWald

from statsmodels.stats.multitest import multipletests

from scipy.cluster.hierarchy import linkage,dendrogram,fcluster,leaves_list
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/')
import helper as h
import importlib
importlib.reload(h)
import helper as h
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/data_exploration/')
import helper_diffxpy as hde
importlib.reload(hde)
import helper_diffxpy as hde

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
# library(proxy)
# library(seriation)
# library(dendextend)

# %%
ro.r('library(edgeR)')
ro.r('library("hypeR")')
ro.r("source(paste(Sys.getenv('WSC'),'diabetes_analysis/data_exploration/','helper_hypeR.R',sep=''))")


# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/'
path_save_r='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/celltypes/'
path_fig='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/figures/paper/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'


# %%
# Load adata (for DE)
adata_full=sc.read(path_data+'data_integrated_analysed.h5ad')

# %%
# Load var only
var=sc.read(path_data+'data_integrated_analysed.h5ad',backed='r').raw.var.copy()

# %%
# Load rawnorm (for plottting). Correctly norm expression is in layers X_sf_integrated
adata_rn=sc.read(path_data+'data_rawnorm_integrated_annotated.h5ad')
adata_rn.X=adata_rn.layers['X_sf_integrated']

# %% [markdown]
# #### Make pseudobulk
# Make pseudobulk per sample, sex, and ct, excluding: embryo (study and cts (cell types)), multiplets, proliferative - mixed endocrine cts as we did not resolve them, low quality cluster (but not removed on level of subtypes). 

# %%
# Keep specified cts
cts=['acinar', 'alpha','beta', 'delta', 'ductal', 'endothelial', 'gamma', 'immune',
 'schwann','stellate_activated', 'stellate_quiescent']
adata=adata_full[adata_full.obs.cell_type_integrated_v1.isin(cts),:]
# Remove embryo
adata=adata[adata.obs.study!='embryo',:]
adata.shape

# %% [markdown]
# Normalize data for pseudobulk DE

# %%
# Normalise
adata_norm=h.get_rawnormalised(adata.raw.to_adata(),sf_col='size_factors_integrated',
                               use_log=False,use_raw=False,copy=False)

# %% [markdown]
# Pseudobulk: sample-ct-sex

# %%
# Creat pseudobulk 
xs=[]
obss=[]
min_cells=20
for group,data in adata_norm.obs.groupby([
    'study','cell_type_integrated_v1','sex','study_sample_design']):
    if data.shape[0]>=min_cells:
        xs.append(np.array(adata_norm[data.index,:].X.sum(axis=0)).ravel())
        # Make obs
        # make sure obss is str not int if clusters
        obs={'study':str(group[0]),'cell_type':str(group[1]),
            'sex':str(group[2]),'sample':str(group[3])}
        obss.append(obs)
xs=pd.DataFrame(np.array(xs),columns=adata_norm.var_names)
obss=pd.DataFrame(obss)

# %%
print('xs',xs.shape)
print('obss',obss.shape)

# %% [markdown]
# #### DE

# %%
group='cell_type'

# %% magic_args="-i xs -i obss -i group" language="R"
# # Creat object
# y<-DGEList(counts = t(xs),  samples = obss)
# print(dim(y))

# %% language="R"
# # remove lowly expressed genes
# keep <- filterByExpr(y, group=y$samples[,group])
# y<-y[keep, , keep.lib.sizes=FALSE]
# print(dim(y))

# %% language="R"
# # Effective library size
# y <- calcNormFactors(y)

# %%
# Build design matrix
dmat_loc=pd.DataFrame(index=obss.index)
dmat_loc['Intercept']=1


condition='cell_type'
for val_idx,val in enumerate(sorted(obss[condition].unique())):
    # Do not add ct factor immune (intercept - as then getting DE results is 
    # easier if this not one of cts to be analyses)
    if val!='immune':
        dmat_loc.loc[obss[condition]==val,condition+'_'+str(val)]=1

condition='sex'
for val_idx,val in enumerate(
    sorted([cl for cl in obss[condition].unique() if cl!='male'])):
    dmat_loc.loc[obss[condition]==val,condition+'_'+str(val)]=1

condition='sample'
for val_idx,val in enumerate(
    sorted([cl for cl in obss[condition].unique() if cl!='STZ_G1_control'])):
    dmat_loc.loc[obss[condition]==val,condition+'_'+str(val)]=1

dmat_loc.fillna(0,inplace=True)
dmat_loc=dmat_loc.astype('float')

print('dmat_loc')
display(dmat_loc)

# %%
# Add design to R and name rows
ro.globalenv['design']=dmat_loc
ro.r('row.names(design)<-row.names(y$samples)')
ro.r('design')

# %%
rcParams['figure.figsize']=(20,10)
sb.heatmap(dmat_loc,xticklabels=True)

# %% language="R"
# # Robust dispersion 
# y <- estimateGLMRobustDisp(y, design)

# %%
print(1)

# %% language="R"
# # Fit - GLM not QL as used robust dispersion
# fit <- glmFit(y, design)

# %%
# %R fit$design

# %%
# Get DE tables - each endo ct to all other cts
summaries={}
for ct_test in ['alpha','beta','gamma','delta']:
    summaries_ct={}
    for ct_ref in [c for c in obss.cell_type.unique() if c!=ct_test]:
        # Coef2 in python indexing format
        coef2=np.argwhere(dmat_loc.columns=='cell_type_'+ct_test)[0][0]
        # Coef 1 is not needed if not existing as then testing signif of coef2 only 
        # (compared to intercept, should not be added explicitly!!!)
        # If coef1 is present then compare coef1 and coef2
        coef1=np.argwhere(dmat_loc.columns=='cell_type_'+ct_ref)
        if len(coef1)>0:
            coef1=coef1[0][0]
            contrast=np.zeros(dmat_loc.shape[1])
            contrast[coef1]=-1
            contrast[coef2]=1
            print('ct_test',ct_test,'ct_ref',ct_ref,'coef1',coef1,'coef2',coef2)
            print('Constrast',contrast)
            ro.globalenv['contrast']=contrast
            res=ro.r('glmLRT(fit, contrast=contrast)$table')
        else:
            print('ct_test',ct_test,'ct_ref',ct_ref,'coef2',coef2)
            coef2=coef2+1
            # Coef2 in R indexing format
            print('coef2 R',coef2)
            ro.globalenv['coef2']=coef2
            res=ro.r('glmLRT(fit, coef=coef2)$table')
        # Add padj
        res['padj']=multipletests(res["PValue"].values, alpha=0.05, method='fdr_bh')[1]
        summaries_ct[ct_ref]=res
    summaries[ct_test]=summaries_ct

# %% [markdown]
# #### Save

# %%
file=path_save_r+'DEedgeR_'
file

# %% magic_args="-i file" language="R"
# # Save fits
# save(y,fit,file=paste0(file,'fits.RData'))

# %%
# #%%R -i file
# Reload
#load(file=paste0(file,'fits.RData'))

# %%
# Save summary tables
pickle.dump(summaries,open(file+'endo_summaries.pkl','wb'))

# %%
# Reload results
#summaries=pickle.load(open(file+'endo_summaries.pkl','rb'))

# %% [markdown]
# ## Analyse DE

# %%
# Collect DE datas for single ct
# Test ct is compared against all otehr ref cts
cts_test=summaries.keys()
de_datas=defaultdict(list)
for ct_test,summaries_sub in summaries.items():
    for ct_ref,res in summaries_sub.items():
        de_data=res.copy()
        de_data['ct_test']=ct_test
        de_data['ct_ref']=ct_ref
        de_datas[ct_test].append(de_data)
for ct,de_data in de_datas.items():
    de_datas[ct]=pd.concat(de_data,axis=0)
    
# Find significant genes (against all cts) and min lfc across compared cts
LFC=1
PADJ=0.05
de_res={}
for ct,de_data in de_datas.items():
    # Find signif comparisons
    n_cts_ref=de_data['ct_ref'].nunique()
    de_data_signif=de_data.query('padj<@PADJ & logFC>@LFC'
                         ).reset_index().rename({'index':'gene'},axis=1).groupby('gene')
    de_genes=pd.DataFrame(de_data_signif.size().rename('n')).query('n==@n_cts_ref').index
    # Min lfc across comparisons
    de_data=de_data['logFC'].reset_index().rename({'index':'gene'},axis=1).\
        groupby('gene').min().loc[de_genes,:]
    de_res[ct]=de_data
    print(ct,'N genes:',de_data.shape[0])

# %% [markdown]
# Visualise selected genes with lowest LFC at different LFC thresholds to determine where the cut should be made to still get good markers

# %%
# At lFC=2
# Plot lowest lFC genes to make sure lFC thr is ok
for ct,de in de_res.items():
    print(ct)
    rcParams['figure.figsize']=(5,5)
    genes=var.loc[de.sort_values('logFC').query('logFC>2').head(4).index,'gene_symbol'].to_list()
    random_indices=np.random.permutation(list(range(adata_rn.shape[0])))
    sc.pl.umap(adata_rn[random_indices,:],color=genes,gene_symbols='gene_symbol',
               s=20,sort_order=False)
    sc.pl.umap(adata_rn,color=genes,gene_symbols='gene_symbol',
               s=20)

# %%
# At lFC=1.5
# Plot lowest lFC genes to make sure lFC thr is ok
for ct,de in de_res.items():
    print(ct)
    rcParams['figure.figsize']=(5,5)
    genes=var.loc[de.sort_values('logFC').query('logFC>1.5').head(4).index,'gene_symbol'].to_list()
    random_indices=np.random.permutation(list(range(adata_rn.shape[0])))
    sc.pl.umap(adata_rn[random_indices,:],color=genes,gene_symbols='gene_symbol',
               s=20,sort_order=False)
    sc.pl.umap(adata_rn,color=genes,gene_symbols='gene_symbol',
               s=20)

# %%
# At lFC=1
# Plot lowest lFC genes to make sure lFC thr is ok
for ct,de in de_res.items():
    print(ct)
    rcParams['figure.figsize']=(5,5)
    genes=var.loc[de.sort_values('logFC').query('logFC>1').head(4).index,'gene_symbol'].to_list()
    random_indices=np.random.permutation(list(range(adata_rn.shape[0])))
    sc.pl.umap(adata_rn[random_indices,:],color=genes,gene_symbols='gene_symbol',
               s=20,sort_order=False)
    sc.pl.umap(adata_rn,color=genes,gene_symbols='gene_symbol',
               s=20)

# %% [markdown]
# C: Low lFC genes may be not specific enough - expressed elsewhere or only in part of the cells. In general, lFC=1 thr is probably too low.

# %%
# Select lFC threshold
LFC_thr=1.5

# %%
# Print markers sorted by lFC
for ct,de, in de_res.items():
    print(ct)
    genes=var.loc[de.sort_values('logFC',ascending=False).query('logFC>@LFC_thr').index,
                  'gene_symbol'].to_list()
    print(len(genes))
    print(genes)

# %% [markdown]
# C: Some markers at the end of lists may not be as specific (e.g. for gamma)

# %%
# Plot top markers
for ct,de in de_res.items():
    print(ct)
    rcParams['figure.figsize']=(5,5)
    genes=var.loc[de.sort_values('logFC',ascending=False).query('logFC>@LFC_thr').head(4).index,'gene_symbol'].to_list()
    random_indices=np.random.permutation(list(range(adata_rn.shape[0])))
    sc.pl.umap(adata_rn[random_indices,:],color=genes,gene_symbols='gene_symbol',
               s=20,sort_order=False)
    sc.pl.umap(adata_rn,color=genes,gene_symbols='gene_symbol',
               s=20)

# %% [markdown]
# C: top markers look ok although some may be a bit lowly expressed

# %% [markdown]
# #### Save
# Report for all genes max padj and lFC as follows: if all LFC>0 report min, if all lFC<0 report max, if mixed signs report 0.

# %%
# Report max padj and parsed lfc
writer = pd.ExcelWriter(file+'summaries_ctParsed.xlsx',engine='xlsxwriter') 
for ct,de_data in de_datas.items():
    # Group by gene and compute min lFC and max padj across genes
    gene_data=de_data.reset_index().rename({'index':'gene'},axis=1).groupby('gene')
    de_summary=pd.concat([gene_data['logFC'].apply(
        lambda x: 0 if (x>0).any() and (x<0).any() else 
        (x.min() if (x>0).all() else x.max())),
                          gene_data['padj'].min()],axis=1
                        ).sort_values('logFC',ascending=False)
    de_summary['gene_symbol']=var.loc[de_summary.index,'gene_symbol']
    de_summary.to_excel(writer, sheet_name=str(ct))   
writer.save()

# %%
file+'summaries_ctParsed.xlsx'

# %% [markdown]
# ## Compare to human markers

# %% [markdown]
# #### Load human data form the paper

# %%
# Find position of table for each ct in the excel sheet from the paper
cts=['ALPHA','BETA','DELTA','GAMMA']
header=pd.read_excel(path_save_r+'10_1038_s41467-022-29588-8_source_data.xlsx',
                     sheet_name='figure 1',nrows=1)
ct_icol_start={}
ct_icol_end={}
ct_previous=None
for idx,col in enumerate(header.columns):
    if ct_previous is not None:
        ct_icol_end[ct_previous]=idx-1
    if col in cts:
        ct_icol_start[col]=idx
        ct_previous=col
print('Starts',ct_icol_start,'Ends',ct_icol_end)

# %%
# read markers
markers={}
for ct in cts:
    # read ct sheet subset
    m=pd.read_excel(path_save_r+'10_1038_s41467-022-29588-8_source_data.xlsx',
                     sheet_name='figure 1',usecols=range(ct_icol_start[ct],ct_icol_end[ct])
                   ).dropna(subset=[ct],axis=0)
    # Rename cols according to 2 line header
    col1_prev=None
    cols=[]
    for col1,col2 in zip(m.columns,m.iloc[0,:].values):
        if col1==ct:
            cols.append(col2)
        else:
            if not 'Unnamed' in col1:
                col1_prev=col1
            cols.append(col1_prev.replace(' ','_').replace('#','n')+'-'+col2)
    m.columns=cols
    # Subset to remove 2nd header row from data rows
    m=m.iloc[1:,:]
    # Result
    print(ct,m.shape)
    markers[ct]=m

# %% [markdown]
# Map to orthologues

# %%
# Orthologues
orthologues=pd.read_table(path_genes+'orthologues_ORGmus_musculus_ORG2homo_sapiens_V103.tsv'
                         ).rename(
    {'Gene name':'gs_mm','Human gene name':'gs_hs','Gene stable ID':'eid_mm',
    'Human gene stable ID':'eid_hs'},axis=1)
# Add Ppy as missing in this version of ortholgues
orthologues=orthologues.append({'eid_mm':'ENSMUSG00000017316','gs_mm':'Ppy', 
                    'eid_hs':'ENSG00000108849','gs_hs':'PPY'},ignore_index=True)

# %%
# Map to orthologues
markers_mm={}
for ct,markers_ct in markers.items():
    print(ct)
    markers_mm_ct=[]
    for idx,row in markers_ct.iterrows():
        gene=row['gene']
        for eid_mm in orthologues.query('gs_hs==@gene')['eid_mm']:
            row_mm=row.copy()
            row_mm['EID_mm']=eid_mm
            markers_mm_ct.append(row_mm)
    markers_mm[ct]=pd.DataFrame(markers_mm_ct)
    print("Succesfully maped human genes:",markers_mm[ct].gene.nunique(),
          'to N mouse genes:',markers_mm[ct].EID_mm.nunique(),
          'in N gene pairs:',markers_mm[ct].shape[0])

# %% [markdown]
# #### Mouse-human comparison

# %% [markdown]
# For each human marker orthologue retain smallest N datasets across ct comparisons (each endo ct was tested OvO against other endo cts) and where there are multiple human genes mapped to one mouse gene use the human gene with best/highest score as it is to be expected that there may be some non-functional orthologues.

# %%
# Min number of datasets where gene is marker across all ct comparisons
marker_scores={}
for ct,markers_mm_ct in markers_mm.items():
    marker_scores[ct]=pd.DataFrame(
        markers_mm_ct[[col for col in markers_mm_ct.columns 
                       if 'n_datasets' in col and 'vs' in col]
                     ].min(axis=1).values, index=markers_mm_ct["EID_mm"]
        ).groupby('EID_mm').max()

# %% [markdown]
# Mouse DE vulcano plot - max padj across cts and lFC across cts (if all lFC>0 use min, if all lFC<0 use max and if mixed lFC signs use 0), on it then color human markers by min N datasets where signif  across cts.

# %%
# Prepare data for vulcano plot
de_vulcano={}
for ct,de_data in de_datas.items():
    # Min lfc (based on abs) across comparisons
    de_group=de_data.reset_index().rename({'index':'gene'},axis=1).groupby('gene')
    de_vulcano[ct]=pd.concat([de_group['logFC'].apply(
        lambda x: 0 if (x>0).any() and (x<0).any() else 
        (x.min() if (x>0).all() else x.max())),
                              de_group['padj'].max()],axis=1)

# %%
# Are any padjs equal to 0? - if not can just do log
for ct,de in de_vulcano.items():
    print(ct,'any padj==0:', any(de.padj==0))

# %%
# make -log10padj for plotting
for ct,de in de_vulcano.items():
    de['-log10(padj)']=-np.log10(de.padj)

# %%
size=4
fig,axs=plt.subplots(1,len(de_vulcano),figsize=(size*len(de_vulcano),size),sharey=True,sharex=True)
legend_colors={}
for idx,ct in enumerate(de_vulcano):
    ax=axs[idx]
    base_color='#c9c9c9'
    genes_shared=set(de_vulcano[ct].index)&set(marker_scores[ct.upper()].index)
    genes_mouse=set(de_vulcano[ct].index)-genes_shared
    ax.scatter(x=de_vulcano[ct].loc[genes_mouse,'logFC'],
               y=de_vulcano[ct].loc[genes_mouse,'-log10(padj)'],c='#c9c9c9',s=10)
    # Min/max datasets based on the study info - to be consistent across plots
    min_datasets=0
    max_datasets=7
    for n_datasets in marker_scores[ct.upper()].loc[genes_shared,0].unique():
        sb.scatterplot(x=de_vulcano[ct].loc[genes_shared,'logFC'],
                   y=de_vulcano[ct].loc[genes_shared,'-log10(padj)'],
                   hue=marker_scores[ct.upper()].loc[genes_shared,:].\
                       rename({0:'n_datasets'},axis=1).\
                       query('n_datasets==@n_datasets')['n_datasets'],
                   ax=ax,linewidth=0,s=20, hue_norm=(min_datasets,max_datasets))
    ax.set_title(ct)
    hs,ls=ax.get_legend_handles_labels()
    for l,h in zip(ls,hs):
        legend_colors[l]=h.get_facecolor()
    ax.get_legend().remove()
    
    # Label genes also predicted as good markers in human
    texts = []
    for eid in marker_scores[ct.upper()].loc[genes_shared,:].rename({0:'n'},axis=1).\
                query('n>=5').index:
        gene_symbol=orthologues.query('eid_mm==@eid').gs_mm.to_list()[0]
        x=de_vulcano[ct].loc[eid,'logFC']
        y=de_vulcano[ct].loc[eid,'-log10(padj)']
        color_point='k'
        texts.append(ax.text(x, y, gene_symbol, color=color_point, fontsize=10))
    adjust_text(texts, expand_points=(2, 2),
        arrowprops=dict(arrowstyle="->",  color='k',  lw=1), ax=ax)
        
# make custom legend
legend_colors={k:legend_colors[k] for k in sorted(legend_colors)}
legend_elements =  [Line2D([0], [0], marker='o', color=c, lw=0, 
                          markerfacecolor=c, markersize=5,label=l) 
                    for l,c in legend_colors.items()]+[
                    Line2D([0], [0], marker='o', color=base_color, lw=0, 
                          markerfacecolor=base_color, markersize=5,label='NA')]
ax.legend(handles=legend_elements, bbox_to_anchor=(1.05,1),title='N human\ndatasets\n')


# %% [markdown]
# C: Seems that most robust human markers are also markers in mouse.

# %% [markdown]
# C: Note that some markers may be good when comapring to some cts but not others, thus geting overall low lFC

# %%
