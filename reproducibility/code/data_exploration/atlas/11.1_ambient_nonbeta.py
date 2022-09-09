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
# # Batch effects due to cell composition ambience
# Check if beta cell cluster spearates based on expression of other cell type markers (e.g. ambient genes), indicating poor batch effect removal. 
#
# Find markers of other cell types (OvR) and retain those that are highly ambient in individual batches and do not overlap beta markers.

# %%
import scanpy as sc
import pandas as pd
import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import h5py
import numpy as np
import seaborn as sb
import diffxpy.api as de
import pickle
import anndata
from scipy import sparse
from scipy.stats import spearmanr
from collections import defaultdict 
import matplotlib.patches as mpatches
from sklearn.preprocessing import minmax_scale

from importlib import reload  
import helper  as h
reload(h)
import helper as h

from scipy.cluster.hierarchy import linkage,dendrogram,fcluster,leaves_list
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import sklearn.preprocessing as pp
from matplotlib.patches import Patch

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/'
path_save='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/ovr_cell_type_integrated/'
path_save_ambient='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/ambient/'

UID2='integrated_ambient_composition'
path_de='/lustre/groups/ml01/workspace/karin.hrovatin//data/pancreas/scRNA/combined/ovr_cell_type_integrated/'

# %%
adata=sc.read(path_data+'data_integrated_analysed.h5ad')

# %%
# Extract rawnormalised data for gene scoring and plotting
adata_rawnorm=h.open_h5ad(path_data+'data_rawnorm_integrated_annotated.h5ad', unique_id2=UID2)

# %% [markdown]
# ## Top ambient genes across studies
# Find genes ambient in individual batches and make an overall union. Select a threshold used to classify gene as ambient based on number of genes being marked as ambient at that threshold (it is assumed that most genes are not highly ambient and will cluster at lower ambient proportions).

# %%
ambient_data=[('Fltp_2y','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islets_aged_fltp_iCre/rev6/'),
      ('Fltp_adult','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islet_fltp_headtail/rev4/'),
      ('Fltp_P16','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/salinno_project/rev4/'),
      ('NOD','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE144471/'),
      ('NOD_elimination','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE117770/'),
      ('spikein_drug','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE142465/'),
      ('embryo','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE132188/rev7/'),
      ('VSG','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/VSG_PF_WT_cohort/rev7/'),
      ('STZ','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islet_glpest_lickert/rev7/')]


# %%
# Find how many ambient genes would be selected as union across studies at different ambient ratio thresholds
ambient=defaultdict(set)
thresholds=list(1/np.logspace(3,18,num=200,base=2,dtype='int'))
for study,folder in ambient_data:
    ambient_df=pd.read_table(folder+"ambient_genes_topN_scores.tsv",index_col=0)
    for thresh in thresholds:
        ambient_study=ambient_df.index[(ambient_df>=thresh).any(axis=1)]
        ambient[thresh].update(list(ambient_study))
    #print(study,ambient_df.shape,ambient_study.shape)
n_ambient={thr:len(genes) for thr,genes in ambient.items()}

# %% [markdown]
# N slected ambient genes across studies/samples at different ambient ratio thresholds. If most genes are not ambient they will have low ambient proportion, thus find threshold that separates the few genes with high ambience from many genes with low ambience.

# %%
fig,ax=plt.subplots(1,2,figsize=(12,4))
ax[0].plot(list(n_ambient.keys()),list(n_ambient.values()))
ax[1].plot(list(n_ambient.keys()),list(n_ambient.values()))
ax[1].set_xscale('log')
ax[1].axvline(0.0003)

# %%
# Extract ambient genes at selected threshold
thr=0.0003
ambient=set()
for study,folder in ambient_data:
    ambient_df=pd.read_table(folder+"ambient_genes_topN_scores.tsv",index_col=0)
    ambient_study=ambient_df.index[(ambient_df>=thr).any(axis=1)]
    ambient.update(list(ambient_study))
    #print(study,ambient_df.shape,ambient_study.shape)
    print(study,'N ambient:',ambient_study.shape[0])
print('\nN combiend ambient:',len(ambient))

# %% [markdown]
# ## Ambient genes clusters
# Cluster top ambient genes and analyse their expression across cell types and on beta cells to mark top ambient genes coming from non-beta cells (for postnatal analysis).

# %% [markdown]
# #### Cluster cells to use as subpopulations that may generate ambient genes.

# %% [markdown]
# Subset to non-embryo data to recalculate embedding and clusters - since we are interested in postnatal ambient genes.

# %%
# remove embryo study
adata_sub=adata[adata.obs.study!='embryo']
adata_sub.shape

# %%
# recalculate UMAP on subsetted data
sc.pp.neighbors(adata_sub,n_pcs=0,use_rep='X_integrated')
sc.tl.umap(adata_sub)

# %%
# Compute fine clusters
res=2
sc.tl.leiden(adata_sub,resolution=res,key_added='leiden_r'+str(res))

# %%
ct_col='cell_type_integrated_v1'

# %%
# Cell types and fine clusters
rcParams['figure.figsize']=(9,9)
sc.pl.umap(adata_sub,color=[ct_col]+['leiden_r'+str(res)],wspace=0.4,s=20,ncols=2)

# %%
# Clusters used for computing gene similarity
cl_col='leiden_r2'

# %% [markdown]
# Correspondence between cl (cluster) and ct (cell type) scaled by rows.

# %%
# Correspondence between cl and ct
confusion_cl=adata_sub.obs.groupby([cl_col])[
    ct_col].value_counts(normalize=True).unstack().fillna(0)
confusion_ct=adata_sub.obs.groupby([ct_col])[
    cl_col].value_counts(normalize=True).unstack().fillna(0)
fig,ax=plt.subplots(1,2,figsize=(15,7))
sb.heatmap(confusion_cl,xticklabels=True,yticklabels=True,ax=ax[0])
sb.heatmap(confusion_ct,xticklabels=True,yticklabels=True,ax=ax[1])

# %% [markdown]
# Find beta clusters

# %%
# Find clusters with many beta cells (>10%)
beta_cls=set()
for cl in adata_sub.obs[cl_col].unique():
    ct_ratios=adata_sub[adata_sub.obs[cl_col]==cl,:].obs[ct_col].value_counts(normalize=True)
    if  'beta' in ct_ratios.index and ct_ratios['beta']>0.1:
        print(cl,'size:',adata_sub[adata_sub.obs[cl_col]==cl,:].shape[0])
        print(ct_ratios)
        beta_cls.add(cl)

# %% [markdown]
# C: Cluster 28 will not be regarded as beta as it has mainly poor quality cells. Cluster 41 will also not be regarded as beta as it is very mixed.

# %%
# Remove the two clusters
beta_cls.remove('28')  
beta_cls.remove('41') 
print(beta_cls)

# %% [markdown]
# #### Compute gene clusters

# %% [markdown]
# Prepare mean gene expression data across clusters

# %%
# Add cell cl and ct and UMAP info to rawnorm adata
adata_sub_rawnorm=adata_rawnorm[adata_sub.obs_names,:].copy()
adata_sub_rawnorm.obs[cl_col]=adata_sub.obs[cl_col].copy()
adata_sub_rawnorm.obs[ct_col]=adata_sub.obs[ct_col].copy()
adata_sub_rawnorm.obsm['X_umap']=adata_sub.obsm['X_umap'].copy()
adata_sub_rawnorm.obsp['connectivities']=adata_sub.obsp['connectivities'].copy()
adata_sub_rawnorm.obsp['distances']=adata_sub.obsp['distances'].copy()
# remove any previous info on scores
adata_sub_rawnorm.obs.drop([col for col in adata_sub_rawnorm.obs.columns
                        if 'ambient_score_cluster' in col],axis=1,inplace=True)

# %%
# Make df of ambient genes vs mean expression across cell clusters 
cell_cls=adata_sub.obs[cl_col].unique()
ambient_features=pd.DataFrame(index=ambient,columns=cell_cls)
for cell_cl in cell_cls:
    ambient_features[cell_cl]=np.array(adata_sub_rawnorm[
        adata_sub.obs[cl_col]==cell_cl,ambient_features.index].X.mean(axis=0)).ravel()

# %%
# prepare gene data for clustering
adata_genes=anndata.AnnData(ambient_features)

# %% [markdown]
# Leiden clustering of genes

# %%
# Embedding of genes for leiden
# Here vars should be at same scale as they are across cells and cells are all scaled - so
# no scaling is needed????
sc.pp.neighbors(adata_genes,use_rep='X',metric='correlation',knn=5) 
sc.tl.leiden(adata_genes,resolution=1)
sc.tl.umap(adata_genes)

# %%
# Cluster genes leiden
sc.tl.leiden(adata_genes,resolution=2)
print('N gene clusters:',adata_genes.obs.leiden.nunique())

# %% [markdown]
# #### Analyse gene clusters

# %%
#Add better colors to leiden clusters
adata_genes.uns['leiden_colors']=['tab:blue',  'tab:orange','tab:green','tab:red', 'tab:purple',
                'tab:brown','tab:pink', 'tab:gray', 'tab:olive','tab:cyan',
                'lightsteelblue','bisque','limegreen','lightcoral','plum',
                'peru','pink','darkgrey','yellowgreen','paleturquoise','yellow','black',
                'gold','brown']

# %%
sc.pl.umap(adata_genes,color=['leiden','hc'],wspace=0.3)

# %% [markdown]
# Compute gene scores for ambient clusters across cells

# %%
# Ambient gene cluster scores
gene_cl='leiden'
for ct in adata_genes.obs[gene_cl].unique():
    score_name='ambient_score_cluster_'+gene_cl+str(ct)
    sc.tl.score_genes(adata_sub_rawnorm, 
                      gene_list=adata_genes.obs_names[adata_genes.obs[gene_cl]==ct], 
                     score_name=score_name, use_raw=False)
    adata_sub_rawnorm.obs[score_name+'_scaled']=minmax_scale(adata_sub_rawnorm.obs[score_name])

# %% [markdown]
# Plot ambient cl gene expression (via scores) across cell clusters

# %%
# Row colors for heatmap, containing cell type info
ct_color_map=dict(zip(adata_sub.obs[ct_col].cat.categories,adata_sub.uns[ct_col+'_colors']))
cl_names=adata_sub.obs[cl_col].unique()
row_colors=pd.DataFrame({
              'is_beta':['g' if cl in beta_cls else 'r' for cl in cl_names],
            'main_'+ct_col:[ct_color_map[
                adata_sub[adata_sub.obs[cl_col]==cl].obs[ct_col].value_counts().index[0]
            ] for cl in cl_names]
              },index=cl_names)

# %%
# Prepare data for gene clusters heatmap
# Which clustering to show
gene_clustering='leiden'
# ordered genes by clusters
gene_list=[gene for cl in adata_genes.obs[gene_clustering].cat.categories
           for gene in ambient_features.index[adata_genes.obs[gene_clustering]==cl]]
# ordered gene colors 
cl_list=[adata_genes.uns[gene_clustering+'_colors'][idx] for idx,cl in 
         enumerate(adata_genes.obs[gene_clustering].cat.categories)
         for gene in ambient_features.index[adata_genes.obs[gene_clustering]==cl]]

x_temp=pd.DataFrame(pp.minmax_scale(ambient_features.T),
                    index=ambient_features.columns,columns=ambient_features.index)[gene_list]
fg=sb.clustermap(x_temp.loc[[str(x) for x in sorted([int(x) for x in x_temp.index])],:], 
              col_colors=cl_list, 
              col_cluster=False,row_cluster=True,
             xticklabels=False, yticklabels=True,
                row_colors=row_colors.loc[x_temp.index,:])
 # Adds block annotation titles as axis labels
fg.ax_col_colors.set_xlabel('Gene clusters',fontsize=20)
fg.ax_col_colors.xaxis.set_label_position('top') 
# legend for gene clusters
handles = [Patch(facecolor=c) for c in adata_genes.uns[gene_clustering+'_colors']]
plt.legend(handles, adata_genes.obs[gene_clustering].cat.categories, title='Gene cluster',
           bbox_to_anchor=(1.1, 1), bbox_transform=plt.gcf().transFigure)

# %% [markdown]
# Ambient gene cluster scores on uMAP

# %%
rcParams['figure.figsize']=(4,4)
sc.pl.umap(adata_sub_rawnorm,color=[col for col in adata_sub_rawnorm.obs.columns 
    if 'ambient_score_cluster_leiden' in col and '_scaled' in col], s=10)

# %% [markdown]
# #### Extract non-beta ambient genes
# Clusters with relatively low expression in beta cells.

# %%
# Non-beta ambient clusters
ambient_nobeta_leiden=[cl for cl in adata_genes.obs['leiden'].unique()
                      if cl not in ['0','3','5','6','11','14','15']]
print('Non-beta ambient clusters:',ambient_nobeta_leiden)
ambient_nobeta=adata_genes.obs_names[adata_genes.obs.leiden.isin(ambient_nobeta_leiden)]
print("N non-beta ambient genes:",len(ambient_nobeta))

# %% [markdown]
# ### Analyse

# %% [markdown]
# #### Ambient beta and non-beta gene numbers

# %% [markdown]
# How many non-beta ambient genes are among ambient genes of each study and sample

# %%
thr=0.0003
ambient=set()
ambient_missing_integrated=set()
ambient_nonbeta_info=[]
for study,folder in ambient_data:
    ambient_df=pd.read_table(folder+"ambient_genes_topN_scores.tsv",index_col=0)
    genes_anno_temp=genes_anno.copy()
    genes_anno_temp.index=genes_anno_temp['gene_symbol_'+study]
    for col in [col for col in ambient_df.columns if col !='mean_ambient_n_counts']:
        ambient_sample=list(ambient_df.index[ambient_df[col]>=thr])
        # EIDs of ambient genes
        ambient_sample_id=genes_anno_temp.loc[
            [g for g in ambient_sample if g in genes_anno_temp.index],'EID'].values
        sample=col.replace('mean_ambient_n_counts_','')
        ambient_nonbeta_info.append(
            {'study':study,'sample':sample,'study_sample':study+'_'+sample,
             'n_beta_ambient':len(set(ambient_sample_id)-set(ambient_nobeta)),
             'n_nonbeta_ambient':len(set(ambient_sample_id)&set(ambient_nobeta)),
             'n_ambient':len(ambient_sample)
            })
ambient_nonbeta_info=pd.DataFrame(ambient_nonbeta_info)

# %%
fig, ax = plt.subplots(figsize=(20,4))
ax.bar(ambient_nonbeta_info['study_sample'], ambient_nonbeta_info['n_beta_ambient'], 
       0.5 ,  label='beta')
ax.bar(ambient_nonbeta_info['study_sample'], ambient_nonbeta_info['n_nonbeta_ambient'], 
       0.5 , bottom=ambient_nonbeta_info['n_beta_ambient'],label='non-beta')
a=plt.xticks(rotation=90)
a=ax.legend(title='Ambient')
a=ax.set_ylabel('n genes')

# %% [markdown]
# #### Check where are known non-beta ambient genes
# Are known non-beta ambient genes (e.g. other hormones, markers of other cts) in beta or non-beta cluster.

# %%
# Load gene anno
genes_anno=pd.read_table('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/genomeAnno_ORGmus_musculus_V103.tsv',
                         index_col=0)

# %%
# Known non-beta ambient genes EIDs
known_ct_ambient=['Gcg','Ppy','Pyy','Sst','Cxcl2','Prss2','Cel','Krt19']
genes_anno.index=genes_anno['Gene name']
eids=genes_anno.loc[known_ct_ambient,'EID']
genes_anno.index=genes_anno['EID']
print('Example non-beta ambient genes:',eids)

# %%
# In which gene cl are genes
print('Gene cls containing non-beta example ambient genes')
gene_cl[eids]

# %%
# Which genes were not marked as non-beta
print('Ambient genes from other cts not removed with ambient_nonbeta')
print(genes_anno.loc[[e for e in eids if e not in ambient_nonbeta],'Gene name'])

# %% [markdown]
# C: It seems that some genes from other cell types were not removed as they are located in cl 5, which is also high in beta cells (some endocrine genes).

# %% [markdown]
# Which other genes are in cluster 5

# %%
genes5=gene_cl.index[gene_cl=='5']
print('N genes in cl 5:',len(genes5))
print(genes_anno.loc[genes5,'Gene name'].values.tolist())

# %% [markdown]
# C: Besides hormones from other cell types this cluster also contains interesting genes like Rbp4, Chga, Chgb, Wnt4, ... Thus it will not be removed. If needed these genes can be removed later in downstream analyses.

# %% [markdown]
# #### Save data

# %% [markdown]
# Save ambient genes not commming from beta cells

# %%
pickle.dump(
    # Also save ambinet threshold at which genew were obtained so 
    # that same filtering can be used lattere
    {'ambient_thr':thr,'ambient_nonbeta':ambient_nobeta},
    open(path_save_ambient+'ambient_nonbeta.pkl','wb'))

# %% [markdown]
# Save used clustering information

# %%
pickle.dump({'cell_cl':adata_sub.obs[cl_col], 'gene_cl':adata_genes.obs['leiden']},
            open(path_save_ambient+'ambient_clustering.pkl','wb'))
