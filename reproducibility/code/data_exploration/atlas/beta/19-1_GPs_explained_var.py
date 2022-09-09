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
# # Explained var by GPs
#
# Compute amount of explained variation by GPs using top PCs as baseline. Ideally would compute over all samples from all studies (included in atlas and external) to make sure that variance between all possible conditions is included, but cant really as different preprocessing due to different technologies so expression across studies is not directly comparable - so do per study, except for the atlas, where we made sure that this is comparable.

# %%
import scanpy as sc
import pandas as pd
import numpy as np
import pickle as pkl

from sklearn.preprocessing import minmax_scale,maxabs_scale

from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from scib import metrics as sm
from sklearn.linear_model import LinearRegression

# %%
path_gp='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/moransi/sfintegrated/'
path_rna='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/'
path_data=path_rna+'combined/'
path_save=path_gp+'explained_var/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'

# %%
# Saving figures
path_fig='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/figures/paper/'


# %%
# orthologue information
orthologues=pd.read_table(path_genes+'orthologues_ORGmus_musculus_ORG2homo_sapiens_V103.tsv'
                         ).rename(
    {'Gene name':'gs_mm','Human gene name':'gs_hs',
     'Gene stable ID':'eid_mm','Human gene stable ID':'eid_hs'},axis=1)

# %%
# Load GPs
genes_hc=pd.read_table(path_gp+'gene_hc_t'+str(2.4)+'.tsv',sep='\t',index_col=0)
genes_hc['hc']=genes_hc['hc'].astype('category')

# %% [markdown]
# ## Compute explained var
# Compute explained var based on each GP or their combination in top PCs computed on expression data per dataset (using atlas as single dataset). In atlas also compute explained var by study. 
#
# Use linear regresison to see how well individual PCs are explained (R2) and weight the R2 score by variance explained by each PC, summing explained var from all PCs to get final explained var.
#
# Compare explained var of GPs to explained var of random gene groups of matched size. Compute random groups multiple times to get a proxy of random gene groups explained var distn.

# %%
# N PCs to use
NPC=50


# %% [markdown]
# Helper functions

# %%
def explained_pca_var(X_pca,covariates,pca_var,n_comps):
    """
    How much variance from PCs is explained.
    Adapted from scIB.
    :param X_pca: PCA embedding
    :param covariates: Design matrix for regression
    :param pca_var: Relative var explained by each PC
    :param n_comps: N PC components in the embedding
    """
    r2 = []
    for i in range(n_comps):
        pc = X_pca[:, [i]]
        lm = LinearRegression()
        lm.fit(covariates, pc)
        r2_score = np.maximum(0, lm.score(covariates, pc))
        r2.append(r2_score)
    Var = pca_var / sum(pca_var) 
    R2Var = sum(r2 * Var) 
    return R2Var


# %%
def explaied_var_helper(gp_study:list,adata,gene_groups_collection:dict,study_cov:np.array):
    """
    Helper to calculate explained var by GPs or studies or both (concatenates both as predictor)
    Also computes for individual GPs (without study, not computed if study is True)
    :param gp_study: list of tuples giving 
    (GP name or None (if not used), True/False for using study)
    :param adata: Pp adata
    :param gene_groups_collection: Dict with keys GP type names and values GP dicts with
    keys being gp names and values gene lists matching adata var names:
    {gp_type:{gp_name:genes,...}}
    :param study_cov: Array of study embedding or None if not used
    """
    # Score for GPs (true or random), study, and true GP+study
    # In each GP set compute explained var over all GPs
    explained_var=[]
    for gp,study in gp_study:
        print('GP and study:',gp,study)
        cov=[]
        # Use GP scores for prediction
        scores_gp={}
        if gp is not None:
            gene_groups=gene_groups_collection[gp]
            for gp_component,genes in gene_groups.items():
                scores_gp[gp_component]=sc.tl.score_genes(
                    adata, gene_list=genes, score_name='score', 
                    use_raw=False,copy=True,random_state=0
                    ).obs['score'].values
            cov.append(np.array(list(scores_gp.values())).T)
        # Use study cov as prediction
        if study:
            cov.append(study_cov) # Not full rank but ok?
        cov=np.concatenate(cov,axis=1)
        print('cov shape:',cov.shape)
        explained_var.append({
            'GPs':gp,'study':study,
            'explained_var_ratio':explained_pca_var(X_pca=adata.obsm['X_pca'],
                          covariates=cov,
                          pca_var=adata.uns['pca']['variance'],
                          n_comps=NPC)})
        # Compute also var exmplained by each GP if present
        if not study:
            for gp_component,score in scores_gp.items():
                explained_var.append({
                    'GPs':gp+'_GPcomponent'+str(gp_component),'study':False,
                    'explained_var_ratio':explained_pca_var(X_pca=adata.obsm['X_pca'],
                              covariates=np.array(score).reshape(-1,1),
                              pca_var=adata.uns['pca']['variance'],
                              n_comps=NPC)})
        del cov
        del scores_gp
    explained_var=pd.DataFrame(explained_var)
    return explained_var


# %% [markdown]
# Store dataset metadata and size infro from all datasets for below plots

# %%
# Store dataset info
dataset_info=[]

# %% [markdown]
# Compute explained var for all datasets individually

# %% [markdown]
# ### Atlas per study
# This is done as sanity check to make sure that GPs look like expected on individual studies within atlas.

# %%
dataset='mouseAtlas'
adata_full=sc.read(path_data+'data_rawnorm_integrated_analysed_beta_v1s1_sfintegrated.h5ad',backed='r')

# %%
studies=[s for s in adata_full.obs.study.unique() if s!='embryo']

# %%
for study in studies:
    print(study)
    adata=adata_full[adata_full.obs.study==study,:].copy()
    print(adata.shape)
    
    # PP
    
    # Remove lowly expressed genes
    sc.pp.filter_genes(adata,min_cells=adata.shape[0]*0.01)
    print('After lowly expr filtering:',adata.shape)
    # Compute HVG and PCA
    sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes =2000)
    sc.tl.pca(adata, n_comps=NPC, use_highly_variable=True)
    
    # GP selection
    
    # Subset GPs to only genes in data
    genes_hc_sub=genes_hc.loc[[g for g in adata.var_names if g in genes_hc.index],:]
    print('N GP genes:',genes_hc_sub.shape[0])
    # Make sizes of GPs for random creation
    # Splits for np split
    splits=[]
    n_curr=0
    # MAke sure this is ordered as gps
    for n in genes_hc_sub['hc'].value_counts(sort=False).values:
        splits.append(n_curr+n)
        n_curr+=n
    # Do not use last one as automatically createcd
    splits=splits[:-1]   
    
    # Gene groups - GP and random
    gene_groups_collection={}
    # Add true GPs
    gene_groups_collection['GPs']={hc:list(g) 
                                   for hc,g in genes_hc_sub.groupby('hc').groups.items()}
    # Add random gene groups of GP sizes
    np.random.seed(0)
    for i in range(10):
        # Random idx groups of GP sizes
        random_indices=np.split(np.random.permutation(list(range(adata.shape[1]))
                                                 )[:genes_hc_sub.shape[0]],splits)
        # Map idx to genes
        gene_groups_collection['GPs_random'+str(i)]={
            genes_hc_sub.hc.cat.categories[gp_idx]:adata.var_names[idxs] 
            for gp_idx,idxs in enumerate(random_indices)}
        
        
    # Explained var
    explained_var=explaied_var_helper(
        gp_study=[(gp,False) for gp in gene_groups_collection.keys()],
        adata=adata,gene_groups_collection=gene_groups_collection,study_cov=None)
    display(explained_var)
    explained_var.to_csv(path_save+dataset+'-study_'+study+'.tsv',sep='\t',index=False)

# %%
del adata_full
del adata

# %% [markdown]
# #### Analyse

# %%
# Load and parse all results into shared table
explained_var_all=[]
for study in studies:
    explained_var=pd.read_table(path_save+dataset+'-study_'+study+'.tsv')
    explained_var['component']=explained_var['GPs'].fillna('NA').apply(
        lambda x: np.nan if x =='NA' else 
        ('GPs' if 'component' not in x else x.split('component')[1]))
    explained_var['random']=explained_var['GPs'].str.contains('random')
    explained_var['dataset']=study
    explained_var_all.append(explained_var)
explained_var_all=pd.concat(explained_var_all)    

# %% [markdown]
# Explained var in each study by all GPs

# %%
# Plot explained var vs random explained var for each dataset
rcParams['figure.figsize']=(5,2.5)
sb.swarmplot(x='dataset',y='explained_var_ratio',hue='random',s=5,
            data=explained_var_all.query('~GPs.str.contains("component")',engine='python'))

a=plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1.01,1.03))

# %% [markdown]
# Explained var in each study by individual GPs. How much var is explained and how many stds is this from mean on random gene groups.

# %%
# Get explained var for each component in each dataset and compare to random var
explained_component=[]
for (dataset, component),data in explained_var_all.query(
    'study==False & component!="GPs"').groupby(['dataset','component']):
    explained=data[~data.random.astype(bool)]['explained_var_ratio'].values[0]
    explained_random=data[data.random.astype(bool)]['explained_var_ratio']
    explained_n_std=(explained-explained_random.mean())/explained_random.std()
    explained_component.append(
        {'dataset':dataset,'component':component,
        'explained_var':explained,'explained_var_n_std':explained_n_std})
explained_component=pd.DataFrame(explained_component)  
# Sort components for plotting
explained_component['component']=pd.Categorical(
    explained_component['component'],
    [str(c) for c in sorted(explained_component['component'].astype(int).unique())],
    True)
explained_component['dataset']=pd.Categorical(
    explained_component['dataset'],sorted(explained_component['dataset'].unique()),True)

# %%
# Explained var by component
rcParams['figure.figsize']=(explained_component.dataset.nunique()*0.32,
                            explained_component.component.nunique()*0.32)
sb.scatterplot(x='dataset',y='component',hue='explained_var',size='explained_var_n_std',
               data=explained_component,palette='magma')
plt.legend(bbox_to_anchor=(1.05, 1))
a=plt.xticks(rotation=90)

# %% [markdown]
# Contribution of GP components for each dataset, scaled (maxabs) across GPs

# %%
# Scaling contibution across components
figsize=(explained_component.dataset.nunique()*0.4,
         explained_component.component.nunique()*0.32)
rcParams['figure.figsize']=figsize
scaled=pd.crosstab(columns=explained_component['dataset'],index=explained_component['component'],
            values=explained_component['explained_var'],aggfunc='mean')
scaled=pd.DataFrame(maxabs_scale(scaled),index=scaled.index,columns=scaled.columns)
sb.heatmap(scaled,xticklabels=True,yticklabels=True)

# %% [markdown]
# C: Expected GPs are most explanatory in studies of atlas. E.g. GPs connected to diabetes for datasets containing healthy and diabetic samples; GPs separating aged sexes in 2y datasets, ...

# %% [markdown]
# ### Atlas - using as single study

# %%
dataset='mouseAtlas'
info={}
info['dataset']=dataset
info['organism']='mouse'
info['technology']='10X'

# %%
adata=sc.read(path_data+'data_rawnorm_integrated_analysed_beta_v1s1_sfintegrated.h5ad')
info['n_cells']=adata.shape[0]
adata.shape

# %%
# Remove lowly expressed genes
sc.pp.filter_genes(adata,min_cells=adata.shape[0]*0.01)
print(adata.shape)

# %%
# Compute HVG and PCA
sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes =2000)
sc.tl.pca(adata, n_comps=NPC, use_highly_variable=True)

# %%
# Subset GPs to only genes in data
genes_hc_sub=genes_hc.loc[[g for g in adata.var_names if g in genes_hc.index],:]
print(genes_hc_sub.shape[0])
info['n_gp_genes']=genes_hc_sub.shape[0]
# Make sizes of GPs for random creation
# Splits for np split
splits=[]
n_curr=0
# MAke sure this is ordered as gps
for n in genes_hc_sub['hc'].value_counts(sort=False).values:
    splits.append(n_curr+n)
    n_curr+=n
# Do not use last one as automatically createcd
splits=splits[:-1]    

# %%
# Gene groups - GP and random
gene_groups_collection={}
# Add true GPs
gene_groups_collection['GPs']={hc:list(g) 
                               for hc,g in genes_hc_sub.groupby('hc').groups.items()}
# Add random gene groups of GP sizes
np.random.seed(0)
for i in range(10):
    # Random idx groups of GP sizes
    random_indices=np.split(np.random.permutation(list(range(adata.shape[1]))
                                             )[:genes_hc_sub.shape[0]],splits)
    # Map idx to genes
    gene_groups_collection['GPs_random'+str(i)]={
        genes_hc_sub.hc.cat.categories[gp_idx]:adata.var_names[idxs] 
        for gp_idx,idxs in enumerate(random_indices)}

# %%
# Study dimmies
study_cov = pd.get_dummies(adata.obs['study'])

# %%
# Explained var
explained_var=explaied_var_helper(
    gp_study=[(gp,False) for gp in gene_groups_collection.keys()]+[(None,True),('GPs',True)],
    adata=adata,gene_groups_collection=gene_groups_collection,study_cov=study_cov)
display(explained_var)

# %%
explained_var.to_csv(path_save+dataset+'.tsv',sep='\t',index=False)

# %%
info

# %%
dataset_info.append(info)

# %% [markdown]
# ### GSE137909

# %%
dataset='GSE137909'
info={}
info['dataset']=dataset
info['organism']='mouse'
info['technology']='STRT-seq'

# %%
ddir=dataset+'/GEO/'
adata=sc.read(path_rna+ddir+'adata.h5ad')
adata=adata[(adata.obs.cell_type=='beta').values ]
info['n_cells']=adata.shape[0]
adata.shape

# %%
# Remove lowly expressed genes
sc.pp.filter_genes(adata,min_cells=adata.shape[0]*0.01)
print(adata.shape)

# %%
# Compute HVG and PCA
sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes =2000)
sc.tl.pca(adata, n_comps=NPC, use_highly_variable=True)

# %%
adata.var_names[:3]

# %%
# Subset GPs to only genes in data
genes_hc_sub=genes_hc.loc[[g for g in adata.var_names if g in genes_hc.index],:]
print(genes_hc_sub.shape[0])
info['n_gp_genes']=genes_hc_sub.shape[0]
# Make sizes of GPs for random creation
# Splits for np split
splits=[]
n_curr=0
# Make sure this is ordered as gps
for n in genes_hc_sub['hc'].value_counts(sort=False).values:
    splits.append(n_curr+n)
    n_curr+=n
# Do not use last one as automatically createcd
splits=splits[:-1]    

# %%
# Gene groups - GP and random
gene_groups_collection={}
# Add true GPs
gene_groups_collection['GPs']={hc:list(g) 
                               for hc,g in genes_hc_sub.groupby('hc').groups.items()}
# Add random gene groups of GP sizes
np.random.seed(0)
for i in range(10):
    # Random idx groups of GP sizes
    random_indices=np.split(np.random.permutation(list(range(adata.shape[1]))
                                             )[:genes_hc_sub.shape[0]],splits)
    # Map idx to genes
    gene_groups_collection['GPs_random'+str(i)]={
        genes_hc_sub.hc.cat.categories[gp_idx]:adata.var_names[idxs] 
        for gp_idx,idxs in enumerate(random_indices)}

# %%
# Explained var
explained_var=explaied_var_helper(
    gp_study=[(gp,False) for gp in gene_groups_collection.keys()],
    adata=adata,gene_groups_collection=gene_groups_collection,study_cov=None)
display(explained_var)

# %%
explained_var.to_csv(path_save+dataset+'.tsv',sep='\t',index=False)

# %%
info

# %%
dataset_info.append(info)

# %% [markdown]
# ### GSE83146

# %%
dataset='GSE83146'
info={}
info['dataset']=dataset
info['organism']='mouse'
info['technology']='FluidigmC1_SMARTer'

# %%
ddir=dataset+'/GEO/'
adata=sc.read(path_rna+ddir+'adata.h5ad')
adata=adata[(adata.obs.cell_type=='beta').values ]
info['n_cells']=adata.shape[0]
adata.shape

# %%
# Remove lowly expressed genes
sc.pp.filter_genes(adata,min_cells=adata.shape[0]*0.01)
print(adata.shape)

# %%
# Compute HVG and PCA
sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes =2000)
sc.tl.pca(adata, n_comps=NPC, use_highly_variable=True)

# %%
adata.var_names[:3]

# %%
# Subset GPs to only genes in data
# Map to gene ids in data before making GP sub
hcs=[]
gids=[]
for eid,hc in genes_hc['hc'].iteritems():
    for gid in adata.var.query('EID==@eid').index:
        hcs.append(hc)
        gids.append(gid)
genes_hc_sub=pd.DataFrame({'hc':hcs},index=gids,dtype='category')
print(genes_hc_sub.shape[0])
info['n_gp_genes']=genes_hc_sub.shape[0]
# Make sizes of GPs for random creation
# Splits for np split
splits=[]
n_curr=0
# Make sure this is ordered as gps
for n in genes_hc_sub['hc'].value_counts(sort=False).values:
    splits.append(n_curr+n)
    n_curr+=n
# Do not use last one as automatically createcd
splits=splits[:-1]    

# %%
# Gene groups - GP and random
gene_groups_collection={}
# Add true GPs
gene_groups_collection['GPs']={hc:list(g) 
                               for hc,g in genes_hc_sub.groupby('hc').groups.items()}
# Add random gene groups of GP sizes
np.random.seed(0)
for i in range(10):
    # Random idx groups of GP sizes
    random_indices=np.split(np.random.permutation(list(range(adata.shape[1]))
                                             )[:genes_hc_sub.shape[0]],splits)
    # Map idx to genes
    gene_groups_collection['GPs_random'+str(i)]={
        genes_hc_sub.hc.cat.categories[gp_idx]:adata.var_names[idxs] 
        for gp_idx,idxs in enumerate(random_indices)}

# %%
# Explained var
explained_var=explaied_var_helper(
    gp_study=[(gp,False) for gp in gene_groups_collection.keys()],
    adata=adata,gene_groups_collection=gene_groups_collection,study_cov=None)
display(explained_var)

# %%
explained_var.to_csv(path_save+dataset+'.tsv',sep='\t',index=False)

# %%
info

# %%
dataset_info.append(info)

# %% [markdown]
# ### GSE83139

# %%
dataset='GSE83139'
info={}
info['dataset']=dataset
info['organism']='human'
info['technology']='Fluidigm_SMART-seq'

# %%
ddir=dataset+'/GEO/'
adata=sc.read(path_rna+ddir+'adata_filtered.h5ad')
adata=adata[(adata.obs.cell_type=='beta').values ]
info['n_cells']=adata.shape[0]
adata.shape

# %%
# Remove lowly expressed genes
sc.pp.filter_genes(adata,min_cells=adata.shape[0]*0.01)
print(adata.shape)

# %%
# Compute HVG and PCA
sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes =2000)
sc.tl.pca(adata, n_comps=NPC, use_highly_variable=True)

# %%
adata.var.iloc[:1]

# %%
# Subset GPs to only genes in data
# Map to gene ids and orthologues in data before making GP sub
hcs=[]
gids=[]
for eid,hc in genes_hc['hc'].iteritems():
    for gid in [gid for gid in orthologues.query('eid_mm==@eid').gs_hs 
                if gid in adata.var_names]:
        hcs.append(hc)
        gids.append(gid)
genes_hc_sub=pd.DataFrame({'hc':hcs},index=gids,dtype='category')
print(genes_hc_sub.shape[0])
info['n_gp_genes']=genes_hc_sub.shape[0]
# Make sizes of GPs for random creation
# Splits for np split
splits=[]
n_curr=0
# Make sure this is ordered as gps
for n in genes_hc_sub['hc'].value_counts(sort=False).values:
    splits.append(n_curr+n)
    n_curr+=n
# Do not use last one as automatically createcd
splits=splits[:-1]    

# %%
# Gene groups - GP and random
gene_groups_collection={}
# Add true GPs
gene_groups_collection['GPs']={hc:list(g) 
                               for hc,g in genes_hc_sub.groupby('hc').groups.items()}
# Add random gene groups of GP sizes
np.random.seed(0)
for i in range(10):
    # Random idx groups of GP sizes
    random_indices=np.split(np.random.permutation(list(range(adata.shape[1]))
                                             )[:genes_hc_sub.shape[0]],splits)
    # Map idx to genes
    gene_groups_collection['GPs_random'+str(i)]={
        genes_hc_sub.hc.cat.categories[gp_idx]:adata.var_names[idxs] 
        for gp_idx,idxs in enumerate(random_indices)}

# %%
# Explained var
explained_var=explaied_var_helper(
    gp_study=[(gp,False) for gp in gene_groups_collection.keys()],
    adata=adata,gene_groups_collection=gene_groups_collection,study_cov=None)
display(explained_var)

# %%
explained_var.to_csv(path_save+dataset+'.tsv',sep='\t',index=False)

# %%
info

# %%
dataset_info.append(info)

# %% [markdown]
# ### GSE154126

# %%
dataset='GSE154126'
info={}
info['dataset']=dataset
info['organism']='human'
info['technology']='SMART-seq'

# %%
ddir=dataset+'/GEO/'
adata=sc.read(path_rna+ddir+'adata_filtered.h5ad')
adata=adata[(adata.obs.cell_type=='beta').values ]
info['n_cells']=adata.shape[0]
adata.shape

# %%
# Remove lowly expressed genes
sc.pp.filter_genes(adata,min_cells=adata.shape[0]*0.01)
print(adata.shape)

# %%
# Compute HVG and PCA
sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes =2000)
sc.tl.pca(adata, n_comps=NPC, use_highly_variable=True)

# %%
adata.var.iloc[:1]

# %%
# Subset GPs to only genes in data
# Map to gene ids and orthologues in data before making GP sub
hcs=[]
gids=[]
for eid,hc in genes_hc['hc'].iteritems():
    for gid in [gid for gid in orthologues.query('eid_mm==@eid').gs_hs 
                if gid in adata.var_names]:
        hcs.append(hc)
        gids.append(gid)
genes_hc_sub=pd.DataFrame({'hc':hcs},index=gids,dtype='category')
print(genes_hc_sub.shape[0])
info['n_gp_genes']=genes_hc_sub.shape[0]
# Make sizes of GPs for random creation
# Splits for np split
splits=[]
n_curr=0
# Make sure this is ordered as gps
for n in genes_hc_sub['hc'].value_counts(sort=False).values:
    splits.append(n_curr+n)
    n_curr+=n
# Do not use last one as automatically createcd
splits=splits[:-1]    

# %%
# Gene groups - GP and random
gene_groups_collection={}
# Add true GPs
gene_groups_collection['GPs']={hc:list(g) 
                               for hc,g in genes_hc_sub.groupby('hc').groups.items()}
# Add random gene groups of GP sizes
np.random.seed(0)
for i in range(10):
    # Random idx groups of GP sizes
    random_indices=np.split(np.random.permutation(list(range(adata.shape[1]))
                                             )[:genes_hc_sub.shape[0]],splits)
    # Map idx to genes
    gene_groups_collection['GPs_random'+str(i)]={
        genes_hc_sub.hc.cat.categories[gp_idx]:adata.var_names[idxs] 
        for gp_idx,idxs in enumerate(random_indices)}

# %%
# Explained var
explained_var=explaied_var_helper(
    gp_study=[(gp,False) for gp in gene_groups_collection.keys()],
    adata=adata,gene_groups_collection=gene_groups_collection,study_cov=None)
display(explained_var)

# %%
explained_var.to_csv(path_save+dataset+'.tsv',sep='\t',index=False)

# %%
info

# %%
dataset_info.append(info)

# %% [markdown]
# ### GSE101207

# %%
dataset='GSE101207'
info={}
info['dataset']=dataset
info['organism']='human'
info['technology']='Drop-seq'

# %%
ddir=dataset+'/GEO/'
adata=sc.read(path_rna+ddir+'adata_filtered.h5ad')
adata=adata[(adata.obs.cell_type=='beta').values ]
info['n_cells']=adata.shape[0]
adata.shape

# %%
# Remove lowly expressed genes
sc.pp.filter_genes(adata,min_cells=adata.shape[0]*0.01)
print(adata.shape)

# %%
# Compute HVG and PCA
sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes =2000)
sc.tl.pca(adata, n_comps=NPC, use_highly_variable=True)

# %%
adata.var.iloc[:1]

# %%
# Subset GPs to only genes in data
# Map to gene ids and orthologues in data before making GP sub
hcs=[]
gids=[]
for eid,hc in genes_hc['hc'].iteritems():
    for gid in [gid for gid in orthologues.query('eid_mm==@eid').gs_hs 
                if gid in adata.var_names]:
        hcs.append(hc)
        gids.append(gid)
genes_hc_sub=pd.DataFrame({'hc':hcs},index=gids,dtype='category')
print(genes_hc_sub.shape[0])
info['n_gp_genes']=genes_hc_sub.shape[0]
# Make sizes of GPs for random creation
# Splits for np split
splits=[]
n_curr=0
# Make sure this is ordered as gps
for n in genes_hc_sub['hc'].value_counts(sort=False).values:
    splits.append(n_curr+n)
    n_curr+=n
# Do not use last one as automatically createcd
splits=splits[:-1]    

# %%
# Gene groups - GP and random
gene_groups_collection={}
# Add true GPs
gene_groups_collection['GPs']={hc:list(g) 
                               for hc,g in genes_hc_sub.groupby('hc').groups.items()}
# Add random gene groups of GP sizes
np.random.seed(0)
for i in range(10):
    # Random idx groups of GP sizes
    random_indices=np.split(np.random.permutation(list(range(adata.shape[1]))
                                             )[:genes_hc_sub.shape[0]],splits)
    # Map idx to genes
    gene_groups_collection['GPs_random'+str(i)]={
        genes_hc_sub.hc.cat.categories[gp_idx]:adata.var_names[idxs] 
        for gp_idx,idxs in enumerate(random_indices)}

# %%
# Explained var
explained_var=explaied_var_helper(
    gp_study=[(gp,False) for gp in gene_groups_collection.keys()],
    adata=adata,gene_groups_collection=gene_groups_collection,study_cov=None)
display(explained_var)

# %%
explained_var.to_csv(path_save+dataset+'.tsv',sep='\t',index=False)

# %%
info

# %%
dataset_info.append(info)

# %% [markdown]
# ### GSE124742 and GSE164875 - patch 

# %%
dataset='GSE124742_GSE164875-patch'
info={}
info['dataset']=dataset
info['organism']='human'
info['technology']='SMART-seq2'

# %%
ddir='/GSE124742_GSE164875/GEO/patch/'
adata=sc.read(path_rna+ddir+'adata_filtered.h5ad')
adata=adata[(adata.obs.cell_type=='beta').values ]
info['n_cells']=adata.shape[0]
adata.shape

# %%
# Remove lowly expressed genes
sc.pp.filter_genes(adata,min_cells=adata.shape[0]*0.01)
print(adata.shape)

# %%
# Compute HVG and PCA
sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes =2000)
sc.tl.pca(adata, n_comps=NPC, use_highly_variable=True)

# %%
adata.var.iloc[:1]

# %%
# Subset GPs to only genes in data
# Map to gene ids and orthologues in data before making GP sub
hcs=[]
gids=[]
for eid,hc in genes_hc['hc'].iteritems():
    for gid in [gid for gid in orthologues.query('eid_mm==@eid').gs_hs 
                if gid in adata.var_names]:
        hcs.append(hc)
        gids.append(gid)
genes_hc_sub=pd.DataFrame({'hc':hcs},index=gids,dtype='category')
print(genes_hc_sub.shape[0])
info['n_gp_genes']=genes_hc_sub.shape[0]
# Make sizes of GPs for random creation
# Splits for np split
splits=[]
n_curr=0
# Make sure this is ordered as gps
for n in genes_hc_sub['hc'].value_counts(sort=False).values:
    splits.append(n_curr+n)
    n_curr+=n
# Do not use last one as automatically createcd
splits=splits[:-1]    

# %%
# Gene groups - GP and random
gene_groups_collection={}
# Add true GPs
gene_groups_collection['GPs']={hc:list(g) 
                               for hc,g in genes_hc_sub.groupby('hc').groups.items()}
# Add random gene groups of GP sizes
np.random.seed(0)
for i in range(10):
    # Random idx groups of GP sizes
    random_indices=np.split(np.random.permutation(list(range(adata.shape[1]))
                                             )[:genes_hc_sub.shape[0]],splits)
    # Map idx to genes
    gene_groups_collection['GPs_random'+str(i)]={
        genes_hc_sub.hc.cat.categories[gp_idx]:adata.var_names[idxs] 
        for gp_idx,idxs in enumerate(random_indices)}

# %%
# Explained var
explained_var=explaied_var_helper(
    gp_study=[(gp,False) for gp in gene_groups_collection.keys()],
    adata=adata,gene_groups_collection=gene_groups_collection,study_cov=None)
display(explained_var)

# %%
explained_var.to_csv(path_save+dataset+'.tsv',sep='\t',index=False)

# %%
info

# %%
dataset_info.append(info)

# %% [markdown]
# ### GSE124742 and GSE164875 - FACS 

# %%
dataset='GSE124742_GSE164875-facs'
info={}
info['dataset']=dataset
info['organism']='human'
info['technology']='SMART-seq2'

# %%
ddir='/GSE124742_GSE164875/GEO/FACS/'
adata=sc.read(path_rna+ddir+'adata_filtered.h5ad')
adata=adata[(adata.obs.cell_type=='beta').values ]
info['n_cells']=adata.shape[0]
adata.shape

# %%
# Remove lowly expressed genes
sc.pp.filter_genes(adata,min_cells=adata.shape[0]*0.01)
print(adata.shape)

# %%
# Compute HVG and PCA
sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes =2000)
sc.tl.pca(adata, n_comps=NPC, use_highly_variable=True)

# %%
adata.var.iloc[:1]

# %%
# Subset GPs to only genes in data
# Map to gene ids and orthologues in data before making GP sub
hcs=[]
gids=[]
for eid,hc in genes_hc['hc'].iteritems():
    for gid in [gid for gid in orthologues.query('eid_mm==@eid').gs_hs 
                if gid in adata.var_names]:
        hcs.append(hc)
        gids.append(gid)
genes_hc_sub=pd.DataFrame({'hc':hcs},index=gids,dtype='category')
print(genes_hc_sub.shape[0])
info['n_gp_genes']=genes_hc_sub.shape[0]
# Make sizes of GPs for random creation
# Splits for np split
splits=[]
n_curr=0
# Make sure this is ordered as gps
for n in genes_hc_sub['hc'].value_counts(sort=False).values:
    splits.append(n_curr+n)
    n_curr+=n
# Do not use last one as automatically createcd
splits=splits[:-1]    

# %%
# Gene groups - GP and random
gene_groups_collection={}
# Add true GPs
gene_groups_collection['GPs']={hc:list(g) 
                               for hc,g in genes_hc_sub.groupby('hc').groups.items()}
# Add random gene groups of GP sizes
np.random.seed(0)
for i in range(10):
    # Random idx groups of GP sizes
    random_indices=np.split(np.random.permutation(list(range(adata.shape[1]))
                                             )[:genes_hc_sub.shape[0]],splits)
    # Map idx to genes
    gene_groups_collection['GPs_random'+str(i)]={
        genes_hc_sub.hc.cat.categories[gp_idx]:adata.var_names[idxs] 
        for gp_idx,idxs in enumerate(random_indices)}

# %%
# Explained var
explained_var=explaied_var_helper(
    gp_study=[(gp,False) for gp in gene_groups_collection.keys()],
    adata=adata,gene_groups_collection=gene_groups_collection,study_cov=None)
display(explained_var)

# %%
explained_var.to_csv(path_save+dataset+'.tsv',sep='\t',index=False)

# %%
info

# %%
dataset_info.append(info)

# %% [markdown]
# ### GSE86469

# %%
dataset='GSE86469'
info={}
info['dataset']=dataset
info['organism']='human'
info['technology']='FluidigmC1_SMARTer'

# %%
ddir=dataset+'/GEO/'
adata=sc.read(path_rna+ddir+'adata_filtered.h5ad')
adata=adata[(adata.obs.cell_type=='beta').values ]
info['n_cells']=adata.shape[0]
adata.shape

# %%
adata.var_names=adata.var.EID

# %%
# Check that now var names are unique
adata.var_names.value_counts()

# %%
# Remove lowly expressed genes
sc.pp.filter_genes(adata,min_cells=adata.shape[0]*0.01)
print(adata.shape)

# %%
# Compute HVG and PCA
sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes =2000)
sc.tl.pca(adata, n_comps=NPC, use_highly_variable=True)

# %%
adata.var.iloc[:1]

# %%
# Subset GPs to only genes in data
# Map to gene ids and orthologues in data before making GP sub
hcs=[]
gids=[]
for eid,hc in genes_hc['hc'].iteritems():
    for gid in [gid for gid in orthologues.query('eid_mm==@eid').eid_hs 
                if gid in adata.var_names]:
        hcs.append(hc)
        gids.append(gid)
genes_hc_sub=pd.DataFrame({'hc':hcs},index=gids,dtype='category')
print(genes_hc_sub.shape[0])
info['n_gp_genes']=genes_hc_sub.shape[0]
# Make sizes of GPs for random creation
# Splits for np split
splits=[]
n_curr=0
# Make sure this is ordered as gps
for n in genes_hc_sub['hc'].value_counts(sort=False).values:
    splits.append(n_curr+n)
    n_curr+=n
# Do not use last one as automatically createcd
splits=splits[:-1]    

# %%
# Gene groups - GP and random
gene_groups_collection={}
# Add true GPs
gene_groups_collection['GPs']={hc:list(g) 
                               for hc,g in genes_hc_sub.groupby('hc').groups.items()}
# Add random gene groups of GP sizes
np.random.seed(0)
for i in range(10):
    # Random idx groups of GP sizes
    random_indices=np.split(np.random.permutation(list(range(adata.shape[1]))
                                             )[:genes_hc_sub.shape[0]],splits)
    # Map idx to genes
    gene_groups_collection['GPs_random'+str(i)]={
        genes_hc_sub.hc.cat.categories[gp_idx]:adata.var_names[idxs] 
        for gp_idx,idxs in enumerate(random_indices)}

# %%
# Explained var
explained_var=explaied_var_helper(
    gp_study=[(gp,False) for gp in gene_groups_collection.keys()],
    adata=adata,gene_groups_collection=gene_groups_collection,study_cov=None)
display(explained_var)

# %%
explained_var.to_csv(path_save+dataset+'.tsv',sep='\t',index=False)

# %%
info

# %%
dataset_info.append(info)

# %% [markdown]
# ### GSE81547

# %%
dataset='GSE81547'
info={}
info['dataset']=dataset
info['organism']='human'
info['technology']='SMART-seq2'

# %%
ddir=dataset+'/GEO/'
adata=sc.read(path_rna+ddir+'adata_filtered.h5ad')
adata=adata[(adata.obs.cell_type=='beta').values ]
info['n_cells']=adata.shape[0]
adata.shape

# %%
# Remove lowly expressed genes
sc.pp.filter_genes(adata,min_cells=adata.shape[0]*0.01)
print(adata.shape)

# %%
# Compute HVG and PCA
sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes =2000)
sc.tl.pca(adata, n_comps=NPC, use_highly_variable=True)

# %%
adata.var.iloc[:1]

# %%
# Subset GPs to only genes in data
# Map to gene ids and orthologues in data before making GP sub
hcs=[]
gids=[]
for eid,hc in genes_hc['hc'].iteritems():
    for gid in [gid for gid in orthologues.query('eid_mm==@eid').gs_hs 
                if gid in adata.var_names]:
        hcs.append(hc)
        gids.append(gid)
genes_hc_sub=pd.DataFrame({'hc':hcs},index=gids,dtype='category')
print(genes_hc_sub.shape[0])
info['n_gp_genes']=genes_hc_sub.shape[0]
# Make sizes of GPs for random creation
# Splits for np split
splits=[]
n_curr=0
# Make sure this is ordered as gps
for n in genes_hc_sub['hc'].value_counts(sort=False).values:
    splits.append(n_curr+n)
    n_curr+=n
# Do not use last one as automatically createcd
splits=splits[:-1]    

# %%
# Gene groups - GP and random
gene_groups_collection={}
# Add true GPs
gene_groups_collection['GPs']={hc:list(g) 
                               for hc,g in genes_hc_sub.groupby('hc').groups.items()}
# Add random gene groups of GP sizes
np.random.seed(0)
for i in range(10):
    # Random idx groups of GP sizes
    random_indices=np.split(np.random.permutation(list(range(adata.shape[1]))
                                             )[:genes_hc_sub.shape[0]],splits)
    # Map idx to genes
    gene_groups_collection['GPs_random'+str(i)]={
        genes_hc_sub.hc.cat.categories[gp_idx]:adata.var_names[idxs] 
        for gp_idx,idxs in enumerate(random_indices)}

# %%
# Explained var
explained_var=explaied_var_helper(
    gp_study=[(gp,False) for gp in gene_groups_collection.keys()],
    adata=adata,gene_groups_collection=gene_groups_collection,study_cov=None)
display(explained_var)

# %%
explained_var.to_csv(path_save+dataset+'.tsv',sep='\t',index=False)

# %%
info

# %%
dataset_info.append(info)

# %% [markdown]
# ### GSE81608

# %%
dataset='GSE81608'
info={}
info['dataset']=dataset
info['organism']='human'
info['technology']='FluidigmC1_SMARTer'

# %%
ddir=dataset+'/GEO/'
adata=sc.read(path_rna+ddir+'adata_filtered.h5ad')
adata=adata[(adata.obs.cell_type=='beta').values ]
info['n_cells']=adata.shape[0]
adata.shape

# %%
# Remove lowly expressed genes
sc.pp.filter_genes(adata,min_cells=adata.shape[0]*0.01)
print(adata.shape)

# %%
# Compute HVG and PCA
sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes =2000)
sc.tl.pca(adata, n_comps=NPC, use_highly_variable=True)

# %%
adata.var.iloc[:1]

# %%
# Subset GPs to only genes in data
# Map to gene ids and orthologues in data before making GP sub
hcs=[]
gids=[]
for eid,hc in genes_hc['hc'].iteritems():
    for gid in [gid for eid in orthologues.query('eid_mm==@eid').eid_hs 
                for gid in adata.var.query('EID==@eid').index ]:
        hcs.append(hc)
        gids.append(gid)
genes_hc_sub=pd.DataFrame({'hc':hcs},index=gids,dtype='category')
print(genes_hc_sub.shape[0])
info['n_gp_genes']=genes_hc_sub.shape[0]
# Make sizes of GPs for random creation
# Splits for np split
splits=[]
n_curr=0
# Make sure this is ordered as gps
for n in genes_hc_sub['hc'].value_counts(sort=False).values:
    splits.append(n_curr+n)
    n_curr+=n
# Do not use last one as automatically createcd
splits=splits[:-1]    

# %%
# Gene groups - GP and random
gene_groups_collection={}
# Add true GPs
gene_groups_collection['GPs']={hc:list(g) 
                               for hc,g in genes_hc_sub.groupby('hc').groups.items()}
# Add random gene groups of GP sizes
np.random.seed(0)
for i in range(10):
    # Random idx groups of GP sizes
    random_indices=np.split(np.random.permutation(list(range(adata.shape[1]))
                                             )[:genes_hc_sub.shape[0]],splits)
    # Map idx to genes
    gene_groups_collection['GPs_random'+str(i)]={
        genes_hc_sub.hc.cat.categories[gp_idx]:adata.var_names[idxs] 
        for gp_idx,idxs in enumerate(random_indices)}

# %%
# Explained var
explained_var=explaied_var_helper(
    gp_study=[(gp,False) for gp in gene_groups_collection.keys()],
    adata=adata,gene_groups_collection=gene_groups_collection,study_cov=None)
display(explained_var)

# %%
explained_var.to_csv(path_save+dataset+'.tsv',sep='\t',index=False)

# %%
info

# %%
dataset_info.append(info)

# %% [markdown]
#  ### GSE198623 (Sophie's data)

# %%
dataset='sophieHuman'
info={}
info['dataset']=dataset
info['organism']='human'
info['technology']='10X'

# %%
ddir='P21000'+'/sophie/human/'
adata=sc.read(path_rna+ddir+'adata_filtered.h5ad')
adata=adata[(adata.obs.cell_type=='beta').values ]
info['n_cells']=adata.shape[0]
adata.shape

# %%
# Remove lowly expressed genes
sc.pp.filter_genes(adata,min_cells=adata.shape[0]*0.01)
print(adata.shape)

# %%
# Compute HVG and PCA
sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes =2000)
sc.tl.pca(adata, n_comps=NPC, use_highly_variable=True)

# %%
adata.var.iloc[:3]

# %%
# Subset GPs to only genes in data
# Map to gene ids and orthologues in data before making GP sub
hcs=[]
gids=[]
for eid,hc in genes_hc['hc'].iteritems():
    for gid in [gid for eid in orthologues.query('eid_mm==@eid').eid_hs 
                for gid in adata.var.query('EID ==@eid').index ]:
        hcs.append(hc)
        gids.append(gid)
genes_hc_sub=pd.DataFrame({'hc':hcs},index=gids,dtype='category')
print(genes_hc_sub.shape[0])
info['n_gp_genes']=genes_hc_sub.shape[0]
# Make sizes of GPs for random creation
# Splits for np split
splits=[]
n_curr=0
# Make sure this is ordered as gps
for n in genes_hc_sub['hc'].value_counts(sort=False).values:
    splits.append(n_curr+n)
    n_curr+=n
# Do not use last one as automatically createcd
splits=splits[:-1]    

# %%
# Gene groups - GP and random
gene_groups_collection={}
# Add true GPs
gene_groups_collection['GPs']={hc:list(g) 
                               for hc,g in genes_hc_sub.groupby('hc').groups.items()}
# Add random gene groups of GP sizes
np.random.seed(0)
for i in range(10):
    # Random idx groups of GP sizes
    random_indices=np.split(np.random.permutation(list(range(adata.shape[1]))
                                             )[:genes_hc_sub.shape[0]],splits)
    # Map idx to genes
    gene_groups_collection['GPs_random'+str(i)]={
        genes_hc_sub.hc.cat.categories[gp_idx]:adata.var_names[idxs] 
        for gp_idx,idxs in enumerate(random_indices)}

# %%
# Explained var
explained_var=explaied_var_helper(
    gp_study=[(gp,False) for gp in gene_groups_collection.keys()],
    adata=adata,gene_groups_collection=gene_groups_collection,study_cov=None)
display(explained_var)

# %%
explained_var.to_csv(path_save+dataset+'.tsv',sep='\t',index=False)

# %%
info

# %%
dataset_info.append(info)

# %% [markdown]
#  ### GSE148073

# %%
dataset='GSE148073'
info={}
info['dataset']=dataset
info['organism']='human'
info['technology']='10X'

# %%
ddir=dataset+'/GEO/'
adata=sc.read(path_rna+ddir+'adata_filtered.h5ad')
adata=adata[(adata.obs.cell_type=='beta').values ]
info['n_cells']=adata.shape[0]
adata.shape

# %%
# Remove lowly expressed genes
sc.pp.filter_genes(adata,min_cells=adata.shape[0]*0.01)
print(adata.shape)

# %%
# Compute HVG and PCA
sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes =2000)
sc.tl.pca(adata, n_comps=NPC, use_highly_variable=True)

# %%
adata.var.iloc[:3]

# %%
# Subset GPs to only genes in data
# Map to gene ids and orthologues in data before making GP sub
hcs=[]
gids=[]
for eid,hc in genes_hc['hc'].iteritems():
    for gid in [gid for eid in orthologues.query('eid_mm==@eid').eid_hs 
                for gid in adata.var.query('EID ==@eid').index ]:
        hcs.append(hc)
        gids.append(gid)
genes_hc_sub=pd.DataFrame({'hc':hcs},index=gids,dtype='category')
print(genes_hc_sub.shape[0])
info['n_gp_genes']=genes_hc_sub.shape[0]
# Make sizes of GPs for random creation
# Splits for np split
splits=[]
n_curr=0
# Make sure this is ordered as gps
for n in genes_hc_sub['hc'].value_counts(sort=False).values:
    splits.append(n_curr+n)
    n_curr+=n
# Do not use last one as automatically createcd
splits=splits[:-1]    

# %%
# Gene groups - GP and random
gene_groups_collection={}
# Add true GPs
gene_groups_collection['GPs']={hc:list(g) 
                               for hc,g in genes_hc_sub.groupby('hc').groups.items()}
# Add random gene groups of GP sizes
np.random.seed(0)
for i in range(10):
    # Random idx groups of GP sizes
    random_indices=np.split(np.random.permutation(list(range(adata.shape[1]))
                                             )[:genes_hc_sub.shape[0]],splits)
    # Map idx to genes
    gene_groups_collection['GPs_random'+str(i)]={
        genes_hc_sub.hc.cat.categories[gp_idx]:adata.var_names[idxs] 
        for gp_idx,idxs in enumerate(random_indices)}

# %%
# Explained var
explained_var=explaied_var_helper(
    gp_study=[(gp,False) for gp in gene_groups_collection.keys()],
    adata=adata,gene_groups_collection=gene_groups_collection,study_cov=None)
display(explained_var)

# %%
explained_var.to_csv(path_save+dataset+'.tsv',sep='\t',index=False)

# %%
info

# %%
dataset_info.append(info)

# %% [markdown]
# ## Analyse explained var across datasets

# %%
# Make DF from dict of dataset info
dataset_info=pd.DataFrame(dataset_info)
dataset_info.index=dataset_info.dataset

# %%
# Save dataset info
dataset_info.to_csv(path_save+'dataset_info.tsv',sep='\t')

# %% [markdown]
# ### Explained var by GP in atlas

# %%
explained_var=pd.read_table(path_save+'mouseAtlas.tsv'
     ).query('study == False & GPs.str.contains("component")',engine='python')
explained_var['component']=explained_var['GPs'].apply(lambda x:x.split('GPcomponent')[1])
explained_var['random']=explained_var['GPs'].str.contains('random')

# %%
rcParams['figure.figsize']=(17,3)
sb.swarmplot(x='component',y='explained_var_ratio',hue='random',data=explained_var)
plt.legend(bbox_to_anchor=(1.12,1))

# %% [markdown]
# C: Some GPs explain overall less var, but for 26&27 probably due to less cells there - see that separate a lot by study. But 5 is probably more random.

# %% [markdown]
# ### Explained var in each dataset

# %%
# Load and summarise data of explained var and also parse all results into shared table
explained_var_all=[]
for dataset in dataset_info.index:
    explained_var=pd.read_table(path_save+dataset+'.tsv')
    evar=explained_var.query('GPs=="GPs" & study==False')['explained_var_ratio'].values
    if evar.shape[0]!=1:
        raise ValueError('More than 1 explained var value')
    dataset_info.at[dataset,'explained_var']=evar[0]
    random_var= explained_var.fillna('NA').query(
        'GPs.str.startswith("GPs_random") & ~GPs.str.contains("component") & study==False',
        engine='python'
        )['explained_var_ratio']
    dataset_info.at[dataset,'explained_var_radom_m']= random_var.mean()
    dataset_info.at[dataset,'explained_var_radom_std']= random_var.std()
    dataset_info['explained_var_n_std']=\
        ((dataset_info['explained_var']-dataset_info['explained_var_radom_m'])/
        dataset_info['explained_var_radom_std'])
    explained_var['component']=explained_var['GPs'].fillna('NA').apply(
        lambda x: np.nan if x =='NA' else 
        ('GPs' if 'component' not in x else x.split('component')[1]))
    explained_var['random']=explained_var['GPs'].str.contains('random')
    explained_var['dataset']=dataset
    explained_var['organism']=dataset_info.at[dataset,'organism']
    explained_var_all.append(explained_var)
explained_var_all=pd.concat(explained_var_all)    

# %%
dataset_info

# %% [markdown]
# #### Effect of data characteristic on explained var
# How much var is explained with all GPs across studies and how covariates (species, n cells in dataset, n GP genes present in data) affect the overall explained var.

# %%
# Plot explained var vs covariates
for y in ['explained_var','explained_var_n_std']:
    for x in ['explained_var_n_std','n_cells','n_gp_genes']:
        if x!=y:
            fig,ax=plt.subplots(figsize=(3,2.5))
            sb.scatterplot(x=x,y=y,hue='organism',data=dataset_info,s=40)
            plt.legend(bbox_to_anchor=(1.05, 1))
            if x=='n_cells':
                plt.xscale('log')


# %%
fig,ax=plt.subplots(figsize=(3,2.5))
sb.scatterplot(x='n_gp_genes',y='n_cells',hue='organism',data=dataset_info,s=40)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.yscale('log')

# %% [markdown]
# #### Explained var significance
# Per-dataset comparison of var explained by GPs and random gene groups

# %%
species_palette={'human':'#8a9e59','mouse':'#c97fac'}

# %%
# For renaming datasets
rename_ds={
    'GSE124742_GSE164875-patch':'GSE124742_GSE164875_patch',
    'GSE124742_GSE164875-facs':'GSE124742_FACS',
    'sophieHuman':'GSE198623'
}
# rename datasets
explained_var_all.dataset=explained_var_all.dataset.replace(rename_ds)

# %%
# Plot explained var vs random explained var for each dataset
rcParams['figure.figsize']=(8,3)
sb.swarmplot(x='dataset',y='explained_var_ratio',hue='organism',marker='X',s=10,
            data=explained_var_all.query('component=="GPs" & study==False & random==False'),
            palette=species_palette)
g=sb.swarmplot(x='dataset',y='explained_var_ratio',hue='organism',
            data=explained_var_all.query('component=="GPs" & study==False & random==True'),
            palette=species_palette)
g.get_legend().remove()
g.set_ylim((0,1))
a=plt.xticks(rotation=90)

handles,labels=g.get_legend_handles_labels()
legend_elements=dict()
legend_elements['species']=Line2D([0], [0], markersize=0,lw=0)
legend_elements.update(zip(labels,handles))
legend_elements['\ngene sets']=Line2D([0], [0], markersize=0,lw=0)                                       
legend_elements['GPs']=Line2D([0], [0], marker='X', color='k', 
                          markerfacecolor='k', markersize=8,lw=0)
legend_elements['random']=Line2D([0], [0], marker='o', color='k', 
                          markerfacecolor='k', markersize=8,lw=0)
a=g.legend(handles=legend_elements.values(),labels=legend_elements.keys(), 
          bbox_to_anchor=(1.01,1.03))
# Transparency
g.set(facecolor = (0,0,0,0))
g.spines['top'].set_visible(False)
g.spines['right'].set_visible(False)
a=g.set_ylabel('explained var ratio')
g.get_legend().get_frame().set_alpha(None)
g.get_legend().get_frame().set_facecolor((0, 0, 0, 0))
g.get_legend().get_frame().set_edgecolor((0, 0, 0, 0))
plt.savefig(path_fig+'swarmplot_beta_GPexplainedVar_datasets.png',dpi=300,bbox_inches='tight')

# %% [markdown]
# #### Explained var in each dataset by individual GPs

# %% [markdown]
# Explained var in each study by individual GPs. How much var is explained and how many stds is this from mean on random gene groups.

# %%
# Get explained var for each component in each dataset and compare to random var
explained_component=[]
for (dataset, component),data in explained_var_all.query(
    'study==False & component!="GPs"').groupby(['dataset','component']):
    explained=data[~data.random.astype(bool)]['explained_var_ratio'].values[0]
    explained_random=data[data.random.astype(bool)]['explained_var_ratio']
    explained_n_std=(explained-explained_random.mean())/explained_random.std()
    organism=data.organism.values[0]
    organism_short='mm' if organism =='mouse' else 'hs'
    explained_component.append(
        {'dataset':dataset,'component':component,'organism':organism,
         'org_dataset':organism_short+'_'+dataset,
        'explained_var':explained,'explained_var_n_std':explained_n_std})
explained_component=pd.DataFrame(explained_component)  
# Sort components for plotting
explained_component['component']=pd.Categorical(
    explained_component['component'],
    [str(c) for c in sorted(explained_component['component'].astype(int).unique())],
    True)
explained_component['org_dataset']=pd.Categorical(
    explained_component['org_dataset'],sorted(explained_component['org_dataset'].unique()),True)

# %%
# Explained var by component
rcParams['figure.figsize']=(explained_component.dataset.nunique()*0.32,
                            explained_component.component.nunique()*0.32)
sb.scatterplot(x='org_dataset',y='component',hue='explained_var',size='explained_var_n_std',
               data=explained_component,palette='magma')
plt.legend(bbox_to_anchor=(1.05, 1))
a=plt.xticks(rotation=90)

# %% [markdown]
# Contribution of GP components for each dataset, scaled (maxabs) across GPs

# %%
# Scale explained components
scaled=pd.crosstab(columns=explained_component['dataset'],index=explained_component['component'],
            values=explained_component['explained_var'],aggfunc='mean')
scaled=pd.DataFrame(maxabs_scale(scaled),index=scaled.index,columns=scaled.columns)

# %%
# rename datasets
scaled.rename(rename_ds,axis=1,inplace=True)
dataset_info.rename(rename_ds,axis=0,inplace=True)

# %%
# Add T1D and T2D info
dataset_info['diabetes']=None
for ds in dataset_info.index:
    if ds!='mouseAtlas':
        disease=set(pd.read_excel(path_rna+'external_metadata.xlsx',sheet_name=ds)['disease'])
        if 'T1D' in disease and 'T2D' in disease:
            diabetes='T1D and T2D'
        elif 'T1D' in disease :
            diabetes='T1D'
        elif 'T2D' in disease :
            diabetes='T2D'
        else:
            diabetes='none'
    else:
        diabetes='T1D and T2D'
    dataset_info.at[ds,'diabetes']=diabetes

# %%
# dataset anno
diabetes_palette={'T1D':'#E04C4C','T2D':'#1E88E5','T1D and T2D':'#E2BF56','none':'#D4D4D4'}
col_colors=pd.concat([
    # Diabetes anno
    pd.Series(scaled.columns,index=scaled.columns,name='diabetes'
         ).map(dataset_info['diabetes'].to_dict()
         ).map(diabetes_palette),
    # Species annotation
    pd.Series(scaled.columns,index=scaled.columns,name='species'
         ).map(dataset_info['organism'].to_dict()
         ).map(species_palette)
],axis=1)

# %%
# Plot clustermap of explained var per GP normalised per dataset
# Sizes
w_dend=1.5
w_colors=0.4
nrow=scaled.shape[0]*0.4
ncol=scaled.shape[1]*0.4
w=ncol+w_dend
h=nrow+w_colors*2+w_dend
# Heatmap
g=sb.clustermap(scaled.T,xticklabels=True,yticklabels=True,
              row_colors=col_colors,
              figsize=(h,w),
             colors_ratio=(w_colors/h,w_colors/w),
            dendrogram_ratio=(w_dend/h,w_dend/w),vmin=0,vmax=1,
                cbar_pos=(-0.01,0.43,0.02,0.2),cmap='viridis')
g.ax_cbar.set_title('relative\nexplained var\n')   

#remove dendrogram
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)

# Legend for species and diabetes
l=plt.legend(handles=\
             [mpatches.Patch(alpha=0, label='diabetes')]+
             [mpatches.Patch(color=c, label=l) 
                      for l,c in diabetes_palette.items()]+
             [mpatches.Patch(alpha=0, label='\nspecies')]+
             [mpatches.Patch(color=c, label=l) 
                      for l,c in species_palette.items()],
          bbox_to_anchor=(4.8,-0.3))
l.get_frame().set_alpha(0)

# Remove row anno tick
g.ax_row_colors.xaxis.set_ticks_position('none') 

plt.savefig(path_fig+'heatmap_beta_GPexplainedVar_datasets_perGPnorm.png',
            dpi=300,bbox_inches='tight')

# %% [markdown]
# C: In most datasets the var is best explained by GPs differing between healthy and T2D
