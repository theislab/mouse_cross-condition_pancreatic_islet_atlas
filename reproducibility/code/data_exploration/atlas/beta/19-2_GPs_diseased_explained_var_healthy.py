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
import pickle as pkl
import glob

from sklearn.preprocessing import minmax_scale,maxabs_scale

from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import pdist,squareform

import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/')
from importlib import reload  
import helper as h
reload(h)
import helper as h

# %%
path_gp='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/moransi/sfintegrated/'
path_rna='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/'
path_data=path_rna+'combined/'
path_save=path_gp+'explained_var/per_sample/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'

# %%
# Saving figures
path_fig='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/figures/paper/'


# %%
# Info on parsed study names
beta_obs=sc.read(path_data+'data_rawnorm_integrated_analysed_beta_v1s1_sfintegrated.h5ad',
                backed='r').obs.copy()
beta_obs['study_parsed_design_sample']=beta_obs.apply(
lambda x:x['study_parsed']+'_'+x['design']+'_'+x['file'],axis=1)

# %%
orthologues=pd.read_table(path_genes+'orthologues_ORGmus_musculus_ORG2homo_sapiens_V103.tsv'
                         ).rename(
    {'Gene name':'gs_mm','Human gene name':'gs_hs',
     'Gene stable ID':'eid_mm','Human gene stable ID':'eid_hs'},axis=1)

# %%
# Load GPs
genes_hc=pd.read_table(path_gp+'gene_hc_t'+str(2.4)+'.tsv',sep='\t',index_col=0)

# %% [markdown]
# GPs that show a large difference between healthy and diseased beta cell clusters, related to health/disease status

# %%
# Select GO subset to analyse
gps=[3,4,19,20]
genes_hc=genes_hc.query('hc in @gps')
genes_hc['hc']=genes_hc['hc'].astype('category')
print(genes_hc.shape[0],genes_hc.hc.nunique())

# %% [markdown]
# ## Compute explained var
# Calculate explained var of each selected GP in helathy adult samples from mice/human with enough beta cells. Also compute correlation between GP scores in each sample.

# %%
NPC=50


# %%
def explained_pca_var(X_pca,covariates,pca_var,n_comps):
    # Adapted from scIB
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
def explaied_var_helper(gp_study:list,adata,gene_groups_collection:dict,study_cov:np.array,
                       return_scores_gp):
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
    :param return_scores_gp: gp name of the gp collection to return or None to not return any.
    Returns as an optional second component in return
    """
    # Score for GPs (true or random), study, and true GP+study
    # In each GP set compute explained var over all GPs
    explained_var=[]
    scores_return=None
    for gp,study in gp_study:
        #print('GP and study:',gp,study)
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
        #print('cov shape:',cov.shape)
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
        if return_scores_gp is not None and gp==return_scores_gp:
            scores_return=scores_gp
        del cov
        del scores_gp
    explained_var=pd.DataFrame(explained_var)
    if scores_return is None:
        return explained_var
    else:
        return explained_var,scores_return


# %%
# Exclude samples with less than this N of beta cells
min_cells=100
sample_meta=[]
# Datasets list
# For external datasets select helathy samples later 
for data_info in [
    # Human external
     {'adata_name': 'adata_filtered.h5ad',
      'dataset': 'GSE83139',
      'ddir': 'GSE83139/GEO/',
      'species': 'hs'},
     {'adata_name': 'adata_filtered.h5ad',
      'dataset': 'GSE154126',
      'ddir': 'GSE154126/GEO/',
      'species': 'hs'},
     {'adata_name': 'adata_filtered.h5ad',
      'dataset': 'GSE101207',
      'ddir': 'GSE101207/GEO/',
      'species': 'hs'},
     {'adata_name': 'adata_filtered.h5ad',
      'dataset': 'GSE124742_GSE164875_patch',
      'ddir': 'GSE124742_GSE164875/GEO/patch/',
      'species': 'hs'},
     {'adata_name': 'adata_filtered.h5ad',
      'dataset': 'GSE124742_GSE164875_FACS',
      'ddir': 'GSE124742_GSE164875/GEO/FACS/',
      'species': 'hs'},
     {'adata_name': 'adata_filtered.h5ad',
      'dataset': 'GSE86469',
      'ddir': 'GSE86469/GEO/',
      'species': 'hs'},
     {'adata_name': 'adata_filtered.h5ad',
      'dataset': 'GSE81547',
      'ddir': 'GSE81547/GEO/',
      'species': 'hs'},
     {'adata_name': 'adata_filtered.h5ad',
      'dataset': 'GSE198623',
      'ddir': 'P21000/sophie/human/',
      'species': 'hs'},
     {'adata_name': 'adata_filtered.h5ad',
      'dataset': 'GSE81608',
      'ddir': 'GSE81608/GEO/',
      'species': 'hs'},
     {'adata_name': 'adata_filtered.h5ad',
      'dataset': 'GSE148073',
      'ddir': 'GSE148073/GEO/',
      'species': 'hs'},
    
    # Mouse external
    {'adata_name': 'adata.h5ad',
      'dataset': 'GSE137909',
      'ddir': 'GSE137909/GEO/',
      'species': 'mm'},
    # Dont use this dataset as low quality 
    #{'adata_name': 'adata.h5ad',
    #  'dataset': 'GSE83146',
    #  'ddir': 'GSE83146/GEO/',
    #  'species': 'mm'},
    
    # Atlas
    # manually specifiy healthy samples
    {'adata_name': 'data_rawnorm_integrated_analysed_beta_v1s1_sfintegrated.h5ad',
      'dataset': 'mouseAtlas',
      'ddir': 'combined/',
      'species': 'mm',
      'samples':[
          'STZ_G1_control',
          'VSG_MUC13633_chow_WT','VSG_MUC13634_chow_WT',
          'Fltp_adult_mouse1_head_Fltp-','Fltp_adult_mouse2_head_Fltp+',
          'Fltp_adult_mouse4_tail_Fltp+','Fltp_adult_mouse3_tail_Fltp-',
          'NOD_elimination_SRR7610295_8w', 'NOD_elimination_SRR7610296_8w',
          'NOD_elimination_SRR7610297_8w',
          'spikein_drug_SRR10751504_DMSO_r1','spikein_drug_SRR10751509_DMSO_r2',
          'spikein_drug_SRR10751514_DMSO_r3'],
      'sample_col':'study_sample_design',
      'ct_col':'cell_type_integrated_v1'} 
]:
    print('*** Dataset',data_info['dataset'])
    
    # Load dataset
    adata=sc.read(path_rna+data_info['ddir']+data_info['adata_name'],)
    
    # Add default ct col info if absent
    if 'ct_col' not in data_info:
        data_info['ct_col']='cell_type'
    
    # Select samples if not pregiven
    # Use helathy adults (based on devel stage ontologies)
    if 'samples' not in data_info:
        data_info['samples']=[]
        # Add default sample col info if absent
        if 'sample_col' not in data_info:
            data_info['sample_col']='donor'
        for sample, data in adata.obs.groupby(data_info['sample_col']):
            keep=True
            if data['disease'][0]!='healthy':
                keep=False
            if data_info['species']=='mm':
                age=h.age_weeks(data['age'][0])
                if age<6 or age>60:
                    keep=False
            elif data_info['species']=='hs':
                age=h.age_years(data['age'][0])
                if age<19 or age>64:
                    keep=False
            if keep:
                data_info['samples'].append(sample)
                
    # Explained var in each sample
    for sample in data_info['samples']:
        # Check if enough beta cells per sample are present
        cells_sub=adata.obs_names[(adata.obs[data_info['sample_col']]==sample).values &
                                  (adata.obs[data_info['ct_col']]=='beta').values]
        if cells_sub.shape[0]>=min_cells:
            print('\n',sample)
            # Subset adata
            adata_sub=adata[cells_sub,:].copy()
            
            # Remove lowly expressed genes
            sc.pp.filter_genes(adata_sub,min_cells=adata_sub.shape[0]*0.05)
            print(adata_sub.shape)
            
            # Use EIDs if availiable   
            if 'EID' in adata_sub.var.columns:
                adata_sub.var_names=adata_sub.var.EID
             # Subset GPs to only genes in data
            if data_info['species']=='hs':
                hcs=[]
                gids=[]   
                # For human use either EID or gs to map to orthologues
                if not pd.Series(adata_sub.var_names).apply(lambda x: x.startswith('ENS')).all():
                    gene_col='gs_hs'
                else:
                    gene_col='eid_hs'
                for eid,hc in genes_hc['hc'].iteritems():
                    for gid in [gs for gs in orthologues.query('eid_mm==@eid')[gene_col]
                            if gs in adata_sub.var_names ]:
                        hcs.append(hc)
                        gids.append(gid)
                genes_hc_sub=pd.DataFrame({'hc':hcs},index=gids,dtype='category')
            # For mouse assumes taht EIDs now in var names
            elif data_info['species']=='mm':
                genes_hc_sub=pd.DataFrame(genes_hc.loc[[e for e  in genes_hc.index 
                                          if e in adata_sub.var_names],'hc'].astype('category'))
            print('N all GP genes:',genes_hc_sub.shape[0])
            
            # Make sizes of GPs for random creation
            # Splits for np split
            splits=[]
            n_curr=0
            # Make sure this is ordered as gps
            for hc,n in genes_hc_sub['hc'].value_counts(sort=False).iteritems():
                splits.append(n_curr+n)
                n_curr+=n
                print('N genes in GP',hc,':',n)
            # Do not use last one as automatically createcd
            splits=splits[:-1]    
            # Gene groups - GP and random
            gene_groups_collection={}
            # Add true GPs
            gene_groups_collection['GPs']={hc:list(g) 
                                           for hc,g in genes_hc_sub.groupby('hc').groups.items()}
            # Add random gene groups of GP sizes
            np.random.seed(0)
            for i in range(100):
                # Random idx groups of GP sizes
                random_indices=np.split(np.random.permutation(list(range(adata_sub.shape[1]))
                                                         )[:genes_hc_sub.shape[0]],splits)
                # Map idx to genes
                gene_groups_collection['GPs_random'+str(i)]={
                    genes_hc_sub.hc.cat.categories[gp_idx]:adata_sub.var_names[idxs] 
                    for gp_idx,idxs in enumerate(random_indices)}

            # Compute HVG and PCA
            sc.pp.highly_variable_genes(adata_sub, flavor='cell_ranger', n_top_genes =2000)
            sc.tl.pca(adata_sub, n_comps=50, use_highly_variable=True)
    
            # Explained var
            explained_var,scores_gp=explaied_var_helper(
                gp_study=[(gp,False) for gp in gene_groups_collection.keys()],
                adata=adata_sub,gene_groups_collection=gene_groups_collection,study_cov=None,
                return_scores_gp='GPs')
            display(explained_var)
            # Save explained var
            explained_var.to_csv(path_save+data_info['dataset']+'_'+sample+'_explainedVar.tsv',
                                 sep='\t',index=False)
            
            # Correlation between GPs
            # Done on cells - hopefully combats sparsity as using multiple genes, 
            # pb clustering may be problematic as little cells in some samples
            scores_gp=pd.DataFrame(scores_gp)
            dist=pdist(scores_gp.T,metric='correlation')
            # Convert to similarity from correlation metric and to square form
            sim=1-dist
            sim=pd.DataFrame(squareform(sim),index=scores_gp.columns,columns=scores_gp.columns)
            # Save correlations between GPs
            sim.to_csv(path_save+data_info['dataset']+'_'+sample+'_GPcorr.tsv',sep='\t',
                       index=True)
            
            # Save metadata of sample
            sample_meta.append({'sample':sample,
                                'dataset':data_info['dataset'],
                                'species':data_info['species'],
                                'n_cells':adata_sub.shape[0]
                               })
# Save sample metadata
pd.DataFrame(sample_meta).to_csv(path_save+'sample_metadata.tsv',sep='\t',index=False)

# %% [markdown]
# ## Analyse explained var

# %%
# Load explained var per sample
files=glob.glob(path_save+'*explainedVar.tsv')
explained_vars=[]
for f in sorted(files):
    explained_var=pd.read_table(f)
    explained_var['sample']=f.split('/')[-1].replace('_explainedVar.tsv','')
    explained_vars.append(explained_var)
explained_vars=pd.concat(explained_vars)

# %%
# Add info on GP components and random/GP
explained_vars['component']=explained_vars['GPs'].apply(
    lambda x: x.split('_GPcomponent')[1] if '_GPcomponent' in x else 'GPs')
explained_vars['random']=explained_vars['GPs'].apply(lambda x: 'GPs_random' in x)

# %% [markdown]
# ### Significance of explained var

# %% [markdown]
# Plot explained vs random var for each GP and datset

# %%
# Explained var of each GP across samples
# Plot random score boxplots
g=sb.catplot(y='sample',x='explained_var_ratio',
             data=explained_vars.query('random'),col='component',kind='box',
           height=15, aspect=.3,color='black')
# Plot real GP score as red dot
for ax in g.axes[0]:
    gp=ax.get_title().split('= ')[1]
    sb.scatterplot(y='sample',x='explained_var_ratio',ax=ax,
             data=explained_vars.query('random==False & component==@gp'),color='red')

# %% [markdown]
# Significance of explained var compared to random for each GP and dataset

# %%
sample_meta=pd.read_table(path_save+'sample_metadata.tsv')

# %%
# Add dataset_sample info to match to explained var data
sample_meta.index=sample_meta.apply(
    lambda x:x['dataset']+'_'+x['sample'],axis=1)

# %%
# Empirical pval 
signif=explained_vars.groupby(['sample','component']).apply(
    lambda x: (x[x.random].explained_var_ratio>=
               x[~x.random].explained_var_ratio.values[0]).sum()/x[x.random].shape[0]
).rename('pval').reset_index()
# Make readatble table
signif_summary=pd.crosstab(signif['sample'],signif.component,signif.pval,aggfunc=max)
signif_summary.columns=['GP'+str(c) if 'GPs' not in c else c for c in signif_summary.columns ]
signif_summary.index=[i.rstrip('_') for i in signif_summary.index]
signif_summary[['species','n_cells','dataset','sample']]=\
    sample_meta.loc[signif_summary.index,['species','n_cells','dataset','sample']]
signif_summary.reset_index(inplace=True)
signif_summary.drop('index',axis=1,inplace=True)
# Use parse atlas study
signif_summary.columns.name=None
signif_summary=signif_summary.sort_values(['species','dataset','sample'],
                                         ascending=[False,True,True])
display(signif_summary)
signif_summary.to_csv(path_save+'explainedVar_GPsample_significanceSummary.tsv',
                      index=False,sep='\t')
# Save another version with parsed study for paper
signif_summary['sample']=signif_summary['sample'].replace(
    dict(zip(beta_obs['study_sample_design'],beta_obs['study_parsed_design_sample'])))
signif_summary.drop('GPs',axis=1,inplace=True)
signif_summary.to_csv(path_save+'explainedVar_GPsample_significanceSummary_parsed.tsv',
                      index=False,sep='\t')

# %% [markdown]
# ### Correlation between GPs per dataset
# Are the two healthy and the two diseased GPs correlated in individual samples and the healthy-diseased GPs anticorelated.

# %%
# Loadd GP correlations from each sample
files=glob.glob(path_save+'*GPcorr.tsv')
corrs=[]
for f in sorted(files):
    corr=pd.read_table(f,index_col=0)
    corr_parsed={}
    for i in range(corr.shape[1]-1):
        for j in range(i+1,corr.shape[1]):
            c=corr.iat[i,j]
            corr_parsed[corr.columns[i]+'_'+corr.columns[j]]=c
    corr_parsed=pd.Series(corr_parsed)
    corr_parsed.name=f.split('/')[-1].replace('_GPcorr.tsv','')
    corrs.append(corr_parsed)        
corrs=pd.concat(corrs,axis=1)

# %%
# GP correlations plot across samples
rcParams['figure.figsize']=(5,10)
sb.heatmap(corrs.T,vmin=-1,vmax=1,cmap='coolwarm')

# %% [markdown]
# C: Correlation is mostly as expected in mouse (except for 4 and 19), but not in human. Could be because some GPs explaining low variance in some of these samples anyways - not very explanatory, so not expected to show expected covariance structure.

# %%
