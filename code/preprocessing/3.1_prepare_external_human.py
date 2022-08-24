# -*- coding: utf-8 -*-
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
# # Prepare preannotated data from GEO

# %%
import scanpy as sc
import numpy as np
import pandas as pd
import glob
from scipy.sparse import csr_matrix
import gzip
from tempfile import TemporaryDirectory
import shutil
from scipy import sparse

import matplotlib.pyplot as plt

import mygene

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()
# # %load_ext rpy2.ipython

import rpy2.rinterface_lib.callbacks
import logging
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)

import sys  
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/')
import helper as h
from importlib import reload
reload(h)
import helper as h
from constants import SAVE


# %%
ro.r('library("scran")')
ro.r('library("BiocParallel")')

# %%
data_path='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'

# %% [markdown]
# ## GSE83139 - SMART-seq

# %%
dataset='GSE83139'
path_ds=data_path+dataset+'/GEO/'

# %% [markdown]
# Parse expression data

# %%
x=pd.read_table(path_ds+'GSE83139_tbx-v-f-norm-ntv-cpms.csv')

# %%
x

# %%
# Are gene names unique or are there multiple transcripts per gene?
x.gene.value_counts()

# %%
# Subset to expression only, drop other gene info
x.index=x.gene
x=x[[c for c in x.columns if 'reads.' in c]]
print(x.shape)

# %% [markdown]
# Load metadata

# %%
# metadata
# For some reason there are 2 obs tables on GEO that each contain part of the data
obs1=pd.read_table(path_ds+'GSE83139-GPL16791_series_matrix.txt',
                  skiprows=36,index_col=0)
obs2=pd.read_table(path_ds+'GSE83139-GPL11154_series_matrix.txt',
                  skiprows=36,index_col=0)
obs=pd.concat([obs1,obs2],axis=1)
print(obs.shape)

# %%
obs

# %%
# Subset
obs=obs.loc[['!Sample_characteristics_ch1','!Sample_geo_accession',
         '!Sample_source_name_ch1','!Sample_organism_ch1'],:]


# %%
# Edit rownmaes
obs.index=['tissue','age_group','disease','cell_type','geo_accession','organ','organism']
obs.drop('age_group',inplace=True)

# %%
# Edit values
for row in ['tissue','disease','cell_type']:
    obs.loc[row,:]=obs.loc[row].apply(lambda x: x.split(': ')[1])
obs.loc['organism']='human'

# %%
# Add donor info
obs.loc['donor',:]=[c.split()[0] for c in obs.columns]
donor_df=pd.read_table(path_ds+'GSE83139-meta-Table1.csv',sep=',',index_col=0)
for col in donor_df.columns:
    donor_df[col]=donor_df[col].apply(lambda x:x.rstrip())

# %%
# Parse donor info
donor_df['Age']=donor_df['Age'].apply(lambda x:x.split()[0]+' '+x.split()[1][0])
donor_df['Sex']=donor_df['Sex'].str.lower()
donor_df['Ethnicity']=donor_df['Ethnicity'].str.lower()
donor_df['Ethnicity']=donor_df['Ethnicity'].replace(
    {'na':'NA','african american':'african_american'})
donor_df['State']=donor_df['State'].str.rstrip().map(
    {'Control':'healthy','Type 1 diabetes':'T1D','Type 2 diabetes':'T2D','Child':'healthy'})
donor_df.rename({"BMI (kg/m2)":'BMI','Cultured (days)':'cultured_days','State':'disease',
                'Age':'age','Sex':'sex','Ethnicity':'ethnicity'},
                axis=1,inplace=True)
donor_df.index=[r.rstrip() for r in donor_df.index]
donor_df['cultured_days'].astype(float)
donor_df['cultured_days']=donor_df['cultured_days'].astype(float)
donor_df['BMI']=donor_df['BMI'].astype(float)

# %%
donor_df

# %%
# Check if health matches in donor and cell dfs
obs.T.groupby('donor')['disease'].unique()

# %%
# Rename some donors in obs
obs.loc['donor',:]=obs.loc['donor',:].replace(
    {'HP-15085-01T2D::8dcult':'HP-15085: cultured','HP-15085-01T2D::fresh':'HP-15085'})

# %%
# Add donor info to obs
for col in donor_df.columns:
    obs.loc[col,:]=obs.loc['donor',:].map(donor_df[col].to_dict())

# %%
# rename obs columns to match x
# Check that removing donor information from cols still produces unique cols
print('Cols unique across donors:',len(set([c.split()[1] for c in  obs.columns]))==obs.shape[1])
obs.columns=['reads.'+c.split()[1] for c in obs.columns]

# %%
# Cell types
obs.loc['cell_type',:].unique().tolist()

# %%
# Rename cell types
obs.loc['cell_type_original',:]=obs.loc['cell_type',:]
obs.loc['cell_type',:]=obs.loc['cell_type',:].replace(
    {'duct':'ductal','pp':'gamma'})
obs.loc['cell_type',:].unique().tolist()

# %%
obs.loc['cell_type',:].value_counts(dropna=False)

# %%
obs

# %%
obs.T.drop_duplicates('donor')

# %% [markdown]
# Make adata

# %%
adata=sc.AnnData(X=csr_matrix(x.T),obs=obs.T.reindex(x.T.index),
                 var=pd.DataFrame(index=x.T.columns))

# %%
adata

# %%
adata.obs.cell_type.isna().sum()

# %%
# Save orginal X
adata.layers['normalised_original']=adata.X.copy()

# %%
# Log normalise
sc.pp.log1p(adata)

# %%
adata

# %%
for col in adata.obs.columns:
    print(col)
    print(adata.obs[col].unique())

# %%
if SAVE:
    adata.write(path_ds+'adata.h5ad')

# %%
# Subset to annotated cells
adata_sub=adata[~adata.obs.cell_type.isin(['dropped','unknown']),:]
print(adata_sub.shape)
if SAVE:
    adata_sub.write(path_ds+'adata_filtered.h5ad')

# %% [markdown]
# ## GSE154126 - SMART-seq 

# %%
dataset='GSE154126'
path_ds=data_path+dataset+'/GEO/'

# %% [markdown]
# Parse expression data

# %%
x_norm=pd.read_table(path_ds+'GSE154126_tbx-v-m-b-norm-ntv-cpms.aug.tsv',index_col=0,skiprows=6)
x=pd.read_table(path_ds+'GSE154126_tbx-v-m-b-norm-ntv-reads.aug.tsv',index_col=0,skiprows=6)

# %%
x_norm

# %%
x

# %%
x.columns=[c.replace('cell.','') for c in x.columns]

# %%
print('Some gene names have no name:',x.index.isna().sum())

# %%
# Subset with genes with names
x=x[~x.index.isna()]
x_norm=x_norm[~x_norm.index.isna()]

# %% [markdown]
# Obs

# %%
# Some obs data also in expression table
obs_x1=pd.read_table(path_ds+'GSE154126_tbx-v-m-b-norm-ntv-cpms.aug.tsv',index_col=0,nrows=6)
obs_x2=pd.read_table(path_ds+'GSE154126_tbx-v-m-b-norm-ntv-reads.aug.tsv',index_col=0,nrows=6)
print('Both obs datasets in X files are the same:',(obs_x1==obs_x2).all().all())

# %%
obs_x1

# %%
# metadata
# For some reason there are 2 obs tables on GEO that each contain part of the data
obs1=pd.read_table(path_ds+'GSE154126-GPL11154_series_matrix.txt',
                  skiprows=38,index_col=0)
obs2=pd.read_table(path_ds+'GSE154126-GPL16791_series_matrix.txt',
                  skiprows=38,index_col=0)
obs=pd.concat([obs1,obs2],axis=1)
print(obs.shape)

# %%
# Correct col names to remove donor info from cell names as it is already in the table, 
# for x matching
obs.columns=[c.split(':')[0] for c in obs.columns]

# %%
obs

# %%
# Select columns
obs=obs.T[['!Sample_geo_accession']]
obs.columns=['geo_accession']

# %%
# Check if some cells from x are missing in obs
[c for c in x.columns if c not in obs.index]

# %%
# Concat both obs datasets
# Format col names
obs_x1.loc['source_id',:]=obs_x1.columns
obs_x1.columns=obs_x1.loc['gene|cell_id',:].str.replace('cell.','',regex=False)
obs_x1=obs_x1.T.rename({'source_id':'donor'})
obs=pd.concat([obs,obs_x1],axis=1)

# %%
obs

# %% [markdown]
# C: Some cells miss geo_accession, but that not supplied as matrix series has less cells than expresion matrix

# %%
# Parse obs
# Select only some cols as others in donor df
obs=obs[['geo_accession','condition_health_status','CellType','source_id']]
obs.rename({'condition_health_status':'disease','CellType':'cell_type_original',
           'source_id':'donor'},axis=1,inplace=True)
obs['disease']=obs['disease'].replace({'Control':'healthy'})

# %%
# Parse donors
obs['donor']=obs.donor.apply(lambda x: x.split('.')[0])

# %% [markdown]
# Add donor info

# %%
# Load donor info
donor_df=pd.read_excel(path_ds+'GSE154126_donors.xlsx',index_col=0).iloc[:-1,:]

# %%
donor_df

# %%
# Parse donor df
donor_df.drop('Condition',axis=1,inplace=True) # Drop as better in other table
donor_df.rename(
    {'Age':'age','Gender':'sex','Ethnicity':'ethnicity','BMI':'BMI',
                 'Condition':'disease'},axis=1,inplace=True)
donor_df['age']=donor_df['age'].apply(lambda x: x[:-1]+' '+x[-1])
donor_df['ethnicity']=donor_df['ethnicity'
                              ].str.lower().str.replace('.','_').replace({'NA':np.nan})
donor_df.replace('NA',np.nan)

# %%
donor_df=donor_df.replace('\xa0NA',np.nan).replace('\xa0na',np.nan)

# %%
donor_df

# %%
# Add donor to obs
for col in donor_df.columns:
    obs[col]=obs['donor'].map(donor_df[col].to_dict())

# %%
obs.cell_type_original.value_counts()

# %%
obs['cell_type']=obs['cell_type_original'].replace({
    'duct':'ductal','pp':'gamma','masked':'dropped'})
obs.cell_type.value_counts(dropna=False)

# %%
# Check donors
obs.drop_duplicates('donor')

# %% [markdown]
# Make adata

# %%
# Check that x and x_norm match
if not (x.index==x_norm.index).all() and \
    (x.columns==[c.replace('cell.','') for c in x_norm.columns]).all():
    raise ValueError('x and x_norm not matching')

# %%
adata=sc.AnnData(X=csr_matrix(x.T),obs=obs.reindex(x.T.index),
                layers={'normalised_original':csr_matrix(x_norm.T)},
                 var=pd.DataFrame(index=x.T.columns))

# %%
adata

# %%
# Save orginal X
adata.layers['raw']=adata.X.copy()

# %%
# Log normalise
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# %%
adata

# %%
for col in adata.obs.columns:
    print(col)
    print(adata.obs[col].unique())

# %%
if SAVE:
    adata.write(path_ds+'adata.h5ad')

# %%
# Subset to annotated cells
adata_sub=adata[~adata.obs.cell_type.isin(['dropped','unknown']),:]
print(adata_sub.shape)
if SAVE:
    adata_sub.write(path_ds+'adata_filtered.h5ad')

# %% [markdown]
# ## GSE101207 - Drop-seq

# %%
dataset='GSE101207'
path_ds=data_path+dataset+'/GEO/'

# %% [markdown]
# Parse metadata

# %%
sm=pd.read_table(path_ds+'GSE101207-GPL11154_series_matrix.txt',skiprows=34,index_col=0)
sm1=pd.read_table(path_ds+'GSE101207-GPL17021_series_matrix.txt',skiprows=34,index_col=0)
print('SM1 is mouse:',(sm1.loc['!Sample_organism_ch1',:]=='Mus musculus').all())
print('SM shape:',sm.shape)

# %%
sm

# %% [markdown]
# Series matrix contains only donor info

# %%
obs=pd.read_excel(path_ds+'1-s2.0-S2211124719302141-mmc2.xlsx',sheet_name='Celltype.info',
                  skiprows=2,index_col=0)

# %%
obs

# %%
# Drop unused columns
obs.drop(['Tx.all','Tx.D'],axis=1,inplace=True)
# rename ct col
obs.rename({'celltype':'cell_type','Donor':'donor'},axis=1,inplace=True)

# %%
# Cell types
obs['cell_type'].value_counts(dropna=False)

# %%
# Parse cts
obs['cell_type']=obs['cell_type'].replace({'allBeta':'Beta'})
obs['cell_type_original']=obs['cell_type']
obs['cell_type']=obs['cell_type'].str.lower().replace({
    'duct':'ductal','psc':'stellate','pp':'gamma'}).fillna('dropped')

# %%
obs['cell_type'].value_counts(dropna=False)

# %% [markdown]
# From the paper: "28,026 “clean” cells without ambiguity" - others I have named here as dropped

# %% [markdown]
# Find mapping between donor id and cell suffix

# %%
donor_suf=set()
for idx,data in obs.iterrows():
    donor_suf.add(tuple([data['donor'],(idx.split('_')[1] if len(idx.split('_'))>1 else None)]))
print(donor_suf)

# %%
# Change suffix to donor to make processing easier downstream when mattching to expression
obs.index=[i.split('_')[0]+'_'+donor for i,donor in zip(obs.index,obs.donor)]

# %%
# Donor info
donor_df=pd.read_excel(path_ds+'1-s2.0-S2211124719302141-mmc2.xlsx',sheet_name='Donor_Full_info',
                  skiprows=2,index_col=0)

# %%
donor_df

# %%
# Edit colnames and values
donor_df.fillna('NA',inplace=True)
donor_df.drop(['Weight','Height'],axis=1,inplace=True)
donor_df.rename({'Gender':'sex','Age':'age','Race':'ethnicity',
                 'Death cause':'death_cause','disease status':'disease'},axis=1,inplace=True)
donor_df['sex'].replace({'M':'male','F':'female'},inplace=True)
donor_df['age']=donor_df['age'].apply(lambda x: str(x)+' y')
donor_df['ethnicity']=donor_df['ethnicity'].str.lower()
donor_df['death_cause']=donor_df['death_cause'].str.lower()
donor_df['disease'].replace(
    {'non-diabetic':'healthy','Type 2 diabetes':'T2D'},inplace=True)

# %%
donor_df

# %%
# Add donor info to obs
for col in donor_df.columns:
    obs[col]=obs['donor'].map(donor_df[col].to_dict())

# %%
obs['organ']='pancreas'
obs['tissue']='pancreatic islets'
obs['organism']='human'

# %%
obs

# %%
obs.drop_duplicates('donor')

# %% [markdown]
# Read expression data and convert to sparse adata

# %%
adata=[]
for donor in obs.donor.unique():
    files=glob.glob(path_ds+'*_'+donor+'.down.gene_exon_tagged.cleaned.dge.txt')
    if len(files)!=1:
        raise ValueError('Not exactly 1 file per donor')
    x_sub=pd.read_table(files[0], index_col=0).T
    x_sub.index=[i+'_'+donor for i in x_sub.index]
    x_sub=sc.AnnData(x_sub)
    x_sub.X=csr_matrix(x_sub.X)
    adata.append(x_sub)
adata=sc.concat(adata,join='outer')

# %%
print('N cells not in obs:',len([o for o in adata.obs_names if o not in obs.index]),
      'N cells in obs:',len([o for o in adata.obs_names if o in obs.index]),
      'N obs cells not in data:',len([o for o in obs.index if o not in adata.obs_names]))
print('Obs cells not in x:')
print([o for o in obs.index if o not in adata.obs_names])

# %%
# Add obs
adata.obs=obs.reindex(adata.obs_names)

# %%
# Fill ct info if not present
adata.obs['cell_type'].fillna('dropped',inplace=True)

# %%
adata

# %%
# Save orginal X
adata.layers['raw']=adata.X.copy()

# %%
# Log normalise
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# %%
adata

# %%
for col in adata.obs.columns:
    print(col)
    print(adata.obs[col].unique())

# %%
if SAVE:
    adata.write(path_ds+'adata.h5ad')

# %%
# Subset to annotated cells
adata_sub=adata[~adata.obs.cell_type.isin(['dropped','unknown']),:]
print(adata_sub.shape)
if SAVE:
    adata_sub.write(path_ds+'adata_filtered.h5ad')

# %% [markdown]
# ## GSE124742 and GSE164875 - SmartSeq2

# %%
dataset='GSE124742_GSE164875'
path_ds=data_path+dataset+'/GEO/'

# %% [markdown]
# ### Patch-Seq

# %%
path_ds_sub=path_ds+'patch/'

# %% [markdown]
# Parse metadata

# %%
obs=pd.read_table(path_ds_sub+'12-02-2022_CellMetabolism_2022_Patch-Seq.csv',
                  nrows=48,index_col=0,sep=',')

# %%
obs

# %%
obs_gh=pd.read_table(path_ds_sub+'patchclamp_wcryo_human.metadata.tab',index_col=0)

# %%
display(obs_gh)
print(list(obs_gh.columns))

# %%
print('obs_gh CellTypeEstimatePatching')
print(obs_gh['CellTypeEstimatePatching'].value_counts())
print('obs_gh cell_type')
print(obs_gh['cell_type'].value_counts())
print('obs cell_type')
print(obs.loc['cell_type',:].value_counts())

# %%
print('obs donors:',obs.T['Donor ID'].unique())
print('obs_gh donors:',obs_gh['DonorID'].unique())

# %%
# Check if all obs_gh donors are in obs
print('Any gh donors missing from Theorors data:',[d for d in obs_gh['DonorID'].unique() 
 if d.replace('cryo_','').replace('_T1D','') not in obs.T['Donor ID'].unique()])

# %% [markdown]
# The dataset from Theodore (not GitHub) has extra donors from the new study. Thus this data will be used.

# %%
obs.index

# %%
# Parse obs
obs=obs.T
# Rename and keep some cols
obs.rename({'Donor ID':'donor','Age':'age','Sex':'sex',
           'Diabetes Status':'disease','Years with Diabetes':'years_diagnosis',
           'cell_type':'cell_type_original'},
           axis=1,inplace=True)
obs=obs[['donor','age','sex','disease','years_diagnosis',
                                 'cell_type_original','HbA1c','BMI']]
obs['sex']=obs['sex'].map({'Female':'female','Male':'male'})
obs['age']=obs['age'].apply(lambda x: str(x)+' y')
obs['cell_type']=obs['cell_type_original']
obs['disease']=obs.disease.replace({'ND':'healthy'})
obs['years_diagnosis']=obs['years_diagnosis'].replace({'ND':np.nan})

# %%
obs['cell_type']=obs['cell_type_original'].replace({
    'fail_qc':'dropped','other':'unknown','lost':'dropped','PSCs':'stellate'}).fillna('dropped')
print(obs['cell_type'].value_counts(dropna=False))

# %%
obs

# %%
obs.drop_duplicates('donor')

# %% [markdown]
# Parse expression data

# %%
x=pd.read_table(path_ds_sub+'12-02-2022_CellMetabolism_2022_Patch-Seq.csv',
                  skiprows=range(1,49),index_col=0,sep=',')

# %%
x

# %% [markdown]
# Make adata

# %%
adata=sc.AnnData(X=csr_matrix(x.T),obs=obs,
                 var=pd.DataFrame(index=x.T.columns))

# %%
adata

# %%
# Save orginal X
adata.layers['raw']=adata.X.copy()

# %%
# Log normalise
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# %%
adata

# %%
for col in adata.obs.columns:
    print(col)
    print(adata.obs[col].unique().tolist())

# %%
if SAVE:
    adata.write(path_ds_sub+'adata.h5ad')

# %%
# Subset to annotated cells
adata_sub=adata[~adata.obs.cell_type.isin(['dropped','unknown']),:]
print(adata_sub.shape)
if SAVE:
    adata_sub.write(path_ds_sub+'adata_filtered.h5ad')

# %% [markdown]
# ### FACS

# %%
path_ds_sub=path_ds+'FACS/'

# %%
obs=pd.read_table(path_ds_sub+'patchclamp_FACS_human.metadata.tab',index_col=0)

# %%
display(obs)
print(list(obs.columns))

# %%
# Add ct info as not in original obs
cts=pd.read_table(path_ds_sub+'cell_typing_FACS_endocrine.csv',index_col=0,header=None)
obs['cell_type']=cts.reindex(obs.index)[1]

# %%
obs['cell_type'].value_counts(dropna=False)

# %%
# Quality of annotated and non anno cells
print('Non-anno:',obs[obs['cell_type'].isna()][['percent_mito', 'all_counts']].mean())
print('Anno:',obs[~obs['cell_type'].isna()][['percent_mito', 'all_counts']].mean())

# %% [markdown]
# C: Non-annotated cells are low quality

# %%
# rename cts
obs['cell_type_original']=obs['cell_type']
obs['cell_type']=obs['cell_type'].replace({
    'alpha_mt':'alpha','endocrine':'unknown'}).fillna('dropped')
obs['cell_type'].value_counts(dropna=False)

# %%
# Parse obs
obs=obs[['cell_type','DonorID','DiabetesStatus','Age','Sex','Body mass index (BMI):',
        'Glycated hemoglobin (HbA1c)']]
obs.rename({'DonorID':'donor','DiabetesStatus':'disease','Age':'age','Sex':'sex',
            'Body mass index (BMI):':'BMI','Glycated hemoglobin (HbA1c)':'HbA1c'
           },axis=1,inplace=True)
obs['age']=obs.age.apply(lambda x:str(x)+' y')
obs.sex=obs.sex.map({'M':'male','F':'female'})

# %%
obs

# %%
obs.drop_duplicates('donor')

# %% [markdown]
# Expression

# %%
x=pd.read_table(path_ds_sub+'patchclamp_FACS_human.counts.tab',index_col=0)

# %%
x

# %% [markdown]
# Make adata

# %%
adata=sc.AnnData(X=csr_matrix(x.T),obs=obs,
                 var=pd.DataFrame(index=x.T.columns))

# %%
adata

# %%
# Save orginal X
adata.layers['raw']=adata.X.copy()

# %%
# Log normalise
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# %%
adata

# %%
for col in adata.obs.columns:
    print(col)
    print(adata.obs[col].unique().tolist())

# %%
if SAVE:
    adata.write(path_ds_sub+'adata.h5ad')

# %%
# Subset to annotated cells
adata_sub=adata[~adata.obs.cell_type.isin(['dropped','unknown']),:]
print(adata_sub.shape)
if SAVE:
    adata_sub.write(path_ds_sub+'adata_filtered.h5ad')

# %% [markdown]
# ## GSE81608 - SMARTer Ultra Low RNA

# %%
dataset='GSE81608'
path_ds=data_path+dataset+'/GEO/'

# %% [markdown]
# Parse metadata

# %%
obs=pd.read_table(path_ds+'GSE81608_series_matrix.txt',
                  skiprows=26,index_col=0)

# %%
obs

# %% [markdown]
# Load expression data

# %%
x=pd.read_table(path_ds+'GSE81608_human_islets_rpkm.txt', index_col=0)

# %%
x

# %% [markdown]
# C: The gene ids are entrez gene numbers.

# %%
mg = mygene.MyGeneInfo()
genemap = mg.querymany(x.index.to_list(), scopes='entrezgene', 
                       fields=['ensembl.gene','symbol'], species='human')

# %%
genemap_df=[]
for g in genemap:
    g_parsed={'uid':g['query']}
    g_parsed['gene_symbol']=g['symbol'] if 'symbol' in g else np.nan
    # Genes with multiple EIDs have these as list
    if 'ensembl' in g:
        if isinstance(g['ensembl'],list):
            g_parsed['EID']=','.join([gs['gene'] for gs in g['ensembl']])
        else:
            g_parsed['EID']=g['ensembl']['gene']
    genemap_df.append(g_parsed)
genemap_df=pd.DataFrame(genemap_df)
genemap_df.index=genemap_df.uid
genemap_df.drop('uid',axis=1,inplace=True)

# %%
genemap_df

# %%
adata=sc.AnnData(X=csr_matrix(x.T),obs=pd.DataFrame(index=x.columns),
                 var=pd.DataFrame(index=x.index))

# %%
# Logtransform
adata.layers['normalised_original']=adata.X.copy()
sc.pp.log1p(adata)

# %%
# Add gene info
for col in genemap_df:
    adata.var[col]=genemap_df[col]

# %% [markdown]
# Obs

# %%
# Parse obs
obs=obs.T[['!Sample_geo_accession','!Sample_characteristics_ch1']]
obs.columns=['geo_accession','tissue','donor','disease','age',
             'ethnicity','sex','cell_type_original']

# %%
obs['cell_type_original'].value_counts(dropna=False)

# %%
# Parse obs
obs.drop('tissue',axis=1,inplace=True)
obs['donor']=obs['donor'].str.replace('donor id: ','')
obs['disease']=obs['disease'].str.replace('condition: ','').str.replace('non-diabetic','healthy')
obs['age']=obs['age'].apply(lambda x: x.replace('age: ','')+' y')
obs['ethnicity']=obs['ethnicity'].str.replace('ethnicity: ','').map(
    {'AA':'african_american','C':'caucasian','AI':'asian_indian','H':'hispanic'})
obs['sex']=obs['sex'].str.replace('gender: ','').map({'M':'male','F':'female'})
obs['cell_type_original']=obs['cell_type_original'].str.replace('cell subtype: ','')
obs['cell_type']=obs['cell_type_original'].replace({'PP':'gamma'})

# %%
obs['cell_type'].value_counts(dropna=False)

# %%
donor_df=pd.read_excel(path_ds+'GSE81608_donors.xlsx',index_col=0)

# %%
donor_df

# %%
for col in ['BMI','HbA1c']:
    obs[col]=obs.donor.map(donor_df[col].to_dict())

# %%
obs.index=[i.replace('Pancreatic islet cell sample ','Sample_') for i in obs.index]

# %%
obs.drop_duplicates('donor')

# %% [markdown]
# Adata

# %%
adata.obs=obs.reindex(adata.obs_names)

# %%
adata

# %%
for col in adata.obs.columns:
    if col!='geo_accession':
        print(col)
        print(adata.obs[col].unique().tolist())

# %%
if SAVE:
    adata.write(path_ds+'adata.h5ad')

# %%
# Subset to annotated cells
adata_sub=adata[~adata.obs.cell_type.isin(['dropped','unknown']),:]
print(adata_sub.shape)
if SAVE:
    adata_sub.write(path_ds+'adata_filtered.h5ad')

# %% [markdown]
# ## GSE86469 - SMARTer v1

# %%
dataset='GSE86469'
path_ds=data_path+dataset+'/GEO/'

# %% [markdown]
# Parse metadata

# %%
obs=pd.read_table(path_ds+'GSE86469_series_matrix.txt',
                  skiprows=38,index_col=0)

# %%
obs

# %%
# Parse obs
# Transpose and select cols
obs=obs.T[['!Sample_geo_accession','!Sample_organism_ch1',
           '!Sample_characteristics_ch1']]

# %%
# Rename and subset cols
obs.columns=['geo_accession','DROP','cell_type_original','DROP',
             'sex','disease','age','ethnicity','BMI','donor']
obs.drop('DROP',axis=1,inplace=True)

# %%
obs['cell_type_original']=obs['cell_type_original'].str.replace('cell type: ','')
obs['sex']=obs['sex'].str.replace('Sex: ','').str.lower()
obs['disease']=obs['disease'].str.replace('disease: ','').map(
    {'Type 2 Diabetic':'T2D','Non-Diabetic':'healthy'})
obs['age']=obs['age'].str.replace('age: ','').apply( lambda x:str(x)+' y')
obs['ethnicity']=obs['ethnicity'].str.replace('race: ','').str.lower().str.replace(' ','_')
obs['BMI']=obs['BMI'].str.replace('bmi: ','')
obs['donor']=obs['donor'].str.replace('islet unos id: ','')

# %%
obs['cell_type_original'].value_counts(dropna=False)

# %%
obs['cell_type']=obs['cell_type_original'].str.lower().replace({
    'none/other':'dropped','gamma/pp':'gamma'})
obs['cell_type'].value_counts(dropna=False)

# %%
# Add donor info
donor_df=pd.read_excel(path_ds+'Supplemental_Table_S1.xlsx',skiprows=2,index_col=1)

# %%
donor_df

# %%
# Parse donor_df
# Subset to cols not in obs
donor_df=donor_df[['Race','On Diabetes Medication?','HbA1c']]
donor_df.columns=['ethnicity','medication','HbA1c']
donor_df['ethnicity']=donor_df['ethnicity'].map(
    {'W':'white','AA':'african_american','H':'hispanic'})
donor_df['medication']=donor_df['medication'].str.lower()

# %%
donor_df

# %%
# Add donor info to obs
for col in donor_df.columns:
    obs[col]=obs['donor'].map(donor_df[col].to_dict())

# %%
obs

# %%
obs.drop_duplicates('donor')

# %% [markdown]
# Expression data

# %%
x=pd.read_table(path_ds+'GSE86469_GEO.islet.single.cell.processed.data.RSEM.raw.expected.counts.csv',
               sep=',',index_col=0).T

# %%
x

# %%
plt.boxplot(x.sum(axis=0))
plt.yscale('log')

# %% [markdown]
# C: Data is expected raw counts - needs normalisation

# %% [markdown]
# Make adata

# %%
genes=pd.read_table(path_ds+'mart_grch37_hs_genes.txt',index_col=0)

# %% [markdown]
# Make adata

# %%
adata=sc.AnnData(X=csr_matrix(x),obs=obs,
                 var=pd.DataFrame(index=x.columns))

# %%
# Edit var
adata.var['EID']=adata.var_names
adata.var_names=genes.loc[x.columns,'Gene name']

# %%
adata.var

# %%
adata

# %%
# Save orginal X
adata.layers['raw']=adata.X.copy()

# %%
# Log normalise
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# %%
adata

# %%
for col in [c for c in adata.obs.columns if c!='geo_accession']:
    print(col)
    print(adata.obs[col].unique().tolist())

# %%
if SAVE:
    adata.write(path_ds_sub+'adata.h5ad')

# %%
# Subset to annotated cells
adata_sub=adata[~adata.obs.cell_type.isin(['dropped','unknown']),:]
print(adata_sub.shape)
if SAVE:
    adata_sub.write(path_ds+'adata_filtered.h5ad')

# %% [markdown]
# ## GSE81547

# %%
dataset='GSE81547'
path_ds=data_path+dataset+'/GEO/'

# %% [markdown]
# Metadata

# %%
obs=pd.read_table(path_ds+'GSE81547_series_matrix.txt',
                  skiprows=29,index_col=0)

# %%
obs

# %%
# Subset obs
obs=obs.T[['!Sample_geo_accession','!Sample_characteristics_ch1']]
obs.columns=['geo_accession','age','sex','cell_type_original']

# %% [markdown]
# Add donor info

# %%
# Add donor info
donor_df=pd.read_table(path_ds+'GSE81547_donors.txt',index_col=0)

# %%
donor_df

# %%
# Parse obs
obs['age']=obs['age'].apply(lambda x: x.split(': ')[1])
obs['sex']=obs['sex'].str.replace('gender: ','')
obs['cell_type_original']=obs['cell_type_original'].str.replace('inferred_cell_type: ','')

# %%
# Match to donor based on metadata
for idx in obs.index:
    sex=obs.at[idx,'sex']
    age=obs.at[idx,'age']
    donors=donor_df.query('sex == @sex and age.str.split(' ').str[0] == @age', 
                          engine="python").index
    if donors.shape[0]!=1:
        print(obs.loc[idx,:],donors)
        raise ValueError('Could not match to single donor')
    else:
        obs.at[idx,'donor']=donors[0]

# %%
# Add donor metadata
for col in ['sex','age','ethnicity','BMI']:
    obs[col]=obs['donor'].map(donor_df[col].to_dict())

# %%
obs['cell_type_original'].value_counts(dropna=False)

# %%
obs['cell_type']=obs['cell_type_original'].replace(
    {'unsure':'unknown','mesenchymal':'mesenchyme'})
obs['cell_type'].value_counts(dropna=False)

# %%
# Change index for matching with x
obs['cell_name']=obs.index
obs.index=obs.geo_accession

# %%
# All donors where healthy
obs['disease']='healthy'

# %%
obs

# %%
obs.drop_duplicates('donor')


# %% [markdown]
# Expression

# %%
# Functions for loading expression files
def read_fn(f):
    # Read ingle matrix file
    return pd.read_table(f,sep='\t',index_col=0,header=None)

def read_gz(fn,read_fn=read_fn):
    # Read gzipped file
    uncompressed_file_type = fn.split('.')[-2]
    with TemporaryDirectory() as tmpdir:
        tmppth = tmpdir + f"/decompressed.{uncompressed_file_type}"
        with gzip.open(fn, "rb") as input_f, open(tmppth, "wb") as output_f:
            shutil.copyfileobj(input_f, output_f)
        x = read_fn(tmppth)
    return x


# %%
remove_var=['tAKR', 'no_feature', 'ambiguous', 'too_low_aQual', 'not_aligned',
       'alignment_not_unique']
x=[]
for f in glob.glob(path_ds+'expression/*'):
    # Read, drop non-genes, rename to cell name
    x.append(read_gz(f
                   ).drop(remove_var).rename({1:f.split('/')[-1].split('_')[0]},axis=1))
x=pd.concat(x,axis=1)    

# %%
x

# %% [markdown]
# Make adata

# %%
var=pd.DataFrame(index=x.T.columns)
var.index.name=None
adata=sc.AnnData(X=csr_matrix(x.T),obs=obs.reindex(x.columns),
                 var=var)

# %%
adata

# %%
# Save orginal X
adata.layers['raw']=adata.X.copy()

# %%
# Log normalise
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# %%
adata

# %%
for col in [c for c in adata.obs.columns if c!='geo_accession' and c!='cell_name']:
    print(col)
    print(adata.obs[col].unique().tolist())

# %%
if SAVE:
    adata.write(path_ds+'adata.h5ad')

# %%
# Subset to annotated cells
adata_sub=adata[~adata.obs.cell_type.isin(['dropped','unknown']),:]
print(adata_sub.shape)
if SAVE:
    adata_sub.write(path_ds+'adata_filtered.h5ad')

# %% [markdown]
# ## GSE114297 - no ct anno

# %% [markdown]
# ## GSE198623
# At the time of the analysis this dataset was not yet published (in-house data). The data is now availiable on GEO. In some of the notebooks this dataset is also called "Sophie's".
#
# Requires just some cell column renaming

# %%
dataset='P21000'
path_ds=data_path+dataset+'/sophie/human/'

# %%
adata_original=sc.read(path_ds+'human_processed.h5ad')

# %%
adata_original

# %%
# Modify adata to match others
adata=adata_original.copy()
adata.obs.drop(adata.obs.columns,axis=1,inplace=True)
adata.layers['raw']=adata.raw.X.copy()
adata.layers['normalised_original']=adata.X.copy()
adata.X=adata.raw.X
del adata.raw
del adata.obsm['X_pca']
del adata.obsm['X_umap']
del adata.uns

# %%
# Log normalise
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# %%
# Modify obs
adata.obs['donor']=adata_original.obs['id']
adata.obs['age']=adata_original.obs['sample'].apply(lambda x:x.split('_')[1]+ ' y')
adata.obs['sex']=adata_original.obs['sex_ontology_term_id'].map(
    {'PATO_0000383':'female', 'PATO_0000384':'male'})
adata.obs['BMI']=adata_original.obs['BMI'].astype(float)
adata.obs['HbA1c']=adata_original.obs['HbA1c'].astype(float)
adata.obs['cell_type_original']=adata_original.obs['louvain_anno_broad']
adata.obs['cell_type']=adata_original.obs['louvain_anno_broad'].replace({'PP':'gamma'})
adata.obs['cell_subtype_original']=adata_original.obs['louvain_anno_fine']
adata.obs['disease']='healthy'

# %%
adata.obs['cell_type'].value_counts(dropna=False)

# %%
# Map var symbols, use EID if can not map symbol
gene_map=pd.read_table(path_genes+'HGNC_export_104HS_EnsemblID.txt',index_col=1)
adata.var['EID']=adata_original.var.index
eids=set(gene_map.index)
symbols=[]
for g in adata.var_names:
    if g in eids:
        s=gene_map.at[g,'Approved symbol']
        if isinstance(s,str):
            symbols.append(s)
        else:
            symbols.append(g)
    else:
        symbols.append(g)
adata.var_names=symbols

# %%
(adata.var.index==adata.var.EID).sum()

# %%
adata

# %%
for col in [c for c in adata.obs.columns if c!='geo_accession' and c!='cell_name']:
    print(col)
    print(adata.obs[col].unique().tolist())

# %%
if SAVE:
    adata.write(path_ds+'adata_filtered.h5ad') # Already no NA cells

# %% [markdown]
# ## GSE148073 - 10x v2/v3

# %%
dataset='GSE148073'
path_ds=data_path+dataset+'/GEO/'

# %%
adata_original=sc.read(path_ds+'local.h5ad')

# %%
adata_original

# %%
adata_original.raw

# %%
# Modify adata to match others
adata=adata_original.copy()
adata.obs.drop(adata.obs.columns,axis=1,inplace=True)
adata.var.drop(adata.var.columns,axis=1,inplace=True)
adata.layers['raw']=adata.raw.X.copy()
adata.layers['normalised_original']=adata.X.copy()
adata.X=adata.raw.X
del adata.raw
del adata.obsm['X_pca']
del adata.obsm['X_umap']
del adata.uns
del adata.layers['scale.data']

# %%
# Log normalise
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# %%
adata_original.obs[['disease_state', 'cell_label', 'cell_type',  'disease',  ]]

# %%
pd.crosstab(adata_original.obs.disease_state,adata_original.obs.disease)

# %%
pd.crosstab(adata_original.obs.cell_label,adata_original.obs.cell_type)

# %%
adata.obs['cell_type_original']=adata_original.obs['cell_label']
adata.obs['cell_type']=adata.obs['cell_type_original'].replace({
    'acinar_minor_mhcclassII':'acinar','beta_major':'beta','beta_minor':'beta',
    'duct_acinar_related':'ductal','duct_major':'ductal','hybrid':'unknown',
    'immune_stellates':'immune_stellate','pp':'gamma','stellates':'stellate'})
adata.obs['disease']=adata_original.obs.disease_state.replace({'Control':'healthy'})

# %%
adata.obs['cell_type'].value_counts()

# %%
# Load donor info
donor_df=pd.read_excel(path_ds+'42255_2022_531_MOESM3_ESM.xlsx',
                       header=[0,1],skiprows=[0,44,45,46,47,48],
                       # read as str for below parsing
                      dtype='str')

# %%
# Merge rows as data of some patients spans multiple rows
temp=[]
curr_donor=-1
for row_idx,data in donor_df.iterrows():
    donor=data[('Identifier','Unnamed: 0_level_1')]
    if isinstance(donor,str):
        temp.append(data.to_dict())
        curr_donor+=1
    else:
        for col,val in data.iteritems():
            if isinstance(val,str):
                temp[curr_donor][col]=str(temp[curr_donor][col])+' '+str(val)
donor_df=pd.DataFrame(temp)
del temp

# %%
donor_df

# %%
# Parse donor df
donor_df.index=donor_df[('HPAP', 'Identifier')]
donor_df.rename({('Sex/', 'Age'):'sex_age',
                 ('Ancestry', 'Unnamed: 3_level_1'):'ethnicity',
                 ('BMI', 'Unnamed: 4_level_1'):"BMI",
                 ('Medical History', 'Unnamed: 5_level_1'):'diabetes',
                 ('HbA1c', '(C-Peptide)'):'HbA1c'
                },axis=1,inplace=True)
donor_df['sex']=donor_df['sex_age'].apply(lambda x: x.split('/')[0]).map(
    {'F':'female','M':'male'})
donor_df['age']=donor_df['sex_age'].apply(
    lambda x: x.split('/')[1].replace('yo',' y').replace('month',' m'))
donor_df['ethnicity']=donor_df['ethnicity'].str.lower()
donor_df['BMI']=donor_df['BMI'].apply(lambda x: x.split()[0] 
                                      if not x.split()[0].startswith("BMI") else x.split()[1])
donor_df['years_diagnosis']=donor_df['diabetes'].apply(
    lambda x:np.nan if 'yrs duration' not in x else x.split('(')[1].split()[0])
donor_df['HbA1c']=donor_df['HbA1c'].apply(lambda x: x.split()[0].replace('*','') 
                                          if  x.split()[0]!='Not' else np.nan)
donor_df=donor_df[['sex','age','ethnicity','BMI','years_diagnosis','HbA1c']]

# %%
donor_df

# %%
# Load data with donor info
ro.globalenv['path_ds']=path_ds
# Load Seurat rds and extract metadata
obs1=ro.r("readRDS(paste0(path_ds,'fasolino_et_al.rds'))[[]]")

# %%
obs1.iloc[0,:].to_dict()

# %% [markdown]
# C: only donor col is useful from here

# %%
print('Adata and Seurat obs names match:',(adata.obs_names==obs1.index).all())

# %%
# Add donor to obs
adata.obs['donor']=obs1['hpap_id']

# %%
# Add donor info to obs
for col in donor_df.columns:
    adata.obs[col]=adata.obs['donor'].map(donor_df[col].to_dict())

# %%
adata.obs['organism']='human'

# %%
adata.obs.drop_duplicates('donor')

# %%
# Map var symbols, use EID if can not map symbol
gene_map=pd.read_table(path_genes+'HGNC_export_104HS_EnsemblID.txt',index_col=1)
adata.var['EID']=adata_original.var.index
eids=set(gene_map.index)
symbols=[]
for g in adata.var_names:
    if g in eids:
        s=gene_map.at[g,'Approved symbol']
        if isinstance(s,str):
            symbols.append(s)
        else:
            symbols.append(g)
    else:
        symbols.append(g)
adata.var_names=symbols

# %%
(adata.var.index==adata.var.EID).sum()

# %%
adata

# %%
for col in [c for c in adata.obs.columns if c!='geo_accession' and c!='cell_name']:
    print(col)
    print(adata.obs[col].unique().tolist())

# %%
if SAVE:
    adata.write(path_ds+'adata.h5ad')

# %%
# Subset to annotated cells
adata_sub=adata[~adata.obs.cell_type.isin(['dropped','unknown']),:]
print(adata_sub.shape)
if SAVE:
    adata_sub.write(path_ds+'adata_filtered.h5ad')

# %% [markdown]
# Prepare per-sample scran normalisation (on filtered cells only)

# %%
# Load data
adata=sc.read(path_ds+'adata_filtered.h5ad')

# %%
# Use raw
adata.X=adata.layers['raw']

# %%
for sample, idx_sample in adata.obs.groupby('donor').groups.items():
    # Subset data
    adata_sub=adata[idx_sample,:].copy()
    print('Normalising:',sample,adata_sub.shape)
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
    ro.globalenv['data_mat'] = adata_sub.X.T.todense()
    ro.globalenv['input_groups'] = adata_sub_pp.obs['groups']
    try:
        size_factors = ro.r(f'calculateSumFactors(data_mat, clusters = input_groups, min.mean = 0.1, BPPARAM=MulticoreParam(workers = 16))')
    except:
        # Sometimes the above does not work so change parameter
        size_factors = ro.r(f'calculateSumFactors(data_mat, clusters = input_groups, min.mean = 0.2, BPPARAM=MulticoreParam(workers = 16))')
    adata.obs.loc[adata_sub.obs.index,'size_factors_sample'] = size_factors

del adata_sub
del adata_sub_pp

# %%
# Save parse anno and colors
if SAVE:
    h.update_adata(
            adata_new=adata, path=path_ds+'adata_filtered.h5ad',
            io_copy=False,
            add=[('obs',True,'size_factors_sample','size_factors_sample')],
        rm=None)

# %%
