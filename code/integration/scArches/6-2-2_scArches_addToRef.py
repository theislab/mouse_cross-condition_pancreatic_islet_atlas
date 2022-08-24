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
#     display_name: py3.6-scarches
#     language: python
#     name: py3.6-scarches
# ---

# %%
# *** scArches integration of new datasets to reference 
# (integration run selected from the original test on the smaller ref data subset)

# %%
import scanpy as sc
import pandas as pd
import pickle
import scarches as sca
import datetime
import os
import argparse
import glob
import anndata as ann
import numpy as np

import sys  
sys.path.insert(0, '/lustre/groups/ml01/workspace/karin.hrovatin/code/diabetes_analysis/')
import helper as h

# %%
# Paths for loading data and saving results
path_ref='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/ref_combined/preprocessed/'
path_model='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/ref_combined/scArches/'
path_out='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/scArches/integrate_add_ref/'
#Unique ID2 for reading/writing h5ad files with helper function
UID2='scArches_addToRef'

# %% [markdown]
# From the initially tested scArches models with different parameters on the ref subset we manually copied the selected one into a directory called ref_run, as the directories themselvles are named based on timestamp, so they would never have comparable name across re-runs.

# %%
# Load parameters of ref model
task_name_ref='ref_run'
params=pickle.load(open(path_model+task_name_ref+'/params.pkl','rb'))
print('Old params:',params)
params['task_name']=task_name_ref
print('Corrected old params:',params)

# %%
# Change params for new networks
params_new=params.copy()
params_new['task_name']='run_scArches'+str(datetime.datetime.now().timestamp())
params_new['sca_version']='scArches v2'
#params_new['early_stop']=early_stop
print('New params',params_new)

# %%
# *** Reference adata
adata_ref=h.open_h5ad(file=path_ref+'data_normalised.h5ad',unique_id2=UID2)

adata_ref.var.rename(columns={'highly_variable': 'highly_variable_2000'}, inplace=True)
hvg_col='highly_variable_'+params['hvg_n']
# Select HVG
adata_ref=adata_ref[:,adata_ref.var[hvg_col]]

# Ensure that adata.raw.X (used by scArches in nb) has the same genes as adata.X
adata_ref.raw=adata_ref.raw[:,adata_ref.raw.var_names.isin(adata_ref.var_names)].to_adata()
# Rename size factors so that scArches finds them
if 'size_factors' in adata_ref.obs.columns:
    print('Size factors are already present - renaming them to size_factors_old')
adata_ref.obs.rename(columns={'size_factors': 'size_factors_old', 'size_factors_sample': 'size_factors'}, inplace=True)

# %%
# *** New data
data=[('NOD','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE144471/data_normlisedForIntegration.h5ad'),
      ('NOD_elimination','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE117770/data_normlisedForIntegration.h5ad'),
      ('spikein_drug','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE142465/data_normlisedForIntegration.h5ad'),
      ('embryo','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE132188/rev7/data_normlisedForIntegration.h5ad'),
      ('VSG','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/VSG_PF_WT_cohort/rev7/nonref/data_normlisedForIntegration.h5ad'),
      ('STZ','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islet_glpest_lickert/rev7/nonref/data_normlisedForIntegration.h5ad')]

adatas=[]
for study,path_pp in data:
    print(study)
    #Load data
    adata_pp=h.open_h5ad(file=path_pp,unique_id2=UID2)
    print('Preprocessed data shape:',adata_pp.shape)  
    # Extract raw data, subset it to QC cells and integration genes and normalise raw
    adata=adata_pp.raw.to_adata()
    adata=adata[adata_pp.obs_names,adata_ref.var_names]
    adata.obs=adata_pp.obs.copy()
    # Normalise raw data with previously compited size factors
    adata.raw=adata.copy()
    adata.X /= adata.obs['size_factors_sample'].values[:,None] # This reshapes the size-factors array
    sc.pp.log1p(adata)
    adata.X = np.asarray(adata.X)
    adatas.append(adata)
    print('Integration data shape:',adata.shape) 

# Combine datasets    
adata_query = ann.AnnData.concatenate( *adatas,  batch_key = 'study', batch_categories = [d[0] for d in data ]).copy()

# %%
# Rename size factors so that scArches finds them
if 'size_factors' in adata_query.obs.columns:
    print('Size factors are already present - renaming them to size_factors_old')
else:
    adata_query.obs.rename(columns={'size_factors': 'size_factors_old', 'size_factors_sample': 'size_factors'}, inplace=True)

# %%
print('Reference adata')
print(adata_ref)
print('Query adata')
print(adata_query)

# %%
# *** Restore model
network = sca.models.scArches(task_name=params['task_name'],
    x_dimension=params['x_dimension'],
    z_dimension=params['z_dimension'],
    architecture=params['architecture'],
    gene_names=adata_ref.var_names.tolist(),
    conditions=adata_ref.obs['study_sample'].unique().tolist(),
    alpha=params['alpha'], 
    beta=params['beta'],
    loss_fn=params['loss_fn'],
    model_path=path_model,
    )

# %%
network.train(adata_ref,
              condition_key='study_sample',
              retrain=False,
              # These conditions are not really needed as network is not retrained
              n_epochs=params['n_epochs'],
              batch_size=params['batch_size'],
              save=False
             )

# %%
# *** Modify scArches model with new studies
network_new=network

network_new = sca.operate(network_new,
    new_task_name=params_new['task_name'],
    new_conditions=adata_query.obs['study_sample'].unique(),
    # Does not work, so change below
    #new_network_kwargs={'model_path':path_out}
    # Mo's suggestion
    #new_network_kwargs={'use_batchnorm':params_new['use_batchnorm']},
    version=params_new['sca_version']
                     )
network_new.model_path=path_out+network_new.task_name+os.sep
network_new.train(adata_query,
          condition_key='study_sample',
          retrain=True,
          n_epochs=params_new['n_epochs'],
          batch_size=params_new['batch_size'],
          save=True,
          #early_stop_limit=params_new['early_stop']
         )   

# %%
# *** Save integration results

# %%
# Save params
pickle.dump(params_new, open( path_out+params_new['task_name']+"/params.pkl", "wb" ) )

# %%
# Get latent representation
# Combined data for latent representation
adata = ann.AnnData.concatenate( adata_query,adata_ref, batch_categories =['nonref','ref']).copy()
# Make sure that obs_names match ref
adata.obs_names=[name.replace('-ref','').replace('-nonref','') for name in adata.obs_names]
latent_adata = network_new.get_latent(adata, 'study_sample')
print('Latent adata:\n',latent_adata.shape)

# %%
#Compute neighbours and UMAP
sc.pp.neighbors(latent_adata,n_pcs=0)
sc.tl.umap(latent_adata)

# %%
# Save latent data
h.save_h5ad(adata=latent_adata,file=path_out+params_new['task_name']+'/latent.h5ad',unique_id2=UID2)

# %%
