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

# %% [markdown]
# # scArches integration of multiple datasets from different adatas

# %%
import sys  
import scanpy as sc
import pandas as pd
import pickle
import scarches as sca
import datetime
import os
import argparse
import anndata as ann
import scIB as scib

sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/')
import helper as h

# %%
#Paths for loading ref params
path_refmodel='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/ref_combined/scArches/ref_run/'
# Path for storing results
path_out='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/scArches/integrate_combine_individual/'

data=[('Fltp_2y','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islets_aged_fltp_iCre/rev6/'),
      ('Fltp_adult','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islet_fltp_headtail/rev4/'),
      ('Fltp_P16','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/salinno_project/rev4/'),
      ('NOD','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE144471/'),
      ('NOD_elimination','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE117770/'),
      ('spikein_drug','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE142465/'),
      ('embryo','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE132188/rev7/'),
      ('VSG','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/VSG_PF_WT_cohort/rev7/'),
      ('STZ','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islet_glpest_lickert/rev7/')]

# %%
if False:
    UID3='a'
    input_file_name='data_normlisedForIntegration.h5ad'
    path_subset='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/beta_cell_names.pkl'

# %%
print(sys.argv)
UID3=sys.argv[1]
input_file_name=sys.argv[2]
# See default above
if len(sys.argv)>3:
    arg3=sys.argv[3]
    if arg3!="None":
        data=[tuple(study.split(',')) for study in arg3.split(';') ]
        print('Using data:\n',data)   
# See default above
if len(sys.argv)>4:
    arg4=sys.argv[4]
    if arg4!='None':
        path_out=arg4
        print('Will save to:',path_out)
path_subset=None
if len(sys.argv)>5:
    arg5=sys.argv[5]
    if arg5!='None':
        path_subset=arg5
        print('Will subset cells based on:',path_subset)
alpha=None
if len(sys.argv)>6:
    arg6=sys.argv[6]
    if arg6!='None':
        alpha=float(arg6)
        print('Reset alpha to:',alpha)
loss=None
if len(sys.argv)>7:
    arg7=sys.argv[7]
    if arg7!='None':
        loss=arg7
        print('Reset loss to:',loss)
quick=False
if len(sys.argv)>8:
    # 0 (False) or 1  (True) or None for default (False)
    arg8=sys.argv[8]
    if arg8!='None':
        quick=bool(int(arg8))
        print('Stopping is quick:',quick)        

# %%
#Unique ID2 for reading/writing h5ad files with helper function
UID2='scArches_combineIndividual'+UID3

# %% [markdown]
# ## Prepare data and params

# %%
# Load parameters of ref model
params=pickle.load(open(path_refmodel+'params.pkl','rb'))
print('Old params:',params)

# %%
# Change params for new networks
params_new=params.copy()
params_new['task_name']='run_scArches'+str(datetime.datetime.now().timestamp())
params_new['input_file_name']=input_file_name
if not quick:
    params_new['learning_rate']=0.0001
    params_new['early_stop_limit']=30

if alpha is not None:
    params_new['alpha']=alpha
if loss is not None:
    params_new['loss_fn']=loss
    
print('New params',params_new)

# %%
# *** New data
adatas=[]
for study,path in data:
    print(study)
    #Load data
    adata=h.open_h5ad(file=path+input_file_name,unique_id2=UID2)
    print(adata.shape)            
    adatas.append(adata)
    
# Combine datasets    
adata = ann.AnnData.concatenate( *adatas,  batch_key = 'study', 
                                batch_categories = [d[0] for d in data ]).copy()
# Edit obs_names to match reference
adata.obs_names=[name.replace('_ref','').replace('_nonref','') for name in adata.obs_names]

# %%
# Subset cells 
if path_subset is not None:
    cells_sub=pickle.load(open(path_subset,'rb'))
    cells_sub_present=[cell for cell in cells_sub if cell in adata.obs_names]
    print('Subsetting cells. N cells to subset:',len(cells_sub),
          'N present cells to subset:',len(cells_sub_present))
    print('N cells before subset:',adata.shape[0])
    adata=adata[cells_sub_present,:]
    print('N cells after subset:',adata.shape[0])

# %%
# Rename size factors so that scArches finds them
if 'size_factors' in adata.obs.columns:
    print('Size factors are already present - renaming them to size_factors_old')
adata.obs.rename(columns={'size_factors': 'size_factors_old', 'size_factors_sample': 'size_factors'}, inplace=True)

# %%
# Compute HVGs on combined dataset
if params_new['hvg_n']=='2000':
    # Compute HVG across batches (samples) using scIB function
    adata.obs['study_sample'] = adata.obs['study_sample'].astype('category')
    hvgs=scib.preprocessing.hvg_batch(adata, batch_key='study_sample', target_genes=2000, flavor='cell_ranger')
    # Add HVGs to adata
    hvgs=pd.DataFrame([True]*len(hvgs),index=hvgs,columns=['highly_variable'])
    hvgs=hvgs.reindex(adata.var_names,copy=False)
    hvgs=hvgs.fillna(False)
    adata.var['highly_variable']=hvgs.values
    print('Number of highly variable genes: {:d}'.format(adata.var['highly_variable'].sum()))
else:
    raise ValueError('HVG mode in params not recongnised')

# %%
# Retain only HVGs
adata=adata[:,adata.var['highly_variable']]
# Ensure that adata.raw.X (used by scArches in nb) has the same genes as adata.X
adata.raw=adata.raw[:,adata.raw.var_names.isin(adata.var_names)].to_adata()
print('Adata shape:',adata.shape,'Raw shape:',adata.raw.shape)

# %% [markdown]
# ## Create model

# %%
# Create model
if not quick:
    network = sca.models.scArches(task_name=params_new['task_name'],
        x_dimension=params_new['x_dimension'],
        z_dimension=params_new['z_dimension'],
        architecture=params_new['architecture'],
        gene_names=adata.var_names.tolist(),
        conditions=adata.obs['study_sample'].unique().tolist(),
        alpha=params_new['alpha'], 
        beta=params_new['beta'],
        loss_fn=params_new['loss_fn'],
        model_path=path_out,
        learning_rate=params_new['learning_rate']
        )
else:
    network = sca.models.scArches(task_name=params_new['task_name'],
        x_dimension=params_new['x_dimension'],
        z_dimension=params_new['z_dimension'],
        architecture=params_new['architecture'],
        gene_names=adata.var_names.tolist(),
        conditions=adata.obs['study_sample'].unique().tolist(),
        alpha=params_new['alpha'], 
        beta=params_new['beta'],
        loss_fn=params_new['loss_fn'],
        model_path=path_out,
        )    

# %%
# Run scArches
if not quick:
    network.train(adata,
              n_epochs=params_new['n_epochs'],
              batch_size=params_new['batch_size'], 
              condition_key='study_sample',
              save=True,
              retrain=True,
              early_stop_limit=params_new['early_stop_limit']
             )
else:
    network.train(adata,
              n_epochs=params_new['n_epochs'],
              batch_size=params_new['batch_size'], 
              condition_key='study_sample',
              save=True,
              retrain=True,
             )

# %% [markdown]
# ### Save additional information

# %%
path_save=path_out+params_new['task_name']+'/'

# %%
# Save params
pickle.dump(params_new, open( path_save+"params.pkl", "wb" ) )

# %%
# Get latent reepresentation
latent_adata = network.get_latent(adata, 'study_sample')
latent_adata

# %%
del adata

# %%
#Compute neighbours and UMAP
sc.pp.neighbors(latent_adata,n_pcs=0)
sc.tl.umap(latent_adata)

# %%
# Save latent data
h.save_h5ad(adata=latent_adata,file=path_save+'/latent.h5ad',unique_id2=UID2)

# %%
