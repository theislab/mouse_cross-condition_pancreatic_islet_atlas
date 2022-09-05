# # Run scArches on smaller data subset (ref) to test different params

import time
t0 = time.time()

import scanpy as sc
import pandas as pd
import numpy as np
import pickle 
import datetime
import os.path
from pathlib import Path

from matplotlib import rcParams
import matplotlib
matplotlib.use('Agg')

import scarches as sca

import sys  
sys.path.insert(0, '/lustre/groups/ml01/workspace/karin.hrovatin/code/diabetes_analysis/')
import helper as h

#*** Prepare args
print("sys.argv", sys.argv)
UID3=str(sys.argv[1])
hvg_n=str(sys.argv[2])
z_dimension=int(sys.argv[3])
#Given as numbers separated by '_'
architecture=[int(layer) for layer in str(sys.argv[4]).split('_')]
beta=float(sys.argv[5])
alpha=float(sys.argv[6])
loss_fn=str(sys.argv[7])
n_epochs=int(sys.argv[8])
batch_size=int(sys.argv[9])
subset_beta=bool(int(sys.argv[10]))

if False:
    # For testing out the scipt
    UID3='1'
    hvg_n='2000'
    z_dimension=15
    #Given as numbers separated by '_'
    architecture=[128,128,128]
    beta=0
    alpha=0.9
    loss_fn='sse'
    n_epochs=150
    batch_size=128
    subset_beta=False

#Paths for loading data and saving results
path_in='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/ref_combined/preprocessed/'
path_out='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/ref_combined/scArches/evaluate/'
#Unique ID2 for reading/writing h5ad files with helper function
UID2='scArches_ref'+UID3

# *** Prepare adata for scarches

# Load data
adata=h.open_h5ad(file=path_in+'data_normalised.h5ad',unique_id2=UID2)
if subset_beta:
    selected_beta=['beta', 'beta_proliferative','beta_subpopulation_STZref', 'beta_ins_low']
    adata=adata[adata.obs.cell_type.isin(selected_beta),:]
adata_full=adata.copy()
# Initial HVG was 2000
adata.var.rename(columns={'highly_variable': 'highly_variable_2000'}, inplace=True)
hvg_col='highly_variable_'+hvg_n
# Select HVG
adata=adata[:,adata.var[hvg_col]]
# Ensure that adata.raw.X (used by scArches) has the same genes as adata.X
adata.raw=adata.raw[:,adata.raw.var_names.isin(adata.var_names)].to_adata()
# Rename size factors so that scArches finds them
if 'size_factors' in adata.obs.columns:
    print('Size factors are already present - renaming them to size_factors_old')
else:
    adata.obs.rename(columns={'size_factors': 'size_factors_old', 'size_factors_sample': 'size_factors'}, inplace=True)

#*** Run scArches
params={'z_dimension':z_dimension,'architecture':architecture,'task_name':'run_scArches'+str(datetime.datetime.now().timestamp()),
        'x_dimension':adata.shape[1],
       'beta':beta,'alpha':alpha,'loss_fn':loss_fn,'n_epochs':n_epochs,'batch_size':batch_size,
       'subset_beta':subset_beta,'hvg_n':hvg_n}

print(params)

# Create model
network = sca.models.scArches(task_name=params['task_name'],
    x_dimension=params['x_dimension'],
    z_dimension=params['z_dimension'],
    architecture=params['architecture'],
    gene_names=adata.var_names.tolist(),
    conditions=adata.obs['study_sample'].unique().tolist(),
    alpha=params['alpha'], 
    beta=params['beta'],
    loss_fn=params['loss_fn'],
    model_path=path_out,
    )

# Run scArches
network.train(adata,
              n_epochs=params['n_epochs'],
              batch_size=params['batch_size'], 
              condition_key='study_sample',
              save=True,
              retrain=True,
              #verbose=2
             )

# Save params
pickle.dump(params, open( path_out+params['task_name']+"/params.pkl", "wb" ) )

# Get latent reepresentation
latent_adata = network.get_latent(adata, 'study_sample')
latent_adata

del adata

#Compute neighbours and UMAP
sc.pp.neighbors(latent_adata,n_pcs=0)
sc.tl.umap(latent_adata)

# Save latent data
h.save_h5ad(adata=latent_adata,file=path_out+params['task_name']+'/latent.h5ad',unique_id2=UID2)

print('Time:',time.time()-t0)
