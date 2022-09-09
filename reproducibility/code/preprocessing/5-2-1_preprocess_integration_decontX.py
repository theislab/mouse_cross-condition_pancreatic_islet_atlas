# +
import scanpy as sc
import pandas as pd
import numpy as np
import glob
import anndata as ann

import sys  
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/')
import helper as h
from scipy import sparse

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import anndata2ri
pandas2ri.activate()
anndata2ri.activate()
# %load_ext rpy2.ipython

import rpy2.rinterface_lib.callbacks
import logging
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)
# -

ro.r('library("scran")')
ro.r('library("BiocParallel")')

UID2='pp_integration_decontX'

print("sys.argv", sys.argv)
UID3=str(sys.argv[1])
STUDY=str(sys.argv[2])
FOLDER=str(sys.argv[3])
FILE=str(sys.argv[4])
PATH_NONAMBPP=str(sys.argv[5])

if False:
    UID3='a'
    STUDY='NOD'
    main_path='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE144471/'
    #main_path='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE142465/'
    FOLDER=main_path+'decontX/'
    FILE='data_decontX.h5ad'
    PATH_NONAMBPP=main_path+'data_normlisedForIntegration.h5ad'

UID2=UID2+UID3

# ## Load adata and metadata
#  - Normalise by sample (using genes selected for each study - e.g. without lowly expressed and ambient genes of that study - as selected for normalised data and annotation). 
#  - Add raw data with all genes.

# Read ambient preprocessed adata
adata=h.open_h5ad(file=FOLDER+FILE,unique_id2=UID2)
# Read integration pp data without ambient pp
adata_integrationpp=h.open_h5ad(file=PATH_NONAMBPP,unique_id2=UID2,backed='r')

# Keep only genes from normalised data (remove ambient ,...) 
adata.raw=adata.copy()
# Some war might be empty as gene filtering was done before doublet filtering - 
# thus select only var present in ambient corrected data when filtering by var present in data for integration
adata=adata[:,[var for var in adata_integrationpp.var_names if var in adata.var_names]]
del adata_integrationpp

print('Adata shape:',adata.shape,'raw shape:',adata.raw.shape)

# +
# Add study column
adata.obs['study']=STUDY
# Add merged column for study and sample: study_sample
adata.obs['study_sample']=[STUDY+'_'+sample for study, sample in zip(adata.obs.study,adata.obs.file)]

#Retain only relevant obs columns
relevant_obs_cols = ['file', 'study','study_sample']
relevant_obs_cols = set(relevant_obs_cols)&set(adata.obs.columns)
adata.obs=adata.obs[relevant_obs_cols]

# Add ref or not annotation
metadata=pd.read_excel('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/scRNA-seq_pancreas_metadata.xlsx',
                      sheet_name=STUDY)
samples=adata.obs.file.unique()
value_map={sample:metadata.query('sample_name =="'+sample+'"')['for_reference'].values[0] for sample in samples}
adata.obs['reference']=adata.obs.file.map(value_map)

# -

# Remove obsm
del adata.obsm

# ## Normalise and log transform

# +
# Stored in raw (with additional genes)
#adata.layers['counts'] = adata.X.copy()

# +
for sample, idx_sample in adata.obs.groupby(['study','file']).groups.items():
    # Subset data
    adata_sub=adata[idx_sample,:].copy()
    print('Computing size factors for:',sample,adata_sub.shape)
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
    ro.globalenv['data_mat'] = adata_sub.X.T
    ro.globalenv['input_groups'] = adata_sub_pp.obs['groups']
    size_factors = ro.r(f'calculateSumFactors(data_mat, clusters = input_groups, min.mean = 0.1, BPPARAM=MulticoreParam(workers = 16))')
    adata.obs.loc[adata_sub.obs.index,'size_factors_sample'] = size_factors

del adata_sub
del adata_sub_pp
# -

# Scale data with size factors
adata.X /= adata.obs['size_factors_sample'].values[:,None] # This reshapes the size-factors array
sc.pp.log1p(adata)
adata.X = np.asarray(adata.X)

# ## Save

print(adata)

path_save=FOLDER+FILE.replace('.h5ad','_normlisedForIntegration.h5ad')

print('Saving to:',path_save)

h.save_h5ad(adata=adata,file=path_save,unique_id2=UID2+UID3)


