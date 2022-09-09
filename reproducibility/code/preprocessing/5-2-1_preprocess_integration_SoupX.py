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

UID2='pp_integration_SoupX'

print("sys.argv", sys.argv)
UID3=str(sys.argv[1])
PATH_ANALYSED=str(sys.argv[2])
PATHENDING_AMBIENTPP=str(sys.argv[3])
AMBIENTPP=str(sys.argv[4])

if False:
    UID3='a'
    PATH_ANALYSED='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE144471/'
    PATHENDING_AMBIENTPP='scanpy_AnnData/SRR1098509*/SoupX/SoupX_filtered_rhoadd005.h5ad'
    AMBIENTPP='SoupX_rhoadd005'

PATH_AMBIENTPP=PATH_ANALYSED+PATHENDING_AMBIENTPP
FILE_ANALYSED=PATH_ANALYSED+'data_normlisedForIntegration.h5ad'

# ## Load adata and metadata
#  - Use genes and cells selected before.
#  - Normalise by sample. 
#  - Add raw data with all genes.

# Read adata that was prepared for integration beforehand
adata_integrationpp=h.open_h5ad(file=FILE_ANALYSED,unique_id2=UID2,backed='r')

# +
#Add ambient pp data
    
# List all files
files=glob.glob(PATH_AMBIENTPP)
print('N Selected sample files:',len(files))
# Find which parts of file paths differ between files to later use them as file id
diff_path_idx=[]
for position in range(len(PATH_AMBIENTPP.split('/'))):
    values=set([file.split('/')[position] for file in files])
    if len(values)>1:
        diff_path_idx.append(position)
# Load files and extract parts of file path that identifies the file, compared to other loaded files
adatas=[]
file_diffs=[]
for file in files:
    print('Reading file',file)
    adatas.append(h.open_h5ad(file=file,unique_id2=UID2+UID3))
    file_diffs.append('_'.join([file.split('/')[i] for i in diff_path_idx]))

adata = ann.AnnData.concatenate( *adatas,  batch_key = 'file', batch_categories = file_diffs).copy()  
print('Adata shape before filtering:',adata.shape)

# Subset raw adata to cells that passed QC and ensure same ordering
adata=adata[adata_integrationpp.obs_names,:]
# Save raw to adata
adata.raw=adata
# Subset to genes previously termed to be used for integration
adata=adata[:,[var for var in adata_integrationpp.var_names if var in adata.var_names]]
print('Raw shape:',adata.raw.shape,'Filtered shape:',adata.shape)
# -

# This works as cell order is matched above
# Add study column
adata.obs['study']=adata_integrationpp.obs['study']
# Add merged column for study and sample: study_sample
adata.obs['study_sample']=adata_integrationpp.obs['study_sample']
# Add ref or not annotation
adata.obs['reference']=adata_integrationpp.obs['reference']

# ## Normalise and log transform

# +
for sample, idx_sample in adata.obs.groupby(['study','file']).groups.items():
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

path_save=PATH_ANALYSED+'data_'+AMBIENTPP+'_normlisedForIntegration.h5ad'

print('Saving to:',path_save)

h.save_h5ad(adata=adata,file=path_save,unique_id2=UID2+UID3)

print('Finished!')
