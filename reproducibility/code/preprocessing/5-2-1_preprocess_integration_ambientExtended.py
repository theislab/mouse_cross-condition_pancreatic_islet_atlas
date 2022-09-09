# +
import scanpy as sc
import pandas as pd
import numpy as np
import glob
import anndata as ann
import pickle

import sys  
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/mm_pancreas_atlas_rep/code/')
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

UID2='pp_integration'

print("sys.argv", sys.argv)
UID3=str(sys.argv[1])
STUDY=str(sys.argv[2])
FILE=str(sys.argv[3])
PATH_RAW=str(sys.argv[4])
FOR_REF=int(sys.argv[5])
if FOR_REF in [0,1]:
    FOR_REF=bool(FOR_REF)
elif FOR_REF== -1:
    FOR_REF=None
else:
    raise ValueError('Unrecognised for_ref')

if False:
    UID3='a'
    STUDY='NOD'
    FILE='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE144471/data_annotated.h5ad'
    PATH_RAW='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE144471/scanpy_AnnData/SRR1098509*/filtered_feature_bc_matrix.h5ad'
    FOR_REF=False

PATH_STUDY='/'.join(FILE.split('/')[:-1])+'/'

# ## Load adata and metadata
#  - Normalise by sample (using genes selected for each study - e.g. without lowly expressed and ambient genes of that study - as selected for normalised data and annotation). 
#  - Add raw data with all genes.

# Read adata
adata_norm=h.open_h5ad(file=FILE,unique_id2=UID2+UID3)
adata=adata_norm.raw.to_adata()
ambient=pickle.load(open(PATH_STUDY+'ambient_genes_selection_extended.pkl','rb'))

# Keep only genes from normalised data (remove ambient ,...) and cells
adata=adata[adata_norm.obs_names,[var for var in adata.var_names if var not in ambient]]
del adata_norm

# +
#Add raw data with all genes
    
# Load raw data to have all necesary genes
# Load metadata for the project
metadata=pd.read_excel('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/scRNA-seq_pancreas_metadata.xlsx',
         sheet_name=STUDY)
if FOR_REF is not None:
    samples=metadata.query('for_reference == @FOR_REF')
else:
    samples=metadata.copy()
# List all files
files=glob.glob(PATH_RAW)
# Find which parts of file paths differ between files to later use them as file id
diff_path_idx=[]
for position in range(len(PATH_RAW.split('/'))):
    values=set([file.split('/')[position] for file in files])
    if len(values)>1:
        diff_path_idx.append(position)
# Subset to files used for reference
files_subset=[]
for file in files:
    sample='_'.join([file.split('/')[i] for i in diff_path_idx])
    if any(sample_name in sample for sample_name in samples.sample_name.values):
        files_subset.append(file)
print('N Selected sample files:',len(files_subset))
#print(files_subset)
# Load files and extract parts of file path that identifies the file, compared to other loaded files
adatas_raw=[]
file_diffs=[]
for file in files_subset:
    print('Reading file',file)
    adatas_raw.append(h.open_h5ad(file=file,unique_id2=UID2+UID3))
    file_diffs.append('_'.join([file.split('/')[i] for i in diff_path_idx]))

adata_raw = ann.AnnData.concatenate( *adatas_raw,  batch_key = 'file', batch_categories = file_diffs).copy()  

# Subset raw adata to cells that passed QC 
adata_raw=adata_raw[adata.obs_names,:]
# Add to adata
adata.raw=adata_raw

# -

print('Adata shape:',adata.shape,'raw shape:',adata.raw.shape)

# +
# Add study column
adata.obs['study']=STUDY
# Add merged column for study and sample: study_sample
adata.obs['study_sample']=[STUDY+'_'+sample for study, sample in zip(adata.obs.study,adata.obs.file)]

#Retain only relevant obs columns
relevant_obs_cols = ['file',   'study','study_sample']
relevant_obs_cols = set(relevant_obs_cols)&set(adata.obs.columns)
adata.obs=adata.obs[relevant_obs_cols]

# Add ref or not annotation
metadata=pd.read_excel('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/scRNA-seq_pancreas_metadata.xlsx',
                      sheet_name=STUDY)
samples=adata.obs.file.unique()
value_map={sample:metadata.query('sample_name =="'+sample+'"')['for_reference'].values[0] for sample in samples}
adata.obs['reference']=adata.obs.file.map(value_map)


#Rename per study size factors - UNUSED: Not saved
#adata.obs.rename(columns={'size_factors':'size_factors_study'}, inplace=True)

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

path_save=PATH_STUDY+'data_normlisedForIntegration_ambientExtended.h5ad'

print('Saving to:',path_save)

h.save_h5ad(adata=adata,file=path_save,unique_id2=UID2+UID3)


