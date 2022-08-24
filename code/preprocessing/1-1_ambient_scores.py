# Calculate emptyDrops ambient score for given sample

import scanpy as sc
import sys  

# +
#R interface
import rpy2.rinterface_lib.callbacks
import logging
from rpy2.robjects import pandas2ri
import anndata2ri

rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)
pandas2ri.activate()
anndata2ri.activate()
# %load_ext rpy2.ipython

import rpy2.robjects as ro

import sys  
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/mm_pancreas_atlas_rep/code/')
import helper as h
# -

ro.r('library(DropletUtils)')
ro.r('library(BiocParallel)')

if False:
    # For testing
    path='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE144471/scanpy_AnnData/SRR10985097/'
    ending='feature_bc_matrix.h5ad'

print("sys.argv", sys.argv)
path=str(sys.argv[1])
ending=str(sys.argv[2])

file=path+'raw_'+ending

adata_raw=sc.read(file)
initial_shape=adata_raw.shape
print('Raw shape:',adata_raw.shape)

#Remove empty genes and cells - copy so that initial file can be later saved
cell_filter=sc.pp.filter_cells(adata_raw, min_counts=1,inplace=False)[0]
gene_filter=sc.pp.filter_genes(adata_raw, min_cells=1,inplace=False)[0]
adata_raw_filtered=adata_raw[cell_filter,gene_filter].copy()
print('Raw shape:',adata_raw.shape,'Filtered shape',adata_raw_filtered.shape)

# Prepare data for emptyDrops
sparse_mat = adata_raw_filtered.X.T
genes = adata_raw_filtered.var_names
barcodes = adata_raw_filtered.obs_names

# Add data to r env
ro.globalenv['sparse_mat']=sparse_mat

# Get ambient scores
ro.r('ambient<-emptyDrops(counts(SingleCellExperiment(assays = list(counts = sparse_mat))),BPPARAM=MulticoreParam(workers = 4))')
ambient_scores=ro.r('as.data.frame(ambient@metadata$ambient)')
ambient=ro.r('ambient')

# Save cell ambience information
ambient.index=barcodes
for col in ambient.columns:
    adata_raw.obs.loc[ambient.index,'emptyDrops_'+col]=ambient[col].values

# Save gene ambient information
ambient_scores.index=genes
ambient_scores.columns=['ambience']
adata_raw.var.loc[ambient_scores.index,'emptyDrops_ambience']=ambient_scores['ambience']

# Make sure that correct data is saved
if initial_shape==adata_raw.shape:
    adata_raw.write(file)
else:
    raise ValueError('Can not save data - does not match original dimensions')

print('Finished!')
