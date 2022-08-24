# # decontX ambient correction and visual evaluation
# Correct data with decontX and plot ambient expression before and after correction.

# +
import scanpy as sc
import anndata as ann
import numpy as np 
import pandas as pd
from scipy import sparse
import os

sc.settings.verbosity = 3

from matplotlib import rcParams
import matplotlib.pyplot as plt

import sys  
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/mm_pancreas_atlas_rep/code/')
import helper as h

#R interface
import rpy2.rinterface_lib.callbacks
import logging
from rpy2.robjects import pandas2ri
import anndata2ri
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)
pandas2ri.activate()
anndata2ri.activate()
# %load_ext rpy2.ipython

import matplotlib
matplotlib.use('Agg')
# -

ro.r('library("celda")')

if False:
    # For testing
    UID3='a'
    folder='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE144471/'
    #folder='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE142465/'
    fix_delta=True

UID3=sys.argv[1]
folder=sys.argv[2]
if len(sys.argv)>=4:
    # Fix decontX delta
    fix_delta=bool(int(sys.argv[3]))
else:
    fix_delta=False
print('UID3:',UID3,'folder:',folder)

UID2='decontX'+UID3

# Normalised data ready for integration (in raw non-normalised with all genes)
adata=h.open_h5ad(file=folder+"data_normlisedForIntegration.h5ad",unique_id2=UID2)

# Make clusters for decontX using data prepared for integration (normalised)
sc.pp.highly_variable_genes(adata, flavor='cell_ranger', batch_key='file',n_top_genes =2000)
adata_scl=adata.copy()
sc.pp.scale(adata_scl,max_value=10)
sc.pp.pca(adata_scl, n_comps=15, use_highly_variable=True, svd_solver='arpack')
sc.pp.neighbors(adata_scl,n_pcs = 15) 
sc.tl.leiden(adata_scl,resolution=1)
sc.tl.umap(adata_scl)

# extrat raw data for decontX
adata_raw=adata.raw.to_adata()
sc.pp.filter_genes(adata_raw, min_cells=1)
ro.globalenv['raw_mat'] = sparse.csr_matrix(adata_raw.X.T)
ro.globalenv['clusters'] = np.array(adata_scl.obs['leiden'].values)
ro.globalenv['batches']=np.array(adata_scl.obs['file'].values)
ro.globalenv['estimate_delta']=not(fix_delta)
if fix_delta:
    # Preselected value for fixed decontX delta
    ro.globalenv['deltas'] = [5,10]
else:
    # Default for nonfixed decontX delta
    ro.globalenv['deltas'] = [10,10]

# Corrected
corrected_res = ro.r(f'decontX( x = raw_mat,z = clusters,batch = batches,delta = unlist(deltas),estimateDelta = estimate_delta,seed = 12345, verbose = TRUE)')

# Print deltas (decontX strength params)
for batch in adata_scl.obs['file'].unique():
    print('Delta',batch,corrected_res.rx2("estimates").rx2(batch).rx2("delta"))

contamination=corrected_res.rx2("contamination")
corrected_mat=corrected_res.rx2("decontXcounts")

# Save result
if fix_delta:
    folder_out=folder+'decontX_fixed/'
else:
    folder_out=folder+'decontX/'
try:
    os.mkdir(folder_out)
except FileExistsError:
    pass
adata_corrected=sc.AnnData(X=corrected_mat.T, var=adata_raw.var,
           obs=adata_raw.obs[['file', 'study', 'study_sample', 'reference']])
h.save_h5ad(adata_corrected, folder_out+"data_decontX.h5ad",unique_id2=UID2)

# Top corrected genes
correction=pd.DataFrame(index=adata_corrected.var_names)
for file in adata_corrected.obs.file.unique():
    adata_raw_sub=adata_raw[adata_raw.obs.file==file,:]
    adata_sub=adata_corrected[adata_corrected.obs.file==file,:]
    correction['correction_mean_'+file]=np.squeeze(np.asarray((adata_sub.X-adata_raw_sub.X).mean(axis=0)))
del adata_sub
del adata_raw_sub
correction['overallMean_correction_mean']=correction[[col for col in correction.columns if 'correction_mean' in col]].mean(axis=1)
correction['abs_overallMean_correction_mean']=abs(correction['overallMean_correction_mean'])
for col in [col for col in correction.columns if 'correction_mean' in col and 'abs' not in col]:
    print(col)
    print(correction.sort_values('abs_overallMean_correction_mean',ascending=False)[col].iloc[:20])

# Normalise corrected data for UMAP plotting and add UMAP from uncorrected scaled data
#adata_corrected=adata_raw.copy()
#adata_corrected.X=corrected_mat.T
sc.pp.normalize_total(adata_corrected, target_sum=1e6, exclude_highly_expressed=True)
sc.pp.log1p(adata_corrected)
adata_corrected.obs['contamination']=contamination
adata_corrected.obsm['X_umap']=adata_scl.obsm['X_umap']

# Normalise raw data for UMAP plotting and add UMAP from uncorrected scaled data
adata_raw_norm=adata_raw.copy()
sc.pp.normalize_total(adata_raw_norm, target_sum=1e6, exclude_highly_expressed=True)
sc.pp.log1p(adata_raw_norm)
adata_raw_norm.obsm['X_umap']=adata_scl.obsm['X_umap']

# Plot top ambient genes on normalised uncorrected and corrected data
ambient=['Ins1','Ins2','Gcg','Sst','Ppy','Pyy','Malat1','Iapp','mt-Co3']
plotsize=3
rcParams['figure.figsize']=(plotsize*(len(ambient)+1),plotsize*2)
fig,axs=plt.subplots(2,(len(ambient)+1))
plt.subplots_adjust(wspace=0.3,hspace=0.3)
for idx,gene in enumerate(ambient):
    vmax=max([adata_raw_norm[:,gene].X.toarray().max(),adata_corrected[:,gene].X.toarray().max()])
    sc.pl.umap(adata_raw_norm,color=gene,ax=axs[0,idx],title=gene+' not-corrected',show=False,
               vmin=0,vmax=vmax,sort_order=False,s=3)
    sc.pl.umap(adata_corrected,color=gene,ax=axs[1,idx],
               title=gene+' corrected',
               show=False,vmin=0,vmax=vmax,sort_order=False,s=3)
sc.pl.umap(adata_raw_norm,color='file',ax=axs[0,idx+1],title='file',show=False, sort_order=False,s=3)
plt.savefig(folder_out+'decontX_ambeintUMAP.png')


