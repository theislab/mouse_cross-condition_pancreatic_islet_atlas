# # PP raw filtered data
# PP filtered_feature_bc_matrix.h5ad to add it clusters and embedding.

import sys
import scanpy as sc

if False:
    # For testing
    UID3='a'
    file_ext='feature_bc_matrix.h5ad'
    folder='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/inceptor/citeseq_mouse_islets_DK/WT/expression/'

UID3=sys.argv[1]
file_ext=sys.argv[2]
folder=sys.argv[3]

file_name=folder+'filtered_'+file_ext

print('UID3:',UID3,'file extention:',file_ext,'folder:',folder,'file_name:',file_name)

adata=sc.read(file_name)

# Normalise data and get UMAP and clusters
adata_temp=adata.copy()
sc.pp.filter_genes(adata_temp, min_cells=1)
sc.pp.normalize_total(adata_temp, target_sum=1e6, exclude_highly_expressed=True)
sc.pp.log1p(adata_temp)
sc.pp.highly_variable_genes(adata_temp, flavor='cell_ranger',n_top_genes =2000)
sc.pp.scale(adata_temp,max_value=10)
sc.pp.pca(adata_temp, n_comps=15, use_highly_variable=True, svd_solver='arpack')
sc.pp.neighbors(adata_temp,n_pcs = 15) 
res=1
sc.tl.leiden(adata_temp,resolution=res)
sc.tl.umap(adata_temp)
adata.obs['leiden_r'+str(res)+'_normscl']=adata_temp.obs['leiden']
adata.obsm['X_umap_normscl']=adata_temp.obsm['X_umap']

adata.write(file_name)

print('Finished!')
