# # Plot cellbender results
# Plot top ambient expression at different CellBender FPR (previously generated data) thresholds and in uncorrected data.

import scanpy as sc
import sys
import matplotlib.pyplot as plt
import sys  
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/mm_pancreas_atlas_rep/code/')
import helper as h
from matplotlib import rcParams

if False:
    # For testing
    UID3='a'
    folder='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE142465/scanpy_AnnData/SRR10751502/'
    raw_file='raw_feature_bc_matrix.h5ad'

UID3=sys.argv[1]
folder=sys.argv[2]
raw_file=sys.argv[3]

UID2='cellbender_ambient_plot'+UID3

# load cellbender data across FPRs
fprs=[0.01,0.05,0.1,0.2,0.3,0.5,0.7,0.9]
adatas_corrected=[]
adata_obss=None
for fpr in fprs:
    print(fpr)
    # Load cell bender data and normalise
    adata_corrected=h.readCellbenderH5(folder+'cellbender/cellbender_FPR_'+str(fpr)+'_filtered.h5')
    sc.pp.normalize_total(adata_corrected, target_sum=1e6, exclude_highly_expressed=True)
    sc.pp.log1p(adata_corrected)
    adatas_corrected.append(adata_corrected)
    print(adata_corrected.shape)
    if adata_obss is None:
        adata_obss==adata_corrected.obs_names
    else:
        if adata_obss !=adata_corrected.obs_names:
            raise ValueError('Filtered cells do not match between fpr thersholds')


# Load raw data, subset to cells retained by cell bender and normalise
adata_raw=h.open_h5ad(file=folder+raw_file,unique_id2=UID2)[adatas_corrected[0].obs_names,adatas_corrected[0].var_names]
sc.pp.normalize_total(adata_raw, target_sum=1e6, exclude_highly_expressed=True)
sc.pp.log1p(adata_raw)

# Compute UMAP on raw data
sc.pp.pca(adata_raw,n_comps=10,use_highly_variable =False)
sc.pp.neighbors(adata_raw, n_neighbors=15, n_pcs=10, metric='correlation')
sc.tl.umap(adata_raw)

# Plot top ambient genes (see list) in columns and 
# different cellbender thresholds in rows with uncorrected on the top
ambient=['Ins1','Ins2','Gcg','Sst','Ppy','Pyy','Malat1','Iapp','mt-Co3']
plotsize=3
rcParams['figure.figsize']=(plotsize*len(ambient),plotsize*(len(fprs)+1))
fig,axs=plt.subplots((len(fprs)+1),len(ambient))
plt.subplots_adjust(wspace=0.3,hspace=0.3)
for idx,gene in enumerate(ambient):
    vmaxs=[adata_raw[:,gene].X.toarray().max()]
    for fpr_idx in range(len(fprs)):
        vmaxs.append(adatas_corrected[fpr_idx][:,gene].X.toarray().max())
    vmax=max(vmaxs)
    sc.pl.umap(adata_raw,color=gene,ax=axs[0,idx],title=gene+' not-corrected',show=False,
               vmin=0,vmax=vmax,sort_order=False,s=3)
    for fpr_idx,fpr in enumerate(fprs):
        adata_corrected=adatas_corrected[fpr_idx]
        adata_corrected.obsm['X_umap']=adata_raw.obsm['X_umap']
        sc.pl.umap(adata_corrected,color=gene,ax=axs[fpr_idx+1,idx],
                   title=gene+' corrected fpr'+str(fpr),
                   show=False,vmin=0,vmax=vmax,sort_order=False,s=3)
plt.savefig(folder+'cellbender/cellbender_FPR_ambeintUMAP.png')


