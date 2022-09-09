# # Plot cellbender results
# Plot top ambient expression at different SoupX rhos (auto and different increased) (previously generated data) thresholds and in uncorrected data.

import scanpy as sc
import sys
import matplotlib.pyplot as plt
import sys  
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/')
import helper as h
from matplotlib import rcParams

if False:
    UID3='a'
    folder='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE144471/scanpy_AnnData/SRR10985097/'
    file_ext='feature_bc_matrix.h5ad'

UID3=sys.argv[1]
file_ext=sys.argv[2]
folder=sys.argv[3]

print('UID3',UID3,'file_ext',file_ext,'folder',folder)

UID2='soupX_ambient_plot'+UID3

# load soupx data across added rhos
rhos_add=[0,0.05,0.1]
adatas_corrected=[]
for rho_add in rhos_add:
    print('Corrected with added rho',rho_add)
    # Load soupx data and normalise
    adata_corrected=h.open_h5ad(
        folder+'SoupX/SoupX_filtered_rhoadd'+str(rho_add).replace('.','')+'.h5ad',
        unique_id2=UID2)
    sc.pp.normalize_total(adata_corrected, target_sum=1e6, exclude_highly_expressed=True)
    sc.pp.log1p(adata_corrected)
    adatas_corrected.append(adata_corrected)
    print(adata_corrected.shape)


# Load raw data, subset to cells retained by cell bender and normalise
print('Raw data')
adata_raw=h.open_h5ad(file=folder+'filtered_'+file_ext,unique_id2=UID2)
print(adata_raw.shape)
sc.pp.normalize_total(adata_raw, target_sum=1e6, exclude_highly_expressed=True)
sc.pp.log1p(adata_raw)

# Plot top ambient genes (see list) in columns and 
# different cellbender thresholds in rows with uncorrected on the top
ambient=['Ins1','Ins2','Gcg','Sst','Ppy','Pyy','Malat1','Iapp','mt-Co3']
plotsize=3
rcParams['figure.figsize']=(plotsize*len(ambient),plotsize*(len(rhos_add)+1))
fig,axs=plt.subplots((len(rhos_add)+1),len(ambient))
plt.subplots_adjust(wspace=0.3,hspace=0.3)
for idx,gene in enumerate(ambient):
    vmaxs=[adata_raw[:,gene].X.toarray().max()]
    for rho_add_idx in range(len(rhos_add)):
        vmaxs.append(adatas_corrected[rho_add_idx][:,gene].X.toarray().max())
    vmax=max(vmaxs)
    sc.pl.embedding(adata_raw,'X_umap_normscl',color=gene,
                    ax=axs[0,idx],title=gene+' not-corrected',show=False,
               vmin=0,vmax=vmax,sort_order=False,s=3)
    for rho_add_idx,rho_add in enumerate(rhos_add):
        adata_corrected=adatas_corrected[rho_add_idx]
        adata_corrected.obsm['X_umap_normscl']=adata_raw.obsm['X_umap_normscl']
        sc.pl.embedding(adata_corrected,'X_umap_normscl',color=gene,ax=axs[rho_add_idx+1,idx],
                   title=gene+' corrected rho add'+str(rho_add),
                   show=False,vmin=0,vmax=vmax,sort_order=False,s=3)
plt.savefig(folder+'SoupX/SoupX_rhoadd_ambeintUMAP.png')

print('Finished!')
