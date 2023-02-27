# +
import scanpy as sc
import numpy as np
from pathlib import Path
import warnings
import os
from scib.metrics.clustering import opt_louvain
from scib.metrics.utils import NeighborsError
import scib.metrics as sm
from matplotlib import rcParams
# Use instead moransi_conservation from moransi_conservation.py
import scib.metrics.moransi_conservation as smmc

def compute_metrics(adata_full,latent_adata,metrics,
            cell_type_col,batch_col,path_save,TMP,
            PC_regression=True,ASW_batch=True,
            kBET=True,
            graph_connectivity=True,graph_iLISI=True,
            graph_cLISI=True,NMI=True, ARI=True,ASW_cell_type=True,
            isolated_label_F1=True,isolated_label_ASW=True, moransi_conservation=True,
            batch_anno_clusters=False,
            copy_expr_full=True, task_name='',):
    """
    :param adata_full: Non-integrated adata with expression
    :param latent_adata: Integrated adata with integrated embedding in X_emb
    :param metrics: Dict which will be modified in place with new metrics results
    :param cell_type_col: Obs col with cell types
    :param batch_col: Obs col with batches
    :param path_save: Save some figures and tables generated for calculating metrics
    :param TMP: Temporary dir for saving temporary data
    :param batch_anno_clusters: Use clusters instead of ct col for computation of batch metrics. 
    Useful when no anno is availiable and only batch integration metrics are computed.
    :param copy_expr_full: In Moran's I use expression from adata_full also for integrated data.
    :param task_name: Added to saved figures/tables filenames
    
    """
    print('N cells adata_full:',adata_full.shape[0],
          'N cells latent_adata',latent_adata.shape[0])
    print('Batches:',latent_adata.obs[batch_col].nunique(),
          list(latent_adata.obs[batch_col].unique()))
    print('Cell types:',latent_adata.obs[cell_type_col].nunique(),
         list(latent_adata.obs[cell_type_col].unique()))
    
    def remove_neigh(adata):
        # Remove any neighbour related entries from  adata
        adata.obsp.pop('connectivities',None)
        adata.obsp.pop('distances',None)
        adata.uns.pop('neighbors',None)
        # UMAP is retained at it is needed for plotting in kBET based on clusters

    adata_full=adata_full.copy()
    remove_neigh(adata_full)
    latent_adata=latent_adata.copy()
    remove_neigh(latent_adata)

    # recompute neighbours on latent data (not needed for adata_full as they are recomputed 
    # within functions where needed)
    latent_adata = sc.pp.neighbors(latent_adata, n_pcs=0, use_rep='X_emb', copy=True)

    # Compute opt clustering
    if batch_anno_clusters or NMI or ARI:
        res_max, nmi_max, nmi_all = opt_louvain(adata=latent_adata,
                label_key=cell_type_col, cluster_key='opt_louvain', function=sm.nmi,
                plot=False, verbose=True, inplace=True, force=True, ignore_na=True)            
        sc._settings.ScanpyConfig.figdir=Path(path_save)
        rcParams['figure.figsize']= (10,10)
        sc.pl.umap(latent_adata,color=['opt_louvain'],
               size=10,save='_batch_optlouvain_'+task_name+'.png',show=False)
    
    # Use optimised clustering instead of cell type for batch eval
    if batch_anno_clusters:
        cell_type_col_batch='opt_louvain'
    else:
        cell_type_col_batch=cell_type_col
        

    # BATCH

    # Principal component regression
    if PC_regression:
        print('Computing PC_regression')
        metrics['PC_regression']=sm.pcr_comparison(adata_pre=adata_full, 
                                                   adata_post=latent_adata, 
                                                   covariate=batch_col,
                                            embed='X_emb', n_comps=50, scale=True, verbose=False)

    # Batch ASW
    if ASW_batch:
        print('Computing ASW_batch')
        metrics['ASW_batch']=sm.silhouette_batch(latent_adata, 
                                                 batch_key=batch_col,
                                                 group_key=cell_type_col_batch,
                                                 metric='euclidean',  embed='X_emb', 
                                                 verbose=False, scale=True
                             )

    # kBET
    if kBET:
        print('Computing kBET')
        try:
            metrics['kBET']=sm.kBET(adata=latent_adata, batch_key=batch_col, 
                                   label_key=cell_type_col_batch, 
                                             embed='X_emb', 
                                         type_ = 'embed',
                                   scaled=True, verbose=False) 

        except NeighborsError as err:
            metrics['kBET']=0
            warnings.warn('kBET can not be calculated and was given value 0:')
            print("ValueError error: {0}".format(err))

    # Graph connectivity
    if graph_connectivity:
        print('Computing graph_connectivity')
        metrics['graph_connectivity']=sm.graph_connectivity(adata=latent_adata, 
                                                            label_key=cell_type_col_batch)

    if graph_iLISI:
        print('Computing graph_LISI')
        try:
            metrics['graph_iLISI'] = sm.ilisi_graph(adata=latent_adata,batch_key=batch_col,
                k0=90, type_='embed', subsample=0.5*100,scale=True,
                multiprocessing=True,nodes=4, verbose=False)
        except FileNotFoundError as err:
            warnings.warn('Could not compute iLISI scores due to FileNotFoundError')
            metrics['graph_iLISI'] = np.nan
            print("FileNotFoundError: {0}".format(err))

    # BIO

    # Graph cLISI
    if graph_cLISI:
        print('Computing graph_LISI')
        try:
            metrics['graph_cLISI'] = sm.clisi_graph(adata=latent_adata,
                batch_key=batch_col,label_key=cell_type_col,
                k0=90,type_='embed',subsample=0.5*100,scale=True,
                multiprocessing=True,nodes=4,verbose=False)
        except FileNotFoundError as err:
            warnings.warn('Could not compute cLISI scores due to FileNotFoundError')
            metrics['graph_cLISI'] = np.nan
            print("FileNotFoundError: {0}".format(err))

    # NMI
    if NMI:
        print('Computing NMI')
        metrics['NMI']=sm.nmi(adata=latent_adata, 
                              group1=cell_type_col, group2='opt_louvain', 
                                method="arithmetic",nmi_dir=None)

    # ARI
    if ARI:
        print('Computing ARI')
        metrics['ARI']=sm.ari(adata=latent_adata, 
                              group1=cell_type_col, group2='opt_louvain')

    # ASW cell type
    if ASW_cell_type:
        print('Computing ASW_cell_type')
        metrics['ASW_cell_type']=sm.silhouette(adata=latent_adata, 
                                               group_key=cell_type_col, embed='X_emb', 
                                               metric='euclidean')


    # Isolated label F1
    if isolated_label_F1:
        print('Computing isolated_label_F1')
        metrics['isolated_label_F1']=sm.isolated_labels(
            adata=latent_adata,  label_key=cell_type_col,  batch_key=batch_col,
            embed='X_emb', cluster=True,  iso_threshold=None, verbose=False)

    # Isolated label ASW
    if isolated_label_ASW:
        print('Computing isolated_label_ASW')
        metrics['isolated_label_ASW']= sm.isolated_labels(
            adata=latent_adata, label_key=cell_type_col,  batch_key=batch_col,
            embed='X_emb', cluster=False, iso_threshold=None, verbose=False)
        
    # moran's I conservation
    if moransi_conservation:
        print('Computing moransi_conservation')
        if copy_expr_full:
            latent_adata_temp=adata_full.copy()
            latent_adata_temp.obsm['X_emb']=latent_adata.obsm['X_emb']
        else:
            latent_adata_temp=latent_adata
        metrics['moransi_conservation']= smmc.moransi_conservation(
            adata_pre=adata_full,  adata_post=latent_adata_temp, n_hvg=1000,
            batch_key=batch_col, embed='X_emb',compare_pre=False, rescale=True)
