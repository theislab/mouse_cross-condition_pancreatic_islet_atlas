# # Integration evaluation
# - Plot embedding with different batch and cell type related information.
# - Evaluate integrated embedding with a subset of scIB metrics on different data subsets (all cells - with and without cell type, beta cells).

print('START')
import time
t0 = time.time()

# +
import scanpy as sc
import pandas as pd
import numpy as np
import pickle 
import datetime
import os.path
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import numpy as np
import anndata as ann
import sys  

sys.path.insert(0, '/lustre/groups/ml01/workspace/karin.hrovatin/code/diabetes_analysis/integration/')
from integration_eval_helper import *

from matplotlib import rcParams
import matplotlib
matplotlib.use('Agg')

from scIB import metrics as sm
# -

#*** Prepare args
print("sys.argv", sys.argv)
UID3=str(sys.argv[1])
folder=str(sys.argv[2])
raw_file=str(sys.argv[3])
split_refnonref=bool(int(sys.argv[4]))
if len(sys.argv) > 5:
# Some of the saving dependet on the server we used, 
# if re-run with a different setup this can be ommitted and odified accordingly 
    slurm=bool(int(sys.argv[5]))
else:
    slurm=False

if False:
    # For testing
    UID3='a'
    #folder='ref_combined/scVI/hyperopt/1599165129.871496_2000_mll/'
    #folder='ref_combined/scArches/ref_run/'
    folder='combined/scArches/integrate_combine_individual/run_scArches1605798604.647761/'
    slurm=False
    split_refnonref=False
    raw_file='data_normlisedForIntegration_ambientExtended.h5ad'

# +
# If "overwrite" add metrics to existing file (if it exists), 
# owerwritting those metrics that were previously computed.
# If "overwrite_all" make new file that will overwritte all metrics (also those that are not computed anew)
# If "missing" compute only metrics that are missing or np.nan in already saved file. Option
# to be potentially used with rerun_metrics - saved metrics are not modified, only the ones
# specified as rerun are
# If "none" do not compute any metrics
MOA='overwrite_all'
MO="overwrite" 
MM="missing"
NM='none'
# Set the mode for updating metrics
UPDATE_METRICS=MOA

# Metrics to force rerun
#kBET_study is done only on all_allCells
rerun_metrics=[]

# Do not run metrics for these data partitions
# all - all cells, anno - annotated cells, anno_beta -annotated beta cells
P_ALL='all'
P_ANNO='anno'
P_ANNO_BETA='anno_beta'
OMIT_PARTITIONS=[] 

# If True recompute plots even if they are already present
REPLOT=True

# Run kBET with more output
kBET_verbose=True

print('Update metrics:',UPDATE_METRICS)
print('Do not run metrics for:',OMIT_PARTITIONS)
print('Force rerun_metrics:',rerun_metrics)
print('Replot:',REPLOT)
print('kBET_verbose:',kBET_verbose)
# -

if not slurm:
    # At some point we had problems saving the data depending on the server we used,
    # so we in some cases saved the data with our helper function
    sys.path.insert(0, '/lustre/groups/ml01/workspace/karin.hrovatin/code/diabetes_analysis/')
    import helper as h

if slurm:
    TMP='/lustre/groups/ml01/workspace/karin.hrovatin/tmp/'
else:
    TMP='/tmp/'

#Paths for loading data and saving results
path_in='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/ref_combined/preprocessed/'
path_out='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/'
#Unique ID2 for reading/writing h5ad files with helper function
UID2='integration_ref_eval'+UID3
path_save=path_out+folder

# *** Load embedding

# Load non-integrated expression data
if slurm:
    adata_ref=sc.read_h5ad(path_in+'data_normalised.h5ad')
else:
    #adata_ref=h.open_h5ad(file=path_in+'data_normalised.h5ad',unique_id2=UID2)
    adata_ref=sc.read_h5ad(path_in+'data_normalised.h5ad')

# Load integrated latent data
if slurm:
    latent_adata=sc.read_h5ad(path_save+'latent.h5ad')
else:
    #latent_adata=h.open_h5ad(file=path_save+'latent.h5ad',unique_id2=UID2)
    latent_adata=sc.read_h5ad(path_save+'latent.h5ad')

# Remove any ref/nonref info from study col and index
latent_adata.obs.study=[study.replace('_ref','').replace('_nonref','') for study in latent_adata.obs.study]
latent_adata.obs_names=[idx.replace('-ref','').replace('-nonref','').replace('_ref','').replace('_nonref','')
                        for idx in latent_adata.obs_names]

# Unify latent cell types with current annotation in original data
cell_type_cols=['cell_type','cell_type_multiplet','cell_subtype','cell_subtype_multiplet']
latent_adata.obs.drop([col for col in cell_type_cols if col in latent_adata.obs.columns],axis=1,inplace=True)
for col in cell_type_cols:
    latent_adata.obs.loc[:,col]=adata_ref.obs.reindex(latent_adata.obs.index)[col]

# Fill all nan in str and categorical columns with "NA" for plotting
for col in latent_adata.obs.columns:
    if latent_adata.obs[col].isna().any() and latent_adata.obs[col].dtype.name in ['category','string']:
        if latent_adata.obs[col].dtype.name == 'category' and 'NA' not in latent_adata.obs[col].cat.categories:
            latent_adata.obs[col]=latent_adata.obs[col].cat.add_categories("NA")
        latent_adata.obs[col].fillna('NA',inplace=True)

# Add design info to column for UMAP titles
if 'design' in latent_adata.obs.columns:
    latent_adata.obs.drop('design',axis=1, inplace=True)
for study in latent_adata.obs.study.unique():
    metadata=pd.read_excel('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/scRNA-seq_pancreas_metadata.xlsx',
                          sheet_name=study.replace('_ref','').replace('_nonref','')) 
    obs_sub=latent_adata.obs.query('study ==@study')
    samples=obs_sub.file.unique()
    value_map={sample:metadata.query('sample_name =="'+sample+'"')['design'].values[0] for sample in samples}
    latent_adata.obs.loc[obs_sub.index,'design']=obs_sub.file.map(value_map)
# Make col with study and design info
latent_adata.obs['study_sample_design']=[study_sample+'_'+str(design) for study_sample, design in 
                                         zip(latent_adata.obs.study_sample,latent_adata.obs.design)]

# *** Evaluate result

# ** Plots

# Plot UMAP
replot = REPLOT
if not replot:
    if not os.path.isfile(path_save+'umap_latent.png'):
        replot=True
if replot:    
    print('Plotting UMAP metadata')
    random_indices=np.random.permutation(list(range(latent_adata.shape[0])))
    sc._settings.ScanpyConfig.figdir=Path(path_save)
    rcParams['figure.figsize']= (10,10)
    sc.pl.umap(latent_adata[random_indices,:],color=['study_sample_design','study','cell_subtype','cell_type_multiplet'],
               #wspace=0.6,
               ncols=1,hspace=0.8,
               size=10,save='_latent.png',show=False)

# Plot denisty
replot = REPLOT
if not replot:
    if not os.path.isfile(path_save+'umap_density_study_sample_design_latent_density.png'):
        replot=True
if replot:  
    print("Plotting density")
    design_order=['mRFP','mTmG','mGFP',
              'head_Fltp-','tail_Fltp-', 'head_Fltp+', 'tail_Fltp+',
              'IRE1alphafl/fl','IRE1alphabeta-/-', 
              '8w','14w', '16w',
              'DMSO_r1','DMSO_r2', 'DMSO_r3','GABA_r1','GABA_r2', 'GABA_r3',
                   'A1_r1','A1_r2','A1_r3','A10_r1','A10_r2', 'A10_r3',  'FOXO_r1', 'FOXO_r2', 'FOXO_r3', 
              'E12.5','E13.5','E14.5', 'E15.5', 
              'chow_WT','sham_Lepr-/-','PF_Lepr-/-','VSG_Lepr-/-',   
              'control','STZ', 'STZ_GLP-1','STZ_estrogen', 'STZ_GLP-1_estrogen',
                  'STZ_insulin','STZ_GLP-1_estrogen+insulin' 
            ]
    # Prepare figure grid
    fig_rows=latent_adata.obs.study.unique().shape[0]
    fig_cols=max(
        [latent_adata.obs.query('study ==@study').file.unique().shape[0] 
         for study in latent_adata.obs.study.unique()])
    rcParams['figure.figsize']= (6*fig_cols,6*fig_rows)
    fig,axs=plt.subplots(fig_rows,fig_cols)
    # Calculagte density
    for row_idx,study in enumerate(latent_adata.obs.study.unique()):
        designs=latent_adata.obs.query('study ==@study')[['study_sample_design','design']].drop_duplicates()
        designs.design=pd.Categorical(designs.design, 
                      categories=[design for design in design_order if design in list(designs.design.values)],
                      ordered=True)
        for col_idx,sample in enumerate(designs.sort_values('design')['study_sample_design'].values):
            subset=latent_adata.obs.study_sample_design==sample
            adata_sub=latent_adata[subset,:]
            print(sample,adata_sub.shape)
            sc.tl.embedding_density(adata_sub)
            sc.pl.umap(latent_adata,ax=axs[row_idx,col_idx],s=10)
            sc.pl.embedding_density(adata_sub,ax=axs[row_idx,col_idx],title=sample) 
            axs[row_idx,col_idx].set_xlabel('')
            if col_idx==0:
                axs[row_idx,col_idx].set_ylabel(study, rotation=90, fontsize=20)
            else:
                axs[row_idx,col_idx].set_ylabel('') 
    fig.tight_layout()
    plt.savefig(path_save+'umap_density_study_sample_design_latent_density.png')


# ** Run scIB
if UPDATE_METRICS !=NM:

    # Required for scIB
    latent_adata.obsm['X_emb']=latent_adata.X

    # ** Compute batch metircs on all cells (all cell types, including without cell types)
    #
    # kBET is here computed on clustering optimised based on reference cell types, as many cells are missing information (NA/multiplet)
    if P_ALL not in OMIT_PARTITIONS:
        print('Computing metrics on all cells (including nonannotated)')

        # +
        # First figure out which metrics must be calculated - if all are False do not load data
        metric_file=path_save+"integration_metrics_allCells.pkl"
        if os.path.isfile(metric_file) and UPDATE_METRICS != MOA:
            metrics = pickle.load( open( metric_file, "rb" ) )
        else:
            metrics={}

        metrics_study={metric.replace('_study',''):val for metric,val in metrics.items() if '_study' in metric}  
        metrics={metric:val for metric,val in metrics.items() if '_study' not in metric} 

        all_metrics=['PC_regression','ASW_batch','kBET','graph_connectivity','graph_iLISI',
                    'graph_cLISI','NMI', 'ARI','ASW_cell_type','isolated_label_F1','isolated_label_ASW']

        # Should any metrics be omitted - if MM and metric is present and not nan
        # Assumes that defaults for all metrics are True
        metrics_omit=dict()
        if UPDATE_METRICS in [MM]:
            for metric,value in metrics.items():
                if not np.isnan(value):
                    metrics_omit[metric]=False 
        metrics_omit_study=dict()
        if UPDATE_METRICS in [MM]:
            for metric,value in metrics_study.items():
                if not np.isnan(value):
                    metrics_omit_study[metric]=False 

        # Force compute (must be before ommiting cell type based metrics as those can not be computed here)
        for metric in rerun_metrics:
            if '_study' not in metric:
                metrics_omit[metric]=True
            else:
                metrics_omit_study[metric.replace('_study','')]=True

        # Omit bio metrics - require annotation, do not omit kBET as it is performed on clusters
        # Also do not omit moransi_conservation as it does not need cell labels
        for metric in ['graph_cLISI', 'NMI', 'ARI', 
                       'ASW_cell_type', 'isolated_label_F1', 'isolated_label_ASW',
                      ]:
            metrics_omit[metric]=False 
            metrics_omit_study[metric]=False 
        # For batch study only kBET is computed
        metrics_batch_study=['kBET']
        for metric in all_metrics:
            if metric not in metrics_batch_study:
                metrics_omit_study[metric]=False

        # Check if there are any metrics that still need to be calculated, otherwise do not load data
        metrics_to_compute=[metric for metric in all_metrics if metrics_omit.get(metric,True) ]
        metrics_to_compute_study=[metric for metric in all_metrics if metrics_omit_study.get(metric,True) ]

        print('Metrics to compute:',metrics_omit,'; batch study:',metrics_omit_study)
        if len(metrics_to_compute)>0 or len(metrics_to_compute_study)>0: 
            # *** New data
            data=[('Fltp_2y','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islets_aged_fltp_iCre/rev6/'),
                  ('Fltp_adult','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islet_fltp_headtail/rev4/'),
                  ('Fltp_P16','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/salinno_project/rev4/'),
                  ('NOD','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE144471/'),
                  ('NOD_elimination','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE117770/'),
                  ('spikein_drug','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE142465/'),
                  ('embryo','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE132188/rev7/')
                 ]
            if split_refnonref:
                data.extend([
                  ('VSG_ref','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/VSG_PF_WT_cohort/rev7/ref/'),
                  ('STZ_ref','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islet_glpest_lickert/rev7/ref/'),
                  ('VSG_nonref','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/VSG_PF_WT_cohort/rev7/nonref/'),
                  ('STZ_nonref','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islet_glpest_lickert/rev7/nonref/')
                ])
            else:
                data.extend([
                  ('VSG','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/VSG_PF_WT_cohort/rev7/'),
                  ('STZ','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islet_glpest_lickert/rev7/')
                ])
            adatas=[]
            for study,path in data:
                print(study)
                #Load data
                if slurm:
                    adata=sc.read_h5ad(path+raw_file)
                else:
                    adata=h.open_h5ad(file=path+raw_file,unique_id2=UID2)
                print(adata.shape)            
                adatas.append(adata)

            # Combine datasets    
            adata_full = ann.AnnData.concatenate( *adatas,  batch_categories = [d[0] for d in data ]).copy()
            # Edit obs_names to match reference
            # Only _ref, _nonref (not "-") is needed here as "-study_ref/nonref" is added to obs names
            adata_full.obs_names=[name.replace('_ref','').replace('_nonref','') for name in adata_full.obs_names]

            # Adata full should contain all latent_adata cells
            adata_full=adata_full[latent_adata.obs_names,:]

            # Replace NA/multiplet cell types with np.nan for kBET opt clustering NMI calculation
            latent_adata_temp=latent_adata.copy()
            na_cells=latent_adata_temp.obs_names[latent_adata_temp.obs.cell_type_multiplet.str.contains('NA|multiplet')]
            print('N cells with NA or multiplet annotation:',len(na_cells))
            latent_adata_temp.obs.loc[na_cells,'cell_type_multiplet']=np.nan

            # Run this on general cell types (not subtype which has beta subtypes) so that it is less biased towards beta cells and 
            # less sprone to nioise due to challeging annotation of subtypes 
            compute_metrics(adata_full=adata_full,latent_adata=latent_adata_temp,metrics=metrics,
                            cell_type_col='cell_type_multiplet', batch_col='study_sample',
                            path_save=path_save,TMP=TMP,
                            batch_anno_clusters=True,
                            task_name='all_allCells',
                            **metrics_omit)
            compute_metrics(adata_full=adata_full,latent_adata=latent_adata_temp,metrics=metrics_study,
                            cell_type_col='cell_type_multiplet', batch_col='study', 
                            path_save=path_save,TMP=TMP,
                            batch_anno_clusters=True,
                            task_name='all_allCells_study',
                            **metrics_omit_study)
            for metric,val in metrics_study.items():
                metrics[metric+'_study']=val

            # Save metrics
            pickle.dump(metrics, open( metric_file, "wb" ) )

            print('Metrics:',metrics)

            # remove data which will not longer be used
            del adata_full
            del latent_adata_temp

    # ** Compute bio and batch metrics on ref cells that are not NA/multiplet
    if P_ANNO not in OMIT_PARTITIONS:
        print('Computing metrics on all cell types (excluding nonannotated/multiplet)')

        # Keep only cells present in reference combined data

        n_cells_all=latent_adata.shape[0]
        latent_adata=latent_adata[[obs for obs in adata_ref.obs_names if obs in latent_adata.obs_names],:]
        if n_cells_all > latent_adata.shape[0]:
            warnings.warn('Only cells present in reference will be used for calculation of integration scores.')
            print('N all cels %i, N cells for score evaluation (shared with ref) %i'%(n_cells_all,latent_adata.shape[0]))

        # Remove cells that contain NA or multiplet in their name (in cell_type_multiplet) from the evaluation.

        # Remove cells not ok for evaluation
        eval_cells=~latent_adata.obs.cell_type_multiplet.str.contains('NA|multiplet')
        print('All cells:',latent_adata.shape[0],'To keep for scIB (non NA/multiplet):', eval_cells.sum())
        latent_adata=latent_adata[eval_cells,:]
        # Subset all data
        adata_ref=adata_ref[latent_adata.obs_names,:]

        metric_file=path_save+"integration_metrics.pkl"
        if os.path.isfile(metric_file) and UPDATE_METRICS != MOA:
            metrics = pickle.load( open( metric_file, "rb" ) )
        else:
            metrics={}

        # Should any metrics be omitted - if MM and metric is present and not nan
        # Assumes that defaults for all metrics are True
        metrics_omit=dict()
        if UPDATE_METRICS in [MM]:
            for metric,value in metrics.items():
                if not np.isnan(value):
                    metrics_omit[metric]=False 

        # Force compute
        for metric in rerun_metrics:
            if '_study' not in metric:
                metrics_omit[metric]=True
        print('Metrics to compute:',metrics_omit)

        # Run this on general cell types (not  beta subtypes) so that it is less biased towards beta cells and 
        # less sprone to nioise due to challenging annotation of subtypes    
        compute_metrics(adata_full=adata_ref,latent_adata=latent_adata,metrics=metrics,
                        cell_type_col='cell_type', batch_col='study_sample',
                        path_save=path_save,TMP=TMP,
                        task_name='all_annotatedCells',
                        **metrics_omit)

        # Save metrics
        pickle.dump(metrics, open( metric_file, "wb" ) )

        print('Metrics:',metrics)

    # Compute metrics on beta cells
    if P_ANNO_BETA not in OMIT_PARTITIONS:
        print('Computing metrics on beta cell subtypes (excluding nonannotated/multiplet)')

        excluded_beta=[]
        selected_beta=[ct for ct in latent_adata.obs.cell_subtype.unique() if 'beta' in ct and ct not in excluded_beta]
        print('Selected beta:',selected_beta,'\nExcluded beta:',excluded_beta,'\n')
        latent_adata_beta=latent_adata[latent_adata.obs.cell_subtype.isin(selected_beta),:].copy()
        adata_ref_beta=adata_ref[latent_adata_beta.obs_names,:]
        print('Beta cells for scIB:',latent_adata_beta.shape[0])

        metric_file=path_save+"integration_metrics_beta.pkl"
        if os.path.isfile(metric_file) and UPDATE_METRICS != MOA:
            metrics_beta = pickle.load( open( metric_file, "rb" ) )
        else:
            metrics_beta={}

        # Should any metrics be omitted - if MM and metric is present and not nan
        # Assumes that defaults for all metrics are True
        metrics_omit=dict()
        if UPDATE_METRICS in [MM]:
            for metric,value in metrics_beta.items():
                if not np.isnan(value):
                    metrics_omit[metric]=False 

        # Force compute
        for metric in rerun_metrics:
            if '_study' not in metric:
                metrics_omit[metric]=True

        print('Metrics to compute:',metrics_omit)
        compute_metrics(adata_full=adata_ref_beta,
                        latent_adata=latent_adata_beta,metrics=metrics_beta,
                        cell_type_col='cell_subtype',batch_col='study_sample',
                        path_save=path_save,TMP=TMP,
                        task_name='beta_annotatedCells',**metrics_omit)

        # Save metrics
        pickle.dump(metrics_beta, open( metric_file, "wb" ) )

        print('Metrics beta:',metrics_beta)

print('Time:',time.time()-t0)
