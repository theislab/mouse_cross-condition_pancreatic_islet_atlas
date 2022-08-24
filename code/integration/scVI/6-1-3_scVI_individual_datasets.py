# # Atlas integration with scVI

import time
t1=time.time()

import logging
import os
import scanpy as sc
from hyperopt import hp
from scvi.inference.autotune import auto_tune_scvi_model
#from scvi.dataset.anndataset import AnnDatasetFromAnnData
import scvi.data
import datetime
import sys
import argparse
import anndata as ann
import pandas as pd
import scIB as scib

logger = logging.getLogger("scvi.inference.autotune")
logger.setLevel(logging.WARNING)

#Paths for loading data and saving results
path_out='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/scVI/runs/'

# +
#*** Prepare args

# Sepcify args
parser = argparse.ArgumentParser()
parser.add_argument('--hvg_n' , required=True,default='2000') # int or all
parser.add_argument('--supervised' , required=True,default=0) # 0 or 1
parser.add_argument('--metric' , required=False, default='mll') #mll, ebm not working
parser.add_argument('--test_mode', required=False, default='0') # 0 or 1
parser.add_argument('--subset_beta', required=False, default='0') # 0 or 1
parser.add_argument('--slurm', required=False, default='1') # 0 or 1
parser.add_argument('--use_cuda', required=False, default='1') # 0 or 1
parser.add_argument('--file', required=True)

# Read args
args = parser.parse_args()
print(args)
test_mode=bool(int(args.test_mode))
hvg_n=str(args.hvg_n)
metric_key=str(args.metric)
supervised=bool(int(args.supervised))
subset_beta=bool(int(args.subset_beta))
slurm=bool(int(args.slurm))
use_cuda=bool(int(args.use_cuda))
input_file_name=str(args.file)

print('test_mode:',test_mode,'hvg_n:',hvg_n,'metric_key:',metric_key,
      'supervised:',supervised,'subset_beta',subset_beta,'input_file_name:',input_file_name)


# -

if False:
    test_mode=True
    hvg_n='2000'
    metric_key='mll'
    supervised=False
    subset_beta=False
    slurm=False
    UID2='scVI_hyperopt_individual'
    use_cuda=False
    input_file_name='data_normlisedForIntegration.h5ad'

if supervised:
    raise ValueError('Supervused not implemented in this file.')

if not slurm:
    import sys  
    sys.path.insert(0, '/lustre/groups/ml01/workspace/karin.hrovatin/code/diabetes_analysis/')
    import helper as h

metric_name={'mll':'marginal_ll','ebm':"entropy_batch_mixing"}[metric_key]

# *** Prepare data

# +
# Copiing to tmp does not work on slurm

data=[('Fltp_2y','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islets_aged_fltp_iCre/rev6/'),
      ('Fltp_adult','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islet_fltp_headtail/rev4/'),
      ('Fltp_P16','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/salinno_project/rev4/'),
      ('NOD','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE144471/'),
      ('NOD_elimination','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE117770/'),
      ('spikein_drug','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE142465/'),
      ('embryo','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE132188/rev7/'),
      ('VSG','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/VSG_PF_WT_cohort/rev7/'),
      ('STZ','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islet_glpest_lickert/rev7/')]
adatas=[]
for study,path in data:
    print(study)
    #Load data
    if slurm:
        adata=sc.read_h5ad(path+input_file_name)
    else:
        adata=h.open_h5ad(file=path+input_file_name,unique_id2=UID2)
    print(adata.shape)            
    adatas.append(adata)
    
# Combine datasets    
data = ann.AnnData.concatenate( *adatas,  batch_categories = [d[0] for d in data ]).copy()
# Edit obs_names to match reference
data.obs_names=[name.replace('_ref','').replace('_nonref','') for name in data.obs_names]
# -

if subset_beta:
    selected_beta=['beta', 'beta_proliferative','beta_subpopulation_STZref', 'beta_ins_low']
    data=data[data.obs.cell_type.isin(selected_beta),:].copy()
    print('Subseted to beta, shape',data.shape,'beta cells',selected_beta)

# +
# Select genes before using only counts
# Compute HVGs on combined dataset
if hvg_n=='2000':
    # Compute HVG across batches (samples) using scIB function
    data.obs['study_sample'] = data.obs['study_sample'].astype('category')
    hvgs=scib.preprocessing.hvg_batch(data, batch_key='study_sample', target_genes=2000, flavor='cell_ranger')
    # Add HVGs to adata
    hvgs=pd.DataFrame([True]*len(hvgs),index=hvgs,columns=['highly_variable'])
    hvgs=hvgs.reindex(data.var_names,copy=False)
    hvgs=hvgs.fillna(False)
    data.var['highly_variable']=hvgs.values
    print('Number of highly variable genes: {:d}'.format(data.var['highly_variable'].sum()))
else:
    raise ValueError('HVG mode in params not recongnised')
    
# Select HVGs
data=data[:,data.var['highly_variable']]
# -

# Replace X with counts
data.X=data.raw.to_adata()[data.obs_names,data.var_names].X.toarray()
if 'counts' in data.layers.keys():
    del data.layers['counts']
del data.raw

print('Data shape:',data.shape)

# +
# Add batch info for scVI
#batches=data.obs.study_sample.unique()
#data.obs['batch_int']=data.obs.study_sample.replace(dict(zip(batches,range(len(batches)))))
# -

# If supervised (with cell types) remove NA/multiplet cell types and add cell type information
if supervised:
    eval_cells=~data.obs.cell_type_multiplet.str.contains('NA|multiplet')
    print('All cells:',data.shape[0],'Not NA/multiplet:', eval_cells.sum())
    data=data[eval_cells,:]

# Make scVI data object
if supervised:
    # Add cell type info if supervised
    scvi.data.setup_anndata(data, batch_key="study_sample",labels_key='cell_type')
    #data=AnnDatasetFromAnnData(ad = data,batch_label='batch_int',ctype_label ='cell_type')
else:
    scvi.data.setup_anndata(data, batch_key="study_sample")
    #data=AnnDatasetFromAnnData(ad = data,batch_label='batch_int')


# *** Set parameters

def if_not_test_else(x, y):
    if not test_mode:
        return x
    else:
        return y

# +
#n_epochs = if_not_test_else(int(400*20000/data.shape[0]), 1)
#reserve_timeout = if_not_test_else(180, 5)
#fmin_timeout = if_not_test_else(300, 10)
# -

# *** Run scVI

# Run out dir
path_out_run=path_out+str(datetime.datetime.now().timestamp())+'_SB'+str(int(subset_beta))+'/'
print('Save to dir: '+path_out_run)
#os.mkdir(path_out_run)

# Execute
# Params based on ref hyperopt eval
model = scvi.model.SCVI(data,n_hidden=256,n_latent=20,n_layers=3,dropout_rate=0.1,
                        dispersion='gene-batch',gene_likelihood='zinb',
                       use_cuda=use_cuda)
model.train(lr=0.001)
model.save(path_out_run)

# Get latent data
latent_adata = sc.AnnData(X=model.get_latent_representation(), 
           obs=data.obs[['file', 'study', 'study_sample', 'reference']])
sc.pp.neighbors(latent_adata,n_pcs=0)
sc.tl.umap(latent_adata)

# Save latent data
if not slurm:
    h.save_h5ad(adata=latent_adata,file=path_out_run+'latent.h5ad',unique_id2=UID2)
else:
    latent_adata.write(path_out_run+'latent.h5ad')

# Save time and params not passed to hyperopt/scVI
with open(path_out_run+'scvi_autotune_logfile.txt','a') as f:
    f.write('Time script:'+str(time.time()-t1)+'\n')
    f.write('N genes:'+str(hvg_n)+'\n')
    f.write('Metric:'+metric_name+'\n')
    f.write('Supervised:'+str(supervised)+'\n')
    f.write('Subset beta:'+str(subset_beta)+'\n')
    f.write('Input file name:'+input_file_name+'\n')


