# # Find scVI hyperparams with hyperopt on subset of atlas data

import time
t1=time.time()

import logging
import os
import scanpy as sc
from hyperopt import hp
from scvi.inference.autotune import auto_tune_scvi_model
from scvi.dataset.anndataset import AnnDatasetFromAnnData
import datetime
import sys
import argparse

logger = logging.getLogger("scvi.inference.autotune")
logger.setLevel(logging.WARNING)

#Paths for loading data and saving results
path_in='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/ref_combined/preprocessed/'
path_out='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/ref_combined/scVI/'

# +
#*** Prepare args

# Sepcify args
parser = argparse.ArgumentParser()
parser.add_argument('--hvg_n' , required=True) # int or all
parser.add_argument('--supervised' , required=True) # 0 or 1
parser.add_argument('--max_evals' , required=False, default='100')
parser.add_argument('--metric' , required=False, default='mll') #mll, ebm not working
parser.add_argument('--test_mode', required=False, default='0') # 0 or 1
parser.add_argument('--subset_beta', required=False, default='0') # 0 or 1

# Read args
args = parser.parse_args()
print(args)
test_mode=bool(int(args.test_mode))
hvg_n=str(args.hvg_n)
metric_key=str(args.metric)
max_evals=int(args.max_evals)
supervised=bool(int(args.supervised))
subset_beta=bool(int(args.subset_beta))
print('test_mode:',test_mode,'hvg_n:',hvg_n,'metric_key:',metric_key,'max_evals:',
      max_evals,'supervised:',supervised,'subset_beta',subset_beta)

# Old way using sys.argv
#print("sys.argv", sys.argv)
#test_mode=bool(int(sys.argv[1]))
# Number for which HVGs were calculated or 'all'
#hvg_n=str(sys.argv[2])
#metric_key=str(sys.argv[3])
#if len(sys.argv) >= 5:
#    max_evals = int(sys.argv[4])
#else:
#    max_evals = 100
# -

metric_name={'mll':'marginal_ll','ebm':"entropy_batch_mixing"}[metric_key]

# *** Prepare data

# Copiing to tmp does not work on slurm
data=sc.read_h5ad(path_in+'data_normalised.h5ad')
if subset_beta:
    selected_beta=['beta', 'beta_proliferative','beta_subpopulation_STZref', 'beta_ins_low']
    data=data[data.obs.cell_type.isin(selected_beta),:].copy()
    print('Subseted to beta, shape',data.shape)

# Replace X with counts
data.X=data.layers['counts']
del data.layers['counts']
del data.raw

# Select genes
# Select HVGs if N of genes is smaller than N of all genes
data.var.rename(columns={'highly_variable': 'highly_variable_2000'}, inplace=True)
if hvg_n != 'all':
    data=data[:,data.var['highly_variable_'+hvg_n]]

# Add batch info for scVI
batches=data.obs.study_sample.unique()
data.obs['batch_int']=data.obs.study_sample.replace(dict(zip(batches,range(len(batches)))))

# If supervised (with cell types) remove NA/multiplet cell types and add cell type information
if supervised:
    eval_cells=~data.obs.cell_type_multiplet.str.contains('NA|multiplet')
    print('All cells:',data.shape[0],'Not NA/multiplet:', eval_cells.sum())
    data=data[eval_cells,:]

# Make scVI data object
if supervised:
    # Add cell type info if supervised
    data=AnnDatasetFromAnnData(ad = data,batch_label='batch_int',ctype_label ='cell_type')
else:
    data=AnnDatasetFromAnnData(ad = data,batch_label='batch_int')


# *** Set parameters

def if_not_test_else(x, y):
    if not test_mode:
        return x
    else:
        return y

n_epochs = if_not_test_else(int(400*20000/data.nb_cells), 1)
max_evals = if_not_test_else(max_evals, 1)
reserve_timeout = if_not_test_else(180, 5)
fmin_timeout = if_not_test_else(300, 10)
use_cuda = True

if supervised:
    model_specific_kwargs={'n_labels':data.n_labels}
    dispersions=["gene", "gene-batch",'gene-label']
else:
    model_specific_kwargs=None
    dispersions=["gene", "gene-batch"]

# Search space
# See https://github.com/YosefLab/scVI/blob/59df5417eaa7815a68041653f13ab30fc3302aba/scvi/inference/autotune.py#L375
space = {
            "model_tunable_kwargs": {
                "n_latent": 5 + hp.randint("n_latent", 16),  # [5, 20]
                "n_hidden": hp.choice("n_hidden", [64, 128, 256]),
                "n_layers": 1 + hp.randint("n_layers", 5),
                "dropout_rate": hp.choice("dropout_rate", [0.1, 0.3, 0.5, 0.7]),
                "reconstruction_loss": hp.choice("reconstruction_loss", ["zinb", "nb"]),
                "dispersion": hp.choice("dispersion", dispersions)
            },
            "train_func_tunable_kwargs": {
                "lr": hp.choice("lr", [0.01, 0.005, 0.001, 0.0005, 0.0001])
            },
        }

# *** Run hyperopt

# Run out dir
path_out_run=path_out+'hyperopt/'+str(datetime.datetime.now().timestamp())+'_SB'+str(int(subset_beta))+'/'
print('Save to dir: '+path_out_run)
os.mkdir(path_out_run)
# Execute
best_trainer, trials = auto_tune_scvi_model(
    gene_dataset=data,
    metric_name=metric_name,
    trainer_specific_kwargs ={'train_size':0.90,'use_cuda':use_cuda,
                              'n_epochs_kl_warmup':None,'n_iter_kl_warmup':int(128*5000/400)},
    train_func_specific_kwargs={"n_epochs": n_epochs},
    model_specific_kwargs=model_specific_kwargs,
    space=space,
    max_evals=max_evals,
    save_path=path_out_run,
    use_batches=True,
    # MongoDB is not availiable - cant use paralel
    parallel=False,
    #n_workers_per_gpu=4,
    reserve_timeout=reserve_timeout,
    fmin_timeout=fmin_timeout,
    exp_key="ref",
)

# Save time and params not passed to hyperopt/scVI
with open(path_out_run+'scvi_autotune_logfile.txt','a') as f:
    f.write('Time script:'+str(time.time()-t1)+'\n')
    f.write('N genes:'+str(hvg_n)+'\n')
    f.write('Metric:'+metric_name+'\n')
    f.write('Supervised:'+str(supervised)+'\n')
    f.write('Subset beta:'+str(subset_beta)+'\n')
