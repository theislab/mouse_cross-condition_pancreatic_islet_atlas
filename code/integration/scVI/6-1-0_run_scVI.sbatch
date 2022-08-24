#!/bin/bash

#SBATCH -o /home/icb/%u/slurm_scVI_%j.job
#SBATCH -e /home/icb/%u/slurm_scVI_%j.job
#SBATCH -J scVI
#SBATCH -c 1 #CPU cores required
#SBATCH --mem=200G #Memory required
#SBATCH -t 48:00:00 #Job runtime
#SBATCH --nice=10000 #Manual priority. Do not change this.
#SBATCH --gres=gpu:1 # N GPUs
#SBATCH --exclude=icb-gpusrv0[1-2] # GPU server exclude interactive nodes
#SBATCH -p gpu_p # Correct partition for GPU

source $HOME/.bashrc
conda activate scvi-env

# Which option to run - un/re-commend a single option to be run

# First step:
# Hyperopt optimisation on reference data (subset of atlas)
# This was also tested with 5000 HVGs or all genes, but based on visual evaluation of UMAPs we decided to use the 2000 HVGs for further evaluation
#python $WSCL/mm_pancreas_atlas_rep/code/integration/scVI/6-1-1_ref_hyperopt_script.py --hvg_n 2000 --metric mll --supervised 0 --subset_beta 0 

# Second step
# Atlas runs

# No ambient correction - only mask top ambient
#python $WSCL/mm_pancreas_atlas_rep/code/integration/scVI/6-1-3_scVI_individual_datasets.py --hvg_n 2000 --metric mll --supervised 0 --subset_beta 0 --file data_normlisedForIntegration.h5ad 

# DecontX ambient corrected
#python $WSCL/mm_pancreas_atlas_rep/code/integration/scVI/6-1-3_scVI_individual_datasets.py --hvg_n 2000 --metric mll --supervised 0 --subset_beta 0 --file decontX/data_decontX_normlisedForIntegration.h5ad 
# DecontX with fixed correction param
#python $WSCL/mm_pancreas_atlas_rep/code/integration/scVI/6-1-3_scVI_individual_datasets.py --hvg_n 2000 --metric mll --supervised 0 --subset_beta 0 --file decontX_fixed/data_decontX_normlisedForIntegration.h5ad

# Different SoupX options
#python $WSCL/mm_pancreas_atlas_rep/code/integration/scVI/6-1-3_scVI_individual_datasets.py --hvg_n 2000 --metric mll --supervised 0 --subset_beta 0 --file data_SoupX_rhoadd0_normlisedForIntegration.h5ad
#python $WSCL/mm_pancreas_atlas_rep/code/integration/scVI/6-1-3_scVI_individual_datasets.py --hvg_n 2000 --metric mll --supervised 0 --subset_beta 0 --file data_SoupX_rhoadd005_normlisedForIntegration.h5ad
python $WSCL/mm_pancreas_atlas_rep/code/integration/scVI/6-1-3_scVI_individual_datasets.py --hvg_n 2000 --metric mll --supervised 0 --subset_beta 0 --file data_SoupX_rhoadd01_normlisedForIntegration.h5ad