#!/bin/bash

# Sbatch for running SoupX correction script

#SBATCH -o /lustre/groups/ml01/workspace/karin.hrovatin/slurm_SoupX_%j.job
#SBATCH -e /lustre/groups/ml01/workspace/karin.hrovatin/slurm_SoupX_%j.job
#SBATCH -J SoupX
#SBATCH -c 1 #CPU cores required
#SBATCH --mem=20G #Memory required
#SBATCH -t 5:00:00 #Job runtime
#SBATCH --nice=10000 #Manual priority. Do not change this.
#SBATCH -p icb_cpu # Correct partition for CPU

source $HOME/.bashrc
conda activate rpy2_3
echo "file_ext:" "$1" "folder:" "$2" 
Rscript "$WSCL"/mm_pancreas_atlas_rep/code/preprocessing/4-3-2_SoupX.R  "$1" "$2"