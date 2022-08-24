#!/bin/bash

# Sbatch for running emptyDrops

#SBATCH -o /lustre/groups/ml01/workspace/karin.hrovatin/slurm_emptyDrops_%j.job
#SBATCH -e /lustre/groups/ml01/workspace/karin.hrovatin/slurm_emptyDrops_%j.job
#SBATCH -J emptyDrops
#SBATCH -c 4 #CPU cores required
#SBATCH --mem=10G #Memory required
#SBATCH -t 5:00:00 #Job runtime
#SBATCH --nice=10000 #Manual priority. Do not change this.
#SBATCH -p cpu_p # Correct partition for CPU
#SBATCH --exclude=ibis-ceph-0[02-05,08-19],ibis216-010-0[20-37,64],icb-rsrv[05-06,08] 

source $HOME/.bashrc
conda activate rpy2_3
echo "folder:" "$1" "file_ext:" "$2" 
python $WSCL/mm_pancreas_atlas_rep/code/preprocessing/1-1_ambient_scores.py  "$1" "$2"