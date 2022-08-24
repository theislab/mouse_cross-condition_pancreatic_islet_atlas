#!/bin/bash

# Script for running UMAp and clustering on raw filtered data
#$1 - UID
#$2 - file extention, e.g. feature_bc_matrix.h5ad
#$3 - dir


#SBATCH -o /home/icb/%u/slurm_pp_rawfiltered_%j.job
#SBATCH -e /home/icb/%u/slurm_pp_rawfiltered_%j.job
#SBATCH -J pp_rawfiltered
#SBATCH -c 1 #CPU cores required
#SBATCH --mem=10G #Memory required
#SBATCH -t 2:00:00 #Job runtime
#SBATCH --nice=10000 #Manual priority. Do not change this.
#SBATCH -p cpu_p # Correct partition 

source $HOME/.bashrc
conda activate rpy2_3
echo "UID $1 file extention $2 dir $3"
python $WSCL/mm_pancreas_atlas_rep/code/preprocessing/4-3-1_analyse_filtered_gene_matrices.py "$1" "$2" "$3"

