#!/bin/bash

# Script for running cellbender
#$1 - folder with raw data, new data will be saved in subfolder cellbender
#$2 - N expected cells for cell bender
#$3 - N droplets to use for cell bender
#$3 - ignore droplets with lower count for cell bender
#$4 - learning rate for cell bender
#$5 - FPR for cell bender


#SBATCH -o /home/icb/%u/slurm_cellbender_%j.job
#SBATCH -e /home/icb/%u/slurm_cellbender_%j.job
#SBATCH -J cellbender
#SBATCH -c 1 #CPU cores required
#SBATCH --mem=20G #Memory required
#SBATCH -t 10:00:00 #Job runtime
#SBATCH --nice=10000 #Manual priority. Do not change this.
#SBATCH --gres=gpu:1 # N GPUs
#SBATCH --exclude=icb-gpusrv0[1-2] # GPU server exclude interactive nodes
#SBATCH -p icb_gpu # Correct partition for GPU
#SBATCH --exclude=ibis-ceph-0[02-05,08-19],ibis216-010-0[20-37,64],icb-rsrv[05-06,08] #Required for tf as these nodes can not run it

source $HOME/.bashrc
conda activate cellbender
fprs=${6//,/ }
echo "folder $1 cells $2 droplets $3 min_count $4 learning_rate $5 fprs $fprs"

cellbender remove-background \
--input "$1"cellbender/ \
--output "$1"cellbender/cellbender.h5 \
--cuda \
--expected-cells $2 \
--total-droplets-included $3 \
--low-count-threshold $4 \
--fpr 0.01 \
--epochs 150 \
--learning-rate $5 \
--fpr $fprs