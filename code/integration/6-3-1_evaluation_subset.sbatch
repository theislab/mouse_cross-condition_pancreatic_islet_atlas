#!/bin/bash

#SBATCH -o /lustre/groups/ml01/workspace/karin.hrovatin/slurm_integrationEval_%j.job
#SBATCH -e /lustre/groups/ml01/workspace/karin.hrovatin/slurm_integrationEval_%j.job
#SBATCH -J integrationEval
#SBATCH -c 4 #CPU cores required 4
#SBATCH --mem=150G #Memory required, TODO put back to 290/150
#SBATCH -t 14:00:00 #Job runtime 14:00:00
#SBATCH --nice=10000 #Manual priority. Do not change this.
#SBATCH -p cpu_p
#SBATCH --exclude=ibis-ceph-014,ibis216-010-037,ibis216-224-010,ibis216-010-[035,051,064],ibis216-224-011,ibis-ceph-015,ibis-ceph-[002-005,008-013,016-019],ibis216-010-[068-071],icb-iris,icb-mona

source $HOME/.bashrc
conda activate rpy2_3
# $1 - UID3 for io adata
# $2 - integration dir names (without path)  
# $3 - raw_file - counts used for integration evaluation
# $4 - split_refnonref - was VSG/STZ split to ref/nonref in integration
# $5 - full path to cell names of the cell subset
# $6 - folder name in which to save results, placed within run folder
# $7 - obs col used as cell group
echo $1 $2 $3 $4 $5 $6 $7
python "$WSC""diabetes_analysis/integration/6-3-1_evaluation_subset_script.py" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "1"