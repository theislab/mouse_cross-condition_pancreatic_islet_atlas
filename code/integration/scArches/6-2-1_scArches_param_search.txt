# Specify parameter combinations to run with scArches
# The latest version of parameter file contains only the last set of params that were tested
# for integration optimisation; however, we tested more in earlier runs
# Summary of other parameters we tried out: 
# - loss: nb, mse, sse
# - with or without MMD
# - architectures with 128 nodes in 2-4 layers
# - alpha from 0.0001 to 1
# - n epochs from 100 to 300
# - different HVG selections: using sample as batch key (N 2000 or 5000) or 
#   selecting HVGs per sample on whole sample and within subclusters, combined as union of 
#   approximately 2000 top HVGs across clusters and samples
i=0
while read -r line
    do  read hvg_n z_dimension architecture beta alpha loss_fn n_epochs batch_size <<<$line
    i=$((i+1))
    nohup python code/diabetes_analysis/integration/scArches/6-2-1_ref_scArches_script.py $i $hvg_n $z_dimension $architecture $beta $alpha $loss_fn $n_epochs $batch_size 0 > r$i.out 2> r$i.err &
done <<<  $(tail -n+2 data/pancreas/scRNA/ref_combined/scArches/scArches_parameters_unix.tsv)
