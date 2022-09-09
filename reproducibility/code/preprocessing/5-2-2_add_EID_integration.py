# # Add EID info
# Add EID info to adatas that have gene symbols as var_names. EID info is read from matched CellRanger files (per study).

import scanpy as sc
import pandas as pd
import argparse
import glob
import anndata as ann

# +
parser = argparse.ArgumentParser()
parser.add_argument('-f','--file', metavar='N',
                    help='File to which to add EIDs')
parser.add_argument('-fe','--file_eid', metavar='N', 
                    help='Files from which EID-symbol mapping can be read')

args = parser.parse_args()
# -

if False:
    # For testing
    #file='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/P21002/rev7/data_normlisedForIntegration.h5ad'
    #file='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE117770/data_normlisedForIntegration.h5ad'
    #file_eid='/lustre/groups/ml01/projects/2020_pancreas_karin.hrovatin/data/pancreas/scRNA/P21002/rev7/cellranger/*/count_matrices/filtered_feature_bc_matrix/features.tsv'
    #file_eid='/lustre/groups/ml01/projects/2020_pancreas_karin.hrovatin/data/pancreas/scRNA/GSE117770/cellranger/*/count_matrices/filtered_feature_bc_matrix/features.tsv'
    args = parser.parse_args([
        '-f','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE117770/data_normlisedForIntegration.h5ad',
        '-fe','/lustre/groups/ml01/projects/2020_pancreas_karin.hrovatin/data/pancreas/scRNA/GSE117770/cellranger/*/count_matrices/filtered_feature_bc_matrix/features.tsv'
        ])


file=args.file
file_eid=args.file_eid
print('file:',file,'\n',
     'file_eid:',file_eid,'\n')

# Load adata
adata=sc.read(file)

# ## Add EIDs

# +
# Use EID info from cellrange data
# Assumes that all samples from dataset have same features, thus performs this for single sample

# Read CellRanger feature info
# Read from the frist file in the sequence of files, assuming they are the same
file_eid1=glob.glob(file_eid)[0]
features_f=pd.read_table(file_eid1,header=None)
# Some files do not have feature_type
features_f.columns=['EID','gene_symbol_adata','feature_type'][:features_f.shape[1]]

# map gene names from Cellranger to gene names used in adata
af=ann.AnnData(var=features_f)
af.var_names=features_f['gene_symbol_adata']
af.var_names_make_unique()
features_f['gene_symbol_adata']=af.var_names
# Add symbols to idx to enable easier mapping
features_f.index=features_f['gene_symbol_adata']
# -

# Add EIDs to adata
adata.var['EID']=features_f.loc[adata.var_names,'EID']
if adata.raw is not None:
    adata.raw.var['EID']=features_f.loc[adata.raw.var_names,'EID']

# Save adata
adata.write(file)
