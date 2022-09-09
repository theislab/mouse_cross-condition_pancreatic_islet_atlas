# +
# Save h5ad  file in mtx that can be used by cellbender
# Or save to other dir for e.g. compass
# -

from scipy.io import mmwrite
import scanpy as sc
import sys  
#sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/')
#import helper as h
import pandas as pd
import os
from scipy import sparse
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--uid', help='UNUSED: unique id for copying file when reading/writting',
                   default='0')
parser.add_argument('--folder', help='data folder')
parser.add_argument('--file', help='data file name without folder')
parser.add_argument('--folder_out', help='folder for saving')
parser.add_argument('--out_prefix', help='prefix for output files',default='')
parser.add_argument('--data_type', 
                    help='X, raw, or raw_norm (for thiw it normalises raw with specified sf col)',
                   default='X')
parser.add_argument('--sf_name', help='prefix for output files',default='size_factors_sample')
parser.add_argument('--var_name', help=' Gene symbol name.'+\
                    'If var use var_names if gene_symbol use var.gene_symbol '+\
                    'if something else use col from anno file, '+\
                    'matching index by var_names and filing Nan with anno file index',
                    default='var')
parser.add_argument('--save_latent',help='if given save this obsm field, else do not save',
                    default=None)
parser.add_argument('--save_mtx',help='save expression as mtx with barcodes and genes files',
                    default=False,type=lambda x: bool(int(x)))
parser.add_argument('--save_csv',help='save expression as csv',
                    default=False, type=lambda x: bool(int(x)))


args = parser.parse_args()

if False:
    # For testing
    args = parser.parse_args([
        '--folder','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/',
        '--file','data_rawnorm_integrated_analysed_beta_v1s1.h5ad',
        '--folder_out','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/scFEA/data_integrated_analysed_beta_v1s1/',
        '--out_prefix','data_integrated_analysed_beta_v1s1_',
        '--data_type','X',
        '--sf_name','size_factors_sample',
        '--var_name','MGI symbol',
        '--save_csv','1'
        ])

print(args)

UID2='to_mtx'+args.uid

#adata=h.open_h5ad(file=folder+file,unique_id2=UID2)
adata=sc.read(args.folder+args.file)

# if folder out does not exist make it
try:
    os.mkdir(args.folder_out)
except FileExistsError:
    pass

# Extract expression matrix
if args.data_type=='X':
    data=adata.X
# If using raw also change adata afterwards so that var names can be matched
elif args.data_type=='raw':
    data=adata.raw.X
    adata=adata.raw.to_adata()
elif args.data_type=='raw_norm':
    data=np.asarray(adata.raw.X/adata.obs[sf_name].values[:,None])
    adata=adata.raw.to_adata()
# Transpose 
data=data.transpose()

# get gene names
if args.var_name=='var':
    genes=adata.var_names
elif args.var_name=='gene_symbol':
    genes=adata.var.gene_symbol
else:
    # Load gene anno
    genes_anno=pd.read_table('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/genomeAnno_ORGmus_musculus_V103.tsv',
                             index_col=0)
    # Select values from specified column, fill NA values with index values
    genes=genes_anno.loc[adata.var_names,args.var_name].fillna(
        dict(zip(adata.var_names,adata.var_names)))

if args.save_mtx:
    # Save sparse expression matrix var*obs
    mmwrite(args.folder_out+args.out_prefix+'matrix.mtx', sparse.csr_matrix(data))
    # Save gene names
    pd.DataFrame(genes).to_csv(
        args.folder_out+args.out_prefix+'genes.tsv',sep='\t',index=False,header=False)
    # Save obs names
    pd.DataFrame(adata.obs_names).to_csv(
        args.folder_out+args.out_prefix+'barcodes.tsv',sep='\t',index=False,header=False)

# Save latent
if args.save_latent is not None:
    pd.DataFrame(adata.obsm[args.save_latent]).to_csv(
        args.folder_out+args.out_prefix+'latent.tsv',sep='\t')

# Save csv
if args.save_csv:
    if sparse.issparse(data):
        data=data.todense()
    pd.DataFrame(data,index=genes,columns=adata.obs_names).to_csv(
        args.folder_out+args.out_prefix+'expression.csv',sep=',')
