# %% [markdown]
# # Gene annotation
# The original per sample adatas did not contain EIDs (Ensembl IDs, the adatas contained only gene symbols) thus map between gene symbols used across adatas and EIDs (provided by core processing facility separately for each sample (matching within dataset)), since these will differ due to different used genomic versions. Retrieve and save gene symbol to EID and genomic location mapping.

# %%
import pandas as pd
import pyensembl
import scanpy as sc
import anndata
from pathlib import Path
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

import sys  
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/')
from constants import SAVE

# %% [markdown]
# ## Ens ids from features file from cellranger 

# %% [markdown]
# Map symbols to EIDs for eacdh dataset and subset to genes only/remove reporter transcript features.

# %%
features=[]
have_nongenes=set()
nongenes=set()
# File reporting EID files location for each sample
files_data=pd.read_table('/lustre/groups/ml01/workspace/karin.hrovatin//data/pancreas/scRNA/raw_file_list.tsv')
for i,sample in files_data.iterrows():
    has_nongenes=False
    # Load features from cellranger
    ending='features.tsv'
    if sample['ending']=='gene_bc_matrices.h5ad':
        ending='genes.tsv'
    features_f=pd.read_table(sample['dir_cellranger']+ending,header=None)
    features_f.columns=['EID','gene_symbol_adata','feature_type'][:features_f.shape[1]]
    features_f.index=features_f['EID']
    # map gene names to gene names used in adata
    af=anndata.AnnData(var=features_f)
    af.var_names=features_f['gene_symbol_adata']
    af.var_names_make_unique()
    features_f['gene_symbol_adata']=af.var_names
    # load features from adata
    features_used=sc.read(sample['dir']+'filtered_'+sample['ending'],backed='r').var_names
    
    # Keep only features in adata
    features_previous=features_f.index.values
    features_f.index=features_f['gene_symbol_adata']
    features_f=features_f.loc[features_used,:]
    features_f.index=features_f['EID']
    if (features_previous.shape[0]-features_f.shape[0])!=0:
        print('not in adata:',set(features_previous)-set(features_f.index.values),'in',
             sample['study'],sample['sample'])
        
    # keep only genes
    features_previous=features_f.index.values
    if 'feature_type' in features_f.columns:
        # featuyre type NA
        features_f=features_f[~features_f.feature_type.isna()]
        # Feature type not Gene Expression
        features_f=features_f[
            features_f.feature_type=='Gene Expression']
    if (features_previous.shape[0]-features_f.shape[0])!=0:
        print('non-genes:',set(features_previous)-set(features_f.index.values),'in',
             sample['study'],sample['sample'])
        has_nongenes=True
    # index not starting with EID - custom reporter genes added to genome
    features_previous=features_f.index.values
    features_f=features_f[features_f.index.str.startswith('ENSMUSG')]
    if (features_previous.shape[0]-features_f.shape[0])!=0:
        print('non-ENSid:',set(features_previous)-set(features_f.index.values),'in',
              sample['study'],sample['sample'])
        has_nongenes=True
    if has_nongenes:
        have_nongenes.add(sample['study']+'__'+sample['sample'])
    
    # Save features to df
    features_f=features_f['gene_symbol_adata']
    features_f.name=sample['study']+'__'+sample['sample']
    features.append(features_f)
features=pd.concat(features,axis=1)

# %% [markdown]
# ### Do gene symbols match for EIDs
# Which symbols match across data. Namely, there are genes for which symbols do not match or are nan in some samples but annotated in other samples.

# %%
# Extract non-matching genes
different_anno=dict()
for row,row_data in features.iterrows():
    if row_data.unique().shape[0]>1:
        different_anno[row]={group:group_data.index 
                                for group,group_data in 
                                row_data.fillna('NA').groupby(row_data.fillna('NA'))}
# Per-sample view of non-matching genes
features.loc[different_anno.keys(),:]

# %% [markdown]
# Genes for which symbols do not match (exluding those that are nan in some samples but not others)

# %%
different_symbol=dict()
for row,row_data in features.iterrows():
    if row_data.nunique(dropna=True)>1:
        different_symbol[row]={group:group_data.index 
                                for group,group_data in 
                                row_data.groupby(row_data)}
features.loc[different_symbol.keys(),:]

# %% [markdown]
# Genes with NA anno in some studies

# %%
# Genes with NA anno in some studies
different_presence={k:v for k,v in different_anno.items() if k not in different_symbol}
features.loc[different_presence.keys(),:]

# %% [markdown]
# Check if labels (symbols) match and are present. This gives True (1) for 'a','a' but False (0) for 'a',nan and nan,nan

# %%
# N matched features
features_matching=pd.DataFrame(index=features.columns,columns=features.columns)
for f1 in features.columns:
    for f2 in features.columns:
        # This gives True for 'a','a' but 0 for 'a',nan and nan,nan
        features_matching.at[f1,f2]=(features[f1]==features[f2]).sum()

# %% [markdown]
# Number of genes for which symbols match correctly (based on EIDs) across studies.

# %%
# Dataset Legend
studies_cmap=dict(zip(files_data.study.unique(),cm.tab20.colors))
studies_colors=[studies_cmap[ss.split('__')[0]] for ss in features_matching.columns]
patches = [ mpatches.Patch(color=c, label=l ) for l,c in studies_cmap.items() ]

# %%
# Plot
sb.clustermap(features_matching.astype('int'),xticklabels=False,yticklabels=False,
             row_colors=studies_colors,col_colors=studies_colors,figsize=(7,7))
plt.legend(handles=patches, bbox_to_anchor=(17, 1),ncol=3)

# %% [markdown]
# Also check if all labels match (also take nan,nan as match).

# %%
features_all_matching=pd.DataFrame(index=features.columns,columns=features.columns)
for f1 in features.columns:
    for f2 in features.columns:
        # This gives True for 'a','a' but 0 for 'a',nan and nan,nan
        features_all_matching.at[f1,f2]=int(
            (features[f1].fillna('NA')==features[f2].fillna('NA')).all())

# %%
sb.clustermap(features_all_matching.astype('int'),xticklabels=False,yticklabels=False,
             row_colors=studies_colors,col_colors=studies_colors,figsize=(7,7))
plt.legend(handles=patches, bbox_to_anchor=(17, 1),ncol=3)

# %% [markdown]
# C: The two dataset cluster occur due to two different genome version used for alignment.

# %% [markdown]
# #### How strongly are expressed genes for which EIDs do not match
# Calculate per study how strongly are expressed genes with unmatching annotation. This is done per study not sample as annotation is the same across samples of individual study.

# %%
data_study=[
      ('Fltp_2y','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islets_aged_fltp_iCre/rev6/'),
      ('Fltp_adult','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islet_fltp_headtail/rev4/'),
      ('Fltp_P16','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/salinno_project/rev4/'),
      ('NOD','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE144471/'),
      ('NOD_elimination','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE117770/'),
      ('spikein_drug','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE142465/'),
      ('embryo','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE132188/rev7/'),
      ('VSG','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/VSG_PF_WT_cohort/rev7/'),
      ('STZ','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islet_glpest_lickert/rev7/')
    ]


# %% [markdown]
# How many genes with symbol inconsistency are present in QC-ed data of each study. 
#
# Diff symbol - different symbol across studies. Diff anno - different symbol or gene annotation being present in only some studies' genomes. All genes - all QC-ed genes.

# %%
for study,file in data_study:
    # QC-ed data of the study
    print(study)
    adata_study=sc.read(file+'data_normalised.h5ad')
    # Use normalised expresion without log transform
    adata_study.X=np.asarray(
        adata_study.layers['counts']/adata_study.obs.size_factors.values.reshape(-1,1))
    # recompute n cells as this was done before cell filtering
    adata_study.var['n_cells_filtered']=(adata_study.X>0).sum(axis=0)
    adata_study.var['mean_expr_in_expr_cells'
                   ]=adata_study.X.sum(axis=0)/adata_study.var['n_cells_filtered']

    # Prepare EID-symbol mapping to match symbols in adata
    # Symbols from the study
    features_temp=pd.DataFrame(
    features[[col for col in features.columns if col.split('__')[0]==study][0]].copy())
    features_temp.columns=['symbol']
    features_temp['EID']=features_temp.index.values
    features_temp.index=features_temp.symbol.values
    # Keep only genes present in studie's genome
    features_temp=features_temp[~features_temp.index.isna()]
    # Make sure that gene symbols match those in adata
    adata_temp=anndata.AnnData(var=features_temp)
    adata_temp.var_names_make_unique()
    features_temp['symbol_adata']=adata_temp.var_names
    features_temp.index=features_temp['EID']

    # Find out how many genes with symbol name conflicts are in QC-ed data
    diff_symbol_genes=features_temp.loc[different_symbol.keys(),'symbol_adata'].values
    diff_symbol_genes_adata=[gene for gene in diff_symbol_genes if gene in adata_study.var_names]
    print('N genes with different symbol in filtered adata:',len(diff_symbol_genes_adata) )
    diff_anno_genes=features_temp.loc[
        [ gene for gene in different_anno.keys() if gene in features_temp.index],
        'symbol_adata'].values
    diff_anno_genes_adata=[gene for gene in diff_anno_genes if gene in adata_study.var_names]
    print('N genes with different annotation (symbol+not present in some samples) in filtered adata:',
          len(diff_anno_genes_adata) )

    # Summary of mean expr values across genes
    expr_summary=pd.DataFrame(columns=['n_cells','mean_expr_in_expr_cells'],
                              index=['all','diff symbol','diff anno'])
    expr_summary.at['all','n_cells']=adata_study.var.loc[:,'n_cells_filtered'].mean()
    expr_summary.at['all','mean_expr_in_expr_cells'
                   ]=adata_study.var.loc[:,'mean_expr_in_expr_cells'].mean()
    expr_summary.at['diff symbol','n_cells'
                   ]=adata_study.var.loc[diff_symbol_genes_adata,'n_cells_filtered'].mean()
    expr_summary.at['diff symbol','mean_expr_in_expr_cells'
                   ]=adata_study.var.loc[diff_symbol_genes_adata,'mean_expr_in_expr_cells'].mean()
    expr_summary.at['diff anno','n_cells'
                   ]=adata_study.var.loc[diff_anno_genes_adata,'n_cells_filtered'].mean()
    expr_summary.at['diff anno','mean_expr_in_expr_cells'
                   ]=adata_study.var.loc[diff_anno_genes_adata,'mean_expr_in_expr_cells'].mean()
    print('\nMean across genes:')
    display(expr_summary)
    # Summary of median expr values across genes
    expr_summary=pd.DataFrame(columns=['n_cells','mean_expr_in_expr_cells'],
                              index=['all','diff symbol','diff anno'])
    expr_summary.at['all','n_cells']=adata_study.var.loc[:,'n_cells_filtered'].median()
    expr_summary.at['all','mean_expr_in_expr_cells'
                   ]=adata_study.var.loc[:,'mean_expr_in_expr_cells'].median()
    expr_summary.at['diff symbol','n_cells'
                   ]=adata_study.var.loc[diff_symbol_genes_adata,'n_cells_filtered'].median()
    expr_summary.at['diff symbol','mean_expr_in_expr_cells'
                   ]=adata_study.var.loc[diff_symbol_genes_adata,'mean_expr_in_expr_cells'].median()
    expr_summary.at['diff anno','n_cells'
                   ]=adata_study.var.loc[diff_anno_genes_adata,'n_cells_filtered'].median()
    expr_summary.at['diff anno','mean_expr_in_expr_cells'
                   ]=adata_study.var.loc[diff_anno_genes_adata,'mean_expr_in_expr_cells'].median()
    print('\nMedian across genes:')
    display(expr_summary)

    # plot N cells and mean expresion in expr cells per gene 
    # for genes in QC-ed data that have gene symbol conflicts
    fig,ax=plt.subplots(1,2,figsize=(18,5))
    ax[0].hist(adata_study.var.loc[:,'n_cells_filtered'],bins=20,label='all genes',color='grey')
    ax[1].hist(adata_study.var.loc[:,'mean_expr_in_expr_cells'],bins=20,label='all genes',color='grey')
    ax2=ax[0].twinx()
    ax3=ax[1].twinx()
    ax2.hist(adata_study.var.loc[diff_symbol_genes_adata,'n_cells_filtered'],color='red',
            bins=20,label='diff symbol',alpha=0.5)
    ax2.hist(adata_study.var.loc[diff_anno_genes_adata,'n_cells_filtered'],color='blue',
            bins=20,label='diff anno',alpha=0.5)
    ax3.hist(adata_study.var.loc[diff_symbol_genes_adata,'mean_expr_in_expr_cells'],color='red',
            bins=20,label='diff symbol',alpha=0.5)
    ax3.hist(adata_study.var.loc[diff_anno_genes_adata,'mean_expr_in_expr_cells'],color='blue',
            bins=20,label='diff anno',alpha=0.5)
    ax[0].set_title(study)
    ax[0].set_yscale('log')
    ax2.set_yscale('log')
    ax[1].set_yscale('log')
    ax3.set_yscale('log')
    ax[0].set_xlabel('N cells expressing gene')
    ax[1].set_xlabel('Mean expression in cells expressing gene')
    ax[0].set_ylabel('N genes (all)')
    ax2.set_ylabel('N genes (diff)')
    ax[1].set_ylabel('N genes (all)')
    ax3.set_ylabel('N genes (diff)')
    handles, labels = ax[0].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles+handles2)
    display(fig)
    plt.close()
 
del adata_temp
del adata_study
del features_temp


# %% [markdown]
# ## Add data from Ensembl
# Add gene info downloaded from BioMart.

# %% [markdown]
# Ensembl data

# %%
# Genome info
org='mus_musculus'
if org=='mus_musculus':
    org_short='MM'
anno_v=103

# %%
# Get data from https://m.ensembl.org/biomart/martview/
anno_path='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'
genes_df=[]
curr_cols=set()
for f in Path(anno_path).rglob('mart_export_'+str(anno_v)+org_short+'*.txt'):
    if 'NCBI_ID' not in f.name:
        print(f)
        df=pd.read_table(f,index_col='Gene stable ID')
        print('Drop second duplicated:',df[df.index.duplicated(keep=False)])
        df=df[~df.index.duplicated()]
        df.drop([col for col in df.columns if col in curr_cols],axis=1,inplace=True)
        curr_cols.update(df.columns.values)
        genes_df.append(df)
genes_df=pd.concat(genes_df,axis=1)
genes_df['EID']=genes_df.index


# %%
# Ensembl data
genes_df

# %% [markdown]
# #### Combined anno from adata and ensebl

# %%
missing4=set(features.index.values)-set(genes_df.EID.values)
print('Missing gene EIDs from ensembl annotation:',len(missing4),'/',features.shape[0])
print('N genes not in ensembl anno that are withing genes that have different symbols:',
      len(missing4 & set(different_symbol.keys())))
print('N genes not in ensembl anno that are withing genes that have different presence:',
      len(missing4 & set(different_presence.keys())))
print('N genes not in ensembl anno that are withing genes that have different anno:',
      len(missing4 & set(different_anno.keys())))

# %%
genes_df_merged=genes_df.copy()
genes_df_merged.index=genes_df_merged['EID']
genes_df_merged=genes_df_merged.reindex(features.index)
genes_df_merged['EID']=genes_df_merged.index
added_studies=[]
for col in features.columns:
    study=col.split('__')[0]
    if study not in added_studies:
        genes_df_merged['gene_symbol_'+study]=features[col]
        added_studies.append(study)

# %%
genes_df_merged

# %%
genes_df_merged['MGI symbol'].isna().sum()

# %%
print('Unique EIDs:',genes_df_merged['EID'].nunique(),'/',genes_df_merged.shape[0])
print('Unique symbol:',genes_df_merged['Gene name'].nunique(),'/',
      genes_df_merged['Gene name'].dropna().shape[0])

# %%
# Save
if SAVE:
    genes_df_merged.to_csv(
        '/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/genomeAnno_ORG'+\
        org+'_V'+str(anno_v)+'.tsv',
        sep='\t')

# %% [markdown]
# ## Analyse gene symbols used in integration
# This part of the notebook can be run only after creating the integrated adata object (next notebook).

# %%
adata_integrated=sc.read(
    '/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/data_integrated_annotated.h5ad',
    backed='r')

# %%
# Symbols of genes used in integration
integration_genes=adata_integrated.var_names[adata_integrated.var['used_integration']]

# %%
# Symbols of genes with different anno, 
# assure that multiple symbols are not used for the same gene
different_anno_symbols=[key for val in different_anno.values() for key in val.keys() 
                            if key !='NA']

# %%
# Overlap between mismatch gene symbols and integration symbols
print('Mismatch anno in integration:',len(set(integration_genes)&set(different_anno_symbols)))

# %%
# N missmatch symbols in whole data
print('N diff anno symbols:',len(set(different_anno_symbols)),
      'N non-unique diff anno symbols (symbol may repeat across EIDs?):',
      len(different_anno_symbols))

# %%
