# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: cellxgene
#     language: python
#     name: cellxgene
# ---

# %%
import scanpy as sc
import numpy as np
import pandas as pd
import pprint
import gc
from collections import defaultdict
from scipy.sparse import csr_matrix
import pickle 

import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/')
from importlib import reload  
import helper
reload(helper)
import helper as h

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/'
path_save=path_data+'submission/geo/'

# %% [markdown]
# Objects to supplement adata expression with

# %%
# Adata metadata fields
meta=dict()

# %% [markdown]
# ## Prepare adata fields

# %% [markdown]
# ### Add fields from existing adata objects
# Check what is in adatas and subsequently add missing info

# %% [markdown]
# ##### data_integrated_analysed.h5ad

# %%
adata_temp=sc.read(path_data+'data_integrated_analysed.h5ad', backed='r')
adata_temp

# %% [markdown]
# Fields to add

# %%
# Fields to add
add_anna={
    'obs':['study_sample', 'study', 'file', 'reference', 'size_factors_sample', 
           'phase_cyclone', 's_cyclone', 'g2m_cyclone', 
           'g1_cyclone', 'sex', 'ins_score', 'ins_high', 'gcg_score', 
           'gcg_high', 'sst_score', 'sst_high', 'ppy_score', 'ppy_high', 'cell_filtering', 
           'age', 'strain', 'tissue', 'technique', 
           'study_sample_design', 'cell_type', 'cell_type_multiplet', 'cell_subtype', 
           'cell_subtype_multiplet', 'design', 
           'cell_type_integrated_v1',
           'size_factors_integrated', 'pre_cell_type_unified', 'pre_cell_type_original', 
           'study_parsed', 'cell_type_integrated_v1_parsed', 'cell_type_parsed', 'low_q'],
    'uns':['cell_type_integrated_v1_colors', 
           'cell_type_integrated_v1_parsed_order', 
           'study_order', 
           'study_parsed_colors', 'study_parsed_order'],
    'obsm':['X_integrated', 'X_umap'],
}

# %%
# Save above specified fields
for m, fields in add_anna.items():
    if not isinstance(vars(adata_temp)['_'+m],pd.DataFrame):
        meta[m]={}
        for field in fields:
            if m!='obsm':
                meta[m][field]=vars(adata_temp)['_'+m][field]
            else:
                meta[m][field]=pd.DataFrame(vars(adata_temp)['_'+m][field],
                                            index=adata_temp.obs_names)
    else:
        meta[m]=[]
        meta[m].append(vars(adata_temp)['_'+m][fields])

# %%
del adata_temp

# %%
gc.collect()

# %% [markdown]
# ##### data_rawnorm_integrated_annotated.h5ad  

# %%
adata_temp=sc.read(path_data+'data_rawnorm_integrated_annotated.h5ad', backed='r')
adata_temp

# %%
# Add var info
m='var'
fields=['gene_symbol', 'used_integration', 'gene_symbol_original_matched']
meta[m]=[]
meta[m].append(vars(adata_temp)['_'+m][fields])

# %%
del adata_temp

# %%
gc.collect()

# %% [markdown]
# ##### data_integrated_annotated.h5ad

# %%
sc.read(path_data+'data_integrated_annotated.h5ad', backed='r')

# %% [markdown]
# All important fields from "annotated" are also in the "analysed" file.

# %% [markdown]
# ##### data_rawnorm_integrated_analysed_beta_v1s1_sfintegrated.h5ad

# %%
adata_temp=sc.read(path_data+'data_rawnorm_integrated_analysed_beta_v1s1_sfintegrated.h5ad', 
                   backed='r')
adata_temp

# %% [markdown]
# Define data parts that should be added from beta

# %%
# Fields to add
add_anna={
    'obs':[
           'leiden_r1.5',  'leiden_r20', 'hc_gene_programs',
           'hc_gene_programs_parsed', 'leiden_r1.5_parsed', 
           'leiden_r1.5_parsed_const'],
    'uns':['hc_gene_programs_parsed_colors', 'hc_gene_programs_parsed_order', 
           'leiden_r1.5_parsed_colors', 'leiden_r1.5_parsed_order'],
    'obsm':['X_umap', 'X_umap_opt'],
}

# %%
# Save above specified fields
prefix='BETA-DATA_'
for m, fields in add_anna.items():
    if not isinstance(vars(adata_temp)['_'+m],pd.DataFrame):
        for field in fields:
            if m!='obsm':
                meta[m][prefix+field]=vars(adata_temp)['_'+m][field]
            else:
                meta[m][prefix+field]=pd.DataFrame(vars(adata_temp)['_'+m][field],
                                            index=adata_temp.obs_names)
    else:
        meta[m].append(vars(adata_temp)['_'+m][fields].rename(
            {c:prefix+c for c in fields},axis=1))

# %%
del adata_temp

# %%
gc.collect()

# %% [markdown]
# ##### submission/cellxgene/adata.h5ad

# %%
adata_temp=sc.read(path_data+'submission/cellxgene/adata.h5ad', backed='r')
adata_temp

# %% [markdown]
# Define data parts that should be added from CellxGene adata

# %%
# Fields to add
add_anna={
    'obs':['n_genes', 'mt_frac', 'doublet_score', 
           'log10_n_counts', 'age_approxDays', 
           'cell_subtype_immune_reannotatedIntegrated', 
           'cell_subtype_endothelial_reannotatedIntegrated', 'emptyDrops_LogProb_scaled', 
           'diabetes_model', 'chemical_stress', 'GEO_accession', 
           'sex_annotation'],
    'var':[
        # This also identifies genes in the data used for analyses
        'feature_is_filtered',
        'present_Fltp_2y', 'present_Fltp_adult', 'present_Fltp_P16', 
           'present_NOD', 'present_NOD_elimination', 'present_spikein_drug', 
           'present_embryo', 'present_VSG', 'present_STZ']
}

# %%
# Save above specified fields
prefix='CXG-DATA_'
for m, fields in add_anna.items():
    if not isinstance(vars(adata_temp)['_'+m],pd.DataFrame):
        for field in fields:
            if m!='obsm':
                meta[m][prefix+field]=vars(adata_temp)['_'+m][field]
            else:
                meta[m][prefix+field]=pd.DataFrame(vars(adata_temp)['_'+m][field],
                                            index=adata_temp.obs_names)
    else:
        meta[m].append(vars(adata_temp)['_'+m][fields].rename(
            {c:prefix+c for c in fields},axis=1))

# %%
# Add finally parsed gene symbols for all genes
meta['var'].append(pd.DataFrame(
    h.get_symbols(adata_temp.var.index).rename('gene_symbol_FINAL')))

# %% [markdown]
# #### Combine metadata fields from different objects and do final edits

# %%
# Combine data parts from different objects
for m in ['obs','var']:
    meta[m]=pd.concat(meta[m],axis=1)

# %%
# Add False to used_integration genes unmatched accross datasets
meta['var']['used_integration']=meta['var']['used_integration'].fillna(False)

# %% [markdown]
# ### View metadata fiels
# Combine and add readme info

# %%
for col in sorted(meta['obs'].columns):
    print('\n************')
    print(col)
    print(meta['obs'][col].dtype)
    if meta['obs'][col].nunique()<100:
        print('\n',sorted(meta['obs'][col].astype(str).unique()))

# %%
for col in sorted(meta['var'].columns):
    print('\n************')
    print(col)
    print(meta['var'][col].dtype)
    if meta['var'][col].nunique()<100:
        print('\n',sorted(meta['var'][col].astype(str).unique()))

# %%
meta['uns']

# %%
meta['obsm']

# %%
# make uns descriptions
meta['uns']['field_descriptions']={
    'readme':'The field description dictionary contains adata field explanations. '+\
              'Fields prefixed with "BETA-DATA" come from beta-cell specific adata object '+\
              '(in code saved in data_rawnorm_integrated_analysed_beta_v1s1_sfintegrated.h5ad) '+\
              'and fields prefixed with CXG-DATA come from adata prepared for cellxgene; '+\
              'other fields come from atlas-wide objects (in code saved in '+\
              'data_rawnorm_integrated_annotated.h5ad and data_integrated_analysed.h5ad). '+\
              'Expression contains all genes from count matrices of individual datasets and '+\
              'we specify in the below described var columns which genes were used for '+\
              'integration and atlas exploration. If a gene was missing from a dataset its '+\
              'expression was set to 0 for that dataset.',
    'X':'Expression normalized with integration-based size factors and transformed with log(x+1)',
    'raw.X':'Raw expression counts',
    'obs':{
        'BETA-DATA_hc_gene_programs': 'Beta cell fine subtype annotation on '+\
            'integrated atlas. Fine annotation aimed at capturing all biollogically '+\
            'distinct beta cell subtypes (assesed based on gene program activity patterns).' ,
        'BETA-DATA_hc_gene_programs_parsed': 'As BETA-DATA_hc_gene_programs, but with '+\
            'pretty names.' +\
            'Abbreviations: D-inter. - diabetic intermediate, NOD-D - NOD diabetic, '+\
            'M/F - male/female, chem - chem dataset, imm. - immature, lowQ - low quality.',
        'BETA-DATA_leiden_r1.5': 'Clustering used to define beta cell coarse subtype '+\
            'annotation.',
        'BETA-DATA_leiden_r1.5_parsed': 'Beta cell coarse subtype '+\
            'annotation on integrated atlas. Coarse annotation based on '+\
            'metadata information. Abbreviations: NOD-D - NOD diabetic, M/F - male/female, '+\
            'chem - chem dataset, imm. - immature, lowQ - low quality, '+\
            'hMT - high mitochondrial fraction',
        'BETA-DATA_leiden_r1.5_parsed_const': 'As BETA-DATA_leiden_r1.5_parsed, but without '+\
            'pretty names. Used as an unchanging reference for coarse cell states.',
        'BETA-DATA_leiden_r20': 'Cell clusters used to define beta-cell pseudobulk in '+\
            'some analyses.',
        'CXG-DATA_GEO_accession': 'GEO accession of each dataset',
        'CXG-DATA_age_approxDays': "Approximate mapping of age column to days for "+\
            "the purpose of visualisation",
        'CXG-DATA_cell_subtype_endothelial_reannotatedIntegrated': 'Endothelial cell subtype '+\
            'reannotation on integrated atlas based on known markers',
        'CXG-DATA_cell_subtype_immune_reannotatedIntegrated': 'Immune cell subtype '+\
            'reannotation on integrated atlas based on known markers',
        'CXG-DATA_chemical_stress': 'Application of chemicals to islets',
        'CXG-DATA_diabetes_model': 'Diabetes model and any diabetes treatment',
        'CXG-DATA_doublet_score': 'Scrublet doublet scores computed per sample; '+\
                'higher - more likely doublet',
        'CXG-DATA_emptyDrops_LogProb_scaled': 'Log probability that droplet is empty computed '+\
                'per sample with emptyDrops and scaled to [0,1] per sample; '+\
                'higher - more likely empty droplet',
        'CXG-DATA_log10_n_counts': 'log10(N counts)',
        'CXG-DATA_mt_frac': 'Fraction of mitochondrial genes expression',
        'CXG-DATA_n_genes': 'Number of expressed genes',
        'CXG-DATA_sex_annotation': 'Was sex known from sample metadata (ground-truth) or '+\
                'was it determined bsed on Y-chromosomal gene expression (data-driven)',
        'age': "Age as defined in original publication, E - embryonic days, "+\
                "d - postnatal days, m - postnatal months, y - postnatal years",
        'cell_filtering': "FACS sorting",
        'cell_subtype': 'Per-dataset annotation of cell types and for beta cells cell states '+\
            'that are annotated based on known markers. Used for integartion evaluation. '+\
            'Unnanotated cells start with NA.',
        'cell_subtype_multiplet': 'As cell_subtype, but with potential multiplet cell clusters '+\
            'set to a single annotation that can be jointly filtered out.',
        'cell_type': 'Per-dataset annotation of cell types used for integartion evaluation. '+\
            'Unnanotated cells start with NA.',
        'cell_type_integrated_v1': 'Cell type reannotation on the integrated atlas.',
        'cell_type_integrated_v1_parsed': 'As cell_type_integrated_v1, but with pretty names.'+\
            'Abbreviations: E - embryonic, endo. - endocrine, "+"" symbol - likely doublet, '+\
            'prolif. - proliferative, lowQ - low quality, '+\
            'stellate a./q. - stellate activated/quiescent',
        'cell_type_multiplet': 'As cell_type, but with potential multiplet cell clusters '+\
            'set to a single annotation that can be jointly filtered out.',
        'cell_type_parsed': 'As cell_type, but with pretty names.',
        'design': 'Brief sample description that gives information on differences '+\
                'between samples within dataset',
        'file': 'Technical sample, in some cases equal to biological sample',
        'g1_cyclone': 'G1 score of cyclone',
        'g2m_cyclone': 'G2M score of cyclone',
        '*_high': 'Where * denotes hormone name. '+\
            'Do cells have high expression of the given hormone (ins, gsg, sst, ppy)'+\
            ', determined per sample',
        '*_score': 'Where * denotes hormone name. Hormone scores computed per sample.',
        'low_q': 'False for cells asigned to low quality clusters',
        'phase_cyclone': 'Phase of cell cycle (cyclone)',
        'pre_cell_type_original': "Cell types as reported in the studies that generated "+\
                "the datasets",
        'pre_cell_type_unified': "Cell types as reported in the studies that "+\
                "generated the datasets; manually unified to a common naming scheme. "+\
                'Abberviations: E - embryonic, EP - endocrine progenitor/precursor, '+\
                'Fev+ - Fev positive, prolif. - proliferative, "+"" symbol - likely doublet',
        'reference': 'Was dataset used in initial integration hyperparameter evaluation, '+\
            'see reproducibility code for more information.',
        's_cyclone': 'S score of cyclone',
        'sex': 'Cell sex.',
        'size_factors_integrated': 'Size factors computed on integrated embedding.',
        'size_factors_sample': 'Size factors computed per sample.',
        'strain': 'Mouse strain and genetic background',
        'study': 'Dataset name.',
        'study_parsed': 'As study, but with pretty name.',
        'study_sample': 'Concatentation of columns with sample study and sample '+\
            '(file) information',
        'study_sample_design': 'Concatentation of columns with sample study, sample '+\
            '(file), and design information',
        'technique': 'Sequencing protocol',
        'tissue': 'Tissue type',

    },
    'var':{
        'CXG-DATA_feature_is_filtered': 'False for features not present in all datasets. '+\
            'This defines features used for atlas exploration.',
        'CXG-DATA_present_*': '* Dataset name. True if feature was present in a dataset.',
        'gene_symbol': 'Gene symbols from BioMart V103.',
        'gene_symbol_FINAL': 'Gene symbol assigned for each gene first based on BioMart V103, '+\
            ' and if missing based on gene symbols from alignment information of datasets.',
        'gene_symbol_original_matched': 'Gene symbols shared across annotations of individual '+\
            'datasets, some of which differ due to different genomic versions.'+\
            'If not shared set to np.nan.',
        'used_integration': 'True if feature was used for integration.',

    },
    'uns':{
        '*_colors': 'Colors for the column corresponding to *.',
        '*_order': 'Order of categories in the column corresponding to *.',
    },
    'obsm':{
        'BETA-DATA_X_umap': 'Beta-cell specific UMAP embedding computed with default '+\
            'Scanpy parameters.',
        'BETA-DATA_X_umap_opt': 'As BETA-DATA_X_umap, but with optimised parameters to '+\
            'enable better visualisation of cell clusters.',
        'X_integrated': 'Integrated atlas embedding.',
        'X_umap': 'Atlas-wide UMAP embedding.',

    },
}

# %% [markdown]
# ### Save metadata

# %%
for m,data in meta.items():
    if isinstance (data,pd.DataFrame):
        data.to_csv(path_save+m+'.tsv',sep='\t')
    else:
        pickle.dump(data,open(path_save+m+'.pkl','wb'))

# %% [markdown]
# ## Expression

# %%
adata_temp=sc.read(path_data+'submission/cellxgene/adata.h5ad')

# %%
# Make adata
adata=sc.AnnData(adata_temp.raw.X,
                 obs=pd.DataFrame(index=adata_temp.obs.index),
                 var=pd.DataFrame(index=adata_temp.raw.var.index))
adata.raw=adata.copy()
del adata_temp
gc.collect()

# %%
# Reload obs for size factors
obs=pd.read_table(path_save+'obs.tsv',index_col=0)

# %%
# Normalize X
adata.obs['size_factors_integrated']=obs.loc[adata.obs_names,'size_factors_integrated']
adata=h.get_rawnormalised(adata,sf_col='size_factors_integrated',
                    use_log=True,save_nonlog=False, use_raw=False, copy=False)
adata.obs.drop('size_factors_integrated',axis=1,inplace=True) 
del adata.uns['log1p']

# %%
print(adata)
print(adata.raw)

# %% [markdown]
# ## Add metadata to adata

# %%
# Obs
obs=pd.read_table(path_save+'obs.tsv',index_col=0)
adata.obs=obs.loc[adata.obs_names,:]

# %%
# Var
var=pd.read_table(path_save+'var.tsv',index_col=0)
adata.var=var.loc[adata.var_names,:]

# %%
# Obsm
obsm=pickle.load(open(path_save+'obsm.pkl','rb'))
for o,data in obsm.items():
    adata.obsm[o]=data.reindex(adata.obs_names).values

# %%
# Uns
uns=pickle.load(open(path_save+'uns.pkl','rb'))
for u,data in uns.items():
    adata.uns[u]=data

# %% [markdown]
# ## Save adata

# %%
adata

# %%
adata.write(path_save+'adata.h5ad')

# %%
path_save+'adata.h5ad'

# %%
