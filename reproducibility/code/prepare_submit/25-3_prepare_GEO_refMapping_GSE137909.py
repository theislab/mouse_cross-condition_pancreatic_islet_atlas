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
#     display_name: Python rpy2_3
#     language: python
#     name: rpy2_3
# ---

# %%
import pandas as pd
import scanpy as sc
import numpy as np

# %%
path_rna='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/'
path_data=path_rna+'combined/'
path_query=path_rna+'GSE137909/GEO/'
path_model=path_query+'run_scArches1603792372.695119/'
path_save=path_model+'submission/geo/'

# %% [markdown]
# ## Prepare data
# Make adata with only obs and embedding. Leave expression out as it is availiable on GEO (original publication).

# %%
# Load obs and latent and convert to adata
adata=sc.AnnData(obs=sc.read(path_query+'adata.h5ad',backed='r').obs.copy())
obsm=sc.read(path_model+'latent_query.h5ad' ).to_df()
obsm.index=[i.replace('-GSE137909','') for i in obsm.index]
adata.obsm['X_integrated']=obsm.loc[adata.obs_names,:].values

# %%
# analyse obs columns
for col in sorted(adata.obs.columns):
    print('\n************')
    print(col)
    print(adata.obs[col].dtype)
    if adata.obs[col].nunique()<100:
        print('\n',sorted(adata.obs[col].astype(str).unique()))

# %%
# Describe adata fields
adata.uns['field_descriptions']={
    'obs':{
        'STZ': 'Was the mice treated with single high STZ dose.',
        'age': 'Age at death. Abbreviations: d - days, m - months.',
        'batch': 'Batch as defined in the original publication.',
        'cell_subtype_original': 'Beta cell subtypes from the original publication.',
        'cell_type': 'Cell type name matched to atlas cell type nomenclature.',
        'cell_type_original': 'Cell type name from the original publication.',
        'disease': 'Diabetes model',
        'donor': 'Sample (cell source) information from the original publication.',
        'insulin_implant': 'Was mice treated with insulin implant.',
        'sex': 'Animal sex',
        'size_factors_sample': 'Size factors computed per sample; used to normalise raw '+\
            'expression from GEO for reference mapping.',
        'strain': 'Mouse strain and genetic background.',
        'time_after_STZ': 'Time between STZ treatment and sampling. '+\
            'Abbreviations: d- days, m - months.',
    },
    'obsm':{
        'X_integrated':'Embededing from mapping onto the atlas.'
    }
}

# %%
adata

# %% [markdown]
# ## Save

# %%
adata.write(path_save+'adata.h5ad')

# %%
path_save+'adata.h5ad'

# %%
