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

# %% [markdown]
# Prepare excel file with metadata from adatas for an easier overview of the availiable samples.

# %%
import scanpy as sc
import pandas as pd
import os
import pickle

# %%
path_rna='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/'

# %%
# All datasets
datasets=[
    ('human','GSE83139','GSE83139/GEO/'),
    ('human','GSE154126','GSE154126/GEO/'),
    ('human','GSE101207','GSE101207/GEO/'),
    ('human','GSE124742_GSE164875_patch','GSE124742_GSE164875/GEO/patch/'),
    ('human','GSE124742_FACS','GSE124742_GSE164875/GEO/FACS/'),
    ('human','GSE86469','GSE86469/GEO/'),
    ('human','GSE81547','GSE81547/GEO/'),
    ('human','GSE198623','P21000/sophie/human/'),
    ('human','GSE81608','GSE81608/GEO/'),
    ('human','GSE148073','GSE148073/GEO/'),
    ('mouse','GSE137909','GSE137909/GEO/'),
    ('mouse','GSE83146','GSE83146/GEO/')
]

# %%
# Save info about all datasets
pickle.dump(datasets,open(path_rna+'external_info.pkl','wb'))

# %%
# Save metadata of all datasets
writer = pd.ExcelWriter(path_rna+'external_metadata.xlsx',
                        engine='xlsxwriter') 
for species,name,d in datasets:
    print(species,name)
    # Load adata - filtered if exists
    path_filtered=path_rna+d+'adata_filtered.h5ad'
    file=path_filtered if os.path.exists(path_filtered) else path_rna+d+'adata.h5ad'
    obs=sc.read(file,backed='r').obs.copy()
    donors_all=set(obs.donor.unique()) if 'donor' in obs.columns else set()
    # all data cols
    print('All cols:',list(obs.columns))
    # Remove cells not useful for saving and grouping
    cols_keep=[c for c in obs.columns 
         # Remove organ/organism/tisse/cell info/geo_accession 
         # (as not added to all samples now,may also not be sample specific (e.g. could be cells)
         if 'organ' not in c and 'tissue' not in c and 'cell_' not in c
            and 'geo_accession' not in c ]
    # Make groups based on kept cols
    # Make sure empty groups are dropped and groups with NA are not dsriopped - does not work
    # Thuis fill nan with NA, must first remove categorical
    def uncategorize(col):
        if col.dtype.name == 'category':
            return col.astype(col.cat.categories.dtype)         
        else:
            return col
    obs = obs.apply(uncategorize, axis=0)
    obs=obs.fillna('NA')
    obs=obs.groupby(cols_keep,observed=True,dropna=False)
    # N beta cells per group - add to df
    n_beta=obs.apply(lambda x:x.query('cell_type=="beta"').shape[0])
    # Workaround to add n_beta - remove size latter
    obs=pd.DataFrame(obs.size())
    obs['N_beta_cells']=n_beta.values
    obs=obs.reset_index().drop(0,axis=1)
    # Add species
    obs['organism']=species
    # Display and save
    print('Saved cols:',list(obs.columns))
    display(obs)
    # Make sure donor (if exists) is unique to sinle group - groupping worked as expected
    if 'donor' in obs.columns and obs.value_counts('donor').max()>1:
        raise ValueError('Duplicated donor')
    # make sure all donors are kept
    if len(donors_all)>0:
        if not donors_all==set(obs.donor.unique()):
            raise ValueError('Donors not matching')
    obs.to_excel(writer, sheet_name=name,index=False)   
writer.save()

# %%
path_rna+'external_metadata.xlsx'

# %%
