# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python rpy2_3
#     language: python
#     name: rpy2_3
# ---

# %% [markdown]
# Report number of cells:
# - Cell types per sample, total, lowQ (if applicable)
# - Beta subtypes per sample

# %%
import scanpy as sc
import pandas as pd
import gc
import numpy as np

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/'
path_tables='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/tables/paper/'

# %%
# Get obs and usn of whole atlas and beta subset
temp=sc.read(path_data+'data_integrated_analysed.h5ad',backed='r')
obs=temp.obs.copy()
uns=temp.uns.copy()
del temp
obs_b=sc.read(path_data+'data_rawnorm_integrated_analysed_beta_v1s1_sfintegrated.h5ad',
              backed='r').obs.copy()
gc.collect()

# %%
# Pretty sample names
obs['study_parsed_design_sample']=obs.apply(
    lambda x: ' '.join([x.study_parsed,x.design,x.file]), axis=1)
obs_b['study_parsed_design_sample']=obs_b.apply(
    lambda x: ' '.join([x.study_parsed,x.design,x.file]), axis=1)

# %%
# Count cell types (total and per sample) and per cell type lowq and doublets
ct_count=pd.concat([
    obs['cell_type_integrated_v2_parsed'].value_counts().rename('total'),
    obs.query('low_q==False &  ~cell_type_integrated_v2_parsed.str.contains("\+")',
          engine='python')['cell_type_integrated_v2_parsed'].value_counts().\
          replace(0,np.nan).rename('low_quality'),
    obs.groupby('cell_type_integrated_v2_parsed').apply(lambda x:'+' in x.name
                                                       ).rename('doublet'),
    pd.crosstab(obs['cell_type_integrated_v2_parsed'],obs['study_parsed_design_sample'])
],axis=1)

# %%
ct_count

# %%
# Count coarse beta cell subtypes per sample
ct_b_coarse_count=pd.crosstab(
    obs_b['leiden_r1.5_parsed'],
    obs_b['study_parsed_design_sample'])
ct_b_coarse_count.index.name=None
ct_b_coarse_count

# %%
# Count fine beta cell subtypes per sample
ct_b_fine_count=pd.crosstab(
    obs_b['hc_gene_programs_parsed'],
    obs_b['study_parsed_design_sample'])
ct_b_fine_count.index.name=None
ct_b_fine_count

# %%
# Save count data
writer = pd.ExcelWriter(path_tables+'cell_counts.xlsx',engine='xlsxwriter') 
for sheet,count in [
    ('cell_type',ct_count),
    ('beta_subtype_coarse',ct_b_coarse_count),
    ('beta_subtype_fine',ct_b_fine_count)
]:
    count.to_excel(writer, sheet_name=sheet,index=True)   
writer.save()

# %%
path_tables+'cell_counts.xlsx'

# %%
