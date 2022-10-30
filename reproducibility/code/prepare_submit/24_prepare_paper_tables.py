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
# # Merge multiple tables for the paper

# %%
import pandas as pd

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/'

# %%
path_tables='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/tables/paper/'

# %% [markdown]
# ## Endocrine markers (postnatal, embryo)

# %%
# Load table parts
res={}
res['postnatal_DE']=pd.read_excel(
    path_data+'combined/celltypes/DEedgeR_summaries_ctParsed.xlsx',sheet_name=None)
res['embryonic_DE']=pd.read_excel(
    path_data+'combined/celltypes_embryo/DEedgeR_summaries_ctParsed.xlsx',sheet_name=None)

# %%
res

# %%
# Save as single excel file
writer = pd.ExcelWriter(path_tables+'endocrine_markers.xlsx',engine='xlsxwriter') 
for age,res_sub1 in res.items():
    for ct,res_sub2 in res_sub1.items():
        res_sub2.to_excel(writer, sheet_name=age+'_'+ct.lower(),index=False)   
writer.save()

# %%
path_tables+'endocrine_markers.xlsx'

# %% [markdown]
# ## Human diabetes comparison

# %%
# Load table parts
res={}
res['T1D_DE']=pd.read_table(path_data+'de_human/diabetes/T1D_vs_healthy.tsv')
res['T2D_DE']=pd.read_table(path_data+'de_human/diabetes/T1D_vs_healthy.tsv')
for dt,res_sub in pd.read_excel(
    path_data+'de_human/diabetes/human_overlap_enrichment.xlsx',sheet_name=None,index_col=0).items():
    res_sub.rename({'label':'gene_set'},axis=1,inplace=True)
    res[f'{dt}_DE_enrichment']=res_sub
res['mouse_genesets_DE']=pd.read_table(
    path_data+'de_human/diabetes/mouse_models_genesets_lfcDvsH.tsv')

# %%
res

# %%
# Save single excel file
writer = pd.ExcelWriter(path_tables+'human_diabetes_de.xlsx',engine='xlsxwriter') 
for name,res_sub1 in res.items():
    res_sub1.to_excel(writer, sheet_name=name,index=False)   
writer.save()

# %%
path_tables+'human_diabetes_de.xlsx'

# %% [markdown]
# ## GPs

# %%
# Load table parts
res={}
res['GPs']=pd.read_table(path_data+'combined/moransi/sfintegrated/gene_hc_t2.4.tsv',
                         index_col=0)
res['GPs'].index.name='EID'
for gp,res_sub in pd.read_excel(
    path_data+'combined/moransi/sfintegrated/gene_hc_t2.4_enrichment.xlsx', 
    index_col=0,sheet_name=None).items():
    res_sub.index.name='gene_set'
    res_sub.drop('label',axis=1,inplace=True)
    res[f'GP{gp}_enrichment']=res_sub
res['sample_explained_var']=pd.read_table(
    path_data+'combined/moransi/sfintegrated/explained_var/per_sample/explainedVar_GPsample_significanceSummary_parsed.tsv',
    index_col=7)


# %%
res

# %%
# Save single excel file
writer = pd.ExcelWriter(path_tables+'GPs.xlsx',engine='xlsxwriter') 
for name,res_sub1 in res.items():
    res_sub1.to_excel(writer, sheet_name=name,index=True)   
writer.save()

# %%
path_tables+'GPs.xlsx'

# %% [markdown]
# ## Healthy GP

# %%
# Load table parts
res={}
res['GPs']=pd.read_table(path_data+'combined/moransi_healthy/sfintegrated/gene_hc_t3.tsv',
                         index_col=0)
res['GPs'].index.name='EID'
for gp,res_sub in pd.read_excel(
    path_data+'combined/moransi_healthy/sfintegrated/gene_hc_t3_enrichment.xlsx', 
    index_col=0,sheet_name=None).items():
    res_sub.index.name='gene_set'
    res_sub.drop('label',axis=1,inplace=True)
    res[f'GP{gp}_enrichment']=res_sub


# %%
res

# %%
# Save single excel file
writer = pd.ExcelWriter(path_tables+'GPs_healthy.xlsx',engine='xlsxwriter') 
for name,res_sub1 in res.items():
    res_sub1.to_excel(writer, sheet_name=name,index=True)   
writer.save()

# %%
path_tables+'GPs_healthy.xlsx'

# %% [markdown]
# ## Diabetes DE in beta cells

# %%
# Load table parts
res={}
for dt,subdir in [('T1D','de_diseased_T1_NODelim_meld'),
                  ('T2D','de_diseased_T2_VSGSTZ_meld_covarStudy')]:
    res[dt+'_DE']=pd.read_table(path_data+f'combined/de/{subdir}/deDataClusters.tsv',
              index_col=0)
    for hc,res_sub in pd.read_excel(
        path_data+f'combined/de/{subdir}/deDataClusters_enrichment.xlsx', 
        index_col=0,sheet_name=None).items():
        res_sub.index.name='gene_set'
        res_sub.drop('label',axis=1,inplace=True)
        res[f'{dt}_DE_{hc}_enrichment']=res_sub
for direction, res_sub in pd.read_excel(
    path_data+'combined/de/compare_T1D-NODelim_T2D-VSGSTZ/sharedClUpDown_enrichment.xlsx', 
    index_col=0,sheet_name=None).items():
    res_sub.index.name='gene_set'
    res_sub.drop('label',axis=1,inplace=True)
    res['sharedT1DT2D_DE_enrichment_'+direction]=res_sub

# %%
res

# %%
# Save single excel file
writer = pd.ExcelWriter(path_tables+'DE_diabetes.xlsx',engine='xlsxwriter') 
for name,res_sub1 in res.items():
    res_sub1.to_excel(writer, sheet_name=name,index=True)   
writer.save()

# %%
path_tables+'DE_diabetes.xlsx'

# %% [markdown]
# ## DE diabetes in endocrine cells

# %%
# Load table parts
res={}
for de,res_sub in pd.read_excel(path_data+'combined/de/deR_endocrine/endo_summaries.xlsx',
                                index_col=0,sheet_name=None).items():
    res_sub.index.name='EID'
    res[de+'_DE']=res_sub
for direction,res_sub in pd.read_excel(
    path_data+'combined/de/deR_endocrine/endoADG_shared_enrichment.xlsx', 
    index_col=0,sheet_name=None).items():
    res_sub.index.name='gene_set'
    res_sub.drop('label',axis=1,inplace=True)
    res[f'sharedADG_DE_enrichment_{direction}']=res_sub

# %%
res


# %%
# Save single excel file
writer = pd.ExcelWriter(path_tables+'DE_diabetes_endo.xlsx',engine='xlsxwriter') 
for name,res_sub1 in res.items():
    res_sub1.to_excel(writer, sheet_name=name,index=True)   
writer.save()

# %%
path_tables+'DE_diabetes_endo.xlsx'

# %% [markdown]
# ## DE sex

# %%
# Load table parts
res={}
res['P16']=pd.read_table(path_data+'combined/de/de_sexaging_covarSample/maleFemale_Fltp_P16_0.05_summary_sdFiltered.tsv',
                         index_col=0)
res['aged']=pd.read_table(path_data+'combined/de/de_sexaging_covarSample/deDataClusters.tsv',
                         index_col=0)

# %%
res

# %%
# Save single excel file
writer = pd.ExcelWriter(path_tables+'DE_sex_ages.xlsx',engine='xlsxwriter') 
for name,res_sub1 in res.items():
    res_sub1.to_excel(writer, sheet_name=name,index=True)   
writer.save()

# %%
path_tables+'DE_sex_ages.xlsx'

# %%
