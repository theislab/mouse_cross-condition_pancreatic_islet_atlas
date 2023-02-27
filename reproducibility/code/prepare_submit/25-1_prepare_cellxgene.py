# -*- coding: utf-8 -*-
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

from sklearn.preprocessing import minmax_scale

import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import rcParams

import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/')
from importlib import reload  
import helper
reload(helper)
import helper as h

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/'
path_save=path_data+'submission/cellxgene/'

# %%
adata_int=sc.read(path_data+'data_integrated_analysed.h5ad')

# %% [markdown]
# ## Prepare obs

# %% [markdown]
# ### Add info from integrated object

# %%
# Load integrated object obs
obs_a=sc.read(path_data+'data_integrated_analysed.h5ad',backed='r').obs.copy()

# %%
obs_a.columns

# %%
# Select and rename columns
cols=['study_sample', 'study_parsed', 'file', 'design',
      'phase_cyclone', #'s_cyclone','g2m_cyclone', 'g1_cyclone', 
      'ins_high', 'gcg_high','sst_high', 'ppy_high', 
      'cell_filtering', 'strain', 'age', 
      'pre_cell_type_unified','pre_cell_type_original',
      'cell_type_integrated_v2_parsed','low_q']
# Other cols to add parsed latter for CZI: 'tissue', 'technique', 
# Other colls to be parsed: 'cell_type_multiplet', 
obs=obs_a[cols].copy()
cols_rename={
    'study_sample':'batch_integration',
    'study_parsed':'dataset',
    'file':'sample',
    'phase_cyclone':'cell_cycle_phase',
    'pre_cell_type_original':'cell_type_originalDataset',
    'pre_cell_type_unified':'cell_type_originalDataset_unified',
    'cell_type_integrated_v2_parsed':'cell_type_reannotatedIntegrated'
}
obs.rename(cols_rename,axis=1,inplace=True)
obs.columns.to_list()

# %%
# Negate lowQ as it is stored in oposite boolean value
obs['low_q']=~obs['low_q']
obs['low_q'].sum()

# %%
# Re-add updated study and ct info if neccesary
if False:
    obs['dataset']=obs_a['study_parsed']
    obs['cell_type_originalDataset']=obs_a['pre_cell_type_original']
    obs['cell_type_originalDataset_unified']=obs_a['pre_cell_type_unified']
    obs['cell_type_reannotatedIntegrated']=obs_a['cell_type_integrated_v2_parsed']

# %%
# Convert to category
for col in [
     'batch_integration',
     'dataset',
     'sample',
     'design',
     'cell_cycle_phase',
     'cell_filtering',
     'strain',
     'age',
     'cell_type_originalDataset_unified',
     'cell_type_originalDataset',
     'cell_type_reannotatedIntegrated']:
    obs[col]=obs[col].astype('category')

# %%
# Study_design_sample
obs['dataset__design__sample']=['__'.join([da,de,s]) for da,de,s in 
                                zip(obs['dataset'],obs['design'],obs['sample'])]
obs['dataset__design__sample']=obs['dataset__design__sample'].astype('category')
# Remove sample as now contained elsewhere
obs.drop('sample',inplace=True,axis=1)

# %%
# Map age to days for visualisation
# Parse ages
e_days=21
ages_parsed_map={}
for a in obs['age'].unique():
    if a.endswith(' E'):
        a_parsed=str(float(a.split()[0])-e_days)+' d'
    elif '-' in a:
        a_parsed=str(np.mean([float(a) for a in a.split(' ')[0].split('-')]))+' '+a.split(' ')[1]
    else:
        a_parsed=a
    ages_parsed_map[a]=a_parsed
print('Remap ages:')
pprint.pprint(ages_parsed_map)
# Map age strings to approx days
age_unit_map={'y':365,'w':7,'m':30,'d':1}
age_map={age: float(age.split()[0])*age_unit_map[age.split()[1]] 
         for age in obs['age'].map(ages_parsed_map).unique()}
print('Map approx days:')
pprint.pprint(age_map)
obs['age_approxDays']=obs['age'].map(ages_parsed_map).map(age_map).dtype


# %% [markdown]
# Map different stress treatments to more readable cols than design

# %%
obs['diabetes_model']=obs_a['study_sample_design'].map({
     'NOD_elimination_SRR7610298_14w':'T1D_NOD',
     'NOD_elimination_SRR7610299_14w':'T1D_NOD',
     'NOD_elimination_SRR7610300_14w':'T1D_NOD',
     'NOD_elimination_SRR7610301_16w':'T1D_NOD',
     'NOD_elimination_SRR7610302_16w':'T1D_NOD',
     'NOD_elimination_SRR7610303_16w':'T1D_NOD',
     'NOD_elimination_SRR7610295_8w':'T1D_NOD_prediabetic',
     'NOD_elimination_SRR7610296_8w':'T1D_NOD_prediabetic',
     'NOD_elimination_SRR7610297_8w':'T1D_NOD_prediabetic',
     'NOD_SRR10985097_IRE1alphabeta-/-':'T1D_NOD_prediabetic',
     'NOD_SRR10985098_IRE1alphabeta-/-':'T1D_NOD_prediabetic',
     'NOD_SRR10985099_IRE1alphafl/fl':'T1D_NOD_prediabetic',
     'STZ_G4_STZ_GLP-1':'T2D_mSTZ-treated_GLP-1',
     'STZ_G8_STZ_GLP-1_estrogen+insulin':'T2D_mSTZ-treated_GLP-1_estrogen+insulin',
     'STZ_G6_STZ_GLP-1_estrogen':'T2D_mSTZ-treated_GLP-1_estrogen',
     'STZ_G2_STZ':'T2D_mSTZ',
     'STZ_G5_STZ_estrogen':'T2D_mSTZ-treated_estrogen',
     'STZ_G3_STZ_insulin':'T2D_mSTZ-treated_insulin',
     'VSG_MUC13631_PF_Lepr-/-':'T2D_db/db-treated_PairFeed',
     'VSG_MUC13632_PF_Lepr-/-':'T2D_db/db-treated_PairFeed',
     'VSG_MUC13640_VSG_Lepr-/-':'T2D_db/db-treated_VSG',
     'VSG_MUC13642_VSG_Lepr-/-':'T2D_db/db-treated_VSG',
     'VSG_MUC13639_sham_Lepr-/-':'T2D_db/db',
     'VSG_MUC13641_sham_Lepr-/-':'T2D_db/db'})
obs['diabetes_model']=obs['diabetes_model'].astype('category')

# %%
obs['chemical_stress']=obs_a['study_sample_design'].map({
 'spikein_drug_SRR10751503_A10_r1':'artemether_10uM',
 'spikein_drug_SRR10751508_A10_r2':'artemether_10uM',
 'spikein_drug_SRR10751513_A10_r3':'artemether_10uM',
 'spikein_drug_SRR10751502_A1_r1':'artemether_1uM',
 'spikein_drug_SRR10751507_A1_r2':'artemether_1uM',
 'spikein_drug_SRR10751512_A1_r3':'artemether_1uM',
 'spikein_drug_SRR10751505_FOXO_r1':'FoxOinhibitor',
 'spikein_drug_SRR10751510_FOXO_r2':'FoxOinhibitor',
 'spikein_drug_SRR10751515_FOXO_r3':'FoxOinhibitor',
 'spikein_drug_SRR10751506_GABA_r1':'GABA',
 'spikein_drug_SRR10751511_GABA_r2':'GABA',
 'spikein_drug_SRR10751516_GABA_r3':'GABA'})
obs['chemical_stress']=obs['chemical_stress'].astype('category')

# %% [markdown]
# Publication info

# %%
# GEO
obs['GEO_accession']=obs_a['study'].map({
    'Fltp_2y':'GSE211795', 
    'Fltp_adult':'GSE211796', 
    'Fltp_P16':'GSE161966', 
    'NOD':'GSE144471', 
    'NOD_elimination':'GSE117770', 
    'spikein_drug':'GSE142465 (GSM4228185 - GSM4228199)', 
    'embryo':'GSE132188', 
    'VSG':'GSE174194', 
    'STZ':'GSE128565'
})
obs['GEO_accession']=obs['GEO_accession'].astype('category')

# %% [markdown]
# ### Add QC info from preprocessing of individual datasets/samples

# %%
# Add emptyDrops LogProba
files=pd.read_table('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/raw_file_list.tsv')
files.index=[study+'_'+sample for study,sample in zip(files['study'],files['sample'])]
for study_sample in obs_a.study_sample.unique():
    file=files.at[study_sample,'dir']+'raw_'+files.at[study_sample,'ending']
    # Index used in raw and merged data
    index_raw=obs_a.index[obs_a.study_sample==study_sample]
    index_parsed=[
        idx.replace('-'+files.at[study_sample,'sample']+'-'+files.at[study_sample,'study'],'')
        for idx in index_raw]
    # Load ambient info
    obs.loc[index_raw,'emptyDrops_LogProb'
                     ]=sc.read(file,backed='r').obs.loc[index_parsed,'emptyDrops_LogProb'].values

# %%
# Scale empty drops proba per sample (as was computed) - gives on e.g. beta cells clearer scores
obs['emptyDrops_LogProb_scaled']=obs.groupby('batch_integration')['emptyDrops_LogProb'].apply(
        lambda x: pd.DataFrame(minmax_scale(x),index=x.index,columns=['emptyDrops_LogProb_scaled'])
).unstack()['emptyDrops_LogProb_scaled']
obs.drop('emptyDrops_LogProb',axis=1,inplace=True)

# %%
# Add other QC metrics from datasets
data=[('Fltp_2y','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islets_aged_fltp_iCre/rev6/'),
      ('Fltp_adult','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islet_fltp_headtail/rev4/'),
      ('Fltp_P16','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/salinno_project/rev4/'),
      ('NOD','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE144471/'),
      ('NOD_elimination','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE117770/'),
      ('spikein_drug','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE142465/'),
      ('embryo','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/GSE132188/rev7/'),
      ('VSG','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/VSG_PF_WT_cohort/rev7/'),
      ('STZ','/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/islet_glpest_lickert/rev7/')]
for study,path in data:
    #print(study)
    #Load annotation
    obs_sub=sc.read_h5ad(path+'data_annotated.h5ad',backed='r').obs[
        [ 'n_counts', 'n_genes', 'mt_frac', 'doublet_score']].copy()
    # parse idx to match integrated one
    obs_sub.index=[i+'-'+study for i in obs_sub.index]
    for col in obs_sub.columns:
        obs.loc[obs_sub.index,col]=obs_sub[col]


# %%
# Convert n_counts to log for visualisation
obs['log10_n_counts']=np.log10(obs['n_counts'])
obs.drop('n_counts',axis=1,inplace=True)

# %% [markdown]
# ### Annotation from cell subtype analyses

# %% [markdown]
# #### Other cell type subtypes

# %%
# Immune
obs_sub=sc.read(path_data+'data_rawnorm_integrated_analysed_immune.h5ad',
                backed='r').obs['cell_subtype_v1_parsed_coarse_v2'].copy()
obs.loc[obs_sub.index,'cell_subtype_immune_reannotatedIntegrated']=obs_sub

# %%
# Endothelial
obs_sub=sc.read(path_data+'data_rawnorm_integrated_analysed_endothelial.h5ad',
                backed='r').obs['cell_subtype_v1_parsed_coarse'].copy()
obs.loc[obs_sub.index,'cell_subtype_endothelial_reannotatedIntegrated']=obs_sub

# %% [markdown]
# #### Beta cell info

# %%
adata_rn_b=sc.read(path_data+'data_rawnorm_integrated_analysed_beta_v1s1_sfintegrated.h5ad')

# %%
# Add cell subtypes
obs.loc[adata_rn_b.obs_names,'cell_subtype_beta_coarse_reannotatedIntegrated']=adata_rn_b.obs[
    'leiden_r1.5_parsed']
obs.loc[adata_rn_b.obs_names,'cell_subtype_beta_fine_reannotatedIntegrated']=adata_rn_b.obs[
    'hc_gene_programs_parsed']

# %%
# Add gene program activity
genes_hc=pd.read_table('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/moransi/sfintegrated/gene_hc_t2.4.tsv',
                       sep='\t',index_col=0)
gene_cl='hc'
for ct in sorted(genes_hc[gene_cl].unique()):
    score_name='GP_'+str(ct)
    sc.tl.score_genes(adata_rn_b, 
                      gene_list=genes_hc.index[genes_hc[gene_cl]==ct], 
                     score_name=score_name, use_raw=False)
    obs.loc[adata_rn_b.obs_names,score_name]=adata_rn_b.obs[score_name]

# %% [markdown]
# ## Parse for cellxgene

# %% [markdown]
# Replace NA with nan

# %%
# Check if any cols contain 'NA'
[col for col in obs.columns if (obs[col]=='NA').any()]

# %%
for col in [col for col in obs.columns if (obs[col]=='NA').any()]:
    obs.loc[obs.index[obs[col]=='NA'],col]=np.nan

# %% [markdown]
# ### Required fields

# %%
obs['assay_ontology_term_id']=obs_a['technique'].map({
    'Chromium v2':'EFO:0009899', 
    'Chromium v3':'EFO:0009922',
    'Chromium v3.1':'EFO:0009922'}).astype('category')

# %%
obs['cell_type_ontology_term_id']=obs_a['cell_type_integrated_v2'].map({
     'acinar':'CL:0002064',
     'alpha':'CL:0000171',
     'alpha_beta':'CL:0008024',
     'alpha_delta':'CL:0008024',
     'ambient':'CL:0000000',
     'beta':'CL:0000169',
     'beta_delta':'CL:0008024',
     'beta_gamma':'CL:0008024',
     'delta':'CL:0000173',
     'delta_gamma':'CL:0008024',
     'ductal':'CL:0002079',
     'embryo':'CL:0000003',
     'embryo endocrine':'CL:0000003',
     'endocrine proliferative':'CL:0008024',
     'endothelial':'CL:0000115',
     'gamma':'CL:0002275',
     'immune':'CL:0000988',
     'schwann':'CL:0002573',
     'stellate_activated':'CL:0002410',
     'stellate_quiescent':'CL:0002410'
}).astype('category')

# %%
obs['development_stage_ontology_term_id']=obs['age'].map({
     '12.5 E':'MmusDv:0000027',
     '13.5 E':'MmusDv:0000028',
     '14 w':'MmusDv:0000063',
     '14.5 E':'MmusDv:0000029',
     '15.5 E':'MmusDv:0000032',
     '16 d':'MmusDv:0000037',
     '16 w':'MmusDv:0000063',
     '182 d':'MmusDv:0000077',
     '2 y':'MmusDv:0000091',
     '2-3 m':'MmusDv:0000062',
     '16-18 w':'MmusDv:0000064',
     '4 m':'MmusDv:0000064',
     '5 w':'MmusDv:0000049',
     '8 w':'MmusDv:0000052'
}).astype('category')

# %%
# Disease
# Would be best not to annotate disease due to model, disease progression, 
# treatment, and chem stress
obs['disease_ontology_term_id']=obs_a['study_sample_design'].map({
     'NOD_elimination_SRR7610298_14w': 'MONDO:0005147',
     'NOD_elimination_SRR7610299_14w': 'MONDO:0005147',
     'NOD_elimination_SRR7610300_14w': 'MONDO:0005147',
     'NOD_elimination_SRR7610301_16w': 'MONDO:0005147',
     'NOD_elimination_SRR7610302_16w': 'MONDO:0005147',
     'NOD_elimination_SRR7610303_16w': 'MONDO:0005147',
     'STZ_G4_STZ_GLP-1': 'MONDO:0005148',
     'STZ_G8_STZ_GLP-1_estrogen+insulin': 'MONDO:0005148',
     'STZ_G6_STZ_GLP-1_estrogen': 'MONDO:0005148',
     'STZ_G2_STZ': 'MONDO:0005148',
     'STZ_G5_STZ_estrogen': 'MONDO:0005148',
     'STZ_G3_STZ_insulin': 'MONDO:0005148',
     'VSG_MUC13631_PF_Lepr-/-': 'MONDO:0005148',
     'VSG_MUC13632_PF_Lepr-/-': 'MONDO:0005148',
     'VSG_MUC13640_VSG_Lepr-/-': 'MONDO:0005148',
     'VSG_MUC13642_VSG_Lepr-/-': 'MONDO:0005148',
     'VSG_MUC13639_sham_Lepr-/-': 'MONDO:0005148',
     'VSG_MUC13641_sham_Lepr-/-': 'MONDO:0005148',
     'spikein_drug_SRR10751503_A10_r1': 'MONDO:0001933',
     'spikein_drug_SRR10751508_A10_r2': 'MONDO:0001933',
     'spikein_drug_SRR10751513_A10_r3': 'MONDO:0001933',
     'spikein_drug_SRR10751502_A1_r1': 'MONDO:0001933',
     'spikein_drug_SRR10751507_A1_r2': 'MONDO:0001933',
     'spikein_drug_SRR10751512_A1_r3': 'MONDO:0001933',
     'spikein_drug_SRR10751505_FOXO_r1': 'MONDO:0001933',
     'spikein_drug_SRR10751510_FOXO_r2': 'MONDO:0001933',
     'spikein_drug_SRR10751515_FOXO_r3': 'MONDO:0001933',
     'spikein_drug_SRR10751506_GABA_r1': 'MONDO:0001933',
     'spikein_drug_SRR10751511_GABA_r2': 'MONDO:0001933',
     'spikein_drug_SRR10751516_GABA_r3': 'MONDO:0001933'
}).fillna('PATO:0000461').astype('category')

# %%
obs['self_reported_ethnicity_ontology_term_id']='na'
obs['self_reported_ethnicity_ontology_term_id']=obs['self_reported_ethnicity_ontology_term_id'].astype('category')

# %%
obs['is_primary_data']=True

# %%
obs['organism_ontology_term_id']='NCBITaxon:10090'
obs['organism_ontology_term_id']=obs['organism_ontology_term_id'].astype('category')

# %%
obs['sex_ontology_term_id']=obs_a['sex'].map({
    'female':'PATO:0000383',
    'male':'PATO:0000384'
}).astype('category')
# Also add info on sex annotation
obs['sex_annotation']=obs_a['study'].map({
    'Fltp_2y':'data-driven', 
    'Fltp_adult':'ground-truth', 
    'Fltp_P16':'data-driven', 
    'NOD':'ground-truth', 
    'NOD_elimination':'ground-truth', 
    'spikein_drug':'ground-truth', 
    'embryo':'data-driven', 
    'VSG':'ground-truth', 
    'STZ':'ground-truth'
})
obs['sex_annotation']=obs['sex_annotation'].astype('category')

# %%
obs['tissue_ontology_term_id']='UBERON:0000006'
obs.loc[obs_a[obs_a.study=='embryo'].index,'tissue_ontology_term_id']='UBERON:0001264'
obs['tissue_ontology_term_id']=obs['tissue_ontology_term_id'].astype('category')

# %%
obs['donor_id']=obs_a.loc[obs.index,:].apply(
    lambda x:'mouse_pancreatic_islet_atlas_Hrovatin__'+x['study']+'__'+x['file'], axis=1)
obs['donor_id']=obs['donor_id'].astype('category')

# %%
obs['suspension_type']='cell'
obs['suspension_type']=obs['suspension_type'].astype('category')

# %% [markdown]
# ### Examine the prepared obs

# %%
# List columns and their content
for col in sorted(obs.columns):
    print('\n************')
    print(col)
    print(obs[col].dtype)
    if obs[col].nunique()<100:
        print('\n',sorted(obs[col].astype(str).unique()))

# %% [markdown]
# ### Save

# %%
obs.to_csv(path_save+'obs.tsv',sep='\t')

# %% [markdown]
# ## Prepare adata X, var, and obsm

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


# %%
genes_anno=pd.read_table('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/genomeAnno_ORG'+\
    'mus_musculus_V103.tsv',index_col=0)

# %%
# Load all raw adatas
adatas={}
genes_study={}
for study,path in data_study:
    print(study)
    adata_sub=sc.read(path+'data_normlisedForIntegration.h5ad').raw.to_adata()
    print(adata_sub.shape)
    # Map symbols to EIDs
    genes_anno_sub=genes_anno.copy()
    genes_anno_sub['EID']=genes_anno_sub.index
    # make sure no nan symbols (not really expected)
    if adata_sub.var_names.isna().any():
        raise ValueError('Some var names are na')
    # map symbols to eids based on per-study map (accounting for modified symbols in adata)
    genes_anno_sub.index=genes_anno_sub['gene_symbol_'+study]
    eid_symbols=set(genes_anno_sub.index)
    eid_vars=[v for v in adata_sub.var_names if v in eid_symbols]
    adata_sub=adata_sub[:,eid_vars]
    print('Shape of valid symbols',adata_sub.shape)
    adata_sub.var_names=genes_anno_sub.loc[adata_sub.var_names,'EID']
    # Genes present in adata
    genes_study[study]=set(adata_sub.var_names)
    # Save adata
    adatas[study]=adata_sub
# Join adata objects
adata = sc.concat( adatas,   join='outer',index_unique='-')
print('Full:',adata.shape)
print('N all genes:',len(set().union(*genes_study.values())))

del adatas
del adata_sub
gc.collect()

# %%
# Which gene is present in which study
for study,genes in genes_study.items():
    adata.var.loc[genes,'present_'+study]=True
    adata.var['present_'+study].fillna(False,inplace=True)

# %%
# Filter out features not present in all adatas (keep in raw)
adata.raw=adata.copy()
# Set filtered genes to 0
genes_intersect=set.intersection(*genes_study.values())
print("N intersecting genes:",len(genes_intersect))
adata.var['feature_is_filtered']=[g not in genes_intersect for g in adata.var_names]
adata[:,adata.var['feature_is_filtered']].X=0

# %%
# Normalize X
adata.obs['size_factors_integrated']=obs_a.loc[adata.obs_names,'size_factors_integrated']
adata=h.get_rawnormalised(adata,sf_col='size_factors_integrated',
                    use_log=True,save_nonlog=False, use_raw=False)
del adata.uns['log1p']

# %%
# Add obsm
adata.obsm['X_integrated_umap']=sc.read(path_data+'data_integrated_analysed.h5ad',
                                        backed='r')[adata.obs_names,:].obsm['X_umap']

# %%
# Addd obsm of beta cells only
adata_temp=sc.read(path_data+'data_rawnorm_integrated_analysed_beta_v1s1_sfintegrated.h5ad',
                   backed='r')
embed=pd.DataFrame(adata_temp.obsm['X_umap_opt'],index=adata_temp.obs_names
                  ).reindex(adata.obs_names)
adata.obsm['X_integrated_umap_beta']=embed.values
del adata_temp
gc.collect()

# %% [markdown]
# ## Add obs to adata

# %%
# reload obs
obs=pd.read_table(path_save+'obs.tsv',index_col=0)

# %%
# Add to adta
adata.obs=obs.loc[adata.obs_names,:]

# %% [markdown]
# ## Prepare uns

# %%
adata.uns['schema_version']="3.0.0"

# %%
adata.uns['title']='Mouse pancreatic islet scRNA-seq atlas across sexes, ages, and stress conditions including diabetes'

# %%
adata.uns['batch_condition']=['batch_integration']

# %%
adata.uns['default_embedding']='X_integrated_umap'

# %%
# Description of individual columns
adata.uns['field_descriptions']={
    'obs':{
         'GEO_accession':'GEO accession of each dataset',
         'GP_*':'Gene program (GP) activity score in beta cells.'+\
                'We collected genes variable across beta cell atlas subset '+\
                'and clustered them into GPs based on their co-expression.',
         'age':"Age as defined in original publication, E - embryonic days, "+\
                "d - postnatal days, m - postnatal months, y - postnatal years",
         'age_approxDays':"Approximate mapping of age column to days for "+\
                "the purpose of visualisation",
         'batch_integration':'Batch used for integration',
         'cell_cycle_phase':'Phase of cell cycle (cyclone)',
         'cell_filtering':"FACS sorting",
         'cell_subtype_beta_coarse_reannotatedIntegrated':'Beta cell subtype '+\
                'reannotation on integrated atlas. Coarse annotation based on '+\
                'metadata information. Abbreviations: NOD-D - NOD diabetic, M/F - male/female, '+\
                'chem - chem dataset, imm. - immature, lowQ - low quality, '+\
                'hMT - high mitochondrial fraction',
         'cell_subtype_beta_fine_reannotatedIntegrated':'Beta cell subtype reannotation on '+\
                'integrated atlas. Fine annotation aimed at capturing all biollogically '+\
                'distinct beta cell subtypes (assesed based on gene program activity patterns).' +\
                'Abbreviations: D-inter. - diabetic intermediate, NOD-D - NOD diabetic, '+\
                'M/F - male/female, chem - chem dataset, imm. - immature, lowQ - low quality.',
         'cell_subtype_endothelial_reannotatedIntegrated':'Endothelial cell subtype '+\
                'reannotation on integrated atlas based on known markers',
         'cell_subtype_immune_reannotatedIntegrated':'Immune cell subtype reannotation on '+\
                'integrated atlas based on known markers',
         'cell_type_originalDataset':"Cell types as reported in the studies that generated "+\
                "the datasets",
         'cell_type_originalDataset_unified':"Cell types as reported in the studies that "+\
                "generated the datasets; manually unified to a common naming scheme. "+\
                'Abberviations: E - embryonic, EP - endocrine progenitor/precursor, '+\
                'Fev+ - Fev positive, prolif. - proliferative, "+"" symbol - likely doublet',
         'cell_type_reannotatedIntegrated':'Cell type reannotation on integrated atlas. '+\
                'Abbreviations: E - embryonic, endo. - endocrine, "+"" symbol - likely doublet, '+\
                'prolif. - proliferative, lowQ - low quality, '+\
                'stellate a./q. - stellate activated/quiescent',
         'chemical_stress':'Application of chemicals to islets',
         'dataset':'Dataset comprised of multiple samples that were generated/published together',
         'dataset__design__sample':'Concatentation of multiple columns with sample information',
         'design':'Brief sample description that gives information on differences '+\
                'between samples within dataset',
         'diabetes_model':'Diabetes model and any diabetes treatment',
         'doublet_score':'Scrublet doublet scores computed per sample; '+\
                'higher - more likely doublet',
         'emptyDrops_LogProb_scaled':'Log probability that droplet is empty computed '+\
                'per sample with emptyDrops and scaled to [0,1] per sample; '+\
                'higher - more likely empty droplet',
         '*_high':'Do cells have high expression of the given hormone (ins, gsg, sst, ppy)'+\
                ', determined per sample',
         'log10_n_counts':'log10(N counts)',
         'low_q':'True for cells asigned to low quality clusters',
         'mt_frac':'Fraction of mitochondrial genes expression',
         'n_genes':'Number of expressed genes',
         'sample':'Technical sample, in some cases equal to biological sample',
         'sex_annotation':'Was sex known from sample metadata (ground-truth) or '+\
                'was it determined bsed on Y-chromosomal gene expression (data-driven)',
         'strain':'Mouse strain and genetic background',
         'donor_id':'This is ID of a sample and not donor. Some samples were pooled '+\
                 'accross animals.',
    },
    'var':{
        'present_*':'Was gene present in the genome version used for count matrix generation '+\
                'of given dataset',
    },
    'obsm':{
        'X_integrated_umap':'UMAP computed on integrated embedding', 
        'X_integrated_umap_beta':'UMAP computed on integrated embedding of beta cell subset'
    },
    'X':{
        'X_normalization':'Joint scran normalisation on integrated embedding followed'+\
                'by log(expr+1) transformation',
    }
}

# %% [markdown]
# ## Cell ordering
# Randomly order cells to improve plotting

# %%
random_indices=np.random.permutation(list(range(adata.shape[0])))
adata=adata[random_indices,:]

# %% [markdown]
# ## Save

# %%
adata

# %%
adata.write(path_save+'adata.h5ad')

# %%
path_save+'adata.h5ad'

# %% [markdown]
# ## Remove genes not in cellxgene genome version

# %% [raw]
# In terminal:
# conda activate cellxgene # env with cellxgene formatting package
# cd /lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/submission/cellxgene
# cellxgene-schema validate adata.h5ad
#
# # Get invalid genes (saved to file)
# cellxgene-schema validate adata.h5ad &> invalid_eids.txt
# cat invalid_eids.txt |grep "is not a valid feature ID in" > invalid_eids.txt

# %%
# Eids not accepted by cellxgene
invalid_eids=pd.read_table(path_save+'invalid_eids.txt',sep=' ',header=None,usecols=[1,9],
                           quotechar="'")
invalid_eids.columns=['eid','obj']

# %%
# Invalid eids should be the same for var and raw.var
if not all(invalid_eids.query('obj=="var."')['eid'].values==\
    invalid_eids.query('obj=="raw.var."')['eid'].values):
    raise ValueError('Genes to be removed not same in var and raw.var')
if not all(adata.var_names==adata.raw.var_names):
    raise ValueError('Var and raw.var not matching')

# %%
# Removes unacceptable genes
# Assumes the same in var and raw.var (checked above) reported as err and present in adata
eids_remove=set(invalid_eids['eid'].unique())
print("N eids to remove:",len(eids_remove))
eids_keep=[e for e in adata.var_names if e not in eids_remove]
# remove from adata
adata_filtered=adata[:,eids_keep].copy()
adata_filtered.raw=adata.raw.to_adata()[:,eids_keep].copy()
print('adata',adata_filtered.shape,'raw',adata_filtered.raw.shape)

# %% [markdown]
# #### Save filtered adata

# %%
adata_filtered

# %%
adata_filtered.write(path_save+'adata_filtered.h5ad')

# %% [markdown]
# ### Validation of final object

# %% [raw]
# In terminal:
# conda activate cellxgene
# cd /lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/submission/cellxgene
# cellxgene-schema validate adata_filtered.h5ad

# %% [markdown]
# ## Other info
#
# - Title: Mouse pancreatic islet scRNA-seq atlas across sexes, ages, and stress conditions including diabetes
# - Description: To better understand pancreatic β-cell heterogeneity we generated a mouse pancreatic islet atlas capturing a wide range of biological conditions. The atlas contains scRNA-seq datasets of over 300,000 mouse pancreatic islet cells, of which more than 100,000 are β-cells, from nine datasets with 56 samples, including two previously unpublished datasets. The samples vary in sex, age (ranging from embryonic to aged), chemical stress, and disease status (including T1D NOD model development and two T2D models, mSTZ and db/db) together with different diabetes treatments. Additional information about data fields is availiable in adata uns field 'field_descriptions'.
# - Contact: Karin Hrovatin, karin.hrovatin@helmholtz-muenchen.de
# - Publication/preprint DOI: TBD
# - URLs: 
#     - GitHub: https://github.com/theislab/mm_pancreas_atlas_rep/
#     - GEO: GSE211799

# %%
