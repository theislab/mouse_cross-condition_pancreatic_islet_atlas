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
#     display_name: Python rpy2_3
#     language: python
#     name: rpy2_3
# ---

# %%
import scanpy as sc
import pandas as pd
import numpy as np

import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib import rcParams

from collections import defaultdict

from scipy.stats import mannwhitneyu

from sklearn.preprocessing import minmax_scale


import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()
# %load_ext rpy2.ipython

import rpy2.rinterface_lib.callbacks
import logging
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)

import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/')
import helper as h
import importlib
importlib.reload(h)
import helper as h

ro.r('library("hypeR")')
ro.r("source(paste(Sys.getenv('WSCL'),'diabetes_analysis/data_exploration/','helper_hypeR.R',sep=''))")
ro.r('library(pvclust)')


# %%
path_rna='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/'
path_data=path_rna+'combined/'
path_de=path_rna+'de_human/diabetes/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'

# %%
path_de

# %%
# Saving figures
path_fig='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/figures/paper/'

# %%
adata_rn_b=sc.read(path_data+'data_rawnorm_integrated_analysed_beta_v1s1_sfintegrated.h5ad')

# %% [markdown]
# ## DE in human T1D/T2D
# Perform DE betwee T1D/T2D and healthy donors (using all ages, sex, ...) on per-dataset basis. Find genes consistent across datasets.

# %%
# Do not test DE if either group has less than this many cells
min_cells=5
# Remove genes expressed in less than this ratio of cells
min_cells_ratio=0.1

# %%
# Compute OvR DE for markers
de_results=defaultdict(list)
for dataset,ddir in [('GSE83139','GSE83139/GEO/'),
                    ('GSE154126','GSE154126/GEO/'),
                    ('GSE101207','GSE101207/GEO/'),
                    ('GSE124742_GSE164875_patch','GSE124742_GSE164875/GEO/patch/'),
                    ('GSE124742_GSE164875_FACS','GSE124742_GSE164875/GEO/FACS/'),
                    ('GSE86469','GSE86469/GEO/'),
                    ('GSE81547','GSE81547/GEO/'),
                    ('GSE198623','P21000/sophie/human/'),
                    ('GSE81608','GSE81608/GEO/'),
                    ('GSE148073','GSE148073/GEO/')
                    ]:
    # Load adata
    print(dataset)
    adata_full=sc.read(path_rna+ddir+'adata_filtered.h5ad')
    for group in ['T1D','T2D']:
        print(group)
        # Subset to beta cells and disease phenotypes (removing pre pehnotypes and other diabetic)
        adata=adata_full[(adata_full.obs.cell_type=='beta').values& 
                    (adata_full.obs.disease.isin(['healthy',group])).values].copy()
        print(adata.shape)
        if 'gene_symbol' in adata.var.columns:
            adata.var_names=adata.var.gene_symbol

        # Only do DE if enough cells in each group
        cells_group=list(adata.obs.query('disease==@group').index)
        cells_rest=list(adata.obs.query('disease=="healthy"').index)
        print("N cells group %i and reference %i"%(len(cells_group),len(cells_rest)))
        if len(cells_group)>=min_cells and len(cells_rest)>=min_cells:

            # Remove genes expressed in less than N% cells in both groups individually
            ratio_group=np.asarray(
                (adata[cells_group,:].X>0).sum(axis=0)).ravel()/len(cells_group)
            mask_gene=np.logical_or(
                ratio_group>=min_cells_ratio,
                np.asarray(
                    (adata[cells_rest,:].X>0).sum(axis=0)).ravel()>=(len(cells_rest)*min_cells_ratio))
            # Subset data
            adata=adata[:,mask_gene].copy()

            # DE test
            key_added='rank_genes_'+group+'_vs_healthy'
            sc.tl.rank_genes_groups(adata, groupby='disease',
                                  use_raw=False, groups=[group], reference='healthy', 
                                  rankby_abs=True, method='t-test',
                                  key_added=key_added)

            de_res=pd.DataFrame({'gene_symbol':adata.uns[key_added]['names'][group],
                           'padj':adata.uns[key_added]['pvals_adj'][group],
                           'logFC':adata.uns[key_added]['logfoldchanges'][group]})
            # Add otehr info to results DF: 
            # dataset, n cells in each group, n cells in group expressing gene
            de_res['dataset']=dataset
            de_res['n_group']=len(cells_group)
            de_res['n_ref']=len(cells_rest)
            de_res['ratio_group']=ratio_group[mask_gene]
            display(de_res)
            de_results[group].append(de_res)
for group,data in de_results.items():
    de_results[group]=pd.concat(de_results[group])
    # Remove genes without gene symbol
    de_results[group]=de_results[group].query('gene_symbol!="nan"')
del adata
del adata_full

# %%
# Save
for group,data in de_results.items():
    data.to_csv(path_de+group+'_vs_healthy.tsv',index=False,sep='\t')

# %% [markdown]
# Report top genes in human: First sort by N datasets where they are signif up (based on padj and lFC threshold) and then by median lFC across all datasets.

# %%
# Top genes in human - T1D
group='T1D'
data=pd.DataFrame({'n_datasets':de_results[group].query('padj<0.25 & logFC>0.5').groupby('gene_symbol').size(),
 'm_lFC':de_results[group].groupby('gene_symbol')['logFC'].mean(),
 'm_ratio_group' :  de_results[group].groupby('gene_symbol')['ratio_group'].mean()}
            ).sort_values(['n_datasets','m_lFC'],ascending=[False,False])
min_datasets=de_results[group]['dataset'].nunique()/2.0
print('n genes signif in at least 50% of datasets:',
      data.query('n_datasets>=@min_datasets').shape[0])
display(data.head(20))

# %%
## Top genes in human - T2D
group='T2D'
data=pd.DataFrame({'n_datasets':de_results[group].query('padj<0.25 & logFC>0.5').groupby('gene_symbol').size(),
 'm_lFC':de_results[group].groupby('gene_symbol')['logFC'].mean(),
 'm_ratio_group' :  de_results[group].groupby('gene_symbol')['ratio_group'].mean()}
            ).sort_values(['n_datasets','m_lFC'],ascending=[False,False])
min_datasets=de_results[group]['dataset'].nunique()/2.0
print('n genes signif in at least 50% of datasets:',
      data.query('n_datasets>=@min_datasets').shape[0])
display(data.head(20))

# %% [markdown]
# ## Compare to mouse

# %% [markdown]
# ### Expression of individual genes
# Expression of consistently DE human genes in mouse - plot mouse dataset-wise helathy and diseased groups.

# %%
# Sample groups in mouse
group_map={
 'NOD_elimination_SRR7610295_8w':'NOD_elimination: healthy',
 'NOD_elimination_SRR7610296_8w':'NOD_elimination: healthy',
 'NOD_elimination_SRR7610297_8w':'NOD_elimination: healthy',
 'NOD_elimination_SRR7610298_14w':'NOD_elimination: diabetic',
 'NOD_elimination_SRR7610299_14w':'NOD_elimination: diabetic',
 'NOD_elimination_SRR7610300_14w':'NOD_elimination: diabetic',
 'NOD_elimination_SRR7610301_16w':'NOD_elimination: diabetic',
 'NOD_elimination_SRR7610302_16w':'NOD_elimination: diabetic',
 'NOD_elimination_SRR7610303_16w':'NOD_elimination: diabetic',
 'STZ_G1_control':'STZ: healthy',
 'STZ_G2_STZ':'STZ: diabetic',
 'VSG_MUC13633_chow_WT':'VSG: healthy',
 'VSG_MUC13634_chow_WT':'VSG: healthy',
 'VSG_MUC13639_sham_Lepr-/-':'VSG: diabetic',
 'VSG_MUC13641_sham_Lepr-/-':'VSG: diabetic',
}
adata_rn_b.obs['group']=adata_rn_b.obs.study_sample_design.map(group_map)

# %% [markdown]
# Extract all human genes that are signif in at least 50% of datasets, but filter so that each gene must be expressed in at least 10% of cells within any of the mouse groups to ensure that it is mouse-relevant gene.

# %%
# Orthologues
orthologues=pd.read_table(path_genes+'orthologues_ORGmus_musculus_ORG2homo_sapiens_V103.tsv'
                         ).rename(
    {'Gene name':'gs_mm','Human gene name':'gs_hs','Gene stable ID':'eid_mm'},axis=1)

# %%
# Plot markers
plot_genes={}
for group in de_results.keys():
    print(group)
    # Human gene prioritisation
    data=pd.DataFrame({'n_datasets':de_results[group].query('padj<0.25 & logFC>0.5').groupby('gene_symbol').size(),
     'm_lFC':de_results[group].groupby('gene_symbol')['logFC'].mean(),
     'm_ratio_group' :  de_results[group].groupby('gene_symbol')['ratio_group'].mean()}
                ).sort_values(['n_datasets','m_lFC'],ascending=[False,False])
    # Filter by N datasets
    min_datasets=de_results[group]['dataset'].nunique()/2.0
    data=data.query('n_datasets>=@min_datasets')
    
    genes_added=list()
    genes=list()
    # Translate to mouse
    for gene in data.index:
        added=False
        # Orthologues present
        eids=[e for e in orthologues.query('gs_hs==@gene')['eid_mm'] 
              if e in adata_rn_b.var_names]
        for eid in eids:
            # Expression in mouse in any group is high enough
            x=adata_rn_b[~adata_rn_b.obs.group.isna(),eid].to_df()
            x['group']=adata_rn_b.obs.loc[x.index,'group']
            if (x.groupby('group').apply(lambda x:(x>0).sum()/x.shape[0])>
                min_cells_ratio).any()[0]:
                genes.append(eid)
                added=True
        # Which human genes were added
        if added:
            genes_added.append(gene)
    # Gene info in human and dotplot
    display(data.loc[genes_added,:])
    # Save genes to be plotted in mice
    plot_genes[group]=genes


# %% [markdown]
# Plot human consistent genes on mouse data

# %%
# Plot on mouse
for group, genes in plot_genes.items():
    genes=adata_rn_b.var.loc[genes,'gene_symbol'].values
    sc.pl.dotplot(adata_rn_b[~adata_rn_b.obs.group.isna(),:],var_names=genes,
                  gene_symbols='gene_symbol',use_raw=False,
                  groupby='group',standard_scale='var',title=group)

# %% [markdown]
# C: It seems that human diabetes genes do not directly translate to mouse. Not sure if due to mouse-human differences or different metadata of samples (e.g. age, ...). But those genes that show difference between helathy and diseased seem to be in the right direction.

# %% [markdown]
# ### Gene set comparison
# Compare on level of gene sets instead - may be more robust as not looking at individual genes.
#
# Enrichment of human consietnet DE genes for gene sets - then use gene sets to score in mouse. We also added manually currated gene sets from literature.

# %% [markdown]
# #### Human DE enrichment
# Gene set enrichment of genes consistently DE in human

# %%
# Compute enrichment in human
enr_data={}
for group in ['T1D','T2D']:
    # Ref genes - must be separate for T1D and T2D
    ref=list(de_results[group].gene_symbol.unique())
    ro.globalenv['ref']=ref

    # Get gene sets
    print('MSIGdb version:',ro.r(f'msigdb_version()'))
    gene_sets_go=ro.r(f"msigdb_gsets_custom(species='Homo sapiens',category='C5',subcategories=c('GO:BP','GO:CC','GO:MF'),size_range=c(5,500),filter_gene_sets=NULL,background=ref)")
    gene_sets_kegg=ro.r(f"msigdb_gsets_custom(species='Homo sapiens',category='C2',subcategories=c('KEGG'),size_range=c(5,500),filter_gene_sets=NULL,background=ref)")
    gene_sets_reactome=ro.r(f"msigdb_gsets_custom(species='Homo sapiens',category='C2',subcategories=c('REACTOME'),size_range=c(5,500),filter_gene_sets=NULL,background=ref)")
    %R -i gene_sets_go -i gene_sets_kegg -i gene_sets_reactome -o gene_sets gene_sets=c(gene_sets_go,gene_sets_kegg,gene_sets_reactome)
    print('N gene sets:',len(gene_sets))
    ro.globalenv['gene_sets']=gene_sets
    
    enrich_fdr=0.25
    ro.globalenv['gs_fdr']=enrich_fdr
    
    data=pd.DataFrame({'n_datasets':de_results[group].query('padj<0.25 & logFC>0.5').groupby('gene_symbol').size(),
     'm_lFC':de_results[group].groupby('gene_symbol')['logFC'].mean(),
     'm_ratio_group' :  de_results[group].groupby('gene_symbol')['ratio_group'].mean()}
                )
    # Filter by N datasets
    min_datasets=de_results[group]['dataset'].nunique()/2.0
    genes=list(data.query('n_datasets>=@min_datasets').index)
    print('CL %s N genes %i'%(group,len(genes)))

    # Calculate enrichment
    ro.globalenv['genes']=genes
    res=ro.r(f'hypeR(signature=as.vector(unlist(genes)),genesets=gene_sets,test = "hypergeometric",background =  as.vector(unlist(ref)),pval = 1,fdr = gs_fdr,plotting = FALSE,quiet = TRUE)')
    ro.globalenv['res']=res
    data=ro.r(f'res$data')
    print('N enriched gene sets:',data.shape[0])
    enr_data[group]=data

    if data.shape[0]>0:
        # Plot top enriched gene sets
        print('Top enriched gene sets')
        data['recall']=data['overlap']/data['geneset']
        data['query_size']=len(genes)
        h.plot_enrich(data=data.rename(
            {'label':'name','fdr':'p_value','overlap':'intersection_size'},axis=1),
            n_terms=20, save=False,min_pval=10**-30, max_pval=enrich_fdr,percent_size=True,
               recall_lim=(0,1))
        h.plot_enr_heatmap(data=data,n_gs=None,xticklabels=False,yticklabels=True)

# %%
# Save enrichment tables
writer = pd.ExcelWriter(path_de+'human_overlap_enrichment.xlsx',
                        engine='xlsxwriter') 
for sheet,data in enr_data.items():
    data.to_excel(writer, sheet_name=str(sheet))   
writer.save()

# %% [markdown]
# #### Select gene sets
# Manually select relevant gene sets from enrichment results and literature

# %% [markdown]
# Extract gene sets without "background" filtering for scoring in mouse data to have full gene sets.

# %%
# Gene sets without ref as only for scoring
gene_sets_go=ro.r(f"msigdb_gsets_custom(species='Mus musculus',category='C5',subcategories=c('GO:BP','GO:CC','GO:MF'),size_range=c(5,500),filter_gene_sets=NULL,background=NULL)")
gene_sets_go_dict=dict(gene_sets_go.items())

# %% [markdown]
# Gene sets mentioned in other papers:
#
# T2D
#
# Fang 2019
# - ATP mt production, oxidative phosphorylation (also Segerstolpe 2016)
# - ribosome
# - hormone
# - glycolysis
# - hypoxia
# - proteasome
# - ER phagosome
#
# Wang 2016
# - dedif
#
# Camunas Soler 2020
# - compensatory
#
# T1D
#
# Camunas Soler 2020
# - immune
# - endosomal vacuolar
# - ER phagosome

# %% [markdown]
# Gene sets to analyse in mouse - containing both gene sets from our DE analysis as well as manually currated gene sets from other papers

# %%
gene_sets=[
    # Selected enriched gene sets (based on human genes DE in multiple datasets)
    # Some gene sets are shared in T1D and T2D, 
    # but based on other gene sets that share "hit" genes
    # it can be assumed what is the real reason for enrichment
    
    # T1D
    'GO:0042612_GOCC_MHC_CLASS_I_PROTEIN_COMPLEX',
    'GO:0035580_GOCC_SPECIFIC_GRANULE_LUMEN',
    'GO:0019730_GOBP_ANTIMICROBIAL_HUMORAL_RESPONSE',
    
    # T2D
    'GO:0032104_GOBP_REGULATION_OF_RESPONSE_TO_EXTRACELLULAR_STIMULUS',
    'GO:0030658_GOCC_TRANSPORT_VESICLE_MEMBRANE',
    'GO:0031018_GOBP_ENDOCRINE_PANCREAS_DEVELOPMENT',
    'GO:0042445_GOBP_HORMONE_METABOLIC_PROCESS',
    
    # External - from papers
    # T1D
    'GO:0098927_GOBP_VESICLE_MEDIATED_TRANSPORT_BETWEEN_ENDOSOMAL_COMPARTMENTS',
    
    #T2D
    'GO:0042776_GOBP_MITOCHONDRIAL_ATP_SYNTHESIS_COUPLED_PROTON_TRANSPORT',
    'GO:0006119_GOBP_OXIDATIVE_PHOSPHORYLATION',
    'GO:0042254_GOBP_RIBOSOME_BIOGENESIS',
    'GO:0006007_GOBP_GLUCOSE_CATABOLIC_PROCESS',
    'GO:1900037_GOBP_REGULATION_OF_CELLULAR_RESPONSE_TO_HYPOXIA',
    'GO:0010498_GOBP_PROTEASOMAL_PROTEIN_CATABOLIC_PROCESS'
]

# %% [markdown]
# Plot selected gene sets gene overlaps (given as overlap/len_smaller_gs)

# %%
# Overlap as ratio of smaller gene set
overlaps=pd.DataFrame(index=gene_sets,columns=gene_sets)
for i in range(len(gene_sets)-1):
    for j in range(i+1,len(gene_sets)):
        gi=gene_sets[i]
        genesi=set(gene_sets_go_dict[gi])
        gj=gene_sets[j]
        genesj=set(gene_sets_go_dict[gj])
        o=len(genesi&genesj)/min([len(genesi),len(genesj)])
        overlaps.at[gi,gj]=o
        overlaps.at[gj,gi]=o
overlaps=overlaps.fillna(1)


# %%
# plot overlap
sb.clustermap(overlaps,figsize=(5,5),
             xticklabels=True,yticklabels=True)

# %%
# N genes per gene set
for gs in gene_sets:
    print(gs,len(gene_sets_go_dict[gs]))

# %% [markdown]
# C: A few gene sets could potentially be dropped as they have a large overlap with anotehr gene sets.

# %% [markdown]
# #### Expression of gene sets in mouse

# %% [markdown]
# Prepare mouse data for plotting - add cell groups

# %%
# Subset data for scoring
adata_rn_b_sub=adata_rn_b[~adata_rn_b.obs.group.isna(),:].copy()

# %%
# Add healthy/diseased info for plot
adata_rn_b_sub.obs['status']=adata_rn_b_sub.obs['group'].apply(lambda x: x.split(': ')[1])
adata_rn_b_sub.obs['model']=adata_rn_b_sub.obs['group'].apply(lambda x: x.split(': ')[0])

# %% [markdown]
# Score mouse data for gene sets

# %%
# Compute scores
# First remove pre-computed scores
adata_rn_b_sub.obs.drop([c for c in adata_rn_b_sub.obs.columns 
                         if 'score_gs_' in c],axis=1,inplace=True)
for gs in gene_sets:
    sc.tl.score_genes(adata_rn_b_sub,
                      adata_rn_b_sub.var_names[adata_rn_b_sub.var.gene_symbol.isin(
                          gene_sets_go_dict[gs])],
                      score_name='score_gs_'+gs )
    
    # Alternative score computation - compute PC1 on genes and direct based on correlation
    # of mean expression of genes in each cell (as PCA not directed)
    # Dont use this as may increase batch effect - the orther score computed per cell, 
    # but this one tries to capture maximal var which could also be batch 
    # (could sometimes see much stronger differences between models even on healthy)
    # CHANGED - compute per dataset and then scale to 0,1 per dataset 
    # to make comparable across datasets
    # UNUSED - most scores were more distinct, but one looked odd - decided not to use 
    # as could be too affected by weights of individual genes - the genes
    # would be differently weighted across datasets
    
    #genes=gene_sets_go_dict[gs]
    #genes=adata_rn_b_sub.var.query('gene_symbol in @genes').index
    #for model in adata_rn_b_sub.obs['model'].unique():
    #    adata_temp=adata_rn_b_sub[adata_rn_b_sub.obs.model==model,genes]
    #    score=sc.pp.pca(data=adata_temp.to_df().values,n_comps=1)[:,0].ravel()
    #    score=score*np.corrcoef(
    #        score,adata_temp.to_df().values.mean(axis=1))[0,1]
    #    adata_rn_b_sub.obs.loc[adata_temp.obs_names,'score_gs_'+gs]=minmax_scale(score)

# %%
# Plot distn (no significance)
for gs in gene_sets:
    fig,ax=plt.subplots(figsize=(3,4))
    g=sb.violinplot(y='model',x='score_gs_'+gs,data=adata_rn_b_sub.obs,hue='status')
    plt.legend(bbox_to_anchor=(1,1))

# %%
# Plot distn with significance between healthy;/diab per model
for gs in gene_sets:
    fig,ax=plt.subplots(figsize=(4,4))
    g=sb.violinplot(y='model',x='score_gs_'+gs,data=adata_rn_b_sub.obs,hue='status')
    labels=[t.get_text() for t in ax.get_yticklabels()]
    for model in adata_rn_b_sub.obs.model.unique():
        g1=model+': diabetic'
        g2=model+': healthy'
        p=mannwhitneyu(adata_rn_b_sub.obs.query('group==@g1')['score_gs_'+gs],
                     adata_rn_b_sub.obs.query('group==@g2')['score_gs_'+gs])[1]
        g='D' if  adata_rn_b_sub.obs.query('group==@g1')['score_gs_'+gs].mean()>\
            adata_rn_b_sub.obs.query('group==@g2')['score_gs_'+gs].mean() else 'H'
        ax.annotate('p='+"{:.2e}".format(p)+' ('+g+')', 
                    xy=(ax.get_xlim()[1]+(ax.get_xlim()[1]-ax.get_xlim()[0])*0.1,
                        np.argwhere(np.array(labels)==model)[0][0]), 
                    zorder=10)
    ax.set_xlim(ax.get_xlim()[0],ax.get_xlim()[1]+(ax.get_xlim()[1]-ax.get_xlim()[0])*1.5)
    plt.legend(bbox_to_anchor=(1.5,1))

# %% [markdown]
# Selection of gene sets that are informative, plot for paper

# %%
gene_sets_sub=[
    'GO:0042612_GOCC_MHC_CLASS_I_PROTEIN_COMPLEX',
    'GO:0019730_GOBP_ANTIMICROBIAL_HUMORAL_RESPONSE',
    'GO:0032104_GOBP_REGULATION_OF_RESPONSE_TO_EXTRACELLULAR_STIMULUS',
    'GO:0030658_GOCC_TRANSPORT_VESICLE_MEMBRANE',
    'GO:0031018_GOBP_ENDOCRINE_PANCREAS_DEVELOPMENT',
    'GO:0042445_GOBP_HORMONE_METABOLIC_PROCESS',
    'GO:0006119_GOBP_OXIDATIVE_PHOSPHORYLATION',
    'GO:0042254_GOBP_RIBOSOME_BIOGENESIS',
    'GO:1900037_GOBP_REGULATION_OF_CELLULAR_RESPONSE_TO_HYPOXIA',
    'GO:0010498_GOBP_PROTEASOMAL_PROTEIN_CATABOLIC_PROCESS'
]

# %%
# Data for plotting
model_map={'NOD_elimination':'NOD','VSG':'db/db','STZ':'mSTZ'}
data=[]
gs_map={}
for gs in gene_sets_sub:
    d=adata_rn_b_sub.obs[['score_gs_'+gs,'model','status']].copy()
    d.rename({'score_gs_'+gs:'score'},axis=1,inplace=True)
    gs_name=' '.join(gs.split('_')[2:])
    d['gene_set']=gs_name
    gs_map[gs_name]=gs
    d['model']=d['model'].map(model_map)
    data.append(d)
data=pd.concat(data)

# %%
# Plot gene set scores across models
g=sb.catplot(y='model',x='score',col='gene_set',hue='status',data=data,kind='violin',
           sharex=False,aspect=.6,palette={'healthy':'#8a9e59','diabetic':'#c97fac'})
ylabels=[l.get_text() for l in g.axes[0][0].get_yticklabels()]
for ax in g.axes[0]:
    t=ax.get_title().replace('gene_set = ','')
    split_at=[0]
    l=20
    i=1
    while len(t)>i*l and ' ' in t[(i-1)*l:]:
        start=(i-1)*l
        end=i*l
        split_at.append((t[start:end].rfind(' ') if ' ' in t[start:end] else 0)+start)
        i+=1
    t_parts=[]
    if len(split_at)>1:
        for i in range(1,len(split_at)):
            t_parts.append(t[split_at[i-1]:split_at[i]])
        t_parts.append(t[split_at[i]:])
    else:
        t_parts=[t]
    ax.set_title('\n'.join(t_parts))   
    
    # Make transparent
    ax.set(facecolor = (0,0,0,0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.savefig(path_fig+'violin_beta_humanDiabeticGS_score.png',dpi=300,bbox_inches='tight')

# %% [markdown]
# For each gene set compute pval and lFCs of healthy vs diabetic mice model on per dataset level

# %%
# lFCs of scores per model
res=[]
for gs in gene_sets_sub:
    for model in adata_rn_b_sub.obs.model.unique():
        g1=model+': diabetic'
        g2=model+': healthy'
        # minmax scale score for each model as else scores may be negative 
        # producing errors in lFC
        score_model=pd.DataFrame(
            minmax_scale(adata_rn_b_sub.obs.query('model==@model')['score_gs_'+gs]),
            index=adata_rn_b_sub.obs.query('model==@model').index,
            columns=['score'])
        score_model['group']=adata_rn_b_sub.obs.query('model==@model').group
        # lFC of medians
        s1=score_model.query('group==@g1')['score']
        s2=score_model.query('group==@g2')['score']
        me_lfc=np.log( (s1.median())/   (s2.median()))
        # pval
        p=mannwhitneyu(s1,s2)[1]
        res.append({'model':model,'gene_set':gs,'me_logFC':me_lfc,'pval':p})
res=pd.DataFrame(res)
res['model']=res['model'].map(model_map)

# %%
# Gene set score DE results
res

# %%
# Save gs DE
res.to_csv(path_de+'mouse_models_genesets_lfcDvsH.tsv',index=False,sep='\t')
