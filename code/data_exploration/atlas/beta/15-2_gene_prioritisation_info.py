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
# # Gene prioritisation
# Collect additional information about genes to prioritise them during results interpretation of other analyses (with focus on postnatal beta cells):
# - Expression strength (N cells, mean expression in expr cells)
# - Relative expression in beta cells
# - N gene sets (Recon, KEGG, GO)
# - N PubMed IDs

# %%
import scanpy as sc
import pandas as pd
import numpy as np
import pickle as pkl
import time

from sklearn.preprocessing import minmax_scale, maxabs_scale

import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import rcParams

from indra import literature

import sys
sys.path.insert(0, '/lustre/groups/ml01/code/karin.hrovatin/diabetes_analysis/')
import helper as h
import importlib
importlib.reload(h)
import helper as h

# %%
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()
# %load_ext rpy2.ipython

import rpy2.rinterface_lib.callbacks
import logging
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)

ro.r('library("hypeR")')
ro.r("source(paste(Sys.getenv('WSC'),'diabetes_analysis/data_exploration/','helper_hypeR.R',sep=''))")

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/'
path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'

# %%
adata_rn_b=sc.read(path_data+'data_rawnorm_integrated_analysed_beta_v1s1_sfintegrated.h5ad')
adata_rn_b.shape

# %%
# Remove genes expressed in less than 20 cells
sc.pp.filter_genes(adata_rn_b, min_cells=20)
adata_rn_b.shape

# %% [markdown]
# ## Make genes DF

# %%
# genes DF
genes=pd.DataFrame(index=adata_rn_b.var_names)

# %%
# gene symbol info
genes['gene_symbol']=adata_rn_b.var['gene_symbol']

# %% [markdown]
# ### Add relative beta expression
# Relative expression in beta cells compared to other cell types.

# %% [markdown]
# #### Relative across cts

# %%
# Load avg expression across cts
avg_expr_ct=pd.read_table('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/data_integrated_ctAvgScaled_sfintegrated.tsv',index_col=0)
avg_maxscl_expr_ct=pd.read_table('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/data_integrated_ctAvgMaxScaled_sfintegrated.tsv',index_col=0)

# %%
# Add relative expr info
genes['rel_beta_expr']=avg_expr_ct.loc[genes.index,'beta']
genes['rel_maxscl_beta_expr']=avg_maxscl_expr_ct.loc[genes.index,'beta']

# %% [markdown]
# #### Relative across cls
# Since cells may be heterogeneous within cell types also compute relative expression across finer clusters and then assign to beta cells the maximal relative expression across clusters that are predominately composed of beta cells.

# %%
# Load adata
adata_full=sc.read(path_data+'data_integrated_analysed.h5ad',backed='r')
# Load rawnorm. Correctly norm expression is in layers X_sf_integrated
adata_rawnorm=sc.read(path_data+'data_rawnorm_integrated_annotated.h5ad',backed='r')

# %% [markdown]
# ##### Cluster data

# %%
cl_col_full='leiden_r2'
ct_col='cell_type_integrated_v1'
adata_rawnorm.obs[cl_col_full]=adata_full.obs[cl_col_full]

# %%
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata_full,color=[ct_col,cl_col_full],s=20,wspace=0.6)

# %% [markdown]
# Most common ct in each cl and its ratio

# %%
# Most common ct ratio per cl
main_ct=adata_full.obs.groupby(cl_col_full).apply(lambda x: x[ct_col].value_counts(normalize=True
                                                                          ).head(n=1))
display(main_ct)

# %% [markdown]
# Other most common cts in cls that are not composed predominately of one ct.

# %%
# Most common cts in each cl that is not composed mainly of one ct
unclean_cls=[cl for cl,ct in main_ct[main_ct<0.9].index]
adata_full[adata_full.obs[cl_col_full].isin(unclean_cls),:].\
    obs.groupby(cl_col_full).apply(lambda x: x[ct_col].value_counts(normalize=True
                                                                          ).head(n=3))

# %% [markdown]
# C: Most cell clusters are mainly composed of single cell type. There are a few exceptions. Those that are non-beta are not that relevant. Those that are beta mainly seem to be well represented by the beta ct. Exception is cl 29 which may be low quality beta cells.
# Doublets will not be used as they are mixture of reall cell types. Embryo clusters will be removed from comparison as they for sure can not contribute to ambience in postnatal beta cells. 

# %%
# Map cl to most common ct
cl_ct_map=dict(main_ct.index)

# %% [markdown]
# #### Relative expression

# %%
# Prepare pb for DE genes

# Creat pseudobulk
adata_rawnorm.obs[cl_col_full]=adata_full.obs[cl_col_full]
xs1=[]
vnames1=[]
for group,data in adata_rawnorm.obs.groupby(cl_col_full):
    # Ignore mainly embryo cts
    if 'embryo' not in cl_ct_map[group]:
        xs1.append(np.array(adata_rawnorm[data.index,:
                                         ].layers['X_sf_integrated'].mean(axis=0)).ravel())
        # Make obs
        # make sure obss is str not int if clusters
        vnames1.append(group)
# Make DF of pb
xs1=np.array(xs1)
xs1=pd.DataFrame(np.array(xs1),columns=adata_rawnorm.var_names,index=vnames1)
xs1.index=xs1.index.map(cl_ct_map)

# %%
# Add to genes info DF
genes['rel_beta_expr_cl']=pd.DataFrame(minmax_scale(xs1[genes.index]),
                                      index=xs1.index,columns=genes.index).loc['beta',:].max()
genes['rel_maxscl_beta_expr_cl']=pd.DataFrame(maxabs_scale(xs1[genes.index]),
                                      index=xs1.index,columns=genes.index).loc['beta',:].max()

# %% [markdown]
# ### Add expression strength info
# n_cells - n beta cells expressing a gene (may be biased due to different population sizes across conditions)
#
# mean_expr_in_expr_cells - mean expression in cells that do not have zero expression (this aims to account for heterogeneous beta cell populations - e.g. some may not express a gene; but may not work for highly ambient genes as they are ambiently expressed all over)

# %%
# Expr stats
n_cells=pd.Series(np.asarray((adata_rn_b[:,genes.index].X.todense()!=0).sum(axis=0)).ravel(),
                  index=genes.index)
mean_expr_in_expr_cells=pd.Series(np.asarray(
    adata_rn_b[:,genes.index].X.sum(axis=0)).ravel(),index=genes.index)/n_cells

# %%
# Add expr stats
genes['n_cells']=n_cells[genes.index]
genes['mean_expr_in_expr_cells']=mean_expr_in_expr_cells[genes.index]

# %% [markdown]
# ### Gene set presence info
# How much is known about a gene - estimate based on N gene sets in which it is contained.

# %%
print('MSIGdb version:',ro.r(f'msigdb_version()'))
gene_sets_go=ro.r(f"msigdb_gsets_custom(species='Mus musculus',category='C5',subcategories=c('GO:BP','GO:CC','GO:MF'),size_range=c(5,500),filter_gene_sets=NULL,background=NULL)")
gene_sets_kegg=ro.r(f"msigdb_gsets_custom(species='Mus musculus',category='C2',subcategories=c('KEGG'),size_range=c(5,500),filter_gene_sets=NULL,background=NULL)")
gene_sets_reactome=ro.r(f"msigdb_gsets_custom(species='Mus musculus',category='C2',subcategories=c('REACTOME'),size_range=c(5,500),filter_gene_sets=NULL,background=NULL)")
print('GO:',len(gene_sets_go))
print('KEGG:',len(gene_sets_kegg))
print('Reactome:',len(gene_sets_reactome))

# %%
# Check in how many genes per ontology is gene
for ontology,gss in {'GO':gene_sets_go,'KEGG':gene_sets_kegg,'Reactome':gene_sets_reactome
                    }.items():
    gss=h.gs_to_dict(gss)
    for gene in genes.index:
        gene_symbol=genes.at[gene,'gene_symbol']
        count=0
        for gs in gss.values():
            if gene_symbol in gs:
                count+=1
        genes.at[gene,ontology]=count

# %% [markdown]
# ### N PubMed IDs associated with a gene
# Approximate how much is known about a gene based on how often it was cited.
#
# Coun all gene-related PMIDs or those that are also related to pancreas or diabetes.

# %% [markdown]
# #### Map to orthologues
# Mapping with indra requires human genes

# %%
# Load orthologues
orthologues=pd.read_table(path_genes+'orthologues_ORGmus_musculus_ORG2homo_sapiens_V103.tsv',
                          index_col=0)

# %% [markdown]
# Check if orthologue mapping is 1-to-1

# %%
# All orthologues
print('All genes are unique:',orthologues.index.nunique()==orthologues.index.shape[0])
print('N genes:',orthologues.index.shape[0],'N unique:',orthologues.index.nunique())

# %% [markdown]
# Check if orthologue mapping is 1-to-1 for expressed (here used) genes

# %%
# orthologues of expressed genes
# Some expressed genes fo not have orthologues
ortho_genes=orthologues.loc[[g for g in genes.index if g in orthologues.index],:]
print('All genes are unique:',ortho_genes.index.nunique()==ortho_genes.index.shape[0])
print('N genes:',ortho_genes.index.shape[0],'N unique:',ortho_genes.index.nunique())

# %% [markdown]
# Since orthologues are duplicated it needs to be assesed how to count PMIDs for them (see below).

# %% [markdown]
# #### Count PMIDs

# %%
# Dict for pmid sets
pmid_sets={'keys':{},'genes':{},'time':{}}

# %% [markdown]
# ##### Analyse publications related to pancreas
# Get PMIDs associated with specific pancreas-related term and compare endocrine and non-endocrine pancreas studies. This info was used to inform what could be useful for PMIDs filtering in the downstream gene-level analyses and for the filtering itself.

# %%
# Get pancreas and diabetes related PMIDs
pancreas_pmids=set()
for term in ['pancreas','pancreatic','diabetes','diabetic']:
    n_ids=literature.pubmed_client.get_id_count(term)
    pancreas_pmids.update(literature.pubmed_client.get_ids(term,retmax=n_ids))
pmid_sets['keys']['pancreas']=pancreas_pmids
pmid_sets['time']['pancreas']=time.ctime()
print('N pancreas-related PMIDs',len(pancreas_pmids))

# %% [markdown]
# Exclude some PMIDs, like cancer or exocrine

# %%
# Get cancer and pancreatic cancer related PMIDs
cancer_pmids=set()
for term in ['cancer','pancreatitis','carcinoma','cystic','adenocarcinoma',
             'lesions','tumor','tumors','benign','malignant','carcinogenesis',
            'neoplasm','metastases','metastasis']:
    n_ids=literature.pubmed_client.get_id_count(term)
    pmids=set(literature.pubmed_client.get_ids(term,retmax=n_ids))
    print(term,'overlapping with pancreas:',len(pmids&pancreas_pmids),'/',len(pmids))
    cancer_pmids.update(pmids)
pmid_sets['keys']['cancer']=cancer_pmids
pmid_sets['time']['cancer']=time.ctime()
print('N cancer-related PMIDs',len(cancer_pmids),
      'overlapping with pancreas:',len(cancer_pmids&pancreas_pmids))

# %%
# Get non-endocrine related PMIDs
nonendo_pmids=set()
for term in ['exocrine','endothelial','endothelium','ductal','duct','acinar','epithelial','epithelium']:
    n_ids=literature.pubmed_client.get_id_count(term)
    pmids=set(literature.pubmed_client.get_ids(term,retmax=n_ids))
    print(term,'overlapping with pancreas:',len(pmids&pancreas_pmids),'/',len(pmids))
    nonendo_pmids.update(pmids)
pmid_sets['keys']['nonendocrine']=nonendo_pmids
pmid_sets['time']['nonendocrine']=time.ctime()
print('N non-endocrine related PMIDs',len(nonendo_pmids),
      'overlapping with pancreas:',len(nonendo_pmids&pancreas_pmids))

# %%
# Get endocrine related PMIDs
maybe_endo_pmids=set()
for term in ['islet','alpha','beta','delta','gamma','epsilon']:
    n_ids=literature.pubmed_client.get_id_count(term)
    pmids=set(literature.pubmed_client.get_ids(term,retmax=n_ids))
    print(term,'overlapping with pancreas:',len(pmids&pancreas_pmids),'/',len(pmids))
    maybe_endo_pmids.update(pmids)
pmid_sets['keys']['endocrine']=maybe_endo_pmids&pancreas_pmids
print('N endocrine related PMIDs',len(pmid_sets['keys']['endocrine']))
print("N overlap endo and non-endo PMIDs out of all non-endo PMIDs %i/%i"%
     (len(pmid_sets['keys']['endocrine']&pmid_sets['keys']['nonendocrine']),
      len(pmid_sets['keys']['nonendocrine'])))
pmid_sets['keys']['nonendocrine_refined']=\
    pmid_sets['keys']['nonendocrine']-pmid_sets['keys']['endocrine']
pmid_sets['time']['nonendocrine_refined']=time.ctime()
print('N refined non-endo',len(pmid_sets['keys']['nonendocrine_refined']))

# %% [markdown]
# C: Some of non-endo mentioning PMIDs also mention endo.

# %% [markdown]
# ##### Get PubMed IDs associated with individual genes
# Extract information from currated entries with indra.

# %%
# retrieve PubMed IDs for each gene
# from indra: Get the curated set of articles for a gene in the Entrez database.
n_genes=genes.shape[0]
curr_gene_idx=0
for gene in genes.index:
    pmid_sets['genes'][gene]={}
    if gene in orthologues.index:
        #pmid_ns=[]
        #pmid_pancreas_ns=[]
        # Get orthologues
        orthologues_gene=orthologues.loc[gene,'Human gene name']
        if isinstance(orthologues_gene,str):
            orthologues_gene=[orthologues_gene]
            # For each orthologue get PMIDs
        for gene_hgnc in orthologues_gene:
            #print(gene_hgnc)
            # Some genes may not have valid HGNC names 
            try:
                pmids=set(literature.pubmed_client.get_ids_for_gene(gene_hgnc))
                pmid_sets['genes'][gene][gene_hgnc]=pmids
                 # Some genes may not have valid HGNC names thus no results are returned
            except ValueError:
                pass
    
    curr_gene_idx+=1
    if curr_gene_idx%500==0:
        print('Analysed %i/%i genes'% (curr_gene_idx,n_genes))
    pmid_sets['time']['genes']=time.ctime()

# %%
# Save pmid sets info
pkl.dump(pmid_sets,open(path_genes+'pmid_sets.pkl','wb'))

# %% [markdown]
# Does N orthologues and N total pmid citations correspond? E.g. can we use union across orthologues or should we use N pmids from orthologue with max PMIDs to account for potential effect of multiple orthologues on higher citation numbers.

# %%
citation_stats=[]
for gene,orthologues_gene in pmid_sets['genes'].items():
    if len(orthologues_gene)>0:
        n_pmids_union=len(set([pmid for pmids in orthologues_gene.values() for pmid in pmids]))
        n_pmids_max=max([len(pmids) for pmids in orthologues_gene.values()])
    else:
        n_pmids_union=0
        n_pmids_max=0
    citation_stats.append({
        'gene':gene,'gene_symbol':genes.at[gene,'gene_symbol'],
        'n_orthologues':len(orthologues_gene),
          'n_pmid_orthologue_union':n_pmids_union,
          'n_pmids_max':n_pmids_max,
                          })
citation_stats=pd.DataFrame(citation_stats)
citation_stats['max_vs_union']=citation_stats['n_pmids_max']/citation_stats['n_pmid_orthologue_union']

# %%
plt.hist(citation_stats['max_vs_union'])
plt.xlabel('max per-orthologue citations/union of citations')
plt.yscale('log')

# %%
print('n genes with citation sum>max:',(citation_stats['max_vs_union']<1).sum())

# %% [markdown]
# C: It seems that for most genes it would not matter if sum or max would be taken as the most cited orthologue is the only/mainly cited one.

# %% [markdown]
# Genes with lowest max/sum ratio - multiple orthologues strongly contributing to citations.

# %%
# Genes with lowest max/sum ratio
citation_stats.sort_values('max_vs_union').head(20)

# %% [markdown]
# Since orthologues that are all cited may have shared function the sum will be taken rather than max.

# %% [markdown]
# Cunt PMIDs and also make adjusted count that excludes publications that may not be relevant for our diabetes and beta cell dysfunction analyses (e.g. not related to pancreas or focused on exocrine or cancer).

# %%
# PMID counts
genes.drop([col for col in genes if 'N_PMID' in col],axis=1,inplace=True)
curr_gene_idx=0
for gene in genes.index:
    pmids_gene=set([pmid for pmids in list(pmid_sets['genes'][gene].values()) for pmid in pmids])
    genes.at[gene,'N_PMID']=len(pmids_gene)  
    genes.at[gene,'N_PMID_pancreas']=len(pmids_gene&pmid_sets['keys']['pancreas'])
    genes.at[gene,'N_PMID_pancreas_notCancer']=\
        len(pmids_gene&pmid_sets['keys']['pancreas']-pmid_sets['keys']['cancer'])
    genes.at[gene,'N_PMID_pancreas_notNonendo']=\
        len(pmids_gene&pmid_sets['keys']['pancreas']-pmid_sets['keys']['nonendocrine_refined'])
    genes.at[gene,'N_PMID_pancreas_notCancerNonendo']=\
        len(pmids_gene&pmid_sets['keys']['pancreas']-pmid_sets['keys']['cancer']\
            -pmid_sets['keys']['nonendocrine_refined'])
    curr_gene_idx+=1
    if curr_gene_idx%500==0:
        print('Analysed %i/%i genes'% (curr_gene_idx,n_genes))

# %% [markdown]
# Check how many genes have citations in each of the citation groups

# %%
# N genes having >0 citations in each category
(genes[[col for col in genes if 'N_PMID' in col]]>0).sum()

# %% [markdown]
# C: The  procedure for removing non-endo papers seems to work well, most of the removed genes overlap with cancer-removed genes.

# %% [markdown]
# Check how high rankl based on N_PMID or N_PMID_pancreas have genes extracted from pancreas literature (e.g. papers that I got for geting familiarised wit the field at the start of my PhD and then made notes on relevant genes). These genes are expected to be important and should thus be highly cited if we want to use this metric.

# %%
# Genes from lietarture - put in lowercase as names may be human/mouse, etc
# Not all names are according to nomenclature
gene_collection=pd.read_excel('/lustre/groups/ml01/workspace/karin.hrovatin//data/pancreas/gene_lists/summary.xlsx',sheet_name='genes')
markers=pd.read_excel('/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/markers.xlsx', 
                      sheet_name='mice')
collected_genes=set([g.lower() for g in gene_collection.Gene.tolist()+markers.Gene.tolist() 
                     if isinstance(g,str)])

# %% [markdown]
# Rank all genes by N PMIDs (with different filters) and then plot ranks distn for genes collected in my notes. Compare ranks distn on all genes and likely-relevant (notes) genes under different N PMID filtering regimes.

# %%
# Rank genes based on N_PMID
# Use lowercase symbols to match above
pmid_rank=pd.DataFrame({'rank_N_PMID':genes['N_PMID'].rank(ascending=False).values,
                        'rank_N_PMID_pancreas':genes['N_PMID_pancreas'].rank(ascending=False).values,
                       'rank_N_PMID_pancreas_notCancerNonendo':\
                        genes['N_PMID_pancreas_notCancerNonendo'].rank(ascending=False).values,
                       },index=[g.lower() if isinstance(g,str) else g for  g in genes.gene_symbol])

# %%
# Compare ranks of known genes with PMID or PMID_pancreas and with gene-wide distn of both
genes_known_overlap=[g for g in collected_genes if g in pmid_rank.index]
fig,ax=plt.subplots(3,1,figsize=(8,15),sharey=True,sharex=True)
ax2s=[]
for idx,col in enumerate(['rank_N_PMID','rank_N_PMID_pancreas',
                          'rank_N_PMID_pancreas_notCancerNonendo']):
    ax[idx].hist(pmid_rank.loc[:,col],bins=50,label='all genes')
    ax2=ax[idx].twinx()
    ax2s.append(ax2)
    ax2.hist(pmid_rank.loc[genes_known_overlap,col],label='known genes', 
             alpha=0.5,bins=50,color='r')
    ax[idx].set_title('Distn of gene ranks based on '+col)
    ax[idx].legend(loc='upper left')
    ax2.legend(loc='upper right')
max_y_ax2=max([ax2.get_ylim()[1] for a in ax2s])
for ax2 in ax2s:
    ax2.set_ylim(0,max_y_ax2)

# %% [markdown]
# C: N_PMID_pancreas has many more 0-s thus many genes have the same rank. But using pancreas/diabetes related PMIDs seems to shift the distn of known genes towards high ranks more strongly.
#
# C: Some known pancreas/diabetes genes were found in no papers with indra (lowest rank is 0 found PMIDs). This must be kept in mind when using this for prioritisation. 

# %% [markdown]
# Further look into "relevant" genes that got ranked as worst (e.g. when I was making the list of the relevant genes I was starting so I could have actually collected less relevant genes as well).

# %%
pmid_rank.loc[genes_known_overlap,'rank_N_PMID_pancreas'].sort_values(ascending=False)[:50]

# %% [markdown]
# C: Some important genes have very low rank (found in no studies related to diabetes/pancreas). There are also some genes with no studies at all (not limiting to pancreas) - may not have orthologues, their PMID entires are not currated well, ...

# %% [markdown]
# Some genes have less currated entries than all entries (e.g. genes may be mentioned in a paper, but this is not currated in the gene entry). Thus check (for random set of genes due to compute time contraints) how many entries they have in currated vs general serach.

# %%
# Compare N obtained PMIDs if using curated or uncurated gene search
n_pmids=[]
i=0
for gene in np.random.permutation(orthologues['Human gene name'].unique()):
    try:
        n_pmids.append({'n_currated':len(set(literature.pubmed_client.get_ids_for_gene(gene))),
                   'n_search':literature.pubmed_client.get_id_count(gene)})
        i+=1
    except ValueError:
        pass
    if i==100:
        break
n_pmids=pd.DataFrame(n_pmids)

# %%
g=sb.scatterplot(x='n_currated',y='n_search',data=n_pmids)
plt.plot([0,n_pmids['n_currated'].max()],[0,n_pmids['n_currated'].max()],c='k')

# %% [markdown]
# C: In some cases using uncurated serach produces many more paper associations. But most genes follow 1,1 line (black).
#
# Thus I will use the currated entries as they wil not be plagued by false positives.

# %% [markdown]
# ## Metrics comparison

# %% [markdown]
# Relationship between n_cells and relative beta expression.

# %%
sb.jointplot(genes['n_cells'],genes['rel_maxscl_beta_expr'],s=1)

# %% [markdown]
# C: Most highly expressed genes do not look like potential ambient genes.

# %% [markdown]
# Relationship between citations likely associated with endocrine compartment and n_cells where gene is expressed

# %%
sb.jointplot(genes['n_cells'],genes['N_PMID_pancreas_notCancerNonendo'],s=1)

# %% [markdown]
# C: It seems that even lowly expressed genes have many citations. Thus the citation counting is likely not biased by expression.

# %% [markdown]
# Different relative expression scaling startegies.

# %%
sb.jointplot(genes['rel_beta_expr'],genes['rel_maxscl_beta_expr'],s=1)

# %% [markdown]
# C: If using minmax instead of maxabs expr some genes would get much lower scores. Problem as minmax scales to [0,1] across cts so even if highly expressed in min expr ct (e.g. difference between highest and lowest expr ct is small) it will get lower score.

# %% [markdown]
# Relative expression across cell types or clusters.

# %%
sb.jointplot(genes['rel_maxscl_beta_expr'],genes['rel_maxscl_beta_expr_cl'],s=1)

# %% [markdown]
# C: Using expression scaled based on ct or cl gives much different results. Using cls is probably better as accounts for cell state heterogeneity.

# %% [markdown]
# ## Save

# %%
genes.to_csv(path_genes+'genePrioritisation_beta.tsv',sep='\t')

# %%
# Load
genes=pd.read_table(path_genes+'genePrioritisation_beta.tsv',index_col=0)

# %% [markdown]
# ## Is n_cells statistic representative also for state-specific genes
# The n_cells over all beta cells may be biased as cell states have different cell numbers. Compare n_cells (all cells) and ratio of cells in a given beta cell cluster that express the gene. 
#
# NOTE: This section can be run only runing the notebooks for defining fine beta cell clusters.

# %%
# Ratio of cells per cl expressing each gene
ratio_subtypes=pd.DataFrame(index=adata_rn_b.var_names)
for cl in sorted(adata_rn_b.obs.hc_gene_programs.unique()):
    n_cells=(adata_rn_b.obs.hc_gene_programs==cl).sum()
    ratio_subtypes['ratio_expr_cl'+str(cl)]=np.array(
        (adata_rn_b[adata_rn_b.obs.hc_gene_programs==cl,:].X>0).sum(axis=0)
                ).ravel()/n_cells

# %%
# Save
ratio_subtypes.to_csv(path_genes+'ratioExpr_betaSubtypes.tsv',sep='\t')

# %%
path_genes+'ratioExpr_betaSubtypes.tsv'

# %%
# Load
#ratio_subtypes=pd.read_table(path_genes+'ratioExpr_betaSubtypes.tsv',index_col=0)

# %% [markdown]
# Check how expression across clusters coresponds to expression across all beta cells

# %%
fig,ax=plt.subplots(ratio_subtypes.shape[1],1,figsize=(4,4*ratio_subtypes.shape[1]),
                    sharey=True,sharex=True)
for idx,cl in enumerate(ratio_subtypes.columns):
    ax[idx].scatter(genes.loc[ratio_subtypes.index,'n_cells'],ratio_subtypes[cl],s=0.1 )
    ax[idx].set_xlabel('n beta cells')
    ax[idx].set_ylabel(cl)

# %% [markdown]
# C: For most cell clusters expression cell ratio coresponds to all beta cells. However, for a few clusters this metric may be biased. This is mainly true for the cluster 5, which is low quality cluster (similar, but to lesser extent for cl4), so it is expected that genes are less expressed there than on the whole population. For other cluster some genes relatively highly expressed within the clusters may get a bit lower score on the whole beta cell population, but in general now extremely lower.

# %%
