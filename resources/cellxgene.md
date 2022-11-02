# cellxgene object

Link: https://cellxgene.cziscience.com/collections/296237e2-393d-4e31-b590-b03f74ac5070

## Field description

### Obs
- GEO_accession: GEO accession of each dataset.
- GP_*: Gene program (GP) activity score in beta cells. We collected genes variable across beta cell atlas subset and clustered them into GPs based on their co-expression.
- age: Age as defined in original publication, E - embryonic days, d - postnatal days, m - postnatal months, y - postnatal years.
- age_approxDays: Approximate mapping of age column to days for the purpose of visualisation.
- batch_integration: Batch used for integration.
- cell_cycle_phase: Phase of cell cycle (cyclone).
- cell_filtering: FACS sorting.
- cell_subtype_beta_coarse_reannotatedIntegrated: Beta cell subtype reannotation on integrated atlas. Coarse annotation based on metadata information. Abbreviations: NOD-D - NOD diabetic, M/F - male/female, chem - chem dataset, imm. - immature, lowQ - low quality, hMT - high mitochondrial fraction.
- cell_subtype_beta_fine_reannotatedIntegrated':'Beta cell subtype reannotation on '+\
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

