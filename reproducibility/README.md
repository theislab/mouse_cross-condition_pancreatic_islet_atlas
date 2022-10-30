# mm_pancreas_atlas_rep

Reproducibility code for mouse pancreatic islet scRNA-seq atlas from Hrovatin et al.


## Processing steps:

All processing code files are in the directory code. For additional instructions on how to run the code see Notes below.

### Analysis steps

Preprocessing:
- 1: Data preprocessing for annotation - per-dataset ambient genes, QC, normalisation, HVGs (preprocessing/1_*) <sup>1</sup>
 - 1.1: Compute empty droplet score of each cell (preprocessing/1-1_ambient_scores*)
- 2: Data annotation - for each dataset add UMAP, info on cell cycle, sex (if mixed), and for some also endocrine marker scores, cell types, and cell subtypes (preprocessing/2*) <sup>1, 2</sup>
- 3: Preparation of datasets used for atlas validation (not part of the atlas - called "external")
 - 3.1: Prepare normalised expression data and currate metadata for mouse and human (prepprocessing/3-1*)
 - 3.2: Collect metadata of all datasets into a single table (prepprocessing/3-2*)
- 4: Ambient correction - correct data and analyse correction results for data used in final integration. For each method there is a txt file (4-\*-0) that contains ordered commands for running different part of workflow.
 - 4.1: CellBender (preprocessing/4-1*). 
 - 4.2: DecontX (preprocessing/4-2*)
 - 4.3: SoupX (preprocessing/4-3*)
- 5: Preprocess for integration - prepare expression data for test and final integrations (prepprocessing/5*)
 - 5.1: Prepare data for integration on a smaller annotated data subset only. For testing out hyperparameters (prepprocessing/5-1*)
 - 5.2: Prepare data for full atlas integration:  prepare study data, including per sample normalization, etc. (prepprocessing/5-2-1*) and add EIDs to data preprocessed for integration (prepprocessing/5-2-2*)
 
Integration:
- 6: Integration (integration/6*)
 - 6.1: scVI - select hyperparams on ref data and then train model on whole data, including different ambient preprocessing (integration/scVI/6-1*)
 - 6.2: scArches - select hyperparams on ref data and then train model on whole data, including different ambient preprocessing or using only beta cell subset (integration/scVI/6-2* scArches). 
  - We also tried maping the non-ref data subset (external or diabetic samples) to the selected best ref subset integration (only non-diabetic in house samples), but the reference mapping performance was never of comparable quality to de novo integration of all datasets, possibly due to low complexity of the reference (the ref dataset). The code for mapping on top of the ref integration is in 6-2-2_scArches_addToRef.ipynb, however, we do not show the scIB results in the evaluation summary plots (described below) as this was not used in the paper due to the above mentioned poor performance.
 - 6.3: Evaluation with scIB metrics - compute evaluation metrics on different data subsets (e.g. all cells, annotated cells, beta cells) and compare results across integration runs (integration/6-3*)
 - 6.4: Prepare integrated adata
   - 6.4.1: Comparison of gene annotation across datasets needed to merge all samples into a single adata (preprocessing/6-4-1*)
   - 6.4.2: Merge datasets into a single integrated object (integration/6-4-2*)
  
Atlas-level analyses of integrated data:
- 7: Cell type re-annotation on integrated data 
 - 7.1: Perform annotation (data_exploration/atlas/7-1*)
 - 7.2: Compare to per-dataset annotation (data_exploration/atlas/7-2*)
- 8: Re-normalise integrated data with Scran, using integrated clusters (data_exploreation/atlas/8*)
- 9: Cell subtype annotation of non-beta cells
 - 9.1: Assign a potential low quality cluster in alpha cells for downsteram plots without potential low quality cells (data_exploration/atlas/9-1*)
 - 9.2: Cell subtype annotation of immune and endothelial cells (data_exploration/atlas/9-2*)
- 10: Summary plot of sample metadata (data_exploration/atlas/10*)
- 11: Analysis of ambient genes:
 - 11.1: Compute top ambient genes comming from postnatal non-beta cells (data_exploration/atlas/11.1*)
 - 11.2: Ambient contribution of top ambient genes across samples (preprocessing/11-2*)
- 12: Endcorine markers 
 - 12.1: Compute postnatal and embyonic markers (data_exploration/atlas/12-1*)
 - 12.2: Comparison of postnatal and embryonic markers (data_exploration/atlas/12-2*)
- 13: Embedding localisation of endocrine embryonic and postnatal cells. Analysis of why embryonic delta cells are mapping to the postnatal region of the integrated embedding. (data_exploration/atlas/13*)
- 14: DE in T1D and T2D in endocrine cell types (data_exploration/atlas/14*)

Beta cell specific analyses of integrated data:
- 15: Gene information for prioritisation of genes during beta cell analyses interpretation
 - 15.1: Average expression of genes across cell types (data_exploration/atlas/15-1*)
 - 15.2: Collect extra information on genes relevant for interpretation, such as expression strength, relative expression in beta cells, how much is known about a gene in online databases, ... (data_exploration/atlas/beta/15-2*)
- 16: Beta cell clusters
 - 16.1: Coarse clusters
   - 16.1.1: Compute clusters and their markers. Also contains a part on comparison to fine clusters (can be run only after these are computed, see below) (data_exploration/atlas/beta/16-1-1*)
   - 16.1.2: Compare cluster markers to external mouse and human datasets (data_exploration/atlas/beta/16-1-2*)
 - 16.2: Fine clusters
   - 16.2.1: Gene programs (GPs) (data_exploration/atlas/beta/16-2-1*)
   - 16.2.2: Define clusters based on GPs (data_exploration/atlas/beta/16-2-2*)
   - 16.2.3: Compare clusters based on GPs (data_exploration/atlas/beta/16-2-3*)
- 17: Beta cell metadata localisation on embedding
 - 17.1: Optimisation of visual representation (UMAP) of the beta cell embedding (data_exploration/atlas/beta/17-1*)
 - 17.2: Plot of markers (data_exploration/atlas/beta/17-2*)
 - 17.3: Plot of condition metadata (data_exploration/atlas/beta/17-3*)
- 18: Expression of known beta cell heterogeneity markers across beta cell states captured within the atlas (data_exploration/atlas/beta/18*)
- 19: Variance in data explained by GPs and healthy conserved gene groups
 - 19.1: Variance explained by GPs across datasets, including additional datasets not included in the atlas (data_exploration/atlas/beta/19-1*)
 - 19.2: Healthy-samples (including samples outside of the atlas) variance explained by GPs that are likely related to differences between health and disease (data_exploration/atlas/beta/19-2*)
 - 19.3: Groups of genes consistently co-variable across healthy samples (data_exploration/atlas/beta/19-3*)
- 20: Comparison of diabetic models
 - 20.1: Simialrity of diabetes models to human data - find gene sets DE in human diabetes (based on scRNA-seq human data and literature) and compare their expression between healthy and diabetic samples from diabetes model datasets (data_exploration/atlas/beta/20-1*)
 - 20.2: Expression of known function and stress genes across diabetes models (data_exploration/atlas/beta/20-2*)
- 21: DE in diabetic models
 - 21.1: Compute DE genes for T1D (NOD) and T2D (db/db, mSTZ) (data_exploration/atlas/beta/21-1*)
 - 21.2: Analysis of DE genes: clustering, embryo comparison (data_exploration/atlas/beta/21-2*)
 - 21.3: Comparison of T1D and T2D gene groups (data_exploration/atlas/beta/21-3*)
 - 21.4: Activity of DE gene groups across all beta cell states (data_exploration/atlas/beta/21-4*)
 - 22.5: Activity of DE genes across healthy-diabetes trajectory (data_exploration/atlas/beta/21-5*)
 - 22.6: Comparison of DE gene groups to external mouse and human datasets (data_exploration/atlas/beta/21-6*)
- 22: Sex differences
 - 22.1: DE between sexes per study (data_exploration/atlas/beta/22-1*)
 - 22.2: Analyse and cluster DE genes in the aged dataset (data_exploration/atlas/beta/22-2*)

Atlas reference mapping
- 23: Mapping of external mouse dataset to the atlas
 - 23.1: Reference mapping (integration/scArches/23-1*)
 - 23.2: Joint embedding of the atlas and the external mouse dataset with cell state transfer (data_exploration/atlas/beta/23-2*)

Prepare data for submission
- 24: Prepare certain tables for paper and supplements that require combining and editing of multiple previously generated tables. Some tables were prepared manualy or directly in other files and are thus not edited here. (prepare_submit/24*)
- 25: Anndata objects
 - 25.1: Atlas adata for cellxgene (prepare_submit/25-1*)
 - 25.2: Atlas adata for GEO (prepare_submit/25-2*)
 - 25.3: Result of reference mapping of GSE137909 for GEO

### Notes on execution order and file content

- Files are numbered according to the order in which they should be run. However, some of them contain small parts of later added code that relies on a notebook that comes afterwards in the ordering (always noted). Thus these code parts should be run after the specified notebook had been ran. 
- In the atlas exploration phase some files that are numbered in order do not require some of the former files to be run. However, to keep the files sorted, groupped, and named consistently we still add consectutive numbers to file names.
- When multiple files contain the same number this is either due to: the order not being important (e.g. preprocessing per-study), export of ipynb to py (also has the same name), or we used an additional file to store commands for calling the analysys script on multiple inputs.
- There are some .txt/.sbatch files (often nammed with number ending in -0) that contain shell comands for runing individual analyses or multiple steps (in order) on multiple input files, often via .sbatch for deploying code on internal servers, which in most cases at the end results in calling .py or .R files that contain the processing code. Some of the .txt files require different processing options to be selected by un/re-commenting code, as explained in the comments.
- Function scripts - store functions, but not analysis. Thus they are not numbered as they are never ran directly.
- Some scripts contain variables WSCL and WSL - these variables were set in bash and contain information on paths to code directory (WSCL, /lustre/groups/ml01/code/karin.hrovatin/) and data directory (WSL, /lustre/groups/ml01/workspace/karin.hrovatin/).
- In some of the notebooks teh data was saved with custom functions to overcome saving problems on or servers. These functions can be replaced with Scanpy functions:

    ```py
    # Saving
    h.save_h5ad(adata=adata,file=fn)
    adata.write(fn)
    # Loading
    h.open_h5ad(file=fn)
    sc.read(fn)
    ```
- Initially, we used different study names as decided upon at the end. For the mapping please see notebook for the summary plot of sample metadata.
- <sup>1</sup> In steps 1 and 2 the db/db (VSG) and mSTZ (STZ) datasets were split in reference and non-reference subsets for initial analyses of reference building with hyperparam testing and query mapping. For the atlas itself the versions that do not contain ref/noref suffix were used.
- <sup>2</sup> Initial annotation used for integration evaluation contained cell type names stellate and pericyte that were latter corrected to stellate quiescent and activated, respectively. This however does not affect the integration evaluation outcomes, as the two cell types were annotated consistently across all datasets and represented distinct cell populations.

## Data:

Directory data contains additional tables used in the code. The directory data contains the same structure as in the code refered to path /lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/ 

Raw scRNA-seq data is availiable on GEO, as specified in the publication. 

