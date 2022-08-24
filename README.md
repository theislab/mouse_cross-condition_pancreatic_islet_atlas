# mm_pancreas_atlas_rep

Reproducibility code for mouse pancreatic islet scRNA-seq atlas from Hrovatin et al.

<!-- #region -->
Data:


Processing steps:
- To enable saving makse sure that the variable SAVE in file code/constants.py is set to True
- 1: Data preprocessing - per-dataset ambient genes, QC, normalisation,HVG (preprocessing/1_*) <sup>1</sup>
 - 1.1: Compute empty droplet score of each cell (preprocessing/1-1_ambient_scores*)
- 2: Data annotation - for each dataset add UMAP, info on cell cycle, sex (if mixed), and for some also endocrine marker scores, cell types, and cell subtypes (preprocessing/2*) <sup>1, 2</sup>
- 3: Preprocessing and curration of datasets used for atlas validation - prepare normalised expression data and currate metadata (not part of the atlas) (prepprocessing/3*)
- 4: Ambient correction - correct data and analyse correction results for data used in final integration
 - 4.1: CellBender (preprocessing/4-1*)
 - 4.2: DecontX (preprocessing/4-2*)
 - 4.3: SoupX (preprocessing/4-3*)
- 5: Preprocess for integration - prepare expression data for test and final integrations (prepprocessing/5*)
 - 5.1: Prepare data for integration on a smaller annotated data subset only. For testing out hyperparameters (prepprocessing/5-1*)
 - 5.2: Prepare data for full atlas integration:  prepare study data, including per sample normalization, etc. (prepprocessing/5-2-1*) and add EIDs to data preprocessed for integration (prepprocessing/5-2-2*)
- 6: Integration (integration/6*)
 - 6.1: scVI - select hyperparams on ref data and then train model on whole data, including different ambient preprocessing (integration/scVI/6-1*)
 - 6.2: scArches - select hyperparams on ref data and then train model on whole data, including different ambient preprocessing and using only beta cell subset (integration/scVI/6-2* scArches). 
  - We also tried maping the non-ref data subset (external or diabetic samples) to the selected best ref integration (only non-diabetic in house samples), but the reference mapping performance was never of comparable quality to de novo integration of all datasets, possibly due to low complexity of the reference (the ref dataset). The code for mapping on top of the ref integration is in 6-2-2_scArches_addToRef.ipynb, however, we do not show the scIB results in the evaluation summary plots (described below) as this was not used in the paper due to the above mentioned poor performance.
 - 6.3: Evaluation with scIB metrics - compute evaluation metrics on different data subsets (e.g. all cells, annotated cells, beta cells) and compare results across integration runs (integration/6-3*)
 - 6.4: Prepare integrated adata
- 



TODO
- gene_annotation - Must be after integration
- P7 ambient plot - must be after integration
- integration/scArches/6B_map_mouseExternal_example.py

TODO
- make sure WSC/WS match new one (e.g. $WSC/, "$WSC", ....)

Notes: 
- When multiple files contain the same number this is either due to: the order not being important (e.g. preprocessing per-study), export of ipynb to py (also same name), or we used a txt file to store commands for calling the analysys script on multiple inputs.
- The .txt files usually contain shell comands for runing analyses on multiple input files, often via .sbatch for deploying code on internal servers, which in most cases at the end results in calling .py or .R files that contain custom processing code. Some of the .txt files require different processing options to be selected by un/re-commenting code, as explained in the comments.
- Function scripts - store functions, but not analysis. Thus they are not numbered as they are never ran directly.
- <sup>1</sup> In steps 1 and 2 the db/db (VSG) and mSTZ (STZ) datasets were split in reference and non-reference subsets for initial analyses of reference building and query mapping. For the atlas itself the versions that do not contain ref/noref suffix were used.
- In some of the notebooks teh data was saved with custom functions to overcome saving problems on or servers. These functions can be replaced with Scanpy functions:

    ```py
    # Saving
    h.save_h5ad(adata=adata,file=fn)
    adata.write(fn)
    # Loading
    h.open_h5ad(file=fn)
    sc.read(fn)
    ```
- Initially, we used different study names as decided upon at the end. For the mapping please see **TODO**
- <sup>2</sup> Initial annotation used for integration evaluation contained cell type names stellate and pericyte that were latter corrected to stellate quiescent and activated, respectively. This however does not affect the integration evaluation outcomes, as the two cell types were annotated consistently across all datasets and represented distinct cell populations.
<!-- #endregion -->

```python

```
