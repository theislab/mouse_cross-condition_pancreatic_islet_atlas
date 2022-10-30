# Mapping of new data to the mouse pancreatic islet scRNA-seq atlas


The atlas is described in ... 

There we also describe how reference mapping onto the atlas enables cell type and state contextualisation for annotation transfer and cross-condition  or cross-study comparison.

The mapping can be done simply by ruining the example jupyter notebook (code/reference_mapping.ipynb). For this we use a scArches model. The mapping requires cloning of the git repository to obtain the atlas model and the example code and data, installation of the below specified python packages, and preparation of the query data as specified below.


#### Data and code

We provide the scArches atlas model in data/scArches and the reference and query datasets needed to run the example mapping in data/scArches and data/mapref_scArches*, respectively.

The reference mapping code example is provided in code/reference_mapping.ipynb.
<br/><br/>

#### Query data prerequisites for the re-use of the example reference mapping script
- Data saved in AnnData format.
- Expression (in X) should be per-sample Scran normalized and log(x+1) transformed, as also described in the manuscript methods for integration and reference mapping. 
- We recomend the use of UMI-based scRNA-seq data: The reference contains datasets generated with different 10X versions and we validated reference mapping with dataset obtained with a modified STRT-seq protocol (please refer to the original dataset publications for details, dataset accession numbers are specified in our manuscript).
- Genes should be named with Ensebl IDs in var_names. 
- Batch covariate should be sample and in the code example we specify it in obs column 'batch_integration'. 
<br/><br/>

#### Environment

The mapping can be done with the environment specified in data/env/pip_freeze.txt.


#### References
scArches: [Lotfollahi et al., Nature Biotechnology 2021](https://www.nature.com/articles/s41587-021-01001-7), doi:10.1038/s41587-021-01001-7<br>

