library(hypeR)
library(msigdbr)
library(visNetwork)

#' Same as msigdb_gsets from hypeR but with added params
#'
#' @param size_range - Do not include gene sets with size outside of this range - 
#'
#' @param filter_gene_sets - If not NULL include only gene sets 
#' with gs_exact_source contained within this vector
#'
#' @param background If not NULL include only genes contained in background in the gene sets
#' This needs to be done here to correctly determine gene set size
msigdb_gsets_custom <- function(species, category, subcategories=c('GO:BP','GO:CC','GO:MF'), 
                                clean=FALSE,
                               size_range=c(5,500),filter_gene_sets=NULL,
                               background=NULL) {
    # Check species
    msigdb_check_species <- function(species="") {
    if (!species %in% msigdb_species()) {
        stop("Species must be one of the following: \n", paste(msigdb_species(), "\n"))
    }
}

    msigdb_check_species(species)
    
    if (!is.null(background)) background=as.vector(unlist(background))
    genesets<-list()
    for (subcategory in subcategories){
        response <- msigdbr(species, category, subcategory)
        if (nrow(response) == 0) {
            stop("No data found: Please review available species and genesets\n", msigdb_info())
        }

        # Download genesets
        mdf <- msigdbr(species, category, subcategory) %>%
               dplyr::select(gs_name,gs_exact_source, gene_symbol) %>%
               as.data.frame() %>%
               stats::aggregate(gene_symbol ~ gs_name+gs_exact_source, data=., c)

        # Convert to list
        for (i in 1:nrow(mdf)) {
            row <- mdf[i,]
            if(is.null(filter_gene_sets) | row$gs_exact_source %in% filter_gene_sets){
                genes<-unlist(as.list(row$gene_symbol))
                if(!is.null(background)) genes<-intersect(genes,background)
                n_genes<-length(genes)
                if(n_genes>=size_range[1] & n_genes<=size_range[2]){
                    genesets[paste(row$gs_exact_source,row$gs_name,sep='_')]<-list(genes)
                }
            }
        }
        
    }
    #name <- ifelse(subcategory == "", category, paste(category, subcategory, sep="."))
    #version <- msigdb_version()
    #gsets$new(genesets, name=name, version=version, clean=clean)
    return(genesets)
}
#' Same as hypeR .enrichment_map_save, modified saving
#'
#' @param file If not NULL save image to this path, adding .html
.enrichment_map_save <- function(hyp_df,
                            genesets, 
                            similarity_metric=c("jaccard_similarity", "overlap_similarity"),
                            similarity_cutoff=0.2,
                            pval_cutoff=1, 
                            fdr_cutoff=1,
                            val=c("fdr", "pval"),
                            top=NULL,
                            title="",
                           file=NULL) {

    # Subset results
    hyp_df <- hyp_df %>%
              dplyr::filter(pval <= pval_cutoff) %>%
              dplyr::filter(fdr <= fdr_cutoff) %>%
              purrr::when(!is.null(top) ~ head(., top), ~ .)

    # Handle empty dataframes
    if (nrow(hyp_df) == 0) return(NULL)
    
    # Geneset similarity matrix
    hyp.genesets <- genesets[hyp_df$label]
    hyp.genesets.mat <- sapply(hyp.genesets, function(x) {
        sapply(hyp.genesets, function(y,x) {
            if (similarity_metric == "jaccard_similarity") hypeR:::.jaccard_similarity(x, y)
            else if (similarity_metric == "overlap_similarity") hypeR:::.overlap_similarity(x, y)     
            else stop(.format_str("{1} is an invalid metric", similarity_metric))
        }, x)
    })
    
    m <- as.matrix(hyp.genesets.mat)

    # Sparsity settings
    m[m < similarity_cutoff] <- 0
    
    # Similarity matrix to weighted network
    inet <- igraph::graph.adjacency(m, mode="undirected", weighted=TRUE, diag=FALSE)
    
    # igraph to visnet
    vnet <- toVisNetworkData(inet)

    nodes <- vnet$nodes
    edges <- vnet$edges
    
    # Add edge weights
    edges$value <- vnet$edges$weight

    # Add node scaled sizes based on genset size
    size.scaler <- function(x) (x-min(x))/(max(x)-min(x))*30 
    node.sizes <- sapply(igraph::V(inet), function(x) hyp_df[x, "geneset"])
    nodes$size <-  size.scaler(node.sizes)+20
    
    val.pretty <- ifelse(val == "fdr", "FDR", "P-Value")
    nodes$title <- sapply(igraph::V(inet), function(x) {
                            paste(val.pretty, hyp_df[x, val], sep=": ")
                   })
    
    # Add node scaled weights based on significance
    weight.scaler <- function(x) (x-max(x))/(min(x)-max(x))
    node.weights <- sapply(igraph::V(inet), function(x) hyp_df[x, val])
    nodes$color.border <- "rgb(0,0,0)"
    nodes$color.highlight <- "rgba(199,0,57,0.9)"
    nodes$color.background <- sapply(weight.scaler(node.weights), function(x) { 
                                  if (is.na(x)) {
                                      return("rgba(199,0,57,0)")
                                  } else{
                                      return(paste("rgba(199,0,57,", round(x, 3), ")", sep=""))   
                                  }
                          })

    graph=visNetwork(nodes, edges, main=list(text=title, style="font-family:Helvetica")) %>%
    visNodes(borderWidth=1, borderWidthSelected=0) %>%
    visEdges(color="rgb(88,24,69)") %>%
    visOptions(highlightNearest=TRUE) %>%
    visInteraction(multiselect=TRUE, tooltipDelay=300) %>%
    visIgraphLayout(layout="layout_nicely")
    if (!is.null(file)) visSave(graph, file=paste0(file,'.html'), 
                                selfcontained = TRUE, background = "white")
}

#' Same as hypeR hyp_emap, modified saving
#'
#' @param file See .enrichment_map_save
hyp_emap_save <- function(hyp_obj, 
                     similarity_metric=c("jaccard_similarity", "overlap_similarity"),
                     similarity_cutoff=0.2,
                     pval=1, 
                     fdr=1,
                     val=c("fdr", "pval"),
                     top=NULL,
                     title="",
                     file=NULL) {

    stopifnot(is(hyp_obj, "hyp") | is(hyp_obj, "multihyp"))

    # Default arguments
    similarity_metric <- match.arg(similarity_metric)
    val <- match.arg(val)

    # Handling of multiple signatures
    if (is(hyp_obj, "multihyp")) {
        multihyp_obj <- hyp_obj

        mapply(function(hyp_obj, title) {

            hyp_emap_save(hyp_obj,
                     similarity_metric=similarity_metric, 
                     similarity_cutoff=similarity_cutoff,
                     pval=pval,
                     fdr=fdr,
                     val=val,
                     top=top,
                     title=title,
                    file=file) 

        }, multihyp_obj$data, names(multihyp_obj$data), USE.NAMES=TRUE, SIMPLIFY=FALSE)
    } 
    else {
        hyp_df <- hyp_obj$data
        genesets <- hyp_obj$args$genesets$genesets
        .enrichment_map_save(hyp_df, genesets, similarity_metric, 
                        similarity_cutoff, pval, fdr, val, top, title,file=file)
    }
}
