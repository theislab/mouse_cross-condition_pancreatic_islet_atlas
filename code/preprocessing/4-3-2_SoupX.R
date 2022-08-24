library("SoupX")
library(zellkonverter)
library(SingleCellExperiment)
library(Matrix)
library(SparseM)

if (FALSE){
    #matplotlib.use('TkAgg')
    file_ext='feature_bc_matrix.h5ad'
    folder='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/inceptor/citeseq_mouse_islets_DK/DB/expression/'
}

args = commandArgs(trailingOnly=TRUE)
print(args)
file_ext<-args[1]
folder<-args[2]


print(paste('file_ext:',file_ext,'folder:',folder))

folder_out<-paste0(folder,'SoupX/')
dir.create(file.path(folder_out))

sc_raw<-readH5AD(paste0(folder,'raw_',file_ext))
sc_filtered<-readH5AD(paste0(folder,'filtered_',file_ext))

clusters<-colData(sc_filtered)['leiden_r1_normscl']
colnames(clusters)<-c('clusters')

sc_filtered_dgc<-as(as(assay(sc_filtered),'matrix.csr'),'dgCMatrix')
dimnames(sc_filtered_dgc)<-dimnames(sc_filtered)

sc<-SoupChannel(tod =  as(assay(sc_raw),"sparseMatrix"),
                toc = sc_filtered_dgc,
                metaData = clusters, calcSoupProfile = TRUE)

# Modification of autoEstCont function that can save plot
autoEstCont_plot<-function (sc, topMarkers = NULL, tfidfMin = 1, soupQuantile = 0.9, 
    maxMarkers = 100, contaminationRange = c(0.01, 0.8), rhoMaxFDR = 0.2, 
    priorRho = 0.05, priorRhoStdDev = 0.1, doPlot = TRUE, forceAccept = FALSE, 
    verbose = TRUE, figsave = NULL) 
{
    if (!"clusters" %in% colnames(sc$metaData)) 
        stop("Clustering information must be supplied, run setClusters first.")
    s = split(rownames(sc$metaData), sc$metaData$clusters)
    tmp = do.call(cbind, lapply(s, function(e) rowSums(sc$toc[, 
        e, drop = FALSE])))
    ssc = sc
    ssc$toc = tmp
    ssc$metaData = data.frame(nUMIs = colSums(tmp), row.names = colnames(tmp))
    soupProf = ssc$soupProfile[order(ssc$soupProfile$est, decreasing = TRUE), 
        ]
    soupMin = quantile(soupProf$est, soupQuantile)
    if (is.null(topMarkers)) {
        mrks = quickMarkers(sc$toc, sc$metaData$clusters, N = Inf)
        mrks = mrks[order(mrks$gene, -mrks$tfidf), ]
        mrks = mrks[!duplicated(mrks$gene), ]
        mrks = mrks[order(-mrks$tfidf), ]
        mrks = mrks[mrks$tfidf > tfidfMin, ]
    }
    else {
        mrks = topMarkers
    }
    tgts = rownames(soupProf)[soupProf$est > soupMin]
    filtPass = mrks[mrks$gene %in% tgts, ]
    tgts = head(filtPass$gene, n = maxMarkers)
    if (verbose) 
        message(sprintf("%d genes passed tf-idf cut-off and %d soup quantile filter.  Taking the top %d.", 
            nrow(mrks), nrow(filtPass), length(tgts)))
    if (length(tgts) == 0) {
        stop("No plausible marker genes found.  Reduce tfidfMin or soupQuantile")
    }
    if (length(tgts) < 10) {
        warning("Fewer than 10 marker genes found.  Consider reducing tfidfMin or soupQuantile")
    }
    tmp = as.list(tgts)
    names(tmp) = tgts
    ute = estimateNonExpressingCells(sc, tmp, maximumContamination = max(contaminationRange), 
        FDR = rhoMaxFDR)
    m = rownames(sc$metaData)[match(rownames(ssc$metaData), sc$metaData$clusters)]
    ute = t(ute[m, , drop = FALSE])
    colnames(ute) = rownames(ssc$metaData)
    expCnts = outer(ssc$soupProfile$est, ssc$metaData$nUMIs)
    rownames(expCnts) = rownames(ssc$soupProfile)
    colnames(expCnts) = rownames(ssc$metaData)
    expCnts = expCnts[tgts, , drop = FALSE]
    obsCnts = ssc$toc[tgts, , drop = FALSE]
    pp = ppois(obsCnts, expCnts * max(contaminationRange), lower.tail = TRUE)
    qq = p.adjust(pp, method = "BH")
    qq = matrix(qq, nrow = nrow(pp), ncol = ncol(pp), dimnames = dimnames(pp))
    rhos = obsCnts/expCnts
    rhoIdx = t(apply(rhos, 1, function(e) order(order(e))))
    dd = data.frame(gene = rep(rownames(ute), ncol(ute)), passNonExp = as.vector(ute), 
        rhoEst = as.vector(rhos), rhoIdx = as.vector(rhoIdx), 
        obsCnt = as.vector(obsCnts), expCnt = as.vector(expCnts), 
        isExpressedFDR = as.vector(qq))
    dd$geneIdx = match(dd$gene, mrks$gene)
    dd$tfidf = mrks$tfidf[dd$geneIdx]
    dd$soupIdx = match(dd$gene, rownames(soupProf))
    dd$soupExp = soupProf$est[dd$soupIdx]
    dd$useEst = dd$passNonExp
    if (sum(dd$useEst) < 10) 
        warning("Fewer than 10 independent estimates, rho estimation is likely to be unstable.  Consider reducing tfidfMin or increasing SoupMin.")
    if (verbose) 
        message(sprintf("Using %d independent estimates of rho.", 
            sum(dd$useEst)))
    p.L = function(x, alpha) {
        if (x == 0) {
            0
        }
        else {
            qgamma(alpha, x)
        }
    }
    p.U = function(x, alpha) {
        qgamma(1 - alpha, x + 1)
    }
    alpha = 0.95
    alpha = (1 - alpha)/2
    dd$rhoHigh = sapply(seq(nrow(dd)), function(e) p.U(dd$obsCnt[e], 
        alpha)/dd$expCnt[e])
    dd$rhoLow = sapply(seq(nrow(dd)), function(e) p.L(dd$obsCnt[e], 
        alpha)/dd$expCnt[e])
    rhoProbes = seq(0, 1, 0.001)
    v2 = (priorRhoStdDev/priorRho)^2
    k = 1 + v2^-2/2 * (1 + sqrt(1 + 4 * v2))
    theta = priorRho/(k - 1)
    tmp = sapply(rhoProbes, function(e) {
        tmp = dd[dd$useEst, ]
        mean(dgamma(e, k + tmp$obsCnt, scale = theta/(1 + theta * 
            tmp$expCnt)))
    })
    xx = dgamma(rhoProbes, k, scale = theta)
    w = which(rhoProbes >= contaminationRange[1] & rhoProbes <= 
        contaminationRange[2])
    rhoEst = (rhoProbes[w])[which.max(tmp[w])]
    rhoFWHM = range((rhoProbes[w])[which(tmp[w] >= (max(tmp[w])/2))])
    contEst = rhoEst
    if (verbose) 
        message(sprintf("Estimated global rho of %.2f", rhoEst))
    if (doPlot) {
        if(!is.null(figsave)) png(figsave)
        plot(rhoProbes, tmp, "l", xlim = c(0, 1), ylim = c(0, 
            max(c(xx, tmp))), frame.plot = FALSE, xlab = "Contamination Fraction", 
            ylab = "Probability Density")
        lines(rhoProbes, xx, lty = 2)
        abline(v = rhoProbes[which.max(tmp)], col = "red")
        legend(x = "topright", legend = c(sprintf("prior rho %g(+/-%g)", 
            priorRho, priorRhoStdDev), sprintf("post rho %g(%g,%g)", 
            rhoEst, rhoFWHM[1], rhoFWHM[2]), "rho max"), lty = c(2, 
            1, 1), col = c("black", "black", "red"), bty = "n")
        if(!is.null(figsave)) dev.off()
    }
    sc$fit = list(dd = dd, priorRho = priorRho, priorRhoStdDev = priorRhoStdDev, 
        posterior = tmp, rhoEst = rhoEst, rhoFWHM = rhoFWHM)
    sc = setContaminationFraction(sc, contEst, forceAccept = forceAccept)
    return(sc)
}


sc <-autoEstCont_plot(sc,figsave=paste0(folder_out,'ambient_estimation.png'),forceAccept=TRUE)

rho<-mean(sc$metaData$rho)
# The below should not really ever be True
if(length(unique(sc$metaData$rho))!=1){
    warning('Not all rhos are equal across cells')
}
cat(rho,file=paste0(folder_out,'rho.txt'))

for(add_contamination in c(0,0.05,0.1)){
    # Increase rho of each cell
    rhos<-sc$metaData$rho+add_contamination
    names(rhos) <- colnames(sc$toc)
    sc_set_cont = setContaminationFraction(sc, rhos,forceAccept=TRUE)
    # Adjust counts
    adjusted = adjustCounts(sc_set_cont)
    # Save adjusted counts as anndata
    
    writeH5AD(SingleCellExperiment(list('X'=adjusted)), 
          file = paste0(folder_out,'SoupX','_filtered_',
                        'rhoadd',gsub('.','',paste0('',add_contamination),fixed=TRUE),'.h5ad'
                       )
         )
}

print('Finished!')
