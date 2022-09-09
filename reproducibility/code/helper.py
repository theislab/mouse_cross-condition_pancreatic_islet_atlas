import shutil
import os
import scanpy as sc
import uuid
import warnings
import h5py
import scipy
import pandas as pd
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colors
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler
import seaborn as sb
from scipy.cluster.hierarchy import linkage,leaves_list
from scipy.spatial.distance import pdist
from sklearn.neighbors import KNeighborsTransformer


def open_h5ad(file, temp_location='/tmp/hsperfdata_karin.hrovatin/',unique_id2='',**kwargs):
    """
    Load h5ad adata by first moving it to temporal location (from where it is easier to load it).
    Files are copied to same default loaction and named based on file name (without path). This could create issues when opening 
    files from different threads. Thus files are added random ID and unique_id2 (if provided). This reduces the 
    chance of corruption due to race conditions.
    :param file: File to open.
    :param temp_location: Location (path without file name) where to copy the file for the duration of reading.
    :param unique_id2: ID added to file copy to prevent the issues due to race conditions when opening multiple equaly named files 
    from different threads. 
    :return: adata
    """
    if not os.path.exists('/tmp/hsperfdata_karin.hrovatin/'):
        os.makedirs('/tmp/hsperfdata_karin.hrovatin/')
    if unique_id2=='':
        warnings.warn('Consider specifying unique_id2 if you are performing the same operation from another thread.')
    file_name=file.split('/')[-1]
    temp_file=file_uid(file=temp_location+file_name,unique_id2=unique_id2)
    shutil.copyfile(file, temp_file)
    adata=sc.read_h5ad(temp_file,**kwargs)
    os.remove(temp_file)
    return adata

def save_h5ad(adata, file, temp_location='/tmp/hsperfdata_karin.hrovatin/',unique_id2='',**kwargs):
    """
    Save adata as h5ad by first moving it to temporal location (where it is easier to save).
    Files are temporarily saved to same default loaction and named based on file name (without path). 
    This could create issues when saving 
    files from different threads. Thus files are added random ID and unique_id2 (if provided). This reduces the 
    chance of corruption due to race conditions.
    :param adata: Adata to save as h5ad.
    :param file: File to save.
    :param temp_location: Location (path without file name) where to temporarily save the file during writting.
    :param unique_id2: ID added to temporary file to prevent the issues due to race conditions when opening multiple equaly named files 
    from different threads. 
    """    
    if not os.path.exists('/tmp/hsperfdata_karin.hrovatin/'):
        os.makedirs('/tmp/hsperfdata_karin.hrovatin/')
    if unique_id2=='':
        warnings.warn('Consider specifying unique_id2 if you are performing the same operation from another thread.')
    file_name=file.split('/')[-1]
    temp_file=file_uid(file=temp_location+file_name,unique_id2=unique_id2)
    adata.write(temp_file,**kwargs)
    shutil.move(temp_file, file)

def update_adata(adata_new,path,add=None,rm=None,unique_id2=None,io_copy=True):
    """
    Update saved adata by information from new adata or remove data.
    Items are first removed and the added.
    :param add: List of iters for data to add, 
    i[0] - adata slot (obs, uns, var, obsm, obsp), 
    i[1] - overwritte - if False error will be raised if item is already present in data 
    saved in path, 
    i[2] - key of slot in adata_new (add this data to original data), 
    i[3] - key of slot in original data to which data will be stored
    :param rm: List of iters for data to remove, 
    i[0] - adata slot (obs, uns, var, obsm, obsp), 
    i[1] - key of slot in original data to remove
    :param adata_new: Adata from which to get info for add
    :param path: Path from which to read data to be updated and then save data back to it.
    :param unique_id2: UID2 for reading/writting if io_copy=True
    :param io_copy: Use read/write from this script if True, else use AnnData functions directly
    """
    if io_copy:
        adata_original=open_h5ad(path, unique_id2=unique_id2)
    else:
        adata_original=sc.read(path)
    if rm is not None:
        for item in rm:
            del getattr(adata_original,item[0])[item[1]]
    if add is not None:
        for item in add:
            if item[3] not in getattr(adata_original,item[0]).keys() or item[1]:
                getattr(adata_original,item[0])[item[3]]=getattr(adata_new,item[0])[item[2]]
            else:
                raise ValueError(item, 'is already in original data and overwritte is False')
    if io_copy:
        save_h5ad(adata_original, path, unique_id2=unique_id2)
    else:
        adata_original.write(path)


def file_uid(file,unique_id2=''):
    """
    Add unique ID to file name.
    Returned file format: file_uniqueID_uniqueID2.OriginalFileEnding
    :param file: File path
    :param unique_id2: Custom unique ID.
    :return: Unique file name
    """
    file_ending='.'+file.split('.')[-1]
    return file+'_'+str(uuid.uuid4())+'_'+unique_id2+file_ending

def replace_all(string, replace):
    """
    Replace multiple substrings of a string.
    :param string: String to edit
    :param replace: Dict with keys being strings to replace and values being corresponding replacements.
    :return: Edited string.
    """
    for old,new in replace.items():
        string=string.replace(old,new)
    return string


def readCellbenderH5(filename):
    f = h5py.File(filename, 'r')
    mat=f['background_removed']
    cols=['latent_cell_probability','latent_RT_efficiency']
    rows=['ambient_expression']
    obsdict={x:mat[x] for x in cols}
    rowsdict={x:mat[x] for x in rows}
    ad=sc.AnnData(X=scipy.sparse.csr_matrix((mat['data'][:], 
                                          mat['indices'][:], 
                                          mat['indptr'][:]),
                                        shape=(mat['shape'][1],mat['shape'][0])),
                  # I had only one column of gene names
              var=pd.DataFrame(rowsdict,index=[x.decode('ascii') for x in mat['gene_names']]),
              obs=pd.DataFrame(obsdict,index=[x.decode('ascii') for x in mat['barcodes']]),
              uns={'test_elbo':list(mat['test_elbo']),'test_epoch':list(mat['test_epoch'])})
    return(ad)

def get_rawnormalised(adata,sf_col='size_factors',use_log=True,save_nonlog=False, use_raw=True,
    copy=True):
    """
    Copy raw data from adata and normalise it.
    :param adata: To normalise. Is copied.
    :param sf_col: Sf col in obs to use
    :param use_log: Do log transform after norm
    :param use_raw: If true use adata.raw, else use adata.X
    :param save_nonlog: Save non-log in layers['normalised_counts']
    :param copy: Call copy on adata or adata.raw.to_adata()
    """
    if use_raw:
        adata_rawnorm = adata.raw.to_adata()
    else:
        adata_rawnorm = adata
    if copy:
        adata_rawnorm=adata_rawnorm.copy()
    adata_rawnorm.X /= adata.obs[sf_col].values[:,None] # This reshapes the size-factors array
    if use_log:
        if save_nonlog:
            adata_rawnorm.layers['normalised_counts']=sparse.csr_matrix(np.asarray(
                adata_rawnorm.X.copy()))
        sc.pp.log1p(adata_rawnorm)
    adata_rawnorm.X = sparse.csr_matrix(np.asarray(adata_rawnorm.X))
    return adata_rawnorm


def add_category(df,idxs,col,category):
    """
    Add single value to multiple rows of DF column (useful when column might be categorical). 
    If column is categorical the value is beforehand added in the list of present categories 
    (required for categorical columns). 
    :param df: DF to which to add values
    :param idxs: Index names of rows where the value should be assigned to the column.
    :param col: Column to which to add the value.
    :param category: The value to add to rows,column.
    """
    # If column is already present, is categorical and value is not in categories add the value to categories first.
    if col in df.columns and df[col].dtype.name=='category' and category not in df[col].cat.categories:
        df[col] = df[col].cat.add_categories([category])
    df.loc[idxs,col]=category


def plot_density(adata,design_order=['mRFP','mTmG','mGFP',
          'head_Fltp-','tail_Fltp-', 'head_Fltp+', 'tail_Fltp+',
          'IRE1alphafl/fl','IRE1alphabeta-/-', 
          '8w','14w', '16w',
          'DMSO_r1','DMSO_r2', 'DMSO_r3','GABA_r1','GABA_r2', 'GABA_r3',
               'A1_r1','A1_r2','A1_r3','A10_r1','A10_r2', 'A10_r3',  'FOXO_r1', 'FOXO_r2', 'FOXO_r3', 
          'E12.5','E13.5','E14.5', 'E15.5', 
          'chow_WT','sham_Lepr-/-','PF_Lepr-/-','VSG_Lepr-/-',   
          'control','STZ', 'STZ_GLP-1','STZ_estrogen', 'STZ_GLP-1_estrogen',
              'STZ_insulin','STZ_GLP-1_estrogen+insulin' 
        ],min_cells=20):
    """
    Plot density for each batch - study_sample, ordered by study and design. 
    Studies are in rows and samples in columns.
    :param adata: Adata for which to plot denisty
    :param design_order: Sort samples (file) within study by design when plotting study samples.
    :param min_cell:
    """
    # Prepare figure grid
    fig_rows=adata.obs.study.unique().shape[0]
    fig_cols=max(
        [adata.obs.query('study ==@study').file.unique().shape[0] 
         for study in adata.obs.study.unique()])
    rcParams['figure.figsize']= (6*fig_cols,6*fig_rows)
    fig,axs=plt.subplots(fig_rows,fig_cols)
    # Calculagte density
    for row_idx,study in enumerate(adata.obs.study.unique()):
        designs=adata.obs.query('study ==@study')[['study_sample_design','design']].drop_duplicates()
        designs.design=pd.Categorical(designs.design, 
                      categories=[design for design in design_order if design in list(designs.design.values)],
                      ordered=True)
        for col_idx,sample in enumerate(designs.sort_values('design')['study_sample_design'].values):
            subset=adata.obs.study_sample_design==sample
            adata_sub=adata[subset,:].copy()
            if adata_sub.shape[0]>=min_cells:
                #print(sample,adata_sub.shape)
                sc.tl.embedding_density(adata_sub)
                sc.pl.umap(adata,ax=axs[row_idx,col_idx],s=10,show=False)
                sc.pl.embedding_density(adata_sub,ax=axs[row_idx,col_idx],title=sample,show=False) 
                axs[row_idx,col_idx].set_xlabel('')
                if col_idx==0:
                    axs[row_idx,col_idx].set_ylabel(study, rotation=90, fontsize=12)
                else:
                    axs[row_idx,col_idx].set_ylabel('') 


# +
def scale_data_5_75(data,mind,maxd):
    data=np.array(data)
    if maxd == mind:
        maxd=maxd+1
        mind=mind-1
        
    drange = maxd - mind
    return ((((data - mind)/drange*0.70)+0.05)*100)

def plot_enrich(data, n_terms=20, save=False,min_pval=10**-30, max_pval=0.05,percent_size=True,
               recall_lim=(0,0.5)):
    """
    :param data: Should have the following columns: 'p_value' (corrected p value), 
    'name' (gene set name), 'intersection_size' (gene set - query imntersection), 
    'recall' (ratio of gene set contained in query), 'query_size'
    """
    # Test data input
    if not isinstance(data, pd.DataFrame):
        raise ValueError('Please input a Pandas Dataframe output by gprofiler.')
        
    if not np.all([term in data.columns for term in ['p_value', 'name', 'intersection_size','recall']]):
        raise TypeError('The data frame {} does not contain enrichment results from gprofiler.'.format(data))
    
    data_to_plot = data.iloc[:n_terms,:].copy()
    data_to_plot['row.id'] = data_to_plot.index

    min_pval = data_to_plot['p_value'].min() if min_pval is None else min_pval
    max_pval = data_to_plot['p_value'].max() if max_pval is None else max_pval
    
    # Min/max overlap
    if percent_size:
        n_genes=data.query_size.unique()[0]
    else:
        n_genes=None
        
    if n_genes is None:
        min_olap = data_to_plot['intersection_size'].min()
        max_olap = data_to_plot['intersection_size'].max()
    else:
        min_olap=1
        max_olap=n_genes
    
    # Scale intersection_size to be between 5 and 75 for plotting
    #Note: this is done as calibration was done for values between 5 and 75
    data_to_plot['scaled.overlap'] = scale_data_5_75(data_to_plot['intersection_size'],
                                                    mind=min_olap, maxd=max_olap)
    cmap="viridis_r"
    norm = colors.LogNorm(min_pval, max_pval)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    rcParams.update({'font.size': 12,'figure.figsize':(12,max(5,n_terms*0.3))})

    #sb.set(style="whitegrid")
    
    fig,ax=plt.subplots()
    path = plt.scatter(x='recall', y="name", c='p_value', cmap=cmap, 
                       norm=norm, 
                       data=data_to_plot, linewidth=1, edgecolor="grey", 
                       s=[(i+10)**1.5 for i in data_to_plot['scaled.overlap']])

    ax = plt.gca()
    ax.invert_yaxis()

    ax.set_ylabel('')
    if recall_lim is not None:
        max_recall=data_to_plot['recall'].max()
        if recall_lim[1]<max_recall:
            raise ValueError('Max recall lim is below max recall:',max_recall)
        ax.set_xlim(recall_lim[0],recall_lim[1])
    ax.set_xlabel('Gene set recall', fontsize=12)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Get tick marks for this plot
    #Note: 6 ticks maximum
    min_tick = np.floor(np.log10(min_pval)).astype(int)
    max_tick = np.ceil(np.log10(max_pval)).astype(int)
    tick_step = np.ceil((max_tick - min_tick)/6).astype(int)
    
    # Ensure no 0 values
    if tick_step == 0:
        tick_step = 1
        min_tick = max_tick-1
    
    ticks_vals = [10**i for i in range(max_tick, min_tick-1, -tick_step)]
    ticks_labs = ['$10^{'+str(i)+'}$' for i in range(max_tick, min_tick-1, -tick_step)]

    #Colorbar
    fig = plt.gcf()
    cbaxes = fig.add_axes([0.8, 0.15, 0.03, 0.4])
    cbar = ax.figure.colorbar(sm, ticks=ticks_vals, shrink=0.5, anchor=(0,0.1), cax=cbaxes)
    cbar.ax.set_yticklabels(ticks_labs)
    cbar.set_label("Adjusted p-value", fontsize=12)

    #Size legend
    
    #Note: approximate scaled 5, 25, 50, 75 values are calculated
    #      and then rounded to nearest number divisible by 5
    if n_genes is None:
        olap_range = max_olap - min_olap
        if max_olap>=25:
            size_leg_vals = [np.ceil(i/5)*5 for i in 
                         [min_olap, min_olap+(20/70)*olap_range, min_olap+(45/70)*olap_range, 
                                   max_olap]]
        else:
            size_leg_vals = [np.ceil(i) for i in 
                        [min_olap, min_olap+(20/70)*olap_range, min_olap+(45/70)*olap_range, 
                           max_olap]]
        labels = [str(int(i)) for i in size_leg_vals]
    else:
        ratio_vals=[0.05,0.25,0.5,0.75]
        size_leg_vals = [int(i*n_genes) for i in ratio_vals]
        
        labels = [str(i*100)+'%' for i in ratio_vals]
    size_leg_scaled_vals = scale_data_5_75(size_leg_vals,mind=min_olap,maxd=max_olap)
            
    l1 = plt.scatter([],[], s=(size_leg_scaled_vals[0]+10)**1.5, edgecolors='none', color='black')
    l2 = plt.scatter([],[], s=(size_leg_scaled_vals[1]+10)**1.5, edgecolors='none', color='black')
    l3 = plt.scatter([],[], s=(size_leg_scaled_vals[2]+10)**1.5, edgecolors='none', color='black')
    l4 = plt.scatter([],[], s=(size_leg_scaled_vals[3]+10)**1.5, edgecolors='none', color='black')
    

    leg = plt.legend([l1, l2, l3, l4], labels, ncol=1, frameon=False, fontsize=12,
                     handlelength=1, loc = 'center left', borderpad = 1, labelspacing = 1.4,
                     handletextpad=2, title='Gene overlap', scatterpoints = 1, 
                     bbox_to_anchor=(-2, 1.5), 
                     facecolor='black')

    if save:
        plt.savefig(save, dpi=300, format='pdf', bbox_inches='tight')

    plt.show()
    


# +
def gs_to_dict(gene_sets):
    """
    For transforming gene sets to dict
    :param gene_sets: gene sets in form of ListVector from hyperR
    :return: Dict k: gs name, v: genes
    """ 
    return dict(zip(gene_sets.names, map(list,list(gene_sets))))

def plot_enr_heatmap(data,n_gs=20,**kwargs):
    """
    Plot relationship between top enriched gene sets (sorted by fdr) and genes.
    :param data: Enrichment DF with columns fdr and hits with gs in index
    :param n_gs: N gene sets to plot (sorted by fdr). If None plot all
    """
    #def gs_to_dict(gene_sets):
    # For transforming gene sets to dict - unused as genes are listed in the results table
    #    return dict(zip(gene_sets.names, map(list,list(gene_sets))))
    data=data.copy()
    # For sorting by fold enrichment - unused. For this "genes" would be needed - this is query
    #data['fold_enrichment']=(data['overlap']/len(genes)
    #                                )/(data['geneset']/data['background'])
    data_to_plot=pd.DataFrame()
    #for gs,gs_enr in data.sort_values('fold_enrichment',ascending=False).iloc[:n_gs,:].iterrows():
    if n_gs is None:
        n_gs=data.shape[0]
    for gs,gs_enr in data.sort_values('fdr').iloc[:n_gs,:].iterrows():
        for gene in gs_enr['hits'].split(','):
            data_to_plot.at[gs,gene]=1
    data_to_plot.fillna(0,inplace=True)
    # TODO adjust cell width as relatively larger width is needed for less genes
    g=sb.clustermap(data_to_plot,
                    row_cluster=True if data_to_plot.shape[0]>2 else False,
                    col_cluster=True if data_to_plot.shape[1]>2 else False,
              figsize=(data_to_plot.shape[1]*0.2+10,data_to_plot.shape[0]*0.4),
              **kwargs)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 10)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 10)
    g.cax.set_visible(False)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    display(g.fig)
    plt.close()


# -

def hypeR_gs_to_dict(gene_sets):
    """
    Transform R object (hyperR) gene sets to dict
    """
    return dict(zip(gene_sets.names, map(list,list(gene_sets))))


def summarize_enr_coverage(data,genes,gene_sets):
    """
    Report N of query genes, N of query genes contained within used gene sets, and
    N query genes contained within enriched gene sets
    :param data: Enrichment results with column hits
    :param genes: Query genes
    :param gene_sets: Used gene_sets as dict
    """
    n_genes_enr=len(set([hit for hits in data['hits'] for hit in hits.split(',')]))
    genes=set(genes)
    anno_genes=set()
    for gs,gs_genes in gene_sets.items():
        anno_genes.update(set(gs_genes)&genes)
    print('N query genes:',len(genes),
          '\nN gs-annotated query genes:',len(anno_genes),
         '\nN query genes in enriched gene sets:',n_genes_enr) 


# +
class Scaler_1D:
    """
    Sklearn Scaler wrapper that is fit on 1D data and transforms individual values. 
    Additional preprocessing can be applied before scaling.
    """
    def __init__(self,scaler,scale_pp=lambda a:a):
        """
        :param scaler: Scaler instance, unfit.
        :param scale_pp: Function to apply to data values before fitting/transforming.
        Default is identity function.
        """
        self.scaler=scaler
        self.scale_pp=scale_pp
    def fit(self,values:iter):
        """
        :param values: values to fit the scaler on
        """
        values=[self.scale_pp(v) for v in values]
        values=np.array(values).reshape(-1,1)
        self.scaler.fit(values)

    def transform_value(self,val:float):
        """
        Transform single value.
        :param val: The value.
        :return: Transformed value.
        """
        val=self.scale_pp(val)
        return self.scaler.transform([[val]])[0][0]
    
def paga_composition(adata,groupby,thr_paga:float,
                     figsize_edge:float=8,lw:float=10,radius_ratio:tuple=(0.01,0.04),
                    cell_ratios_legend=[0.005,0.02,0.05,0.1],
                     ratio_legend_locy=0.4):
    """
    Plot PAGA graph with nodes plotted as piecharts showing composition of each node.
    :param adata: Adata with computed and plotted PAGA and saved groupby colours
    :param groupby: Adata.obs col for which composition will be shown.
    :param thr_paga: PAGA threshold which was used for plotting PAGA.
    :param figsize_edge: Figsize edge size, figure is always square.
    :param lw: linewidth of each edge is connectivity*lw
    :param radius_ratio: Radius of piecharts and size legend elements (min and max of 
    their union) will be between radius_ratio[0]*max_ax_range and radius_ratio[1]*max_ax_range, 
    where max_ax_range is the ax range of the longer axis. 
    This can be interpreted as radius being ratio of image.
    :param cell_ratios_legend: Cell ratios plotted in the size legend. If None will be set
    automatically.
    :param ratio_legend_locy: At which y to anchor ratios legend
    """
    paga=adata.uns['paga']
    # Extract connectivities and apply threshold
    connectivities=paga['connectivities'].copy().todense()
    connectivities[connectivities<thr_paga]=0
    # For scaling edges to specific range - UNUSED
    #scaler_lw=Scaler_1D(MinMaxScaler(feature_range=lw))
    #scaler_lw.fit(np.asarray(connectivities).ravel())
    # MAke square figure
    fig,ax=plt.subplots(figsize=(figsize_edge,figsize_edge))
    
    # Plot edges 
    
    # Edges are extracted from triangular matrix of connectivities
    n_nodes=connectivities.shape[0]
    for i in range(n_nodes-1):
        for j in range(i+1, n_nodes):
            if connectivities[i,j]>0:
                ax.plot([paga['pos'][i,0],paga['pos'][j,0]],[paga['pos'][i,1],paga['pos'][j,1]],
                        c='k',
                        #linewidth=scaler_lw.transform_value(connectivities[i,j]),
                        linewidth=lw*connectivities[i,j],
                        solid_capstyle='round')
    # Plot nodes
    
    # Get min and max radius size
    ax_range=max([paga['pos'][:,0].max()-paga['pos'][:,0].min(),# x range length
                  paga['pos'][:,0].max()-paga['pos'][:,0].min()]) #y range length      
    radius=np.array(radius_ratio)*ax_range
    # Get node labels and col used for making paga
    paga_col=paga['groups']            
    node_labels= adata.obs[paga_col].cat.categories
    # Get groupby ratios of each node
    # They are normalised in the pie plotting
    ratios=adata.obs.groupby(paga_col)[groupby].value_counts().rename(groupby+'_ratio'
                                                                     ).reset_index()
    # Set sort order for groupby to match the one from adata (for matching colours)
    ratios[groupby]=pd.Categorical(ratios[groupby], adata.obs[groupby].cat.categories)
    # Sizes of nodes - represent cell proportion from adata
    sizes=adata.obs[paga_col].value_counts(normalize=True)
    # Scale pie chart sizes so that their surface grows with amount of cells in each node
    scaler_s=Scaler_1D(MinMaxScaler(radius),scale_pp=lambda x: (x/3.14)**0.5)
    # Add automatic size legend items if absent (min, min+0.3range, min+0.7range, max)
    if cell_ratios_legend is None:
        cell_ratios_legend=[
                        eval("%.0e" % (sizes.min())),
                        eval("%.0e" % ((sizes.max()-sizes.min())*0.3+sizes.min())),
                        eval("%.0e" % ((sizes.max()-sizes.min())*0.7+sizes.min())),
                        eval("%.0e" % (sizes.max()))]
    # Fit scaler to union of actual sizes and legend sizes.
    sizes_all=cell_ratios_legend+list(sizes.values)
    scaler_s.fit([min(sizes_all),max(sizes_all)])
    # Plot nodes
    for i in range(n_nodes):
        node_label=node_labels[i]
        # Extract groupby ratios and sort them to match the colours from adata
        ratios_node=ratios[ratios[paga_col]==node_label
                          ].sort_values(groupby)[groupby+'_ratio'].values
        # Plot piechart
        pie_r=scaler_s.transform_value(sizes.at[node_label])
        pie  = ax.pie(ratios_node,center=paga['pos'][i], 
            colors = adata.uns[groupby+'_colors'], # Colour as in adata
            radius=pie_r,
                      # For testing radius size legend
                      #radius=scaler_s.transform_value(0.1),
            labels=[node_label]+['']*(ratios_node.shape[0]-1),
                textprops=dict(color="grey",weight='bold'),
                      labeldistance=0.26/pie_r, 
            frame=True)
        # plot piecharts on top
        for w in pie[0]:
            w.set_zorder(1000)
        # plot piechart labels on top
        pie[1][0].set_zorder(2000)
        #pie[1][0].set_verticalalignment("top")
        #pie[1][0].set_horizontalalignment("right")
        # At the beginning draw legend for piechart wedge colours - not as would only plot first pie colors

    legend_pathes=[]
    for group_idx,group in  enumerate(adata.obs[groupby].cat.categories):
        legend_pathes.append(Patch( color=adata.uns[groupby+'_colors'][group_idx]))
    legend1=ax.legend(legend_pathes, adata.obs[groupby].cat.categories,
              title=groupby,
              loc="upper left",
              bbox_to_anchor=(1.05,1))
    
    # Do not display axes ticks
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # make sure that figure has no margins before triing to plot legend 
    # that matches radius to scatterplot sizes
    fig.subplots_adjust(0, 0, 1-0, 1-0, 0, 0)
    def radius_to_s(fig,ax,radius):
        """
        Based on figure size and dpi and ax range convert radius to equivalent scatterplot s.
        Figure must be square and must have 0 margins
        scatterplot size.
        :param fig: Figure
        :param ax: Axes
        :param radius: Radius to match.
        :return: return s for scatterplot that so that point will be of equal size as 
        patch with given radius.
        """
        figsize_in=fig.get_size_inches()
        y_range=ax.get_ylim()[1]-ax.get_ylim()[0]
        x_range=ax.get_xlim()[1]-ax.get_xlim()[0]
        if figsize_in[0]!=figsize_in[1]:
            raise ValueError('Can be used only on squared figures.')
        # figsize_in[0] * fig.dpi = total points along edge
        # Divide by max([x_range,y_range]) - s-size of one unit (TODO figure out why
        # max of both axes seems to work here instead of sqrt of their product)
        # multiply s per unit with radius*2 - get s for the dimatere
        # **2 - for scatterplot s must be given **2
        return(figsize_in[0] * fig.dpi/max([x_range,y_range])*radius*2 )**2
    
    # Size legend
    
    # Make scatterpoints for legend handles
    legend_handles=[]
    for ratio in cell_ratios_legend:
        # radius of legend circle
        radius_legend=scaler_s.transform_value(ratio)
        # s of legend scatter point
        s_legend=radius_to_s(fig=fig,ax=ax,radius=radius_legend)
        # Add legend handle
        legend_handles.append( 
            plt.scatter([],[], s=s_legend, edgecolors='none', color='black'))
        
    # plot legend
    leg = ax.legend(legend_handles, cell_ratios_legend, ncol=1, frameon=True, fontsize=12,
                     handlelength=1, loc = 'center left', borderpad = 1, labelspacing = 1.6,
                     handletextpad=2, title='Population ratio', scatterpoints = 1, 
                     bbox_to_anchor=(1.05, ratio_legend_locy), 
                     facecolor='white')
    
    # Add the first (colour) legend that was now replaced
    fig.gca().add_artist(legend1)


# -
def split_citeseq_adata(adata,feature_type_col='feature_types',
                        feature_type_expr='Gene Expression',feature_type_ab='Antibody Capture',
                        suffix_expr='expr',suffix_ab='ab'
                       ):
    """
    Split cite-seq data to different modalities.
    :parm adata: Cite seq adata to split
    :param feature_type_col: Obs col containing modality info
    :param feature_type_expr: feature_type_col val denoting RNA modality
    :param feature_type_ab: feature_type_col val denoting protein modality
    :return: original adata, RNA modality, protein modality
    """
    adata_full=adata.copy()
    pdata=adata_full[:,adata_full.var[feature_type_col]==feature_type_ab].copy()
    adata=adata_full[:,adata_full.var[feature_type_col]==feature_type_expr].copy()
    return adata_full,adata,pdata


def map_gene_names(genes,use_col='MGI ID',path_genewalk=None,
                      path_anno='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/genomeAnno_ORGmus_musculus_V103_SSFltp_2y__MUC13974.tsv'
                     ):
    """
    Save gene symbols in format for gene walk
    :param genes: List of genes symbols from adata
    :param path_genewalk: Path prefix, genes are saved in 
    path_genewalk+'_DEgeneList_'+use_col.replace(' ','')+'.txt'. If None return instead of save
    :param path_anno: Location of gene symbol-EID map file
    """
    anno=pd.read_table(path_anno,index_col=0)
    ids=anno.loc[genes,use_col].dropna().values
    print('Mapped %i / %i genes'%(ids.shape[0],len(list(genes))))
    if path_genewalk is not None:
        with open(path_genewalk+'_DEgeneList_'+use_col.replace(' ','')+'.txt', 'w') as myfile:
            myfile.write('\n'.join(ids))
    else:
        return list(ids)


def opt_order(X,metric='correlation',method='ward'):
    """
    Get HC opt ordering
    :param X: obs*features
    :param metric: dist metric for pdist
    :param method: HC method for linkage
    """
    dist=pdist(X,metric=metric)
    hc=linkage(dist, method=method,  optimal_ordering=True)
    return leaves_list(hc)


def opt_order_withincl(X:pd.DataFrame,clusters:pd.Series,cl_order:list=None,
                       metric='correlation',method='ward'):
    """
    Get HC opt ordering for within clusters. Clusters are ordered as in groupby
    :param X: obs*features
    :param clusters: Clusters for obvs, to be added to X as X['cl']=clusters
    :param cl_order: In which order gene clusters should be reported
    :param metric: dist metric for pdist
    :param method: HC method for linkage
    """
    X=X.copy()
    # Sometimes not working for some reason
    #X['cl']=clusters
    clusters.name='cl'
    X=pd.concat([X,clusters],axis=1)
    # Sort genes within clusters
    genes_dict=X.groupby('cl').apply(
                 # Within each cl sort genes
                 lambda x: x.index.values[
                     opt_order(x.drop('cl',axis=1),metric='correlation',method='ward')]
                ).to_dict()
    # Report genes in this cl order (e.g. ordered genes within each cluster reported with
    # specific cluster order)
    if cl_order is None:
        cl_order=genes_dict.keys()
    return [obs for cl in cl_order
            for obs in genes_dict[cl]]


# +
def age_months(x):
    """
    Return age in months
    :param x: Age string of format N d/m/y for days/months/years
    :return: Age in months
    """
    unit=x.split()[1]
    n=float(x.split()[0])
    if unit=='d':
        n=n/30.0
    elif unit=='m':
        n=n
    elif unit=='y':
        n=n*12.0
    else:
        raise ValueError('Unrecognised unit')
    return n

def age_years(x):
    """
    Return age in years
    :param x: Age string of format N d/m/y for days/months/years
    :return: Age in years
    """
    unit=x.split()[1]
    n=float(x.split()[0])
    if unit=='d':
        n=n/365.0
    elif unit=='m':
        n=n/12.0
    elif unit=='y':
        n=n
    else:
        raise ValueError('Unrecognised unit')
    return n

def age_days(x):
    """
    Return age in days
    :param x: Age string of format N d/m/y for days/months/years
    :return: Age in days
    """
    unit=x.split()[1]
    n=float(x.split()[0])
    if unit=='d':
        n=n
    elif unit=='m':
        n=n*30.0
    elif unit=='y':
        n=n*365.0
    else:
        raise ValueError('Unrecognised unit')
    return n

def age_weeks(x):
    """
    Return age in weeks
    :param x: Age string of format N d/m/y for days/months/years
    :return: Age in weeks
    """
    unit=x.split()[1]
    n=float(x.split()[0])
    if unit=='d':
        n=n/7
    elif unit=='m':
        n=n*30.0/7
    elif unit=='y':
        n=n*365.0/7
    else:
        raise ValueError('Unrecognised unit')
    return n



# +

def weighted_knn(train_adata, valid_adata, label_key, n_neighbors=50, threshold=0.5,
                 pred_unknown=True, #mode='package',
                 label_key_valid=None):
    """Annotates ``valid_adata`` cells with a trained weighted KNN classifier on ``train_adata``.
        Parameters
        Adapted from https://github.com/NUPulmonary/scarches-covid-reference/blob/master/sankey.py and https://github.com/theislab/scarches/blob/51f9ef4ce816bdc0522cb49c6a0eb5d976a59f22/scarches/annotation.py#L9 
        ----------
        train_adata: :class:`~anndata.AnnData`
            Annotated dataset to be used to train KNN classifier with ``label_key`` as the target variable.
        valid_adata: :class:`~anndata.AnnData`
            Annotated dataset to be used to validate KNN classifier.
        label_key: str
            Name of the column to be used as target variable (e.g. cell_type) in ``train_adata`` 
        n_neighbors: int
            Number of nearest neighbors in KNN classifier.
        threshold: float
            Threshold of uncertainty used to annotating cells as "Unknown". cells with uncertainties upper than this
             value will be annotated as "Unknown".
        pred_unknown: bool
            ``True`` by default. Whether to annotate any cell as "unknown" or not. If `False`, will not use
            ``threshold`` and annotate each cell with the label which is the most common in its
            ``n_neighbors`` nearest cells.
        mode: str
            Has to be one of "paper" If mode is set to "package", uncertainties will be 1 - P(pred_label),
            Unavailiable "paper" (uncertainities will be 1 - P(true_label)).
        label_key_valid: str
            Name of the column to be used as target variable (e.g. cell_type) in ``valid_adata``.
            If None does not report prediction accuracy.
    """
    print(f'Weighted KNN with n_neighbors = {n_neighbors} and threshold = {threshold} ... ', end='')
    k_neighbors_transformer = KNeighborsTransformer(n_neighbors=n_neighbors, mode='distance',
                                                    algorithm='brute', metric='euclidean',
                                                    n_jobs=-1)
    k_neighbors_transformer.fit(train_adata.X)

    y_train_labels = train_adata.obs[label_key].values
    if label_key_valid is not None:
        y_valid_labels = valid_adata.obs[label_key_valid].values

    top_k_distances, top_k_indices = k_neighbors_transformer.kneighbors(X=valid_adata.X)

    stds = np.std(top_k_distances, axis=1)
    stds = (2. / stds) ** 2
    stds = stds.reshape(-1, 1)

    top_k_distances_tilda = np.exp(-np.true_divide(top_k_distances, stds))

    weights = top_k_distances_tilda / np.sum(top_k_distances_tilda, axis=1, keepdims=True)

    uncertainties = []
    pred_labels = []
    for i in range(len(weights)):
        unique_labels = np.unique(y_train_labels[top_k_indices[i]])
        best_label, best_prob = None, 0.0
        for candidate_label in unique_labels:
            candidate_prob = weights[i, y_train_labels[top_k_indices[i]] == candidate_label].sum()
            if best_prob < candidate_prob:
                best_prob = candidate_prob
                best_label = candidate_label
        
        if pred_unknown:
            if best_prob >= threshold:
                pred_label = best_label
            else:
                pred_label = 'Unknown'
        else:
            pred_label = best_label

        #if mode == 'package':
        uncertainties.append(max(1 - best_prob, 0))

        #elif mode == 'paper':
        #    if pred_label == y_valid_labels[i]:
        #        uncertainties.append(max(1 - best_prob, 0))
        #    else:
         #       true_prob = weights[i, y_train_labels[top_k_indices[i]] == y_valid_labels[i]].sum()
        #        if true_prob > 0.5:
        #            pass
        #        uncertainties.append(max(1 - true_prob, 0))
        #else:
        #    raise Exception("Invalid Mode!")

        pred_labels.append(pred_label)

    pred_labels = np.array(pred_labels).reshape(-1,)
    uncertainties = np.array(uncertainties).reshape(-1,)
    
    if label_key_valid is not None:
        labels_eval = pred_labels == y_valid_labels
        labels_eval = labels_eval.astype(object)
    
        n_correct = len(labels_eval[labels_eval == True])
        n_incorrect = len(labels_eval[labels_eval == False]) - len(labels_eval[pred_labels == 'Unknown'])
        n_unknown = len(labels_eval[pred_labels == 'Unknown'])
    
        labels_eval[labels_eval == True] = f'Correct'
        labels_eval[labels_eval == False] = f'InCorrect'
        labels_eval[pred_labels == 'Unknown'] = f'Unknown'
        
        valid_adata.obs['evaluation'] = labels_eval
        
        print(f"Number of correctly classified samples: {n_correct}")
        print(f"Number of misclassified samples: {n_incorrect}")
        print(f"Number of samples classified as unknown: {n_unknown}")
    
    valid_adata.obs['uncertainty'] = uncertainties
    valid_adata.obs[f'pred_{label_key}'] = pred_labels
    
    
    print('finished!')

# -

def get_symbols(eids:list):
    """
    Map eids from mouse atlas adata back to symbols
    :return: symbols
    """
    # Load symbol info
    path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/combined/'
    path_genes='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/gene_lists/'
    var=sc.read(path_data+'data_rawnorm_integrated_analysed_beta_v1s1_sfintegrated.h5ad',
               backed='r').var.copy().fillna('nan')
    gene_names_df=pd.read_table(path_genes+'genomeAnno_ORGmus_musculus_V103.tsv',index_col=0
                           ).fillna('nan')
    
    # Add official gene symbols from Ensembl
    symbols=gene_names_df.loc[eids,'Gene name']
    symbols.name='gene_symbol'
    symbols=pd.DataFrame(symbols)
    # If missing add symbols from adata (matched across studies)
    # (Some genes not matching across studies thus nan)
    symbols['gene_symbol']=symbols.apply(
        lambda x: x['gene_symbol'] if  x['gene_symbol']!='nan'
            else var.loc[x.name,'gene_symbol_original_matched'],axis=1)
    # If still missing add symbol from original study (not all studies have all genes)
    symbols['gene_symbol']=symbols.apply(
        lambda x: x['gene_symbol'] if  x['gene_symbol']!='nan'
        else [g for g in gene_names_df.loc['ENSMUSG00000117790',
                      [c for c in gene_names_df.columns if 'gene_symbol_' in c]].unique()
                 if g!='nan'][0],axis=1)
    return symbols['gene_symbol']
