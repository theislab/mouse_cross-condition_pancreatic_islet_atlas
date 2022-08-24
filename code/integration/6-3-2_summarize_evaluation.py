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
# # Analyse integration evalaluation results
# Compared are:
# - Different ambient correction methods: top ambient genes removal ("ambientMasked" for the shorter ambient genes list or "ambientMaskedExtended" for the longer list) and different ambient correction tools followed by top ambient removal. Ambient correction tools: DecontX with default learnt corection parameters or more strong amibent removal (termed "fixed"); SoupX wit default correction strength or with increased correction strength (indicated by "rhoadd" - higher number means larger correction). 
# - Different integration methods: scVI, scArches.
# - scArches with default learning protocol (termed "quick") or protocol with slower learning (termed "slow")
# - Multiple re-runs of the same setting to estimate variability across multiple integration runs (denoted by a number from 0 to 2 at the end of the run name).
#
# For integration of all cell types we computed evaluation on cell-type annotated cells (annotated for the ref datasets) - all cell types or beta cells only (evaluation of bio preservation and batch correction) or on all cells (including those without cell type annotation; computed only a subset of metrics that could be computed without cell type annotation). For beta cell integration we computed batch metrics on all cells and bio metrics on cell-type annotated cells onnly.
#

# %%
import pandas as pd
import glob
import os
import pickle
import numpy as np

import matplotlib.pylab as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from  matplotlib.patches import  Patch
import seaborn as sb

from sklearn.preprocessing import minmax_scale

# %%
# Path to run results
runs_path='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/'

# %%
# Saving figures
path_fig='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/figures/paper/'

# %%
BIO=['graph_cLISI', 'NMI', 'ARI', 'ASW_cell_type', 'isolated_label_F1', 'isolated_label_ASW','moransi_conservation'] 
BATCH=['PC_regression', 'ASW_batch', 'kBET', 'kBET_study','graph_connectivity', 'graph_iLISI']

# %% [markdown]
# For initial evaluation we computed kBET both on sample (kBET, as other metrics) and study (kBET_study) as batch. However, for computation of merics summaries in paper we only used the one using sample as batch.

# %% [markdown]
# ## Integration of all cell types

# %%
# Get all evaluation metric results 
run_dirs=[
          ('combined/scArches/integrate_combine_individual/run_scArches1603792372.695119/','scArches;ambientMasked;quick;0'),
          ('combined/scArches/integrate_combine_individual/run_scArches1649916019.148133/','scArches;ambientMasked;quick;1'),
          ('combined/scArches/integrate_combine_individual/run_scArches1650008190.874977/','scArches;ambientMasked;quick;2'),
          ('combined/scArches/integrate_combine_individual/run_scArches1605997831.626417/','scArches;ambientMasked;slow;0'),
          ('combined/scArches/integrate_combine_individual/run_scArches1605997831.626664/','scArches;ambientMasked;slow;1'),
          ('combined/scArches/integrate_combine_individual/run_scArches1606480383.88613/','scArches;ambientMasked;slow;2'),
          ('combined/scArches/integrate_combine_individual/run_scArches1605798604.647761/','scArches;ambientMaskedExtended;slow;0'),
          ('combined/scArches/integrate_combine_individual/run_scArches1606061035.453162/','scArches;ambientMaskedExtended;slow;1'),
          ('combined/scArches/integrate_combine_individual/run_scArches1606061035.453407/','scArches;ambientMaskedExtended;slow;2'),
          ('combined/scArches/integrate_combine_individual/run_scArches1603881821.939477/','scArches;ambientMasked+decontX;slow'),
          ('combined/scArches/integrate_combine_individual/run_scArches1604502149.078628/','scArches;ambientMasked+decontX_fixed;slow'),
          ('combined/scArches/integrate_combine_individual/run_scArches1612129003.819243/','scArches;ambientMasked+SoupX_rhoadd0;slow'),
          ('combined/scArches/integrate_combine_individual/run_scArches1612129003.818745/','scArches;ambientMasked+SoupX_rhoadd005;slow'),
          ('combined/scArches/integrate_combine_individual/run_scArches1612129003.818814/','scArches;ambientMasked+SoupX_rhoadd01;slow'),
          ('combined/scArches/integrate_combine_individual/run_scArches1612217039.745045/','scArches;ambientMasked+SoupX_rhoadd0;quick;0'),
          ('combined/scArches/integrate_combine_individual/run_scArches1649794207.047576/','scArches;ambientMasked+SoupX_rhoadd0;quick;1'),
          ('combined/scArches/integrate_combine_individual/run_scArches1649618335.163001/','scArches;ambientMasked+SoupX_rhoadd0;quick;2'),
          ('combined/scArches/integrate_combine_individual/run_scArches1612217039.744893/','scArches;ambientMasked+SoupX_rhoadd005;quick'),
          ('combined/scArches/integrate_combine_individual/run_scArches1612217039.744912/','scArches;ambientMasked+SoupX_rhoadd01;quick'),
          ('combined/scVI/runs/1603814626.572086_SB0/','scVI;ambientMasked'),
          ('combined/scVI/runs/1603814626.761494_SB0/','scVI;ambientMasked+decontX'),
          ('combined/scVI/runs/1604500510.973771_SB0/','scVI;ambientMasked+decontX_fixed'),
          ('combined/scVI/runs/1612129492.366433_SB0/','scVI;ambientMasked+SoupX_rhoadd0'),
          ('combined/scVI/runs/1612129479.815399_SB0/','scVI;ambientMasked+SoupX_rhoadd005'),
          ('combined/scVI/runs/1612129526.764645_SB0/','scVI;ambientMasked+SoupX_rhoadd01')
         ]
    
    

# %%
# Store metrics results and corresponding parameters in df
res_df=[]
index=[]
cell_subsets=[('integration_metrics.pkl','all'),('integration_metrics_beta.pkl','beta'),
             ('integration_metrics_allCells.pkl','all_allCells')]
for run_dir, run_name in run_dirs:
    for file, subset in cell_subsets:
        if os.path.isfile(runs_path+run_dir+file):
            run_metrics=pickle.load( open(  runs_path+run_dir+file, "rb" ) )
            run_metrics['subset']=subset
            res_df.append(run_metrics)
            index.append(run_name)
res_df=pd.DataFrame(res_df)
res_df.index=index

# %%
# Sort runs for plotting
run_datas=[]
for run_name in res_df.index.unique():
    run_datas.append({'run':run_name,
                      'method':run_name.split(';')[0],
                      'ambient_corr':run_name.split(';')[1]
                         })
run_datas=pd.DataFrame(run_datas)
run_datas.sort_values(['method','ambient_corr'],inplace=True)
res_df=res_df.loc[run_datas['run'],:]

# %%
res_df.sort_values('subset')


# %%
def dotplot(res_df,h=None,w=None,h_scale=0.7,w_scale=0.6,margins=(0.1,0.03),color='rank'):
    """
    Plot scores as dotplot, separately for bio and batch. Keep only metrics that are not na in any sample.
    :param res_df: Results df to use.
    :param color: "rank" - color by rank or "normalised" - color by score normalised to [0,l]
    """
    # Keep metric that are present and not na in any sample 
    # and that are correctly scaled between 0 and 1 - the latter check is no longer needed 
    # as since metrics have been corrected
    bio=[metric for metric in BIO if metric in res_df.columns and 
         not res_df[metric].isna().any() 
         and (res_df[metric]>=0).all() and (res_df[metric]<=1).all()]
    batch=[metric for metric in BATCH if metric in res_df.columns and 
           not res_df[metric].isna().any()
          and (res_df[metric]>=0).all() and (res_df[metric]<=1).all()]
    # Vstack columns for plotting
    res_df_stacked= res_df[bio+batch].copy()
    res_df_stacked= res_df_stacked.stack().reset_index()
    res_df_stacked.columns=['run','metric','score']
    res_df_stacked['sqrt_score']=np.sqrt(res_df_stacked['score'])

    # Rank  runs in individual metrics
    res_df_stacked['rank']=res_df[bio+batch].rank(axis=0,ascending=True).stack().values # Give larger number (rank) to higher (better) value
    # Normalised scores
    res_df_stacked['normalised']=pd.DataFrame(minmax_scale(res_df[bio+batch]),
                                                    index=res_df.index,columns=bio+batch
                                             ).stack().values
    
    # Plot dotplot
    if h is None:
        h=res_df.shape[0]*h_scale
    if w is None:
        w=len(bio+batch)*w_scale
    rcParams['figure.figsize']= (w,h)
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
    data_bio=res_df_stacked.query('metric in @bio')
    data_batch=res_df_stacked.query('metric in @batch')
    g=sb.scatterplot(x='metric',y='run',data=data_bio,
                     hue=color,size='sqrt_score',
                     legend=False,
                     sizes=(data_bio['sqrt_score'].min()*(600-80)+80, 
                        data_bio['sqrt_score'].max()*(600-80)+80), 
                     palette='viridis', ax=ax1)
    ax1.margins(*margins)
    ax1.set_xlabel('bio')
    g=sb.scatterplot(x='metric',y='run',data=data_batch,hue=color,size='sqrt_score',
                     legend=False,
                     sizes=(data_batch['sqrt_score'].min()*(600-80)+80, 
                            data_batch['sqrt_score'].max()*(600-80)+80),
                     palette='viridis',ax=ax2)
    ax2.margins(*margins)
    ax2.set_xlabel('batch')
    a=fig.autofmt_xdate(rotation=90 )


# %% [markdown]
# In the dotplots the color indicates scaled score across runs (yelow - better, violet - worse) and the size the non-scaled score value.

# %% [markdown]
# #### Scores computed on all annotated cells

# %%
dotplot(res_df.query('subset == "all"'))

# %% [markdown]
# #### Scores computed on cells annotated as beta cells

# %%
dotplot(res_df.query('subset == "beta"'))

# %% [markdown]
# #### Scores computed on all cells 
# Regardless of having the annotation or not and what the annotation was. Some metrics could not be computed due tto lack of cell type annotation. For some metrics (e.g. batch) we used clusters instead of cell types for computation.

# %%
dotplot(res_df.query('subset == "all_allCells"'),w=7)


# %% [markdown]
# ### Rank based on overall score
# Compute rank as in scBI: 0.6 * bio + 0.4 * batch. Excluder kBET_study from computation of scores as this metric does not have the same batch variable as other metrics (study instead of sample; there is a kBET also computed on sample).

# %%
def rank(res_df,sort:bool=True,ignore:list=['kBET_study'],return_df=False,plot=True):
    """
    Compute average bio and batch score and their weighted mean. 
    Use only metrics that are not NA in any of the samples.
    Plot as heatmap.
    :param res_df: Results df to use.
    :param ignore: metrics to ignore
    """
    # Scale metrics to be on same scale [0,1]
    bio=[metric for metric in BIO 
         if metric in res_df.columns and not res_df[metric].isna().any() 
         and (res_df[metric]>=0).all() and (res_df[metric]<=1).all()
         and metric not in ignore]
    batch=[metric for metric in BATCH 
           if metric in res_df.columns and not res_df[metric].isna().any() 
           and (res_df[metric]>=0).all() and (res_df[metric]<=1).all()
           and metric not in ignore]
    res_df_scaled=pd.DataFrame(minmax_scale(res_df[bio+batch]),index=res_df.index,columns=bio+batch)
    # Average scores
    mean_bio=res_df_scaled[bio].mean(axis=1)
    mean_batch=res_df_scaled[batch].mean(axis=1)
    mean_score=(mean_bio*0.6 + mean_batch*0.4)
    mean_score=pd.DataFrame([mean_bio,mean_batch,mean_score],index=['bio','batch','overall']).T
    # Plot heatmap of summarized scores
    if sort:
        sort_by='overall'
        if len(bio) ==0 and len(batch)==0:
            raise ValueError("Neither bio or batch were computed - can't sort")
        elif len(bio)==0:
            sort_by='batch'
        elif len(batch)==0:
            sort_by='bio'
        mean_score=mean_score.sort_values(sort_by,ascending=False)
    if plot:
        rcParams['figure.figsize']= (5,res_df.shape[0]*0.4)
        sb.heatmap(mean_score, vmin=0, vmax=1)
    if return_df:
        return mean_score


# %% [markdown]
# Ranking based on scores of annotated cells

# %%
rank_all=rank(res_df.query('subset == "all"'),return_df=True)

# %%
rank_all['method']=[idx.split(';')[0] for idx in rank_all.index]
rank_all['ambient']=[idx.split(';')[1] for idx in rank_all.index]
rank_all['lr']=[idx.split(';')[2] if len(idx.split(';'))>=3 else 'NA' for idx in rank_all.index]

# %%
methods=['scVI','scArches']
g = sb.catplot(
    data=rank_all.query('method in @methods'), 
    x="method", y="bio", hue="ambient",palette="dark", alpha=.6, height=6,s=10,
    kind='swarm'
)
g.despine(left=True)
g.set_axis_labels("method", "bio")
g.fig.suptitle('ref annotated cells')

# %%
methods=['scVI','scArches']
g = sb.catplot(
    data=rank_all.query('method in @methods'), 
    x="method", y="batch", hue="ambient", palette="dark", alpha=.6, height=6,s=10,
    kind='swarm'
)
g.despine(left=True)
g.set_axis_labels("method", "batch")
g.fig.suptitle('ref annotated cells')

# %% [markdown]
# Ranking based on scores computed on cell annotated as beta cells

# %%
rank_beta=rank(res_df.query('subset == "beta"'),return_df=True)

# %%
rank_beta['method']=[idx.split(';')[0] for idx in rank_beta.index]
rank_beta['ambient']=[idx.split(';')[1] for idx in rank_beta.index]

# %%
methods=['scVI','scArches']
g = sb.catplot(
    data=rank_beta.query('method in @methods'), 
    x="method", y="bio", hue="ambient", palette="dark", alpha=.6, height=6,s=10,
    kind='swarm'
)
g.despine(left=True)
g.set_axis_labels("method", "bio")
g.fig.suptitle('ref annotated beta cells')

# %%
methods=['scVI','scArches']
g = sb.catplot(
    data=rank_beta.query('method in @methods'), 
    x="method", y="batch", hue="ambient", palette="dark", alpha=.6, height=6,s=10,
    kind='swarm'
)
g.despine(left=True)
g.set_axis_labels("method", "batch")
g.fig.suptitle('ref annotated beta cells')

# %% [markdown]
# Ranking based on all cells, including non-annotated ones. Note that here a single metric was used in bio score.

# %%
rank_allAll=rank(res_df.query('subset == "all_allCells"'),return_df=True)

# %%
rank_allAll['method']=[idx.split(';')[0] for idx in rank_allAll.index]
rank_allAll['ambient']=[idx.split(';')[1] for idx in rank_allAll.index]

# %%
methods=['scVI','scArches']
g = sb.catplot(
    data=rank_allAll.query('method in @methods'), 
    x="method", y="bio", hue="ambient", palette="dark", alpha=.6, height=6,s=10,
    kind='swarm'
)
g.despine(left=True)
g.set_axis_labels("method", "batch")
g.fig.suptitle('all cells')

# %%
methods=['scVI','scArches']
g = sb.catplot(
    data=rank_allAll.query('method in @methods'), 
    x="method", y="batch", hue="ambient", palette="dark", alpha=.6, height=6,s=10,
    kind='swarm'
)
g.despine(left=True)
g.set_axis_labels("method", "batch")
g.fig.suptitle('all cells')

# %% [markdown]
# ### Final plot

# %%
subset_map={'all':'all annotated',
            'all_allCells':'all',
            'beta':'beta annotated',
           }

# %%
# Compute bio and batch scores for all subsets
ranks=[]
for subset in res_df.subset.unique():
    r=rank(res_df.query('subset==@subset'),ignore=['kBET_study'],return_df=True,plot=False)
    r['subset']=subset_map[subset]
    ranks.append(r)
ranks=pd.concat(ranks)
ranks['method']=[idx.split(';')[0] for idx in ranks.index]
ranks['ambient']=[idx.split(';')[1] for idx in ranks.index]
ranks['lr']=[idx.split(';')[2] if len(idx.split(';'))>=3 else 'NA' for idx in ranks.index]

# %%
# Palettes
palette={
    'ambientMasked':'#757575', 
    'ambientMaskedExtended':'#000000',
    'ambientMasked+SoupX_rhoadd0':'#e0b002',
    'ambientMasked+SoupX_rhoadd005':'#e07502',
    'ambientMasked+SoupX_rhoadd01':'#b03005', 
    'ambientMasked+decontX':'#45ddf5',
    'ambientMasked+decontX_fixed':'#0a8ec7', 
}
markers={'quick':'x',
        'slow':'o',
        'NA':'v'}


# %%
def simple_beeswarm(y, nbins=None):
    # From https://stackoverflow.com/questions/36153410/how-to-create-a-swarm-plot-with-matplotlib
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    y = np.asarray(y)
    if nbins is None:
        nbins = len(y) // 4

    # Get upper bounds of bins
    x = np.zeros(len(y))
    ylo = np.min(y)
    yhi = np.max(y)
    dy = (yhi - ylo) / nbins
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide indices into bins
    i = np.arange(len(y))
    ibs = [0] * nbins
    ybs = [0] * nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices
    dx = 1 / (nmax // 2)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(y)]
            a = i[j::2]
            b = i[j+1::2]
            x[a] = np.arange(1, 1 + len(a)) * dx
            x[b] = np.arange(1, 1 + len(b)) * -dx

    return x
    


# %%
# Plot scatter per subset and bio/batch, marking pp methods, lr, and final selection
# Fig of size bio/batch*subsets
ncol=ranks.subset.nunique()
fig,axs=plt.subplots(2,ncol,figsize=(3*ncol,2.5*2),sharex=True)
plt.subplots_adjust(hspace=0.0)
for idxr,metric in enumerate(['bio','batch']):
    for idxc,subset in enumerate(subset_map.values()):
        ax=axs[idxr,idxc]
        ranks_sub=ranks.query('subset==@subset')
        n_methods=ranks_sub.method.nunique()
        methods={}
        # plot methods (scIV/scarches) into 2 columns
        for idx1,(method,ranks_sub1) in enumerate(ranks_sub.groupby('method')):
            # Compute x of each point - swarm and col position
            ns=ranks_sub1.index
            ys=ranks_sub1[metric]
            xs=simple_beeswarm(ys)
            x_mid=idx1*3
            xs=xs+x_mid
            # Plot each point
            methods[method]=x_mid
            cs=[palette[c] for c in ranks_sub1.ambient]
            ms=[markers[c] for c in ranks_sub1.lr]
            for  x,y,c,m,n in zip(xs,ys,cs,ms,ns):
                ax.scatter(x,y,c=c,marker=m,s=40)
                if n=='scArches;ambientMasked;quick;0':
                     ax.scatter(x,y,marker='o',s=180,
                                facecolors='none', edgecolors='#91db5c')
        # Label columns with method names                
        ax.set_xlim(-1.5,x_mid+1.5)
        ax.set_xticks(list(methods.values()))
        ax.set_xticklabels(labels=list(methods.keys()))
        
        # Titles
        if idxr==0:
            ax.set_title(subset)
        if idxc==0:
            ax.set_ylabel(metric)
        
        # Seethrough
        ax.set(facecolor = (0,0,0,0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend
        if idxr==0 and idxc==ncol-1:
            legend_elements = [ Patch(alpha=0,label='ambient correction')]+[
                 Patch(facecolor=c,label=s) for s,c in palette.items()
            ]+[ Patch(alpha=0,label='\nlearning rate')]+[
                 Line2D([0],[0],marker=m, color='k',  markerfacecolor='k', 
                        markersize=8,lw=0,label=l) for l,m in markers.items()
            ]+[Line2D([0],[0],marker='o', color='k',  markerfacecolor='none', 
                        markersize=0,markeredgecolor='none',lw=0,label='') 
              ]+[Line2D([0],[0],marker='o', color='k',  markerfacecolor='none', 
                        markersize=12,markeredgecolor='#91db5c',lw=0,label='selected') ]
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05,1.01),frameon=False)
plt.savefig(path_fig+'swarmplot_atlas_integrationSummary.png',dpi=300,bbox_inches='tight')

# %% [markdown]
# ## Beta cell integration
# Evaluation of integration when using only beta cells (using beta cells that were annotated as such of the all-cells integration (above) re-annotation). Compare to the best integration of all cell types (above) using the same set of cells for evaluation.

# %%
# Get all evaluation metric results 
run_dirs=[('combined/scArches/integrate_combine_individual/run_scArches1603792372.695119/beta_integrated/','allCT_scArches;ambientMasked;alpha0.99'),
          ('combined/scArches/integrate_combine_individual/run_scArches1612798260.979096/beta_integrated/','beta_scArches;ambientMasked;alpha0.99'),
          ('combined/scArches/integrate_combine_individual/run_scArches1612875550.182888/beta_integrated/','beta_scArches;ambientMasked;alpha0.9'),
          ('combined/scArches/integrate_combine_individual/run_scArches1612875550.182689/beta_integrated/','beta_scArches;ambientMasked;alpha0.8'),
          ('combined/scArches/integrate_combine_individual/run_scArches1613645086.693854/beta_integrated/','beta_scArches;ambientMasked;alpha0.1'),
          ('combined/scArches/integrate_combine_individual/run_scArches1613645086.693856/beta_integrated/','beta_scArches;ambientMasked;alpha0.2'),
          ('combined/scArches/integrate_combine_individual/run_scArches1613744258.263764/beta_integrated/','beta_scArches;ambientMasked;alpha0.3'),
          ('combined/scArches/integrate_combine_individual/run_scArches1613744258.219381/beta_integrated/','beta_scArches;ambientMasked;alpha0.4'),
          ('combined/scArches/integrate_combine_individual/run_scArches1613645086.694957/beta_integrated/','beta_scArches;ambientMasked;alpha0.5'),
          ('combined/scArches/integrate_combine_individual/run_scArches1613653844.206358/beta_integrated/','beta_scArches;ambientMasked;alpha0.01')
         ]
    


# %%
# Store metrics results and corresponding parameters in df
res_df=[]
index=[]
cell_subsets=[('integration_metrics.pkl','annotated'),
             ('integration_metrics_allCells.pkl','allCells')]
for run_dir, run_name in run_dirs:
    for file, subset in cell_subsets:
        if os.path.isfile(runs_path+run_dir+file):
            run_metrics=pickle.load( open(  runs_path+run_dir+file, "rb" ) )
            run_metrics['subset']=subset
            res_df.append(run_metrics)
            index.append(run_name)
res_df=pd.DataFrame(res_df)
res_df.index=index

# %%
# Sort runs for plotting
run_datas=[]
for run_name in res_df.index.unique():
    run_datas.append({'run':run_name,
                      'method':run_name.split(';')[0],
                      'alpha':float(run_name.split(';')[2].replace('alpha',''))
                         })
run_datas=pd.DataFrame(run_datas)
run_datas.sort_values(['method','alpha'],inplace=True)
res_df=res_df.loc[run_datas['run'],:]

# %%
res_df

# %% [markdown]
# #### On annotated cells with annotated cell subtype
# Here we did not claculate batch score on annotated cells as we decided to use the one computed on all cells.

# %%
res_df.query('subset == "annotated"')

# %% [markdown]
# Scores colored by rank

# %%
dotplot(res_df.query('subset == "annotated"'),w_scale=0.9,margins=(0.12,0.2))

# %% [markdown]
# Scores colored by normalised score across runs

# %%
dotplot(res_df.query('subset == "annotated"'),w_scale=0.9,margins=(0.12,0.2),color='normalised')

# %%
rank_annotated=rank(res_df.query('subset=="annotated"'),return_df=True)


# %% [markdown]
# #### On all cells, regardless of subtype annotation

# %%
res_df.query('subset == "allCells"')

# %% [markdown]
# Scores colored by rank

# %%
dotplot(res_df.query('subset == "allCells"'),w_scale=1,margins=(0.12,0.2))

# %% [markdown]
# Scores colored by normalised score across runs

# %%
dotplot(res_df.query('subset == "allCells"'),w_scale=1,margins=(0.12,0.2),color='normalised')

# %% [markdown]
# In ranking only Moran's I is used for bio preservation as other bio metrics were not computed.

# %%
rank_annotated=rank(res_df.query('subset=="allCells"'),return_df=True)


# %% [markdown]
# ### Final plot
# Batch metric comupted on all cells vs bio metric computed on annotated cells.
# We decided to use bio metric from annotated cells as it is an average over more individual metrics.

# %%
subset_map={'annotated':'annotated subtype',
            'allCells':'all',
           }

# %%
# Get score summaries
ranks=[]
for subset in res_df.subset.unique():
    r=rank(res_df.query('subset==@subset'),ignore=['kBET_study'],return_df=True,plot=False)
    r['subset']=subset_map[subset]
    ranks.append(r)
ranks=pd.concat(ranks)
ranks['integrated\ncell types']=[r.split('_')[0].replace('CT','') for r  in ranks.index]

# %%
# Bio (compute on annotated cells) vs batch (compute on all cells)
rcParams['figure.figsize']=(2.5,2.5)
g=sb.scatterplot(x=ranks.query('subset=="all"')['batch'].rename('batch - all cells'),
               y=ranks.query('subset=="annotated subtype"')['bio'].rename('bio - subtype annotated cells'),
              hue=ranks.query('subset=="all"')['integrated\ncell types'],s=50,
                palette={'all':'#8a9e59','beta':'#c97fac'})
g.legend(bbox_to_anchor=(1.6,1.03),title='integrated\ncell types',frameon=False)
g.set(facecolor = (0,0,0,0))
g.spines['top'].set_visible(False)
g.spines['right'].set_visible(False)
g.yaxis.set_label_coords(-0.2,0.4)
plt.savefig(path_fig+'scatterplot_beta_integrationSummary.png',dpi=300,bbox_inches='tight')
