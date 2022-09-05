# +
import numpy as np
import pandas as pd
import patsy
from typing import Union, List, Dict, Callable, Tuple
import anndata
try:
    from anndata.base import Raw
except ImportError:
    from anndata import Raw
import scipy
import batchglm.api as glm
import diffxpy
import logging

from diffxpy.testing.utils import parse_gene_names, parse_sample_description, parse_size_factors,\
  constraint_system_from_star
from diffxpy.testing.tests import _fit

from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests


# -

def model(
        data: Union[anndata.AnnData, Raw, np.ndarray, scipy.sparse.csr_matrix, glm.typing.InputDataBase],
        formula_loc: Union[None, str] = None,
        formula_scale: Union[None, str] = "~1",
        as_numeric: Union[List[str], Tuple[str], str] = (),
        init_a: Union[np.ndarray, str] = "AUTO",
        init_b: Union[np.ndarray, str] = "AUTO",
        gene_names: Union[np.ndarray, list] = None,
        sample_description: Union[None, pd.DataFrame] = None,
        dmat_loc: Union[patsy.design_info.DesignMatrix] = None,
        dmat_scale: Union[patsy.design_info.DesignMatrix] = None,
        constraints_loc: Union[None, List[str], Tuple[str, str], dict, np.ndarray] = None,
        constraints_scale: Union[None, List[str], Tuple[str, str], dict, np.ndarray] = None,
        noise_model: str = "nb",
        size_factors: Union[np.ndarray, pd.core.series.Series, str] = None,
        batch_size: int = None,
        backend: str = "numpy",
        train_args: dict = {},
        training_strategy: Union[str, List[Dict[str, object]], Callable] = "AUTO",
        quick_scale: bool = False,
        dtype="float64",
        **kwargs
):

    if len(kwargs) != 0:
        logging.getLogger("diffxpy").debug("additional kwargs: %s", str(kwargs))

    if (dmat_loc is None and formula_loc is None) or \
            (dmat_loc is not None and formula_loc is not None):
        raise ValueError("Supply either dmat_loc or formula_loc.")
    if (dmat_scale is None and formula_scale is None) or \
            (dmat_scale is not None and formula_scale != "~1"):
        raise ValueError("Supply either dmat_scale or formula_scale.")

    # # Parse input data formats:
    gene_names = parse_gene_names(data, gene_names)
    if dmat_loc is None and dmat_scale is None:
        sample_description = parse_sample_description(data, sample_description)
    size_factors = parse_size_factors(
        size_factors=size_factors,
        data=data,
        sample_description=sample_description
    )

    design_loc, design_loc_names, constraints_loc, term_names_loc = constraint_system_from_star(
        dmat=dmat_loc,
        sample_description=sample_description,
        formula=formula_loc,
        as_numeric=as_numeric,
        constraints=constraints_loc,
        return_type="patsy"
    )
    design_scale, design_scale_names, constraints_scale, term_names_scale = constraint_system_from_star(
        dmat=dmat_scale,
        sample_description=sample_description,
        formula=formula_scale,
        as_numeric=as_numeric,
        constraints=constraints_scale,
        return_type="patsy"
    )

    model = _fit(
        noise_model=noise_model,
        data=data,
        design_loc=design_loc,
        design_scale=design_scale,
        design_loc_names=design_loc_names,
        design_scale_names=design_scale_names,
        constraints_loc=constraints_loc,
        constraints_scale=constraints_scale,
        init_a=init_a,
        init_b=init_b,
        gene_names=gene_names,
        size_factors=size_factors,
        batch_size=batch_size,
        backend=backend,
        train_args=train_args,
        training_strategy=training_strategy,
        quick_scale=quick_scale,
        dtype=dtype,
        **kwargs,
    )
    return model

def parse_design_testing(
    dmat_loc=None,
    formula_loc=None,
    constraints_loc=None,
    factor_loc_totest=None,
    coef_to_test=None,
    sample_description=None,
    as_numeric=None):
    
    from diffxpy.testing.utils import preview_coef_names,constraint_system_from_star
    
    design_loc, design_loc_names, constraints_loc, term_names_loc = constraint_system_from_star(
        dmat=dmat_loc,
        sample_description=sample_description,
        formula=formula_loc,
        as_numeric=as_numeric,
        constraints=constraints_loc,
        return_type="patsy"
    )
    
    if dmat_loc is not None and factor_loc_totest is not None:
        raise ValueError("Supply coef_to_test and not factor_loc_totest if dmat_loc is supplied.")
    # Define indices of coefficients to test:
    constraints_loc_temp = constraints_loc if constraints_loc is not None else np.eye(design_loc.shape[-1])
    # Check that design_loc is patsy, otherwise  use term_names for slicing.
    if factor_loc_totest is not None:
        if not isinstance(design_loc, patsy.design_info.DesignMatrix):
            col_indices = np.where([
                x in factor_loc_totest
                for x in term_names_loc
            ])[0]
        else:
            # Select coefficients to test via formula model:
            col_indices = np.concatenate([
                np.arange(design_loc.shape[-1])[design_loc.design_info.slice(x)]
                for x in factor_loc_totest
            ])
        assert len(col_indices) > 0, "Could not find any matching columns!"
        if coef_to_test is not None:
            if len(factor_loc_totest) > 1:
                raise ValueError("do not set coef_to_test if more than one factor_loc_totest is given")
            samples = sample_description[factor_loc_totest].astype(type(coef_to_test)) == coef_to_test
            one_cols = np.where(design_loc[samples][:, col_indices][0] == 1)
            if one_cols.size == 0:
                # there is no such column; modify design matrix to create one
                design_loc[:, col_indices] = np.where(samples, 1, 0)
    elif coef_to_test is not None:
        # Directly select coefficients to test from design matrix:
        if sample_description is not None:
            coef_loc_names = preview_coef_names(
                sample_description=sample_description,
                formula=formula_loc,
                as_numeric=as_numeric
            )
        else:
            coef_loc_names = dmat_loc.columns.tolist()
        if not np.all([x in coef_loc_names for x in coef_to_test]):
            raise ValueError(
                "the requested test coefficients %s were found in model coefficients %s" %
                (", ".join([x for x in coef_to_test if x not in coef_loc_names]),
                 ", ".join(coef_loc_names))
            )
        col_indices = np.asarray([
            coef_loc_names.index(x) for x in coef_to_test
        ])
    else:
        raise ValueError("either set factor_loc_totest or coef_to_test")
    # Check that all tested coefficients are independent:
    for x in col_indices:
        if np.sum(constraints_loc_temp[x, :]) != 1:
            raise ValueError("Constraints input is wrong: not all tested coefficients are unconstrained.")
    # Adjust tested coefficients from dependent to independent (fitted) parameters:
    col_indices = np.array([np.where(constraints_loc_temp[x, :] == 1)[0][0] for x in col_indices])
    
    return col_indices


class DEContinousNorm():
    """
    Test DE expression of continous covariate in non-linear setting (splines).
    Using normalised log-transformed expression as input and assuming Normal noise.
    pval - computed with LRT between full model (intercept, 
        covar of interest & splines, other covars) and reduced model (intercept, other covars)
    log2fc - log2 difference between min and max predicted expression value 
        (when non-tested covars are 0); min & max are capped to be > 0
    """
        
    def __init__(self,sample_data:pd.DataFrame,expression:pd.DataFrame,
                 coef_to_test:str,n_splines:int=3,test_method='lrt'):
        """
        :param sample_data: DF with coef_to_test and covariates
        :param expression: DF cells*genes of normalised log scaled expression. 
            Should have the same index as sample_data.
        :param coef_to_test: Which coef from sample data should be used for DE test.
        :param n_splines: N splines to use
        """
        self._sample_data=sample_data.copy()
        self._expression=expression.copy()
        self._coef_to_test=coef_to_test
        self._coef_covar=[col for col in sample_data.columns if col!=coef_to_test] 
        self._n_splines=n_splines
        self.test_method=test_method
        
        # For log2fc set min value to be > 0 as 0 would lead to problems in log. 
        # Also predicted values below 0 do not make sense.
        self._min_above_zero=np.nextafter(0, 1)
        self._models={}
        self._summary_temp=pd.DataFrame()
        self._prediction_interpolated=pd.DataFrame()
    
        # Make splines for design matrix
        spline_basis = np.array(patsy.highlevel.dmatrix(
                    "0+bs(" + self._coef_to_test +\
            ", df=" + str(self._n_splines) + ", degree=3, include_intercept=False) - 1",
                    self._sample_data
                ))
        spline_cols=['spline'+str(i) for i in range(spline_basis.shape[1])]
        self._coefs_to_test=[self._coef_to_test]+spline_cols
        # Design matrix
        self._dmat_loc=pd.DataFrame(np.concatenate(
            [np.array([1]*spline_basis.shape[0]).reshape(-1,1), # Intercept
             self._sample_data,spline_basis], # Coefs and splines
            axis=1), 
            index=self._sample_data.index, # Index should match input data, for test
            columns=['Intercept']+self._sample_data.columns.values.tolist()+spline_cols # Col names
            )
        
        # For predict - make linear range of coef to test values across 
        # coef to test value range to predict each of the points
        interpolated_interval = np.linspace(
            np.min(self._dmat_loc[self._coef_to_test].values),
            np.max(self._dmat_loc[self._coef_to_test].values),
            100
        )
        # Make matching splines across the linear range
        # Stack intercept, interpolated coef to test, and splines
        interpolated_spline_basis = pd.DataFrame(np.hstack([
            np.ones([100, 1]),
            np.expand_dims(interpolated_interval, axis=1),
            patsy.highlevel.dmatrix(
                "0+bs(" + self._coef_to_test + ", df=" + str(self._n_splines) + ")",
                pd.DataFrame({self._coef_to_test: interpolated_interval})
            ).base
        ]),columns=['Intercept',self._coef_to_test]+spline_cols)
        #  Data for prediction: interpolated basis (intercept, coef to test, splines) +
        # covariates set to 0
        self._interpolated_dmat_loc=pd.concat(
            [interpolated_spline_basis,
            pd.DataFrame(
                np.zeros((interpolated_spline_basis.shape[0],len(self._coef_covar))), 
                         columns=self._coef_covar)
             # Make sure that columns are same as in design matrix as else it will 
             # not work properly
            ], axis=1)[self._dmat_loc.columns]
    
    def test(self):
        n_tested=0
        for gene in self._expression.columns:
            # Fit models - full 
            self._models[gene]={}
            self._models[gene]['res_full']=OLS(self._expression[gene], 
                                               exog=self._dmat_loc).fit()
            self._summary_temp.at[gene,'llf']=self._models[gene]['res_full'].llf
            
            if self.test_method=='wald':
                # This currently gives warning:
                # ValueWarning: covariance of constraints does not have full rank.
                raise NotImplementedError('Wald test not implemented')
                # Wald test for all continous componnets
                self._summary_temp.at[gene,'pval']=\
                    self._models[gene]['res_full'].wald_test(
                    ' = '.join(self._coefs_to_test)+' = 0').pvalue
                
            elif self.test_method=='lrt':
                # Reduced model - only intercept and covar
                self._models[gene]['res_red']=OLS(self._expression[gene], 
                            exog=self._dmat_loc[['Intercept']+self._coef_covar]).fit()
                self._summary_temp.at[gene,'llf_red']=self._models[gene]['res_red'].llf

                # LRT test
                self._summary_temp.at[gene,'pval']=\
                    self._models[gene]['res_full'].compare_lr_test(self._models[gene]['res_red'])[1]
            
            # Predicted values over interpolated coef_to_test range
            self._prediction_interpolated[gene]=\
                self._models[gene]['res_full'].predict(self._interpolated_dmat_loc)
            
            # log2fc
            # TODO correct this currently dioes not work well 
            # 1.) Problem since expression values here can be negative
            # 2.) The addition of small value to for replacing 0 leads to large lfc
            # Diff between min and max predictions across all cells, 
            # use prediction without covar
            # Log2fc by first doing log and then diff as else may get inf
            # Assumes that: minimal expression value >= a small positive non-zero float
            #self._summary_temp.at[gene,'log2fc']=\
            #    np.log2(max([self._prediction_interpolated[gene].max(),self._min_above_zero]))-\
            #    np.log2(max([self._prediction_interpolated[gene].min(),self._min_above_zero]))
            
            # Instead of lFC compute mean, std, and mean/std on raw data
            # This is useful as expression is assumed to follow normal distn anyways
            #self._summary_temp.at[gene,'mean']=self._expression[gene].mean()
            #self._summary_temp.at[gene,'std']=self._expression[gene].std()
            #self._summary_temp.at[gene,'std_relative']=\
            #    self._summary_temp.at[gene,'std']/self._summary_temp.at[gene,'mean']
            
            # Reposr progress
            n_tested+=1
            if n_tested%100==0:
                print('Tested %i/%i genes'%(n_tested,self._expression.shape[1]))
    
    def summary(self):
        summary=self._summary_temp.copy()
        # Calculate padj of currently computed genes
        summary['padj']=multipletests(summary['pval'],method='fdr_bh')[1]
        return summary
    
    def predict_interpolated(self):
        return self._prediction_interpolated.copy()
