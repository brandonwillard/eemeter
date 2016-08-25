import warnings

import numpy as np
import pandas as pd
import patsy
import pymc
import sklearn.metrics as skm

from amimodels.normal_hmm import (gmm_norm_hmm_init_params, make_normal_hmm)
from amimodels.step_methods import (TransProbMatStep, HMMStatesStep,
                                    NormalNormalStep)


class NormalHMMModel(object):
    ''' Hidden Markov Model using daily frequency data to build a model of
    formatted energy trace data that takes into account HDD, CDD, day of week,
    month factors.

    Parameters
    ----------
    cooling_base_temp : float
        Base temperature (degrees F) used in calculating cooling degree days.
    heating_base_temp : float
        Base temperature (degrees F) used in calculating heating degree days.
    '''

    def __init__(self, cooling_base_temp, heating_base_temp):

        self.cooling_base_temp = cooling_base_temp
        self.heating_base_temp = heating_base_temp

        self.model_freq = pd.tseries.frequencies.Day()
        self.base_reg_formula = 'energy ~ 1 + CDD + HDD + CDD:HDD'
        self.mcmc_samples = 100
        self.params = None
        self.X_matrices = None
        self.y = None
        self.estimated = None
        self.r2 = None
        self.rmse = None
        self.cvrmse = None
        self.n = None

    def __repr__(self):
        return (
            'NormalHMMModel(cooling_base_temp={},'
            ' heating_base_temp={})'
            .format(self.cooling_base_temp, self.heating_base_temp)
        )

    def fit(self, input_data):
        ''' Fits a model to the input data.

        Parameters
        ----------
        input_data : pandas.DataFrame
            Formatted input data as returned by
            :code:`ModelDataFormatter.create_input()`

        Returns
        -------
        out : dict
            Results of this model fit:

            - :code:`"r2"`: R-squared value from this fit.
            - :code:`"model_params"`: Fitted parameters.

              - :code:`X_design_matrix`: patsy design matrix used in
                formatting design matrix.
              - :code:`formula`: patsy formula used in creating design matrix.
              - :code:`coefficients`: ElasticNetCV coefficients.
              - :code:`intercept`: ElasticNetCV intercept.

            - :code:`"rmse"`: Root mean square error
            - :code:`"cvrmse"`: Normalized root mean square error
              (Coefficient of variation of root mean square error).
            - :code:`"upper"`: self.upper,
            - :code:`"lower"`: self.lower,
            - :code:`"n"`: self.n
        '''
        # convert to daily
        model_data = input_data.resample(self.model_freq).agg(
                {'energy': np.sum, 'tempF': np.mean})

        #model_data = model_data.dropna()

        if model_data.empty:
            raise ValueError("No model data (consumption + weather)")

        model_data.loc[:, 'CDD'] = np.fmax(model_data.tempF -
                                           self.cooling_base_temp, 0.)
        model_data.loc[:, 'HDD'] = np.fmax(self.heating_base_temp -
                                           model_data.tempF, 0.)

        regression_formula = self.base_reg_formula

        # Make sure these factors have enough levels to not
        # cause issues.
        if len(np.unique(model_data.index.month)) >= 12:
            regression_formula += '''\
            + CDD * C(tempF.index.month) \
            + HDD * C(tempF.index.month) \
            + C(tempF.index.month) \
            '''

        if len(np.unique(model_data.index.weekday)) >= 7:
            regression_formula += '''\
            + (CDD) * C(tempF.index.weekday) \
            + (HDD) * C(tempF.index.weekday) \
            + C(tempF.index.weekday)\
            '''

        formulas = ["energy ~ 1", regression_formula]
        X_matrices = []
        for formula in formulas:
            y, X = patsy.dmatrices(formula, model_data,
                                   return_type='dataframe')
            X_matrices += [X]

        init_params = gmm_norm_hmm_init_params(y, X_matrices)
        norm_hmm = make_normal_hmm(y, X_matrices, init_params)

        mcmc_step = pymc.MCMC(norm_hmm.variables)

        mcmc_step.use_step_method(HMMStatesStep, norm_hmm.states)
        mcmc_step.use_step_method(TransProbMatStep, norm_hmm.trans_mat)
        for b_ in norm_hmm.betas:
            mcmc_step.use_step_method(NormalNormalStep, b_)

        mcmc_step.sample(self.mcmc_samples)

        mu_samples = pd.DataFrame(mcmc_step.trace('mu')[:].T,
                                  index=model_data.tempF.index)

        estimated = mu_samples.mean(axis=1)

        self.X_matrices = X_matrices
        self.y = y
        self.estimated = estimated

        hmm_r2_samples = mu_samples.apply(lambda x: skm.r2_score(y, x),
                                          axis=0)
        r2 = hmm_r2_samples.mean()

        rmse = ((y.values.ravel() - estimated)**2).mean()**.5

        if y.mean != 0:
            cvrmse = rmse / float(y.values.ravel().mean())
        else:
            cvrmse = np.nan

        self.r2 = r2
        self.rmse = rmse
        self.cvrmse = cvrmse

        n = self.estimated.shape[0]

        # or we could use ...['quantiles']
        self.lower, self.upper = mcmc_step.stats('mu')['mu']['95% HPD interval']
        self.n = n

        self.plot()


        # TODO: These parameters don't apply anymore; what should
        # they be/do?
        last_samples = {v.__name__: v.trace()[-1]
                        for v in mcmc_step.stochastics}
        self.params = {
            # We're dealing with samples now; use those, or their means?
            # We'll use the last sample for each stochastic for now.
            "init_params": init_params,
            "last_samples": last_samples,
            "X_design_infos": [X_.design_info for X_ in X_matrices],
        }

        output = {
            "r2": self.r2,
            "model_params": self.params,
            "rmse": self.rmse,
            "cvrmse": self.cvrmse,
            "upper": self.upper,
            "lower": self.lower,
            "n": self.n
        }
        return output

    def predict(self, demand_fixture_data, params=None):
        ''' Predicts across index using fitted model params

        Parameters
        ----------
        demand_fixture_data : pandas.DataFrame
            Formatted input data as returned by
            :code:`ModelDataFormatter.create_demand_fixture()`
        params : dict, default None
            Parameters found during model fit. If None, `.fit()` must be called
            before this method can be used.

        Returns
        -------
        output : pandas.DataFrame
            Dataframe of energy values as given by the fitted model across the
            index given in :code:`demand_fixture_data`.
        '''
        # needs only tempF
        if params is None:
            # TODO: Use/check this object for stored fit results.
            # Otherwise, error!
            params = self.params

            if params is None:
                raise ValueError("No stored or passed fit params")

        model_data = demand_fixture_data.resample(self.model_freq).agg(
                {'tempF': np.mean})

        model_data.loc[:, 'CDD'] = np.fmax(model_data.tempF -
                                           self.cooling_base_temp, 0.)
        model_data.loc[:, 'HDD'] = np.fmax(self.heating_base_temp -
                                           model_data.tempF, 0.)

        X_matrices = []
        for design_info in self.params['X_design_infos']:
            (X,) = patsy.build_design_matrices([design_info],
                                               model_data,
                                               return_type='dataframe')
            X_matrices += [X]

        # TODO FIXME: Setup initial values for stochastics from
        # previous sample results...
        init_params = self.params['init_params']

        # Creating the model with None observations should
        # give a (non-observed) stochastic for the observations
        # variable, 'y_rv'.  We can use that for posterior predictive
        # samples.
        norm_hmm = make_normal_hmm(None, X_matrices, init_params)

        mcmc_step = pymc.MCMC(norm_hmm.variables)

        # Set the stochastic values to their previous sampled values.
        # TODO: Do this a better way.
        last_samples = self.params['last_samples']
        for stoch in mcmc_step.stochastics:
            last_value = last_samples.get(stoch.__name__, None)
            if last_value is not None:
                stoch.value = last_value

        mcmc_step.assign_step_methods()

        mcmc_step.sample(self.mcmc_samples)

        # We're not really using posterior predictive values.
        # Use them if you want error estimates (e.g. form HPD region
        # or quantiles of samples from 'y_rv' variable).
        # More specifically,
        #   predict_low, predict_high = mcmc_step.stats('y_rv')['y_rv']['95% HPD interval']
        mu_samples = pd.DataFrame(mcmc_step.trace('mu')[:].T,
                                  index=model_data.tempF.index)

        estimated = mu_samples.mean(axis=1)

        predicted = pd.Series(estimated, index=model_data.index)

        return predicted

    def plot(self):
        ''' Plots fit against input data. Should not be run before the
        :code:`.fit(` method.
        '''

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("Cannot plot - no matplotlib.")
            return None

        plt.title("actual v. estimated w/ 95% confidence")

        self.estimated.plot(color='b', alpha=0.7)

        plt.fill_between(self.estimated.index.to_datetime(),
                         self.estimated + self.upper,
                         self.estimated - self.lower,
                         color='b', alpha=0.3)

        pd.Series(self.y.values.ravel(), index=self.estimated.index).plot(
                  color='k', linewidth=1.5)

        plt.show()
