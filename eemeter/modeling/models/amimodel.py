import warnings

from itertools import chain

import numpy as np
import pandas as pd
import patsy
import pymc
import sklearn.metrics as skm

from amimodels.normal_hmm import (make_normal_hmm, trace_sampler,
                                  get_stochs_excluding)
from amimodels.step_methods import (TransProbMatStep, HMMStatesStep,
                                    NormalNormalStep, GammaNormalStep,
                                    PriorObsSampler)


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

    def __init__(self, cooling_base_temp,
                 heating_base_temp, mcmc_samples=1000):

        self.cooling_base_temp = cooling_base_temp
        self.heating_base_temp = heating_base_temp

        self.model_freq = pd.tseries.frequencies.Day()
        self.base_reg_formula = 'energy ~ 1 + CDD + HDD + CDD:HDD'
        self.mcmc_samples = mcmc_samples
        self.params = None
        self.X_matrices = None
        self.y = None
        self.estimated = None
        self.r2 = None
        self.rmse = None
        self.cvrmse = None

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

              - :code:`initial_values`: initial values for the model.
                Mostly consists of hyper-parameter values and initial
                stochastic values.
              - :code:`X_design_matrix`: patsy design matrix used in
                formatting design matrix.
              - :code:`stoch_traces`: dict of numpy arrays for each
                stochastic needed to sample new predictive values
                (i.e. needed to sample the observation mean, ``mu``,
                pymc object).

            - :code:`"rmse"`: Root mean square error
            - :code:`"cvrmse"`: Normalized root mean square error
              (Coefficient of variation of root mean square error).
            - :code:`"upper"`: self.y_upper, the upper %95 quantile of
                the observation variable's posterior samples.
            - :code:`"lower"`: self.y_lower, the lower %95 quantile of
                the observation variable's posterior samples.
        '''
        # convert to daily
        model_data = input_data.resample(self.model_freq).agg(
                {'energy': np.sum, 'tempF': np.mean})

        #model_data = model_data.dropna()

        if model_data.dropna().empty:
            raise ValueError("No model data (consumption + weather)")

        model_data = model_data.assign(CDD=np.fmax(model_data.tempF -
                                                   self.cooling_base_temp, 0.),
                                       HDD=np.fmax(self.heating_base_temp -
                                                   model_data.tempF, 0.))

        regression_formula = self.base_reg_formula

        # Make sure these factors have enough levels to not
        # cause issues.
        if len(np.unique(model_data.index.month)) >= 12:
            regression_formula += ' + cc(tempF.index.month, df=12) '

        #if len(np.unique(model_data.index.weekday)) >= 7:
        #    regression_formula += ' + cc(tempF.index.weekday, df=12) '

        # Single constant state and regression state.
        formulas = ["energy ~ 1", regression_formula]
        X_matrices = []
        for formula in formulas:
            y, X = patsy.dmatrices(formula, model_data,
                                   return_type='dataframe')
            X_matrices += [X]

        init_params = None  # gmm_norm_hmm_init_params(y, X_matrices)

        norm_hmm = make_normal_hmm(y, X_matrices, init_params,
                                   include_ppy=True)

        # Some very basic, generic initializations for the state parameters.
        norm_hmm.betas[0].value = 0
        b1_0 = norm_hmm.betas[1].value.copy()
        b1_0[0] = float(y.mean())
        norm_hmm.betas[1].value = b1_0

        mcmc_step = pymc.MCMC(norm_hmm.variables)

        mcmc_step.use_step_method(HMMStatesStep, norm_hmm.states)
        mcmc_step.use_step_method(TransProbMatStep, norm_hmm.trans_mat)

        for b_ in norm_hmm.betas:
            mcmc_step.use_step_method(NormalNormalStep, b_)

        for V_ in norm_hmm.V_invs:
            mcmc_step.use_step_method(GammaNormalStep, V_)

        for e_ in chain(norm_hmm.etas, norm_hmm.lambdas):
            mcmc_step.use_step_method(pymc.StepMethods.Metropolis,
                                      e_, proposal_distribution='Prior')

        mcmc_step.use_step_method(PriorObsSampler, norm_hmm.y_pp_rv)

        mcmc_step.sample(self.mcmc_samples, burn=int(self.mcmc_samples/4.))

        mu_samples = pd.DataFrame(norm_hmm.mu.trace().T, index=y.index)

        estimated_stats = norm_hmm.mu.stats()

        self.X_matrices = X_matrices
        self.y = y

        #abs_err_samples = np.abs(mu_samples.sub(y.squeeze(), axis=0)).mean(axis=1).mean()
        r2_samples = mu_samples.apply(lambda x: skm.r2_score(y, x), axis=0)
        r2 = r2_samples.mean()

        estimated = pd.Series(estimated_stats['quantiles'][50.0],
                              index=y.index, name='estimated')

        rmse = ((y.squeeze() - estimated.squeeze())**2).mean()**.5

        # XXX: Reindex last; otherwise, some indices might not match.
        self.estimated = estimated.reindex(model_data.index)

        # These are the bounds for the observation mean:
        self.mu_lower = pd.DataFrame(estimated_stats['quantiles'][2.5],
                                     index=y.index).reindex(model_data.index)
        self.mu_upper = pd.DataFrame(estimated_stats['quantiles'][97.5],
                                     index=y.index).reindex(model_data.index)

        # These are the bounds for the observations:
        pp_stats = norm_hmm.y_pp_rv.stats()
        self.lower = pd.DataFrame(pp_stats['quantiles'][2.5],
                                  index=y.index).reindex(model_data.index)
        self.upper = pd.DataFrame(pp_stats['quantiles'][97.5],
                                  index=y.index).reindex(model_data.index)

        y_mean = y.values.ravel().mean()
        if y_mean != 0:
            cvrmse = rmse / float(y_mean)
        else:
            cvrmse = np.nan

        self.r2 = r2
        self.rmse = rmse
        self.cvrmse = cvrmse

        self.plot()

        # Collect all the traces needed for 'mu'.
        non_time_parents = get_stochs_excluding(norm_hmm.mu,
                                                set(('states', 'N_obs')))
        stoch_traces = {}
        for stoch in non_time_parents:
            stoch_traces[stoch.__name__] = stoch.trace()

        self.params = {
            "init_params": init_params,
            "X_design_infos": [X_.design_info for X_ in X_matrices],
            "stoch_traces": stoch_traces,
        }

        output = {
            "r2": self.r2,
            "model_params": self.params,
            "rmse": self.rmse,
            "cvrmse": self.cvrmse,
            "upper": self.upper,
            "lower": self.lower,
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

        model_data = model_data.assign(CDD=np.fmax(model_data.tempF -
                                                   self.cooling_base_temp, 0.),
                                       HDD=np.fmax(self.heating_base_temp -
                                                   model_data.tempF, 0.))

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

        ram_db = trace_sampler(norm_hmm, 'mu', self.params['stoch_traces'])

        mu_samples = pd.DataFrame(ram_db.trace('mu').gettrace().T,
                                  index=X_matrices[0].index)

        estimated = mu_samples.mean(axis=1)

        predicted = pd.Series(estimated, index=X_matrices[0].index,
                              name='predicted')
        predicted = predicted.reindex(model_data.index)

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
                         self.lower.squeeze(),
                         self.upper.squeeze(),
                         color='b', alpha=0.3,
                         where=np.isfinite(self.estimated.values.squeeze()),
                         label=r'y posterior %95')
        plt.fill_between(self.estimated.index.to_datetime(),
                         self.mu_lower.squeeze(),
                         self.mu_upper.squeeze(),
                         color='r', alpha=0.3,
                         where=np.isfinite(self.estimated.values.squeeze()),
                         label=r'mu posterior %95')
        y_plot = self.y.squeeze().reindex(self.estimated.index)
        y_plot.plot(color='k', linewidth=1.5)

        plt.legend()
        plt.show()