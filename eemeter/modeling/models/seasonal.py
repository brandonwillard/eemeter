import warnings

import holidays
import numpy as np
import pandas as pd
import patsy
from scipy.stats import chi2
from sklearn import linear_model


class SeasonalElasticNetCVModel(object):
    ''' Linear regression using daily frequency data to build a model of
    formatted energy trace data that takes into account HDD, CDD, day of week,
    month, and holiday effects, with elastic net regularization.

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
        self.base_formula = 'energy ~ 1 + CDD + HDD + CDD:HDD'
        self.l1_ratio = 0.5
        self.holidays = holidays.UnitedStates()
        self.params = None
        self.X = None
        self.y = None
        self.estimated = None
        self.r2 = None
        self.rmse = None
        self.cvrmse = None
        self.n = None

    def __repr__(self):
        return (
            'SeasonalElasticNetCVModel(cooling_base_temp={},'
            ' heating_base_temp={})'
            .format(self.cooling_base_temp, self.heating_base_temp)
        )

    def _holidays_indexed(self, dt_index):
        def clean_holiday_name(dt):
            raw_name = self.holidays.get(dt, "none")
            if raw_name.endswith(" (Observed)"):
                return raw_name[:-11]
            else:
                return raw_name

        holiday_names = pd.Series(dt_index.map(clean_holiday_name),
                                  index=dt_index)
        return holiday_names

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

        model_data = model_data.dropna()

        if model_data.empty:
            raise ValueError("No model data (consumption + weather)")

        model_data.loc[:, 'CDD'] = np.maximum(model_data.tempF -
                                              self.cooling_base_temp, 0.)
        model_data.loc[:, 'HDD'] = np.maximum(self.heating_base_temp -
                                              model_data.tempF, 0.)

        formula = self.base_formula

        # Make sure these factors have enough levels to not
        # cause issues.
        if len(np.unique(model_data.index.month)) >= 12:
            formula += '''\
            + CDD * C(tempF.index.month) \
            + HDD * C(tempF.index.month) \
            + C(tempF.index.month) \
            '''

        if len(np.unique(model_data.index.weekday)) >= 7:
            formula += '''\
            + (CDD) * C(tempF.index.weekday) \
            + (HDD) * C(tempF.index.weekday) \
            + C(tempF.index.weekday)\
            '''

        holiday_names = self._holidays_indexed(model_data.index)

        if len(np.unique(holiday_names)) == 11:
            model_data.loc[:, 'holiday_name'] = holiday_names
            formula += " + C(holiday_name)"

        y, X = patsy.dmatrices(formula, model_data, return_type='dataframe')

        model_obj = linear_model.ElasticNetCV(l1_ratio=self.l1_ratio,
                                              fit_intercept=False)
        model_obj.fit(X, y.values.ravel())

        estimated = pd.Series(model_obj.predict(X),
                              index=model_data.tempF.index)

        self.X = X
        self.y = y
        self.estimated = estimated

        r2 = model_obj.score(X, y)
        rmse = ((y.values.ravel() - estimated)**2).mean()**.5

        if y.mean != 0:
            cvrmse = rmse / float(y.values.ravel().mean())
        else:
            cvrmse = np.nan

        self.r2 = r2
        self.rmse = rmse
        self.cvrmse = cvrmse

        # For justification of these 95% confidence intervals, based on rmse,
        # see http://stats.stackexchange.com/questions/78079/
        #     confidence-interval-of-rmse
        #
        # > Let xi be your true value for the ith data point and xhat_i the
        # >   estimated value.
        # > If we assume that the differences between the estimated and
        # > true values have
        # >
        # > 1. mean zero (i.e. the xhat_i are distributed around xi)
        # > 2. follow a Normal distribution
        # > 3. and all have the same standard deviation sigma
        # > then you really want a confidence interval for sigma
        # > ...
        #
        # We might decide these assumptions don't hold.

        n = self.estimated.shape[0]

        c1, c2 = chi2.ppf([0.025, 1-0.025], n)
        self.lower = np.sqrt(n/c2) * self.rmse
        self.upper = np.sqrt(n/c1) * self.rmse
        self.n = n

        self.plot()

        self.params = {
            "coefficients": model_obj.coef_,
            "intercept": model_obj.intercept_,
            "X_design_info": X.design_info,
            "formula": formula,
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

              - :code:`X_design_matrix`: patsy design matrix used in
                formatting design matrix.
              - :code:`formula`: patsy formula used in creating design matrix.
              - :code:`coefficients`: ElasticNetCV coefficients.
              - :code:`intercept`: ElasticNetCV intercept.

        Returns
        -------
        output : pandas.DataFrame
            Dataframe of energy values as given by the fitted model across the
            index given in :code:`demand_fixture_data`.
        '''
        # needs only tempF
        if params is None:
            params = self.params

        model_data = demand_fixture_data.resample(self.model_freq).agg(
                {'tempF': np.mean})

        model_data.loc[:, 'CDD'] = np.maximum(model_data.tempF -
                                              self.cooling_base_temp, 0.)
        model_data.loc[:, 'HDD'] = np.maximum(self.heating_base_temp -
                                              model_data.tempF, 0.)

        holiday_names = self._holidays_indexed(model_data.index)

        model_data.loc[:, 'holiday_name'] = holiday_names

        design_info = params["X_design_info"]

        (X,) = patsy.build_design_matrices([design_info],
                                           model_data,
                                           return_type='dataframe')

        model_obj = linear_model.ElasticNetCV(l1_ratio=self.l1_ratio,
                                              fit_intercept=False)

        model_obj.coef_ = params["coefficients"]
        model_obj.intercept_ = params["intercept"]

        predicted = pd.Series(model_obj.predict(X), index=X.index)

        # add NaNs back in
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
                         self.estimated + self.upper,
                         self.estimated - self.lower,
                         color='b', alpha=0.3)

        pd.Series(self.y.values.ravel(), index=self.estimated.index).plot(
                  color='k', linewidth=1.5)

        plt.show()
