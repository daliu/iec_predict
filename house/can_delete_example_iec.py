"""  Intelligent energy component contains the IEC class, that includes several algorithms
     for predicting consumption of a house, given historical data. It also contains an IECTester
     class that can be used to test and provide results on multiple IEC runs """

from __future__ import division

import pickle
from datetime import timedelta
import datetime
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import GPy
import statsmodels.api as sm
import pandas as pd
import scipy.ndimage.filters
import scipy.signal
from scipy import spatial
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import time 
from sklearn.preprocessing import MinMaxScaler


import matplotlib.pyplot as plt

try:
    from easing import *
except:
    from easing import *

cons_col = 'House Consumption'



def convert_to_timestamp(x):
    """Convert date objects to integers"""
    return time.mktime(x.to_datetime().timetuple())

def normalize_dates(df,col):
    """Normalize the DF using min/max"""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_values= df[col].values.reshape(-1,1)
    dates_scaled = scaler.fit_transform(df_values)
    return pd.DataFrame(dates_scaled)


class NoSimilarMomentsFound(Exception):
    pass


def cosine_similarity(a, b):
    """Calculate the cosine similarity between
    two non-zero vectors of equal length (https://en.wikipedia.org/wiki/Cosine_similarity)
    """
    return 1.0 - spatial.distance.cosine(a, b)


def baseline_similarity(a, b, filter=True):
    if filter is True:
        similarity = -mean_squared_error(gauss_filt(a, 201), gauss_filt(b, 201)) ** 0.5
    else:
        similarity = -mean_squared_error(a, b) ** 0.5
    return similarity


def advanced_similarity(a, b):
    sigma = 10

    base_similarity = baseline_similarity(a, b)

    high_pass_a = highpass_filter(a)
    high_pass_b = highpass_filter(b)

    high_pass_a = scipy.ndimage.filters.gaussian_filter1d(high_pass_a, sigma)
    high_pass_b = scipy.ndimage.filters.gaussian_filter1d(high_pass_b, sigma)

    highpass_similarity = -mean_squared_error(high_pass_a, high_pass_b)

    return base_similarity + highpass_similarity


def mins_in_day(timestamp):
    return timestamp.hour * 60 + timestamp.minute


def find_similar_days(training_data, observation_length, k, interval, method=cosine_similarity):
    now = training_data.index[-1]
    timezone = training_data.index.tz

    # Find moments in our dataset that have the same hour/minute and is_weekend() == weekend.
    # Those are the indexes of those moments in TrainingData

    min_time = training_data.index[0] + timedelta(minutes=observation_length)

    selector = (
        (training_data.index.minute == now.minute) &
        (training_data.index.hour == now.hour) &
        (training_data.index > min_time)
    )
    similar_moments = training_data[selector][:-1].tz_convert('UTC')

    if similar_moments.empty:
        raise NoSimilarMomentsFound

    training_data = training_data.tz_convert('UTC')

    last_day_vector = (training_data
                       .tail(observation_length)
                       .resample(timedelta(minutes=interval))
                       .sum()
                       )

    obs_td = timedelta(minutes=observation_length)

    similar_moments['Similarity'] = [
        method(
            last_day_vector.as_matrix(columns=[cons_col]),
            training_data[i - obs_td:i].resample(timedelta(minutes=interval)).sum().as_matrix(columns=[cons_col])
        ) for i in similar_moments.index
    ]

    indexes = (similar_moments
               .sort_values('Similarity', ascending=False)
               .head(k)
               .index
               .tz_convert(timezone))

    return indexes


def lerp(x, y, alpha):
    assert x.shape == y.shape and x.shape == alpha.shape  # shapes must be equal

    x *= 1 - alpha
    y *= alpha

    return x + y


def med_filt(x, k=201):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    if x.ndim > 1:
        x = np.squeeze(x)
    med = np.median(x)
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = med
    return np.median(y, axis=1)

def gauss_filt(x, k=201):
    """Apply a length-k gaussian filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    if x.ndim > 1:
        x = np.squeeze(x)
    med = np.median(x)
    assert k % 2 == 1, "mean filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = med
    return np.mean(y, axis=1)


def calc_baseline(training_data, similar_moments,
                  prediction_window, half_window=100, method=gauss_filt, interp_range=200):
    prediction_window_in_mins = prediction_window
    if type(prediction_window) is not timedelta:
        prediction_window = timedelta(minutes=prediction_window)

    k = len(similar_moments)

    r = np.zeros((prediction_window_in_mins + 1, 1))
    for i in similar_moments:
        r += (1 / k) * training_data[i:i + prediction_window].rolling(window=half_window * 2, center=True,
                                                                      min_periods=1).mean().as_matrix()
    baseline = np.squeeze(r)

    recent_baseline = training_data[-2 * half_window:-1].mean()[cons_col]

    if interp_range > 0:
        baseline[:interp_range] = lerp(np.repeat(recent_baseline, interp_range),
                                       baseline[:interp_range],
                                       np.arange(interp_range) / interp_range)

    return baseline


def calc_baseline_dumb(training_data, similar_moments,
                       prediction_window):
    if type(prediction_window) is not timedelta:
        prediction_window = timedelta(minutes=prediction_window)

    k = len(similar_moments)

    r = np.zeros((49, 1))
    for i in similar_moments:
        similar_day = (1 / k) * training_data[i:i + prediction_window].resample(timedelta(minutes=15)).mean()
        similar_day = similar_day[0:49]
        r += similar_day
        # r += (1 / k) * training_data[i:i + prediction_window].as_matrix

    baseline = np.squeeze(r)

    b = pd.DataFrame(baseline).set_index(pd.TimedeltaIndex(freq='15T', start=0, periods=49)).resample(
        timedelta(minutes=1)).ffill()
    baseline = np.squeeze(b.as_matrix())
    baseline = np.concatenate((baseline, np.atleast_1d(baseline[-1])))

    return baseline


def highpass_filter(a):
    cutoff = 2

    baseline = gauss_filt(a)
    highpass = a - baseline
    highpass[highpass < baseline * cutoff] = 0

    return highpass


def calc_highpass(training_data, similar_moments,
                  prediction_window, half_window, method=gauss_filt):
    if type(prediction_window) is not timedelta:
        prediction_window = timedelta(minutes=prediction_window)

    k = len(similar_moments)

    similar_data = np.zeros((k, int(prediction_window.total_seconds() / 60) + 2 * half_window))

    for i in range(k):
        similar_data[i] = training_data[
                          similar_moments[i] - timedelta(minutes=half_window)
                          : similar_moments[i] + prediction_window + timedelta(minutes=half_window),
                          2
                          ]

    highpass = np.apply_along_axis(highpass_filter, 1, similar_data)

    highpass = highpass[:, half_window: -half_window]

    w = 3
    confidence_threshold = 0.5

    paded_highpass = np.pad(highpass, ((0,), (w,)), mode='edge')

    highpass_prediction = np.zeros(prediction_window)

    for i in range(w, prediction_window + w):
        window = paded_highpass[:, i - w:i + w]
        confidence = np.count_nonzero(window) / window.size

        if confidence > confidence_threshold:
            highpass_prediction[
                i - w] = np.mean(window[np.nonzero(window)]) * confidence

    return highpass_prediction


class IEC(object):
    """The Intelligent Energy Component of a house.
    IEC will use several methods to predict the energy consumption of a house
    for a given prediction window using historical data.
    """

    def __init__(self, data, prediction_window=16 * 60):
        """Initializing the IEC.

        Args:
            :param data: Historical Dataset. Last value must be current time
        """
        self.data = data
        self.now = data.index[-1]
        self.prediction_window = prediction_window
        self.algorithms = {
            "Simple Mean": self.simple_mean,
            "Usage Zone Finder": self.usage_zone_finder,
            "ARIMA": self.ARIMAforecast,
            "Baseline Finder": partial(self.baseline_finder, training_window=1440 * 60, k=9, long_interp_range=250,
                                       short_interp_range=25, half_window=70, similarity_interval=5,
                                       recent_baseline_length=250,
                                       observation_length_addition=240, short_term_ease_method=easeOutSine,
                                       long_term_ease_method=easeOutCirc),
            "STLF": self.baseline_finder_dumb,
            "b1": partial(self.baseline_finder, training_window=1440 * 60, k=9, long_interp_range=250,
                          short_interp_range=25, half_window=70, similarity_interval=5, recent_baseline_length=250,
                          observation_length_addition=240, short_term_ease_method=easeOutSine,
                          long_term_ease_method=easeOutCirc),
            "b2": partial(self.baseline_finder, training_window=1440 * 60, k=3, long_interp_range=250,
                          short_interp_range=25, half_window=70, similarity_interval=5, recent_baseline_length=300,
                          observation_length_addition=240, short_term_ease_method=easeOutSine,
                          long_term_ease_method=easeOutCirc),
            "b3": partial(self.baseline_finder, training_window=1440 * 60, k=6, long_interp_range=250,
                          short_interp_range=25, half_window=70, similarity_interval=5, recent_baseline_length=200,
                          observation_length_addition=240, short_term_ease_method=easeOutSine,
                          long_term_ease_method=easeOutCirc),
            "b4": partial(self.baseline_finder, training_window=1440 * 60, k=12, long_interp_range=250,
                          short_interp_range=25, half_window=70, similarity_interval=5, recent_baseline_length=200,
                          observation_length_addition=240, short_term_ease_method=easeOutSine,
                          long_term_ease_method=easeOutCirc),
            "b5": partial(self.baseline_finder, training_window=1440 * 60, k=9, long_interp_range=250,
                          short_interp_range=25, half_window=50, similarity_interval=5, recent_baseline_length=250,
                          observation_length_addition=240, short_term_ease_method=easeOutSine,
                          long_term_ease_method=easeOutCirc),
            "b6": partial(self.baseline_finder, training_window=1440 * 60, k=9, long_interp_range=250,
                          short_interp_range=25, half_window=60, similarity_interval=5, recent_baseline_length=300,
                          observation_length_addition=240, short_term_ease_method=easeOutSine,
                          long_term_ease_method=easeOutCirc),
            "b7": partial(self.baseline_finder, training_window=1440 * 60, k=9, long_interp_range=250,
                          short_interp_range=25, half_window=80, similarity_interval=5, recent_baseline_length=200,
                          observation_length_addition=240, short_term_ease_method=easeOutSine,
                          long_term_ease_method=easeOutCirc),
            "GP_periodic": partial(self.gaussian_process, training_window=1440*60, k=7, kernel = "PeriodicExponential")
        }


    def gaussian_process(self, training_window=1440 * 60, k=7, kernel='PeriodicExponential'):

            kernel = GPy.kern.PeriodicExponential(1)

            # TODO: Test Kernels
            if kernel is 'Linear':
                kernel = GPy.kern.Linear(1)
            if kernel is 'LinearFull':
                kernel = GPy.kern.LinearFull(1)
            if kernel is 'LinearSlopeBasisFuncKernel':
                kernel = GPy.kern.LinearSlopeBasisFuncKernel(1)
            if kernel is 'LogisticBasisFuncKernel':
                kernel = GPy.kern.LogisticBasisFuncKernel(1)

            # Untested Kernels
            if kernel is 'BasisFuncKernel':
                kernel = GPy.kern.BasisFuncKernel(1)
            if kernel is 'Bias':
                kernel = GPy.kern.Bias(1)
            if kernel is 'Brownian':
                kernel = GPy.kern.Brownian(1)
            if kernel is 'ChangePointBasisFuncKernel':
                kernel = GPy.kern.ChangePointBasisFuncKernel(1)
            if kernel is 'Coregionalize':
                kernel = GPy.kern.Coregionalize(1)
            if kernel is 'Cosine':
                kernel = GPy.kern.Cosine(1)
            if kernel is 'DEtime':
                kernel = GPy.kern.DEtime(1)
            if kernel is 'DiffGenomeKern':
                kernel = GPy.kern.DiffGenomeKern(1)
            if kernel is 'DomainKernel':
                kernel = GPy.kern.DomainKernel(1)
            if kernel is 'EQ_ODE1':
                kernel = GPy.kern.EQ_ODE1(1)
            if kernel is 'EQ_ODE2':
                kernel = GPy.kern.EQ_ODE2(1)
            if kernel is 'ExpQuad':
                kernel = GPy.kern.ExpQuad(1)
            if kernel is 'Exponential':
                kernel = GPy.kern.Exponential(1)
            if kernel is 'Fixed':
                kernel = GPy.kern.Fixed(1)
            if kernel is 'GridRBF':
                kernel = GPy.kern.GridRBF(1)
            if kernel is 'Hierarchical':
                kernel = GPy.kern.Hierarchical(1)
            if kernel is 'IndependentOutputs':
                kernel = GPy.kern.IndependentOutputs(1)
            if kernel is 'Integral':
                kernel = GPy.kern.Integral(1)
            if kernel is 'Integral_Limits':
                kernel = GPy.kern.Integral_Limits(1)
            if kernel is 'Kern':
                kernel = GPy.kern.Kern(1)

            if kernel is 'MLP':
                kernel = GPy.kern.MLP(1)
            if kernel is 'Matern32':
                kernel = GPy.kern.Matern32(1)
            if kernel is 'Matern52':
                kernel = GPy.kern.Matern52(1)
            if kernel is 'Multidimensional_Integral_Limits':
                kernel = GPy.kern.Multidimensional_Integral_Limits(1)
            if kernel is 'MultioutputKern':
                kernel = GPy.kern.MultioutputKern(1)
            if kernel is 'ODE_UY':
                kernel = GPy.kern.ODE_UY(1)
            if kernel is 'ODE_UYC':
                kernel = GPy.kern.ODE_UYC(1)
            if kernel is 'ODE_st':
                kernel = GPy.kern.ODE_st(1)
            if kernel is 'ODE_t':
                kernel = GPy.kern.ODE_t(1)
            if kernel is 'OU':
                kernel = GPy.kern.OU(1)
            if kernel is 'PeriodicExponential':
                kernel = GPy.kern.PeriodicExponential(1)
            if kernel is 'PeriodicMatern32':
                kernel = GPy.kern.PeriodicMatern32(1)
            if kernel is 'PeriodicMatern52':
                kernel = GPy.kern.PeriodicMatern52(1)
            if kernel is 'Poly':
                kernel = GPy.kern.Poly(1)

            if kernel is 'Precomputed':
                kernel = GPy.kern.Precomputed(1)
            if kernel is 'Prod':
                kernel = GPy.kern.Prod(1)
            if kernel is 'RBF':
                kernel = GPy.kern.RBF(1)
            if kernel is 'RatQuad':
                kernel = GPy.kern.RatQuad(1)
            if kernel is 'Spline':
                kernel = GPy.kern.Spline(1)
            if kernel is 'SplitKern':
                kernel = GPy.kern.SplitKern(1)
            if kernel is 'StdPeriodic':
                kernel = GPy.kern.StdPeriodic(1)
            if kernel is 'TruncLinear':
                kernel = GPy.kern.TruncLinear(1)
            if kernel is 'TruncLinear_inf':
                kernel = GPy.kern.TruncLinear_inf(1)
            if kernel is 'White':
                kernel = GPy.kern.White(1)
            if kernel is 'WhiteHeteroscedastic':
                kernel = GPy.kern.WhiteHeteroscedastic(1)
                kernel = GPy.kern.sde_Bias(1)
            if kernel is 'sde_Brownian':
                kernel = GPy.kern.sde_Brownian(1)
            if kernel is 'sde_Exponential':
                kernel = GPy.kern.sde_Exponential(1)
            if kernel is 'sde_Linear':
                kernel = GPy.kern.sde_Linear(1)
            if kernel is 'sde_Matern32':
                kernel = GPy.kern.sde_Matern32(1)
            if kernel is 'sde_Matern52':
                kernel = GPy.kern.sde_Matern52(1)
            if kernel is 'sde_RBF':
                kernel = GPy.kern.sde_RBF(1)
            if kernel is 'sde_RatQuad':
                kernel = GPy.kern.sde_RatQuad(1)
            if kernel is 'sde_StdPeriodic':
                kernel = GPy.kern.sde_StdPeriodic(1)

        print "TRYING THE GP"
    ## new function to add -- what we get from this is the model. 
        training_data = self.data.tail(training_window)[[cons_col]]

        # observation_length is ALL of the current day (till now) + 4 hours
        observation_length = mins_in_day(self.now) + 4 * 60

        training_df = (
            training_data
            .reset_index()
            .rename(columns={'index':'X', 'House Consumption':'y'})
        )


    
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_values= training_df['X'].values.reshape(-1,1)
        X = scaler.fit_transform(X_values)

        X = pd.DataFrame(X)
        Y = pd.DataFrame(training_df['y'])

      #  X = normalize_dates(X, col = 'X')

        kernel = GPy.kern.PeriodicExponential(1)

        Xs = (X.fillna(value=0)
                # .apply(lambda x: x.total_seconds())
                # .pipe(lambda s: (s - s.mean()) / s.std())
                .sample(1000)
        )

        Ys = (Y.fillna(value=0)
                .pipe(lambda s: (s - s.mean()/ s.std())) #### TODO: fix: is this normalization step correct? 
                .loc[Xs.index]
                .values.reshape((-1,1))
        )

        Xs = Xs.values.reshape(-1,1)

        print("Xs-----------------")
        print(Xs)

        print("Ys-----------------")
        print(Ys) ## <--------------------- these haven't actually been normalized correctly, apparently.
 
        m = GPy.models.GPRegression(Xs,Ys,kernel)
        m.optimize(messages=True)

        current_consumption = training_data.tail(1)[cons_col]

        print "REACHED THE END"

        #### TODO: scale the index (below) appropriately by invoking the MinMaxScalar above. 
        # This, through the fit_transform function, stores global parameters needed to transform other inputs by the training data. 

        index = pd.DatetimeIndex(start=self.now, freq='T', periods=self.prediction_window)
        trans_index = scaler.fit_transform(df_values)
        result = pd.DataFrame(index=index)

        print("index---------------")
        print(index)


        ### the following line SHOULD work if index is scled appropriately. 

        result[index] = m.predict(index)



        return result


    def predict(self, alg_keys):
        index = pd.DatetimeIndex(start=self.now, freq='T', periods=self.prediction_window)
        result = pd.DataFrame(index=index)

        print("Inside the predict function")

        for key in alg_keys:
            print("key------" + key)
            r = self.algorithms[key]()
            print("r-------") #  'r is equal to {} or {}'.format(r, r2)
            print(r)
            if (r.shape[1] if r.ndim > 1 else 1) > 1:
                result[key] = r[:, 0]
                result[key + ' STD'] = r[:, 1]
            else:
                result[key] = r

        return result

    def simple_mean(self, training_window=24 * 60):
        training_data = self.data.tail(training_window)
        mean = training_data[cons_col].mean()
        return np.repeat(mean, self.prediction_window)

    def baseline_finder(self, training_window=1440 * 60, k=7, long_interp_range=300, short_interp_range=15,
                        half_window=100, similarity_interval=15, recent_baseline_length=200,
                        observation_length_addition=240, short_term_ease_method=easeOutQuad,
                        long_term_ease_method=easeOutQuad):
        training_data = self.data.tail(training_window)[[cons_col]]


        # observation_length is ALL of the current day (till now) + 4 hours
        observation_length = mins_in_day(self.now) + observation_length_addition

        try:
            similar_moments = find_similar_days(
                training_data, observation_length, k, similarity_interval, method=baseline_similarity)
        except NoSimilarMomentsFound:
            # no similar moments were found by our approach.. returning a mean of the last few hours
            recent_baseline = training_data[-recent_baseline_length:-1].mean()[cons_col]
            baseline = np.repeat(recent_baseline, self.prediction_window)
            baseline[0] = training_data.tail(1)[cons_col]
            return baseline

        baseline = calc_baseline(
            training_data, similar_moments, self.prediction_window, half_window, method=gauss_filt, interp_range=0)

        # long range interpolate

        interp_range = long_interp_range
        recent_baseline = training_data[-recent_baseline_length:-1].mean()[cons_col]

        method = np.array(list(map(lambda x: long_term_ease_method(x, 0, 1, interp_range), np.arange(interp_range))))

        if interp_range > 0:
            baseline[:interp_range] = lerp(np.repeat(recent_baseline, interp_range),
                                           baseline[:interp_range],
                                           method)

        # interpolate our prediction from current consumption to predicted
        # consumption
        # First index is the current time

        interp_range = short_interp_range
        current_consumption = training_data.tail(1)[cons_col]
        method = np.array(list(map(lambda x: short_term_ease_method(x, 0, 1, interp_range), np.arange(interp_range))))
        if interp_range > 0:
            baseline[:interp_range] = lerp(np.repeat(current_consumption, interp_range),
                                           baseline[:interp_range],
                                           method)

        return baseline[:-1]  # slice last line because we are actually predicting PredictionWindow-1

    def baseline_finder_dumb(self, training_window=1440 * 60, k=7):
        training_data = self.data.tail(training_window)[[cons_col]]

        # observation_length is ALL of the current day (till now) + 4 hours
        observation_length = mins_in_day(self.now) + 4 * 60

        mse = partial(baseline_similarity, filter=False)

        similar_moments = find_similar_days(
            training_data, observation_length, k, 60, method=mse)

        baseline = calc_baseline_dumb(training_data, similar_moments, self.prediction_window)

        # long range interpolate


        # interpolate our prediction from current consumption to predicted
        # consumption
        # First index is the current time

        current_consumption = training_data.tail(1)[cons_col]

        baseline[0] = current_consumption

        return baseline[:-1]  # slice last line because we are actually predicting PredictionWindow-1

    def usage_zone_finder(self, training_window=24 * 60 * 120, k=5):

        training_data = self.data.tail(training_window)[[cons_col]]

        # observation_length is ALL of the current day (till now) + 4 hours
        observation_length = mins_in_day(self.now) + (4 * 60)

        similar_moments = find_similar_days(
            training_data, observation_length, k, 15, method=baseline_similarity)

        half_window = 60

        baseline = calc_baseline(
            training_data, similar_moments, self.prediction_window, half_window, method=gauss_filt)

        highpass = calc_highpass(training_data, similar_moments,
                                 self.prediction_window, half_window, method=gauss_filt)
        final = baseline + highpass

        # interpolate our prediction from current consumption to predicted
        # consumption
        interp_range = 15
        # First index is the current time
        current_consumption = training_data.tail(1)[cons_col]

        final[:interp_range] = lerp(np.repeat(current_consumption, interp_range),
                                    final[:interp_range],
                                    np.arange(interp_range) / interp_range)

        return final[:-1]  # slice last line because we are actually predicting PredictionWindow-1

    def ARIMAforecast(self, training_window=1440 * 7, interval=60):
        training_data = self.data.tail(training_window)[cons_col].values

        TrainingDataIntervals = [sum(training_data[current: current + interval]) / interval for current in
                                 range(0, len(training_data), interval)]
        # test_stationarity(TrainingDataIntervals)
        try:
            model = sm.tsa.SARIMAX(TrainingDataIntervals, order=(1, 0, 0),
                                   seasonal_order=(1, 1, 1, int(1440 // interval)))
            model_fit = model.fit(disp=0)
        except:
            model = sm.tsa.SARIMAX(TrainingDataIntervals, enforce_stationarity=False)
            model_fit = model.fit(disp=0)
        #

        output = model_fit.forecast(int(self.prediction_window / interval))

        # Predictions = np.zeros((self.prediction_window, 4))
        # Predictions[:, 0] = np.arange(CurrentUTCTime, CurrentUTCTime + self.prediction_window * 60, 60)
        # Predictions[:, 1] = np.arange(CurrentLocalTime, CurrentLocalTime + self.prediction_window * 60, 60)
        # Predictions[:, 2] = np.repeat(output, interval)
        # Predictions[:, 3] = 0

        Predictions = np.repeat(output, interval)
        Predictions[0] = training_data[-1]  # current consumption

        return Predictions
