"""  Intelligent energy component contains the IEC class, that includes several algorithms
     for predicting consumption of a house, given historical data. It also contains an IECTester
     class that can be used to test and provide results on multiple IEC runs """

from __future__ import division

import pickle
from datetime import timedelta
import datetime as dt
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count
import os.path

import numpy as np
import GPy
import keras
import statsmodels.api as sm
import pandas as pd
import scipy.ndimage.filters
import scipy.signal
from scipy import spatial
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import time 
import pdb

import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import model_from_json

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

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

def is_weekday(day):
    if day in range(0,5):
        return 1
    else:
        return 0


def is_workday(hour):
    if hour in range(8,18):
        return 1
    else:
        return 0



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



# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


#get season from timestamp

def get_season(t):

    # the input must be a datetime
    if isinstance(t, datetime):
        t = t.to_datetime()
        Y = t.year # dummy leap year to allow input X-02-29 (leap day)

    if t.month < 4:
        season = "winter"
    elif t.month >3 and t.month<7:
        season = "spring"
    elif t.month > 6 and t.month < 10:
        season = "summer"
    else: 
        season = "fall"

    return season


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

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

def kernel_transformer(kernel = "PeriodicMatern52", dim = 1):
    if kernel is 'PeriodicExponential': 
        kernel = GPy.kern.PeriodicExponential(dim)
    if kernel is 'RBF':
        kernel = GPy.kern.RBF(dim) ## decrease starting length -> put some restrictions on optimization 
        ## bounds on optimization 
    if kernel is 'Coregionalize':
        kernel = GPy.kern.Coregionalize(dim)
    if kernel is "Periodic":
        kernel = GPy.kern.Periodic(dim)
    if kernel is 'Cosine':
        kernel = GPy.kern.Cosine(dim)    
    if kernel is 'CosineCustom':
        kernel = GPy.kern.Cosine(dim, lengthscale = 10)
    if kernel is 'DEtime':
        kernel = GPy.kern.DEtime(dim)
    if kernel is 'DiffGenomeKern':
        kernel = GPy.kern.DiffGenomeKern(dim)
    if kernel is 'Hierarchical':
        kernel = GPy.kern.Hierarchical(dim)
    if kernel is 'PeriodicExponential':
        kernel = GPy.kern.PeriodicExponential(dim)
    if kernel is 'PeriodicMatern32':
        kernel = GPy.kern.PeriodicMatern32(dim)
    if kernel is 'PeriodicMatern52':
        kernel = GPy.kern.PeriodicMatern52(dim)
    if kernel is 'Poly':
        kernel = GPy.kern.Poly(dim)
    if kernel is 'StdPeriodic':
        kernel = GPy.kern.StdPeriodic(dim)    
    if kernel is 'StdPeriodicSum':
        kernel = (GPy.kern.StdPeriodic(dim)+
            GPy.kern.StdPeriodic(dim, lengthscale= 10)+
            GPy.kern.StdPeriodic(dim, lengthscale=100))
    if kernel is 'sde_RatQuad':
        kernel = GPy.kern.sde_RatQuad(dim)
    if kernel is 'sde_StdPeriodic':
        kernel = GPy.kern.sde_StdPeriodic(dim)
    if kernel is 'custom1':
        kernel = GPy.kern.RBF(dim)+ GPy.kern.PeriodicMatern52(dim)        
    if kernel is 'custom1a':
        kernel = GPy.kern.PeriodicMatern52(dim, period = 1)        
    if kernel is 'custom1b':
        kernel = GPy.kern.PeriodicMatern52(dim, period = .1)      
    if kernel is 'custom1c':
        kernel = GPy.kern.PeriodicMatern52(dim, period = 10)        
    if kernel is 'custom1d':
        kernel = GPy.kern.PeriodicMatern52(dim, period = 15)       
    if kernel is 'custom1e':
        kernel = GPy.kern.PeriodicMatern52(dim, period = 20)      
    if kernel is 'custom1f':
        kernel = GPy.kern.PeriodicMatern52(dim, period = 25)
    if kernel is 'custom2':
        kernel = (GPy.kern.PeriodicMatern52(dim) + 
            GPy.kern.PeriodicMatern52(dim, period = 10)+
            GPy.kern.PeriodicMatern52(dim, period = 20)+
            GPy.kern.PeriodicMatern52(dim, period = 30))
    print(kernel)

    return kernel

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

def forecast_nn(model, batch_size, row):
    X = row[0:-1]
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]


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
            "Baseline Finder": partial(self.baseline_finder, training_window=1440 * 60, k=9, long_interp_range=50,
                                       short_interp_range=25, half_window=50, similarity_interval=5,
                                       recent_baseline_length=250,
                                       observation_length_addition=240, short_term_ease_method=easeOutSine,
                                       long_term_ease_method=easeOutCirc),
            "STLF": self.baseline_finder_dumb,
            "b1": partial(self.baseline_finder, training_window=1440 * 60, k=9, long_interp_range=50,
                          short_interp_range=25, half_window=50, similarity_interval=5, recent_baseline_length=250,
                          observation_length_addition=240, short_term_ease_method=easeOutSine,
                          long_term_ease_method=easeOutCirc),
            "b2": partial(self.baseline_finder, training_window=1440 * 60, k=3, long_interp_range=50,
                          short_interp_range=25, half_window=50, similarity_interval=5, recent_baseline_length=300,
                          observation_length_addition=240, short_term_ease_method=easeOutSine,
                          long_term_ease_method=easeOutCirc),
            "b3": partial(self.baseline_finder, training_window=1440 * 60, k=6, long_interp_range=50,
                          short_interp_range=25, half_window=50, similarity_interval=5, recent_baseline_length=200,
                          observation_length_addition=240, short_term_ease_method=easeOutSine,
                          long_term_ease_method=easeOutCirc),
            "b4": partial(self.baseline_finder, training_window=1440 * 60, k=12, long_interp_range=50,
                          short_interp_range=25, half_window=50, similarity_interval=5, recent_baseline_length=200,
                          observation_length_addition=240, short_term_ease_method=easeOutSine,
                          long_term_ease_method=easeOutCirc),
            "b5": partial(self.baseline_finder, training_window=1440 * 60, k=9, long_interp_range=50,
                          short_interp_range=25, half_window=50, similarity_interval=5, recent_baseline_length=250,
                          observation_length_addition=240, short_term_ease_method=easeOutSine,
                          long_term_ease_method=easeOutCirc),
            "b6": partial(self.baseline_finder, training_window=1440 * 60, k=9, long_interp_range=50,
                          short_interp_range=25, half_window=60, similarity_interval=5, recent_baseline_length=300,
                          observation_length_addition=240, short_term_ease_method=easeOutSine,
                          long_term_ease_method=easeOutCirc),
            "b7": partial(self.baseline_finder, training_window=1440 * 60, k=9, long_interp_range=50,
                          short_interp_range=25, half_window=50, similarity_interval=5, recent_baseline_length=200,
                          observation_length_addition=240, short_term_ease_method=easeOutSine,
                          long_term_ease_method=easeOutCirc),
            "GP PerExp": partial(self.gaussian_process_regression, training_window=1440*60, k=7, kernel="PeriodicExponential", 
                recent_baseline_length=10),
            "GP PerMatern32": partial(self.gaussian_process_regression, training_window=1440*60, k=7, kernel="PeriodicMatern32", 
                recent_baseline_length=10),
            "GP PerMatern52": partial(self.gaussian_process_regression, training_window=1440*60, k=7, kernel="PeriodicMatern52", 
                recent_baseline_length=10),
            "GP coregionalize": partial(self.gaussian_process_regression, training_window=1440*60, k=5, kernel='Coregionalize', 
                recent_baseline_length=10),
            "GP rbf": partial(self.gaussian_process_regression, training_window=1440*60, k=5, kernel="RBF", 
                recent_baseline_length=10),           
            "GP DEtime": partial(self.gaussian_process_regression, training_window=1440*60, k=5, kernel="DEtime", 
                recent_baseline_length=10),          
            "GP Poly": partial(self.gaussian_process_regression, training_window=1440*60, k=5, kernel="Poly", 
                recent_baseline_length=10),
            "GP Genome": partial(self.gaussian_process_regression, training_window=1440*60, k=5, kernel="DiffGenomeKern", 
                recent_baseline_length=10),
            "GP Custom1": partial(self.gaussian_process_regression, training_window=1440*60, k=5, kernel="custom1", 
                recent_baseline_length=10),
            "GP Custom1a": partial(self.gaussian_process_regression, training_window=1440*60, k=5, kernel="custom1a", 
                recent_baseline_length=10),
            "GP Custom1b": partial(self.gaussian_process_regression, training_window=1440*60, k=5, kernel="custom1b", 
                recent_baseline_length=10),
            "GP Custom1c": partial(self.gaussian_process_regression, training_window=1440*60, k=5, kernel="custom1c", 
                recent_baseline_length=10),
            "GP Custom1d": partial(self.gaussian_process_regression, training_window=1440*60, k=5, kernel="custom1d", 
                recent_baseline_length=10),
            "GP Custom1e": partial(self.gaussian_process_regression, training_window=1440*60, k=5, kernel="custom1e", 
                recent_baseline_length=10),
            "GP Custom1f": partial(self.gaussian_process_regression, training_window=1440*60, k=5, kernel="custom1f", 
                recent_baseline_length=10),
            "GP Custom2": partial(self.gaussian_process_regression, training_window=1440*60, k=5, kernel="custom2", 
                recent_baseline_length=10),
            "nn lstm": partial(self.rnn_lstm, training_window = 1440*60, k=7, recent_baseline_length=5),
            "residGP1": partial(self.GP_resids, training_window=1440*60, k=5,
                kernel="RBF", num_samples = 1600), # shorter than an hour ahead, shorter than predictive horizon 
            "residGP2": partial(self.GP_resids, training_window=1440*60, k=5, 
                kernel="CosineCustom", num_samples=3000),            
            "residGP3": partial(self.GP_resids, training_window=1440*60, k=5, 
                kernel="StdPeriodic", num_samples=2000),            
            "residGP4": partial(self.GP_resids, training_window=1440*60, k=5, 
                kernel="StdPeriodicSum", num_samples=2000),
            "residRNN": partial(self.RNN_resids, training_window= 1440*60, k=7, recent_baseline_length=5, 
                num_samples = 2000, nn_file = False, trained_nn_file = "", lag = 1),
            "residRNN_nolag": partial(self.RNN_resids_no_lag, training_window= 1440*60, k=7, recent_baseline_length=5, 
                num_samples = 2000, nn_file = False, trained_nn_file = "")
        }

        self.grid_search()

    def grid_search(self):
        algo_name = ""
        i = 0
        # adjust k parameter range
        for k in range(1, 10):
            algo_name_k = "b k=" + str(k) + " "
            # adjust baseline length range
            for recent_baseline_length in range(200, 350, 50):
                algo_name_recent = "recent_baseline_length=" + str(recent_baseline_length) + " "
                # Short-term easing functions
                for short_term_ease_method in [easeOutSine, easeInOutSine, easeInOutQuint]:
                    algo_name_short = "short_term_ease_method=" + short_term_ease_method.__name__ + " "
                    # Long-term easing functions
                    for long_term_ease_method in [easeOutCirc, easeInOutCirc, easeInOutExpo]:
                        algo_name_long = "long_term_ease_method=" + long_term_ease_method.__name__ # + " "
                        algo_name = algo_name_k + algo_name_recent + algo_name_short + algo_name_long
                        self.algorithms[algo_name] = partial(self.baseline_finder, training_window=1440 * 60,
                                                                                   k=k, long_interp_range=200,
                                                                                   short_interp_range=25,
                                                                                   half_window=80,
                                                                                   similarity_interval=5,
                                                                                   recent_baseline_length=recent_baseline_length,
                                                                                   observation_length_addition=240,
                                                                                   short_term_ease_method=short_term_ease_method,
                                 
                                                                                   long_term_ease_method=long_term_ease_method)



    def rnn_lstm(self, training_window=1440 * 60, k=7, recent_baseline_length=60):

        training_data = self.data.tail(training_window)[[cons_col]]
        # observation_length is ALL of the current day (till now) + 4 hours
        observation_length = mins_in_day(self.now) + 4 * 60

        training_df = (
            training_data
            .reset_index()
            .rename(columns={'index':'X', 'House Consumption':'y'})
        )

        scaler = MinMaxScaler(feature_range=(0, 1))
        X= training_df['X'].values.reshape(-1,1)

        X = pd.DataFrame(X)
        Y = pd.DataFrame(training_df['y'])

        Xs = (X.fillna(value=0))
        Ys = (Y.fillna(value=0)
#                .pipe(lambda s: (s - s.mean())/s.std()) #### TODO: fix: is this normalization step correct? 
      #          .loc[Xs.index]
                .values.reshape((-1,1))
        )
        #Xs = Xs.values.reshape(-1,1)

        Ys = scaler.fit_transform(Ys)

        lag = 3000

        Ys_supervised = pd.DataFrame(timeseries_to_supervised(Ys, lag=lag).values)

        X = Ys_supervised.take(range(0, lag), axis = 1)
        Y = Ys_supervised.take([lag], axis = 1)

        print(X.shape)
        print(X.head)

        X = X.values.reshape(-1,lag)
        Y = Y.values.reshape(-1,1)
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
   #     Y = np.reshape(Y, (Y.shape[0], 1, Y.shape[1]))

        # create and fit the LSTM network
        batch_size = 1
        model = Sequential()
        model.add(LSTM(4, batch_input_shape=(batch_size, 1, lag), stateful = True, return_sequences=True))
        model.add(LSTM(4, batch_input_shape=(batch_size, 1, lag), stateful = True, return_sequences=True))
        model.add(LSTM(4, batch_input_shape=(batch_size, 1, lag), stateful = True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        for i in range(3):
            model.fit(X, Y, epochs=1, batch_size=batch_size, verbose=2, shuffle = False)
            model.reset_states()

        prediction_feeder = np.reshape(X[-1], (1,1,-1))
        current_prediction = model.predict(prediction_feeder, batch_size=batch_size)
        trainPredict = scaler.inverse_transform(current_prediction)

        for i in range(self.prediction_window-1):
            prediction_feeder = np.append(prediction_feeder[0][0][1:lag], current_prediction).reshape(1,1,-1)
            current_prediction = model.predict(prediction_feeder, batch_size=batch_size)
            trainPredict = np.append(trainPredict, scaler.inverse_transform(current_prediction))

        return trainPredict

    def gaussian_process_regression(self, training_window=1440 * 60, k=7, kernel='PeriodicExponential',recent_baseline_length=5):

        kernel=kernel_transformer(kernel, dim = 1)
        
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
        X= training_df['X'].values.reshape(-1,1)
 
        X = pd.DataFrame(X)
        Y = pd.DataFrame(training_df['y'])

        Xs = (X.fillna(value=0)
                # .apply(lambda x: x.total_seconds())
                # .pipe(lambda s: (s - s.mean()) / s.std())
        #        .sample(1000)
        )

        Ys = (Y.fillna(value=0)
                .pipe(lambda s: (s - s.mean())/s.std()) 
        #        .loc[Xs.index]
                .values.reshape((-1,1))
        )

        Xs = Xs.values.reshape(-1,1)

        m = GPy.models.GPRegression(Xs,Ys,kernel)
        m.optimize(messages=True)

      
        index = pd.DatetimeIndex(start=self.now, freq='T', periods=self.prediction_window).to_datetime()
        result = pd.DataFrame(index=index)
        means, stds = m._raw_predict(index.values.reshape(-1,1))
        result['pred_means'] = (means*Y.std().values+Y.mean().values)

        baseline_finder_means = self.algorithms["b3"]()

        temp_combination = result['pred_means'] #+ baseline_finder_means
        recent_baseline = np.nanmean(Y[-recent_baseline_length:-1].values.reshape(-1,1))
        
        interp_range=120
        long_term_ease_method=easeOutQuad

        method = np.array(list(map(lambda x: long_term_ease_method(x, 0, 1, interp_range), np.arange(interp_range))))
        if interp_range > 0:
            temp_combination[:interp_range] = lerp(np.repeat(recent_baseline, interp_range),
                                           temp_combination[:interp_range],
                                           method)
        return temp_combination



    def GP_resids(self, training_window=1440 * 60, k=7, kernel='PeriodicExponential',
        recent_baseline_length=60, num_samples = 3000, GP_file = False, trained_GP_file = ""):

        dim = 2

        kernel_name = kernel
        kernel=kernel_transformer(kernel_name,dim = dim)

        # TODO: make the model update every 2 weeks. (Someone has to help here -- ask Dave.)

        training_data = pd.read_csv("resids_baseline_year.csv") 


        # make MinMaxScalers for the dataset
        scaler_resids = MinMaxScaler(feature_range=(0, 1))
        scaler_baseline = MinMaxScaler(feature_range=(0, 1))
        scaler_time = MinMaxScaler(feature_range=(0, 1))

        X= training_data[['time', "b_predictor"]].sample(num_samples)
        Y = scaler_resids.fit_transform(training_data[["resids"]].loc[X.index])
        X["b_predictor"]= scaler_baseline.fit_transform(X["b_predictor"].values.reshape(-1,1))
        X = X.iloc[1:]

        X["time"] = [i.replace("-07:00","") for i in X["time"]]
        X["time"] = [i.replace("-08:00","") for i in X["time"]]
        X["time"] = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in X["time"]]


        seasons = [get_season(i) for i in X["time"]]

        # X["is_fall"]  = [int(i == "fall")*.2 for i in seasons]
        # X["is_spring"] = [int(i == "spring")*.2 for i in seasons]
        # X["is_summer"] = [int(i == "summer")*.2 for i in seasons]
        # X["is_winter"] = [int(i == "winter")*.2 for i in seasons]

        # days = [i.weekday() for i in X["time"]]
        # X["weekday"] = [is_weekday(i)*.2 for i in days]


        # X["is_workhour"] = [(is_workday(i.hour) and is_weekday(i.weekday()))*.2 for i in X["time"]]
        X["time"] = scaler_time.fit_transform(X["time"].values.reshape(-1,1))

        X = X.values.reshape(-1,dim)
        Y = Y.reshape(-1,1)
        Y = Y[1:] 

        if not GP_file:
            trained_GP_file = "GP_"+kernel_name + str(num_samples) +".pkl"

        if os.path.isfile(trained_GP_file):
            GP_model_pkl = open(trained_GP_file, 'rb')
            m = pickle.load(GP_model_pkl)
            print "Loaded GP model :: ", m
        
        else:

            # train the model --- 

            m = GPy.models.GPRegression(X,Y,kernel)
            m.optimize(messages=True)

            # save it in a pickle 
            GP_pkl = open(trained_GP_file, 'wb')
            pickle.dump(m, GP_pkl)
            GP_pkl.close()

            
        ## predicting based on the GP 

        # setting up input data  

        index = pd.DatetimeIndex(start=self.now, freq='T', periods=self.prediction_window).to_datetime()
        
        result = (pd.DataFrame(index)
            .rename(columns = {0:"time"})) 

        result["baseline_finder_preds"]= (scaler_baseline
            .transform(self.algorithms["Baseline Finder"]()
                    .reshape(-1,1)))
        

        # seasons = [get_season(i) for i in result["time"]]
        # result["is_fall"]  = [int(i == "fall")*.2 for i in seasons]
        # result["is_spring"] = [int(i == "spring")*.2 for i in seasons]
        # result["is_summer"] = [int(i == "summer")*.2 for i in seasons]
        # result["is_winter"] = [int(i == "winter")*.2 for i in seasons]

        # days = [i.weekday() for i in result["time"]]
        # result["weekday"] = [is_weekday(i)*.2 for i in days]
        # result["is_workday"]= [(is_workday(i.hour) and is_weekday(i.weekday()))*.2 for i in result["time"]]

        result["time"]=scaler_time.transform(result["time"].values.reshape(-1,1))

        # predicting
        means, stds = m._raw_predict(result.values.reshape(-1,dim))

        result["baseline_finder_preds"]= (scaler_baseline
            .inverse_transform(
                    result["baseline_finder_preds"]
                    .values.reshape(-1,1)
                    )
            )


        result['pred_means'] = scaler_resids.inverse_transform(means)

        temp_combination = result['pred_means'] + result["baseline_finder_preds"]

        data_temp = training_data["b_predictor"] + training_data["resids"]
        recent_baseline = np.nanmean(data_temp[-recent_baseline_length:-1].values.reshape(-1,1))
        current_consumption = training_data.tail(1)[cons_col]
        
        # TODO: the smoothing should be done using the GP 
        # how does a GP prediction do more upfront then towards the end of the data? 

        interp_range=120
        long_term_ease_method=easeOutQuad


        method = np.array(list(map(lambda x: long_term_ease_method(x, 0, 1, interp_range), np.arange(interp_range))))
        if interp_range > 0:
            temp_combination[:interp_range] = lerp(np.repeat(current_consumption, interp_range),
                                           temp_combination[:interp_range],
                                           method)

        return temp_combination.values

    def RNN_resids(self, training_window=1440 * 60, k=7,recent_baseline_length=5, 
        num_samples = 2000, nn_file = False, trained_nn_file = "", lag = 30):

        # TODO nn with 1440 input nodes. Train 

        training_data = pd.read_csv("resids_baseline.csv")

        # make MinMaxScalers for the dataset
        
        scaler_resids = MinMaxScaler(feature_range=(-1, 1))
        scaler_baseline = MinMaxScaler(feature_range=(-1, 1))
        scaler_time = MinMaxScaler(feature_range=(-1, 1))

        X= training_data[['time', "b_predictor"]].sample(num_samples)
        Y = scaler_resids.fit_transform(training_data[["resids"]].loc[X.index])

        X = X.fillna(value=0)
        X["b_predictor"]= scaler_baseline.fit_transform(X["b_predictor"].values.reshape(-1,1))
        X = X.iloc[1:]

        if X["time"].iloc[0]==0:
            X=X.iloc[1:]

        X["time"] = [i.replace("-07:00","") for i in X["time"]]
        X["time"] = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in X["time"]]
        days = [i.weekday() for i in X["time"]]
        X["weekday"] = [is_weekday(i) for i in days]
        X["time"] = scaler_time.fit_transform(X["time"].values.reshape(-1,1))
        
        Y = Y.reshape(-1,1)
        Y = Y[1:] 

        Ys_supervised = pd.DataFrame(timeseries_to_supervised(Y, lag=lag).values)

        X = pd.concat([X.reset_index(drop=True), pd.DataFrame(Ys_supervised.take(range(0, lag), axis = 1))], axis = 1)
        X = X.values.reshape(-1,3+lag)
        Y = Ys_supervised.take([lag], axis = 1)
        Y = Y.values.reshape(-1,1)
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

        batch_size = 1
        training_times= 10

        if not nn_file:
            trained_nn_file = "nn_samples_"+ str(num_samples)+"_lag_"+str(lag) +"_epochs_"+str(training_times)+ ".json"
            trained_nn_weights_file = "nn_samples_"+ str(num_samples)+"_lag_"+str(lag) +"_epochs_"+str(training_times)+ ".h5"

        if os.path.isfile(trained_nn_file):
            json_file = open(trained_nn_file, 'r')
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
            model.load_weights(trained_nn_weights_file)
            print "Loaded NN model :: ", model
        
        else:
            print("training!")
            # train the model 
            model = Sequential()
            model.add(LSTM(4, batch_input_shape=(batch_size, 1, 3+lag), stateful = True, return_sequences=True))

            # ^-- remember that 3 is based on the number of "stable" features: currently, time, weekday, and baseline

            model.add(LSTM(4, batch_input_shape=(batch_size, 1, 3+lag), stateful = True, return_sequences=True))
            model.add(LSTM(4, batch_input_shape=(batch_size, 1, 3+lag), stateful = True))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')

            for i in range(training_times):
                model.fit(X, Y, epochs=1, batch_size=batch_size, verbose=2, shuffle = False)
                model.reset_states()

            # save it in a json
            model_json= model.to_json()
            with open(trained_nn_file, "w") as json_file:
                json_file.write(model_json)
            model.save_weights(trained_nn_weights_file)
            print("saved trained net")

        ## predicting based on the NN 

        # setting up input data  

        index = pd.DatetimeIndex(start=self.now, freq='T', periods=self.prediction_window).to_datetime()
        result = (pd.DataFrame(scaler_time
            .transform(index.values.reshape(-1,1)))
            .rename(columns = {0:"time"}))

        secs = scaler_time.inverse_transform(result["time"].reshape(-1,1)).reshape(-1)
        result["weekday"] =  [is_weekday(i.weekday()) for i in pd.to_datetime(secs)]        
        result["baseline_finder_preds"]= (scaler_baseline
            .transform(self.algorithms["Baseline Finder"]()
                    .reshape(-1,1)))

        result["pred_means"] = np.zeros(result.shape[0])
        ## nn stuff 
        time_counter = X[-1][0][-lag:]
        prediction_feeder = np.concatenate([X[-1][0][0:3], time_counter]).reshape(1,-1,lag+3)
        current_prediction = model.predict(prediction_feeder, batch_size=batch_size)
        trainPredict = scaler_resids.inverse_transform(current_prediction)    

        # create and fit the LSTM network
   
        for i in range(self.prediction_window-1):
            if lag ==1:
                time_counter = current_prediction[0]
            else: 
                time_counter = np.append(prediction_feeder[0][0][-(lag-1):], current_prediction).reshape(1,1,-1)

            prediction_feeder = np.append([result["time"].iloc[i], result["baseline_finder_preds"].iloc[i], 
            result["weekday"].iloc[i]], time_counter).reshape(1,-1,lag+3)
            current_prediction = model.predict(prediction_feeder, batch_size=batch_size)
            trainPredict = np.append(trainPredict, scaler_resids.inverse_transform(current_prediction))

        # predicting

        temp_combination = trainPredict+ scaler_resids.inverse_transform(result["baseline_finder_preds"].values.reshape(-1,1))[0]

        data_temp = training_data["b_predictor"] + training_data["resids"]
        recent_baseline = np.nanmean(data_temp[-recent_baseline_length:-1].values.reshape(-1,1))
        
        interp_range=120
        long_term_ease_method=easeOutQuad



        method = np.array(list(map(lambda x: long_term_ease_method(x, 0, 1, interp_range), np.arange(interp_range))))
        if interp_range > 0:
            temp_combination[:interp_range] = lerp(np.repeat(recent_baseline, interp_range),
                                           temp_combination[:interp_range],
                                           method)

        return temp_combination

    def RNN_resids_no_lag(self, training_window=1440 * 60, k=7,recent_baseline_length=5, 
        num_samples = 2000, nn_file = False, trained_nn_file = ""):

        training_data = pd.read_csv("resids_baseline.csv")

        # make MinMaxScalers for the dataset
        
        scaler_resids = MinMaxScaler(feature_range=(-1, 1))
        scaler_baseline = MinMaxScaler(feature_range=(-1, 1))
        scaler_time = MinMaxScaler(feature_range=(-1, 1))

        X= training_data[['time', "b_predictor"]].sample(num_samples)
        Y = scaler_resids.fit_transform(training_data[["resids"]].loc[X.index])

        X = X.fillna(value=0)
        X["b_predictor"]= scaler_baseline.fit_transform(X["b_predictor"].values.reshape(-1,1))
        X = X.iloc[1:]

        if X["time"].iloc[0]==0:
            X=X.iloc[1:]

        X["time"] = [i.replace("-07:00","") for i in X["time"]]
        X["time"] = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in X["time"]]
        days = [i.weekday() for i in X["time"]]
        X["weekday"] = [is_weekday(i) for i in days]
        X["time"] = scaler_time.fit_transform(X["time"].values.reshape(-1,1))
        
        Y = Y.reshape(-1,1)
        Y = Y[1:] 

        X = X.values.reshape(-1,3)
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

        batch_size = 1
        training_times= 10

        if not nn_file:
            trained_nn_file = "nn_samples_"+ str(num_samples) +"_epochs_"+str(training_times)+ ".json"
            trained_nn_weights_file = "nn_samples_"+ str(num_samples)+"_epochs_"+str(training_times)+ ".h5"

        if os.path.isfile(trained_nn_file):
            json_file = open(trained_nn_file, 'r')
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
            model.load_weights(trained_nn_weights_file)
            print "Loaded NN model :: ", model
        
        else:
            print("training!")
            # train the model 
            model = Sequential()
            model.add(LSTM(4, batch_input_shape=(batch_size, 1, 3), stateful = True, return_sequences=True))

            # ^-- remember that 3 is based on the number of "stable" features: currently, time, weekday, and baseline

            model.add(LSTM(4, batch_input_shape=(batch_size, 1, 3), stateful = True, return_sequences=True))
            model.add(LSTM(4, batch_input_shape=(batch_size, 1, 3), stateful = True))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')

            for i in range(training_times):
                model.fit(X, Y, epochs=1, batch_size=batch_size, verbose=2, shuffle = False)
                model.reset_states()

            # save it in a json
            model_json= model.to_json()
            with open(trained_nn_file, "w") as json_file:
                json_file.write(model_json)
            model.save_weights(trained_nn_weights_file)
            print("saved trained net")

        ## predicting based on the NN 

        # setting up input data  

        index = pd.DatetimeIndex(start=self.now, freq='T', periods=self.prediction_window).to_datetime()
        result = (pd.DataFrame(scaler_time
            .transform(index.values.reshape(-1,1)))
            .rename(columns = {0:"time"}))

        secs = scaler_time.inverse_transform(result["time"].reshape(-1,1)).reshape(-1)
        result["weekday"] =  [is_weekday(i.weekday()) for i in pd.to_datetime(secs)]        
        result["baseline_finder_preds"]= (scaler_baseline
            .transform(self.algorithms["Baseline Finder"]()
                    .reshape(-1,1)))

        prediction_feeder = result.values.reshape(1,-1,1)

        result["pred_means"] = scaler_resids.inverse_transform(model.predict(prediction_feeder, batch_size=batch_size))
  

        temp_combination = result["pred_means"]+ scaler_resids.inverse_transform(result["baseline_finder_preds"].values.reshape(-1,1))[0]

        data_temp = training_data["b_predictor"] + training_data["resids"]
        recent_baseline = np.nanmean(data_temp[-recent_baseline_length:-1].values.reshape(-1,1))
        
        interp_range=120
        long_term_ease_method=easeOutQuad

        # method = np.array(list(map(lambda x: long_term_ease_method(x, 0, 1, interp_range), np.arange(interp_range))))
        # if interp_range > 0:
        #     temp_combination[:interp_range] = lerp(np.repeat(recent_baseline, interp_range),
        #                                    temp_combination[:interp_range],
        #                                    method)

        return temp_combination


    def predict(self, alg_keys):

        index = pd.DatetimeIndex(start=self.now, freq='T', periods=self.prediction_window)
        result_pred = pd.DataFrame(index=index)
        for key in alg_keys:
            pred = self.algorithms[key]()
            if (pred.shape[1] if pred.ndim > 1 else 1) > 1:
                result_pred[key] = pred[:, 0]
                result_pred[key + ' STD'] = pred[:, 1]
            else:
                result_pred[key] = pred
        return result_pred

    def simple_mean(self, training_window=24 * 60):
        training_data = self.data.tail(training_window)
        mean = training_data[cons_col].mean()
        return np.repeat(mean, self.prediction_window)

    def baseline_finder(self, training_window=1440 * 60, k=7, long_interp_range=30, short_interp_range=15,
                        half_window=50, similarity_interval=15, recent_baseline_length=60,
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



def worker(ie, alg_keys):
    return ie.predict(alg_keys)

class IECTester:
    """Performs several tests to the Intelligent Energy Component.
    """
    version = 0.1

    def __init__(self, data, prediction_window, testing_range, save_file='save.p'):
        self.data = data
        self.prediction_window = prediction_window
        self.range = testing_range
        self.save_file = save_file

        self.hash = 0

        self.TestedAlgorithms = set()
        self.results = dict()
        if save_file is not None:
            self.load()

    def load(self):
        try:
            with open(self.save_file, "rb") as f:
                savedata = pickle.load(f)
                if (savedata['version'] == self.version
                    and savedata['range'] == self.range
                    and savedata['hash'] == self.hash
                    and savedata['PredictionWindow'] == self.prediction_window):
                    self.TestedAlgorithms = savedata['TestedAlgorithms']
                    self.results = savedata['results']

        except (IOError, EOFError):
            pass

    def save(self):
        savedata = dict()
        savedata['version'] = self.version
        savedata['range'] = self.range
        savedata['hash'] = self.hash
        savedata['PredictionWindow'] = self.prediction_window
        savedata['TestedAlgorithms'] = self.TestedAlgorithms
        savedata['results'] = self.results

        with open(self.save_file, "wb") as f:
            pickle.dump(savedata, f)

    def run(self, multithread=True, force_processes=None, *args):
        """Runs the tester and saves the result
        """
        
        algorithms_to_test = set(args) - self.TestedAlgorithms
        if not algorithms_to_test:
            return

        for key in algorithms_to_test:
            self.results[key] = np.zeros(
                [len(self.range), self.prediction_window])
            self.results[key + " STD"] = np.zeros(
                [len(self.range), self.prediction_window])

        self.results['GroundTruth'] = np.zeros(
            [len(self.range), self.prediction_window])

        IECs = [IEC(self.data[:(-offset)]) for offset in self.range]

        if multithread:
            if force_processes is None:
                p = Pool(processes=cpu_count() - 2)
            else:
                p = Pool(force_processes)
            func_map = p.imap(
                partial(worker, alg_keys=algorithms_to_test),
                IECs)
        else:
            func_map = map(
                partial(worker, alg_keys=algorithms_to_test),
                IECs)
        try:
            with tqdm(total=len(IECs), smoothing=0.0) as pbar:
                for index, (offset, result) in enumerate(zip(self.range, func_map)):

                    for key in algorithms_to_test:
                        std_key = key + " STD"

                        self.results[key][index, :] = result[key].as_matrix()
                        if std_key in result:
                            self.results[std_key][index, :] = result[std_key].as_matrix()

                    self.results['GroundTruth'][index, :] = self.data[
                                                            -offset - 1
                                                            : -offset + self.prediction_window - 1
                                                            ][cons_col].as_matrix()
                    pbar.update(1)

            self.TestedAlgorithms.update(algorithms_to_test)

        except KeyboardInterrupt:
            pass
        finally:
            if multithread:
                p.terminate()
                p.join()

    def rmse(self):
        """For each second in the future find the root mean square prediction error
        """
        rmse = dict()

        for key in self.TestedAlgorithms:
            rmse[key] = [mean_squared_error(
                self.results['GroundTruth'][:, col],
                self.results[key][:, col]) ** 0.5 for col in range(self.prediction_window)]

        return rmse

    def simple_prediction(self, offset):

        prediction = dict()
        for key in self.TestedAlgorithms:
            prediction[key] = self.results[key][offset, :]
        prediction['GroundTruth'] = self.results['GroundTruth'][offset, :]

        return prediction

    def average_rmse(self):
        """Average the RMSE of each algorithms over our runs
        """

        armse = dict()

        for key in self.TestedAlgorithms:
            rmse = [mean_squared_error(self.results['GroundTruth'][i, :],
                                       self.results[key][i, :]
                                       ) for i in range(self.results[key].shape[0])
                    ]
            armse[key] = np.mean(rmse)
        return armse

    def average_total_error(self):
        ate = dict()

        for key in self.TestedAlgorithms:
            total_error = [abs(np.sum(self.results['GroundTruth'][i, :])
                               - np.sum(self.results[key][i, :])
                               ) for i in range(self.results[key].shape[0])]
            ate[key] = np.mean(total_error)
        return ate

    def similarity_tester(self, offset, method=cosine_similarity):
        pass


def main():
    dataset_filename = '../dataset-kw.gz'
    dataset_tz = 'Europe/Zurich'

    data = pd.read_csv(dataset_filename, parse_dates=[0], index_col=0).tz_localize('UTC').tz_convert(dataset_tz)

    prediction_window = 960
    testing_range = range(prediction_window, prediction_window + 200, 1)

    tester = IECTester(data, prediction_window, testing_range, save_file=None)
    tester.run('Baseline Finder Hybrid', multithread=False)


if __name__ == '__main__':
    main()





