from scipy import signal
import numpy as np

# from . import exceptions


def butter_lowpass(cutoff, fs, order=5):
    """
    Design an Nth-order digital or analog Butterworth filter and return the filter coefficients.
    Scipy's butter filter creator...

    Parameters
    ----------
    cutoff : float
        Cutoff frequency for the low pass.

    fs : float
        Sampling frequency.

    order : int
        Order of the filter.

    Returns
    -------
    b, a : ndarray
        Numerator (b) and denominator (a) polynomials of the IIR filter. Feed these to variables to scipy's filter functions.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    This low pass filter uses scipy's butter_lowpass method which comes with phase shift.

    Parameters
    ----------
    data : pd.Series or np.array or such
        Data

    cutoff : float
        Cutoff frequency for the lowpass filter.

    fs : float
        Sampling frequency of the data.

    order : int
        Order of the filter.

    Returns
    -------
    y : ndarray
        Filtered data.

    b, a : ndarray
        The return values from the butter_lowpass function.
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y, b, a


def filter_lowpass(data, cutoff, fs, order=2):
    """
    This low pass filter uses scipy's filtfilt method which comes without phase shift.

    Parameters
    ----------
    data : pd.Series or np.array or such
        Data

    cutoff : float
        Cutoff frequency for the lowpass filter.

    fs : float
        Sampling frequency of the data.

    order : int
        Order of the filter. Due to the filtfilt function these order is double.

    Returns
    -------
    y : ndarray
        Filtered data.

    b, a : ndarray
        The return values from the butter_lowpass function.
    """
    b, a = butter_lowpass(cutoff, fs, order)
    y = signal.filtfilt(b, a, data, method="gust")
    return y, b, a


def quadratic_addition(data):
    """
    Returns the average of the quadratic added values.
    """
    return np.sqrt(data.pow(2).sum()) / data.shape[0]


def linear(x, m, b):
    """
    Returns a linear function f(x) = m*x + b
    """
    return m * x + b


def gaus(x, a, mu, sigma):
    """
    Returns a gaussian function f(x) = a * exp( -1/2 * [(x - mu) / sigma]^(1/2) )

    # TODO: Implement with SciPy.stats.norm function. This method has a fit method, too.
    """
    return a * np.exp(-1 / 2 * np.power((x - mu) / sigma, 2))


def std90(x):
    """
    Returns the standard deviation of the central 90% of the data set.
    """
    x1 = np.nanpercentile(x, 5)
    x2 = np.nanpercentile(x, 95)
    x = x[x > x1]
    x = x[x < x2]
    if len(x) < 1:
        return np.nan
    return np.std(x)


def rms90(x):
    """
    Returns the root mean square of the central 90% of the data set.

    Parameters
    ----------

    x : np.array

    Returns
    -------
    float

    """
    x1 = np.nanpercentile(x, 5)
    x2 = np.nanpercentile(x, 95)
    x = x[x > x1]
    x = x[x < x2]

    if len(x) < 1:
        return np.nan
    return np.sqrt(np.sum(np.power(x, 2)) / len(x))


def mean90(x):
    """
    Returns the mean of the central 90% of the data set.

    Parameters
    ----------

    x : np.array

    Returns
    -------
    float

    """
    x1 = np.nanpercentile(x, 5)
    x2 = np.nanpercentile(x, 95)
    x = x[x > x1]
    x = x[x < x2]
    if len(x) < 1:
        return np.nan
    return np.mean(x)


def median90(x):
    """
    Returns the median of the central 90% of the data set.

    Parameters
    ----------

    x : np.array

    Returns
    -------
    float

    """
    x1 = np.nanpercentile(x, 5)
    x2 = np.nanpercentile(x, 95)
    x = x[x > x1]
    x = x[x < x2]
    if len(x) < 1:
        return np.nan
    return np.median(x)


def histogr(data, **kwargs):
    """
    Returns a histrogram where len(n)=len(hist). Uses np.histrogram and creates
    an x-array with the centers of the previously returned histrogram edges.

    This way the histrogram can easily be plotted.

    The code is easy:
    >>>>>
        n, hist = np.histogram(data, **kwargs)
        dx = hist[1]-hist[0]
        hbins = np.array([d+dx/2. for d in hist[:-1]])

        return hbins, n
    <<<<<

    Parameters
    ----------
    All parameters of the np.histrogram method.


    """
    n, hist = np.histogram(data, **kwargs)
    dx = hist[1] - hist[0]
    hbins = np.array([d + dx / 2.0 for d in hist[:-1]])

    return hbins, n


def break_pd_series_into_parts(series, n):
    """
    Breaks a pandas series into n parts. Returns a list of tuples with the start and end index of each part.

    Parameters
    ----------
    series : pd.Series
        The series to break into parts.

    n : int
        The number of parts to break the series into.

    Returns
    -------
    list
        A list of tuples with the start and end index of each part.
    """
    # Calculate the length of each part
    part_length = len(series) // n

    # Create a list of ranges for each part
    ranges = []
    start = 0
    for i in range(n - 1):
        end = start + part_length
        ranges.append((start, end))
        start = end
    ranges.append((start, len(series)))

    return ranges


# def calc_stats_with_diff_length(signals, min_threshold=25):
#     """
#     Calculates the average and standard deviation of the given signals. If the signals have different lengths, the
#     shortest signal is used as the basis for the calculation. If the shortest signal is shorter than the minimum
#     threshold, a ValueError is raised.

#     Parameters
#     ----------
#     signals : list
#         A list of pandas Series or numpy arrays.
#     min_threshold : int
#         The minimum length of the signals. If the shortest signal is shorter than this threshold, a ValueError is
#         raised.

#     Returns
#     -------
#     tuple
#         A tuple of the average and standard deviation of the signals.
#     """
#     if len(signals) == 1:
#         return signals[0].values, len(signals[0].values) * [0]

#     signal_lengths = [len(s) for s in signals]
#     min_length, max_length = min(signal_lengths), max(signal_lengths)

#     if max_length != min_length:
#         if min_length < min_threshold:
#             raise exceptions.LengthError(
#                 "The minimum length of the signals is less than the minimum threshold."
#             )
#         else:
#             shortened_signal = np.array([s.iloc[:min_length] for s in signals])
#             return np.average(shortened_signal, axis=0), np.std(
#                 shortened_signal, axis=0
#             )
#     return np.average(signals, axis=0), np.std(signals, axis=0)
