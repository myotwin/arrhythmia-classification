"""
This module contains the classes for preprocessing the raw data as implemented by Dr. Hendrik Windel.
"""

import pandas as pd
import numpy as np

from utils.general import linear, gaus, histogr, std90, filter_lowpass

from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from scipy.optimize import curve_fit
from scipy import signal

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

import matplotlib as mpl
import matplotlib.pyplot as plt

# Doane's rule for width of histogram bins
# https://medium.com/@maxmarkovvision/optimal-number-of-bins-for-histograms-3d7c48086fde

# plt.style.use(["thesis", "large"])

import os

# catch optimize warnings
import warnings
from scipy.optimize import OptimizeWarning

pio.templates.default = "plotly"


#####################################################
#
# Global variable needed in most classes and methods
#
#####################################################

# global output_dir_path

# try:
#     if type(output_dir_path) == str:
#         pass
# except NameError:
#     output_dir_path = ''

#####################################################
#
# Global variable needed in most classes and methods
#                       END
#####################################################


class Calcium:
    """
    This class performs a pedestal correction even of a sloped pedestal and returns the peaks and other information if needed.
    """

    def __init__(self, calc_data, smooth=True):
        """
        Initialize the Calcium class with the provided data and parameters.

        Args:
            calc_data (pd.Series): The raw calcium data to be processed.
            smooth (bool): Whether to apply smoothing to the data. (default: True)
        If smooth is False, the pedestal correction, time correction, and savgol filtering are not applied.
        """
        self.res_pedestal = dict()
        self.data_raw = calc_data
        self.smooth = smooth
        # act on raw data
        self._prepare_data()

        # prepare filtered data
        self.data = pd.Series(dtype="float64")
        if self.smooth:
            self._getFilteredData()
        else:
            self.data = self.data_raw
        # if not prepared:
        #     return
        self.peaks, self.peak_widths, self.peak_indexes = self.getPeaks()

    def _prepare_data(self):
        try:
            self._correct_pedestal()
        except ValueError:
            pass
        if self.smooth:
            self._doTimeCorrection()

    def _correct_pedestal(self):
        """
        Do pedestal correction of even sloped pedestals.
        This is done in two steps:
            1. Make a histogram of the data and extract the most probable (pedestal) value.
            2. Prepare a "pedestal only" data set by limiting the data point to be >mu +- 2 sigma<
            3. Use the RANSAC-regressor to estimate the sloped pedestal.
            4. Apply pedestal correction.

        """
        g_bin_width = 0.01  # seems to be a good binning width
        x, y = histogr(
            self.data_raw,
            bins=np.arange(
                self.data_raw.min() - 5 * g_bin_width,
                self.data_raw.max() + 5 * g_bin_width,
                g_bin_width,
            ),
        )

        # prepare fit data
        dbin = 3
        y_fit = y[y.argmax() - dbin : y.argmax() + dbin + 1]
        x_fit = x[y.argmax() - dbin : y.argmax() + dbin + 1]

        # do fit and extract errors
        res = []
        pcov = []
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            try:
                res, pcov = curve_fit(
                    gaus,
                    x_fit,
                    y_fit,
                    p0=[y_fit.max(), x[y.argmax()], std90(self.data_raw)],
                    maxfev=2000,
                )
                perr = np.sqrt(abs(np.diag(pcov)))
            except OptimizeWarning:
                print("OptimizeWarning in curve_fit for Calcium.")
                res = [np.nan, np.median(self.data_raw), std90(self.data_raw)]
                perr = [0, 0, 0]
            except RuntimeError:
                # fit results probably not found, therefore create different version
                print("RuntimeError in curve_fit for Calcium.")
                res = [np.nan, np.median(self.data_raw), std90(self.data_raw)]
                perr = [0, 0, 0]
        # create clean data set
        _f = 2
        calc_clean = self.data_raw.loc[
            (self.data_raw < res[1] + _f * abs(res[2]))
            & (self.data_raw > res[1] - _f * abs(res[2]))
        ]

        # save fit data
        self.res_pedestal["g1_amplitude"] = res[0]
        self.res_pedestal["g1_mu"] = res[1]
        self.res_pedestal["g1_sigma"] = abs(res[2])
        self.res_pedestal["g1_amplitude_err"] = perr[0]
        self.res_pedestal["g1_mu_err"] = perr[1]
        self.res_pedestal["g1_sigma_err"] = perr[2]

        # prepare sloped pedestal estimation
        if self.smooth:
            try:
                ransac = RANSACRegressor(random_state=0)
                X = calc_clean.to_numpy().reshape(-1, 1)
                y = calc_clean.index.to_numpy().reshape(-1, 1)
                ransac.fit(X, y)
            except ValueError:
                print("ValueError in RANSACRegressor for Calcium.")
                raise ValueError

            # do linear fit on inliers
            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)

            res, pcov = curve_fit(
                linear, calc_clean[inlier_mask].index, calc_clean[inlier_mask]
            )
            perr = np.sqrt(abs(np.diag(pcov)))

            # save fit results
            self.res_pedestal["l_slope"] = res[0]
            self.res_pedestal["l_constant"] = res[1]
            self.res_pedestal["l_slope_err"] = perr[0]
            self.res_pedestal["l_constant_err"] = perr[1]

            # apply pedestal correction
            def _correct(row):
                return row.calc - linear(
                    row.name,
                    self.res_pedestal["l_slope"],
                    self.res_pedestal["l_constant"],
                )

            self.data_raw = pd.DataFrame(self.data_raw).apply(_correct, axis=1)

    def _doTimeCorrection(self):
        """
        The MEA and Force signals seem to be in sync. The Calc signal is not. This method aims to correct this.
        The correction factor is derived in the Jupyter notebook dev_calc_time_calibration.ipynb
        """
        _corr_factor = 1 + 0.0028388903486529877  # in sec per recorded sec
        _tmax = self.data_raw.index[-1]

        self.data_raw.index = np.round(
            self.data_raw.index.to_numpy() * _corr_factor, decimals=2
        )
        self.data_raw = self.data_raw.loc[
            :_tmax
        ]  # waveform shall not be larger than before

    def _getFilteredData(self, window_length=21, polyorder=2, **kwargs):
        """
        Returns filtered mea data. Uses scipys signal.savgol_filter as used by PRISM.
        """
        self.data = pd.Series(
            data=signal.savgol_filter(
                self.data_raw, window_length=window_length, polyorder=polyorder
            ),
            index=self.data_raw.index,
        )

    def getPeaks(self, factor_prominence=2, factor_height=0, wlen=401):
        """
        Returns the peaks and other properties in the Calcium wf.

        Parameters
        ----------
        factor_fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)prominence : float
            The minimum prominence is calculated by >mu + factor_prominence * sigma< of the second pedestal fit.

        factor_height : float
            The minimum height is calculated by >mu + factor_prominence * sigma< of the second pedestal fit.

        wlen : int
            Used to calculate peak prominences. See scipy's signal.peak_prominences

        """

        self._limit_low_h = (
            factor_height * self.res_pedestal["g1_sigma"]
        )  # the lower limit should be larger than the pedestal
        self._limit_low_p = (
            factor_prominence * self.res_pedestal["g1_sigma"]
        )  # the lower limit should be larger than the pedestal
        peak_indexes, props = signal.find_peaks(
            self.data, height=self._limit_low_h, prominence=self._limit_low_p, wlen=wlen
        )

        _widths = []
        for _rh in [0.1, 0.5, 0.9, 0.97, 1]:
            _width = dict()
            _ws = signal.peak_widths(
                self.data,
                peak_indexes,
                rel_height=_rh,
                prominence_data=(
                    props["prominences"],
                    props["left_bases"],
                    props["right_bases"],
                ),
                wlen=wlen,
            )

            _width["width"] = _ws[0]
            _width["width_height"] = _ws[1]
            _width["left_ips"] = _ws[2]
            _width["right_ips"] = _ws[3]
            _width["rel_height"] = [int(round(_rh * 100)) for _ in range(len(_ws[0]))]
            _width["peaks"] = peak_indexes

            _widths.append(_width)

        props["peaks"] = peak_indexes
        peaks = pd.DataFrame(props)
        _convert_to_time = ["left_bases", "right_bases", "peaks"]
        for _con in _convert_to_time:
            peaks.loc[:, _con] = self.data.iloc[peaks.loc[:, _con]].index

        widths = pd.concat([pd.DataFrame(_w) for _w in _widths], ignore_index=True)
        _convert_to_time = ["width", "left_ips", "right_ips", "peaks"]
        for _con in _convert_to_time:
            widths.loc[:, _con] = self.data.iloc[widths.loc[:, _con]].index

        # bases values in peaks is not robust, exchange with widths data
        peaks.loc[:, ["left_bases", "right_bases"]] = widths.loc[
            widths.rel_height == 90, ["left_ips", "right_ips"]
        ].values
        peaks.insert(
            0,
            "width_height",
            widths.loc[widths.rel_height == 90, ["width_height"]].values,
        )

        peaks.loc[:, ["left_wings", "right_wings"]] = widths.loc[
            widths.rel_height == 10, ["left_ips", "right_ips"]
        ].values
        peaks.insert(
            0,
            "wing_height",
            widths.loc[widths.rel_height == 10, ["width_height"]].values,
        )

        # time to peak and time to relaxation
        _t2p = (
            widths.loc[widths.rel_height == 10, "left_ips"].to_numpy()
            - widths.loc[widths.rel_height == 90, "left_ips"].to_numpy()
        )
        peaks.insert(0, "t2p", _t2p)

        _t2r = (
            widths.loc[widths.rel_height == 90, "right_ips"].to_numpy()
            - widths.loc[widths.rel_height == 10, "right_ips"].to_numpy()
        )
        peaks.insert(0, "t2r", _t2r)

        return peaks, widths, peak_indexes


class Force:
    """
    This class performs a pedestal correction and returns the peaks and other information if needed.

    Noise reduction is missing so far.

    Parameters
    ----------
    data : pd.DataFrame with only one column named "force" or pd.Series
        The raw force data to be processed.

    filter_params : dict, default None
        Options are:
            cutoff : float  -> Cutoff frequency for a possible low pass filter.
            order : int     -> Filter order.

    smooth : bool, default True
        Whether to apply smoothing to the data.
        - if smooth is False, lowpass filtering, pedestal correction, and smoothing are not applied.
        - However, the method for pedestal correction is still called to calculate new pedestal parameters (mu, sigma)
          which are utilized for peak prominence threshold calculation.

    bin_width : float, default None
        The bin width for the histogram used in the pedestal correction.
        If None, a default value of 0.01 is used.

    """

    def __init__(
        self, data, filter_params=dict(order=5, cutoff=5.2), smooth=True, bin_width=None
    ):
        self.res_pedestal = dict()
        self.filter_params = filter_params
        self._limit_low_h = 0
        self._limit_low_t = 0
        self.smooth = smooth
        self.bin_width = bin_width

        self.data_raw = data
        self.data = self._prepare_data(self.data_raw)
        # do pedestal corection to raw data
        self.data_raw = self.data_raw.apply(lambda x: x - self.res_pedestal["g1_mu"])

        self.peaks, self.peak_widths, self.peak_indexes = self.getPeaks()

    def _prepare_data(self, data):
        """
        Applies the low pass filter if params are given and corrects the pedestal.
        """

        if isinstance(self.filter_params, type(dict())):
            if self.smooth:
                data = self._apply_filter(
                    data, self.filter_params["cutoff"], self.filter_params["order"]
                )

        if self.smooth:
            data = self._getSmoothData(data)

        data = self._correct_pedestal(data)

        # adding smoothing

        return data

    def _getSmoothData(self, data, window_length=21, polyorder=2, **kwargs):
        """
        Returns Smooth force data. Uses scipys signal.savgol_filter as used by PRISM.
        """
        return pd.Series(
            data=signal.savgol_filter(
                data, window_length=window_length, polyorder=polyorder
            ),
            index=data.index,
        )

    def _apply_filter(self, data, cutoff, order):
        """
        The Force data comes often with some noisy frequency of around 10 Hz. This method filters it out.
        """
        fs = 1 / (data.index[1] - data.index[0])
        y, b, a = filter_lowpass(data, cutoff, fs, order)

        self.filter_params["a"] = a
        self.filter_params["b"] = b

        data = pd.DataFrame({"force": y}, index=data.index)

        return data.force

    def _correct_pedestal(self, data):
        """
        Do pedestal correction.
        This is done in two steps:
            1. Make a histogram of the data and extract the most probable (pedestal) value.
            2. Apply pedestal correction.
        """
        if self.bin_width is None:
            g_bin_width = 0.01
        else:
            g_bin_width = self.bin_width
        x, y = histogr(
            data,
            bins=np.arange(
                data.min() - 5 * g_bin_width, data.max() + 5 * g_bin_width, g_bin_width
            ),
        )

        # prepare fit data
        dbin = 2
        y_fit = y[y.argmax() - dbin : y.argmax() + dbin + 1]
        x_fit = x[y.argmax() - dbin : y.argmax() + dbin + 1]

        # do fit and extract errors
        res = []
        pcov = []
        optimze_warning = False
        runtime_error = False
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            try:
                res, pcov = curve_fit(
                    gaus, x_fit, y_fit, p0=[y_fit.max(), x[y.argmax()], std90(data)]
                )
                perr = np.sqrt(abs(np.diag(pcov)))
            except OptimizeWarning:
                print("OptimizeWarning in curve_fit for Force.")
                res = [np.nan, np.median(data), std90(data)]
                perr = [np.nan, np.nan, np.nan]
                optimze_warning = True
            except RuntimeError:
                # fit results probably not found, therefore create different version
                print("RuntimeError in curve_fit for Force.")
                res = [np.nan, np.median(data), std90(data)]
                perr = [np.nan, np.nan, np.nan]
                runtime_error = True

        # save fit data
        self.res_pedestal["g1_amplitude"] = res[0]
        self.res_pedestal["g1_mu"] = res[1]
        self.res_pedestal["g1_sigma"] = abs(res[2])
        self.res_pedestal["g1_amplitude_err"] = perr[0]
        self.res_pedestal["g1_mu_err"] = perr[1]
        self.res_pedestal["g1_sigma_err"] = perr[2]
        self.res_pedestal["hist_nbins"] = len(
            np.arange(
                data.min() - 5 * g_bin_width, data.max() + 5 * g_bin_width, g_bin_width
            )
        )

        self.res_pedestal["hist_bin_width"] = g_bin_width
        self.res_pedestal["hist_x"] = x
        self.res_pedestal["hist_y"] = y
        self.res_pedestal["hist_x_fit"] = x_fit
        self.res_pedestal["hist_y_fit"] = y_fit
        self.res_pedestal["curve_fit_optimize_warning"] = optimze_warning
        self.res_pedestal["curve_fit_runtime_error"] = runtime_error

        if self.smooth:
            data = data.apply(lambda x: (x - res[1]))

        return data

    def getPeaks(self, factor_prominence=4, factor_height=3, wlen=401):
        """
        Returns the peaks and other properties in the Force wf.

        Parameters
        ----------
        factor_prominence : float
            The minimum prominence is calculated by >mu + factor_prominence * sigma< of the second pedestal fit.

        factor_height : float
            The minimum height is calculated by >mu + factor_prominence * sigma< of the second pedestal fit.

        wlen : int
            Used to calculate peak prominences. See scipy's signal.peak_prominences

        """

        self._limit_low_h = (
            factor_height * self.res_pedestal["g1_sigma"]
        )  # the lower limit should be larger than the pedestal
        self._limit_low_p = (
            factor_prominence * self.res_pedestal["g1_sigma"]
        )  # the lower limit should be larger than the pedestal
        peak_indexes, props = signal.find_peaks(
            self.data, height=self._limit_low_h, prominence=self._limit_low_p, wlen=wlen
        )

        _tsampling = self.data.index[1] - self.data.index[0]
        _widths = []
        for _rh in [0.1, 0.5, 0.9, 0.97, 1]:
            _width = dict()
            _ws = signal.peak_widths(
                self.data,
                peak_indexes,
                rel_height=_rh,
                prominence_data=(
                    props["prominences"],
                    props["left_bases"],
                    props["right_bases"],
                ),
                wlen=wlen,
            )

            _width["width"] = _ws[0]
            _width["width_height"] = _ws[1]
            _width["left_ips"] = _ws[2]
            _width["right_ips"] = _ws[3]
            _width["rel_height"] = [int(round(_rh * 100)) for _ in range(len(_ws[0]))]
            _width["peaks"] = peak_indexes

            _widths.append(_width)

        props["peaks"] = peak_indexes
        peaks = pd.DataFrame(props)
        _convert_to_time = ["left_bases", "right_bases", "peaks"]
        peaks.loc[:, _convert_to_time] = peaks.loc[:, _convert_to_time].apply(
            lambda x: x * _tsampling
        )

        widths = pd.concat([pd.DataFrame(_w) for _w in _widths], ignore_index=True)
        _convert_to_time = ["width", "left_ips", "right_ips", "peaks"]
        widths.loc[:, _convert_to_time] = widths.loc[:, _convert_to_time].apply(
            lambda x: x * _tsampling
        )

        # bases values in peaks is not robust, exchange with widths data
        peaks.loc[:, ["left_bases", "right_bases"]] = widths.loc[
            widths.rel_height == 90, ["left_ips", "right_ips"]
        ].values
        peaks.insert(
            0,
            "width_height",
            widths.loc[widths.rel_height == 90, ["width_height"]].values,
        )

        peaks.loc[:, ["left_wings", "right_wings"]] = widths.loc[
            widths.rel_height == 10, ["left_ips", "right_ips"]
        ].values
        peaks.insert(
            0,
            "wing_height",
            widths.loc[widths.rel_height == 10, ["width_height"]].values,
        )

        # time to peak and time to relaxation
        _t2p = (
            widths.loc[widths.rel_height == 10, "left_ips"].to_numpy()
            - widths.loc[widths.rel_height == 90, "left_ips"].to_numpy()
        )
        peaks.insert(0, "t2p", _t2p)

        _t2r = (
            widths.loc[widths.rel_height == 90, "right_ips"].to_numpy()
            - widths.loc[widths.rel_height == 10, "right_ips"].to_numpy()
        )
        peaks.insert(0, "t2r", _t2r)

        return peaks, widths, peak_indexes


class MEA_channel:
    """
    Takes care of a single MEA channel.
    """

    def __init__(self, name, data, apply_triangle_filter=False, bin_width=None):
        self._box_width_default = (
            80  # Default value corresponding to 40ms for t_s = 500us
        )

        self.res_pedestal = dict()
        self.name = name
        self.data_raw = data
        self.apply_triangle_filter = apply_triangle_filter
        self.bin_width = bin_width
        self.data = self._prepare_data(self.data_raw)
        self.peaks, self.peak_props = self.getPeaks()

    def _calcTriganleBoxWidth(self):
        """
        The optimal box width at a sampling frequency of 500us is 80 and corresponds to 40ms.
        For a slower sampling frequency this is different.

        Returns
        -------
        int : Ideal box width for the Triangle convolution with respect to the sampling frequency.
        """
        t_real = self._box_width_default * 0.0005  # sampling time of 40ms
        t_sampling_data = self.data_raw.index[1] - self.data_raw.index[0]
        box_width = int(round(t_real / t_sampling_data, 0))
        return box_width

    def _prepare_data(self, data):
        if self.apply_triangle_filter:
            data = abs(data)  # The filter works better on positiv-only data.
            data = pd.Series(
                index=self.data_raw.index,
                data=self._conv_triang(data, self._calcTriganleBoxWidth()),
            )  # apply convolutional filter with a triangle function as kernel
        return self._correct_pedestal(data)

    def _correct_pedestal(self, data):
        """
        Do pedestal correction. On the filtered data.
        This is done in two steps:
            1. Make a histogram of the data and extract the most probable (pedestal) value.
            2. Apply pedestal correction.
        """
        if self.bin_width is None:
            g_bin_width = 0.05
            # g_bin_width = 100*0.0005  # seems to be a good binning width -> experimentally derived
        else:
            g_bin_width = self.bin_width
        x, y = histogr(
            data,
            bins=np.arange(
                data.min() - 5 * g_bin_width, data.max() + 5 * g_bin_width, g_bin_width
            ),
        )

        # prepare fit data
        # dbin = 3
        # y_fit = y[y.argmax() - dbin : y.argmax() + dbin + 1]
        # x_fit = x[y.argmax() - dbin : y.argmax() + dbin + 1]

        dbin = 5
        y_fit = y[y.argmax() - dbin : y.argmax() + dbin + 1]
        x_fit = x[y.argmax() - dbin : y.argmax() + dbin + 1]
        optimze_warning = False
        runtime_error = False
        try:
            # do fit and extract errors
            res, pcov = curve_fit(
                gaus,
                x_fit,
                y_fit,
                # p0=[y_fit.max(), x[y.argmax()], 1.0]
                p0=[y_fit.max(), x[y.argmax()], std90(data)],
            )
            perr = np.sqrt(abs(np.diag(pcov)))
        except OptimizeWarning as err:
            print("OptimizeWarning in MEA channel.")
            optimze_warning = True
        except RuntimeError as err:
            # fig = px.line(x=x, y=y, title=f'MEA - {self.name}')
            # fig.add_scatter(mode='markers', x=x_fit, y=y_fit, marker=dict(color='red'))
            # fig.write_html(os.path.join('analysis',f'debug_plot_histogram_Cha{self.name}.html'))
            # raise RuntimeError(f'{self.name} : ' + err.args[0])
            print("RUntime Error in MEA channel.")
            # res = [np.nan, np.median(data), std90(data)]
            res = [np.nan, np.median(data), abs(np.std(data))]
            perr = [np.nan, np.nan, np.nan]
            runtime_error = True
            # self.res_pedestal["g1_sigma"] = abs(np.std(data))
            # return data - self.res_pedestal["g1_sigma"]

        # save fit data
        self.res_pedestal["g1_amplitude"] = res[0]
        self.res_pedestal["g1_mu"] = res[1]
        self.res_pedestal["g1_sigma"] = abs(res[2])
        self.res_pedestal["g1_amplitude_err"] = perr[0]
        self.res_pedestal["g1_mu_err"] = perr[1]
        self.res_pedestal["g1_sigma_err"] = perr[2]

        self.res_pedestal["hist_nbins"] = len(
            np.arange(
                data.min() - 5 * g_bin_width, data.max() + 5 * g_bin_width, g_bin_width
            )
        )

        self.res_pedestal["hist_bin_width"] = g_bin_width
        self.res_pedestal["hist_x"] = x
        self.res_pedestal["hist_y"] = y
        self.res_pedestal["hist_x_fit"] = x_fit
        self.res_pedestal["hist_y_fit"] = y_fit
        self.res_pedestal["curve_fit_optimize_warning"] = optimze_warning
        self.res_pedestal["curve_fit_runtime_error"] = runtime_error

        #         data = data.apply(lambda x: x-self.res_pedestal['g1_mu'])
        data = data - self.res_pedestal["g1_mu"]

        return data

    def _conv_triang(self, y, box_pts):
        """
        Applies a convolutional filter to the given function.

        Parameters
        ----------
        y : np.array or pd.series
            Data to be filtered.

        box_pts : int
            Kernel size in samples.

        Returns
        -------
        np.array
        """
        box = signal.windows.triang(box_pts, sym=True)
        box = box / box.sum()
        y_smooth = np.convolve(y, box, mode="same")
        return y_smooth

    def _getPeakClusters(self, df_peaks, algorithm="kmeans", **kwargs):
        """

        Parameters
        ----------
        algorithm : str
            Options are
                - kmeans
                - dbscan

        kwargs
            Are the standard settings for the chosen algorithm
        """

        # this method returns the cluster labels
        labels = []

        # prepare data
        X = df_peaks.loc[:, ["amplitude_proc", "dt"]]
        #         X = StandardScaler().fit_transform(X.reshape(-1,1))
        X = StandardScaler().fit_transform(X)

        # DBSCAN
        if algorithm.find("dbscan") > -1:
            # use default values if no options are given
            if not kwargs:
                kwargs = dict(eps=0.5, min_samples=5)

            # Find the two different clusters (Na-Peak & T-Wave)
            db = DBSCAN(**kwargs).fit(X)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_  # noisy peaks have -1

        # KMEANS
        elif algorithm.find("kmeans") > -1:
            # use default values if no options are given
            if not kwargs:
                kwargs = dict(n_clusters=2, random_state=0)

            kmeans = KMeans(**kwargs).fit(X)
            labels = kmeans.labels_

        else:
            raise NotImplementedError(
                f'Your choice is not implemented. Choose "kmeans" or "dbscan".'
            )

        return labels

    def getPeaks(
        self, factor_prominence=8, distance_min=int(0.3 / 0.1), algorithm="kmeans"
    ):
        """
        Returns the peaks and other properties in the Field Potential wf.

        Parameters
        ----------
        factor_prominence : float
            The minimum prominence is calculated by >mu + factor_prominence * sigma< of the second pedestal fit.

        distance_min : int
            Minimum distance (>0) between two peaks given in samples.
        """
        # if self.apply_triangle_filter:
        #     factor_prominence = 3
        _limit_low = (
            factor_prominence * self.res_pedestal["g1_sigma"]
        )  # the lower limit should be larger than the pedestal
        peaks, props = signal.find_peaks(
            self.data,
            # distance=distance_min,
            # prominence=[_limit_low, np.inf]
            prominence=_limit_low,
        )

        # define DataFrame for easier handling of the data
        df_peaks = pd.DataFrame(
            data={
                "amplitude_raw": self.data_raw.iloc[peaks].to_list(),
                "amplitude_proc": self.data.iloc[peaks].to_list(),
                "time": self.data.iloc[peaks].index.to_list(),
                "iloc_sample": peaks,
            }
        )

        if not self.apply_triangle_filter:
            # separate NaPeak and TWave
            df_peaks.insert(0, "t2f", np.abs(df_peaks.time.diff(1)))
            df_peaks.insert(0, "t2b", np.abs(df_peaks.time.diff(-1)))

            _naPeaks = []
            _twaves = []
            for _i, _d in df_peaks.iterrows():
                try:
                    if (_d.t2f > _d.t2b) and (_d.t2f > df_peaks.loc[_i + 1].t2b):
                        _naPeaks.append(_i)
                    elif abs(2.5 * _d.t2f) < abs(_d.t2b):
                        _twaves.append(_i)
                except KeyError:
                    continue

            df_peaks.insert(0, "ptype", "none")
            df_peaks.loc[_naPeaks, "ptype"] = "na"
            df_peaks.loc[_twaves, "ptype"] = "twave"

            # this actions assume, that the algorithm always finds one NaPeak and one TWave
            df_peaks = (
                df_peaks.loc[df_peaks.ptype.isin(["na", "twave"])]
                .sort_values("time")
                .reset_index(drop=True)
            )
            df_peaks.loc[:, "t2f"] = np.abs(df_peaks.time.diff())
            df_peaks.loc[:, "t2b"] = np.abs(df_peaks.time.diff(-1))

        return df_peaks, props

    def plot(self, verbose=True, output_dir_path=None):
        fig = px.scatter(
            self.peaks,
            x="time",
            y="amplitude",
            color="cluster",
            labels={"time": "Time [s]", "amplitude": "Voltage [mV]"},
            marginal_y="rug",
            title="MEA - Data with Peaks",
            color_continuous_scale=px.colors.sequential.Rainbow,
            symbol="cluster",
        )
        fig.add_scatter(
            x=self.data.index, y=self.data, line=dict(color="black"), name="Conv Data"
        )
        fig.update_layout(showlegend=True)
        fig.update_traces(marker_size=10)
        fig.update(layout_coloraxis_showscale=False)

        #         fig = px.scatter(x=self.data.iloc[self.peaks].index, y=self.data.iloc[self.peaks], marginal_y='rug', title='MEA - Filtered Data with Peaks', opacity=.5,
        #                             color_discrete_sequence=['red',], labels={'x' : 'Time [s]', 'y' : 'Voltage [mV]'})
        #         fig.add_scatter(x=self.data.index, y=self.data, line=dict(color='black'))
        #         fig.update_layout(showlegend=False)

        if not isinstance(output_dir_path, type(None)):
            fig.write_html(
                os.path.join(output_dir_path, "mea_filtered_data_and_peaks.html")
            )

        if verbose:
            fig.show()


class MEA:
    """
    Takes care of all MEA channels.
    """

    def __init__(self, datas):
        """
        Parameters
        ----------
        datas : pd.DataFrame
            Searches for all FlexMea channels available in the data frame.
        """

        self.peaks_dt = 0
        self.data = []
        self.channels = []
        self.peaks = pd.DataFrame()

        # all possible channels of the FlexMea
        self._channel_names = [
            "A2",
            "A3",
            "A4",
            "A5",
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
            "C6",
            "D1",
            "D2",
            "D3",
            "D4",
            "D5",
            "D6",
            "E1",
            "E2",
            "E3",
            "E4",
            "E5",
            "E6",
            "F2",
            "F3",
            "F4",
            "F5",
        ]

        # preselect only channel available in the data set
        self._channel_names = [
            cha for cha in self._channel_names if cha in datas.columns
        ]
        self._prepare_datas(datas.loc[:, self._channel_names])

    def _prepare_datas(self, datas):
        """
        Initialized each channel with the MEA_channel-class which does the typical pedestal subtraction and peak extraction.
        """
        self.data = []
        _dpeaks = []
        for col in datas.columns:
            try:
                _cha = MEA_channel(col, datas[col])
                self.channels.append(col)

            except ValueError as e:
                # add logging here
                continue

            self.data.append(_cha)
            _cha.peaks.insert(0, "cha", _cha.name)  # add channel information
            _dpeaks.append(_cha.peaks)

        try:
            self.peaks = pd.concat(_dpeaks, ignore_index=True)
        except ValueError:
            self.peaks = None
            return

        # find dt Na-Peak and T-Wave
        g_bin_width = 0.03
        data = self.peaks.loc[self.peaks.ptype == "na", "t2b"]

        try:
            x, y = histogr(
                data,
                bins=np.arange(
                    data.min() - 5 * g_bin_width,
                    data.max() + 5 * g_bin_width,
                    g_bin_width,
                ),
            )
            dbin = 3
            y_fit = y[y.argmax() - dbin : y.argmax() + dbin]
            x_fit = x[y.argmax() - dbin : y.argmax() + dbin]

            res, pcov = curve_fit(
                gaus, x_fit, y_fit, p0=[y_fit.max(), np.median(y), np.std(y)]
            )
            perr = np.sqrt(abs(np.diag(pcov)))
        except ValueError:
            res = [np.nan, np.nan, np.nan]
            perr = [np.nan, np.nan, np.nan]
        except RuntimeError:
            res = [np.nan, np.median(y), np.std(y)]
            perr = [np.nan, np.nan, np.std(y) / np.sqrt(data.shape[0])]
        except:
            raise

        self.peaks_dt = dict(results=res, errors=perr)

        return

    def plot(self, mode="processed", verbose=True, output_dir_path=None):
        """
        Plots all channels of the desired data, 'processed' or 'raw' including the peaks.

        Parameters
        ----------
        mode : str, default 'processed'
            Options are 'processed' or 'raw'.

        verbose : bool, default True
            Show the plot or don't.

        output_dir_path : str
            Give any path to save the plot.

        """

        # measures of the mea channels
        _nrows = 6
        _ncols = 6

        _mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}

        fig = make_subplots(
            rows=_nrows,
            cols=_ncols,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.02,
            horizontal_spacing=0.02,
        )

        def _getPTypeColor(ptypes):
            def _mapper(x):
                if x.find("na") > -1:
                    return "#F50057"
                elif x.find("twave") > -1:
                    return "#D500F9"
                else:
                    return "#90CAF9"

            output = []
            for _ in ptypes.values:
                output.append(_mapper(_))

            return output

        if mode.find("processed") > -1:
            # processed data
            for _cha in self.data:
                _l = _cha.name[0].upper()
                _n = int(_cha.name[1])

                fig.add_trace(
                    go.Scatter(
                        x=_cha.data.index,
                        y=_cha.data,
                        name=f"{_l}{_n}",
                        line=dict(color="black"),
                    ),
                    row=_mapping[_l],
                    col=_n,
                )
                fig.add_trace(
                    go.Scatter(
                        x=_cha.peaks.time,
                        y=_cha.peaks.amplitude_proc,
                        marker_color=_getPTypeColor(_cha.peaks.ptype),
                        marker_colorscale=px.colors.qualitative.Light24,
                        mode="markers",
                        name="Peak",
                    ),
                    row=_mapping[_l],
                    col=_n,
                )
                fig.add_annotation(
                    text=f"{_l}{_n}",
                    xref="paper",
                    yref="paper",
                    x=0.0,
                    y=0.0,
                    showarrow=False,
                    row=_mapping[_l],
                    col=_n,
                    xanchor="right",
                    yanchor="bottom",
                )

        elif mode.find("raw") > -1:
            # raw data
            for _cha in self.data:
                _l = _cha.name[0].upper()
                _n = int(_cha.name[1])

                fig.add_trace(
                    go.Scatter(
                        x=_cha.data_raw.index,
                        y=_cha.data_raw,
                        name=f"{_l}{_n}",
                        line=dict(color="black"),
                    ),
                    row=_mapping[_l],
                    col=_n,
                )
                fig.add_trace(
                    go.Scatter(
                        x=_cha.peaks.time,
                        y=_cha.peaks.amplitude_raw,
                        marker_color=_getPTypeColor(_cha.peaks.ptype),
                        marker_colorscale=px.colors.qualitative.Light24,
                        mode="markers",
                        name="Peak",
                    ),
                    row=_mapping[_l],
                    col=_n,
                )
                fig.add_annotation(
                    text=f"{_l}{_n}",
                    xref="paper",
                    yref="paper",
                    x=0.0,
                    y=0.0,
                    showarrow=False,
                    row=_mapping[_l],
                    col=_n,
                    xanchor="right",
                    yanchor="bottom",
                )

        fig.update_layout(
            showlegend=False,
            title_text=f"{mode.title()} MEA Data - y axis in mV - x axis in seconds"
            #                             , width=1600
            ,
            height=800,
        )

        if verbose:
            fig.show()

        if not isinstance(output_dir_path, type(None)):
            fig.write_html(os.path.join(output_dir_path, f"plots_mea_{mode}.html"))


def plot_force_calcium(force, calc, verbose=True, mode="raw", output_dir_path=""):
    # def plot_force_calcium(df, s=None, df_calcium=None, df_calcium_peaks=None, verbose=True, save=False, mode='raw'):
    """
    Plots the Calcium and Force Data in the same figure including the peaks if given.
    The given data frames need to have an index with the time and a force/calc column.

    Parameters
    ----------
    df_{force, calcium} : pd.DataFrame or pd.Series
        DataFrame containing


    output_dir_path : str
        Give any path to save the plot.


        -----------------------------> TODO <-----------------------------------------------------
    """
    _marker_size = 12
    # plot force and calcium
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    # FORCE
    # raw
    fig.add_trace(
        go.Scatter(x=force.data_raw.index, y=force.data_raw, name="Force Raw"),
        row=1,
        col=1,
    )
    # filtered
    fig.add_trace(
        go.Scatter(x=force.data.index, y=force.data, name="Force Filt"), row=1, col=1
    )
    # peaks band
    # fig.add_hrect(y0=force.data.iloc[force.peaks.peaks].mean()-force.data.iloc[force.peaks.peaks].std(), y1=force.data.iloc[force.peaks.peaks].mean()+force.data.iloc[force.peaks.peaks].std(), line_width=0, opacity=0.2,  fillcolor='red', row=1, col=1)
    # peaks
    fig.add_trace(
        go.Scatter(
            x=force.peaks.peaks,
            y=force.peaks.peak_heights,
            marker=dict(opacity=0.5, size=_marker_size),
            mode="markers",
            name="Peak",
        ),
        row=1,
        col=1,
    )
    # left/right bases
    fig.add_trace(
        go.Scatter(
            x=force.peaks.left_bases,
            y=force.peaks.width_height,
            marker=dict(opacity=0.5, size=_marker_size),
            mode="markers",
            name="Left Base",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=force.peaks.right_bases,
            y=force.peaks.width_height,
            marker=dict(opacity=0.5, size=_marker_size),
            mode="markers",
            name="Right Base",
        ),
        row=1,
        col=1,
    )

    # CALCIUM
    # raw
    fig.add_trace(
        go.Scatter(x=calc.data_raw.index, y=calc.data_raw, name="Calcium Raw"),
        row=2,
        col=1,
    )
    # filtered
    fig.add_trace(
        go.Scatter(x=calc.data.index, y=calc.data, name="Calcium Filt"), row=2, col=1
    )
    # peaks band
    # fig.add_hrect(y0=calc.peaks.peaks].mean()-calc.data.iloc[calc.peaks.peaks].std(), y1=calc.data.iloc[calc.peaks.peaks].mean()+calc.data.iloc[calc.peaks.peaks].std(), line_width=0, opacity=0.2,  fillcolor='red', row=2, col=1)
    # peaks
    fig.add_trace(
        go.Scatter(
            x=calc.peaks.peaks,
            y=calc.peaks.peak_heights,
            marker=dict(opacity=0.5, size=_marker_size),
            mode="markers",
            name="Peak",
        ),
        row=2,
        col=1,
    )
    # left/right bases
    fig.add_trace(
        go.Scatter(
            x=calc.peaks.left_bases,
            y=calc.peaks.width_height,
            marker=dict(opacity=0.5, size=_marker_size),
            mode="markers",
            name="Left Base",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=calc.peaks.right_bases,
            y=calc.peaks.width_height,
            marker=dict(opacity=0.5, size=_marker_size),
            mode="markers",
            name="Right Base",
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Force [mN]", row=1, col=1)
    fig.update_yaxes(title_text="Intensity [a.u.]", row=2, col=1)
    fig.update_xaxes(title_text="Time [s]", row=2, col=1)

    if verbose:
        fig.show()

    if not isinstance(output_dir_path, type(None)):
        fig.write_html(
            os.path.join(output_dir_path, f"plots_{mode}_force_calcium.html")
        )

    # plot as pdf
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={"hspace": 0.2})

    _tmax = 10  # in seconds
    _lw = 1

    # force plot
    _data = force.data.loc[:_tmax]
    _peaks = force.peaks.loc[force.peaks.peaks < _tmax]
    ax = axs[0]

    ax.plot(_data.index, _data, lw=_lw, label="Data Filt", color="black", zorder=0.5)
    ax.plot(
        _data.index,
        force.data_raw.loc[:_tmax],
        lw=_lw,
        label="Data Raw",
        color="blue",
        zorder=0.1,
    )
    ax.plot(
        _peaks.peaks,
        _peaks.peak_heights,
        ls="None",
        marker=".",
        ms=12,
        alpha=0.5,
        color="red",
        label="Peaks",
    )

    for _l, _r, _h in zip(
        _peaks.left_bases.values, _peaks.right_bases.values, _peaks.width_height.values
    ):
        ax.plot([_l, _r], [_h, _h], ls="-", lw=_lw, marker="None")

    # _widths = force.peak_widths.loc[(force.peak_widths.peaks < _tmax) & (force.peak_widths.rel_height==10)]

    # calcium
    _data = calc.data.loc[:_tmax]
    _peaks = calc.peaks.loc[calc.peaks.peaks < _tmax]
    ax = axs[1]

    ax.plot(_data.index, _data, lw=_lw, label="Data Filt", color="black", zorder=0.5)
    ax.plot(
        _data.index,
        calc.data_raw.loc[:_tmax],
        lw=_lw,
        label="Data Raw",
        color="blue",
        zorder=0.1,
    )
    ax.plot(
        _peaks.peaks,
        _peaks.peak_heights,
        ls="None",
        marker=".",
        ms=12,
        alpha=0.5,
        color="red",
        label="Peaks",
    )

    for _l, _r, _h in zip(
        _peaks.left_bases.values, _peaks.right_bases.values, _peaks.width_height.values
    ):
        ax.plot([_l, _r], [_h, _h], ls="-", lw=_lw, marker="None")

    ax.set_xlabel("Time [s]")
    # ax.set_ylabel('Amplitude [mN]')

    fig.legend(loc="upper right", bbox_to_anchor=(1.1, 0.75))
    fig.savefig(os.path.join(output_dir_path, f"plot_peaks_force_calc.pdf"))
    plt.close("all")
