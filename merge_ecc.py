import os
import sys
import json
import logging
from typing import List, Tuple, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import preprocessing
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from utils import series as series_utils
from utils import general as general_utils
from glob import glob
from scipy import signal
import tsfel
import joblib
import json
import argparse
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class ECCMerger:
    """
    Handles merging and preprocessing of ECC (Force, Calcium, MEA) data.

    This class processes raw HDF files containing force, calcium, and MEA (Multi-Electrode Array)
    measurements. It performs several key operations:
    1. Filters and validates input files
    2. Preprocesses force and calcium signals
    3. Classifies and filters MEA channels
    4. Merges all signals (Force, Calcium, Field Potential) with appropriate time alignment (for calcium)
    5. Generates visualization plots
    6. Saves processed data and peaks information

    Attributes:
        data_dir (str): Directory containing raw HDF files
        mea_channel_classifier_path (str): Path to trained MEA channel classifier
        error_metadata_file (str): JSON file containing error cases to exclude
        output_dir (str): Directory for output files
        mea_cases_filter (str): Filter string to identify MEA cases
        mea_good_channels_filter_params (dict): Parameters for MEA signal filtering
    """

    def __init__(
        self,
        data_dir: str = "../RawHDFs",
        mea_channel_classifier_path: str = "./mea-channel-classification/modeling/mea-channel-classification_2025-02-20_17:25:52_model.joblib",
        error_metadata_file: str = "./error_metadata.json",
        output_dir: str = "./Preprocessed",
        mea_cases_filter: str = "run1b",
        mea_good_channels_filter_params: Dict = dict(
            cutoff=180,
            order=5,
        ),
    ):
        """
        Initialize the ECCMerger with configuration parameters.

        Args:
            data_dir: Directory containing raw HDF files
            mea_channel_classifier_path: Path to the trained MEA channel classifier model
            error_metadata_file: JSON file listing cases to exclude
            output_dir: Directory for storing processed outputs
            mea_cases_filter: String to identify MEA cases in filenames
            mea_good_channels_filter_params: Dictionary with 'cutoff' and 'order' for Butterworth filter

        Raises:
            AssertionError: If required files or directories don't exist
        """
        logger.info("Initializing ECCMerger")

        # Initialize attributes
        self.data_dir = data_dir
        self.error_metadata_file = error_metadata_file
        self.output_dir = output_dir
        self.mea_channel_classifier_path = mea_channel_classifier_path
        self.mea_cases_filter = mea_cases_filter
        self.mea_good_channels_filter_params = mea_good_channels_filter_params

        # Validate paths
        self._validate_paths()

        # Create output directory structure
        self._create_output_directories()

        logger.info("ECCMerger initialized successfully")

    def _validate_paths(self) -> None:
        """Validate existence of required input files and directories."""
        assert os.path.exists(self.data_dir), "Data directory does not exist"
        assert os.path.exists(
            self.error_metadata_file
        ), "Error metadata file does not exist"
        assert os.path.exists(
            self.mea_channel_classifier_path
        ), "MEA channel classifier file does not exist"

    def _create_output_directories(self) -> None:
        """Create necessary output directory structure."""
        directories = [
            self.output_dir,
            os.path.join(self.output_dir, "Plots"),  # For preprocessed plots
            os.path.join(self.output_dir, "HDFs"),  # For preprocessed HDF files
            os.path.join(self.output_dir, "Peaks"),  # For peaks information
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Created directory: {directory}")

    def _read_valid_raw_hdf_paths(self) -> List[str]:
        """
        Get paths to valid HDF files, excluding known problematic cases.

        Returns:
            List of valid HDF file paths

        Raises:
            AssertionError: If no valid files are found
        """
        logger.info("Reading valid HDF file paths")

        all_cases = glob(os.path.join(self.data_dir, "**", "*.hdf"), recursive=True)
        errored_cases = json.load(open(self.error_metadata_file, "r"))

        # Filter out problematic cases
        valid_cases = [
            case.split("/")[-1]
            for case in all_cases
            if not any(
                case.split("/")[-1] in errored_cases[error_type]
                for error_type in [
                    "duplicates_to_be_removed",
                    "out_of_sync",
                    "unavailable_mea_files",
                ]
            )
        ]

        assert len(valid_cases) > 0, "No valid MEA files found"
        logger.info(f"Found {len(valid_cases)} valid cases")

        return [os.path.join(self.data_dir, case) for case in valid_cases]

    def _load_mea_channel_classifier_and_cfg_file(self):
        """
        Load the MEA channel classifier and the TSFEL configuration file.
        """
        self.classifier = joblib.load(self.mea_channel_classifier_path)
        self.cfg_file = tsfel.get_features_by_domain("statistical")
        self.cfg_file["statistical"]["ECDF Percentile"]["parameters"]["percentile"] = [
            0.1,
            0.2,
            0.4,
            0.6,
            0.8,
            0.9,
        ]

    def _average_filtered_good_mea_channels(self, df_raw):
        """
        Average filtered good MEA channels.

        Finds the good MEA channels by using the MEA channel classifier and TSFEL feature extraction.
        Apply a butterworth lowpass filter to the good MEA channels.
        Returns the average of the good MEA channels and the good MEA channels.

        Args:
            df_raw: Raw MEA data

        Returns:
            Tuple of average of the good MEA channels and the good MEA channels
        """
        mea = preprocessing.MEA(df_raw)
        fs = 1 / (df_raw.index[1] - df_raw.index[0])
        good_channels = []

        for i in range(len(mea.channels)):
            mea_selected_channel = mea.data[i]
            features = tsfel.time_series_features_extractor(
                self.cfg_file, mea_selected_channel.data_raw, fs=fs, verbose=0
            )
            col_name = [col.replace("0_", "") for col in features.columns.to_list()]
            features.columns = col_name
            predicted_class = self.classifier.predict(features)
            if predicted_class == 2:
                good_channels.append(mea_selected_channel.name)

        good_channels_df = df_raw.loc[:, good_channels]
        for col in good_channels_df.columns:
            (
                y,
                *_,
            ) = general_utils.filter_lowpass(
                good_channels_df[col],
                cutoff=self.mea_good_channels_filter_params["cutoff"],
                fs=fs,
                order=self.mea_good_channels_filter_params["order"],
                unormalized_cutoff=True,
            )
            good_channels_df[col] = pd.Series(y, index=df_raw[col].index)

        return good_channels_df.mean(axis=1), good_channels

    def _generate_and_save_case_plot(
        self, df_raw, df_merge, force_peaks, calc_peaks, good_channels, identifier
    ):
        """
        Generate and save a single case plot at the {output directory}/Plots directory.
        1. Force Raw vs Preprocessed
        2. Force Preprocessed Peaks
        3. Calcium Raw vs Preprocessed
        4. Calcium Preprocessed Peaks
        5. MEA Preprocessed

        Args:
            df_raw: Raw data
            df_merge: Merged data
            force_peaks: Force peaks
            calc_peaks: Calcium peaks
            good_channels: Good MEA channels
            identifier: Case identifier

        """
        row_count = 4
        if "mea" in df_merge.columns:
            row_count += 1

        fig = make_subplots(
            rows=row_count,
            cols=1,
            shared_xaxes=True,
            specs=[[{"secondary_y": True}]] * row_count,
            subplot_titles=[
                "Force Raw vs Preprocessed",
                "Force Preprocessed Peaks",
                "Calcium Raw vs Preprocessed",
                "Calcium Preprocessed Peaks",
                (
                    f"MEA ({self.mea_good_channels_filter_params['cutoff']} Hz, {self.mea_good_channels_filter_params['order']}th Order) [{', '.join(good_channels)}]"
                    if "mea" in df_merge.columns
                    else "Force"
                ),
                "Calcium",
            ],
        )

        # Force Plot
        row = 1
        # Raw Force
        fig.add_trace(
            go.Scatter(
                x=df_raw.force.dropna().index, y=df_raw.force.dropna(), name="Force Raw"
            ),
            row=row,
            col=1,
            secondary_y=False,
        )
        # Preprocessed Force
        fig.add_trace(
            go.Scatter(
                x=df_merge.index,
                y=df_merge.force,
                name="Force Preprocessed",
            ),
            row=row,
            col=1,
            secondary_y=True,
        )
        # Preprocessed Force Peaks
        row += 1
        fig.add_trace(
            go.Scatter(
                x=df_merge.index,
                y=df_merge.force,
                name="Force Preprocessed",
            ),
            row=row,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=df_merge.iloc[force_peaks].index,
                y=df_merge.iloc[force_peaks].force,
                mode="markers",
                name="Force Peaks",
            ),
            row=row,
            col=1,
            secondary_y=False,
        )

        # Calcium Plot
        row += 1
        fig.add_trace(
            go.Scatter(
                x=df_raw.calc.dropna().index, y=df_raw.calc.dropna(), name="Calcium Raw"
            ),
            row=row,
            col=1,
            secondary_y=False,
        )
        # Preprocessed Calcium
        fig.add_trace(
            go.Scatter(x=df_merge.index, y=df_merge.calc, name="Calcium Preprocessed"),
            row=row,
            col=1,
            secondary_y=True,
        )
        # Preprocessed Calcium Peaks
        row += 1
        fig.add_trace(
            go.Scatter(x=df_merge.index, y=df_merge.calc, name="Calcium Preprocessed"),
            row=row,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=df_merge.iloc[calc_peaks].index,
                y=df_merge.iloc[calc_peaks].calc,
                mode="markers",
                name="Calcium Peaks",
                marker=dict(color="red", size=10),
            ),
            row=row,
            col=1,
            secondary_y=False,
        )

        if "mea" in df_merge.columns:
            row += 1
            fig.add_trace(
                go.Scatter(
                    x=df_merge.index,
                    y=df_merge.mea,
                    name=f"MEA Preprocessed",
                ),
                row=row,
                col=1,
                secondary_y=False,
            )
        title_text = (
            f"Force and Calcium {identifier}"
            if "mea" not in df_merge.columns
            else f"Force, Calcium & Field Potential {identifier}"
        )
        fig.update_layout(
            autosize=False, width=2300, height=1000, title_text=title_text
        )
        fig.write_html(os.path.join(self.output_dir, "Plots", f"{identifier}.html"))

    def _save_preprocessed_data_and_peaks_info(
        self, df_merge, force_peaks, calc_peaks, identifier
    ):
        """
        Save the preprocessed data and peaks information at the {output directory}/HDFs and {output directory}/Peaks directories.

        Args:
            df_merge: Merged data
            force_peaks: Force peaks
            identifier: Case identifier
        """
        df_merge.to_hdf(
            os.path.join(self.output_dir, "HDFs", f"{identifier}.hdf"),
            key="preprocessed_data",
        )
        peaks_info = {
            "force_peaks_indexes": force_peaks.tolist(),
            "calc_peaks_indexes": calc_peaks.tolist(),
        }
        json.dump(
            peaks_info,
            open(os.path.join(self.output_dir, "Peaks", f"{identifier}.json"), "w"),
            cls=series_utils.NumpyEncoder,
        )

    def _remap_calc_peaks(self, df_merge, calc_peaks, calc_nans_index_mapping=None):
        calc_data = df_merge.calc.copy()
        corr_factor = 1 + 0.0028388903486529877  # in sec per recorded sec
        _tmax = calc_data.index[-1]
        calc_data.index = np.round(calc_data.index.to_numpy() * corr_factor, decimals=2)
        calc_data = calc_data.loc[:_tmax]
        if calc_nans_index_mapping is not None:
            calc_remap_peak_indexes = calc_data.iloc[
                calc_nans_index_mapping[calc_peaks]
            ].index
        else:
            calc_remap_peak_indexes = calc_data.iloc[calc_peaks].index
        calc_remap_peak_indexes = np.where(
            df_merge.calc.index.isin(calc_remap_peak_indexes)
        )[0]
        calc_peaks = calc_remap_peak_indexes
        return calc_peaks

    def _process_raw_file(self, hdf_file):
        df_raw = pd.read_hdf(hdf_file, key="raw_data")

        # Drop Nulls
        force = df_raw.force.dropna()
        calc = df_raw.calc.dropna()

        force_raw_normalized = series_utils.normalize_series(force)
        # getting bin width equivalent of 0.01 on raw force to normalized force -> works better practically for
        # guassian curve fitting around force histogram maximum
        normalized_bin_width = np.round(
            0.01
            * (
                (np.max(force) - np.min(force))
                / (np.max(force_raw_normalized) - np.min(force_raw_normalized))
            ),
            3,
        )

        # First Curve Fit for force and calcium
        # This version smoothes the signal and do the pedestal correction
        # For the force, savgol filtering is also applied
        # Force utilizes the bin width calculated above
        first_force_curve_fit = preprocessing.Force(
            force.rename("force"), bin_width=normalized_bin_width
        )
        first_calc_curve_fit = preprocessing.Calcium(calc.rename("calc"))

        # Second Curve Fit for force and calcium using the first curve fit data as source
        # This version does not smooth the signal and does not do the pedestal correction as
        # the first curve fit already did that (given as input to the second curve fit), force
        # savgol filtering is also not applied
        # The purpose of this curve fit is to update the mean and std of the guassian curve fit
        # that will be utilized for the peak prominence threshold calculation
        second_force_curve_fit = preprocessing.Force(
            first_force_curve_fit.data.rename("force"),
            smooth=False,
            bin_width=normalized_bin_width,
        )
        second_calc_curve_fit = preprocessing.Calcium(
            first_calc_curve_fit.data.rename("calc"), smooth=False
        )

        # Selecting the fit data that is able to detect peaks
        force_data, calc_data = second_force_curve_fit.data, second_calc_curve_fit.data
        force_peaks, calc_peaks = (
            second_force_curve_fit.peak_indexes,
            second_calc_curve_fit.peak_indexes,
        )

        if len(force_peaks) == 0 and len(first_force_curve_fit.peak_indexes) > 0:
            # If the second curve fit is not able to detect peaks, then
            # use the first curve fit version for force data and peaks
            force_data = first_force_curve_fit.data
            force_peaks = first_force_curve_fit.peak_indexes
        if len(calc_peaks) == 0 and len(first_calc_curve_fit.peak_indexes) > 0:
            # same as above for calcium
            calc_data = first_calc_curve_fit.data
            calc_peaks = first_calc_curve_fit.peak_indexes

        df_merge = pd.merge(
            force_data.rename("force"),
            calc_data.rename("calc"),
            how="left",
            left_index=True,
            right_index=True,
        )
        good_channels = None
        # Add MEA if it is not filtered out
        if self.mea_cases_filter in hdf_file:
            mea_avg_filtered_good_channels, good_channels = (
                self._average_filtered_good_mea_channels(df_raw)
            )

            df_merge = pd.merge(
                df_merge,
                mea_avg_filtered_good_channels.rename("mea"),
                how="outer",
                left_index=True,
                right_index=True,
            )
            # Fill NaNs with linear interpolation for force and calc as MEA has higher resolution 2KHz > 100 Hz
            df_merge = df_merge.interpolate("linear")

            # force peaks were determined on Non Null data and we need to remap them to the new index with higher resolution (due to MEA) having NaNs
            force_nans_index_mapping = series_utils.remap_indexes_after_nan_removal(
                df_raw.force
            )
            calc_nans_index_mapping = series_utils.remap_indexes_after_nan_removal(
                df_raw.calc
            )
            force_peaks = force_nans_index_mapping[force_peaks]
            calc_peaks = self._remap_calc_peaks(
                df_merge, calc_peaks, calc_nans_index_mapping
            )
        else:
            # Fill NaNs with linear interpolation for Calcium with undefined values after time correction
            df_merge = df_merge.interpolate("linear")
            calc_peaks = self._remap_calc_peaks(df_merge, calc_peaks)
        identifier = hdf_file.split("/")[-1].replace("__raw_data.hdf", "")
        self._generate_and_save_case_plot(
            df_raw, df_merge, force_peaks, calc_peaks, good_channels, identifier
        )
        self._save_preprocessed_data_and_peaks_info(
            df_merge, force_peaks, calc_peaks, identifier
        )

    def execute(self) -> None:
        """
        Execute the complete ECC merging pipeline.

        This method orchestrates the entire processing pipeline:
        1. Identifies valid input files
        2. Loads the MEA channel classifier
        3. Processes each file individually

        Any errors during processing are logged but don't stop the pipeline.
        """
        logger.info("Starting ECC merge pipeline")

        try:
            hdf_files = self._read_valid_raw_hdf_paths()

            # error_unaligned_mea_files = [
            #     "./RawHDFs/run1b_ca_titration__1.13b_B19__01_0.1_ca++__Take2__raw_data.hdf",
            #     "./RawHDFs/run1b_e-4031__1.20b_B01__05_300nM__Take3__raw_data.hdf",
            #     "./RawHDFs/run1b_ca_titration__1.13b_B19__09_4.0_ca++__Take2__raw_data.hdf",
            #     "./RawHDFs/run1b_nifedipine__1.30b_B03__04_0.2uM__Take3__raw_data.hdf",
            #     "./RawHDFs/run1b_ca_titration__1.20b_B01__06_1.0_ca++__Take2__raw_data.hdf",
            #     "./RawHDFs/run1b_nifedipine__1.30b_B03__02_0.03uM__Take3__raw_data.hdf",
            #     "./RawHDFs/run1b_e-4031__1.17b_B02__05_300nM__Take3__raw_data.hdf",
            #     "./RawHDFs/run1b_nifedipine__1.30b_B03__08_4.0uM__Take3__raw_data.hdf",
            #     "./RawHDFs/run1b_nifedipine__1.28b_B02__02_0.03uM__Take3__raw_data.hdf",
            # ]

            self._load_mea_channel_classifier_and_cfg_file()

            for hdf_file in hdf_files:
                # if hdf_file not in error_unaligned_mea_files:
                #     continue
                logger.info(f"Processing file: {hdf_file}")
                self._process_raw_file(hdf_file)
                logger.info(f"Successfully processed: {hdf_file}")

            logger.info("ECC merge pipeline completed successfully")

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./RawHDFs",
        help="Path to the directory containing raw HDF files",
    )
    parser.add_argument(
        "--mea_channel_classifier_path",
        type=str,
        default="./MeaChannelClassification/modeling/mea-channel-classification_2025-03-04_01:59:51_model.joblib",
        help="Path to the MEA channel classifier file",
    )
    parser.add_argument(
        "--error_metadata_file",
        type=str,
        default="./error_metadata.json",
        help="Path to the error metadata file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./Preprocessed",
        help="Path to the output directory, defaults to ./Preprocessed. It will create subdirectories for Plots, HDFs, and Peaks",
    )
    parser.add_argument(
        "--mea_cases_filter",
        type=str,
        default="run1b",
        help="Filter for MEA cases, defaults to run1b. Will only process MEA for files with this string in the path",
    )
    parser.add_argument(
        "--mea_good_channels_filter_params",
        type=json.loads,
        default=dict(cutoff=180, order=5),
        help="Parameters for the MEA good channels butterworth lowpass filter (Default: cutoff=180, order=5)",
    )
    args = parser.parse_args()
    merger = ECCMerger(
        args.data_dir,
        args.mea_channel_classifier_path,
        args.error_metadata_file,
        args.output_dir,
        args.mea_cases_filter,
    )
    merger.execute()
