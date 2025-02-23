from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import tsfel
import os
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ArrythmiaGroundTruthGenerator:
    """
    Generates ground truth data for arrhythmia detection by extracting features from labeled cardiac signals.

    This class processes three types of cardiac signals:
    - Force measurements
    - Calcium signals
    - Field potential recordings

    For each signal type, it:
    1. Loads labeled peak data
    2. Extracts features using different time windows around peaks
    3. Saves processed features to CSV files organized by signal type and window size

    Attributes:
        WINDOW_SIZES_SECONDS (List[float]): Time windows used for feature extraction
        FORCE_SIGNAL_TYPE (str): Identifier for force measurements
        CALCIUM_SIGNAL_TYPE (str): Identifier for calcium signals
        FIELD_POTENTIAL_SIGNAL_TYPE (str): Identifier for field potential recordings
    """

    WINDOW_SIZES_SECONDS = [0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.3, 1.5]

    FORCE_SIGNAL_TYPE = "force"
    CALCIUM_SIGNAL_TYPE = "calcium"
    FIELD_POTENTIAL_SIGNAL_TYPE = "field_potential"

    def __init__(
        self,
        label_data_folder: str = "./Labeled/Data",
        preprocessed_data_folder: str = "./Preprocessed/HDFs",
        raw_data_folder: str = "./RawHDFs",
        output_folder: str = "./GroundTruth",
    ) -> None:
        """
        Initialize the ground truth generator.

        Args:
            label_data_folder: Directory containing labeled peak data (force.csv, calcium.csv, field_potential.csv)
            preprocessed_data_folder: Directory containing preprocessed HDF files
            raw_data_folder: Directory containing raw data HDF files
            output_folder: Directory where ground truth features will be saved
        """
        self.label_data_folder = label_data_folder
        self.preprocessed_data_folder = preprocessed_data_folder
        self.raw_data_folder = raw_data_folder
        self.output_folder = output_folder
        self._validate_data_folders()
        self._load_tfsel_config()
        self._load_data()

    def _validate_data_folders(self) -> None:
        """
        Validate existence of required input directories and files.

        Raises:
            AssertionError: If any required path does not exist
        """
        assert os.path.exists(
            self.label_data_folder
        ), f"The labeled data folder {self.label_data_folder} does not exist."
        assert os.path.exists(
            self.preprocessed_data_folder
        ), f"The preprocessed data folder {self.preprocessed_data_folder} does not exist."
        assert os.path.exists(
            self.raw_data_folder
        ), f"The raw data folder {self.raw_data_folder} does not exist."
        # check force.csv
        assert os.path.exists(
            os.path.join(self.label_data_folder, "force.csv")
        ), "The Force Labeled does not exist in the labeled data folder."
        # check calcium.csv
        assert os.path.exists(
            os.path.join(self.label_data_folder, "calcium.csv")
        ), "The Calcium Labeled does not exist in the labeled data folder."
        # check field_potential.csv
        assert os.path.exists(
            os.path.join(self.label_data_folder, "field_potential.csv")
        ), "The Field Potential Labeled does not exist in the labeled data folder."

        # create output folder if it does not exist
        os.makedirs(self.output_folder, exist_ok=True)

    def _load_tfsel_config(self) -> None:
        """
        Configure TSFEL feature extraction settings.

        Customizes statistical feature extraction parameters, particularly ECDF percentiles.
        """
        statistical_features_cfg = tsfel.get_features_by_domain("statistical")
        statistical_features_cfg["statistical"]["ECDF Percentile"]["parameters"][
            "percentile"
        ] = [
            0.1,
            0.2,
            0.4,
            0.6,
            0.8,
            0.9,
        ]
        self.statistical_features_cfg = statistical_features_cfg

    def _load_data(self):
        """
        Load labeled data for force, calcium, and field potential signals.
        """
        self._force = pd.read_csv(os.path.join(self.label_data_folder, "force.csv"))
        self._calcium = pd.read_csv(os.path.join(self.label_data_folder, "calcium.csv"))
        self._field_potential = pd.read_csv(
            os.path.join(self.label_data_folder, "field_potential.csv")
        )

        # Remove the rows with label -1 (Discarded cases)
        if not self._force.empty:
            self._force = self._force[self._force["label"] != -1]
        if not self._calcium.empty:
            self._calcium = self._calcium[self._calcium["label"] != -1]
        if not self._field_potential.empty:
            self._field_potential = self._field_potential[
                self._field_potential["label"] != -1
            ]
        self._data_dict = {}
        self._data_dict[self.FORCE_SIGNAL_TYPE] = self._force
        self._data_dict[self.CALCIUM_SIGNAL_TYPE] = self._calcium
        self._data_dict[self.FIELD_POTENTIAL_SIGNAL_TYPE] = self._field_potential

    def _check_if_window_record_exists(
        self,
        identifier: str,
        signal_type: str,
        window_size: float,
        force_peak_time: float,
        force_peak_idx: int,
    ) -> bool:
        """
        Check if features for a specific window have already been extracted.

        Args:
            identifier: Case identifier
            signal_type: Type of signal (force/calcium/field_potential)
            window_size: Window size in seconds
            force_peak_time: Time of the force peak
            force_peak_idx: Index of the force peak

        Returns:
            bool: True if features exist, False otherwise
        """
        dataset_dir = os.path.join(self.output_folder, signal_type)
        dataset_file_name = os.path.join(dataset_dir, f"window_{window_size}s.csv")
        if os.path.exists(dataset_file_name):
            dataset_df = pd.read_csv(dataset_file_name)
            if (
                dataset_df[
                    (dataset_df["identifier"] == identifier)
                    & (dataset_df["force_peak_time"] == force_peak_time)
                    & (dataset_df["force_peak_index"] == force_peak_idx)
                ].shape[0]
                > 0
            ):
                return True
        return False

    def _remove_interpolated_data(
        self,
        data_df: pd.DataFrame,
        signal_type: str,
        window_data: pd.DataFrame,
        identifier: str,
        window_start_time: float,
        window_end_time: float,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Remove interpolated values from preprocessed signal data with the help of raw data.
        Applicable only to Force and Calcium signal in presence of MEA as MEA has high sampling rate compared to force and calcium.

        Args:
            data_df: Preprocessed data
            signal_type: Type of signal
            window_data: Data within the current window
            identifier: Case identifier
            window_start_time: Start of window
            window_end_time: End of window

        Returns:
            Tuple containing:
            - DataFrame with interpolated data removed
            - Boolean indicating if window is out of bounds
        """
        data_column_name = self._get_data_column_name(signal_type)
        if "mea" in data_df.columns:
            raw_data_path = os.path.join(
                self.raw_data_folder, f"{identifier}__raw_data.hdf"
            )
            raw_df = pd.read_hdf(raw_data_path)
            non_null_idx = np.where(raw_df[data_column_name].notnull())[0]
            non_null_df = data_df.iloc[non_null_idx]

            if (
                window_start_time < non_null_df.index[0]
                or window_end_time > non_null_df.index[-1]
            ):
                logger.info(
                    f"Removing interpolated data for [{identifier}]: Window {window_start_time}--{window_end_time}s for {signal_type} is out of bounds. Skipping..."
                )
                return data_df, True
            window_data = non_null_df.loc[window_start_time:window_end_time]
            return window_data, False
        else:
            return data_df, False

    def _append_record_to_dataset(self, record_df, signal_type, window_size):
        """
        Append a record to the dataset for a given signal type and window size.

        Args:
            record_df: DataFrame containing the record to append
            signal_type: Type of signal
            window_size: Size of window in seconds
        """
        dataset_dir = os.path.join(self.output_folder, signal_type)
        dataset_file_name = os.path.join(dataset_dir, f"window_{window_size}s.csv")
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=True)
        if not os.path.exists(dataset_file_name):
            record_df.to_csv(dataset_file_name, index=False)
        else:
            with open(dataset_file_name, "a") as f:
                record_df.to_csv(f, header=False, index=False)

    def _get_data_column_name(self, signal_type: str) -> str:
        """
        Get the column name for the data in preprocessed hdf file of a given signal type.

        Args:
            signal_type: Type of signal

        Returns:
            str: Column name for the data
        """
        if signal_type == self.FIELD_POTENTIAL_SIGNAL_TYPE:
            return "mea"
        elif signal_type == self.CALCIUM_SIGNAL_TYPE:
            return "calc"
        else:
            return signal_type

    def _create_ground_truth_record_for_window(
        self,
        window_data: pd.DataFrame,
        data_df: pd.DataFrame,
        identifier: str,
        processed_hdf_path: str,
        signal_type: str,
        window_start_time: float,
        window_end_time: float,
        global_window_start_time: float,
        global_window_end_time: float,
        force_peak_time: float,
        force_peak_idx: int,
        case_peak_row: Dict,
        window_size: float,
    ) -> None:
        """
        Extract features from a signal window and save as ground truth.

        Process:
        1. Remove interpolated data if necessary
        2. Calculate sampling frequency
        3. Extract statistical features using TSFEL
        4. Format and save features to appropriate CSV file

        Features saved include:
        - Metadata (identifier, times, window parameters)
        - Statistical features (mean, std, percentiles, etc.)
        - Label (arrhythmic/normal)

        Args:
            window_data: Signal data within the window
            data_df: Complete preprocessed data
            identifier: Case identifier
            processed_hdf_path: Path to preprocessed HDF file
            signal_type: Type of signal being processed
            window_start_time: Start of feature window
            window_end_time: End of feature window
            global_window_start_time: Start of entire event
            global_window_end_time: End of entire event
            force_peak_time: Time of force peak
            force_peak_idx: Index of force peak
            case_peak_row: Dictionary containing peak information
            window_size: Size of window in seconds
        """
        data_column_name = self._get_data_column_name(signal_type)
        # Remove interpolated data for force and calcium
        if signal_type in [self.FORCE_SIGNAL_TYPE, self.CALCIUM_SIGNAL_TYPE]:
            window_data, out_of_bounds = self._remove_interpolated_data(
                data_df,
                signal_type,
                window_data,
                identifier,
                window_start_time,
                window_end_time,
            )
            if out_of_bounds:
                return

        fs = np.round(
            1
            / (
                window_data[data_column_name].index[1]
                - window_data[data_column_name].index[0]
            ),
            4,
        )
        base_record = {
            "identifier": identifier,
            "processed_hdf_path": processed_hdf_path,
            "window_start_time": window_start_time,
            "window_end_time": window_end_time,
            "event_start_time": global_window_start_time,
            "event_end_time": global_window_end_time,
            "force_peak_time": force_peak_time,
            "force_peak_index": force_peak_idx,
            "window_size[s]": window_size,
            "sampling_frequency[Hz]": fs,
            "label": case_peak_row["label"],
        }
        features = tsfel.time_series_features_extractor(
            self.statistical_features_cfg,
            window_data[data_column_name].values,
            fs,
            verbose=0,
        )
        # Remove the prefix "0_" from the feature names
        col_name = [col.replace("0_", "") for col in features.columns.to_list()]
        features.columns = col_name
        features_dict = features.to_dict(orient="records")[0]
        # Remove spaces and convert to lowercase for feature names
        features_dict = {
            f"{k.replace(' ', '_').lower()}": np.round(v, 3)
            for k, v in features_dict.items()
        }
        features_dict = dict(base_record, **features_dict)
        features_df = pd.DataFrame([features_dict])
        if not features_df.empty:
            logger.info(
                f"Storing ground truth record for {identifier} with window {window_size}s at {force_peak_time}s for {signal_type}."
            )
            self._append_record_to_dataset(features_df, signal_type, window_size)

    def _generate_ground_truth_for_signal_type(
        self, signal_type: str = FORCE_SIGNAL_TYPE
    ):
        """
        Generate ground truth features for a given signal type.

        Args:
            signal_type: Type of signal (force/calcium/field_potential)
        """
        label_data = self._data_dict[signal_type]
        case_identifiers = label_data["identifier"].unique()
        for identifier in case_identifiers:
            logger.info(f"Processing case {identifier} for {signal_type}.")
            processed_hdf_path = os.path.join(
                self.preprocessed_data_folder, f"{identifier}.hdf"
            )
            data_df = pd.read_hdf(processed_hdf_path)
            case_peaks = label_data[label_data["identifier"] == identifier]

            for case_peak_row in case_peaks.to_dict(orient="records"):
                # contraction event start and end time
                global_window_start_time = np.round(case_peak_row["start_time"], 4)
                global_window_end_time = np.round(case_peak_row["end_time"], 4)
                # force peak time and index
                force_peak_time = np.round(case_peak_row["force_peak_time"], 4)
                force_peak_idx = case_peak_row["force_peak_index"]

                logger.info(
                    f"Processing peak at index {force_peak_idx}: {force_peak_time} s for {signal_type}."
                )

                for window_size in self.WINDOW_SIZES_SECONDS:
                    exist = self._check_if_window_record_exists(
                        identifier,
                        signal_type,
                        window_size,
                        force_peak_time,
                        force_peak_idx,
                    )

                    if exist:
                        logger.info(
                            f"Record exists for window {window_size}s at {force_peak_time}s for {signal_type}. Continuing..."
                        )
                        continue

                    logger.info(
                        f"Processing window {window_size}s at {force_peak_time}s for {signal_type}."
                    )

                    # ground truth window start and end time
                    window_start_time = np.round(force_peak_time - window_size, 4)
                    window_end_time = np.round(force_peak_time + window_size, 4)

                    if (
                        window_start_time < global_window_start_time
                        or window_end_time > global_window_end_time
                    ):
                        logger.info(
                            f"Window {window_size}s at {force_peak_time}s for {signal_type} is out of bounds. Skipping..."
                        )
                        continue

                    window_data = data_df.loc[window_start_time:window_end_time]
                    self._create_ground_truth_record_for_window(
                        window_data,
                        data_df,
                        identifier,
                        processed_hdf_path,
                        signal_type,
                        window_start_time,
                        window_end_time,
                        global_window_start_time,
                        global_window_end_time,
                        force_peak_time,
                        force_peak_idx,
                        case_peak_row,
                        window_size,
                    )

    def generate(self) -> None:
        """
        Generate ground truth features for all signal types.

        Processes each signal type if data is available:
        1. Force measurements
        2. Calcium signals
        3. Field potential recordings

        Features are extracted using multiple window sizes and saved to separate CSV files.
        Progress is logged throughout the process.
        """
        if not self._force.empty:
            self._generate_ground_truth_for_signal_type(self.FORCE_SIGNAL_TYPE)
        if not self._calcium.empty:
            self._generate_ground_truth_for_signal_type(self.CALCIUM_SIGNAL_TYPE)
        if not self._field_potential.empty:
            self._generate_ground_truth_for_signal_type(
                self.FIELD_POTENTIAL_SIGNAL_TYPE
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_data_folder",
        type=str,
        default="./Labeled/Data",
        help="The folder containing the labeled data with peak information.",
    )
    parser.add_argument(
        "--preprocessed_data_folder",
        type=str,
        default="./Preprocessed/HDFs",
        help="The folder containing the preprocessed data.",
    )
    parser.add_argument(
        "--raw_data_folder",
        type=str,
        default="./RawHDFs",
        help="The folder containing the raw data hdf files.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./GroundTruth",
        help="""Directory for storing ground truth feature data, organized as follows:
        
        Structure:
        output_folder/
        ├── force/              # Force signal features
        │   ├── window_0.2s.csv # Features for 0.2s window
        │   ├── window_0.5s.csv # Features for 0.5s window
        │   └── window_1.0s.csv # Features for 1.0s window
        ├── calcium/            # Calcium signal features
        │   └── [similar window CSVs]
        └── field_potential/    # Field potential features
            └── [similar window CSVs]
            
        Each CSV contains:
        - Signal features extracted using TSFEL
        - Metadata (identifier, peak times, labels)
        - Sampling information
        - Window parameters""",
    )
    args = parser.parse_args()
    ground_truth_generator = ArrythmiaGroundTruthGenerator(
        args.label_data_folder,
        args.preprocessed_data_folder,
        args.raw_data_folder,
        args.output_folder,
    )
    ground_truth_generator.generate()
