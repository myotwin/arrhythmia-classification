import pandas as pd
import numpy as np
import os
import json
import matplotlib

matplotlib.use("TkAgg")  # Add this line before importing pyplot
import matplotlib.pyplot as plt
import logging
import argparse
from typing import List, Tuple
from glob import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ArrythmiaLabeler:
    """
    Handles interactive labeling of arrhythmic and non-arrhythmic cardiac data.

    This class provides functionality to:
    1. Load and display preprocessed cardiac data (Force, Calcium, and MEA signals)
    2. Present an interactive interface for labeling peaks
    3. Save labeled data for further analysis
    4. Track labeling progress across multiple cases

    Class Constants:
        COMPLETED: Status for fully labeled cases
        SKIPPED: Status for intentionally skipped cases
        STARTED: Status for cases in progress
        NO_PEAKS_FOUND: Status for cases where no peaks were detected

    Attributes:
        preprocessed_dir (str): Directory containing preprocessed HDFs and Peaks (Force) folders
        error_metadata_file (str): Path to error tracking metadata
        output_dir (str): Directory for saving labeled data
        window_size_seconds (float): Time window around peaks for detailed view
        metadata_file_path (str): Path to labeling progress metadata

    This discards first and last peak because most often they are incomplete.
    """

    COMPLETED = "completed"
    SKIPPED = "skipped"
    STARTED = "started"
    NO_PEAKS_FOUND = "no_peaks_found"

    def __init__(
        self,
        preprocessed_dir: str = "./Preprocessed",
        raw_dir: str = "./RawHDFs",
        error_metadata_file: str = "./error_metadata.json",
        output_dir: str = "./Labeled",
        window_size_seconds: float = 1.5,
    ):
        logger.info("Initializing ArrythmiaLabeler")
        self.preprocessed_data_dir = os.path.join(preprocessed_dir, "HDFs")
        self.preprocessed_peaks_dir = os.path.join(preprocessed_dir, "Peaks")
        self.error_metadata_file = error_metadata_file
        self.output_dir = output_dir
        self.raw_dir = raw_dir
        self.window_size_seconds = window_size_seconds

        # Validate paths
        self._validate_paths()

        # Create output directory structure
        self._create_output_directories()

        logger.info("ArrythmiaLabeler initialized successfully")

    def _validate_paths(self) -> None:
        """
        Validate existence of required input files and directories.

        Raises:
            AssertionError: If any required path does not exist
        """
        assert os.path.exists(
            self.raw_dir,
        ), "Raw data directory does not exist"
        assert os.path.exists(
            self.preprocessed_data_dir
        ), "Preprocessed data directory does not exist"
        assert os.path.exists(
            self.preprocessed_peaks_dir
        ), "Preprocessed peaks directory does not exist"
        assert os.path.exists(
            self.error_metadata_file
        ), "Error metadata file does not exist"

    def _create_output_directories(self) -> None:
        """
        Create necessary output directory structure.

        Creates:
            - Main output directory
            - Data subdirectory for labeled data storage
            - Metadata file for tracking labeling progress
        """
        directories = [
            self.output_dir,
            os.path.join(self.output_dir, "Data"),  # For storage of labeled data
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        self.metadata_file_path = os.path.join(self.output_dir, "metadata.json")

    def _read_processed_hdf_paths(self) -> List[str]:
        """Read paths to processed HDF files."""
        return glob(
            os.path.join(self.preprocessed_data_dir, "**", "*.hdf"), recursive=True
        )

    def _check_processing_status(self, case_identifier: str) -> bool:
        """Check if the case has been processed."""
        if not os.path.exists(self.metadata_file_path):
            return False
        with open(self.metadata_file_path, "r") as file:
            metadata = json.load(file)
            if case_identifier in metadata:
                return metadata[case_identifier]
            return False

    def _take_user_input(self) -> str:
        """
        Present labeling options and collect user input.

        Valid inputs:
            Y: Mark current peak as arrhythmic
            N: Mark current peak as normal
            A: Mark all remaining peaks as arrhythmic
            P: Mark all remaining peaks as normal
            D: Discard current peak
            S: Skip entire case

        Returns:
            str: User's validated input choice
        """
        VALID_INPUTS = {
            "Y": "Current peak is arrhythmic",
            "N": "Current peak is normal",
            "A": "All remaining peaks are arrhythmic",
            "P": "All remaining peaks are normal",
            "D": "Discard current peak",
            "S": "Skip entire case",
        }

        # Create formatted prompt
        prompt = "\nPeak Labeling Options:\n"
        for key, description in VALID_INPUTS.items():
            prompt += f"  [{key}] {description}\n"
        prompt += "\nEnter choice: "

        while True:
            label = input(prompt).upper()
            if label in VALID_INPUTS:
                return label
            logger.warning(
                f"Invalid input '{label}'. Please enter one of: {', '.join(VALID_INPUTS.keys())}"
            )

    def _update_processing_status(self, case_identifier: str, status: str) -> None:
        """
        Update the processing status for a case in metadata.

        Args:
            case_identifier (str): Unique identifier for the case
            status (str): New status to set (one of class status constants)

        Creates metadata file if it doesn't exist.
        """
        if not os.path.exists(self.metadata_file_path):
            with open(self.metadata_file_path, "w") as file:
                json.dump({}, file)
        with open(self.metadata_file_path, "r") as file:
            metadata = json.load(file)
            metadata[case_identifier] = status
        with open(self.metadata_file_path, "w") as file:
            json.dump(metadata, file)

    def _delete_existing_record_for_case(
        self, data_list: List[List[str]], list_label: str
    ) -> None:
        """
        Delete existing record for a case in a CSV file. Used to discard partially labeled cases.

        Args:
            data_list (List[List[str]]): List of labeled data rows
            list_label (str): Label of the data type (e.g., "force", "calcium", "field_potential")
        """
        if os.path.exists(os.path.join(self.output_dir, "Data", f"{list_label}.csv")):
            with open(
                os.path.join(self.output_dir, "Data", f"{list_label}.csv"), "r"
            ) as f:
                lines = f.readlines()
                identifiers = [line.split(",")[0] for line in lines]
                identifiers = identifiers[1:]
            if data_list[0][0] in identifiers:
                lines = [
                    line for line in lines if data_list[0][0] != line.split(",")[0]
                ]
            with open(
                os.path.join(self.output_dir, "Data", f"{list_label}.csv"), "w"
            ) as f:
                for line in lines:
                    f.write(line)

    def _save_labeled_data(
        self,
        force_label_rows: List[List[str]],
        calc_label_rows: List[List[str]],
        mea_label_rows: List[List[str]],
        mea: bool,
    ) -> None:
        """
        Save labeled data to CSV files.

        Args:
            force_label_rows: List of labeled force measurements
            calc_label_rows: List of labeled calcium measurements
            mea_label_rows: List of labeled MEA measurements
            mea (bool): Whether MEA data is present

        Each row contains:
            - case identifier
            - force peak index
            - window start time
            - window end time
            - force peak time
            - label (-1: discarded, 0: normal, 1: arrhythmic)
        """
        data_lists = [force_label_rows, calc_label_rows, mea_label_rows]
        list_labels = ["force", "calcium", "field_potential"]
        for data_list, list_label in zip(data_lists, list_labels):
            if list_label == "field_potential" and not mea:
                continue
            if not os.path.exists(
                os.path.join(self.output_dir, "Data", f"{list_label}.csv")
            ):
                with open(
                    os.path.join(self.output_dir, "Data", f"{list_label}.csv"), "w"
                ) as file:
                    file.write(
                        "identifier,force_peak_index,start_time,end_time,force_peak_time,label\n"
                    )
            self._delete_existing_record_for_case(data_list, list_label)
            with open(
                os.path.join(self.output_dir, "Data", f"{list_label}.csv"), "a"
            ) as file:
                for record in data_list:
                    file.write(",".join([str(x) for x in record]) + "\n")

    def _label_case(self, case_identifier: str) -> Tuple[bool, bool]:
        """
        Handle labeling process for a single case.

        Displays:
            - Full signal plots (Force, Calcium, MEA if available)
            - Detailed views around each detected force peak
            - Command line based labeling interface

        Args:
            case_identifier (str): Unique identifier for the case

        Returns:
            Tuple[bool, bool]: (case_skipped, peaks_found)
                case_skipped: Whether user chose to skip this case
                peaks_found: Whether any peaks were successfully labeled
        """
        preprocessed_data = pd.read_hdf(
            os.path.join(self.preprocessed_data_dir, f"{case_identifier}.hdf")
        )
        raw_data = pd.read_hdf(
            os.path.join(self.raw_dir, f"{case_identifier}__raw_data.hdf")
        )
        preprocessed_peaks_path = os.path.join(
            self.preprocessed_peaks_dir, f"{case_identifier}.json"
        )
        force_peaks_idx = json.load(open(preprocessed_peaks_path, "r"))[
            "force_peaks_indexes"
        ][
            1:-1
        ]  # skipping first and last peak
        force_label_rows, calc_label_rows, mea_label_rows = [], [], []

        all_arrhythmic, all_normal, case_skipped = False, False, False
        for i, force_peak_idx in enumerate(force_peaks_idx):
            logger.info(
                f"Labeling force peak {i} of {len(force_peaks_idx)} at index {force_peak_idx} for case {case_identifier}"
            )
            plt.ion()

            subplot_rows = 6 if "mea" in preprocessed_data.columns else 4
            fig, ax = plt.subplots(subplot_rows, 1, figsize=(20, 20))
            # Adjust spacing between subplots and title
            plt.subplots_adjust(
                hspace=0.5,  # Increase vertical space between subplots (default is 0.2)
                top=0.95,  # Leave less space at the top (default is 0.9)
                bottom=0.05,  # Reduce space at the bottom (default is 0.1)
            )

            # Force
            row = 0
            ax[row].plot(
                preprocessed_data.loc[: raw_data.force.dropna().index[-1]].index,
                preprocessed_data.loc[: raw_data.force.dropna().index[-1]].force,
            )
            ax[row].plot(
                preprocessed_data.iloc[[force_peak_idx]].index,
                preprocessed_data.iloc[[force_peak_idx]].force,
                "ro",
            )
            ax[row].set_title(f"Force")
            ax[row].set_xlabel("Time (s)")
            ax[row].set_ylabel("Force (mN)")

            # Calcium
            row += 1
            ax[row].plot(
                preprocessed_data.loc[: raw_data.calc.dropna().index[-1]].index,
                preprocessed_data.loc[: raw_data.calc.dropna().index[-1]].calc,
            )
            ax[row].set_title(f"Calcium")
            ax[row].set_xlabel("Time (s)")
            ax[row].set_ylabel("Calcium (a.u.)")

            # Field Potential
            if "mea" in preprocessed_data.columns:
                row += 1
                ax[row].plot(preprocessed_data.index, preprocessed_data.mea)
                ax[row].set_title(f"Field Potential")
                ax[row].set_xlabel("Time (s)")
                ax[row].set_ylabel("Field Potential (mV)")

            force_time_idx = preprocessed_data.iloc[[force_peak_idx]].index[0]
            window_start = np.maximum(
                np.round(force_time_idx - self.window_size_seconds, 4),
                preprocessed_data.force.dropna().index[0],
            )
            window_end = np.minimum(
                np.round(force_time_idx + self.window_size_seconds, 4),
                preprocessed_data.force.dropna().index[-1],
            )

            row += 1
            ax[row].plot(
                preprocessed_data.loc[window_start:window_end].index,
                preprocessed_data.loc[window_start:window_end].force,
            )
            ax[row].plot(
                preprocessed_data.iloc[[force_peak_idx]].index,
                preprocessed_data.iloc[[force_peak_idx]].force,
                "ro",
            )
            ax[row].set_title(f"Force Around Peak")
            ax[row].set_xlabel("Time (s)")
            ax[row].set_ylabel("Force (mN)")

            row += 1
            ax[row].plot(
                preprocessed_data.loc[window_start:window_end].index,
                preprocessed_data.loc[window_start:window_end].calc,
            )
            ax[row].set_title(f"Calcium Around Force Peak")
            ax[row].set_xlabel("Time (s)")
            ax[row].set_ylabel("Calcium (a.u.)")

            # Field Potential
            if "mea" in preprocessed_data.columns:
                row += 1
                ax[row].plot(
                    preprocessed_data.loc[window_start:window_end].index,
                    preprocessed_data.loc[window_start:window_end].mea,
                )
                ax[row].set_title(f"Field Potential Around Force Peak")
                ax[row].set_xlabel("Time (s)")
                ax[row].set_ylabel("Field Potential (mV)")

            fig.suptitle(
                f"Labeling - Force Peak at {force_peak_idx}: {np.round(force_time_idx, 4)} for {case_identifier}",
                y=0.99,  # Move title closer to top (default is 0.98)
                fontsize=14,
                fontweight="bold",
            )
            plt.show()
            plt.pause(0.05)
            user_input = self._take_user_input()
            plt.close("all")

            if user_input == "S":
                case_skipped = True
                break

            if user_input in ["Y", "N", "D"]:
                record = [
                    case_identifier,
                    force_peak_idx,
                    window_start,
                    window_end,
                    force_time_idx,
                ]
                # Discard peak
                if user_input == "D":
                    label = -1
                # Mark as arrhythmic or normal
                else:
                    label = 1 if user_input == "Y" else 0
                record.append(label)

                force_label_rows.append(record)
                calc_label_rows.append(record)
                if "mea" in preprocessed_data.columns:
                    mea_label_rows.append(record)

            # Mark all remaining peaks as arrhythmic
            if user_input == "A":
                all_arrhythmic = True
                break
            # Mark all remaining peaks as normal
            if user_input == "P":
                all_normal = True
                break

        if not case_skipped and (all_arrhythmic or all_normal):
            # Mark all remaining peaks as arrhythmic
            if all_arrhythmic:
                label = 1
            # Mark all remaining peaks as normal
            else:
                label = 0

            # Mark all remaining peaks with the same label defined above
            for i in range(len(force_label_rows), len(force_peaks_idx)):
                force_peak_idx = force_peaks_idx[i]
                force_time_idx = preprocessed_data.iloc[[force_peak_idx]].index[0]
                window_start = np.maximum(
                    np.round(force_time_idx - self.window_size_seconds, 4),
                    preprocessed_data.index[0],
                )
                window_end = np.minimum(
                    np.round(force_time_idx + self.window_size_seconds, 4),
                    preprocessed_data.index[-1],
                )
                record = [
                    case_identifier,
                    force_peak_idx,
                    window_start,
                    window_end,
                    force_time_idx,
                    label,
                ]
                force_label_rows.append(record)
                calc_label_rows.append(record)
                if "mea" in preprocessed_data.columns:
                    mea_label_rows.append(record)

        # Save labeled data
        if not case_skipped and len(force_label_rows) > 0:
            self._save_labeled_data(
                force_label_rows,
                calc_label_rows,
                mea_label_rows,
                mea="mea" in preprocessed_data.columns,
            )
        return case_skipped, len(force_label_rows) > 0

    def start_labeling(self) -> None:
        """
        Start the labeling process for all unprocessed cases.

        Process:
            1. Identifies all preprocessed HDF files
            2. Checks processing status of each case
            3. Skips completed cases
            4. Initiates labeling for unprocessed cases
            5. Updates processing status based on results

        Progress is tracked in metadata file to allow resuming interrupted sessions.
        """
        processed_hdf_paths = self._read_processed_hdf_paths()
        case_identifiers = [
            os.path.basename(path).replace(".hdf", "") for path in processed_hdf_paths
        ]

        for case_identifier in case_identifiers:
            processing_status = self._check_processing_status(case_identifier)
            if processing_status and processing_status == self.COMPLETED:
                logger.info(
                    f"Case {case_identifier} has already been processed. Skipping..."
                )
                continue
            if processing_status:
                logger.info(
                    f"\nCurrent Processing status for identifier {case_identifier}: {processing_status}"
                )

            logger.info(f"\nStarting labeling for case {case_identifier}...")
            self._update_processing_status(case_identifier, self.STARTED)

            skipped, peaks_found = self._label_case(case_identifier)
            if skipped:
                self._update_processing_status(case_identifier, self.SKIPPED)
                logger.info(f"Skipping case {case_identifier} due to no peaks found.")
                continue
            if peaks_found:
                self._update_processing_status(case_identifier, self.COMPLETED)
                logger.info(f"Completed labeling for case {case_identifier}.")
            else:
                self._update_processing_status(case_identifier, self.NO_PEAKS_FOUND)
                logger.info(f"No peaks found for case {case_identifier}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessed_dir",
        type=str,
        default="./Preprocessed",
        help="Path to the directory containing preprocessed data",
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
        default="./Labeled",
        help="Path to the output directory for storing labeled data and metadata",
    )
    parser.add_argument(
        "--window_size_seconds",
        type=float,
        default=1.5,
        help="Window size in seconds for labeling",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="./RawHDFs",
        help="Path to the directory containing raw data",
    )
    args = parser.parse_args()
    labeler = ArrythmiaLabeler(
        args.preprocessed_dir,
        args.raw_dir,
        args.error_metadata_file,
        args.output_dir,
        args.window_size_seconds,
    )
    labeler.start_labeling()
