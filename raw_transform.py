import pandas as pd
import numpy as np
import os
import argparse
import logging
from utils import fs

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class RawDataHDFProcessor:
    """
    Process and combine raw experimental data files into a single HDF file.

    This class reads Force, Calcium, and MEA (Multi-Electrode Array) data from separate files
    and combines them into a single HDF file for each experimental take.

    Attributes:
        folder_path (str): Root directory containing the experimental data
        calcium_sampling_rate (int): Sampling rate for calcium measurements in Hz
        output_folder (str): Directory where processed HDF files will be saved
        target_directories (list): List of experiment subdirectories to process
    """

    def __init__(
        self, folder_path, calcium_sampling_rate=100, output_folder="./RawHDFs"
    ):
        self.folder_path = folder_path
        self.calcium_sampling_rate = calcium_sampling_rate
        self.target_directories = [
            "run1_ca_titration/1.12_B04",
            "run1_ca_titration/1.13_B19",
            "run1_ca_titration/1.20_B01",
            "run1_e-4031/1.14_B02",
            "run1_e-4031/1.17_B02",
            "run1_e-4031/1.18_A28",
            "run1_nifedipine/1.25_B17",
            "run1_nifedipine/1.28_B02",
            "run1_nifedipine/1.30_B03",
            "run1b_ca_titration/1.13b_B19",
            "run1b_ca_titration/1.20b_B01",
            "run1b_e-4031/1.17b_B02",
            "run1b_e-4031/1.20b_B01",
            "run1b_nifedipine/1.28b_B02",
            "run1b_nifedipine/1.30b_B03",
        ]
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def _read_force_data(self, take_folder):
        """
        Read and process force measurement data from a CSV file.

        Args:
            take_folder (str): Path to the folder containing force measurement data

        Returns:
            pd.DataFrame: Processed force data with time index and force measurements

        Raises:
            NotImplementedError: If trigger information beyond third trigger is found
        """
        force_file_path = fs.findFileWithKeyword(f"{take_folder}/Force", "Measurement")
        df_force = pd.read_csv(
            force_file_path,
            encoding="iso8859_15",
            decimal=",",
            sep="\t",
            header=0,
            usecols=[1, 2, 3],
        )

        # make data set pretty
        df_force = df_force.rename(
            columns={
                "Relative Time [s]": "time",
                "Force_filtered [mN]": "force",
                "Pace [V]": "pace",
            }
        )

        # remove data before the trigger event, but first, check on which trigger the data was acquired.
        file_trigInfo = fs.findFileWithKeyword(f"{take_folder}/Trigger", "trigger")
        counter_trigger = 0
        if (not file_trigInfo == None) and os.path.isfile(file_trigInfo):
            file_trigInfo = file_trigInfo[
                file_trigInfo.rfind("/") + 1 :
            ]  # get only the filename
            if file_trigInfo.find("second") > -1:
                counter_trigger = 1
            elif file_trigInfo.find("third") > -1:
                counter_trigger = 2
            else:
                raise NotImplementedError("Trigger information > 3 is not implemented.")

        first_trigger = df_force.loc[df_force.pace < 0].index[counter_trigger]
        df_force = df_force.loc[first_trigger:]

        # reset the index and drop the unneeded pace column
        frequency = df_force.time.iloc[1] - df_force.time.iloc[0]
        time = np.round(
            np.arange(0, df_force.shape[0] * frequency, frequency, dtype=np.float64), 2
        )
        time = time[: len(df_force)]
        df_force.time = time
        df_force = df_force.set_index("time")
        df_force = df_force.drop(columns=["pace"])

        return df_force

    def _read_calcium_data(self, take_folder):
        """
        Read and process calcium imaging data from an Excel file.

        Processes raw grey values and background measurements to calculate
        normalized calcium signals (deltaF/F0).

        Args:
            take_folder (str): Path to the folder containing calcium measurement data

        Returns:
            pd.DataFrame: Processed calcium data with time index and normalized signals
        """
        calcium_file_path = fs.findFileWithKeyword(
            f"{take_folder}/Calcium", "Greyvalues"
        )
        df = pd.read_excel(calcium_file_path)
        new_df = pd.DataFrame()
        new_df["Time"] = (df[df.columns[0]] - 1) / self.calcium_sampling_rate
        new_df["f"] = df["GV"]
        new_df["background"] = df["BG"]
        new_df["F-background"] = new_df["f"] - new_df["background"]
        new_df["F0"] = new_df["F-background"].min()
        new_df["deltaF"] = new_df["F-background"] - new_df["F0"]
        new_df["deltaF/F0"] = new_df["deltaF"] / new_df["F0"]
        new_df = new_df.set_index("Time")
        calc_df = new_df.drop(
            columns=["f", "F-background", "background", "F0", "deltaF"]
        )
        calc_df = calc_df.rename(columns={"deltaF/F0": "calc"})
        return calc_df

    def _read_mea_data(self, take_folder):
        """
        Read and process Multi-Electrode Array (MEA) recordings.

        Combines trigger signals with electrode recordings and aligns the data
        to the first trigger event. Converts raw values from pV to mV.

        Args:
            take_folder (str): Path to the folder containing MEA data

        Returns:
            pd.DataFrame: Processed MEA data with time index and electrode recordings

        Raises:
            FileNotFoundError: If required MEA data files are not found
            IOError: If files exist but cannot be opened
        """
        # find data paths
        # path_root = "Experimenter"

        # Digital = Trigger; Electrode = Raw Data; Filter = Filtered Data
        fnames = {"Digital Data": "", "Electrode": "", "Filter": ""}
        for key in fnames:
            fnames[key] = fs.findFileWithKeyword(take_folder, key)

        for _key in fnames:
            if fnames[_key] == None:
                raise FileNotFoundError(f"File not found: {take_folder}, {key}")
            if not os.path.isfile(fnames[_key]):
                raise IOError(f"File not found: {fnames[_key]}")

        # read TRIGGER data
        df_trigger = pd.read_csv(
            fnames["Digital Data"], skiprows=7, delimiter=",", names=["time", "trigger"]
        )

        # set the time as index
        df_trigger = df_trigger.set_index("time")

        # read RAW data
        df_raw = pd.read_csv(fnames["Electrode"], skiprows=6, delimiter=",")
        # rename colums to be easier callable
        cols = {}
        for col in df_raw.columns:
            if col.find("ID") > -1:
                cols[col] = col[:2]
            elif col.find("TimeStamp") > -1:
                cols[col] = "time"
        df_raw = df_raw.rename(columns=cols)

        # change the index to the recording time
        df_raw = df_raw.set_index("time")

        # change the data from pV to mV
        df_raw = df_raw.apply(lambda x: x / 1e9)

        # MERGE
        # trigger and raw data
        df_exp = pd.merge(
            df_trigger, df_raw, how="inner", left_index=True, right_index=True
        )

        # ALIGN to trigger events
        first_trigger = df_exp.loc[df_exp.trigger > 0].index[0]
        df_exp = df_exp.drop(columns=["trigger"])
        df_exp = df_exp.loc[first_trigger:]

        # cut away the signals before the first trigger arrival
        df_exp.index = df_exp.index - first_trigger

        df_exp.index = df_exp.index / 1e6

        return df_exp

    def _process_take(self, take_folder):
        """
        Process all data types for a single experimental take.

        Combines Force, Calcium, and MEA data (if available) into a single DataFrame
        with aligned time indices.

        Args:
            take_folder (str): Path to the folder containing all measurement data

        Returns:
            pd.DataFrame: Combined data from all available measurements
        """
        mea_found = False
        try:
            df_mea = self._read_mea_data(take_folder)
            mea_found = True
        except FileNotFoundError:
            logger.warning(f"MEA data not found for {take_folder}")
        df_calcium = self._read_calcium_data(take_folder)
        df_force = self._read_force_data(take_folder)

        # combine dataframes
        df = pd.merge(
            df_calcium, df_force, how="left", left_index=True, right_index=True
        )
        if mea_found:
            df = pd.merge(df, df_mea, how="outer", left_index=True, right_index=True)
        return df

    def execute(self):
        """
        Process all experimental takes in the target directories.

        Iterates through the directory structure:
        drug_type/batch/concentration/take
        and saves processed data as HDF files in the output folder.
        """
        logger.info("Starting data processing")
        for drug_bct_folder in self.target_directories:
            drug = drug_bct_folder.split("/")[0]
            bct = drug_bct_folder.split("/")[1]
            target_folder = os.path.join(self.folder_path, drug_bct_folder)
            if os.path.isdir(target_folder):
                for concentration in os.listdir(target_folder):
                    if os.path.isdir(os.path.join(target_folder, concentration)):
                        for take in os.listdir(
                            os.path.join(target_folder, concentration)
                        ):
                            if os.path.isdir(
                                os.path.join(target_folder, concentration, take)
                            ):
                                logger.info(
                                    f"Processing {drug}::{bct}::{concentration}::{take}"
                                )
                                df = self._process_take(
                                    os.path.join(target_folder, concentration, take)
                                )
                                output_file = f"{self.output_folder}/{drug}__{bct}__{concentration}__{take}__raw_data.hdf"
                                df.to_hdf(
                                    output_file,
                                    key="raw_data",
                                    mode="w",
                                    complevel=4,
                                    format="table",
                                )
                                logger.debug(f"Saved processed data to {output_file}")
        logger.info("Processing complete")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--raw-data-dir",
        type=str,
        default="./Raw Data",
        help="Path to the folder containing the raw data",
    )
    args.add_argument(
        "--output_dir",
        type=str,
        default="./RawHDFs",
        help="Path to the folder where the processed HDF files will be saved",
    )
    args.add_argument(
        "--calcium_sampling_rate",
        type=int,
        default=100,
        help="Sampling rate for calcium measurements in Hz (default: 100 Hz)",
    )
    args = args.parse_args()
    processor = RawDataHDFProcessor(
        folder_path=args.raw_data_dir,
        output_folder=args.output_dir,
        calcium_sampling_rate=args.calcium_sampling_rate,
    )
    processor.execute()
