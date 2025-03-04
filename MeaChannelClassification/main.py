import os
import sys
import json
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import MEA
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from glob import glob
import tsfel
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from datetime import datetime
import argparse

import warnings

warnings.filterwarnings("ignore")

# Configure logging at global level
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class MEAChannelClassificationController:
    """
    Controller class for MEA (Multi-Electrode Array) channel classification.

    This class handles the classification of MEA channels into three categories:
    - Good (2): High quality signal with clear neural activity
    - Medium (1): Moderate quality signal with some noise/artifacts
    - Bad (0): Poor quality signal, dominated by noise or artifacts

    The controller supports two main modes of operation:
    1. Plotting mode: Visualize raw data, ground truth labels, or model predictions
    2. Training mode: Train and evaluate a Random Forest classifier
    """

    def __init__(
        self,
        data_dir="../RawHDFs",
        error_metadata_file="../error_metadata.json",
        plot=False,
        ground_truth_file=None,
        existing_model_path=None,
        train=False,
        random_state=42,
        test_size=0.2,
        store_predictions=False,
        n_estimators=20,
        model_prefix=None,
        plot_dir="./plots",
        model_dir="./modeling",
    ):
        """
        MEA Channel Classification Controller.

        Args:
            data_dir (str): Directory containing raw MEA data files
            error_metadata_file (str): JSON file containing metadata about errored cases
            plot (bool): Enable plotting mode
            ground_truth_file (str, optional): Path to ground truth labels JSON file
            existing_model_path (str, optional): Path to pre-trained model file
            train (bool): Enable training mode
            random_state (int): Random seed for reproducibility
            test_size (float): Proportion of data to use for testing (0.0-1.0)
            store_predictions (bool): Save prediction plots after training
            n_estimators (int): Number of trees in Random Forest classifier
            model_prefix (str, optional): Prefix for saved model artifacts <default: mea-channel-classification_YYYY-MM-DD_HH:MM:SS>
            plot_dir (str): Directory for saving plots
            model_dir (str): Directory for saving model files
        """
        logger.info("Initializing MEA Channel Classification Controller")
        self.data_dir = data_dir
        self.error_metadata_file = error_metadata_file
        self.plot = plot
        self.ground_truth_file = ground_truth_file
        self.train = train
        self.existing_model_path = existing_model_path
        self.random_state = random_state
        self.test_size = test_size
        self.model_prefix = model_prefix
        self.store_predictions = store_predictions
        self.plot_dir = plot_dir
        self.model_dir = model_dir
        self.ground_truth_data = None
        self.ground_truth_cases_paths = None
        self.n_estimators = n_estimators

        assert os.path.exists(self.data_dir), "Data directory does not exist"
        assert os.path.exists(
            self.error_metadata_file
        ), "Error metadata file does not exist"
        # create a prefix with datetime
        if self.model_prefix is None:
            self.model_prefix = "mea-channel-classification_" + datetime.now().strftime(
                "%Y-%m-%d_%H:%M:%S"
            )
        # tsfel config file with statistical features
        self._cfg_file = tsfel.get_features_by_domain("statistical")
        self._cfg_file["statistical"]["ECDF Percentile"]["parameters"]["percentile"] = [
            0.1,
            0.2,
            0.4,
            0.6,
            0.8,
            0.9,
        ]
        os.makedirs(self.plot_dir, exist_ok=True)
        # check if the plot and train are not set at the same time
        if self.train and self.plot:
            raise Exception(
                "Cannot plot and train at the same time, please set either plot or train to True"
            )
        # check if the plot is set and no ground truth file and no existing model path is provided -> plot all raw cases
        if (
            self.plot
            and self.ground_truth_file is None
            and self.existing_model_path is None
        ):
            raw_plots_dir = os.path.join(self.plot_dir, "raw")
            os.makedirs(raw_plots_dir, exist_ok=True)
            self.plot_all_cases(raw_plots_dir)
        elif self.plot:
            # check if the ground truth file is provided -> plot ground truth cases
            if self.ground_truth_file is not None:
                ground_truth_plots_dir = os.path.join(self.plot_dir, "ground_truth")
                self._init_ground_truth()
                os.makedirs(ground_truth_plots_dir, exist_ok=True)
                self.plot_ground_truth_cases(ground_truth_plots_dir)
            # check if the existing model path is provided -> plot model predictions on all cases
            if self.existing_model_path is not None:
                prediction_plot_dir = os.path.join(self.plot_dir, "predictions")
                os.makedirs(prediction_plot_dir, exist_ok=True)
                if not os.path.exists(self.existing_model_path):
                    raise FileNotFoundError(
                        f"Existing model path {self.existing_model_path} does not exist"
                    )
                model = joblib.load(self.existing_model_path)
                self.plot_model_predictions(model, prediction_plot_dir)
        # check if the train is set -> train the model
        elif self.train:
            # check if the ground truth file is provided
            assert (
                self.ground_truth_file is not None
            ), "Ground truth file is required for training"
            # check if the existing model path is not provided, if provided, raise an error as we are training a new model
            assert (
                self.existing_model_path is None
            ), "In training mode, existing model path should not be provided"
            # create the prediction plot directory
            prediction_plot_dir = os.path.join(self.plot_dir, "predictions")
            os.makedirs(self.model_dir, exist_ok=True)
            os.makedirs(prediction_plot_dir, exist_ok=True)
            # initialize the ground truth data
            self._init_ground_truth()
            # train the model and store the predictions
            self.train_model_with_prediction_plots()

    def get_mea_file_paths(self, cases=[]):
        """
        Get paths to valid MEA data files, excluding known problematic cases.

        Args:
            cases (list): Optional list of specific cases to filter for

        Returns:
            list: Full paths to valid MEA data files

        Raises:
            AssertionError: If no valid files found or requested cases missing
        """
        all_cases = glob(os.path.join(self.data_dir, "**", "*.hdf"), recursive=True)
        errored_cases = json.load(open(self.error_metadata_file, "r"))
        # Remove cases that are in the error metadata file
        all_cases = [
            case.split("/")[-1]
            for case in all_cases
            if (case.split("/")[-1] not in errored_cases["duplicates_to_be_removed"])
            and (case.split("/")[-1] not in errored_cases["out_of_sync"])
            and (case.split("/")[-1] not in errored_cases["unavailable_mea_files"])
        ]
        assert len(all_cases) > 0, "No valid MEA files found"
        if len(cases) == 0:
            return [os.path.join(self.data_dir, case) for case in all_cases]
        else:
            target_cases = [case for case in cases if case in all_cases]
            logger.info(
                f"Found {len(target_cases)} cases out of {len(cases)} requested cases of ground truth"
            )
            missing_cases = [case for case in cases if case not in all_cases]
            if len(missing_cases) > 0:
                logger.warning(f"Missing cases: {missing_cases}")
            assert (
                len(missing_cases) == 0
            ), "Some ground truth cases are not found, check the error metadata file"
            return [os.path.join(self.data_dir, case) for case in target_cases]

    def plot_all_cases(self, raw_plot_dir):
        """
        Generate interactive plots for all valid MEA cases.
        Creates an 8x4 grid of channel plots for each case.

        Args:
            raw_plot_dir (str): Directory to save the plot HTML files
        """
        all_cases = self.get_mea_file_paths()
        logger.info(f"Plotting Mea channels from {len(all_cases)} raw cases")
        for case in all_cases:
            df_raw = pd.read_hdf(case, key="raw_data")
            mea = MEA(df_raw)
            channel_labels = [mea.data[i].name for i in range(len(mea.channels))]
            # 8x4 grid plotly
            fig = make_subplots(
                rows=8,
                cols=4,
                subplot_titles=channel_labels,
            )
            for i in range(8):
                for j in range(4):
                    fig.add_trace(
                        go.Scatter(
                            x=mea.data[i * 4 + j].data_raw.index,
                            y=mea.data[i * 4 + j].data_raw,
                            name=channel_labels[i * 4 + j],
                        ),
                        row=i + 1,
                        col=j + 1,
                    )
            fig.update_layout(
                height=2500,
                width=2300,
                title_text=f"Raw Mea Channels: {case.split('/')[-1].split('__raw_data.hdf')[0]}",
            )
            fig.write_html(
                os.path.join(
                    raw_plot_dir,
                    f"{case.split('/')[-1].split('__raw_data.hdf')[0]}.html",
                )
            )

    def _init_ground_truth(self):
        """
        Initialize ground truth data from JSON file and generate case file paths for raw hdf files.
        Must be called before any ground truth operations.
        """
        with open(self.ground_truth_file, "r") as f:
            self.ground_truth_data = json.load(f)
        ground_truth_cases = [
            f"{case['name']}__raw_data.hdf"
            for case in self.ground_truth_data["experiments"]
        ]
        self.ground_truth_cases_paths = self.get_mea_file_paths(ground_truth_cases)

    def plot_ground_truth_cases(self, ground_truth_plot_dir):
        """
        Generate plots for cases with ground truth labels.
        Each channel is labeled as [Good], [Medium], or [Bad].

        Args:
            ground_truth_plot_dir (str): Directory to save the plot HTML files
        """
        logger.info(f"Plotting {len(self.ground_truth_cases_paths)} ground truth cases")
        for idx, case in enumerate(self.ground_truth_cases_paths):
            df_raw = pd.read_hdf(case, key="raw_data")
            mea = MEA(df_raw)

            channel_labels = [mea.data[i].name for i in range(len(mea.channels))]
            titles = []
            for ch in channel_labels:
                if ch in self.ground_truth_data["experiments"][idx]["good_labels"]:
                    titles.append(f"{ch}: [Good]")
                elif ch in self.ground_truth_data["experiments"][idx]["medium_labels"]:
                    titles.append(f"{ch}: [Medium]")
                else:
                    titles.append(f"{ch}: [Bad]")
            fig = make_subplots(
                rows=8,
                cols=4,
                subplot_titles=titles,
            )

            for i in range(8):
                for j in range(4):
                    fig.add_trace(
                        go.Scatter(
                            x=mea.data[i * 4 + j].data_raw.index,
                            y=mea.data[i * 4 + j].data_raw,
                            name=channel_labels[i * 4 + j],
                        ),
                        row=i + 1,
                        col=j + 1,
                    )
            fig.update_layout(
                height=2500,
                width=2300,
                title_text=f"Ground Truth Mea Channels: {case.split('/')[-1].split('__raw_data.hdf')[0]}",
            )
            fig.write_html(
                os.path.join(
                    ground_truth_plot_dir,
                    f"{case.split('/')[-1].split('__raw_data.hdf')[0]}.html",
                )
            )

    def plot_model_predictions(self, model, predictions_plot_dir):
        """
        Generate plots showing model predictions for each channel.

        Creates an 8x4 grid for each case with channel classifications
        indicated in the subplot titles.

        Args:
            model: Trained classifier model
            predictions_plot_dir (str): Directory to save prediction plots
        """
        all_cases = self.get_mea_file_paths()
        if self.existing_model_path is not None:
            logger.info(
                f"Plotting model predictions on {len(all_cases)} cases using existing model: {self.existing_model_path}"
            )
        else:
            logger.info(f"Plotting model predictions on {len(all_cases)} cases")
        for case in all_cases:
            df_raw = pd.read_hdf(case, key="raw_data")
            mea = MEA(df_raw)
            titles = []
            # sample rate
            fs = 1 / (df_raw.index[1] - df_raw.index[0])
            for i in range(len(mea.channels)):
                channel_data = mea.data[i].data_raw
                channel_features = tsfel.time_series_features_extractor(
                    self._cfg_file, channel_data, fs=fs, verbose=0
                )
                # remove the 0_ prefix from the column names
                col_name = [
                    col.replace("0_", "") for col in channel_features.columns.to_list()
                ]
                channel_features.columns = col_name
                predicted_class = model.predict(channel_features)
                if predicted_class == 2:
                    titles.append("{}: [Good]".format(mea.data[i].name))
                elif predicted_class == 1:
                    titles.append("{}: [Medium]".format(mea.data[i].name))
                elif predicted_class == 0:
                    titles.append("{}: [Bad]".format(mea.data[i].name))

            fig = make_subplots(
                rows=8,
                cols=4,
                subplot_titles=titles,
            )
            for i in range(8):
                for j in range(4):
                    fig.add_trace(
                        go.Scatter(
                            x=mea.data[i * 4 + j].data_raw.index,
                            y=mea.data[i * 4 + j].data_raw,
                            name=mea.data[i * 4 + j].name,
                        ),
                        row=i + 1,
                        col=j + 1,
                    )

            fig.update_layout(
                height=2500,
                width=2300,
                title_text=f"Model Predictions: {case.split('/')[-1].split('__raw_data.hdf')[0]}",
            )
            fig.write_html(
                os.path.join(
                    predictions_plot_dir,
                    f"{case.split('/')[-1].split('__raw_data.hdf')[0]}.html",
                )
            )

    def generate_classification_dataset(self):
        """
        Generate feature dataset from ground truth cases for model training.

        Extracts time series features using TSFELfrom each channel and assigns labels
        based on ground truth data.

        Returns:
            pandas.DataFrame: Dataset with features and class labels
        """
        features = pd.DataFrame()
        for idx, case in enumerate(self.ground_truth_cases_paths):
            df_raw = pd.read_hdf(case, key="raw_data")
            mea = MEA(df_raw)
            case_identifier = case.split("/")[-1].split("__raw_data.hdf")[0]
            fs = 1 / (df_raw.index[1] - df_raw.index[0])
            for i in range(len(mea.channels)):
                channel_data = mea.data[i].data_raw
                channel_identifier = f"{case_identifier}__MEA_{mea.data[i].name}"
                channel_features = tsfel.time_series_features_extractor(
                    self._cfg_file, channel_data, fs=fs, verbose=0
                )
                col_name = [
                    col.replace("0_", "") for col in channel_features.columns.to_list()
                ]
                channel_features.columns = col_name
                channel_features["identifier"] = [channel_identifier]
                if (
                    mea.data[i].name
                    in self.ground_truth_data["experiments"][idx]["good_labels"]
                ):
                    channel_features["class"] = [2]
                elif (
                    mea.data[i].name
                    in self.ground_truth_data["experiments"][idx]["medium_labels"]
                ):
                    channel_features["class"] = [1]
                else:
                    channel_features["class"] = [0]
                features = pd.concat([features, channel_features], axis=0)
        features.index = range(len(features))
        logger.info(
            f"Generated MEA Channel Classification Dataset based on ground truth: {self.ground_truth_file}"
        )
        logger.info(features.head())
        logger.info(
            f"Storing the dataset to disk at {self.model_dir}/classification_dataset.csv"
        )
        features.to_csv(
            os.path.join(
                self.model_dir, f"{self.model_prefix}_classification_dataset.csv"
            )
        )
        return features

    def train_model_with_prediction_plots(self):
        """
        Train Random Forest classifier and evaluate performance.

        Performs the following steps:
        1. Generates classification dataset
        2. Splits data into train/test sets
        3. Trains Random Forest model
        4. Evaluates performance with classification report and confusion matrices
        5. Saves model, results, and optionally prediction plots
        """
        classification_dataset = self.generate_classification_dataset()
        X = classification_dataset.drop(columns=["identifier", "class"])
        y = classification_dataset["class"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        target_names = ["Bad", "Medium", "Good"]
        # Get classification report and confusion matrices
        class_report = classification_report(
            y_test, y_pred, target_names=target_names, output_dict=True
        )
        conf_matrices = multilabel_confusion_matrix(y_test, y_pred)

        # Pretty print the results
        logger.info("\n=== Classification Report ===")
        logger.info(classification_report(y_test, y_pred, target_names=target_names))

        logger.info("\n=== Confusion Matrices ===")
        for i, matrix in enumerate(conf_matrices):
            logger.info(f"\nClass: {target_names[i]}")
            logger.info("True Negative  False Positive")
            logger.info("False Negative True Positive")
            logger.info(matrix)

        # Save Model
        joblib.dump(
            model, os.path.join(self.model_dir, f"{self.model_prefix}_model.joblib")
        )
        # Prepare results for saving
        results = {
            "classification_report": class_report,
            "confusion_matrices": {
                target_names[i]: matrix.tolist()
                for i, matrix in enumerate(conf_matrices)
            },
        }

        # Save results to JSON file
        results_file = os.path.join(
            self.model_dir, f"{self.model_prefix}_classification_results.json"
        )
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

        logger.info(f"\nResults saved to: {results_file}")
        if self.store_predictions:
            logger.info(
                f"Storing model prediction plots in {os.path.join(self.plot_dir, 'predictions')}"
            )
            self.plot_model_predictions(
                model, os.path.join(self.plot_dir, "predictions")
            )


if __name__ == "__main__":

    # Create parser
    parser = argparse.ArgumentParser(
        description="MEA Channel Classification Controller"
    )

    # Create mutually exclusive group for plot and train modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--plots", action="store_true", help="Run in plotting mode")
    mode_group.add_argument("--train", action="store_true", help="Run in training mode")

    # Arguments for plotting mode
    plot_group = parser.add_argument_group("Plotting arguments")
    plot_group.add_argument(
        "--ground_truth_file", type=str, help="Path to ground truth JSON file"
    )
    plot_group.add_argument(
        "--existing_model_path", type=str, help="Path to existing model file"
    )

    # Arguments for training mode
    train_group = parser.add_argument_group("Training arguments")
    train_group.add_argument(
        "--random_state", type=int, default=42, help="Random state for reproducibility"
    )
    train_group.add_argument(
        "--test_size", type=float, default=0.2, help="Test size for train-test split"
    )
    train_group.add_argument(
        "--store_predictions",
        action="store_true",
        help="Store model predictions after training",
    )
    train_group.add_argument(
        "--n_estimators",
        type=int,
        default=20,
        help="Number of estimators for Random Forest",
    )

    # Common arguments
    parser.add_argument(
        "--plot_dir", type=str, default="./plots", help="Directory for storing plots"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./modeling",
        help="Directory for storing model and results",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../RawHDFs",
        help="Directory containing raw data",
    )
    parser.add_argument(
        "--error_metadata_file",
        type=str,
        default="../error_metadata.json",
        help="Path to error metadata file",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.plots:
        # Call controller for different plotting scenarios
        MEAChannelClassificationController(
            data_dir=args.data_dir,
            error_metadata_file=args.error_metadata_file,
            plot=True,
            ground_truth_file=args.ground_truth_file,
            existing_model_path=args.existing_model_path,
            plot_dir=args.plot_dir,
            model_dir=args.model_dir,
        )

    elif args.train:
        if not args.ground_truth_file:
            parser.error("--ground_truth_file is required for training mode")

        # Call controller for training
        MEAChannelClassificationController(
            data_dir=args.data_dir,
            error_metadata_file=args.error_metadata_file,
            ground_truth_file=args.ground_truth_file,
            train=True,
            random_state=args.random_state,
            test_size=args.test_size,
            store_predictions=args.store_predictions,
            n_estimators=args.n_estimators,
            plot_dir=args.plot_dir,
            model_dir=args.model_dir,
        )
