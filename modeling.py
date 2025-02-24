from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
)
from imblearn.metrics import geometric_mean_score
from datetime import datetime

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import json
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ArrythmiaClassificationModeler:
    """
    A class for training and evaluating arrhythmia classification models for different signal types.

    This class handles the training of Random Forest classifiers for force, calcium, and field potential
    signals, with support for different window sizes. It automatically selects the best performing model
    based on specified evaluation criteria and saves comprehensive training metrics and visualizations for best performing model.
    The timestamp is used to identify the training run and is included in the filenames of the saved artifacts.

    The output directory is created if it does not exist.
    The output structure is:
        - <model_dir>/<signal_type>/<training_run_timestamp>/
            - <model_name>.joblib # best performing model
            - <metrics_name>.json # scalar metrics
            - <confusion_matrix_name>.png # confusion matrix
            - <roc_curve_name>.png # roc curve
            - <classification_report_name>.txt # classification report
            - <training_metrics>.csv # training metrics for all models
        where <name> include the best performing window size i.e '_best_window_<window_size>s'

    Attributes:
        FORCE_SIGNAL_TYPE (str): Identifier for force signal data
        CALCIUM_SIGNAL_TYPE (str): Identifier for calcium signal data
        FIELD_POTENTIAL_SIGNAL_TYPE (str): Identifier for field potential signal data
        TARGET_NAMES (list): Class labels for classification ["Normal", "Arrythmia"]
    """

    FORCE_SIGNAL_TYPE = "force"
    CALCIUM_SIGNAL_TYPE = "calcium"
    FIELD_POTENTIAL_SIGNAL_TYPE = "field_potential"
    TARGET_NAMES = ["Normal", "Arrythmia"]

    def __init__(
        self,
        ground_truth_dir: str = "./GroundTruth",
        model_dir: str = "./Models",
        test_size: float = 0.3,
        random_state: int = 42,
        n_estimators: int = 50,
        cross_validation_n_folds: int = 10,
        cross_validation_n_repeats: int = 3,
        evaluation_criteria: str = "weighted_geometric_mean",
    ):
        """
        Initialize the ArrythmiaClassificationModeler with the provided parameters.

        Args:
            ground_truth_dir (str): Directory containing ground truth data
            model_dir (str): Directory to store trained model artifacts
            test_size (float): Proportion of data to include in the test split
            random_state (int): Random seed for reproducibility
            n_estimators (int): Number of trees in the random forest
            cross_validation_n_folds (int): Number of folds in the cross validation
            cross_validation_n_repeats (int): Number of repeats in the cross validation
            evaluation_criteria (str): Evaluation criteria for selecting the best model
        """
        self.ground_truth_dir = ground_truth_dir
        self.model_dir = model_dir
        self.test_size = test_size
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.cross_validation_n_folds = cross_validation_n_folds
        self.cross_validation_n_repeats = cross_validation_n_repeats
        self.evaluation_criteria = evaluation_criteria
        self._cv_scoring = ("f1", "recall", "precision", "balanced_accuracy")
        self._timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logger.info(
            f"Setting timestamp for training run: training_run_{self._timestamp}"
        )
        self._validate_dirs()

    def _validate_dirs(self):
        """
        Validate the directories and create them if they do not exist.
        """
        assert os.path.exists(
            self.ground_truth_dir
        ), f"Ground truth directory {self.ground_truth_dir} does not exist"
        assert os.path.exists(
            os.path.join(self.ground_truth_dir, self.FORCE_SIGNAL_TYPE)
        ), "Force ground truth directory does not exist"
        assert os.path.exists(
            os.path.join(self.ground_truth_dir, self.CALCIUM_SIGNAL_TYPE)
        ), "Calcium ground truth directory does not exist"
        assert os.path.exists(
            os.path.join(self.ground_truth_dir, self.FIELD_POTENTIAL_SIGNAL_TYPE)
        ), "Field potential ground truth directory does not exist"
        # create model directory if it does not exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, self.FORCE_SIGNAL_TYPE), exist_ok=True)
        os.makedirs(
            os.path.join(self.model_dir, self.CALCIUM_SIGNAL_TYPE), exist_ok=True
        )
        os.makedirs(
            os.path.join(self.model_dir, self.FIELD_POTENTIAL_SIGNAL_TYPE),
            exist_ok=True,
        )

    def _load_training_features_for_window_size(self, window_ground_truth_path: str):
        """
        Load and preprocess training features from a CSV file for a specific window size.

        Args:
            window_ground_truth_path (str): Path to the CSV file containing features

        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target vector
                  Features are columns 11 onwards, target is column 10
                  Duplicate feature rows are removed to reduce redundancy
        """
        features_df = pd.read_csv(window_ground_truth_path)
        X, y = features_df.iloc[:, 11:], features_df.iloc[:, 10]
        duplicate_idx = X[X.duplicated()].index
        X = X.drop(duplicate_idx)
        y = y.drop(duplicate_idx)
        return X, y

    def _train_model_for_window_size(self, X, y):
        """
        Train a Random Forest model and compute comprehensive performance metrics.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            tuple: (model, metrics) where:
                - model: Trained RandomForestClassifier
                - metrics: Dict containing:
                    - f1_positive_class: F1 score for arrhythmia class
                    - weighted_geometric_mean: Geometric mean of class-wise performance
                    - balanced_accuracy: Accuracy adjusted for class imbalance
                    - fpr: False positive rates for ROC curve
                    - tpr: True positive rates for ROC curve
                    - roc_auc: Area under ROC curve
                    - classification_report: Detailed classification metrics
                    - confusion_matrix: Confusion matrix
                    - cv_test_mean_*: Cross-validation metrics
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
            shuffle=True,
        )
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

        cv = RepeatedStratifiedKFold(
            n_splits=self.cross_validation_n_folds,
            n_repeats=self.cross_validation_n_repeats,
            random_state=self.random_state,
        )
        cv_results = cross_validate(model, X, y, cv=cv, scoring=self._cv_scoring)
        metrics = {
            "f1_positive_class": np.round(f1_score(y_test, y_pred, pos_label=1), 2),
            "weighted_geometric_mean": np.round(
                geometric_mean_score(y_test, y_pred, average="weighted"), 2
            ),
            "balanced_accuracy": np.round(balanced_accuracy_score(y_test, y_pred), 2),
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": np.round(roc_auc_score(y_test, y_pred_proba), 2),
            "classification_report": classification_report(
                y_test, y_pred, target_names=self.TARGET_NAMES, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=model.classes_),
            "cv_test_mean_f1": np.round(np.mean(cv_results["test_f1"]), 2),
            "cv_test_mean_recall": np.round(np.mean(cv_results["test_recall"]), 2),
            "cv_test_mean_precision": np.round(
                np.mean(cv_results["test_precision"]), 2
            ),
            "cv_test_mean_balanced_accuracy": np.round(
                np.mean(cv_results["test_balanced_accuracy"]), 2
            ),
        }
        return model, metrics

    def _select_best_model_on_evaluation_criteria(self, models_dict: dict):
        """
        Select the best performing model based on specified evaluation criteria.

        The selection process uses a hierarchical approach:
        1. First, sorts by the user-specified evaluation criteria
        2. Then uses F1 score as the first tiebreaker (if not already the primary criteria)
        3. Uses remaining metrics as additional tiebreakers
        4. Finally uses window size as the last tiebreaker (preferring smaller windows)

        Args:
            models_dict (dict): Dictionary of models and their metrics, keyed by window size

        Returns:
            str: Window size of the best performing model
        """
        # Define the metrics order based on user's selection
        metrics_order = []
        all_metrics = [
            "f1_positive_class",
            "weighted_geometric_mean",
            "balanced_accuracy",
            "roc_auc",
        ]

        # Add user-selected metric first
        metrics_order.append(self.evaluation_criteria)

        # Add remaining metrics with f1_positive_class prioritized if not already selected
        remaining_metrics = [m for m in all_metrics if m != self.evaluation_criteria]
        if self.evaluation_criteria != "f1_positive_class":
            # Ensure f1_positive_class is first in remaining metrics
            remaining_metrics.remove("f1_positive_class")
            remaining_metrics.insert(0, "f1_positive_class")

        metrics_order.extend(remaining_metrics)

        def sort_key(window_size):
            # Create tuple of metrics for sorting (negative values for descending sort)
            metric_values = [
                -models_dict[window_size]["metrics"][m] for m in metrics_order
            ]
            # Append window size as final tiebreaker (convert to float for proper comparison)
            metric_values.append(float(window_size))
            return tuple(metric_values)

        # Return the window size with the best metrics
        return min(models_dict.keys(), key=sort_key)

    def _store_specific_model_metrics(self, model_metrics: dict, signal_type: str):
        """
        Store the scalar metrics for all models during training in a CSV file for a specific signal type

        Args:
            model_metrics (dict): Dictionary of models and their metrics
            signal_type (str): Type of signal (force/calcium/field_potential)
        """
        logger.info(f"Storing training metrics for signal type {signal_type}")
        model_metrics_df = pd.DataFrame()
        for window_size in model_metrics:
            metrics = model_metrics[window_size]["metrics"]
            selected_metrics = {
                "window_size": window_size,
                "f1_positive_class": metrics["f1_positive_class"],
                "weighted_geometric_mean": metrics["weighted_geometric_mean"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "roc_auc": metrics["roc_auc"],
                "cross_validation_mean_f1": metrics["cv_test_mean_f1"],
                "cross_validation_mean_recall": metrics["cv_test_mean_recall"],
                "cross_validation_mean_precision": metrics["cv_test_mean_precision"],
                "cross_validation_mean_balanced_accuracy": metrics[
                    "cv_test_mean_balanced_accuracy"
                ],
            }
            model_metrics_df = pd.concat(
                [model_metrics_df, pd.DataFrame([selected_metrics])], ignore_index=True
            )
        # Create output directory for this signal type and training run
        model_output_dir = os.path.join(
            self.model_dir, signal_type, f"training_run_{self._timestamp}"
        )
        os.makedirs(model_output_dir, exist_ok=True)
        model_metrics_df.to_csv(
            os.path.join(model_output_dir, f"training_metrics.csv"),
            index=False,
        )

    def _save_best_model_with_metrics(
        self, best_performing_window_size: str, models_dict: dict, signal_type: str
    ) -> None:
        """
        Save the best performing model along with its metrics and visualization plots.

        Args:
            best_performing_window_size: Window size that achieved best performance
            models_dict: Dictionary containing models and their metrics
            signal_type: Type of signal (force/calcium/field_potential)

        Saves:
            - Best model pickle file
            - Metrics JSON file containing scalar metrics
            - PNG plots for confusion matrix and ROC curve
            All files include timestamp in format: YYYYMMDD_HHMMSS
        """
        logger.info(f"Saving best model for signal type {signal_type}")
        best_metrics = models_dict[best_performing_window_size]["metrics"]
        best_model = models_dict[best_performing_window_size]["model"]

        # Create output directory for this signal type and training run
        model_output_dir = os.path.join(
            self.model_dir, signal_type, f"training_run_{self._timestamp}"
        )
        os.makedirs(model_output_dir, exist_ok=True)

        # Create base filename
        base_filename = f"best_window_{best_performing_window_size}s"

        # Save the model
        model_path = os.path.join(model_output_dir, f"model_{base_filename}.joblib")
        joblib.dump(best_model, model_path)
        logger.info(f"Saved best model to {model_path}")

        # Extract scalar metrics (add timestamp)
        scalar_metrics = {
            "training_timestamp": self._timestamp,
            "window_size": best_performing_window_size,
            "f1_positive_class": best_metrics["f1_positive_class"],
            "weighted_geometric_mean": best_metrics["weighted_geometric_mean"],
            "balanced_accuracy": best_metrics["balanced_accuracy"],
            "roc_auc": best_metrics["roc_auc"],
            "cross_validation": {
                "mean_f1": best_metrics["cv_test_mean_f1"],
                "mean_recall": best_metrics["cv_test_mean_recall"],
                "mean_precision": best_metrics["cv_test_mean_precision"],
                "mean_balanced_accuracy": best_metrics[
                    "cv_test_mean_balanced_accuracy"
                ],
            },
        }

        # Save metrics as JSON
        metrics_path = os.path.join(model_output_dir, f"metrics_{base_filename}.json")
        with open(metrics_path, "w") as f:
            json.dump(scalar_metrics, f, indent=4)
        logger.info(f"Saved metrics to {metrics_path}")

        # Plot and save confusion matrix
        plt.figure(figsize=(10, 8))
        cm_display = ConfusionMatrixDisplay(
            confusion_matrix=best_metrics["confusion_matrix"],
            display_labels=self.TARGET_NAMES,
        )
        cm_display.plot(cmap="Blues", values_format="d")
        plt.title(
            f"Confusion Matrix - {signal_type}\n(Window: {best_performing_window_size}s, {self._timestamp})"
        )
        cm_path = os.path.join(
            model_output_dir, f"confusion_matrix_{base_filename}.png"
        )
        plt.savefig(cm_path, bbox_inches="tight", dpi=300)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {cm_path}")

        # Plot and save ROC curve
        plt.figure(figsize=(10, 8))
        roc_display = RocCurveDisplay(
            fpr=best_metrics["fpr"],
            tpr=best_metrics["tpr"],
            roc_auc=best_metrics["roc_auc"],
        )
        roc_display.plot()
        plt.plot([0, 1], [0, 1], "k--")  # Add diagonal line
        plt.title(
            f"ROC Curve - {signal_type}\n(Window: {best_performing_window_size}s, {self._timestamp})"
        )
        roc_path = os.path.join(model_output_dir, f"roc_curve_{base_filename}.png")
        plt.savefig(roc_path, bbox_inches="tight", dpi=300)
        plt.close()
        logger.info(f"Saved ROC curve plot to {roc_path}")

        # Save classification report
        report_path = os.path.join(
            model_output_dir, f"classification_report_{base_filename}.txt"
        )
        with open(report_path, "w") as f:
            f.write(f"Training Run: {self._timestamp}\n")
            f.write(f"Signal Type: {signal_type}\n")
            f.write(f"Window Size: {best_performing_window_size}s\n\n")
            f.write(best_metrics["classification_report"])
        logger.info(f"Saved classification report to {report_path}")

    def _train_models_for_signal_type(self, signal_type: str):
        """
        Train models for a specific signal type and store the results.

        This function is called for each signal type (force/calcium/field_potential)
        It loads the ground truth data, trains models for each window size,
        stores the metrics, and selects the best performing model.

        Args:
            signal_type (str): Type of signal (force/calcium/field_potential)
        """
        # get all the files in the ground truth directory for the signal type
        logger.info(f"Training models for signal type {signal_type}")
        files = os.listdir(os.path.join(self.ground_truth_dir, signal_type))
        models_dict = {}
        for file in sorted(files):
            # load the features
            window_size = file.split("_")[1].replace(".csv", "").replace("s", "")
            logger.info(
                f"Training model for window size {window_size} for signal type {signal_type}"
            )
            models_dict[window_size] = {}
            X, y = self._load_training_features_for_window_size(
                os.path.join(self.ground_truth_dir, signal_type, file)
            )
            model, metrics = self._train_model_for_window_size(X, y)
            models_dict[window_size]["model"] = model
            models_dict[window_size]["metrics"] = metrics

        self._store_specific_model_metrics(models_dict, signal_type)

        best_performing_window_size = self._select_best_model_on_evaluation_criteria(
            models_dict
        )
        logger.info(
            f"Best performing window size for signal type {signal_type} is {best_performing_window_size} with {self.evaluation_criteria} {models_dict[best_performing_window_size]['metrics'][self.evaluation_criteria]}"
        )
        self._save_best_model_with_metrics(
            best_performing_window_size, models_dict, signal_type
        )

    def start(self):
        """
        Start the training process for all signal types (force/calcium/field_potential)
        """
        for signal_type in [
            self.FORCE_SIGNAL_TYPE,
            self.CALCIUM_SIGNAL_TYPE,
            self.FIELD_POTENTIAL_SIGNAL_TYPE,
        ]:
            self._train_models_for_signal_type(signal_type)


# def _test_model_selection():
#     """
#     Test function to demonstrate model selection behavior with different evaluation criteria.
#     This is for illustration purposes only.
#     """
#     # Mock data representing different window sizes and their metrics
#     test_models_dict = {
#         "2": {
#             "metrics": {
#                 "f1_positive_class": 0.86,
#                 "weighted_geometric_mean": 0.87,
#                 "balanced_accuracy": 0.83,
#                 "roc_auc": 0.88,
#             }
#         },
#         "5": {
#             "metrics": {
#                 "f1_positive_class": 0.88,  # Same f1 as window 2
#                 "weighted_geometric_mean": 0.87,
#                 "balanced_accuracy": 0.82,
#                 "roc_auc": 0.88,
#             }
#         },
#         "10": {
#             "metrics": {
#                 "f1_positive_class": 0.87,
#                 "weighted_geometric_mean": 0.87,
#                 "balanced_accuracy": 0.84,
#                 "roc_auc": 0.88,  # Same ROC as others
#             }
#         },
#     }

#     # Test cases with different evaluation criteria
#     test_cases = [
#         "f1_positive_class",
#         "weighted_geometric_mean",
#         "balanced_accuracy",
#         "roc_auc",
#     ]

#     print("\nModel Selection Test Cases:")
#     print("-" * 50)

#     for criteria in test_cases:
#         modeler = ArrythmiaClassificationModeler(evaluation_criteria=criteria)
#         best_window = modeler._select_best_model_on_evaluation_criteria(
#             test_models_dict
#         )

#         print(f"\nEvaluation Criteria: {criteria}")
#         print(f"Selected Window Size: {best_window}")
#         print("Metrics for selected window:")
#         print(
#             f"- F1 Score: {test_models_dict[best_window]['metrics']['f1_positive_class']}"
#         )
#         print(
#             f"- Geometric Mean: {test_models_dict[best_window]['metrics']['weighted_geometric_mean']}"
#         )
#         print(
#             f"- Balanced Accuracy: {test_models_dict[best_window]['metrics']['balanced_accuracy']}"
#         )
#         print(f"- ROC AUC: {test_models_dict[best_window]['metrics']['roc_auc']}")


if __name__ == "__main__":
    # _test_model_selection()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        default="./GroundTruth",
        help="Directory containing ground truth data",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./Models",
        help="Directory to store trained model artifacts",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.3,
        help="Proportion of data to include in the test split",
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=50,
        help="Number of trees in the random forest",
    )
    parser.add_argument(
        "--cross_validation_n_folds",
        type=int,
        default=10,
        help="Number of folds in the cross validation",
    )
    parser.add_argument(
        "--cross_validation_n_repeats",
        type=int,
        default=3,
        help="Number of repeats in the cross validation",
    )
    parser.add_argument(
        "--evaluation_criteria",
        type=str,
        default="weighted_geometric_mean",
        help="Evaluation criteria for selecting the best model [possible values: f1_positive_class, weighted_geometric_mean, balanced_accuracy, roc_auc], Default: weighted_geometric_mean",
    )
    args = parser.parse_args()
    modeler = ArrythmiaClassificationModeler(
        args.ground_truth_dir,
        args.model_dir,
        args.test_size,
        args.random_state,
        args.n_estimators,
        args.cross_validation_n_folds,
        args.cross_validation_n_repeats,
        args.evaluation_criteria,
    )
    modeler.start()
