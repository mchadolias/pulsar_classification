"""
poller.py
----------

The Poller class is a lightweight central data registry used to collect all
intermediate and final model outputs during training.

It stores:
    • Metrics for each model
    • Confusion matrices
    • Feature importances
    • ROC and PR curves
    • Threshold curves (F1, F2 vs threshold)
    • Correlation matrices
    • Prediction histories

The Poller is written by run_training.py and read by ModelComparisonPlotter.
Its state is serialised to a single JSON file for easy reproducibility.
"""

import json
from pathlib import Path
import numpy as np


class Poller:
    """
    Central data collector for model outputs.

    The Poller acts as a unified repository for:
      - feature importances (per model)
      - metrics (per model)
      - confusion matrices (per model)
      - prediction histories (per model)
      - ROC curves
      - Precision-Recall curves
      - Threshold vs F2 curves
      - Correlation matrices / heatmaps

    run_training.py writes into Poller.
    plotter.py reads from Poller.
    """

    def __init__(self, save_dir: str = "outputs/metrics/poller"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # internal registries
        self.feature_importances = {}
        self.metrics = {}
        self.confusion_matrices = {}
        self.prediction_history = {}
        self.roc_curves = {}
        self.pr_curves = {}
        self.threshold_curves = {}
        self.correlation_matrices = {}

    # --------------------------------------------------------------
    # ADD DATA
    # --------------------------------------------------------------
    def add_feature_importances(self, model_name: str, feature_dict):
        self.feature_importances[model_name] = {k: float(v) for k, v in feature_dict.items()}

    def add_metrics(self, model_name: str, metrics_dict):
        self.metrics[model_name] = metrics_dict

    def add_confusion_matrix(self, model_name: str, cm):
        cm_arr = np.asarray(cm)
        self.confusion_matrices[model_name] = cm_arr.tolist()

    def add_prediction_history(self, model_name: str, history: dict):
        self.prediction_history[model_name] = history

    def add_roc_curve(self, model_name: str, fpr, tpr, auc):
        self.roc_curves[model_name] = {
            "fpr": np.asarray(fpr).tolist(),
            "tpr": np.asarray(tpr).tolist(),
            "auc": float(auc),
        }

    def add_pr_curve(self, model_name: str, precision, recall, ap):
        self.pr_curves[model_name] = {
            "precision": np.asarray(precision).tolist(),
            "recall": np.asarray(recall).tolist(),
            "ap": float(ap),
        }

    def add_threshold_curve(self, model_name: str, thresholds, f1_scores, f2_scores):
        """
        Store threshold arrays for both F1 and F2.
        """
        self.threshold_curves[model_name] = {
            "thresholds": np.asarray(thresholds).tolist(),
            "f1": np.asarray(f1_scores).tolist(),
            "f2": np.asarray(f2_scores).tolist(),
        }

    def add_correlation_matrix(self, name: str, matrix, labels):
        self.correlation_matrices[name] = {
            "matrix": np.asarray(matrix).tolist(),
            "labels": list(labels),
        }

    # --------------------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------------------
    def save(self, filename: str = "poller.json"):
        out = {
            "feature_importances": self.feature_importances,
            "metrics": self.metrics,
            "confusion_matrices": self.confusion_matrices,
            "prediction_history": self.prediction_history,
            "roc_curves": self.roc_curves,
            "pr_curves": self.pr_curves,
            "threshold_curves": self.threshold_curves,
            "correlation_matrices": self.correlation_matrices,
        }
        path = self.save_dir / filename
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        return path

    @classmethod
    def load(cls, path: str = "outputs/metrics/poller/poller.json"):
        poller = cls()
        p = Path(path)
        with open(p, "r") as f:
            data = json.load(f)

        poller.feature_importances = data.get("feature_importances", {})
        poller.metrics = data.get("metrics", {})
        poller.confusion_matrices = data.get("confusion_matrices", {})
        poller.prediction_history = data.get("prediction_history", {})
        poller.roc_curves = data.get("roc_curves", {})
        poller.pr_curves = data.get("pr_curves", {})
        poller.threshold_curves = data.get("threshold_curves", {})
        poller.correlation_matrices = data.get("correlation_matrices", {})

        return poller
