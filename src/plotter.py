"""
plotter.py
-----------

This module provides the ModelComparisonPlotter, a high-level tool that loads
model outputs from the Poller and produces a complete suite of evaluation plots.

These include:
    • ROC & Precision–Recall curves
    • Calibration (reliability) diagrams
    • Confusion matrices
    • Feature importance charts
    • Correlation heatmaps
    • Threshold curves for F1/F2 analysis
    • Unified multi-page PDF report

Typical workflow:
    1. run_training.py collects metrics into a Poller JSON
    2. run_plotting.py (or manual code) loads the Poller
    3. ModelComparisonPlotter generates all comparison plots

All plots are saved to: outputs/plots/comparison/
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from pathlib import Path

from src.poller import Poller


# ============================================================
#  STYLE CONFIGURATION (GLOBAL)
# ============================================================


def configure_style():
    """
    Apply a unified, publication-quality plotting style.
    Called automatically by ModelComparisonPlotter.
    """
    plt.rcParams.update(
        {
            "figure.figsize": (7, 5),
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.color": "#CCCCCC",
        }
    )

    # Minimalist spines
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["legend.frameon"] = False

    # Seaborn foundation
    sns.set_theme(style="whitegrid", font_scale=1.2)

    # Optional: True LaTeX
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Computer Modern Roman", "CMU Serif"]

    # Palette
    sns.set_palette("tab10")


# ============================================================
#  MAIN CLASS
# ============================================================


class ModelComparisonPlotter:
    """
    Generate combined comparison plots from Poller outputs.
    """

    def __init__(self, poller: Poller):
        self.poller = poller
        self.output_dir = Path("outputs/plots/comparison")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_colors = {
            "logistic_regression": "#1f77b4",  # blue
            "random_forest": "#ff7f0e",  # orange
            "gradient_boosting": "#2ca02c",  # green
            "xgboost": "#d62728",  # red
        }

        # apply styling by default
        configure_style()

    @classmethod
    def from_poller_file(cls, path="outputs/metrics/poller/poller.json"):
        poller = Poller.load(path)
        return cls(poller)

    # ---------------------------------------------------------
    #  INTERNAL HELPERS
    # ---------------------------------------------------------

    def _fig(self, w=7, h=5):
        return plt.subplots(figsize=(w, h))

    def _save_fig(self, fig, name: str, formats=("png",), pdf_writer: PdfPages = None):
        base = self.output_dir / name
        stem = base.with_suffix("")

        if "png" in formats:
            fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
        if "svg" in formats:
            fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        if pdf_writer is not None:
            pdf_writer.savefig(fig)
        plt.close(fig)

    def _get_opt_metrics(self, model_name: str):
        m = self.poller.metrics.get(model_name, {})
        optimal_key = next((k for k in m if k.startswith("optimal_threshold_")), None)
        return m[optimal_key] if optimal_key else m

    # ============================================================
    #  CORE COMPARISON PLOTS
    # ============================================================

    def plot_roc(self, formats=("png",), pdf_writer=None):
        fig, ax = self._fig()
        for model, roc_data in self.poller.roc_curves.items():
            fpr = np.array(roc_data["fpr"])
            tpr = np.array(roc_data["tpr"])
            auc = float(roc_data["auc"])
            color = self.model_colors.get(model, None)
            ax.plot(fpr, tpr, label=rf"{model} (AUC = {auc:.3f})", color=color)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve Comparison")
        ax.legend()
        self._save_fig(fig, "roc_comparison", formats, pdf_writer)

    def plot_pr(self, formats=("png",), pdf_writer=None):
        fig, ax = self._fig()
        for model, pr_data in self.poller.pr_curves.items():
            precision = np.array(pr_data["precision"])
            recall = np.array(pr_data["recall"])
            ap = float(pr_data["ap"])
            color = self.model_colors.get(model, None)
            ax.plot(recall, precision, label=rf"{model} (AP = {ap:.3f})", color=color)

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall Comparison")
        ax.legend()
        self._save_fig(fig, "pr_comparison", formats, pdf_writer)

    # ---------------------------------------------------------
    #  CALIBRATION
    # ---------------------------------------------------------

    def plot_calibration(self, formats=("png",), pdf_writer=None, n_bins=10):
        fig, ax = self._fig()
        for model, hist in self.poller.prediction_history.items():
            y_true = np.array(hist["y_true"])
            y_proba = np.array(hist["y_proba"])

            bins = np.linspace(0, 1, n_bins + 1)
            digitized = np.digitize(y_proba, bins) - 1

            avg_pred, avg_true = [], []
            for i in range(n_bins):
                mask = digitized == i
                if mask.sum() > 0:
                    avg_pred.append(y_proba[mask].mean())
                    avg_true.append(y_true[mask].mean())

            if not avg_pred:
                continue
            color = self.model_colors.get(model, None)
            ax.plot(avg_pred, avg_true, marker="o", linewidth=1.3, label=model, color=color)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title("Calibration Curve Comparison")
        ax.legend()
        self._save_fig(fig, "calibration_comparison", formats, pdf_writer)

    # ---------------------------------------------------------
    #  METRIC BARS
    # ---------------------------------------------------------
    def plot_metric_bars(self, formats=("png",), pdf_writer=None):

        metrics = ["f2", "f1", "recall", "precision", "pr_auc", "total_cost"]
        titles = {
            "f2": r"$F_2$",
            "f1": r"$F_1$",
            "recall": "Recall",
            "precision": "Precision",
            "pr_auc": "PR AUC",
            "total_cost": "Total Cost",
        }

        model_names = list(self.poller.metrics.keys())
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = int(np.ceil(n_metrics / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(-1)

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [self._get_opt_metrics(m).get(metric, 0) for m in model_names]

            df_plot = pd.DataFrame({"model": model_names, "value": values})

            sns.barplot(
                data=df_plot,
                x="model",
                y="value",
                hue="model",
                palette=self.model_colors,  # consistent colours
                ax=ax,
                legend=False,  # avoid duplicate legends
            )

            # Fix the tick issue
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=20, ha="right")

            ax.set_title(titles.get(metric, metric))

            # Annotate values above bars
            for bar, val in zip(ax.patches, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + height * 1e-3,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            # Axis limits
            if metric != "total_cost":
                # Normal metrics are between 0–1
                ax.set_ylim(0, 1)
            else:
                # For total_cost → allow 10% extra space above max bar
                max_val = max(values)
                ax.set_ylim(0, max_val * 1.10)

        # Hide extra axes
        for i in range(n_metrics, len(axes)):
            axes[i].axis("off")

        fig.tight_layout()
        self._save_fig(fig, "metric_comparison_bars", formats, pdf_writer)

    # ---------------------------------------------------------
    #  CONFUSION MATRICES
    # ---------------------------------------------------------

    def plot_confusion_matrices(self, formats=("png",), pdf_writer=None):
        model_names = list(self.poller.confusion_matrices.keys())
        if not model_names:
            return

        n_cols = min(2, len(model_names))
        n_rows = int(np.ceil(len(model_names) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
        axes = np.array(axes).reshape(-1)

        for idx, model in enumerate(model_names):
            cm = np.array(self.poller.confusion_matrices[model])
            ax = axes[idx]

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                ax=ax,
                xticklabels=["0", "1"],
                yticklabels=["0", "1"],
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(model)

        for i in range(len(model_names), len(axes)):
            axes[i].axis("off")

        fig.suptitle("Confusion Matrices (Optimal Threshold)", y=1.02)
        fig.tight_layout()
        self._save_fig(fig, "confusion_matrices", formats, pdf_writer)

    # ---------------------------------------------------------
    #  FEATURE IMPORTANCES
    # ---------------------------------------------------------

    def plot_feature_importances(self, formats=("png",), pdf_writer=None):
        if not self.poller.feature_importances:
            return

        for model, fi in self.poller.feature_importances.items():
            fig, ax = self._fig(w=8, h=4.5)
            items = sorted(fi.items(), key=lambda x: x[1], reverse=True)
            labels, values = zip(*items) if items else ([], [])

            sns.barplot(x=labels, y=values, ax=ax)
            ax.set_title(f"Feature Importances – {model}")
            ax.set_xticklabels(labels, rotation=30, ha="right")
            fig.tight_layout()
            self._save_fig(fig, f"feature_importances_{model}", formats, pdf_writer)

    # ---------------------------------------------------------
    #  CORRELATION HEATMAPS
    # ---------------------------------------------------------

    def plot_correlations(self, formats=("png",), pdf_writer=None):
        if not self.poller.correlation_matrices:
            return

        for name, data in self.poller.correlation_matrices.items():
            fig, ax = self._fig()
            mat = np.array(data["matrix"])
            labels = data["labels"]

            sns.heatmap(
                mat, xticklabels=labels, yticklabels=labels, cmap="coolwarm", center=0, ax=ax
            )
            ax.set_title(f"Correlation Heatmap – {name}")
            fig.tight_layout()
            self._save_fig(fig, f"correlation_{name}", formats, pdf_writer)

    # ============================================================
    #  THRESHOLD DIAGNOSTICS
    # ============================================================

    def plot_threshold_f1(self, formats=("png",), pdf_writer=None):
        fig, ax = self._fig()
        for model, th in self.poller.threshold_curves.items():
            thresholds = np.asarray(th.get("thresholds", []))
            f1 = np.asarray(th.get("f1", []))
            if f1.size == 0:
                continue
            color = self.model_colors.get(model, None)
            ax.plot(thresholds, f1, linewidth=1.8, label=model, color=color)

            th_opt = float(
                self.poller.metrics.get(model, {}).get(
                    "optimal_threshold",
                    self.poller.prediction_history.get(model, {}).get("optimal_threshold", 0.5),
                )
            )
            ax.axvline(th_opt, linestyle=":", linewidth=1.4, color=color)

        ax.set_xlabel("Threshold")
        ax.set_ylabel(r"$F_1$")
        ax.set_title(r"$F_1$ vs Threshold")
        ax.legend()
        self._save_fig(fig, "threshold_f1_only", formats, pdf_writer)

    def plot_threshold_f2_only(self, formats=("png",), pdf_writer=None):
        fig, ax = self._fig()
        for model, th in self.poller.threshold_curves.items():
            thresholds = np.asarray(th.get("thresholds", []))
            f2 = np.asarray(th.get("f2", []))
            if f2.size == 0:
                continue
            color = self.model_colors.get(model, None)
            ax.plot(thresholds, f2, linewidth=1.8, label=model, color=color)

            th_opt = float(
                self.poller.metrics.get(model, {}).get(
                    "optimal_threshold",
                    self.poller.prediction_history.get(model, {}).get("optimal_threshold", 0.5),
                )
            )
            ax.axvline(th_opt, linestyle=":", linewidth=1.4, color=color)

        ax.set_xlabel("Threshold")
        ax.set_ylabel(r"$F_2$")
        ax.set_title(r"$F_2$ vs Threshold")
        ax.legend()
        self._save_fig(fig, "threshold_f2_only", formats, pdf_writer)

    def plot_threshold_f1_f2_combined(self, formats=("png",), pdf_writer=None):
        fig, ax = self._fig()

        for model, th in self.poller.threshold_curves.items():
            thresholds = np.asarray(th.get("thresholds", []))
            f1 = np.asarray(th.get("f1", []))
            f2 = np.asarray(th.get("f2", []))
            color = self.model_colors.get(model, None)

            if f2.size == 0:
                continue

            ax.plot(thresholds, f2, linewidth=1.8, label=rf"{model} – $F_2$", color=color)
            if f1.size == thresholds.size:
                ax.plot(
                    thresholds,
                    f1,
                    linestyle="--",
                    linewidth=1.6,
                    label=rf"{model} – $F_1$",
                )

            th_opt = float(
                self.poller.metrics.get(model, {}).get(
                    "optimal_threshold",
                    self.poller.prediction_history.get(model, {}).get("optimal_threshold", 0.5),
                )
            )
            ax.axvline(th_opt, linestyle=":", linewidth=1.4, alpha=0.8, color=color)

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title(r"Threshold Curves: $F_1$ and $F_2$")
        ax.legend()
        self._save_fig(fig, "threshold_f1_f2_comparison", formats, pdf_writer)

    def plot_f1_f2_region(self, formats=("png",), pdf_writer=None):
        fig, ax = self._fig()

        for model, th in self.poller.threshold_curves.items():
            f1 = np.asarray(th.get("f1", []))
            f2 = np.asarray(th.get("f2", []))
            thresholds = np.asarray(th.get("thresholds", []))
            color = self.model_colors.get(model, None)

            if f1.size == 0 or f2.size == 0:
                continue

            ax.plot(f1, f2, linewidth=1.8, label=model, color=color)

            th_opt = float(
                self.poller.metrics.get(model, {}).get(
                    "optimal_threshold",
                    self.poller.prediction_history.get(model, {}).get("optimal_threshold", 0.5),
                )
            )
            idx = np.argmin(np.abs(thresholds - th_opt))
            ax.scatter(f1[idx], f2[idx], s=70, marker="o")

        ax.set_xlabel(r"$F_1$")
        ax.set_ylabel(r"$F_2$")
        ax.set_title(r"$F_1$ vs $F_2$ Region Map")
        ax.legend()
        self._save_fig(fig, "f1_f2_region_map", formats, pdf_writer)

    # ---------------------------------------------------------
    #  MASTER DIAGNOSTICS
    # ---------------------------------------------------------

    def plot_all_threshold_diagnostics(self, formats=("png",), pdf_writer=None):
        self.plot_threshold_f1(formats, pdf_writer)
        self.plot_threshold_f2_only(formats, pdf_writer)
        self.plot_threshold_f1_f2_combined(formats, pdf_writer)
        self.plot_f1_f2_region(formats, pdf_writer)

    # ---------------------------------------------------------
    #  FULL REPORT
    # ---------------------------------------------------------

    def plot_all(self, formats=("png", "pdf")):
        pdf_path = self.output_dir / "model_comparison_report.pdf"
        pdf_writer = PdfPages(pdf_path) if "pdf" in formats else None

        self.plot_roc(formats, pdf_writer)
        self.plot_pr(formats, pdf_writer)
        self.plot_calibration(formats, pdf_writer)
        self.plot_metric_bars(formats, pdf_writer)
        self.plot_confusion_matrices(formats, pdf_writer)
        self.plot_feature_importances(formats, pdf_writer)
        self.plot_correlations(formats, pdf_writer)
        self.plot_all_threshold_diagnostics(formats, pdf_writer)

        if pdf_writer is not None:
            pdf_writer.close()
            print(f"Saved PDF report to: {pdf_path}")
