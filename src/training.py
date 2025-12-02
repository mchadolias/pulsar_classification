"""
training.py
------------

Defines the ModelTrainer class, the central API for model training, validation,
hyper-parameter search, threshold optimisation, and metric computation.

Main responsibilities:
    • Build sklearn pipelines with preprocessing + classifier
    • Run grid/random/halving search (or simple fit)
    • Compute metrics including ROC, PR-AUC, F1/F2, recall, precision, cost
    • Compute and store calibration statistics
    • Produce individual evaluation plots (saved to outputs/plots/training/)
    • Store threshold curves and evaluation metadata in the Poller

Used by run_training.py as the core training engine.
"""

import pickle
import toml
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    HalvingGridSearchCV,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    f1_score,
    precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    fbeta_score,
    average_precision_score,
    make_scorer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier


class ModelTrainer:
    def __init__(self, config_path: str, logger=None):
        if logger is None:
            self.logger = logging.getLogger("pulsar_ml")
        else:
            self.logger = logger

        config = toml.load(config_path)
        config = self._convert_nulls(config)

        self.model_params = {
            "logistic_regression": config.get("logistic_regression", {}),
            "random_forest": config.get("random_forest", {}),
            "gradient_boosting": config.get("gradient_boosting", {}),
            "xgboost": config.get("xgboost", {}),
        }

        grid_config = config.get("grid", {})
        self.param_grids = {
            "logistic_regression": grid_config.get("logistic_regression", {}),
            "random_forest": grid_config.get("random_forest", {}),
            "gradient_boosting": grid_config.get("gradient_boosting", {}),
            "xgboost": grid_config.get("xgboost", {}),
        }

        self.training_cfg = config.get("training", {})
        self.data_cfg = config.get("data", {})
        self.cv_cfg = config.get("cv", {})
        self.threshold_cfg = config.get("threshold", {})
        self.metrics_cfg = config.get("metrics", {})
        self.costs_cfg = config.get("costs", {})

        self.scoring_metric = self.metrics_cfg.get("primary", "f1")
        self.f2_scorer = make_scorer(fbeta_score, beta=2, average="binary", zero_division=0)

        self.optimal_threshold = 0.5
        self.class_weights = None
        self._sample_X_train = None
        self.calibration_data = {}

        self.models = self._initialize_models_with_weights()
        self.best_model = None
        self.best_model_name = None

    def _convert_nulls(self, obj):
        if isinstance(obj, dict):
            return {k: self._convert_nulls(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_nulls(v) for v in obj]
        return None if obj in ("null", "None", None) else obj

    def set_training_sample(self, X_train):
        self._sample_X_train = X_train

    def _clean_params(self, d):
        return {k: v for k, v in d.items() if v is not None}

    def _initialize_models_with_weights(self):
        use_weights = self.data_cfg.get("class_weights", False)
        strategy = self.data_cfg.get("weight_strategy", "balanced")
        custom = self.data_cfg.get("custom_weights", [1.0, 10.0])

        models = {}

        lr_params = self._clean_params(self.model_params["logistic_regression"].copy())
        if use_weights:
            if strategy == "balanced":
                lr_params["class_weight"] = "balanced"
            elif strategy == "custom":
                lr_params["class_weight"] = {0: custom[0], 1: custom[1]}
        models["logistic_regression"] = LogisticRegression(**lr_params)

        rf_params = self._clean_params(self.model_params["random_forest"].copy())
        if use_weights:
            if strategy == "balanced":
                rf_params["class_weight"] = "balanced"
            elif strategy == "balanced_subsample":
                rf_params["class_weight"] = "balanced_subsample"
            elif strategy == "custom":
                rf_params["class_weight"] = {0: custom[0], 1: custom[1]}
        models["random_forest"] = RandomForestClassifier(**rf_params)

        gb_params = self._clean_params(self.model_params["gradient_boosting"].copy())
        models["gradient_boosting"] = GradientBoostingClassifier(**gb_params)

        xgb_params = self._clean_params(self.model_params["xgboost"].copy())
        models["xgboost"] = XGBClassifier(eval_metric="logloss", **xgb_params)

        return models

    def _compute_scale_pos_weight(self, y):
        pos = int(np.sum(y == 1))
        neg = int(np.sum(y == 0))
        if pos == 0:
            return 1.0
        return max(1.0, neg / pos)

    def make_pipeline(self, model_name: str):
        if self.data_cfg.get("numerical_features"):
            numerical = self.data_cfg.get("numerical_features")
        else:
            if self._sample_X_train is None:
                raise ValueError("Call trainer.set_training_sample(X_train) before training")
            numerical = self._sample_X_train.select_dtypes(include=[np.number]).columns.tolist()
            self.data_cfg["numerical_features"] = numerical
            self.logger.info(f"Auto-detected numerical features: {numerical}")

        pre = ColumnTransformer([("num", StandardScaler(), numerical)])
        pipeline = Pipeline([("preprocessor", pre), ("classifier", self.models[model_name])])
        return pipeline

    def _clean_param_grid(self, grid):
        cleaned = {}
        for k, v in grid.items():
            if isinstance(v, list):
                v = [x for x in v if x is not None]
                if v:
                    cleaned[k] = v
            elif v is not None:
                cleaned[k] = v
        return cleaned

    def train(self, X_train, y_train, model_name: str):
        self.logger.info(f"Starting training for {model_name}...")

        if model_name is None or model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        if self._sample_X_train is None:
            self.set_training_sample(X_train)

        scale = self._compute_scale_pos_weight(y_train)
        if model_name == "xgboost":
            try:
                self.models["xgboost"].set_params(scale_pos_weight=scale)
            except Exception:
                self.logger.warning("Could not set scale_pos_weight on xgboost model")

        pipeline = self.make_pipeline(model_name)
        param_grid = self._clean_param_grid(self.param_grids.get(model_name, {}))

        search_mode = self.training_cfg.get("search", "grid").lower()
        score_function = (
            self.f2_scorer if str(self.scoring_metric).lower() == "f2" else self.scoring_metric
        )
        param_grid_prefixed = {f"classifier__{k}": v for k, v in param_grid.items()}

        if param_grid:
            if search_mode == "random":
                self.logger.info("Using RandomizedSearchCV")
                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=param_grid_prefixed,
                    n_iter=int(self.training_cfg.get("random_iters", 40)),
                    scoring=score_function,
                    cv=int(self.cv_cfg.get("folds", 5)),
                    n_jobs=self.training_cfg.get("n_jobs", -1),
                    verbose=self.training_cfg.get("verbose", 1),
                    random_state=self.cv_cfg.get("random_state", 42),
                )

            elif search_mode == "halving":
                self.logger.info("Using HalvingGridSearchCV (successive halving)")
                search = HalvingGridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid_prefixed,
                    factor=3,
                    scoring=score_function,
                    cv=int(self.cv_cfg.get("folds", 5)),
                    verbose=self.training_cfg.get("verbose", 1),
                    n_jobs=self.training_cfg.get("n_jobs", -1),
                )

            else:
                self.logger.info("Using GridSearchCV")
                search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid_prefixed,
                    scoring=score_function,
                    cv=int(self.cv_cfg.get("folds", 5)),
                    n_jobs=self.training_cfg.get("n_jobs", -1),
                    verbose=self.training_cfg.get("verbose", 1),
                )

            search.fit(X_train, y_train)
            self.best_model = search.best_estimator_
            self.logger.info(f"Best CV score: {search.best_score_:.4f}")
        else:
            self.logger.info("No parameter grid, performing simple fit")
            pipeline.fit(X_train, y_train)
            self.best_model = pipeline

        self.best_model_name = model_name
        self.logger.info(f"Training complete for {model_name}")

    def evaluate(self, X_test, y_test, save_plots: bool = False):
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Call train() first.")

        try:
            y_proba = self.best_model.predict_proba(X_test)[:, 1]
        except Exception:
            y_pred_only = self.best_model.predict(X_test)
            y_proba = np.array(y_pred_only, dtype=float)

        if self.threshold_cfg.get("optimization", True):
            self.optimal_threshold = self._optimize_threshold(y_test, y_proba)
        else:
            self.optimal_threshold = 0.5

        y_pred_default = (y_proba >= 0.5).astype(int)
        y_pred_opt = (y_proba >= self.optimal_threshold).astype(int)

        default_metrics = self._calculate_metrics(y_test, y_pred_default, y_proba)
        optimal_metrics = self._calculate_metrics(y_test, y_pred_opt, y_proba)

        metrics = {
            "default_threshold_0.5": default_metrics,
            f"optimal_threshold_{self.optimal_threshold:.3f}": optimal_metrics,
            "optimal_threshold": float(self.optimal_threshold),
            "model_name": self.best_model_name,
        }

        opt = optimal_metrics
        metrics["roc_auc"] = float(opt.get("roc_auc", 0.0))
        metrics["pr_auc"] = float(opt.get("pr_auc", 0.0))
        metrics["f1"] = float(opt.get("f1", 0.0))
        metrics["f1_score"] = float(opt.get("f1", 0.0))
        metrics["recall"] = float(opt.get("recall", 0.0))
        metrics["precision"] = float(opt.get("precision", 0.0))
        metrics["f2"] = float(opt.get("f2", 0.0))

        try:
            self._calculate_calibration(y_test, y_proba)
            metrics["calibration"] = self.calibration_data
        except Exception as e:
            self.logger.debug(f"Calibration failed: {e}")

        if save_plots:
            try:
                self._generate_evaluation_plots(y_test, y_proba, y_pred_opt)
            except Exception as e:
                self.logger.warning(f"Plot generation failed: {e}")

        return metrics

    def _optimize_threshold(self, y_true, y_proba):
        method = self.threshold_cfg.get("method", "precision_recall")
        num_thresholds = int(self.threshold_cfg.get("num_thresholds", 50))
        beta = float(self.threshold_cfg.get("beta", 2.0))

        if method == "precision_recall":
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
            f_scores = (
                (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-12)
            )
            best_idx = int(np.nanargmax(f_scores[:-1])) if len(f_scores) > 1 else 0
            best_threshold = (
                pr_thresholds[min(best_idx, len(pr_thresholds) - 1)]
                if len(pr_thresholds) > 0
                else 0.5
            )
            return float(best_threshold)

        thresholds = np.linspace(0.01, 0.99, num_thresholds)
        best_th = 0.5
        best_score = -1
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
            if score > best_score:
                best_score = score
                best_th = float(t)
        return best_th

    def _calculate_metrics(self, y_true, y_pred, y_proba=None):
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                roc = float(roc_auc_score(y_true, y_proba))
            except Exception:
                roc = 0.0
            try:
                pr_auc = float(average_precision_score(y_true, y_proba))
            except Exception:
                pr_auc = 0.0
        else:
            roc = 0.0
            pr_auc = 0.0

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = int(cm[0, 0]) if cm.size > 0 else 0
            fp = fn = tp = 0

        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        f2 = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))

        try:
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        except Exception:
            report = {}

        fp_cost = float(self.costs_cfg.get("false_positive", 1))
        fn_cost = float(self.costs_cfg.get("false_negative", 10))
        total_cost = fp * fp_cost + fn * fn_cost

        return {
            "roc_auc": roc,
            "pr_auc": pr_auc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "f2": f2,
            "confusion_matrix": cm.tolist(),
            "confusion_counts": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "total_cost": float(total_cost),
            "classification_report": report,
        }

    def _calculate_calibration(self, y_true, y_proba, n_bins: int = 10):
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
        self.calibration_data = {
            "prob_true": [float(x) for x in prob_true.tolist()],
            "prob_pred": [float(x) for x in prob_pred.tolist()],
        }

    def _generate_evaluation_plots(self, y_true, y_proba, y_pred_opt=None):
        """
        Generate ROC, Precision–Recall, and Threshold (F1/F2) curves
        during training evaluation. Also stores threshold curves in the Poller.
        The third parameter is accepted for backward compatibility.
        """
        output_dir = Path("outputs/plots/training")
        output_dir.mkdir(parents=True, exist_ok=True)

        # ======================================================
        # 1) ROC Curve
        # ======================================================
        if len(np.unique(y_true)) == 2:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                auc_score = roc_auc_score(y_true, y_proba)

                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
                plt.plot([0, 1], [0, 1], "k--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve – {self.best_model_name}")
                plt.legend()
                plt.savefig(
                    output_dir / f"roc_{self.best_model_name}.png", dpi=150, bbox_inches="tight"
                )
                plt.close()
            except Exception as e:
                self.logger.debug(f"ROC plot failed: {e}")

        # ======================================================
        # 2) Precision–Recall Curve
        # ======================================================
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            pr_auc = average_precision_score(y_true, y_proba)

            plt.figure(figsize=(6, 5))
            plt.plot(recall, precision, label=f"AP = {pr_auc:.3f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision–Recall – {self.best_model_name}")
            plt.legend()
            plt.savefig(
                output_dir / f"pr_{self.best_model_name}.png", dpi=150, bbox_inches="tight"
            )
            plt.close()
        except Exception as e:
            self.logger.debug(f"PR plot failed: {e}")

        thresholds, f1_scores, f2_scores = None, None, None

        # ======================================================
        # 3) Threshold analysis (F1 and F2)
        # ======================================================
        try:
            thresholds = np.linspace(0.01, 0.99, 100)
            f1_scores = []
            f2_scores = []

            for t in thresholds:
                preds_t = (y_proba >= t).astype(int)
                f1_scores.append(f1_score(y_true, preds_t, zero_division=0))
                f2_scores.append(fbeta_score(y_true, preds_t, beta=2, zero_division=0))

            plt.figure(figsize=(6, 5))
            plt.plot(thresholds, f1_scores, linestyle="--", label="F1")
            plt.plot(thresholds, f2_scores, label="F2")
            plt.axvline(
                self.optimal_threshold,
                color="r",
                linestyle="--",
                label=f"Optimal = {self.optimal_threshold:.3f}",
            )
            plt.axvline(0.5, color="g", linestyle="--", label="Default = 0.5")
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.title(f"Threshold Analysis – {self.best_model_name}")
            plt.legend()
            plt.savefig(
                output_dir / f"threshold_{self.best_model_name}.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

        except Exception as e:
            self.logger.debug(f"Threshold analysis plot failed: {e}")

        # ======================================================
        # 4) Store threshold curves in Poller (always)
        # ======================================================
        try:
            if thresholds is not None:
                self.poller.add_threshold_curve(
                    self.best_model_name, thresholds, f1_scores, f2_scores
                )
        except Exception as e:
            self.logger.debug(f"Saving threshold curves failed: {e}")

        self.logger.debug("Generated evaluation plots and threshold data.")

    def predict_with_threshold(self, X, threshold=None):
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Call train() first.")
        if threshold is None:
            threshold = self.optimal_threshold

        try:
            proba = self.best_model.predict_proba(X)[:, 1]
        except Exception:
            y_pred_only = self.best_model.predict(X)
            proba = np.array(y_pred_only, dtype=float)

        preds = (proba >= threshold).astype(int)
        return {"predictions": preds, "probabilities": proba, "threshold": threshold}

    def get_feature_importances(self, feature_names=None):
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Call train() first.")
        try:
            model = self.best_model.named_steps["classifier"]
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                if feature_names is None or len(feature_names) != len(importances):
                    feature_names = [f"feature_{i}" for i in range(len(importances))]
                return dict(zip(feature_names, importances))
            elif hasattr(model, "coef_"):
                coefs = model.coef_
                if coefs.ndim == 2:
                    coefs = coefs[0]
                if feature_names is None or len(feature_names) != len(coefs):
                    feature_names = [f"feature_{i}" for i in range(len(coefs))]
                return dict(zip(feature_names, coefs))
            else:
                self.logger.warning("Model does not expose feature_importances_ or coef_")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to extract feature importances: {e}")
            return {}

    def predict(self, X):
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Call train() first.")
        return self.best_model.predict(X)

    def predict_proba(self, X):
        if self.best_model is None:
            raise ValueError("No model has been trained yet. Call train() first.")
        return self.best_model.predict_proba(X)

    def save_model(self, file_path=None):
        if self.best_model is None:
            raise ValueError("No model to save. Train a model first.")
        if file_path is None:
            file_path = self.training_cfg.get("save_path", "models/best_model.pkl")
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.best_model, f)

    def load_model(self, file_path):
        with open(file_path, "rb") as f:
            self.best_model = pickle.load(f)
        self.logger.info(f"Model loaded from {file_path}")
