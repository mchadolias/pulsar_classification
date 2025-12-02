# scripts/run_training.py

"""
Main training pipeline for HTRU2 Pulsar Classification.

Features:
- Imbalance handling & F2-optimised thresholding
- Model comparison across LR, RF, GBDT, XGBoost
- tqdm progress bar for model training loop
- Centralised coloured logging via LoggerManager
- Poller for collecting metrics/history/curves
- Optional combined comparison plots via ModelComparisonPlotter
- Optional dry-run mode to validate wiring without training
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    roc_curve,
    average_precision_score,
    fbeta_score,
    precision_recall_curve,
    f1_score,
)

# Ensure src is on Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import DataConfig
from src.data_handler import HTRU2DataHandler
from src.training import ModelTrainer
from src.plotter import ModelComparisonPlotter
from src.poller import Poller
from src.utils import (
    LoggerManager,
    check_data_balance,
    save_model_comparison,
    save_test_predictions,
    save_final_results,
    save_training_history,
    NumpyEncoder,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="HTRU2 Pulsar Classification ML Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML config file (e.g. model.toml).",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Generate combined comparison plots after training.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (more verbose output).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run full data pipeline setup without training any models.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    log_manager = LoggerManager(log_dir="logs", level=(0 if args.debug else 20))
    logger = log_manager.get_logger()
    if args.debug:
        log_manager.enable_debug()

    logger.info("Starting HTRU2 Pulsar Classification Pipeline…")
    logger.info(f"Log file: {log_manager.get_log_path()}")
    logger.info(f"Working directory: {os.getcwd()}")

    # -------- Config file selection -------- #
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        logger.info(f"Using config file from CLI: {config_path}")
    else:
        candidates = [
            "model.toml",
            "test_model.toml",
            "configs/model.toml",
            "configs/test_model.toml",
        ]
        config_path = None
        for c in candidates:
            p = Path(c)
            if p.exists():
                config_path = p.resolve()
                logger.info(f"Using default config file: {config_path}")
                break
        if config_path is None:
            raise FileNotFoundError(
                "No config file provided and no default model.toml / test_model.toml found."
            )

    # -------- Data loading & preprocessing -------- #
    data_config = DataConfig()
    data_handler = HTRU2DataHandler(data_config, logger)

    log_manager.section("STEP 1: DOWNLOADING DATA")
    data_handler.download_kaggle()

    log_manager.section("STEP 2: LOADING DATA")
    df = data_handler.load()
    logger.info(f"Dataset shape: {df.shape}")

    log_manager.section("STEP 3: PREPROCESSING DATA")
    df_processed = data_handler.preprocess()
    logger.info(f"Processed dataset shape: {df_processed.shape}")

    check_data_balance(df_processed, logger=logger)

    log_manager.section("STEP 4: SPLITTING DATA")
    splits = data_handler.split_train_val_test()
    data_handler.export_splits()

    target_col = "signal"
    feature_cols = [c for c in df_processed.columns if c != target_col]

    X_train = splits["train"][feature_cols]
    y_train = splits["train"][target_col]
    X_val = splits["val"][feature_cols]
    y_val = splits["val"][target_col]
    X_test = splits["test"][feature_cols]
    y_test = splits["test"][target_col]

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # -------- Optional dry-run: stop here -------- #
    if args.dry_run:
        log_manager.section("DRY-RUN MODE SUMMARY")
        logger.info(f"Config path: {config_path}")
        logger.info("Dataset shapes:")
        logger.info(f"  Train: {X_train.shape}")
        logger.info(f"  Val:   {X_val.shape}")
        logger.info(f"  Test:  {X_test.shape}")
        logger.info("Feature columns:")
        for col in feature_cols:
            logger.info(f"  - {col}")
        logger.info(
            "Models to be trained: logistic_regression, random_forest, gradient_boosting, xgboost"
        )
        logger.info("Dry run complete — no model training performed. Exiting.")
        return

    # -------- Trainer -------- #
    log_manager.section("STEP 5: INITIALIZING TRAINER")
    trainer = ModelTrainer(str(config_path), logger)
    trainer.data_cfg["numerical_features"] = feature_cols
    logger.info(f"Using numerical features: {feature_cols}")

    # Poller instance
    poller = Poller()

    # Correlation matrix once (for features)
    try:
        corr = df_processed[feature_cols].corr().values
        poller.add_correlation_matrix("features", corr, feature_cols)
    except Exception as e:
        logger.warning(f"Could not compute feature correlation matrix: {e}")

    # -------- Train models with tqdm -------- #
    log_manager.section("STEP 6: TRAINING MODELS")

    models_to_train = ["logistic_regression", "random_forest", "gradient_boosting", "xgboost"]
    all_results = {}
    prediction_history = {}
    best_f2_score = 0.0
    best_model_name = None

    for model_name in tqdm(models_to_train, desc="Training Models", colour="green"):
        log_manager.section(f"TRAINING {model_name.upper()}")
        logger.info(f"Model: {model_name}")

        try:
            trainer.train(X_train, y_train, model_name)

            metrics = trainer.evaluate(X_val, y_val, save_plots=True)
            all_results[model_name] = metrics
            poller.add_metrics(model_name, metrics)

            opt_key = next((k for k in metrics if k.startswith("optimal_threshold_")), None)
            opt_metrics = metrics.get(opt_key, metrics)

            logger.info(
                "Validation (optimal threshold): "
                f"F2={opt_metrics.get('f2', 0):.4f}, "
                f"F1={opt_metrics.get('f1', 0):.4f}, "
                f"Recall={opt_metrics.get('recall', 0):.4f}, "
                f"Precision={opt_metrics.get('precision', 0):.4f}, "
                f"ROC-AUC={opt_metrics.get('roc_auc', 0):.4f}, "
                f"PR-AUC={opt_metrics.get('pr_auc', 0):.4f}"
            )

            # Prediction history
            y_val_proba = trainer.predict_proba(X_val)[:, 1]
            thresh = metrics.get("optimal_threshold", trainer.optimal_threshold)
            y_val_pred = (y_val_proba >= thresh).astype(int)

            hist = {
                "y_true": y_val.tolist(),
                "y_proba": y_val_proba.tolist(),
                "y_pred_opt": y_val_pred.tolist(),
                "optimal_threshold": float(thresh),
                "metrics": metrics,
            }
            prediction_history[model_name] = hist
            poller.add_prediction_history(model_name, hist)

            # Confusion matrix for Poller (from metrics)
            cm = opt_metrics.get("confusion_matrix", None)
            if cm is not None:
                poller.add_confusion_matrix(model_name, cm)

            # ROC & PR curves & threshold-F2 curves from validation
            try:
                fpr, tpr, _ = roc_curve(y_val, y_val_proba)
                auc_val = opt_metrics.get("roc_auc", 0.0)
                poller.add_roc_curve(model_name, fpr, tpr, auc_val)

                precision, recall, pr_th = precision_recall_curve(y_val, y_val_proba)
                ap_val = opt_metrics.get(
                    "pr_auc",
                    average_precision_score(y_val, y_val_proba),
                )
                poller.add_pr_curve(model_name, precision, recall, ap_val)

                thresholds = np.linspace(0.01, 0.99, 100)

                f1_scores = [
                    f1_score(y_val, (y_val_proba >= t).astype(int), zero_division=0)
                    for t in thresholds
                ]
                f2_scores = [
                    fbeta_score(y_val, (y_val_proba >= t).astype(int), beta=2, zero_division=0)
                    for t in thresholds
                ]

                poller.add_threshold_curve(model_name, thresholds, f1_scores, f2_scores)

            except Exception as e:
                logger.debug(f"Could not compute curves for {model_name}: {e}")

            current_f2 = opt_metrics.get("f2", 0.0)
            if current_f2 > best_f2_score:
                best_f2_score = current_f2
                best_model_name = model_name

        except Exception as e:
            logger.error(f"Training failed for {model_name}: {e}", exc_info=True)
            continue

    if best_model_name is None:
        logger.error("No model trained successfully; aborting.")
        return

    save_model_comparison(all_results, best_model_name, best_f2_score, logger)

    # Save raw prediction_history as simple JSON (legacy)
    history_path_simple = Path("outputs/metrics/model_prediction_history.json")
    with open(history_path_simple, "w") as f:
        json.dump(prediction_history, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Saved model prediction history → {history_path_simple}")

    # -------- Retrain best model on full train+val -------- #
    log_manager.section("STEP 7: RETRAINING BEST MODEL ON FULL TRAINING DATA")

    X_full_train = pd.concat([X_train, X_val], axis=0)
    y_full_train = pd.concat([y_train, y_val], axis=0)

    logger.info(f"Best model: {best_model_name} (F2 = {best_f2_score:.4f})")
    trainer.train(X_full_train, y_full_train, best_model_name)

    # -------- Final test evaluation -------- #
    log_manager.section("STEP 8: FINAL TEST EVALUATION")

    test_metrics = trainer.evaluate(X_test, y_test, save_plots=True)
    opt_key_test = next((k for k in test_metrics if k.startswith("optimal_threshold_")), None)
    opt_test_metrics = test_metrics.get(opt_key_test, test_metrics)

    logger.info(
        "Test (optimal threshold): "
        f"F2={opt_test_metrics.get('f2', 0):.4f}, "
        f"F1={opt_test_metrics.get('f1', 0):.4f}, "
        f"Recall={opt_test_metrics.get('recall', 0):.4f}, "
        f"Precision={opt_test_metrics.get('precision', 0):.4f}, "
        f"ROC-AUC={opt_test_metrics.get('roc_auc', 0):.4f}, "
        f"PR-AUC={opt_test_metrics.get('pr_auc', 0):.4f}"
    )

    save_test_predictions(trainer, X_test, y_test, best_model_name, logger)

    # -------- Feature importances for best model -------- #
    log_manager.section("STEP 9: FEATURE IMPORTANCES")

    try:
        feature_importances = trainer.get_feature_importances(feature_cols)
        if feature_importances:
            for feat, imp in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"{feat}: {imp:.4f}")
            poller.add_feature_importances(best_model_name, feature_importances)
        else:
            feature_importances = {}
    except Exception as e:
        logger.warning(f"Could not compute feature importances: {e}")
        feature_importances = {}

    # -------- Save final outputs -------- #
    log_manager.section("STEP 10: SAVING FINAL OUTPUTS")

    model_path = f"outputs/models/best_{best_model_name}.pkl"
    trainer.save_model(model_path)

    save_final_results(
        best_model_name,
        test_metrics,
        feature_importances,
        feature_cols,
        config_path,
        logger,
    )

    save_training_history(all_results, best_model_name, data_handler, logger)

    # Save Poller state
    poller_path = poller.save()
    logger.info(f"Poller state saved → {poller_path}")

    # -------- Optional combined plots -------- #
    if args.compare_models:
        log_manager.section("STEP 11: COMBINED MODEL COMPARISON PLOTS")
        try:
            plotter = ModelComparisonPlotter.from_poller_file(str(poller_path))
            plotter.plot_all(formats=("png", "pdf"))
            logger.info("Combined comparison plots generated.")
        except Exception as e:
            logger.error(f"Failed to generate comparison plots: {e}", exc_info=True)

    logger.info("Pipeline completed successfully.")
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Optimal threshold: {trainer.optimal_threshold:.4f}")


if __name__ == "__main__":
    main()
