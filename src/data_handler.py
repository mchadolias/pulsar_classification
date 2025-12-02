"""
data_handler.py
----------------

Contains HTRU2DataHandler, responsible for the full data pipeline:

    • Download HTRU2 dataset from Kaggle (if missing)
    • Load raw CSV into pandas
    • Rename columns to meaningful names
    • Validate missing values
    • Round numerical columns
    • Split the dataset into train/val/test
    • Export cleaned data and splits

This module isolates all data-loading details from the ML training code.
"""

import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi
from src.config import DataConfig

COLUMN_NAMES = [
    "ip_mean",
    "ip_std",
    "ip_kurtosis",
    "ip_skewness",
    "dm_mean",
    "dm_std",
    "dm_kurtosis",
    "dm_skewness",
    "signal",
]


class HTRU2DataHandler:
    """Handles downloading, loading, and preprocessing of the HTRU2 dataset."""

    def __init__(self, config: DataConfig, logger=None):
        # Set up logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        self.logger.info("Initializing HTRU2DataHandler...")

        self.config = config
        self.data_dir = config.data_dir
        self.external_dir = os.path.join(self.data_dir, config.external_dir)
        self.processed_dir = os.path.join(self.data_dir, config.processed_dir)
        self.data_file = os.path.join(self.external_dir, "HTRU_2.csv")
        self.output_file = self.processed_dir  # This should be a directory, not a file path

        self.df = None
        self.splits = {}

        # Create directories safely
        self.logger.info("Creating directory structure...")
        self.logger.info(f"Data directory: {self.data_dir}")
        self.logger.info(f"External data: {self.external_dir}")
        self.logger.info(f"Processed data: {self.processed_dir}")

        os.makedirs(self.external_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        self.logger.info("HTRU2DataHandler initialized successfully")

    def download_kaggle(self):
        """Download dataset from Kaggle if not found locally."""
        self.logger.info("Checking dataset availability...")

        if os.path.exists(self.data_file):
            self.logger.info("Dataset already exists locally.")
            self.logger.info(f"Location: {self.data_file}")
            return

        self.logger.info("Downloading dataset from Kaggle...")
        self.logger.info(f"Dataset: {self.config.dataset}")
        self.logger.info(f"Target directory: {self.external_dir}")

        try:
            api = KaggleApi()
            self.logger.debug("Authenticating with Kaggle API...")
            api.authenticate()

            self.logger.info("Starting dataset download...")
            api.dataset_download_files(self.config.dataset, path=self.external_dir, unzip=True)

            # Check if download was successful
            if os.path.exists(self.data_file):
                self.logger.info("Download complete.")
                self.logger.info(f"Dataset saved to: {self.data_file}")
            else:
                # Check for any downloaded files
                downloaded_files = os.listdir(self.external_dir)
                if downloaded_files:
                    self.logger.warning(
                        f"Download completed but expected file not found. Found files: {downloaded_files}"
                    )
                    # Try to find and rename the correct file
                    for file in downloaded_files:
                        if file.endswith(".csv"):
                            original_path = os.path.join(self.external_dir, file)
                            new_path = os.path.join(self.external_dir, "HTRU_2.csv")
                            os.rename(original_path, new_path)
                            self.logger.info(f"Renamed {file} to HTRU_2.csv")
                            break
                else:
                    raise FileNotFoundError("No files were downloaded from Kaggle")

        except Exception as e:
            self.logger.error(f"Error downloading dataset: {e}")
            self.logger.error("Please ensure:")
            self.logger.error("1. Kaggle API is properly configured")
            self.logger.error("2. You have access to the dataset")
            self.logger.error("3. Internet connection is available")
            raise

    def load(self):
        """Load the dataset into a pandas DataFrame."""
        self.logger.info("Loading data...")

        if not os.path.exists(self.data_file):
            error_msg = f"Data file not found: {self.data_file}. Run download() first."
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        self.logger.info(f"Reading from: {self.data_file}")

        try:
            # No header in original dataset
            self.df = pd.read_csv(self.data_file, header=None)

            self.logger.info("Data loaded successfully.")
            self.logger.info(f"Dataset shape: {self.df.shape}")
            self.logger.info(f"Columns: {self.df.shape[1]}")
            self.logger.info(f"Rows: {self.df.shape[0]}")
            self.logger.info(
                f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            )

            # Log basic info about the raw data
            self.logger.debug("Raw data info:")
            self.logger.debug(f"Data types:\n{self.df.dtypes}")
            self.logger.debug(f"First few rows:\n{self.df.head(2)}")

            return self.df

        except Exception as e:
            self.logger.error(f"[ERROR] Error loading data: {e}")
            raise

    def get_column_types(self):
        """Get the data types of each column in the DataFrame."""
        if self.df is None:
            error_msg = "Data not loaded. Call load() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.debug("Analyzing column data types...")

        # Check numerical values
        features = self.df.columns[(self.df.columns != "signal")].tolist()
        numerical_cols = (
            self.df[features].select_dtypes(include=["int64", "float64"]).columns.tolist()
        )
        categorical_cols = (
            self.df[features].select_dtypes(include=["object", "category"]).columns.tolist()
        )

        self.logger.debug(f"Numerical columns: {numerical_cols}")
        self.logger.debug(f"Categorical columns: {categorical_cols}")

        return numerical_cols, categorical_cols

    def preprocess(self):
        """Preprocess the dataset: rename columns, check for missing values, and round numerical values."""
        if self.df is None:
            error_msg = "Data not loaded. Call load() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info("Preprocessing data...")
        self.logger.info(f"Initial shape: {self.df.shape}")

        try:
            # Rename columns
            self.logger.info("Renaming columns...")
            self.logger.debug(f"New column names: {COLUMN_NAMES}")
            self.df.columns = COLUMN_NAMES
            self.logger.info("Columns renamed")

            # Check for missing values
            self.logger.info("Checking for missing values...")
            missing_values = self.df.isnull().sum().sum()
            if missing_values > 0:
                missing_details = self.df.isnull().sum()
                self.logger.error(f"[ERROR] Dataset contains {missing_values} missing values:")
                for col, missing_count in missing_details[missing_details > 0].items():
                    self.logger.error(f"    {col}: {missing_count} missing values")
                raise ValueError("Dataset contains missing values.")
            self.logger.info("No missing values found")

            # Get numerical columns and apply rounding
            numerical_cols, _ = self.get_column_types()
            self.logger.info(
                f"Rounding {len(numerical_cols)} numerical columns to 3 decimal places..."
            )

            # Log some basic statistics before rounding
            self.logger.debug("Numerical columns statistics before rounding:")
            for col in numerical_cols[:3]:  # Log first 3 columns to avoid too much output
                self.logger.debug(
                    f"    {col}: min={self.df[col].min():.3f}, max={self.df[col].max():.3f}, mean={self.df[col].mean():.3f}"
                )

            self.df[numerical_cols] = self.df[numerical_cols].apply(lambda x: round(x, 3))
            self.logger.info("Numerical values rounded")

            self.logger.info("Preprocessing complete.")
            self.logger.info(f"Final shape: {self.df.shape}")
            self.logger.debug(f"Processed data sample:\n{self.df.head(2)}")

            return self.df

        except Exception as e:
            self.logger.error(f"[ERROR] Error during preprocessing: {e}")
            raise

    def split_train_val_test(self):
        """Split the dataset into training, validation, and test sets."""
        if self.df is None:
            error_msg = "Data not loaded. Call load() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info("Splitting data into train, val, and test sets...")
        self.logger.info(f"Test size: {self.config.test_size}")
        self.logger.info(f"Validation size: {self.config.val_size}")
        self.logger.info(f"Random state: {self.config.random_state}")

        try:
            # First split: full train + test
            self.logger.info("  Performing train-test split...")
            full_train_df, test_df = train_test_split(
                self.df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=self.df["signal"],
            )

            # Second split: train + val from full train
            self.logger.info("  Performing train-validation split...")
            train_df, val_df = train_test_split(
                full_train_df,
                test_size=self.config.val_size,
                random_state=self.config.random_state,
                stratify=full_train_df["signal"],
            )

            self.splits = {
                "train": train_df.reset_index(drop=True),
                "val": val_df.reset_index(drop=True),
                "test": test_df.reset_index(drop=True),
            }

            # Log split statistics
            self.logger.info("Data splitting complete.")
            self.logger.info("Split Statistics:")
            for split_name, df_split in self.splits.items():
                split_size = len(df_split)
                total_size = len(self.df)
                percentage = (split_size / total_size) * 100
                class_distribution = dict(df_split["signal"].value_counts().sort_index())
                self.logger.info(
                    f"  {split_name:8}: {split_size:6} samples ({percentage:5.1f}%) - Classes: {class_distribution}"
                )

            return self.splits

        except Exception as e:
            self.logger.error(f"[ERROR] Error during data splitting: {e}")
            raise

    def export_splits(self):
        """Export the train, validation, and test splits to CSV files."""
        if not self.splits:
            error_msg = "Data splits not found. Call split_train_val_test() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info("Exporting data splits to CSV files...")
        self.logger.info(f"Output directory: {self.processed_dir}")

        try:
            exported_files = []
            for split_name, df_split in self.splits.items():
                file_path = os.path.join(self.processed_dir, f"{split_name}_data.csv")
                df_split.to_csv(file_path, index=False)
                exported_files.append(file_path)
                self.logger.info(
                    f"Exported {split_name} set to ./data/processed/{split_name}_data.csv"
                )
                self.logger.debug(f"    {split_name} shape: {df_split.shape}")

            self.logger.info("All data splits exported successfully.")
            self.logger.info(f"Total files exported: {len(exported_files)}")

        except Exception as e:
            self.logger.error(f"[ERROR] Error exporting data splits: {e}")
            raise

    def export_cleaned_data(self):
        """Export the cleaned full dataset to a CSV file."""
        if self.df is None:
            error_msg = "Data not loaded. Call load() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info("Exporting cleaned full dataset...")

        try:
            file_path = os.path.join(self.processed_dir, "HTRU2_cleaned_data.csv")
            self.df.to_csv(file_path, index=False)

            self.logger.info("Exported cleaned data to {file_path}.")
            self.logger.info(f"  File size: {os.path.getsize(file_path) / 1024**2:.2f} MB")
            self.logger.info(f"  Dataset shape: {self.df.shape}")

        except Exception as e:
            self.logger.error(f"[ERROR] Error exporting cleaned data: {e}")
            raise
