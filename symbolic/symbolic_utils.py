"""
Utility functions for symbolic regression with PySR.
"""

import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

TARGET_COL_DEFAULT = "FM_data_peak_distorted_echo_power"


def setup_logger(output_dir: str, log_filename: str = "pysr.log") -> str:
    """Configure logging to file + stdout in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Logger initialized. Log file: {log_path}")
    return log_path


def load_xy(path: str, target_col: str = TARGET_COL_DEFAULT) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load CSV and return (X, y, feature_names)."""
    logging.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    logging.info(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    y = df[target_col].values
    X_df = df.drop(columns=[target_col]).select_dtypes(include=["number"])
    feature_names = list(X_df.columns)

    logging.info(f"Using {len(feature_names)} numeric features for symbolic regression.")

    if X_df.empty:
        raise ValueError("No numeric feature columns found after dropping the target.")

    return X_df.values, y, feature_names


def evaluate_split(
    model,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
) -> Dict[str, float]:
    """Compute MSE and R² for a given split."""
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    logging.info(f"[{split_name}] MSE={mse:.6f} | R²={r2:.6f}")
    return {"mse": float(mse), "r2": float(r2)}


def save_pysr_results(
    model,
    output_dir: str,
    metrics: Dict[str, Dict[str, float]],
    equations_filename: str = "pysr_equations.csv",
    metrics_filename: str = "pysr_metrics.json",
) -> None:
    """Save equations table and metrics to disk."""
    os.makedirs(output_dir, exist_ok=True)

    equations_path = os.path.join(output_dir, equations_filename)
    model.equations_.to_csv(equations_path, index=False)
    logging.info(f"PySR equations saved to: {equations_path}")

    metrics_path = os.path.join(output_dir, metrics_filename)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"PySR metrics saved to: {metrics_path}")

    try:
        best_eq = model.get_best()
        logging.info("Best equation found by PySR:")
        logging.info("\n" + str(best_eq))
    except Exception as e:
        logging.warning(f"Could not retrieve best equation from PySR model: {e}")


def save_test_predictions(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    output_dir: str,
    filename: str = "pysr_test_predictions.csv",
) -> None:
    """Save test predictions with features, y_true and y_pred."""
    os.makedirs(output_dir, exist_ok=True)
    y_pred = model.predict(X)

    df = pd.DataFrame(X, columns=feature_names)
    df["y_true"] = y
    df["y_pred"] = y_pred

    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    logging.info(f"Test predictions saved to: {path}")
