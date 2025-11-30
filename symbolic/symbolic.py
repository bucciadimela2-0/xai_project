
"""
Entry point for running symbolic regression with PySR on the Mars radar dataset.

This module:
- Parses command-line arguments
- Loads data and prepares train/val/test splits (internal or external)
- Configures and trains a PySRRegressor model
- Evaluates the model and saves equations, metrics, and test predictions
"""

import argparse
import logging
from typing import Optional

import numpy as np
from pysr import PySRRegressor

from symbolic_utils import (
    TARGET_COL_DEFAULT,
    evaluate_split,
    load_xy,
    save_pysr_results,
    save_test_predictions,
    setup_logger,
)


def train_symbolic_regression(
    train_csv: str,
    val_csv: Optional[str],
    test_csv: Optional[str],
    output_dir: str,
    target_col: str = TARGET_COL_DEFAULT,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> None:
    """
    Train a PySR symbolic regressor given train/val/test CSVs or internal splits.

    Modes
    -----
    1. Internal split (no val_csv, no test_csv):
        - The file at `train_csv` is split into train/val/test using
          `val_ratio` and `test_ratio`.
    2. External validation + optional test:
        - `train_csv` is used as TRAIN
        - `val_csv` is used as VAL
        - `test_csv`, if provided, is used only for evaluation (never for training)

    Parameters
    ----------
    train_csv : str
        Path to the training CSV, or the full dataset if using internal splits.
    val_csv : str or None
        Path to the validation CSV, or None to trigger internal splitting.
    test_csv : str or None
        Path to the test CSV. If None and using internal split, test is created
        from the training file.
    output_dir : str
        Directory where model outputs (metrics, equations, predictions) will be saved.
    target_col : str, default=TARGET_COL_DEFAULT
        Name of the target column.
    val_ratio : float, default=0.15
        Validation fraction when using internal splits.
    test_ratio : float, default=0.15
        Test fraction when using internal splits.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    # Load full training file (for either internal or external split modes)
    X_all, y_all, feature_names = load_xy(train_csv, target_col=target_col)

    use_internal_test = False
    X_train = X_val = X_test = None
    y_train = y_val = y_test = None

    # CASE 1: internal split (no val_csv and no test_csv)
    if val_csv is None and test_csv is None:
        logging.info(
            "No val_csv or test_csv provided. Performing internal split "
            f"with val_ratio={val_ratio}, test_ratio={test_ratio}."
        )

        total = len(X_all)
        if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1.0:
            raise ValueError(
                f"Invalid val_ratio + test_ratio: {val_ratio} + {test_ratio} "
                "must be < 1.0 and >= 0."
            )

        n_val = int(total * val_ratio)
        n_test = int(total * test_ratio)

        if n_val == 0 or n_test == 0:
            raise ValueError(
                "val_ratio or test_ratio too small for dataset of size "
                f"{total}. Got n_val={n_val}, n_test={n_test}."
            )

        rng = np.random.default_rng(random_state)
        indices = rng.permutation(total)

        val_idx = indices[:n_val]
        test_idx = indices[n_val:n_val + n_test]
        train_idx = indices[n_val + n_test:]

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        use_internal_test = True

        logging.info(
            "Internal split sizes -> "
            f"TRAIN: {len(y_train)}, VAL: {len(y_val)}, TEST: {len(y_test)}"
        )

    else:
        # CASE 2: external validation and/or test
        X_train, y_train = X_all, y_all

        # External VAL is required in this mode
        if val_csv is not None:
            logging.info(f"Using separate validation CSV: {val_csv}")
            X_val, y_val, feature_names_val = load_xy(val_csv, target_col=target_col)

            if len(feature_names_val) != len(feature_names):
                raise ValueError(
                    "Train and validation feature dimensions do not match: "
                    f"train={len(feature_names)}, val={len(feature_names_val)}"
                )
        else:
            # To avoid surprising behavior, require val_csv if test_csv is provided.
            raise ValueError(
                "val_csv is None but test_csv is provided. "
                "For this mode, please provide val_csv as well, or omit both "
                "to use internal splitting."
            )

        # External TEST (optional)
        if test_csv is not None:
            logging.info(f"Using separate TEST CSV: {test_csv}")
            X_test, y_test, feature_names_test = load_xy(test_csv, target_col=target_col)

            if len(feature_names_test) != len(feature_names):
                raise ValueError(
                    "Train and test feature dimensions do not match: "
                    f"train={len(feature_names)}, test={len(feature_names_test)}"
                )

    logging.info(f"Final TRAIN size: {len(y_train)}  |  VAL size: {len(y_val)}")

    # Configure PySR model
    model = PySRRegressor(
        niterations=80,                   # can be increased for better performance
        populations=10,
        population_size=1000,
        elementwise_loss="L2DistLoss()",  # updated loss name
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "log", "exp"],
        maxsize=20,
        maxdepth=6,
        model_selection="best",
        progress=True,
        verbosity=1,
        random_state=random_state,
        deterministic=True,
        parallelism="serial",
        # NOTE: feature_names parameter is not used here due to version constraints.
    )

    logging.info("Starting PySR training...")
    model.fit(X_train, y_train)
    logging.info("PySR training completed.")

    # Evaluate on TRAIN and VAL
    metrics = {}
    metrics["train"] = evaluate_split(model, X_train, y_train, split_name="TRAIN")
    metrics["val"] = evaluate_split(model, X_val, y_val, split_name="VAL")

    # Evaluate on TEST (internal or external)
    if use_internal_test or test_csv is not None:
        if X_test is None or y_test is None:
            raise RuntimeError(
                "Internal logic error: TEST arrays are None but test evaluation was requested."
            )

        metrics["test"] = evaluate_split(model, X_test, y_test, split_name="TEST")

        # Save detailed predictions for the test set
        save_test_predictions(
            model=model,
            X=X_test,
            y=y_test,
            feature_names=feature_names,
            output_dir=output_dir,
            filename="pysr_test_predictions.csv",
        )

    # Save discovered equations and metrics
    save_pysr_results(model, output_dir=output_dir, metrics=metrics)


def parse_args():
    """
    Parse command-line arguments for the symbolic regression script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run symbolic regression with PySR on a Mars radar dataset subset."
    )

    parser.add_argument(
        "--train_csv",
        required=True,
        help="Path to training CSV (e.g., a sampled or clustered subset).",
    )
    parser.add_argument(
        "--val_csv",
        help=(
            "Path to validation CSV (optional). "
            "If omitted together with test_csv, an internal split is used."
        ),
    )
    parser.add_argument(
        "--test_csv",
        help="Path to TEST CSV (optional). Used only for final evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where PySR outputs (equations, metrics, logs) will be saved.",
    )
    parser.add_argument(
        "--target_col",
        default=TARGET_COL_DEFAULT,
        help=f"Name of target column (default={TARGET_COL_DEFAULT}).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help=(
            "Validation fraction when no val_csv/test_csv are provided "
            "(default=0.15)."
        ),
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help=(
            "Test fraction when no val_csv/test_csv are provided "
            "(default=0.15)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default=42).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Initialize logging (file + stdout) inside the target output directory
    setup_logger(output_dir=args.output_dir, log_filename="pysr.log")

    logging.info("=== Starting PySR symbolic regression run ===")

    train_symbolic_regression(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        target_col=args.target_col,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed,
    )

    logging.info("=== PySR symbolic regression run completed ===")
