"""
Entry point for running symbolic regression with PySR.
"""

import logging
from typing import Optional, Dict, Any

import numpy as np
from pysr import PySRRegressor
import yaml  # <--- nuovo import

from args.symbolic_args import parse_args
from symbolic_utils import (
    TARGET_COL_DEFAULT,
    evaluate_split,
    load_xy,
    save_pysr_results,
    save_test_predictions,
    setup_logger,
)


def build_pysr_model(random_state: int, config_path: Optional[str]) -> PySRRegressor:
    """Create a PySRRegressor, optionally using a YAML config."""
    default_params: Dict[str, Any] = {
        "niterations": 80,
        "populations": 10,
        "population_size": 1000,
        "elementwise_loss": "L2DistLoss()",
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": ["sin", "cos", "log", "exp"],
        "maxsize": 20,
        "maxdepth": 6,
        "model_selection": "best",
        "progress": True,
        "verbosity": 1,
        "random_state": random_state,
        "deterministic": True,
        "parallelism": "serial",
    }

    if config_path is not None:
        logging.info(f"Loading PySR config from YAML: {config_path}")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        pysr_cfg = cfg.get("pysr", {})
        if pysr_cfg:
            logging.info(f"Overriding default PySR params with YAML keys: {list(pysr_cfg.keys())}")
            default_params.update(pysr_cfg)

    return PySRRegressor(**default_params)


def train_symbolic_regression(
    train_csv: str,
    val_csv: Optional[str],
    test_csv: Optional[str],
    output_dir: str,
    target_col: str = TARGET_COL_DEFAULT,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    config_path: Optional[str] = None,
) -> None:
    """Train a PySR symbolic regressor with internal or external splits."""
    X_all, y_all, feature_names = load_xy(train_csv, target_col=target_col)

    use_internal_test = False
    X_train = X_val = X_test = None
    y_train = y_val = y_test = None

    if val_csv is None and test_csv is None:
        logging.info(
            "No val_csv or test_csv provided. Using internal split "
            f"(val_ratio={val_ratio}, test_ratio={test_ratio})."
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
        X_train, y_train = X_all, y_all

        if val_csv is not None:
            logging.info(f"Using separate validation CSV: {val_csv}")
            X_val, y_val, feature_names_val = load_xy(val_csv, target_col=target_col)

            if len(feature_names_val) != len(feature_names):
                raise ValueError(
                    "Train and validation feature dimensions do not match: "
                    f"train={len(feature_names)}, val={len(feature_names_val)}"
                )
        else:
            raise ValueError(
                "val_csv is None but test_csv is provided. "
                "Provide val_csv as well, or omit both to use internal splitting."
            )

        if test_csv is not None:
            logging.info(f"Using separate TEST CSV: {test_csv}")
            X_test, y_test, feature_names_test = load_xy(test_csv, target_col=target_col)

            if len(feature_names_test) != len(feature_names):
                raise ValueError(
                    "Train and test feature dimensions do not match: "
                    f"train={len(feature_names)}, test={len(feature_names_test)}"
                )

    logging.info(f"Final TRAIN size: {len(y_train)}  |  VAL size: {len(y_val)}")

    # Crea il modello PySR usando defaults + YAML
    model = build_pysr_model(random_state=random_state, config_path=config_path)

    logging.info("Starting PySR training...")
    model.fit(X_train, y_train)
    logging.info("PySR training completed.")

    metrics = {}
    metrics["train"] = evaluate_split(model, X_train, y_train, split_name="TRAIN")
    metrics["val"] = evaluate_split(model, X_val, y_val, split_name="VAL")

    if use_internal_test or test_csv is not None:
        if X_test is None or y_test is None:
            raise RuntimeError(
                "TEST arrays are None but test evaluation was requested."
            )

        metrics["test"] = evaluate_split(model, X_test, y_test, split_name="TEST")

        save_test_predictions(
            model=model,
            X=X_test,
            y=y_test,
            feature_names=feature_names,
            output_dir=output_dir,
            filename="pysr_test_predictions.csv",
        )

    save_pysr_results(model, output_dir=output_dir, metrics=metrics)


if __name__ == "__main__":
    args = parse_args()

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
        config_path=args.config,
    )

    logging.info("=== PySR symbolic regression run completed ===")
