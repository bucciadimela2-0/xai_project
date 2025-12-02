

import argparse

from symbolic_utils import TARGET_COL_DEFAULT


def parse_args():
    """Parse CLI arguments for the symbolic regression script."""
    parser = argparse.ArgumentParser(
        description="Run symbolic regression with PySR on a dataset subset."
    )

    parser.add_argument(
        "--train_csv",
        required=True,
        help="Path to training CSV.",
    )
    parser.add_argument(
        "--val_csv",
        help="Path to validation CSV (optional).",
    )
    parser.add_argument(
        "--test_csv",
        help="Path to test CSV (optional).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where PySR outputs will be saved.",
    )
    parser.add_argument(
        "--target_col",
        default=TARGET_COL_DEFAULT,
        help=f"Name of target column (default: {TARGET_COL_DEFAULT}).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation fraction for internal split (default 0.15).",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test fraction for internal split (default 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default 42).",
    )

    return parser.parse_args()
