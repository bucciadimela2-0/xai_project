import argparse


def parse_args():
    """
    Parse command-line arguments for KMeans clustering used before symbolic regression.

    Returns
    -------
    argparse.Namespace
        Parsed arguments for clustering.
    """
    parser = argparse.ArgumentParser(
        description="Run KMeans clustering on a TRAIN CSV for Symbolic Regression."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the TRAIN CSV file.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where clustered outputs will be saved.",
    )

    parser.add_argument(
        "--n_clusters",
        type=int,
        default=5,
        help="Number of KMeans clusters (default=5).",
    )

    parser.add_argument(
        "--columns",
        nargs="+",
        help=(
            "Optional list of feature columns to use for clustering. "
            "If omitted, all numeric columns are used (optionally excluding the target)."
        ),
    )

    parser.add_argument(
        "--drop_target",
        action="store_true",
        help=(
            "When using all numeric columns, remove the target column "
            "before clustering."
        ),
    )

    parser.add_argument(
        "--target_col",
        type=str,
        default="FM_data_peak_distorted_echo_power",
        help="Name of the target column (default: FM_data_peak_distorted_echo_power).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for KMeans (default=42).",
    )

    parser.add_argument(
        "--save_model",
        type=str,
        help="Optional path where the trained KMeans model should be saved (e.g. models/kmeans.pkl).",
    )

    return parser.parse_args()
