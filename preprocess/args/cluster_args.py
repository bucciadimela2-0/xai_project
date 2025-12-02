import argparse


def parse_args():
    # Parse command-line arguments for KMeans clustering before symbolic regression.
    parser = argparse.ArgumentParser(
        description="Run KMeans clustering on a TRAIN CSV for Symbolic Regression."
    )

    # Path to input training CSV
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the TRAIN CSV file.",
    )

    # Directory where clustered outputs will be stored
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where clustered outputs will be saved.",
    )

    # Number of clusters
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=5,
        help="Number of KMeans clusters (default=5).",
    )

    # Optional list of feature columns
    parser.add_argument(
        "--columns",
        nargs="+",
        help="Optional list of feature columns to use. If omitted, all numeric columns are used.",
    )

    # Drop the target column when using all numeric columns
    parser.add_argument(
        "--drop_target",
        action="store_true",
        help="If set, remove the target column before clustering.",
    )

    # Name of the target column
    parser.add_argument(
        "--target_col",
        type=str,
        default="FM_data_peak_distorted_echo_power",
        help="Target column name (default: FM_data_peak_distorted_echo_power).",
    )

    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for KMeans (default=42).",
    )

    # Optional output path to save the KMeans model
    parser.add_argument(
        "--save_model",
        type=str,
        help="Optional path to save the trained KMeans model.",
    )

     # >>> new: optional auto-k logic
    parser.add_argument(
        "--auto_k",
        action="store_true",
        help="If set, search best k in [k_min, k_max] using silhouette score.",
    )
    parser.add_argument(
        "--k_min",
        type=int,
        default=2,
        help="Minimum k for auto-k search (default: 2).",
    )
    parser.add_argument(
        "--k_max",
        type=int,
        default=20,
        help="Maximum k for auto-k search (default: 10).",
    )


    return parser.parse_args()
