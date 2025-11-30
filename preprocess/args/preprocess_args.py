import argparse

from const import (DEFAULT_CLUSTER_OUTPUT_DIR, DEFAULT_GEO_OUTPUT_DIR,
                   DEFAULT_N_CLUSTERS, DEFAULT_RANDOM_STATE,
                   DEFAULT_SAMPLING_OUTPUT, DEFAULT_TRAIN_SPLIT,
                   DEFAULT_VAL_SPLIT)


def parse_args():
    """
    Parse command-line arguments for preprocess.py.
    This file keeps preprocess.py clean and focused on pipeline logic.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Main preprocessing pipeline: clean + optional subsets (random / geo / cluster)."
    )

    # Input/Output
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the raw CSV file (e.g., data/all_past.csv).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where all cleaned and processed files will be saved "
             "(e.g., data/data_cleaned).",
    )

    # Splits
    parser.add_argument(
        "--train_split",
        type=float,
        default=DEFAULT_TRAIN_SPLIT,
        help=f"Training set proportion (default: {DEFAULT_TRAIN_SPLIT}).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=DEFAULT_VAL_SPLIT,
        help=f"Validation set proportion (default: {DEFAULT_VAL_SPLIT}). "
             "The remaining fraction is used for TEST.",
    )

    # Shuffle options
    parser.add_argument(
        "--shuffle",
        choices=["none", "all", "train_only"],
        default="none",
        help="Shuffling mode: none | all | train_only.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for shuffling, sampling, and clustering "
             f"(default: {DEFAULT_RANDOM_STATE}).",
    )

    # Cleaning options
    parser.add_argument(
        "--remove_flux",
        action="store_true",
        help="If set, drop the FM_data_F10_7_index (solar flux) column during cleaning.",
    )

    # Optional subsets
    parser.add_argument(
        "--do_random",
        action="store_true",
        help="Generate a RANDOM subset from the cleaned dataset.",
    )
    parser.add_argument(
        "--do_geo",
        action="store_true",
        help="Generate a GEO subset (south pole region) from the cleaned dataset.",
    )
    parser.add_argument(
        "--do_cluster",
        action="store_true",
        help="Generate CLUSTER subsets (one per cluster) from the cleaned dataset.",
    )

    # External subprocess file paths
    parser.add_argument(
        "--sampling_output",
        type=str,
        default=DEFAULT_SAMPLING_OUTPUT,
        help=(
            "Output file for sampling.py (default: "
            f"{DEFAULT_SAMPLING_OUTPUT})."
        ),
    )
    parser.add_argument(
        "--geo_output_dir",
        type=str,
        default=DEFAULT_GEO_OUTPUT_DIR,
        help=(
            "Output directory for filter_geo.py (default: "
            f"{DEFAULT_GEO_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--cluster_output_dir",
        type=str,
        default=DEFAULT_CLUSTER_OUTPUT_DIR,
        help=(
            "Output directory for cluster.py (default: "
            f"{DEFAULT_CLUSTER_OUTPUT_DIR})."
        ),
    )

    # KMeans cluster count
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=DEFAULT_N_CLUSTERS,
        help=f"Number of clusters for cluster.py (default: {DEFAULT_N_CLUSTERS}).",
    )

    return parser.parse_args()
