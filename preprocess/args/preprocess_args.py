import argparse

from const import (DEFAULT_CLUSTER_OUTPUT_DIR, DEFAULT_GEO_OUTPUT_DIR,
                   DEFAULT_N_CLUSTERS, DEFAULT_RANDOM_STATE,
                   DEFAULT_SAMPLING_OUTPUT, DEFAULT_TRAIN_SPLIT,
                   DEFAULT_VAL_SPLIT)


def parse_args():
    """Parse all CLI arguments for the preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess + optional subsets (random / geo / cluster)."
    )

    #skip dataset cleaning if already done
    parser.add_argument(
    "--skip_clean",
    action="store_true",
    help="Skip dataset cleaning if clean.csv already exists.",
)

    # I/O
    parser.add_argument("--input", required=True, help="Path to raw CSV.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")

    # Splits
    parser.add_argument("--train_split", type=float, default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--val_split", type=float, default=DEFAULT_VAL_SPLIT)

    # Shuffle
    parser.add_argument("--shuffle", choices=["none", "all", "train_only"], default="none")
    parser.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE)

    # Cleaning
    parser.add_argument("--remove_flux", action="store_true")

    # Subset flags
    parser.add_argument("--do_random", action="store_true")
    parser.add_argument("--do_cluster", action="store_true")

    # Subprocess paths
    parser.add_argument("--sampling_output", type=str, default=DEFAULT_SAMPLING_OUTPUT)
    parser.add_argument("--cluster_output_dir", type=str, default=DEFAULT_CLUSTER_OUTPUT_DIR)

    # Random sampling args
    parser.add_argument("--n_samples", type=int, default=50000)

    # Clustering args
    parser.add_argument("--n_clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument(
        "--columns",
        type=str,
        default=None,
        help="Comma-separated column list for clustering."
    )
    parser.add_argument("--drop_target", action="store_true")
    parser.add_argument("--target_col", type=str, default="FM_data_peak_distorted_echo_power")
    parser.add_argument("--save_model", type=str)

    return parser.parse_args()
