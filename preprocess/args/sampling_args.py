import argparse


def parse_args():
    # Parse command-line arguments for random dataset sampling.
    parser = argparse.ArgumentParser(
        description="Create a random subset of a dataset for symbolic regression."
    )

    # Input CSV path
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV file.",
    )

    # Output CSV path
    parser.add_argument(
        "--output",
        required=True,
        help="Path where the sampled CSV will be saved.",
    )

    # Number of samples to extract
    parser.add_argument(
        "--n_samples",
        type=int,
        default=50000,
        help="Number of samples to extract (default: 50,000).",
    )

    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    return parser.parse_args()
