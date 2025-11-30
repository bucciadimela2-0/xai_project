import argparse


def parse_args():
    """
    Parse command-line arguments for random sampling of a dataset.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create a random subset of a dataset for symbolic regression."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path where the sampled output CSV will be saved.",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=50000,
        help="Number of samples to extract (default: 50,000).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    return parser.parse_args()
