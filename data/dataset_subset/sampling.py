import argparse
import logging
import os
import sys

import pandas as pd

from data.utils.sampling_utils import save_run_config_sampling


def random_sampling(
    input_path: str,
    output_path: str,
    n_samples: int = 50000,
    random_state: int = 42,
    command: str = "",
) -> None:

    ##Performs random sampling from the input CSV and saves the result.
    ##Also logs the configuration into run_config_sampling.json.


    logging.info(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    logging.info(f"Original dataset size: {len(df):,} rows")

    if n_samples <= 0:
        raise ValueError("n_samples must be greater than zero.")
    if n_samples > len(df):
        raise ValueError("Requested number of samples exceeds dataset size.")

    # Perform sampling
    sampled_df = df.sample(n=n_samples, random_state=random_state)
    logging.info(f"Random sampling completed: {n_samples} rows extracted")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save sampled dataset
    sampled_df.to_csv(output_path, index=False)
    logging.info(f"Sampled dataset saved to: {output_path}")

    # Save run configuration
    save_run_config_sampling(
        output_path=output_path,
        input_path=input_path,
        n_samples=n_samples,
        random_state=random_state,
        command=command,
    )
    logging.info("Sampling run configuration recorded.")


def parse_args():
    parser = argparse.ArgumentParser(description="Random Sampling for Symbolic Regression")

    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument("--n_samples", type=int, default=50000,
                        help="Number of samples to extract (default=50k)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Reconstruct full command string for logging
    command_str = "python " + " ".join(sys.argv)

    # Configure logging
    log_dir = os.path.dirname(args.output)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "sampling.log")

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


    random_sampling(
        input_path=args.input,
        output_path=args.output,
        n_samples=args.n_samples,
        random_state=args.seed,
        command=command_str,
    )

  
