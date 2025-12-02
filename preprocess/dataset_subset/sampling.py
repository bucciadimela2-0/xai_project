import logging
import os
import sys

import pandas as pd

from preprocess.args.sampling_args import parse_args


def random_sampling(
    input_path: str,
    output_path: str,
    n_samples: int = 50000,
    random_state: int = 42,
    command: str = "",
) -> None:
   
    #Create a random subset of the dataset by sampling a fixed number of rows.
  
    logging.info(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    logging.info(f"Original dataset size: {len(df):,} rows")

    # Validate sampling size
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than zero.")
    if n_samples > len(df):
        raise ValueError("Requested number of samples exceeds dataset size.")

    # Perform random sampling
    sampled_df = df.sample(n=n_samples, random_state=random_state)
    logging.info(f"Random sampling completed: {n_samples} rows extracted")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save sampled dataset to disk
    sampled_df.to_csv(output_path, index=False)
    logging.info(f"Sampled dataset saved to: {output_path}")


if __name__ == "__main__":
    args = parse_args()

    # Reconstruct full command string for logging such as:
    # python preprocess/sampling.py --input ... --output ...
    command_str = "python " + " ".join(sys.argv)

    # Configure logging
    os.makedirs("log/", exist_ok=True)
    log_path = os.path.join("log", "sampling.log")

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Run random sampling pipeline
    random_sampling(
        input_path=args.input,
        output_path=args.output,
        n_samples=args.n_samples,
        random_state=args.seed,
        command=command_str,
    )
