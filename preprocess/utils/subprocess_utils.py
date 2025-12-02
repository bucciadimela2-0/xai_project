"""
Utility functions for running preprocessing steps as subprocesses.
"""
import logging
import subprocess
from typing import List, Optional, Union


def run_sampling(
    input_path: str,
    output_path: str,
    n_samples: int = 50000,
    seed: int = 42,
) -> None:
    """Run random sampling as a subprocess."""
    logging.info(f"[SUBPROCESS] Sampling {n_samples} rows from {input_path}")

    subprocess.run(
        [
            "python", "-m", "preprocess.dataset_subset.sampling",
            "--input", input_path,
            "--output", output_path,
            "--n_samples", str(n_samples),
            "--seed", str(seed),
        ],
        check=True,
    )


def run_clustering(
    input_path: str,
    output_dir: str,
    n_clusters: int,
    seed: int,
    drop_target: bool = True,
    columns: Optional[Union[List[str], str]] = None,
    target_col: str = "FM_data_peak_distorted_echo_power",
    save_model: Optional[str] = None,
) -> None:
    """Run KMeans clustering as a subprocess."""
    logging.info(f"[SUBPROCESS] Clustering {input_path} into {n_clusters} clusters")

    cmd = [
        "python", "-m", "preprocess.dataset_subset.clusters",
        "--input", input_path,
        "--output_dir", output_dir,
        "--n_clusters", str(n_clusters),
        "--seed", str(seed),
        "--target_col", target_col,
    ]

    if drop_target:
        cmd.append("--drop_target")

    # Handle columns string or list
    if columns:
        if isinstance(columns, list):
            columns = ",".join(columns)
        cmd.extend(["--columns", columns])

    if save_model:
        cmd.extend(["--save_model", save_model])

    subprocess.run(cmd, check=True)
