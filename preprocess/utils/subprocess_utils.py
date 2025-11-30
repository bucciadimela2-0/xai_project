"""
Utility functions for running preprocessing steps as subprocesses.

Each function here wraps a CLI-based preprocessing script and executes it
using `subprocess.run`. This allows building higher-level preprocessing
pipelines without directly importing and calling the underlying modules.
"""
import logging
import subprocess
from typing import Optional


def run_sampling(
    input_path: str,
    output_path: str,
    n_samples: int = 50000,
    seed: int = 42,
) -> None:
    """
    Launch the random sampling step as a subprocess.

    Parameters
    ----------
    input_path : str
        Path to the input CSV.
    output_path : str
        Path where the sampled output CSV will be saved.
    n_samples : int, default=50000
        Number of samples to extract.
    seed : int, default=42
        Random seed for reproducibility.
    """
    logging.info(f"[SUBPROCESS] Running sampling: extracting {n_samples} rows from {input_path}")

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


def run_filter_geo(
    input_path: str,
    output_dir: str,
    south_pole: bool = True,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> None:
    """
    Launch the geographic filtering step as a subprocess.

    Parameters
    ----------
    input_path : str
        Path to the input CSV.
    output_dir : str
        Directory where the filtered dataset will be saved.
    south_pole : bool, default=True
        If True, use the predefined south pole geographic region.
        If False, use custom coordinate bounds (lat_min/max, lon_min/max).
    lat_min : float or None
        Minimum latitude bound (ignored if south_pole=True).
    lat_max : float or None
        Maximum latitude bound.
    lon_min : float or None
        Minimum longitude bound.
    lon_max : float or None
        Maximum longitude bound.
    lat_col : str, default="lat"
        Name of the latitude column.
    lon_col : str, default="lon"
        Name of the longitude column.
    """
    logging.info(f"[SUBPROCESS] Running geographic filtering on {input_path}")

    cmd = [
        "python", "-m", "preprocess.dataset_subset.filter_geo",
        "--input", input_path,
        "--output_dir", output_dir,
        "--lat_col", lat_col,
        "--lon_col", lon_col,
    ]

    if south_pole:
        cmd.append("--south_pole")
    else:
        if lat_min is not None:
            cmd += ["--lat_min", str(lat_min)]
        if lat_max is not None:
            cmd += ["--lat_max", str(lat_max)]
        if lon_min is not None:
            cmd += ["--lon_min", str(lon_min)]
        if lon_max is not None:
            cmd += ["--lon_max", str(lon_max)]

    subprocess.run(cmd, check=True)


def run_clustering(
    input_path: str,
    output_dir: str,
    n_clusters: int,
    seed: int,
    drop_target: bool = True,
) -> None:
    """
    Launch the dataset clustering step as a subprocess.

    Parameters
    ----------
    input_path : str
        Path to the input CSV.
    output_dir : str
        Directory where the cluster-based splits will be saved.
    n_clusters : int
        Number of clusters to generate.
    seed : int
        Random seed for reproducibility.
    drop_target : bool, default=True
        Whether to drop the target column before training KMeans.
    """
    logging.info(f"[SUBPROCESS] Running clustering: {n_clusters} clusters from {input_path}")

    cmd = [
        "python", "-m", "preprocess.dataset_subset.clusters",
        "--input", input_path,
        "--output_dir", output_dir,
        "--n_clusters", str(n_clusters),
        "--seed", str(seed),
    ]

    if drop_target:
        cmd.append("--drop_target")

    subprocess.run(cmd, check=True)
