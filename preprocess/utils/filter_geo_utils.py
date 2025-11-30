"""
Utility functions for geographic filtering of datasets.

These functions allow filtering a DataFrame based on latitude and longitude
bounds and saving the resulting subset to disk. They are used by the
preprocessing pipeline to extract specific geographic regions of Mars.
"""
import logging
import os
from typing import Dict, Optional

import pandas as pd


def build_geo_mask(
    df: pd.DataFrame,
    lat_min: Optional[float],
    lat_max: Optional[float],
    lon_min: Optional[float],
    lon_max: Optional[float],
    lat_col: str,
    lon_col: str,
) -> pd.Series:
    """
    Create a boolean mask to filter rows by latitude and longitude ranges.
    If a bound is None, that constraint is not applied.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to filter.
    lat_min : float or None
        Minimum latitude (inclusive).
    lat_max : float or None
        Maximum latitude (inclusive).
    lon_min : float or None
        Minimum longitude (inclusive).
    lon_max : float or None
        Maximum longitude (inclusive).
    lat_col : str
        Name of the latitude column.
    lon_col : str
        Name of the longitude column.

    Returns
    -------
    pd.Series
        Boolean mask indicating which rows fall within the specified bounds.
    """
    # Start with all rows enabled, then apply constraints
    mask = pd.Series(True, index=df.index)

    if lat_min is not None:
        mask &= df[lat_col] >= lat_min
    if lat_max is not None:
        mask &= df[lat_col] <= lat_max
    if lon_min is not None:
        mask &= df[lon_col] >= lon_min
    if lon_max is not None:
        mask &= df[lon_col] <= lon_max

    return mask


def filter_geo(
    path: str,
    split_name: str,
    output_dir: str,
    lat_min: Optional[float],
    lat_max: Optional[float],
    lon_min: Optional[float],
    lon_max: Optional[float],
    lat_col: str,
    lon_col: str,
) -> Dict[str, int]:
    """
    Load a dataset, apply geographic filtering, and save the filtered split.

    Parameters
    ----------
    path : str
        Path to the input CSV file.
    split_name : str
        Name of the split (e.g. "train", "val", "test") used in log messages and filenames.
    output_dir : str
        Directory where the filtered file will be saved.
    lat_min : float or None
        Minimum latitude bound.
    lat_max : float or None
        Maximum latitude bound.
    lon_min : float or None
        Minimum longitude bound.
    lon_max : float or None
        Maximum longitude bound.
    lat_col : str
        Name of the latitude column in the dataframe.
    lon_col : str
        Name of the longitude column in the dataframe.

    Returns
    -------
    dict
        Dictionary with counts of original and filtered rows.

    Raises
    ------
    ValueError
        If the latitude or longitude columns do not exist.
    """
    logging.info(f"Loading {split_name.upper()} from: {path}")
    df = pd.read_csv(path)

    n_orig = len(df)
    logging.info(f"{split_name.upper()} original rows: {n_orig}")

    # Validate that required columns are present
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Columns not found: {lat_col}, {lon_col}")

    # Build the mask and filter rows
    mask = build_geo_mask(df, lat_min, lat_max, lon_min, lon_max, lat_col, lon_col)
    df_filtered = df[mask].copy()

    n_filt = len(df_filtered)
    logging.info(f"{split_name.upper()} after filtering: {n_filt} rows")

    # Save directly inside output_dir
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, f"{split_name}_geo_filtered.csv")
    df_filtered.to_csv(out_path, index=False)

    logging.info(f"Saved filtered dataset to: {out_path}")

    return {"original_rows": n_orig, "filtered_rows": n_filt}
