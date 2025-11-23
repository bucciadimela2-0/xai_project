import json
import logging
import os
from typing import Dict, List, Optional

import pandas as pd
from sklearn.cluster import KMeans


def build_geo_mask(
    df: pd.DataFrame,
    lat_min: Optional[float],
    lat_max: Optional[float],
    lon_min: Optional[float],
    lon_max: Optional[float],
    lat_col: str,
    lon_col: str,
) -> pd.Series:
   
    #Creates a boolean mask for filtering rows by latitude/longitude ranges.
   
    mask = pd.Series([True] * len(df), index=df.index)

    if lat_min is not None:
        mask &= df[lat_col] >= lat_min
    if lat_max is not None:
        mask &= df[lat_col] <= lat_max
    if lon_min is not None:
        mask &= df[lon_col] >= lon_min
    if lon_max is not None:
        mask &= df[lon_col] <= lon_max

    return mask


def save_run_config_filter_geo(
    output_dir: str,
    train_path: Optional[str],
    val_path: Optional[str],
    test_path: Optional[str],
    lat_min: Optional[float],
    lat_max: Optional[float],
    lon_min: Optional[float],
    lon_max: Optional[float],
    lat_col: str,
    lon_col: str,
    command: str,
) -> None:
   
    #Appends a new geographic filter run configuration to run_config_filter_geo.json.
  

    config_entry = {
        "command": command,
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_col": lat_col,
        "lon_col": lon_col,
    }

    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "run_config_filter_geo.json")

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = [existing]
        except Exception:
            existing = []
    else:
        existing = []

    existing.append(config_entry)

    with open(config_path, "w") as f:
        json.dump(existing, f, indent=4)

    logging.info(f"Geo-filter configuration appended to: {config_path}")


def filter_geo_split(
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

    #Loads a train/val/test split, applies the geo-filter, and saves the filtered CSV.
    #Returns: { original_rows, filtered_rows }
   

    logging.info(f"Loading {split_name.upper()} from: {path}")
    df = pd.read_csv(path)

    n_orig = len(df)
    logging.info(f"{split_name.upper()} original rows: {n_orig}")

    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Columns not found: {lat_col}, {lon_col}")

    mask = build_geo_mask(df, lat_min, lat_max, lon_min, lon_max, lat_col, lon_col)
    df_filtered = df[mask].copy()

    n_filt = len(df_filtered)
    logging.info(f"{split_name.upper()} after filtering: {n_filt} rows")

    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    out_path = os.path.join(split_dir, f"{split_name}_geo_filtered.csv")
    df_filtered.to_csv(out_path, index=False)

    logging.info(f"Saved filtered {split_name} to: {out_path}")

    return {"original_rows": n_orig, "filtered_rows": n_filt}


def save_geo_filter_summary(output_dir: str, summary: Dict[str, Dict[str, int]]) -> None:
    
    #Saves a JSON summary of the geographic filtering results.
   

    summary_path = os.path.join(output_dir, "geo_filter_summary.json")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    logging.info(f"Geo-filter summary saved to: {summary_path}")
