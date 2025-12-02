
import logging
import os
from typing import Tuple

import pandas as pd
from const import (DEFAULT_CLUSTER_OUTPUT_DIR, DEFAULT_GEO_OUTPUT_DIR,
                   DEFAULT_N_CLUSTERS, DEFAULT_RANDOM_STATE,
                   DEFAULT_SAMPLING_OUTPUT, DEFAULT_TRAIN_SPLIT,
                   DEFAULT_VAL_SPLIT, DROP_COLS, FLUX_COL, FREQ_COL,
                   FREQ_REQUIRED, TARGET_COL)


def filter_data(
    df: pd.DataFrame,
    freq: float = FREQ_REQUIRED,
    cancel_column: bool = True
) -> pd.DataFrame:
    
    #Filter rows where the frequency column matches a given value.
    
    if FREQ_COL not in df.columns:
        raise ValueError(f"Missing required column: {FREQ_COL}")

    filtered = df[df[FREQ_COL] == freq].copy()
    if filtered.empty:
        raise ValueError(f"No rows found with {FREQ_COL} == {freq}")

    if cancel_column and FREQ_COL in filtered.columns:
        filtered.drop(columns=[FREQ_COL], inplace=True)

    return filtered


def clean_data(
    df: pd.DataFrame,
    drop_cols=None,
    keep_flux: bool = True
) -> pd.DataFrame:

    #Remove unnecessary columns from the dataset.
    if drop_cols is None:
        drop_cols = DROP_COLS

    df = df.drop(columns=drop_cols, errors="ignore")

    if not keep_flux and FLUX_COL in df.columns:
        df = df.drop(columns=[FLUX_COL])

    return df


# SPLITTING 


def split_data(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    shuffle: str = "none",
    random_state: int = DEFAULT_RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  
   # Split a dataset into train, validation, and test subsets.
  
    if shuffle not in ["none", "all", "train_only"]:
        raise ValueError("shuffle must be 'none', 'all', or 'train_only'")

    # Shuffle entire dataset if required
    if shuffle == "all":
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        df = df.copy()

    n = len(df)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # Shuffle only the training portion if requested
    if shuffle == "train_only":
        train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return train_df, val_df, test_df


def save_split(
    df: pd.DataFrame,
    path_out: str,
    train_ratio: float,
    val_ratio: float,
    shuffle: str,
    random_state: int,
    label: str,
) -> None:
   
    #Perform splitting and save train/val/test CSV files.
    
    os.makedirs(path_out, exist_ok=True)

    train, val, test = split_data(df, train_ratio, val_ratio, shuffle, random_state)

    train.to_csv(os.path.join(path_out, "train.csv"), index=False)
    val.to_csv(os.path.join(path_out, "val.csv"), index=False)
    test.to_csv(os.path.join(path_out, "test.csv"), index=False)

    logging.info(f"[OK] Split '{label}' saved to {path_out}")
    logging.info(f"    Train={len(train)} | Val={len(val)} | Test={len(test)}")
