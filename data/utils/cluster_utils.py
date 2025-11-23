# cluster_utils.py

import json
import logging
import os
from typing import Dict, List, Optional

import joblib
import pandas as pd
from sklearn.cluster import KMeans


def select_features(
    df: pd.DataFrame,
    columns: Optional[List[str]],
    drop_target: bool,
    target_col: str,
) -> pd.DataFrame:
    
    #Selects the features to use for clustering.
    # If `columns` is None: use all numeric columns (optionally dropping target_col).
    # If `columns` is provided: use only those columns.
   

    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Requested columns do not exist in dataset: {missing}")

        features = df[columns].copy()
        logging.info(f"Using specified columns for clustering: {columns}")

    else:
        features = df.copy()

        if drop_target and target_col in features.columns:
            features = features.drop(columns=[target_col])

        # select only numeric columns
        features = features.select_dtypes(include=["number"])
        logging.info(f"Using numeric columns for clustering: {list(features.columns)}")

    if features.empty:
        raise ValueError("No valid columns found for clustering.")

    return features



def train_kmeans(
    train_features: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
) -> KMeans:
    
    #Trains a KMeans model on the given features and returns it.
    
    logging.info(f"Training KMeans with k={n_clusters} ...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    kmeans.fit(train_features)

    logging.info("KMeans training completed.")
    return kmeans


def save_kmeans_model(kmeans: KMeans, save_model_path: Optional[str]) -> None:
    
    #Saves the KMeans model if a path is provided.
   
    if save_model_path is not None:
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        joblib.dump(kmeans, save_model_path)
        logging.info(f"KMeans model saved to: {save_model_path}")



def save_split_with_clusters(
    df: pd.DataFrame,
    split_name: str,
    output_dir: str,
    n_clusters: int,
) -> Dict[str, int]:

    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    full_path = os.path.join(split_dir, f"{split_name}_with_clusters.csv")
    df.to_csv(full_path, index=False)
    logging.info(f"Saved {split_name} with cluster labels to: {full_path}")

    cluster_sizes: Dict[str, int] = {}

    for c in range(n_clusters):
        cluster_df = df[df["cluster"] == c]

        if cluster_df.empty:
            logging.warning(f"{split_name} cluster {c} is empty. Skipping file.")
            continue

        cluster_path = os.path.join(split_dir, f"{split_name}_cluster_{c}.csv")
        cluster_df.to_csv(cluster_path, index=False)

        size = len(cluster_df)
        cluster_sizes[str(c)] = size
        logging.info(f"{split_name} cluster {c}: {size} rows -> {cluster_path}")

    return cluster_sizes



def save_run_config_clustering(
    output_dir: str,
    train_path: str,
    val_path: Optional[str],
    test_path: Optional[str],
    n_clusters: int,
    columns: Optional[List[str]],
    drop_target: bool,
    target_col: str,
    random_state: int,
    save_model_path: Optional[str],
) -> None:
    
    #Writes a text file containing the parameters used for this clustering run.(perchè me ne scordo)
    #Overwrites any previous config file.
    

    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "run_config_clustering.txt")

    with open(config_path, "w") as f:
        f.write("=== KMeans clustering run config ===\n")
        f.write(f"train: {train_path}\n")
        f.write(f"val: {val_path}\n")
        f.write(f"test: {test_path}\n")
        f.write(f"n_clusters: {n_clusters}\n")
        f.write(f"columns: {columns}\n")
        f.write(f"drop_target: {drop_target}\n")
        f.write(f"target_col: {target_col}\n")
        f.write(f"random_state: {random_state}\n")
        f.write(f"save_model_path: {save_model_path}\n")

    logging.info(f"Clustering run configuration saved to: {config_path}")


def save_cluster_summary(
    output_dir: str,
    summary: Dict[str, Dict],
) -> None:
    
    #Saves a JSON summary with cluster sizes for each split.
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "cluster_summary.json")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    logging.info(f"Cluster summary saved to: {summary_path}")
