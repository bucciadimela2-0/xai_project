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
    """
    Select the features to be used for K-Means clustering.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : list of str, optional
        If provided, only these columns will be used as features.
    drop_target : bool
        Whether to remove the target column (if present).
    target_col : str
        Name of the target column to drop if `drop_target=True`.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the selected numeric features.

    Raises
    ------
    ValueError
        If requested columns are missing, or if no valid numeric features remain.
    """
    # Use manually selected columns
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Requested columns do not exist in dataset: {missing}")

        features = df[columns].copy()
        logging.info(f"Using specified columns for clustering: {columns}")

    else:
        # Use full dataset
        features = df.copy()

        # Optionally drop target
        if drop_target and target_col in features.columns:
            features = features.drop(columns=[target_col])

        # Keep numeric-only columns for clustering
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
    """
    Train a K-Means clustering model.

    Parameters
    ----------
    train_features : pd.DataFrame
        Features used for clustering.
    n_clusters : int
        Number of clusters.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    KMeans
        Trained KMeans model.
    """
    logging.info(f"Training KMeans with k={n_clusters} ...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    kmeans.fit(train_features)

    logging.info("KMeans training completed.")
    return kmeans


def save_kmeans_model(kmeans: KMeans, save_model_path: Optional[str]) -> None:
    """
    Save a trained K-Means model to disk using joblib.

    Parameters
    ----------
    kmeans : KMeans
        Trained model.
    save_model_path : str or None
        Path where the model should be saved. If None, saving is skipped.
    """
    if save_model_path is not None:
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        joblib.dump(kmeans, save_model_path)
        logging.info(f"KMeans model saved to: {save_model_path}")


def save_split_with_clusters(
    df: pd.DataFrame,
    output_dir: str,
    n_clusters: int,
    per_cluster: bool = True,
) -> Dict[str, int]:
    """
    Save the clustered dataset and optionally separate files for each cluster.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the 'cluster' column produced by K-Means.
    output_dir : str
        Directory where results will be saved.
    n_clusters : int
        Number of clusters expected.
    per_cluster : bool, default=True
        Whether to save one CSV per cluster.

    Returns
    -------
    dict
        Dictionary mapping cluster index -> number of samples.

    Notes
    -----
    Saves:
        - all_with_clusters.csv → full dataset with cluster labels
        - all_cluster_<k>.csv   → per-cluster CSVs (if enabled)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1) Save full dataset with cluster labels
    full_path = os.path.join(output_dir, "all_with_clusters.csv")
    df.to_csv(full_path, index=False)
    logging.info(f"Saved full dataset with cluster labels to: {full_path}")

    cluster_sizes: Dict[str, int] = {}

    # 2) Count samples in each cluster
    for c in range(n_clusters):
        size = int((df["cluster"] == c).sum())
        if size > 0:
            cluster_sizes[str(c)] = size

    if not per_cluster:
        return cluster_sizes

    # 3) Save per-cluster CSVs
    for c in range(n_clusters):
        cluster_df = df[df["cluster"] == c]

        if cluster_df.empty:
            logging.warning(f"Cluster {c} is empty. Skipping file.")
            continue

        cluster_path = os.path.join(output_dir, f"all_cluster_{c}.csv")
        cluster_df.to_csv(cluster_path, index=False)

        logging.info(f"Cluster {c}: {len(cluster_df)} rows -> {cluster_path}")

    return cluster_sizes
