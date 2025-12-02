import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def select_features(
    df: pd.DataFrame,
    columns: Optional[List[str]],
    drop_target: bool,
    target_col: str,
) -> pd.DataFrame:
    # Select features for K-Means
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
    # Train fixed-k K-Means
    logging.info(f"Training KMeans with k={n_clusters} ...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    kmeans.fit(train_features)
    logging.info("KMeans training completed.")
    return kmeans


def find_best_k(
    train_features: pd.DataFrame,
    k_min: int,
    k_max: int,
    random_state: int = 42,
) -> Tuple[int, KMeans]:
    # Grid search over k using silhouette score
    n_samples = len(train_features)
    if n_samples < 3:
        raise ValueError("Not enough samples for clustering (need at least 3).")

    k_min = max(2, k_min)
    k_max = min(k_max, n_samples - 1)
    if k_min > k_max:
        raise ValueError(f"Invalid k range after clipping: [{k_min}, {k_max}]")

    best_k = None
    best_score = -1.0
    best_model: Optional[KMeans] = None

    logging.info(f"Searching best k in range [{k_min}, {k_max}] using silhouette score")

    for k in range(k_min, k_max + 1):
        logging.info(f"Testing k={k}")
        model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = model.fit_predict(train_features)

        # silhouette_score requires at least 2 clusters and less than n_samples
        score = silhouette_score(train_features, labels)
        logging.info(f"k={k}, silhouette={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_model = model

    logging.info(f"Best k={best_k} with silhouette={best_score:.4f}")
    if best_k is None or best_model is None:
        raise RuntimeError("Failed to find a valid k for clustering.")

    return best_k, best_model


def save_kmeans_model(kmeans: KMeans, save_model_path: Optional[str]) -> None:
    # Save K-Means model
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
    # Save dataset with cluster labels and per-cluster CSVs
    os.makedirs(output_dir, exist_ok=True)

    full_path = os.path.join(output_dir, "all_with_clusters.csv")
    df.to_csv(full_path, index=False)
    logging.info(f"Saved full dataset with cluster labels to: {full_path}")

    cluster_sizes: Dict[str, int] = {}
    for c in range(n_clusters):
        size = int((df["cluster"] == c).sum())
        if size > 0:
            cluster_sizes[str(c)] = size

    if not per_cluster:
        return cluster_sizes

    for c in range(n_clusters):
        cluster_df = df[df["cluster"] == c]
        if cluster_df.empty:
            logging.warning(f"Cluster {c} is empty. Skipping file.")
            continue
        cluster_path = os.path.join(output_dir, f"all_cluster_{c}.csv")
        cluster_df.to_csv(cluster_path, index=False)
        logging.info(f"Cluster {c}: {len(cluster_df)} rows -> {cluster_path}")

    return cluster_sizes
