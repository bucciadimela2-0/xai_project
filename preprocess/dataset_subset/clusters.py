import logging
import os
from typing import List, Optional

import pandas as pd

from preprocess.args.cluster_args import parse_args
from preprocess.utils.cluster_utils import (
    save_kmeans_model,
    save_split_with_clusters,
    select_features,
    train_kmeans,
)


def cluster_kmeans(
    train_path: str,
    output_dir: str,
    n_clusters: int = 5,
    columns: Optional[List[str]] = None,
    drop_target: bool = True,
    target_col: str = "FM_data_peak_distorted_echo_power",
    random_state: int = 42,
    save_model_path: Optional[str] = None,
) -> None:
    """
    Perform K-Means clustering on a dataset and save the resulting clusters.

    Parameters
    ----------
    train_path : str
        Path to the input CSV used for training the clustering model.
    output_dir : str
        Directory where clustered splits will be saved.
    n_clusters : int, default=5
        Number of clusters for K-Means.
    columns : list of str, optional
        List of feature columns to use. If None, all columns except the target are used.
    drop_target : bool, default=True
        Whether to remove the target column before clustering.
    target_col : str, default="FM_data_peak_distorted_echo_power"
        Name of the target column to drop if `drop_target=True`.
    random_state : int, default=42
        Random seed for reproducibility.
    save_model_path : str, optional
        If provided, the trained K-Means model is saved to this path.
    """

    logging.info(f"Loading TRAIN from: {train_path}")
    train_df = pd.read_csv(train_path)
    logging.info(f"TRAIN shape: {len(train_df)} rows, {train_df.shape[1]} columns")

    # Select features used to train the K-Means model
    # If a column subset is provided, only that subset is used.
    # If drop_target=True, the target column is removed.
    train_features = select_features(train_df, columns, drop_target, target_col)

    # Train K-Means
    kmeans = train_kmeans(
        train_features,
        n_clusters=n_clusters,
        random_state=random_state,
    )

    # Assign cluster labels to the original dataset
    train_df["cluster"] = kmeans.labels_
    logging.info("Clustering completed.")

    # Save trained model, if a path was specified
    save_kmeans_model(kmeans, save_model_path)

    # Split the dataset by clusters and save each partition
    save_split_with_clusters(
        df=train_df,
        output_dir=output_dir,
        n_clusters=n_clusters,
        per_cluster=True,
    )


if __name__ == "__main__":
    args = parse_args()

    # Create logging directory if needed
    os.makedirs("log/", exist_ok=True)

    log_path = "log/cluster_kmeans.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(),
        ],
    )

    cluster_kmeans(
        train_path=args.input,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        columns=args.columns,
        drop_target=args.drop_target,
        target_col=args.target_col,
        random_state=args.seed,
        save_model_path=args.save_model,
    )
