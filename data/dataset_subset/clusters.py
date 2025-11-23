
import argparse
import logging
import os
from typing import Dict, List, Optional

import pandas as pd

from data.utils.cluster_utils import (save_cluster_summary, save_kmeans_model,
                                      save_run_config_clustering,
                                      save_split_with_clusters,
                                      select_features, train_kmeans)


def cluster_kmeans(
    train_path: str,
    val_path: Optional[str],
    test_path: Optional[str],
    output_dir: str,
    n_clusters: int = 5,
    columns: Optional[List[str]] = None,
    drop_target: bool = True,
    target_col: str = "FM_data_peak_distorted_echo_power",
    random_state: int = 42,
    save_model_path: Optional[str] = None,
) -> None:
  

    # Save clustering configuration
    save_run_config_clustering(
        output_dir=output_dir,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        n_clusters=n_clusters,
        columns=columns,
        drop_target=drop_target,
        target_col=target_col,
        random_state=random_state,
        save_model_path=save_model_path,
    )

    cluster_summary: Dict[str, Dict] = {}

    # -------- TRAIN --------
    logging.info(f"Loading TRAIN from: {train_path}")
    train_df = pd.read_csv(train_path) #read train csv
    logging.info(f"TRAIN shape: {len(train_df)} rows, {train_df.shape[1]} columns")

    # Select columns to use as features (all numeric if columns is None)
    train_features = select_features(train_df, columns, drop_target, target_col)

    # Train KMeans
    kmeans = train_kmeans(train_features, n_clusters=n_clusters, random_state=random_state)

    # Add cluster labels to train dataframe
    train_df["cluster"] = kmeans.labels_
    logging.info("Clustering on TRAIN completed.")

    # Save trained model (optional)
    save_kmeans_model(kmeans, save_model_path)

    # Save train with clusters + per-cluster CSVs
    train_cluster_sizes = save_split_with_clusters(train_df, "train", output_dir, n_clusters)
    cluster_summary["train"] = {
        "total_rows": len(train_df),
        "clusters": train_cluster_sizes,
    }

    
    # -------- VAL & TEST --------
    # Apply KMeans model to val and test splits
    splits = {"val": val_path, "test": test_path}

    for split_name, path in splits.items():
        if path is None:
            continue

        logging.info(f"Loading {split_name.upper()} from: {path}")
        df = pd.read_csv(path)
        logging.info(f"{split_name.upper()} shape: {len(df)} rows, {df.shape[1]} columns")

        features_df = select_features(df, columns, drop_target, target_col)

        # Ensure the same features as in TRAIN are present
        missing_in_split = [c for c in train_features.columns if c not in features_df.columns]
        if missing_in_split:
            raise ValueError(
                f"The following columns used in TRAIN are missing in {split_name}: {missing_in_split}"
            )

        # Reorder columns to match TRAIN
        features_df = features_df[train_features.columns]

        # Predict cluster labels for this split
        split_clusters = kmeans.predict(features_df)
        df["cluster"] = split_clusters
        logging.info(f"Clustering on {split_name.upper()} completed.")

        split_cluster_sizes = save_split_with_clusters(df, split_name, output_dir, n_clusters)
        cluster_summary[split_name] = {
            "total_rows": len(df),
            "clusters": split_cluster_sizes,
        }

    # Save global cluster summary
    save_cluster_summary(output_dir, cluster_summary)


def parse_args():
    parser = argparse.ArgumentParser(
        description="KMeans clustering on TRAIN + application to VAL/TEST for Symbolic Regression"
    )

    parser.add_argument("--train", required=True, help="Path to TRAIN CSV")
    parser.add_argument("--val", help="Path to VALIDATION CSV (optional)")
    parser.add_argument("--test", help="Path to TEST CSV (optional)")
    parser.add_argument("--output_dir", required=True, help="Output directory")

    parser.add_argument("--n_clusters", type=int, default=5, help="Number of KMeans clusters (default=5)")

    parser.add_argument(
        "--columns",
        nargs="+",
        help=(
            "List of columns to use for clustering (optional). "
            "If not provided, uses all numeric columns (optionally without target)."
        ),
    )

    parser.add_argument(
        "--drop_target",
        action="store_true",
        help="If using all numeric columns, remove the target column from clustering.",
    )

    parser.add_argument(
        "--target_col",
        type=str,
        default="FM_data_peak_distorted_echo_power",
        help="Name of the target column (default=FM_data_peak_distorted_echo_power)",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for KMeans (default=42)")

    parser.add_argument(
        "--save_model",
        type=str,
        help="Path to save the KMeans model (e.g., models/kmeans.pkl). Optional.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Ensure output directory exists before creating the log file
    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "cluster_kmeans.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(),
        ],
    )

    

    cluster_kmeans(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        columns=args.columns,
        drop_target=args.drop_target,
        target_col=args.target_col,
        random_state=args.seed,
        save_model_path=args.save_model,
    )

    
