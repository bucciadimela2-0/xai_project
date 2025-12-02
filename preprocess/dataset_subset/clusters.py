import logging
import os
from typing import List, Optional

import pandas as pd

from preprocess.args.cluster_args import parse_args
from preprocess.utils.cluster_utils import (find_best_k, save_kmeans_model,
                                            save_split_with_clusters,
                                            select_features, train_kmeans)


def cluster_kmeans(
    train_path: str,
    output_dir: str,
    n_clusters: int = 5,
    columns: Optional[List[str]] = None,
    drop_target: bool = True,
    target_col: str = "FM_data_peak_distorted_echo_power",
    random_state: int = 42,
    save_model_path: Optional[str] = None,
    auto_k: bool = False,
    k_min: int = 2,
    k_max: int = 10,
) -> None:
    # Run K-Means clustering and save results
    logging.info(f"Loading dataset from: {train_path}")
    train_df = pd.read_csv(train_path)
    logging.info(f"dataset shape: {len(train_df)} rows, {train_df.shape[1]} columns")

    train_features = select_features(train_df, columns, drop_target, target_col)

    if auto_k:
        best_k, kmeans = find_best_k(
            train_features=train_features,
            k_min=k_min,
            k_max=k_max,
            random_state=random_state,
        )
        logging.info(f"Using best_k={best_k} for final clustering")
        used_k = best_k
    else:
        kmeans = train_kmeans(
            train_features,
            n_clusters=n_clusters,
            random_state=random_state,
        )
        used_k = n_clusters

    train_df["cluster"] = kmeans.labels_
    logging.info("Clustering completed.")

    save_kmeans_model(kmeans, save_model_path)

    save_split_with_clusters(
        df=train_df,
        output_dir=output_dir,
        n_clusters=used_k,
        per_cluster=True,
    )


if __name__ == "__main__":
    args = parse_args()

    os.makedirs("log/", exist_ok=True)
    log_path = "log/cluster_kmeans.log"

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
        auto_k=args.auto_k,
        k_min=args.k_min,
        k_max=args.k_max,
    )
