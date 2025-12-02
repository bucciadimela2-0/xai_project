import logging
import os
from datetime import datetime
from typing import List, Optional

import pandas as pd
from args.preprocess_args import parse_args
from utils.subprocess_utils import run_clustering, run_sampling
from utils.utils_preprocess import clean_data, filter_data, save_split

# Logging setup
os.makedirs("log", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join("log", f"preprocess_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_path, mode="w")],
)

logging.info(f"Logging started. Saving logs to: {log_path}")


def preprocess(
    input_path: str,
    output_dir: str,
    train_ratio: float,
    val_ratio: float,
    keep_flux: bool,
    shuffle: str,
    random_state: int,
    do_random: bool,
    do_cluster: bool,
    sampling_output: str,
    cluster_output_dir: str,
    n_clusters: int,
    n_samples: int,
    drop_target: bool,
    target_col: str,
    save_model: Optional[str],
    skip_clean: bool,
    columns: Optional[List[str]] = None,
) -> None:
    #End-to-end preprocessing pipeline.

    this_dir = os.path.dirname(__file__)
    logging.debug(f"preprocess.py directory: {this_dir}")

    os.makedirs(output_dir, exist_ok=True)
    clean_path = os.path.join(output_dir, "clean.csv")

    # Load or build clean dataset
    if skip_clean and os.path.exists(clean_path):
        logging.info("Skipping cleaning: using existing clean.csv")
        df = pd.read_csv(clean_path)
        logging.info(f"Clean dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    else:
        logging.info(f"Loading original dataset: {input_path}")
        df = pd.read_csv(input_path, sep=";")
        logging.info(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")

        df = filter_data(df)
        df = clean_data(df, keep_flux=keep_flux)
        logging.info(f"After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")

        df.to_csv(clean_path, index=False)
        logging.info(f"Clean file saved to '{clean_path}'")

    # Full split
    split_full_dir = os.path.join(output_dir, "split_full")
    save_split(
        df=df,
        path_out=split_full_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        shuffle=shuffle,
        random_state=random_state,
        label="split_full",
    )
    logging.info(f"Full split saved in '{split_full_dir}'")

    # Random subset + split
    if do_random:
        logging.info("Running sampling on CLEAN dataset")

        sampling_dir = os.path.dirname(sampling_output)
        if sampling_dir:
            os.makedirs(sampling_dir, exist_ok=True)

        run_sampling(
            input_path=clean_path,
            output_path=sampling_output,
            n_samples=n_samples,
            seed=random_state,
        )
        logging.info(f"Random subset created at: {sampling_output}")

        df_random = pd.read_csv(sampling_output)
        split_random_dir = os.path.join(output_dir, "random10", "split_random")
        os.makedirs(split_random_dir, exist_ok=True)

        save_split(
            df=df_random,
            path_out=split_random_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            random_state=random_state,
            label="split_random",
        )
        logging.info(f"Random split saved in '{split_random_dir}'")

    # Clustering + split per cluster
    if do_cluster:
        logging.info("Running clustering on CLEAN dataset")
        os.makedirs(cluster_output_dir, exist_ok=True)

        run_clustering(
            input_path=clean_path,
            output_dir=cluster_output_dir,
            n_clusters=n_clusters,
            seed=random_state,
            drop_target=drop_target,
            columns=columns,
            target_col=target_col,
            save_model=save_model,
        )

        for k in range(n_clusters):
            cluster_file = os.path.join(cluster_output_dir, f"all_cluster_{k}.csv")
            if not os.path.exists(cluster_file):
                logging.warning(f"Cluster file {k} not found: {cluster_file}, skipping.")
                continue

            df_cluster = pd.read_csv(cluster_file)

            split_cluster_dir = os.path.join(cluster_output_dir, f"split_cluster_{k}")
            os.makedirs(split_cluster_dir, exist_ok=True)

            save_split(
                df=df_cluster,
                path_out=split_cluster_dir,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                shuffle=shuffle,
                random_state=random_state,
                label=f"split_cluster_{k}",
            )
            logging.info(f"Cluster {k} split saved in '{split_cluster_dir}'")


if __name__ == "__main__":
    args = parse_args()

    columns_list: Optional[List[str]] = None
    if args.columns is not None:
        columns_list = [c.strip() for c in args.columns.split(",") if c.strip()]

    preprocess(
        input_path=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_split,
        val_ratio=args.val_split,
        keep_flux=not args.remove_flux,
        shuffle=args.shuffle,
        random_state=args.random_state,
        do_random=args.do_random,
        do_cluster=args.do_cluster,
        sampling_output=args.sampling_output,
        cluster_output_dir=args.cluster_output_dir,
        n_clusters=args.n_clusters,
        n_samples=args.n_samples,
        drop_target=args.drop_target,
        target_col=args.target_col,
        save_model=args.save_model,
        skip_clean=args.skip_clean,
        columns=columns_list,
    )
