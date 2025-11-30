import logging
import os
from datetime import datetime

import pandas as pd
from args.preprocess_args import parse_args
from utils.subprocess_utils import run_clustering, run_filter_geo, run_sampling
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
    do_geo: bool,
    do_cluster: bool,
    sampling_output: str,
    geo_output_dir: str,
    cluster_output_dir: str,
    n_clusters: int,
):
    """
    Complete preprocessing pipeline for the FM dataset.

    Steps
    -----
    1. Load and clean the dataset:
        - Keep only 4 MHz rows
        - Remove unnecessary columns
        - Save clean.csv
    2. Split the clean dataset into train/val/test
    3. Optional pipelines:
        - Random subset sampling + split
        - Geographic subset (south pole) + split
        - Clustering + split per cluster

    Parameters
    ----------
    input_path : str
        Path to the raw CSV file.
    output_dir : str
        Directory where all preprocessing outputs will be stored.
    train_ratio : float
        Train split proportion.
    val_ratio : float
        Validation split proportion.
    keep_flux : bool
        Whether to keep the FLUX column.
    shuffle : str
        Shuffle strategy passed to split_data().
    random_state : int
        Random seed.
    do_random : bool
        Whether to run random subset sampling.
    do_geo : bool
        Whether to run geographic filtering.
    do_cluster : bool
        Whether to run clustering.
    sampling_output : str
        Output path for the random sampling CSV.
    geo_output_dir : str
        Output directory for the geographic subset.
    cluster_output_dir : str
        Output directory for clustering outputs.
    n_clusters : int
        Number of clusters for KMeans.
    """

    this_dir = os.path.dirname(__file__)
    logging.debug(f"preprocess.py directory: {this_dir}")

    logging.info(f"Loading original dataset: {input_path}")
    df = pd.read_csv(input_path, sep=";")
    logging.info(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    df = filter_data(df)
    df = clean_data(df, keep_flux=keep_flux)
    logging.info(f"After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")

    os.makedirs(output_dir, exist_ok=True)
    clean_path = os.path.join(output_dir, "clean.csv")
    df.to_csv(clean_path, index=False)
    logging.info(f"Clean file saved to '{clean_path}'")

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

    if do_random:
        logging.info("Running sampling on CLEAN dataset")
        os.makedirs(os.path.dirname(sampling_output), exist_ok=True)

        run_sampling(
            input_path=clean_path,
            output_path=sampling_output,
            n_samples=10000,
            seed=random_state,
        )
        logging.info(f"Random subset created at: {sampling_output}")

        df_random = pd.read_csv(sampling_output)
        split_random_dir = os.path.join(output_dir, "random10/split_random")
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

    if do_geo:
        logging.info("Running geographic filter on CLEAN dataset (south pole)")
        os.makedirs(geo_output_dir, exist_ok=True)

        run_filter_geo(
            input_path=clean_path,
            output_dir=geo_output_dir,
            south_pole=True,
        )

        geo_train_path = os.path.join(geo_output_dir, "train_geo_filtered.csv")
        if not os.path.exists(geo_train_path):
            msg = (
                f"Filtered train file not found: {geo_train_path}. "
                "Check preprocess/utils/filter_geo_utils.py for output filename."
            )
            logging.error(msg)
            raise FileNotFoundError(msg)

        df_geo = pd.read_csv(geo_train_path)

        split_geo_dir = os.path.join(geo_output_dir, "split_geo")
        save_split(
            df=df_geo,
            path_out=split_geo_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            random_state=random_state,
            label="split_geo",
        )
        logging.info(f"Geo split saved in '{split_geo_dir}'")

    if do_cluster:
        logging.info("Running clustering on CLEAN dataset")
        os.makedirs(cluster_output_dir, exist_ok=True)

        run_clustering(
            input_path=clean_path,
            output_dir=cluster_output_dir,
            n_clusters=n_clusters,
            seed=random_state,
            drop_target=True,
        )

        for k in range(n_clusters):
            cluster_file = os.path.join(cluster_output_dir, f"all_cluster_{k}.csv")
            if not os.path.exists(cluster_file):
                logging.warning(f"Cluster file {k} not found: {cluster_file}, skipping.")
                continue

            df_cluster = pd.read_csv(cluster_file)

            split_cluster_dir = os.path.join(cluster_output_dir, f"split_cluster_{k}")
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

    preprocess(
        input_path=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_split,
        val_ratio=args.val_split,
        keep_flux=not args.remove_flux,
        shuffle=args.shuffle,
        random_state=args.random_state,
        do_random=args.do_random,
        do_geo=args.do_geo,
        do_cluster=args.do_cluster,
        sampling_output=args.sampling_output,
        geo_output_dir=args.geo_output_dir,
        cluster_output_dir=args.cluster_output_dir,
        n_clusters=args.n_clusters,
    )
