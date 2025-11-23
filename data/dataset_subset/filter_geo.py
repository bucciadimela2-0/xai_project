
import argparse
import logging
import os
import sys
from typing import Dict, Optional

from data.utils.filter_geo_utils import (filter_geo_split,
                                         save_geo_filter_summary,
                                         save_run_config_filter_geo)


def filter_geo_all(
    train_path: Optional[str],
    val_path: Optional[str],
    test_path: Optional[str],
    output_dir: str,
    lat_min: Optional[float],
    lat_max: Optional[float],
    lon_min: Optional[float],
    lon_max: Optional[float],
    lat_col: str,
    lon_col: str,
    command: str,
) -> None:
    
    #Apply the geographic filter to train/val/test

    if lat_min is None and lat_max is None and lon_min is None and lon_max is None:
        raise ValueError(
            "You must specify at least one of lat_min, lat_max, lon_min, lon_max, "
            "or use --south_pole."
        )

    summary: Dict[str, Dict[str, int]] = {}

    if train_path is not None:
        summary["train"] = filter_geo_split(
            train_path,
            "train",
            output_dir,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            lat_col,
            lon_col,
        )

    if val_path is not None:
        summary["val"] = filter_geo_split(
            val_path,
            "val",
            output_dir,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            lat_col,
            lon_col,
        )

    if test_path is not None:
        summary["test"] = filter_geo_split(
            test_path,
            "test",
            output_dir,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            lat_col,
            lon_col,
        )

    # Save JSON summary
    save_geo_filter_summary(output_dir, summary)

    # Append run config
    save_run_config_filter_geo(
        output_dir=output_dir,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_col=lat_col,
        lon_col=lon_col,
        command=command,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Geographical filter for an area (including Mars south pole region)."
    )

    parser.add_argument("--train", help="Path to TRAIN CSV (optional)")
    parser.add_argument("--val", help="Path to VALIDATION CSV (optional)")
    parser.add_argument("--test", help="Path to TEST CSV (optional)")
    parser.add_argument("--output_dir", required=True, help="Output directory")

    parser.add_argument("--lat_min", type=float, help="Minimum latitude (e.g., -90)")
    parser.add_argument("--lat_max", type=float, help="Maximum latitude (e.g., -70)")
    parser.add_argument("--lon_min", type=float, help="Minimum longitude")
    parser.add_argument("--lon_max", type=float, help="Maximum longitude")

    parser.add_argument(
        "--lat_col",
        type=str,
        default="FM_data_latitude",
        help="Name of the latitude column (default=FM_data_latitude)",
    )
    parser.add_argument(
        "--lon_col",
        type=str,
        default="FM_data_longitude",
        help="Name of the longitude column (default=FM_data_longitude)",
    )

    parser.add_argument(
        "--south_pole",
        action="store_true",
        help="Automatically apply a restricted filter for the Martian south pole (lat <= -84).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Rebuild command string for config logging
    command_str = "python " + " ".join(sys.argv)

    # Determine lat/lon bounds
    if args.south_pole:
        lat_min = None
        lat_max = -84.0  
        lon_min = None
        lon_max = None
    else:
        lat_min = args.lat_min
        lat_max = args.lat_max
        lon_min = args.lon_min
        lon_max = args.lon_max

    # Ensure output_dir exists before configuring logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "filter_geo.log")

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

 

    if args.south_pole:
        logging.info("Applying automatic south pole filter (lat <= -84).")

    filter_geo_all(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        output_dir=args.output_dir,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        command=command_str,
    )



