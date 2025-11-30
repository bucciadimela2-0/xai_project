"""
Create a subset of the dataset by selecting a specific Martian geographic area.

This script is intended to filter a CSV file by latitude and longitude, keeping
only rows that fall within a given bounding box on Mars.
"""
import logging
import os
from typing import Optional

from preprocess.args.filter_geo_args import parse_args
from preprocess.utils.filter_geo_utils import filter_geo


def filter_geo_dataset(
    input_path: str,
    output_dir: str,
    lat_min: Optional[float],
    lat_max: Optional[float],
    lon_min: Optional[float],
    lon_max: Optional[float],
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> None:
    """
    Filter a dataset by geographic coordinates and save the result.

    Parameters
    ----------
    input_path : str
        Path to the input CSV file containing the data to filter.
    output_dir : str
        Directory where the filtered file(s) will be saved.
    lat_min : float or None
        Minimum latitude (inclusive). If None, no lower bound is applied.
    lat_max : float or None
        Maximum latitude (inclusive). If None, no upper bound is applied.
    lon_min : float or None
        Minimum longitude (inclusive). If None, no lower bound is applied.
    lon_max : float or None
        Maximum longitude (inclusive). If None, no upper bound is applied.
    lat_col : str, default "lat"
        Name of the latitude column in the CSV.
    lon_col : str, default "lon"
        Name of the longitude column in the CSV.
    """
    # Ensure that at least one bound is specified, otherwise filtering is meaningless
    if lat_min is None and lat_max is None and lon_min is None and lon_max is None:
        raise ValueError(
            "You must specify at least one of: lat_min, lat_max, lon_min, lon_max"
        )

    logging.info("Starting geographic filtering...")
    logging.info(f"Bounds: lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}]")

    # Perform the actual filtering.
    # `filter_geo` is expected to:
    #   - load the CSV at `path`
    #   - apply the geographic bounds
    #   - save the filtered split to `output_dir`
    #   - return some statistics (e.g., number of rows before/after filtering)
    stats = filter_geo(
        path=input_path,
        split_name="train",  # label for this split, used for naming outputs
        output_dir=output_dir,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_col=lat_col,
        lon_col=lon_col,
    )

    logging.info(f"Filtering complete. Stats: {stats}")


if __name__ == "__main__":
    # Ensure log directory exists
    os.makedirs("log", exist_ok=True)
    log_path = os.path.join("log", "filter_geo.log")

    # Configure logging: log to file (and optionally to console)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),  # append to filter_geo.log
            # Uncomment the next line if you also want logs in the console
            # logging.StreamHandler(),
        ],
    )

    logging.info("=== Starting filter_geo script ===")

    # Parse command-line arguments
    args = parse_args()

    # Handle south_pole preset: predefined latitude band for the south polar region
    if args.south_pole:
        lat_min = -90
        lat_max = -60
        lon_min = None
        lon_max = None
        logging.info("Using south pole preset: lat=[-90, -60]")
    else:
        lat_min = args.lat_min
        lat_max = args.lat_max
        lon_min = args.lon_min
        lon_max = args.lon_max

    # Run the filtering pipeline with the chosen bounds
    filter_geo_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
    )

    logging.info("=== filter_geo script completed ===")
