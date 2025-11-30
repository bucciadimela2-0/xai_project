import argparse


def parse_args():
    """
    Parse command-line arguments for geographic filtering of a dataset.
    This includes support for:
      - Custom latitude/longitude ranges
      - Automatic south-pole preset
      - Optional filtering of validation/test splits
    """
    parser = argparse.ArgumentParser(
        description="Apply geographic filtering to a dataset (including support for the Martian south pole)."
    )

    parser.add_argument(
        "--input",
        help="Path to the TRAIN CSV file. Optional if used within a larger pipeline.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where the filtered dataset will be saved.",
    )

    parser.add_argument(
        "--lat_min",
        type=float,
        help="Minimum latitude value (inclusive). Example: -90",
    )
    parser.add_argument(
        "--lat_max",
        type=float,
        help="Maximum latitude value (inclusive). Example: -70",
    )
    parser.add_argument(
        "--lon_min",
        type=float,
        help="Minimum longitude value (inclusive).",
    )
    parser.add_argument(
        "--lon_max",
        type=float,
        help="Maximum longitude value (inclusive).",
    )

    parser.add_argument(
        "--lat_col",
        type=str,
        default="FM_data_latitude",
        help="Name of the latitude column (default: FM_data_latitude).",
    )
    parser.add_argument(
        "--lon_col",
        type=str,
        default="FM_data_longitude",
        help="Name of the longitude column (default: FM_data_longitude).",
    )

    parser.add_argument(
        "--south_pole",
        action="store_true",
        help=(
            "Automatically filter for the predefined Martian south pole region "
            "(typically lat <= -84). Overrides manual latitude/longitude values."
        ),
    )

    parser.add_argument(
        "--filter_val",
        action="store_true",
        help=(
            "Apply the geographic filter also to the VALIDATION split. "
            "Usually disabled for SR experiments to avoid distribution shifts."
        ),
    )
    parser.add_argument(
        "--filter_test",
        action="store_true",
        help=(
            "Apply the geographic filter also to the TEST split. "
            "Usually disabled for SR experiments to preserve evaluation integrity."
        ),
    )

    return parser.parse_args()
