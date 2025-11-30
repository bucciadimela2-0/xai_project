"""
Global constants used across the preprocessing pipeline.

These include:
- Domain-specific column names
- Default split ratios
- Default random seeds
- Default output paths
- Default clustering parameters
"""

# ----------------- Domain-specific column names -----------------

# Required fixed frequency value (in Hz) for filtering the dataset
FREQ_REQUIRED = 4_000_000.0

# Name of the frequency column in the dataset
FREQ_COL = "FM_data_frequency"

# Name of the solar flux column (F10.7 index)
FLUX_COL = "FM_data_F10_7_index"

# Name of the target column used for regression / prediction
TARGET_COL = "FM_data_peak_distorted_echo_power"

# Columns that are considered unnecessary for the learning task
# and can be dropped during preprocessing
DROP_COLS = [
    "FM_data_ephemeris_time",
    "FM_data_median_corrected_echo_power",
    "FM_data_orbit_number",
    "FM_data_peak_corrected_echo_power",
    "FM_data_peak_simulated_echo_power",
    "FM_data_solar_longitude",
]

# ----------------- Default split & random params -----------------

# Default train/validation split ratios
DEFAULT_TRAIN_SPLIT = 0.7
DEFAULT_VAL_SPLIT = 0.15  # the rest (0.15) will be used for test

# Default random seed for reproducibility
DEFAULT_RANDOM_STATE = 42

# ----------------- Default output paths & clustering params -----------------

# Default output path for random sampling subset
DEFAULT_SAMPLING_OUTPUT = "data/dataset_subset/random/train_50k.csv"

# Default directory for geographic-filtered subsets
DEFAULT_GEO_OUTPUT_DIR = "data/dataset_subset/geo"

# Default directory for cluster-based subsets
DEFAULT_CLUSTER_OUTPUT_DIR = "data/dataset_subset/cluster"

# Default number of clusters for K-Means
DEFAULT_N_CLUSTERS = 5
