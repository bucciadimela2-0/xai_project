"""
Global constants for preprocessing.
"""

# ---- Columns ----
FREQ_REQUIRED = 4_000_000.0
FREQ_COL = "FM_data_frequency"
FLUX_COL = "FM_data_F10_7_index"
TARGET_COL = "FM_data_peak_distorted_echo_power"

DROP_COLS = [
    "FM_data_ephemeris_time",
    "FM_data_median_corrected_echo_power",
    "FM_data_orbit_number",
    "FM_data_peak_corrected_echo_power",
    "FM_data_peak_simulated_echo_power",
    "FM_data_solar_longitude",
]

# ---- Splits / random ----
DEFAULT_TRAIN_SPLIT = 0.7
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_RANDOM_STATE = 42

# ---- Default outputs ----
DEFAULT_SAMPLING_OUTPUT = "data/dataset_subset/random/train_50k.csv"
DEFAULT_GEO_OUTPUT_DIR = "data/dataset_subset/geo"
DEFAULT_CLUSTER_OUTPUT_DIR = "data/dataset_subset/cluster"

# ---- Clustering ----
DEFAULT_N_CLUSTERS = 5

