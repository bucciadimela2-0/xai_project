#!/bin/bash
set -e

###########################################
# CONFIGURATION (edit freely)
###########################################

INPUT="data/all_past.csv"
OUTPUT_DIR="data"

# Cleaning options
SKIP_CLEAN=true          # skip cleaning if clean.csv exists
REMOVE_FLUX=false        # drop solar flux column

# Split ratios + shuffle
TRAIN_SPLIT=0.7
VAL_SPLIT=0.15
SHUFFLE="none"
RANDOM_STATE=42

###########################################
# RANDOM SAMPLING
###########################################
DO_RANDOM=true
N_SAMPLES=10000
SAMPLING_OUTPUT="data/random/data_10k.csv"

###########################################
# CLUSTERING
###########################################
DO_CLUSTER=false
CLUSTER_OUTPUT_DIR="data/cluster"
N_CLUSTERS=20
COLUMNS="FM_data_latitude,FM_data_longitude"   # used for clustering
DROP_TARGET=true                                # drop target column before fit
TARGET_COL="FM_data_peak_distorted_echo_power"
SAVE_MODEL="models/kmeans.pkl"                  # where to save trained model

###########################################
# COLOR LOGGING
###########################################
GREEN="\033[1;32m"
BLUE="\033[1;34m"
YELLOW="\033[1;33m"
RESET="\033[0m"

echo -e "${BLUE}=== Starting Preprocessing Pipeline ===${RESET}"


###########################################
# SAFETY CHECKS
###########################################

# Ensure input CSV exists
if [[ ! -f "$INPUT" ]]; then
    echo -e "${YELLOW}[ERROR] Input file not found: $INPUT${RESET}"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Create sampling directory only if used
if [[ "$DO_RANDOM" = true ]]; then
    mkdir -p "$(dirname "$SAMPLING_OUTPUT")"
fi

# Create cluster directory only if used
if [[ "$DO_CLUSTER" = true ]]; then
    mkdir -p "$CLUSTER_OUTPUT_DIR"
fi

# Create model directory if saving KMeans model
if [[ -n "$SAVE_MODEL" ]]; then
    mkdir -p "$(dirname "$SAVE_MODEL")"
fi


###########################################
# BUILD COMMAND
###########################################

CMD="python preprocess/preprocess.py \
    --input $INPUT \
    --output_dir $OUTPUT_DIR \
    --train_split $TRAIN_SPLIT \
    --val_split $VAL_SPLIT \
    --shuffle $SHUFFLE \
    --random_state $RANDOM_STATE \
"

# Cleaning flags
[[ "$SKIP_CLEAN" = true ]] && CMD+=" --skip_clean"
[[ "$REMOVE_FLUX" = true ]] && CMD+=" --remove_flux"


###########################################
# RANDOM SAMPLING (optional)
###########################################
if [[ "$DO_RANDOM" = true ]]; then
    CMD+=" --do_random"
    CMD+=" --n_samples $N_SAMPLES"
    CMD+=" --sampling_output $SAMPLING_OUTPUT"
fi


###########################################
# CLUSTERING (optional)
###########################################
if [[ "$DO_CLUSTER" = true ]]; then

    # Enable clustering pipeline
    CMD+=" --do_cluster"

    # Output directory for cluster CSVs
    CMD+=" --cluster_output_dir $CLUSTER_OUTPUT_DIR"

    # Number of clusters
    CMD+=" --n_clusters $N_CLUSTERS"

    # Column subset for clustering
    if [[ -n "$COLUMNS" ]]; then
        CMD+=" --columns $COLUMNS"
    fi

    # Whether to drop the target column
    if [[ "$DROP_TARGET" = true ]]; then
        CMD+=" --drop_target"
    fi

    # Target column name
    CMD+=" --target_col $TARGET_COL"

    # Where to save trained KMeans model
    if [[ -n "$SAVE_MODEL" ]]; then
        CMD+=" --save_model $SAVE_MODEL"
    fi
fi


###########################################
# RUN COMMAND
###########################################

echo -e "${GREEN}Running command:${RESET}"
echo -e "${YELLOW}$CMD${RESET}"
echo ""

eval $CMD

echo -e "${BLUE}=== Preprocessing Completed Successfully ===${RESET}"
