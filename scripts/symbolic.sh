#!/bin/bash
set -e

###########################################
# COLORS
###########################################
BLUE="\033[1;34m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

###########################################
# CONFIG
###########################################

TRAIN_CSV="data/random10/split_random/train.csv"
VAL_CSV="data/random10/split_random/val.csv"
TEST_CSV="data/random10/split_random/test.csv"

OUTPUT_DIR="experiments/random10"

TARGET_COL="FM_data_peak_distorted_echo_power"
SEED=42

###########################################
# START
###########################################

echo -e "${BLUE}=== Starting PySR symbolic regression ===${RESET}"
mkdir -p "$OUTPUT_DIR"

# Validate input CSVs
[[ ! -f "$TRAIN_CSV" ]] && echo -e "${RED}[ERROR] TRAIN missing: $TRAIN_CSV${RESET}" && exit 1
[[ ! -f "$VAL_CSV" ]] && echo -e "${RED}[ERROR] VAL missing: $VAL_CSV${RESET}" && exit 1
[[ ! -f "$TEST_CSV" ]] && echo -e "${RED}[ERROR] TEST missing: $TEST_CSV${RESET}" && exit 1

###########################################
# BUILD COMMAND
###########################################

CMD="python symbolic/symbolic.py \
  --train_csv $TRAIN_CSV \
  --val_csv $VAL_CSV \
  --test_csv $TEST_CSV \
  --output_dir $OUTPUT_DIR \
  --target_col $TARGET_COL \
  --seed $SEED \
"

echo -e "${YELLOW}Executing:${RESET}"
echo -e "${GREEN}$CMD${RESET}"
echo ""

###########################################
# RUN
###########################################

eval $CMD

echo -e "${BLUE}=== PySR symbolic regression completed ===${RESET}"
echo -e "${GREEN}Results saved to: $OUTPUT_DIR${RESET}"
