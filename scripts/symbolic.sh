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
# EXPERIMENT CONFIG
###########################################

CONFIG_YAML="configs/experiments/random10_exp1.yaml"

TRAIN_CSV="data/random10/split_random/train.csv"
VAL_CSV="data/random10/split_random/val.csv"
TEST_CSV="data/random10/split_random/test.csv"

OUTPUT_DIR="experiments/random10/exp1"
TARGET_COL="FM_data_peak_distorted_echo_power"
SEED=42

###########################################
# CHECKS
###########################################

echo -e "${BLUE}=== Starting PySR experiment: $CONFIG_YAML ===${RESET}"

mkdir -p "$OUTPUT_DIR"

[[ ! -f "$TRAIN_CSV" ]] && echo -e "${RED}[ERROR] TRAIN missing: $TRAIN_CSV${RESET}" && exit 1
[[ ! -f "$VAL_CSV" ]] && echo -e "${RED}[ERROR] VAL missing: $VAL_CSV${RESET}" && exit 1
[[ ! -f "$TEST_CSV" ]] && echo -e "${RED}[ERROR] TEST missing: $TEST_CSV${RESET}" && exit 1
[[ ! -f "$CONFIG_YAML" ]] && echo -e "${RED}[ERROR] YAML missing: $CONFIG_YAML${RESET}" && exit 1

###########################################
# COMMAND
###########################################

CMD="python symbolic/symbolic.py \
  --config $CONFIG_YAML \
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

eval $CMD

echo -e "${BLUE}=== Experiment completed ===${RESET}"
echo -e "${GREEN}Results saved in: $OUTPUT_DIR${RESET}"
