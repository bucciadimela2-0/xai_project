#!/bin/bash
set -e

# Config
TRAIN_CSV="data/random10/split_random/train.csv"
VAL_CSV="data/random10/split_random/val.csv"
TEST_CSV="data/random10/split_random/test.csv"
OUTPUT_DIR="experiments/random10"


mkdir -p "$OUTPUT_DIR"


python symbolic/symbolic.py \
  --train_csv "$TRAIN_CSV" \
  --val_csv "$VAL_CSV" \
  --test_csv "$TEST_CSV" \
  --output_dir "$OUTPUT_DIR" \
  --target_col FM_data_peak_distorted_echo_power \
  --seed 42

