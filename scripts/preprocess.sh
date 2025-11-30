#!/bin/bash
set -e

python preprocess/preprocess.py \
  --input data/all_past.csv \
  --output_dir data \
  --train_split 0.7 \
  --val_split 0.15 \
  --shuffle none \
  --do_random \
  --sampling_output data/random10/data_10k.csv 
  #--do_geo \
  #--do_cluster \
  #--geo_output_dir data/geo\
 # --cluster_output_dir data/cluster \
  #--n_clusters 5
