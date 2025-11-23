
echo "Running example commands..."

TRAIN="data/data_cleaned/split/train.csv"
VAL="data/data_cleaned/split/val.csv"
TEST="data/data_cleaned/split/test.csv"

# Output folders
OUT_SAMPLING="data/dataset_subset/random"
OUT_GEO="data/dataset_subset/filter_geo"
OUT_CLUSTER="data/dataset_subset/clusters"

echo ">>> PREPROCESSING DATASET"

python preprocess.py \
  --input data/all_past.csv \
  --output_dir data/data_cleaned \
   --train_split 0.7 \
   --val_split 0.15 \
   --shuffle none

 Output:
 data/data_cleaned/split/train.csv
 data/data_cleaned/split/val.csv
 data/data_cleaned/split/test.csv

echo ">>> RANDOM SAMPLING EXAMPLE"

python -m data.dataset_subset.sampling \
    --input data/data_cleaned/split/train.csv \
    --output data/dataset_subset/random/train_50k.csv \
    --n_samples 50000 \
    --seed 42


echo ">>> RUNNING KMEANS CLUSTERING"

python -m data.dataset_subset.clusters \
    --train data/data_cleaned/split/train.csv \
    --val data/data_cleaned/split/val.csv \
    --test data/data_cleaned/split/test.csv \
    --output_dir data/dataset_subset/clusters \
    --n_clusters 5 \
    --drop_target \
    --save_model models/kmeans.pkl

echo ">>> RUNNING GEO-BASED KMEANS (lat/lon)"

python -m data.dataset_subset.clusters \
    --train data/data_cleaned/split/train.csv \
    --val data/data_cleaned/split/val.csv \
    --test data/data_cleaned/split/test.csv \
    --output_dir data/dataset_subset/clusters_geo \
    --n_clusters 6 \
    --columns FM_data_latitude FM_data_longitude \
    --drop_target \
    --save_model models/kmeans_geo.pkl


echo ">>> APPLYING SOUTH POLE GEO FILTER"

python -m data.dataset_subset.filter_geo \
    --train data/data_cleaned/split/train.csv \
    --val data/data_cleaned/split/val.csv \
    --test data/data_cleaned/split/test.csv \
    --output_dir data/dataset_subset/geo \
    --south_pole


echo "All example commands completed."








