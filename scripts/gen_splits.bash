#!/bin/bash

data_version="V5.1"
target="DIR.ml.3CLPro_7BQY_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col"
ml_fname="ml.3CLPro_7BQY_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col.dd.parquet"

sampling=random
# sampling=flatten

# dpath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge.$sampling/data.$SOURCE.dd.ge.parquet
# gout=data/ml.dfs/$data_version/data.$SOURCE.dd.ge.$sampling

dpath=data/$data_version-1M-$sampling/$target/$ml_fname
gout=data/$data_version-1M-$sampling/$target

python ../../ml-data-splits/src/main_data_split.py \
    -dp $dpath \
    --gout $gout \
    --trg_name cls \
    -ns 20 \
    -cvm strat \
    --te_size 0.10
