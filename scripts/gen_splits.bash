#!/bin/bash

data_ver="V5.1"
# target="DIR.ml.3CLPro_7BQY_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col"
# ml_fname="ml.3CLPro_7BQY_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col.descriptors.parquet"
target="DIR.ml.ADRP_6W02_A_1_H.Orderable_zinc_db_enaHLL.sorted.4col"
ml_fname="ml.ADRP_6W02_A_1_H.Orderable_zinc_db_enaHLL.sorted.4col.descriptors.parquet"

sampling=random
# sampling=flatten

# dpath=data/ml.dfs/$data_ver/data.$SOURCE.dd.ge.$sampling/data.$SOURCE.dd.ge.parquet
# gout=data/ml.dfs/$data_ver/data.$SOURCE.dd.ge.$sampling

# dpath=data/$data_ver-1M-$sampling/$target/$ml_fname
# gout=data/$data_ver-1M-$sampling/$target

dpath=data/$data_ver-2M-$sampling-dd-fps/$target/$ml_fname
gout=data/$data_ver-2M-$sampling-dd-fps/$target

# python ../../ml-data-splits/src/main_data_split.py \
python ../ml-data-splits/src/main_data_split.py \
    -dp $dpath \
    --gout $gout \
    --trg_name cls \
    -ns 20 \
    -cvm strat \
    --te_size 0.05
