#!/bin/bash

# Example:
# scripts/gen_lc_keras.bash 3 random 0

DEVICE=$1
SAMPLING=$2
SPLIT=$3

OUTDIR=lc.out.linear
mkdir -p $OUTDIR
echo "Outdir $OUTDIR"

LC_SIZES=7
# LC_SIZES=12

# EPOCH=2
EPOCH=500

# SPLIT=0

export CUDA_VISIBLE_DEVICES=$1
echo "CUDA:     $CUDA_VISIBLE_DEVICES"
echo "Sampling: $SAMPLING"
echo "Split:    $SPLIT"

data_version="V5.1"
target="DIR.ml.3CLPro_7BQY_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col"
ml_fname="ml.3CLPro_7BQY_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col.dd.parquet"
sp_dname="ml.3CLPro_7BQY_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col.dd.splits"
gout=$OUTDIR/lc.3CLPro.${SAMPLING}

dpath=data/$data_version-1M-$SAMPLING/$target/$ml_fname
spath=data/$data_version-1M-$SAMPLING/$target/$sp_dname
# ls_hpo_dir=k-tuner/${SOURCE}_${MODEL}_tuner_out/ls_hpo

# r=7
python src/main_lc.py \
    -dp $dpath \
    -sd $spath \
    --split_id $SPLIT \
    --ml keras \
    --epoch $EPOCH \
    --batchnorm \
    --gout $gout \
    --min_size 50000 \
    --lc_step_scale linear \
    --lc_sizes $LC_SIZES


    # --rout run${SPLIT} \
    # --lc_sizes_arr 500000 570000 640000

#     --min_size 20000 \
#     --lc_sizes $LC_SIZES
    # --ls_hpo_dir $ls_hpo_dir \
