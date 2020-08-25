#!/bin/bash

# Example:
# bash scripts/gen_lc_keras.bash 3 random 0
# bash scripts/gen_lc_keras.bash 3 flatten 0

DEVICE=$1
SAMPLING=$2
SPLIT=$3

OUTDIR=lc.out.linear.2M
mkdir -p $OUTDIR
echo "Outdir $OUTDIR"

LC_SIZES=7
# LC_SIZES=12

# EPOCH=1
# EPOCH=2
EPOCH=400

# SPLIT=0

export CUDA_VISIBLE_DEVICES=$1
echo "CUDA:     $CUDA_VISIBLE_DEVICES"
echo "Sampling: $SAMPLING"
echo "Split:    $SPLIT"

data_version="V5.1"

receptor="ADRP_6W02_A_1_H"
target="DIR.ml.ADRP_6W02_A_1_H.Orderable_zinc_db_enaHLL.sorted.4col"
ml_fname="ml.ADRP_6W02_A_1_H.Orderable_zinc_db_enaHLL.sorted.4col.descriptors.parquet"
sp_dname="ml.ADRP_6W02_A_1_H.Orderable_zinc_db_enaHLL.sorted.4col.descriptors.splits"
gout=$OUTDIR/lc.${receptor}.${SAMPLING}

dpath=data/$data_version-2M-$SAMPLING/$target/$ml_fname
spath=data/$data_version-2M-$SAMPLING/$target/$sp_dname

r=1
python src/main_lc.py \
    -dp $dpath \
    -sd $spath \
    --split_id $SPLIT \
    --rout run${r} \
    --ml keras \
    --epoch $EPOCH \
    --batchnorm \
    --gout $gout \
    --min_size 100000 \
    --lc_step_scale linear \
    --lc_sizes $LC_SIZES

    # --lc_sizes_arr 750000 700000 600000

    # --rout run${SPLIT} \
    # --lc_sizes_arr 500000 570000 640000

#     --min_size 20000 \
#     --lc_sizes $LC_SIZES
    # --ls_hpo_dir $ls_hpo_dir \
