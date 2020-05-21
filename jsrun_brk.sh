#!/bin/bash
# bsub -Is -W 0:30 -nnodes 1 -P MED106 $SHELL
# module load ibm-wml-ce/1.7.0-2
# conda activate gen

# Test
# CUDA_VISIBLE_DEVICES=1 
# python src/main_lc.py -dp $DATAPATH --lc_sizes_arr 128 2174 36937 --epoch $EPOCH --gout $gout --rout run"$id" -sc stnd > "$gout"/run"$id".log 2>&1
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh 1 7 inter-parts/2 2 exec >inter_jsrun.log 2>&1 &

device=$1
id=$2
global_sufx=$3
SET=$4

# DATAPATH="data/sample.data.parquet"

DATAPATH="data/ml.3CLPro_pocket1_dock.parquet"
SPLITPATH="data/ml.3CLPro_pocket1_dock.splits"
TARGET="3CLPro_pocket1"

# DATAPATH="data/ml.ADRP-ADPR_pocket1_dock.parquet"
# SPLITPATH="data/ml.ADRP-ADPR_pocket1_dock.splits"
# TARGET="ADRP-ADPR_pocket1"

gout="/gpfs/alpine/med106/scratch/$USER/$global_sufx/$TARGET"
mkdir -p $gout

export CUDA_VISIBLE_DEVICES=$device

EPOCH=2
# EPOCH=100
# EPOCH=350
# EPOCH=500

echo "Using cuda device $device"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Run id: $id"
echo "Global output: $gout"
echo "Target name: $TARGET"

# [128 329 845 2174 5589 14368 36937 94952 244088]
#  1   2   9
# [128 329 244088]
set1="128 329 244088"
#  3   4    8 
# [845 2174 94952]
set2="845 2174 94952"
#  5    6     7 
# [5589 14368 36937]
set3="5589 14368 36937"

if [[ $SET -eq 1 ]]; then
    echo "LC subset set $SET"
    python src/main_lc.py -dp $DATAPATH -sd $SPLITPATH --split_id $id --lc_sizes_arr $set1 \
        --epoch $EPOCH --gout $gout --rout run"$id" -sc stnd --ml keras > "$gout"/run"$id".log 2>&1
elif [[ $SET -eq 2 ]]; then
    echo "LC subset set $SET"
    python src/main_lc.py -dp $DATAPATH -sd $SPLITPATH --split_id $id --lc_sizes_arr $set2 \
        --epoch $EPOCH --gout $gout --rout run"$id" -sc stnd --ml keras > "$gout"/run"$id".log 2>&1
elif [[ $SET -eq 3 ]]; then
    echo "LC subset set $SET"
    python src/main_lc.py -dp $DATAPATH -sd $SPLITPATH --split_id $id --lc_sizes_arr $set3 \
        --epoch $EPOCH --gout $gout --rout run"$id" -sc stnd --ml keras > "$gout"/run"$id".log 2>&1
fi


# if [[ $SET -eq 1 ]]; then
#     echo "LC subset set $SET"
#     python src/main_lc.py -dp $DATAPATH -sd $SPLITPATH --split_id $id --lc_sizes_arr 128 329 244088 \
#         --epoch $EPOCH --gout $gout --rout run"$id" -sc stnd --ml keras > "$gout"/run"$id".log 2>&1
# elif [[ $SET -eq 2 ]]; then
#     echo "LC subset set $SET"
#     python src/main_lc.py -dp $DATAPATH -sd $SPLITPATH --split_id $id --lc_sizes_arr 845 2174 94952 \
#         --epoch $EPOCH --gout $gout --rout run"$id" -sc stnd --ml keras > "$gout"/run"$id".log 2>&1
# elif [[ $SET -eq 3 ]]; then
#     echo "LC subset set $SET"
#     python src/main_lc.py -dp $DATAPATH -sd $SPLITPATH --split_id $id --lc_sizes_arr 5589 14368 36937 \
#         --epoch $EPOCH --gout $gout --rout run"$id" -sc stnd --ml keras > "$gout"/run"$id".log 2>&1
# fi


