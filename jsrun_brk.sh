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

gout="/gpfs/alpine/med106/scratch/$USER/$global_sufx"
mkdir -p $gout

export CUDA_VISIBLE_DEVICES=$device

# DATAPATH="data/sample.data.parquet"
DATAPATH="data/ml.ADRP.parquet"
EPOCH=2
# EPOCH=350

echo "Using cuda device $device"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Run id: $id"
echo "Global output: $gout"

# [128 329 845 2174 5589 14368 36937 94952 244088]
# [128 2174 36937]
# [329 5589 94952]
# [845 14368 244088]

if [[ $SET -eq 1 ]]; then
    echo "Set $SET"
    python src/main_lc.py -dp $DATAPATH --lc_sizes_arr 128 2174 36937 --epoch $EPOCH \
        --gout $gout --rout run"$id" -sc stnd > "$gout"/run"$id".log 2>&1
elif [[ $SET -eq 2 ]]; then
    echo "Set $SET"
    python src/main_lc.py -dp $DATAPATH --lc_sizes_arr 329 5589 94952 --epoch $EPOCH \
        --gout $gout --rout run"$id" -sc stnd > "$gout"/run"$id".log 2>&1
elif [[ $SET -eq 3 ]]; then
    echo "Set $SET"
    python src/main_lc.py -dp $DATAPATH --lc_sizes_arr 845 14368 244088 --epoch $EPOCH \
        --gout $gout --rout run"$id" -sc stnd > "$gout"/run"$id".log 2>&1
fi


