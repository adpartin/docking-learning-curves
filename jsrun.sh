#!/bin/bash
# bsub -Is -W 0:30 -nnodes 1 -P MED106 $SHELL
# module load ibm-wml-ce/1.7.0-2
# conda activate gen
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 1 0 inter-1node exec >inter_jsrun.log 2>&1 &
# jsrun --nrs 1 --tasks_per_rs 1 --cpu_per_rs 4 --gpu_per_rs 1 ./jsrun.sh 0 0 inter-1node exec >inter_jsru.log 2>&1 &

# Test
# CUDA_VISIBLE_DEVICES=1 python -m pdb src/main_lc.py -dp data/ml.ADRP.parquet --gout /gpfs/alpine/med106/scratch/$USER/direct --rout run0

device=$1
id=$2
global_sufx=$3

gout="/gpfs/alpine/med106/scratch/$USER/$global_sufx"
mkdir -p $gout

export CUDA_VISIBLE_DEVICES=$device

DATAPATH="data/sample.data.parquet"
EPOCH=2
# EPOCH=200

echo "Using cuda device $device"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Run id: $id"
echo "Global output: $gout"

python src/main_lc.py -dp $DATAPATH --epoch $EPOCH --gout $gout --rout run"$id" -sc stnd > "$gout"/run"$id".log 2>&1


