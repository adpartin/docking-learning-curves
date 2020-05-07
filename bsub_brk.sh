#!/bin/bash
#BSUB -P med106
#BSUB -W 00:10
#BSUB -nnodes 50
#BSUB -J my-bsub
# ----------------------------------------------

# You first need to load the appropriate module and activate the conda env!
# module load ibm-wml-ce/1.7.0-2
# conda activate gen

echo "Bash version ${BASH_VERSION}"

# Set resources based on number of nodes
# GPU_PER_NODE=6
# NODES=15
# # NODES=2
# N_SPLITS=$(($NODES * $GPU_PER_NODE))

# Set resources based on number of splits
# N_SPLITS=100
# PARTS=3
# GPUs=$(($N_SPLITS * $PARTS))
# NODES=$(($GPUs / $GPU_PER_NODE))

# GLOBAL_SUFX="bsub_2nodes_lines"
# GLOBAL_SUFX="bsub_2nodes_for"
# GLOBAL_SUFX="bsub_15nodes_for"
GLOBAL_SUFX="bsub_3parts"

# echo "Number of nodes to use: $NODES"
# echo "Number of GPUs per node: $GPU_PER_NODE"
# echo "Number of data splits for LC: $N_SPLITS"
# echo "Global outdir dir name: $GLOBAL_SUFX"

# gpu_counter=0
# node_counter=0
# device=0
# for split in $(seq 0 1 $N_SPLITS); do
#     if [[ $device -lt 6 ]]; then
#         jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_b1.sh $device $id $GLOBAL_SUFX exec >run"$id".log 2>&1 &
#     else
#         device=0
#         jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_b1.sh $device $id $GLOBAL_SUFX exec >run"$id".log 2>&1 &
#     fi
#     device=$(($device + 1))

# gpu_counter=0
# node_counter=0
# device=-1
# for split in $(seq 0 1 $N_SPLITS); do
#     device=$(($device + 1))
#     if [[ $device -eq 6 ]]; then
#         device=0
#     fi
#     echo "Device $device"
#     # jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_b1.sh $device $id $GLOBAL_SUFX exec >run"$id".log 2>&1 &
# done

SET=1
out_dir=$GLOBAL_SUFX/$SET
echo "Dir $out_dir"
for split in $(seq 0 1 $N_SPLITS); do
    device=$(($split % 6))
    jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh $device $split $out_dir $SET exec >run"$split".log 2>&1 &
done

SET=2
out_dir=$GLOBAL_SUFX/$SET
echo "Dir $out_dir"
for split in $(seq 0 1 $N_SPLITS); do
    device=$(($split % 6))
    echo "Dir $out_dir"
    jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh $device $split $out_dir $SET exec >run"$split".log 2>&1 &
done

SET=3
out_dir=$GLOBAL_SUFX/$SET
echo "Dir $out_dir"
for split in $(seq 0 1 $N_SPLITS); do
    device=$(($split % 6))
    jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh $device $split $out_dir $SET exec >run"$split".log 2>&1 &
done



