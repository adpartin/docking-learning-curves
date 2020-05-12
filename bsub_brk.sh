#!/bin/bash
#BSUB -P med106
#BSUB -W 06:00
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
N_SPLITS=100
START_SPLIT=1
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

SET=1
out_dir=$GLOBAL_SUFX/$SET
echo "Dir $out_dir"
for split in $(seq $START_SPLIT 1 $N_SPLITS); do
    device=$(($split % 6))
    echo "Set $SET; Split $split; Device $device"
    jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh $device $split $out_dir $SET exec >logs/run"$split".log 2>&1 &
done

SET=2
out_dir=$GLOBAL_SUFX/$SET
echo "Dir $out_dir"
for split in $(seq $START_SPLIT 1 $N_SPLITS); do
    device=$(($split % 6))
    echo "Set $SET; Split $split; Device $device"
    jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh $device $split $out_dir $SET exec >logs/run"$split".log 2>&1 &
done

SET=3
out_dir=$GLOBAL_SUFX/$SET
echo "Dir $out_dir"
for split in $(seq $START_SPLIT 1 $N_SPLITS); do
    device=$(($split % 6))
    echo "Set $SET; Split $split; Device $device"
    jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh $device $split $out_dir $SET exec >logs/run"$split".log 2>&1 &
done



