#!/bin/bash
#BSUB -P med106
#BSUB -W 0:10
#BSUB -nnodes 15
#BSUB -J my-bsub
# ----------------------------------------------

# You first need to load the appropriate module and activate the conda env!
# module load ibm-wml-ce/1.7.0-2
# conda activate gen

echo "Bash version ${BASH_VERSION}"

GPU_PER_NODE=6
NODES=15
# NODES=2
N_SPLITS=$(($NODES * $GPU_PER_NODE))

# GLOBAL_SUFX="bsub_2nodes_lines"
# GLOBAL_SUFX="bsub_2nodes_for"
GLOBAL_SUFX="bsub_15nodes_for"

echo "Number of nodes to use: $NODES"
echo "Number of GPUs per node: $GPU_PER_NODE"
echo "Number of data splits for LC: $N_SPLITS"
echo "Global outdir dir name: $GLOBAL_SUFX"

id=0
for node in $(seq 0 1 $(($NODES-1)) ); do
	for device in $(seq 0 1 5); do
		echo "Run $id (use device $device on node $node)"
		jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh $device $id $GLOBAL_SUFX exec >run"$id".log 2>&1 &
		id=$(($id+1))
	done
done 

# for node in $(seq 0 1 $(($NODES-1)) ); do
# 	echo "Use device 0 on node $node"
# done 

# for device in $(seq 0 1 5); do
# 	echo "Use device $device on node $node"
# done

# # Resources of node 1.
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 0 0 $GLOBAL_SUFX exec >run0.log 2>&1 &
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 1 1 $GLOBAL_SUFX exec >run1.log 2>&1 &
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 2 2 $GLOBAL_SUFX exec >run2.log 2>&1 &
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 3 3 $GLOBAL_SUFX exec >run3.log 2>&1 &
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 4 4 $GLOBAL_SUFX exec >run4.log 2>&1 &
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 5 5 $GLOBAL_SUFX exec >run5.log 2>&1 &

# # Resources of node 2.
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 0 6 $GLOBAL_SUFX exec >run6.log 2>&1 &
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 1 7 $GLOBAL_SUFX exec >run7.log 2>&1 &
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 2 8 $GLOBAL_SUFX exec >run8.log 2>&1 &
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 3 9 $GLOBAL_SUFX exec >run9.log 2>&1 &
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 4 10 $GLOBAL_SUFX exec >run10.log 2>&1 &
# jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun.sh 5 11 $GLOBAL_SUFX exec >run11.log 2>&1 &



