#!/bin/bash
#SBATCH -p debug
#SBATCH -N 2
#SBATCH -C haswell
#SBATCH -t 00:30:00
#SBATCH -J shusen_kmeans
#SBATCH -L SCRATCH
#SBATCH -e mysparkjob_%j.err
#SBATCH -o mysparkjob_%j.out
#SBATCH --ccm

MASTER=$SPARKURL
INPUT_DIR="$SCRATCH/NystromKernelKmeans"
PYTHON_FILE="$INPUT_DIR/examples/kmeans.py"

export DATA_FILE="$INPUT_DIR/data/mnist"
export OUTPUT_FILE="$INPUT_DIR/result/kmeans.npz"

module load python/3.5-anaconda
module load spark
start-all.sh

spark-submit \
    --master $MASTER \
    $PYTHON_FILE -k 10
    
stop-all.sh
