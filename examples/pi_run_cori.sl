#!/bin/bash
#SBATCH -p debug
#SBATCH -N 2
#SBATCH -C haswell
#SBATCH -t 00:30:00
#SBATCH -J shusen_kmeans
#SBATCH -L SCRATCH
#SBATCH -e mysparkjob_%j.err
#SBATCH -o mysparkjob_%j.out

module load spark
start-all.sh


spark-submit \
    --master $SPARKURL \
    --driver-memory 15G \
    --executor-memory 20G \
    $SPARK_EXAMPLES/python/pi.py
    
stop-all.sh
