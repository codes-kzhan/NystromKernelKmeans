#!/bin/bash
#SBATCH -p debug
#SBATCH -N 2
#SBATCH -C haswell
#SBATCH -t 00:30:00
#SBATCH -J shusen_kmeans
#SBATCH -L SCRATCH
#SBATCH -e mysparkjob_%j.err
#SBATCH -o mysparkjob_%j.out

module load python/3.5-anaconda
module load spark
start-all.sh

spark-submit $SPARK_EXAMPLES/python/pi.py
    
stop-all.sh
