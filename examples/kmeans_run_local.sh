#!/usr/bin/env bash

INPUT_DIR="$HOME/Code/NystromKernelKmeans"
SPARK_HOME="$HOME/Software/spark-2.1.0"
PYTHON_FILE="$INPUT_DIR/examples/kmeans.py"
MASTER="local[4]"

export DATA_FILE="$INPUT_DIR/data/mnist"
export OUTPUT_FILE="$INPUT_DIR/result/kmeans.npz"

$SPARK_HOME/bin/spark-submit \
  --master $MASTER \
  $PYTHON_FILE -k 10