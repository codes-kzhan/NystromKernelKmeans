#!/usr/bin/env bash

SPARK_HOME="$HOME/Software/spark-2.1.0"
INPUT_DIR="$HOME/Code/NystromKernelKmeans"
PYTHON_FILE="$INPUT_DIR/examples/kmeans.py"
MASTER="local[4]"



$SPARK_HOME/bin/spark-submit $PYTHON_FILE \
  --verbose \
  --master $MASTER 