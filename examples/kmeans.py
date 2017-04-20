from __future__ import print_function

from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans

import numpy
import time
import sys
import os
from sklearn.metrics.cluster import normalized_mutual_info_score

HOME_DIR = "../"
DATA_FILE = "data/mnist"
OUTPUT_FILE = "result/kmeans.npz"

if __name__ == "__main__":
    
    spark = SparkSession\
        .builder\
        .appName("PythonKMeans")\
        .getOrCreate()
    
    # Loads data as RDD of (label, feature)
    dataset = spark.read.format("libsvm").load(HOME_DIR + DATA_FILE)

    # Trains a k-means model.
    kmeans = KMeans(k=10, seed=1, maxIter=50)
    model = kmeans.fit(dataset)
    
    # Evaluate normalized mutual information score
    transformed = (model
                   .transform(dataset)
                   .select("label", "prediction")
                   .rdd
                   .map(lambda a: (int(a['label']), a['prediction'])))
    label_pred_pair = transformed.collect()
    labels = [pair[0] for pair in label_pred_pair]
    preds = [pair[1] for pair in label_pred_pair]
    nmi = normalized_mutual_info_score(labels, preds)
    print('#####################################')
    print("NMI = " + str(nmi))

    # Evaluate clustering by computing Within Set Sum of Squared Errors.
    wssse = model.computeCost(dataset)
    print('#####################################')
    print("Within Set Sum of Squared Errors = " + str(wssse))
    
    numpy.savez(HOME_DIR + OUTPUT_FILE, nmi, wssse)
    
    spark.stop()
