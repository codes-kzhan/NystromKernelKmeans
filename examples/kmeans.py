from __future__ import print_function

from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans

import numpy
import optparse
import time
import sys
import os
from sklearn.metrics.cluster import normalized_mutual_info_score

DATA_FILE = os.environ['DATA_FILE']
OUTPUT_FILE = os.environ['OUTPUT_FILE']

def main():
    parser = optparse.OptionParser()
    parser.add_option('-k', '--cluster_num',
                  dest="cluster_num",
                  default=2,
                  type="int",
                  help="cluster number"
                  )
    parser.add_option('-m', '--max_iter',
                  dest="max_iter",
                  default=10,
                  type="int",
                  help="max number of iterations"
                  )
    options, reminder = parser.parse_args()
    CLUSTER_NUM = options.cluster_num
    MAX_ITER = options.max_iter

    
    spark = SparkSession\
        .builder\
        .appName("PythonKMeans")\
        .getOrCreate()
    
    # Loads data as RDD of (label, feature)
    dataset = spark.read.format("libsvm").load(DATA_FILE)

    # Trains a k-means model.
    kmeans = KMeans(k=CLUSTER_NUM, seed=1, maxIter=MAX_ITER)
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
    
    numpy.savez(OUTPUT_FILE, nmi, wssse)
    
    
    print('#####################################')
    print(CLUSTER_NUM)
    print(MAX_ITER)
    
    spark.stop()


if __name__ == "__main__":
    main()
