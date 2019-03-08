import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'shrinkage_clustering')))

import numpy as np 
from sklearn.cluster import DBSCAN
from shrinkage_clustering import read_points, similarity_matrix
import matplotlib.pyplot as plt


def cluster_dbscan(from_file): 
    X = read_points(from_file)
    db = DBSCAN(eps=20, min_samples=5, metric='euclidean').fit(X)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    xs, ys = zip(*db.components_)

    return xs, ys, labels



cluster_dbscan(os.path.dirname(__file__) + '/../data/clusters')