import numpy as np 
from math import sqrt
from random import randint

def euclidean_distance(x, y): 
    """
    Takes two coordinates as input, returns the euclidean distance between them. 
    param x: A pair (x, y). 
    param y: A pair (x, y).
    """
    x1, y1 = x
    x2, y2 = y
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)


def random_cluster_matrix(shape): 
    """
    Creates an (N, M) matrix where a random cell in each row is 1, otherwise 0.
    param shape: A pair (N, M).
    """
    N, k = shape
    A = np.zeros((N, k))
    for row in A: 
        row[randint(0, k - 1)] = 1

    return A


def cluster(S, k=None): 
    """
    Uses Shrinkage clustering to provide a cluster assignments matrix given a 
    similarity matrix S. 
    param S: Similarity matrix for all points to be clustered. 
    param k: Initial number of clusters. Will be reduced as the algorithm runs, 
             so start high.
    """
    if not k: 
        k = len(S)

    N = len(S)

    A = random_cluster_matrix((N, k))

    print(A)


cluster(np.zeros((10, 10)), k=5)