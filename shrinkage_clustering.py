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


def remove_clusters(A, cluster_indeces): 
    return np.delete(A, cluster_indeces, axis=1)


def cluster(S, k=None, max_iter=100): 
    """
    Uses Shrinkage clustering to provide a cluster assignments matrix given a 
    similarity matrix S. 
    param S: Similarity matrix for all points to be clustered. 
    param k: Initial number of clusters. Will be reduced as the algorithm runs, 
             so start high.
    """
    N = len(S)
    if not k: 
        k = N

    S_bar = 1 - 2 * S
    A = random_cluster_matrix((N, k))

    for _i in range(max_iter): 
        # Remove empty clusters
        empty_columns = [i for i, c in enumerate(A.T) if sum(c) == 0]
        A = remove_clusters(A, empty_columns)
        k = len(A[0]) # Adjust number of clusters

        # Permute cluster memberships:
        # (a) Compute M = ~SA
        M = S_bar @ A

        # (b) Compute v
        MA = np.multiply(M, A)
        v = [min([M[i][j] for j in range(k)]) - sum([MA[i][j] for j in range(k)]) for i in range(N)]

        # (c) Find the object X with the greatest optimization potential
        X = np.argmin(v)

        # (d) Reassign X to the cluster C where C = argmin(M[X][j]) w.r.t. j
        C = np.argmin([M[X][j] for j in range(k)])
        prev = A[X][C]
        A[X] = np.zeros((k))
        A[X][C] = 1

    return A

cluster(np.random.random((10, 10)), k=5)