import numpy as np 
from math import sqrt, isclose
from random import randint, shuffle
import matplotlib.pyplot as plt     
import matplotlib.animation as animation             


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


def progress(current, max): 
    prog = int(20 * (current / max))
    print(f"[{''.join('=' for _ in range(prog))}>{''.join(' ' for _ in range(20 - prog - 1))}] ({(current / max) * 100 : 2.2f}%)", end='\r')




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
        MoA = np.multiply(M, A)
        #v = [min([M[i][j] - sum([MoA[i][j]]) for j in range(k)]) for i in range(N)]
        v = [min([M[i][j] for j in range(k)]) - sum([MoA[i][j] for j in range(k)]) for i in range(N)]

        # Check if we converged
        if isclose(sum(v), 0, abs_tol=1e-5): 
            break

        # (c) Find the object X with the greatest optimization potential
        X = np.argmin(v)

        # (d) Reassign X to the cluster C where C = argmin(M[X][j]) w.r.t. j
        C = np.argmin([M[X][j] for j in range(k)])
        A[X] = np.zeros((k))
        A[X][C] = 1
        progress(_i, max_iter)

    return A


def square(S):
    """
    Converts a triangular matrix into a full square matrix. 
    Example: 
    [
        [0  0  0  0]       [0  3  2  4]
        [3  0  0  0]  -->  [3  0  3  1]
        [2  3  0  0]       [2  3  0  5]
        [4  0  0  0]       [4  1  5  0]
    ]
    """
    full = S.T + S
    idx = np.arange(S.shape[0])
    full[idx,idx] = S[idx,idx]
    return full 


def read_points(from_file): 
    points = []
    with open(from_file) as fp: 
        for line in fp.readlines(): 
            feats = line.strip().split()
            points.append((int(feats[0]), int(feats[1])))

    return points


def similarity_matrix(P):
    N = len(P) 
    S = np.zeros((N, N))
    for i in range(N): 
        for j in range(i): 
            S[i][j] = euclidean_distance(P[i], P[j])

    S = square(S)
    S = S / np.max(S)
    S = 1 - S # Higher value = more similar

    return S


def test(): 
    print(f'Reading...')
    points = read_points('data/synth')
    print(f'Calculating similarity matrix...')
    S = similarity_matrix(points)
    print(f'Clustering...')
    A = cluster(S, k=50, max_iter=5000)

    # Visualise
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    labels = [np.argmax(p) for p in A]
    xs, ys = zip(*points)
    ax.scatter(xs, ys, c=labels)
    plt.show()



# cluster(np.random.random((10, 10)), k=5)
test()

