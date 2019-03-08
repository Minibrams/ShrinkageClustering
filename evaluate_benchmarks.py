import numpy as np 
from time import time
from shrinkage_clustering import cluster_shrinkage_clustering
from benchmarks.dbscan import cluster_dbscan
import matplotlib.pyplot as plt


def timeit(method):
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        
        print(f'{method.__name__} took {(te - ts) * 1000} seconds to complete.')
        return result
    return timed


def evaluate(from_file, named_funcs): 
    """
    Provided a list of functions to evaluate, plots their result as a 
    scatterplot and prints its running time. 
    """
    fig = plt.figure()
    cols, rows = 3, 3

    for i, (name, func) in enumerate(named_funcs.items(), start=1): 
        start = time()
        xs, ys, labels = func(from_file)
        end = time()
        scatter(data=(xs, ys), labels=labels, title=name, fig=fig, pos=(rows, cols, i), time_taken=(end-start))

    fig.tight_layout()
    plt.show()


def scatter(data, labels, title, fig, pos, time_taken): 
    r, c, i = pos
    xs, ys = data
    ax = fig.add_subplot(r, c, i)
    ax.scatter(xs, ys, c=labels[:len(xs)], marker='.', alpha=0.5)
    ax.set_title(f'{title}\n({time_taken : 2.2f} seconds )')
    

evaluate(from_file='data/two_circles', named_funcs={
    "Shrinkage Clustering" : cluster_shrinkage_clustering,
    "DBSCAN" : cluster_dbscan
})

