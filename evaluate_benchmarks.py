import numpy as np 
from time import time
from shrinkage_clustering import cluster_shrinkage_clustering
from benchmarks.dbscan import cluster_dbscan

def timeit(method):
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        
        print(f'{method.__name__} took {(te - ts) * 1000} seconds to complete.')
        return result
    return timed


def evaluate(named_funcs): 
    """
    Provided a list of functions to evaluate, plots their result as a 
    scatterplot and prints its running time. 
    """
    for name, func in named_funcs.items(): 
        print(name, func)
        pass


evaluate({
    "Shrinkage Clustering" : cluster_shrinkage_clustering,
    "DBSCAN" : cluster_dbscan
})

