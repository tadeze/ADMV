

import numpy as np
from util.common import *
from sklearn.metrics.pairwise import euclidean_distances
def abod(data, nn):
    scores = -knn_abod(data, max(np.ceil(nrow(data)*nn),min(3,nrow(data)-1)))
    return scores
def knn_abod(x, nn):
    assert isinstance(x, np.ndarray)
    tx = np.t(x)
    n = ncol(x)
    D = euclidean_distances(x)
    np.fill_diagnol(D,np.inf)
    neighbor = [np.argsort(D[:,i])[:nn] for i in range(n)]

    for i in range(n):
        mags = D[neighbor[i], i]
        relx = tx[:,i] - tx[:,i]




if __name__ == '__main__':
    print 8