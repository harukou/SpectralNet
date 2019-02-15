from contextlib import contextmanager
import os, sys

from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.stats import norm
import sklearn.metrics

from keras import backend as K
from keras.callbacks import Callback
import tensorflow as tf

from core import costs as cf
from munkres import Munkres



def spectral_clustering(x, scale, n_nbrs=None, affinity='full', W=None):
    '''
    Computes the eigenvectors of the graph Laplacian of x,
    using the full Gaussian affinity matrix (full), the
    symmetrized Gaussian affinity matrix with k nonzero
    affinities for each point (knn), or the Siamese affinity
    matrix (siamese)
    x:          input data
    n_nbrs:     number of neighbors used
    affinity:   the aforementeiond affinity mode
    returns:    the eigenvectors of the spectral clustering algorithm
    '''
    if affinity == 'full':
        W =  K.eval(cf.full_affinity(K.variable(x), scale))
    elif affinity == 'knn':
        if n_nbrs is None:
            raise ValueError('n_nbrs must be provided if affinity = knn!')
        W =  K.eval(cf.knn_affinity(K.variable(x), scale, n_nbrs))
    elif affinity == 'siamese':
        if W is None:
            print ('no affinity matrix supplied')
            return
    d = np.sum(W, axis=1)
    D = np.diag(d)
    # (unnormalized) graph laplacian for spectral clustering
    L = D - W
    Lambda, V = np.linalg.eigh(L)
    return(Lambda, V)
