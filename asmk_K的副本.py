from keras import backend as K
import numpy as np
from keras.backend.tensorflow_backend import expand_dims
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf

def squared_distance(X, Y=None, W=None):
    '''
    Calculates the pairwise distance between points in X and Y
    X:          n x d matrix
    Y:          m x d matrix
    sigma:      the exp gaussian parameter
    W:          affinity -- if provided, we normalize the distance
    returns:    n x m matrix of all pairwise squared Euclidean distances
    '''
    if Y is None:
        Y = X
    # distance = squaredDistance(X, Y)
    sum_dimensions = list(range(2, K.ndim(X) + 1))
    X = K.expand_dims(X, axis=1)
    if W is not None:
        # if W provided, we normalize X and Y by W
        D_diag = K.expand_dims(K.sqrt(K.sum(W, axis=1)), axis=1)
        X /= D_diag
        Y /= D_diag
    squared_difference = K.square(X - Y)
    distance = K.sum(squared_difference, axis=sum_dimensions)
    

    return distance

# asmk 
def asmk(X,n_nbrs,scale=None,scale_nbr=None, local_scale=None, verbose=False): # (X, k, scale=None, scale_nbr=None, local_scale=None, verbose=False)
    '''
    
    X:              input dataset of size n, tensor
    n_nbrs:          k<n
    scale:          provided scale sigma
    scale_nbr:      k2, used if sigma not provided
    local_scale:    if True, then we use the aforementioned option 3),
                    else we use option 2)
    verbose:        extra printouts
    returns:        n x n affinity matrix
    '''
    if isinstance(n_nbrs, np.float):
        n_nbrs = int(n_nbrs)
    elif isinstance(n_nbrs, tf.Variable) and n_nbrs.dtype.as_numpy_dtype != np.int32:
        n_nbrs = tf.cast(n_nbrs, np.int32)
    #get squared distance
    Dx = squared_distance(X)
    #calculate the top k neighbors of minus the distance (so the k closest neighbors)
    nn = tf.nn.top_k(-Dx, n_nbrs, sorted=True)

    vals = nn[0]
    # apply scale
    if scale is None:
        # if scale not provided, use local scale
        if scale_nbr is None:
            scale_nbr = 0
        else:
            print("getAffinity scale_nbr, n_nbrs:", scale_nbr, n_nbrs)
            assert scale_nbr > 0 and scale_nbr <= n_nbrs
        if local_scale:
            scale = -nn[0][:, scale_nbr - 1]
            scale = tf.reshape(scale, [-1, 1])
            scale = tf.tile(scale, [1, n_nbrs])
            scale = tf.reshape(scale, [-1, 1])
            vals = tf.reshape(vals, [-1, 1])
            if verbose:
                vals = tf.Print(vals, [tf.shape(vals), tf.shape(scale)], "vals, scale shape")
            vals = vals / (2*scale)
            vals = tf.reshape(vals, [-1, n_nbrs])
        else:
            def get_median(scales, m):
                with tf.device('/cpu:0'):
                    scales = tf.nn.top_k(scales, m)[0]
                scale = scales[m - 1]
                return scale, scales
            scales = -vals[:, scale_nbr - 1]
            const = tf.shape(X)[0] // 2
            scale, scales = get_median(scales, const)
            vals = vals / (2 * scale)
    else:
        # otherwise, use provided value for global scale
        vals = vals / (2*scale**2)

    np.set_printoptions(precision=4,suppress=1)
    with tf.Session() as sess:
        X1 = X.eval()
        n = X1.shape[0]
        #print(n)
    # with tf.Session() as sess1:
    #     n1 = n.eval()
    P = np.zeros([n,n])
    #P = tf.Variable(tf.zeros([n,n], tf.float64))
    dst = squared_distance(X)
    d1 = 2-2*K.exp(- dst /(2*scale**2))

    with tf.Session() as sess:
        
        d2 = d1.eval()
    d2v = np.sort(d2,axis=1)
    d2i = np.argsort(d2,axis=1)

    for i in range(n):
        kdv = d2v[i,1:n_nbrs+2]
        kdi = d2i[i,1:n_nbrs+2]
        P[i,kdi]= (kdv[n_nbrs]-kdv)/(
                 n_nbrs*kdv[n_nbrs]-np.sum(kdv[0:n_nbrs])+np.finfo(float).eps)
    W = (P+P.T)/2
    W = tf.convert_to_tensor(W, dtype=tf.float64)
    return W


if __name__ == "__main__":
    
    x = np.arange(25).reshape(5,5,order='F')
    print(x)
    xx = tf.convert_to_tensor(x, dtype=tf.float32)

    w = asmk(xx,n_nbrs=3,scale=1)
    #xd = squared_distance(xx,1)

    with tf.Session() as sess:
        #print(xd.eval())
        print(w.eval())