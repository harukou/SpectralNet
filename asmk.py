import numpy as np
from numpy.linalg import norm
from numpy.matlib import repmat
from keras import backend as K
import tensorflow as tf

# Method 1 L2
def L2(x,y):
    '''
    x,y: 2 matrices, each row is a data
    return L2 distance matrix of x and y
    '''
    xx = np.sum(x*x,axis=1,keepdims=1) # row*1
    yy = np.sum(y*y,axis=1,keepdims=1) # row*1
    xy = x.dot(y.T)                    # row*row
    
    x2 = repmat(xx.T,len(yy),1)        # row*row
    y2 = repmat(yy,1,len(xx))          # row*row
    d = x2 + y2 - 2*xy    
    return d

def d2ij2(X,sigma): 
    return 2-2*np.exp(-L2(X,X)/(2*sigma**2))

# Method 2 L2
def L2v(a,b):
    '''
    a,b: vectors
    return: L2 distance of 2 vector
    '''
    return np.sum(np.square(a-b))

def L2D(x,y):
    '''
    x,y: matrice n*d
    return L2 distance of 2 matrix
    '''
    n = len(x)
    D = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            D[i][j] = L2v(x[i],y[j])
    return D

def d2ij(X,sigma): 
    return 2-2*np.exp(-L2D(X,X)/(2*sigma**2))

# asmk 
def asmk(x,k,sigma):
    '''
    X: input matrix m*d  data num:m
    k: n nearest neighbor
    return w (similarity matrix)
    '''
    np.set_printoptions(precision=4,suppress=1)

    n = len(x)

    P = np.zeros([n,n])
    d2 = d2ij2(x,sigma)
    d2v = np.sort(d2,axis=1)
    d2i = np.argsort(d2,axis=1)

    for i in range(n):
        kdv = d2v[i,1:k+2]
        kdi = d2i[i,1:k+2]
        P[i,kdi] = (kdv[k]-kdv)/(
                  k*kdv[k]-np.sum(kdv[0:k])+np.finfo(float).eps)
    W = (P+P.T)/2
    return W



if __name__ == "__main__":
    x = np.arange(36).reshape(6,6,order='F')
    print(x)
    print('*********')
    print(d2ij2(x,1))
    print(asmk(x,k=4,sigma=1))
    
