#### Singular Value Decomposition ####

import numpy as np
from gram_schmidt import gram_schmidt
from eigen_value_decomposition import eigen_value_decomposition


# for scalar
def is_zero(x):
    return (np.abs(x) < 1e-8)


# for vector 
def is_zero_vector(x):
    return (np.linalg.norm(x) < 1e-8)

    
# SVD
def singular_value_decomposition(A):
    
    m, n = A.shape
    flag = (m < n)
    if (flag):
        A = A.T
        m, n = n, m
        
    eigen_vals, eigen_vecs = eigen_value_decomposition(np.dot(A.T, A))
    rank = np.argsort(eigen_vals)[::-1]
    Lambda = eigen_vals[rank]
    V = eigen_vecs[:, rank]
    
    k = 0
    while (k < n and not(is_zero(Lambda[k]))): k += 1
    
    V_k = V[:, :k]
    U_k = np.dot(np.dot(A, V_k), np.diag(Lambda[:k]))
    U = gram_schmidt(U_k)
    Sigma = np.dot(U.T, np.dot(A, V))
    
    if (flag):
        U, V = V, U
        Sigma = Sigma.T
    
    return U, Sigma, V