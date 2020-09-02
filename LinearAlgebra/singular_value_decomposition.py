#### Singular Value Decomposition ####

import numpy as np
from jacobi import jacobi_method
from gram_schmidt import gram_schmidt


# for scalar
def is_zero(x):
    return (np.abs(x) < 1e-8)


# for vector 
def is_zero_vector(x):
    return (np.linalg.norm(x) < 1e-8)

    
# SVD
def singular_value_decomposition(A):
    
    m, n = A.shape
    if (m < n):
        A = A.T
        m, n = n, m
        
    eigen_vals, eigen_vecs = jacobi_method(np.dot(A.T, A))
    rank = np.argsort(eigen_vals)[::-1]
    Lambda = eigen_vals[rank]
    V = eigen_vecs[:, rank]
    
    k = 0
    while (k < n and not(is_zero(Lambda[k]))): k += 1
    
    V_k = V[:, :k]
    U_k = np.dot(np.dot(A, V_k), np.diag(Lambda[:k]))
    U = gram_schmidt(U_k)
    Sigma = np.dot(U.T, np.dot(A, V))
    
    if (m < n):
        U = V.T
        Sigma = Sigma.T
        V = U.T
    
    return U, Sigma, V