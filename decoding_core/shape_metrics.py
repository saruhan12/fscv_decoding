import numpy as np
from scipy.stats import wasserstein_distance_nd
from scipy.spatial import procrustes
import ot


w = ot.unif(1000)

def w2_matrix(A):
    N = len(A)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            D[i, j] = D[j, i] = ot.emd2(w, w, ot.dist(A[i], A[j]), numItermax=1_000_000)

    return D

def std_global(A):
    #normalization
    f = A.reshape(-1, A.shape[-1])
    return (A - f.mean(0)) / np.maximum(f.std(0), 1e-3)

def procrustes_matrix(A):
    N = len(A)
    D = np.zeros((N,N))
    for i in range(N):
            for j in range(i + 1, N):
                _, _, dist = procrustes(A[i], A[j])
                D[i, j] = D[j, i] = dist
   

    return D
