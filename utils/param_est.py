import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

def multiplelinearreg(
    Y, # kxN
    X  #lxN
):
    N = Y.shape[1]
    sy = np.array([np.sum(Y, axis=1)])
    sx = np.array([np.sum(X, axis=1)])
    syx = np.sum(np.array([Y[:, i:i+1] @ X[:, i:i+1].T for i in range(N)]), axis=0)
    sxx = np.sum(np.array([X[:, i:i+1] @ X[:, i:i+1].T for i in range(N)]), axis=0)

    A = sx.T @ sx - N * sxx
    B = sy.T @ sx - N * syx

    Z = np.linalg.solve(A.T, B.T).T
    x0 = (sy.T - Z @ sx.T) / N
    eps = Y - x0 - Z@X
    Sigma = np.mean([eps[:, i:i+1] @ eps[:,  i:i+1].T for i in range(N)], axis=0)
    return x0, Z, Sigma, eps

def multiplelinearregnointercept(
    Y, 
    X
):
    N = Y.shape[1]
    sxx = np.sum(np.array([X[:, i:i+1] @ X[:, i:i+1].T for i in range(N)]), axis=0)
    syx = np.sum(np.array([Y[:, i:i+1] @ X[:, i:i+1].T for i in range(N)]), axis=0)
    Z = np.linalg.solve(sxx.T, syx.T).T
    eps = Y - Z@X
    Sigma = np.mean([eps[:, i:i+1] @ eps[:,  i:i+1].T for i in range(N)], axis=0)
    return Z, Sigma, eps
    

def covar(u, v):
    return np.mean([u[:, i:i+1] @ v[:, i:i+1].T for i in range(u.shape[1])], axis=0)


def param_estimation(
    X, 
    m
):
    """Estimate the parameters of a ECM(1) model
    
    Parameters
    ----------
    X : np.array
        logprices in (n_assets, n_timestamps) shape
    m : int
        number of common trends

    Returns
    -------
    (a0, A, delta, omega)
        params of the model
    """
    vecm = VECM(X.T, deterministic='co', coint_rank=m, k_ar_diff=0)
    res = vecm.fit()
    A = res.beta
    mu = res.const
    omega = res.sigma_u
    delta = res.alpha

    a0, _, _ = multiplelinearregnointercept((mu + np.array([np.diag(omega)]).T / 2.0).T, delta.T)
    return a0, A, delta, omega


