
import numpy as np

def generate_M(n, d, prc):
    """
    Generate a missing data matrix M with n rows and d columns, with a proportion of missing data prop_NA.
    It guarantees no row with all missing data.
    """
    M = np.random.binomial(n=1, p=prc, size=(n, d))

    all_ones = np.all(M == 1, axis=1)

    while np.any(all_ones):
        M[all_ones] = np.random.binomial(n=1, p=prc, size=(all_ones.sum(), d))
        all_ones = np.all(M == 1, axis=1)  # Recheck after redrawing

    return M


def toep_matrix(d, corr):
    """
    Generate a Toeplitz matrix with correlation corr.
    """
    return np.array([[corr**abs(i-j) for j in range(d)] for i in range(d)])

def generate_X(n, d, corr, mu=None):
    """
    Generate a design matrix X with n rows and d columns, with a correlation of corr.
    """

    if mu is None:
        mu = np.zeros(d)

    cov = toep_matrix(d, corr)
    
    X = np.random.multivariate_normal(mu, cov, size=n)
    
    return X