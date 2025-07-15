

# %%

import numpy as np
import pandas as pd
import random 
import matplotlib.pyplot as plt

# %%

d = 6

def cov_toep(d, eta):

    """
    Generate a Toeplitz covariance matrix with a given dimension and eta.
    """
    cov = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            cov[i, j] = eta ** abs(i - j)
    return cov

COV = cov_toep(d, 0.5)

print("COV:\n", COV)

# %%

MU = np.random.normal(size=d)
print("MU:\n", MU)

# %%

X_i = np.random.multivariate_normal(MU, COV, size=1)
print("X_i:\n", X_i)

missing_idx = [0,1,3]
print("Missing indices:\n", missing_idx)
observed_idx = [i for i in range(d) if i not in missing_idx]

X_obs = X_i.copy()
for idx in missing_idx:
    X_obs[0, idx] = np.nan

print("X_obs:\n", X_obs)


# %%

# get conditional mean and covariance, in the TRUE way

mu_M = MU[missing_idx]
mu_O = MU[observed_idx]

cov_MO = COV[np.ix_(missing_idx, observed_idx)]
cov_OO = COV[np.ix_(observed_idx, observed_idx)]
cov_MM = COV[np.ix_(missing_idx, missing_idx)]

# conditional mean
mu_cond = mu_M + cov_MO @ np.linalg.inv(cov_OO) @ (X_obs[0, observed_idx] - mu_O)
print("Conditional mean (TRUE):\n", mu_cond)

# conditional covariance
cov_cond = cov_MM - cov_MO @ np.linalg.inv(cov_OO) @ cov_MO.T
print("Conditional covariance (TRUE):\n", cov_cond)


# %%

# get conditional mean and covariance, in the FAST way

sigma_inv = np.linalg.inv(COV)

Oi = np.linalg.inv(COV[np.ix_(missing_idx, missing_idx)])
mi = mu_M

# mu_cond = mi - (xi[obs_idx] - mu[obs_idx]) @ sigma_inv[np.ix_(obs_idx, missing_idx)] @ Oi

mu_cond_fast = mu_M - (X_obs[0, observed_idx] - mu_O) @ sigma_inv[np.ix_(observed_idx, missing_idx)] @ Oi
print("Conditional mean (FAST):\n", mu_cond_fast)

conv_cond_fast = Oi
print("Conditional covariance (FAST):\n", conv_cond_fast)

# %% 

# compare the two methods

print("Difference in conditional means:\n", mu_cond - mu_cond_fast)
print("\tmu TRUE: ", mu_cond)
print("\tmu FAST: ", mu_cond_fast)

print("Difference in conditional covariances:\n", cov_cond - conv_cond_fast)
print("\tCov TRUE:\n", cov_cond)
print("\tCov FAST:\n", conv_cond_fast)