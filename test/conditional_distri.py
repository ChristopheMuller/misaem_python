
#%%

import numpy as np

#%%

def toep_matrix(d, corr):
    """
    Generate a Toeplitz matrix with correlation corr.
    """
    return np.array([[corr**abs(i-j) for j in range(d)] for i in range(d)])


SIGMA = toep_matrix(6, 0.45)
MU = np.array([0,1,2,3,4,5])

missing_idx = [0,1,3]
observed_idx = [2,4,5]

X_true = np.random.multivariate_normal(MU, SIGMA, size=1)[0]
X_obs = X_true.copy()
X_obs[missing_idx] = np.nan


print("SIGMA: \n", SIGMA)
print("MU: \n", MU)
print("X_true: \n", X_true)
print("X_obs: \n", X_obs)


#%%

# True conditional distribution

MU_M = MU[missing_idx]
MU_O = MU[observed_idx]

SIGMA_MO = SIGMA[np.ix_(missing_idx, observed_idx)]
SIGMA_OO = SIGMA[np.ix_(observed_idx, observed_idx)]
SIGMA_MM = SIGMA[np.ix_(missing_idx, missing_idx)]


MU_COND = MU_M + SIGMA_MO @ np.linalg.inv(SIGMA_OO) @ (X_obs[observed_idx] - MU_O)
SIGMA_COND = SIGMA_MM - SIGMA_MO @ np.linalg.inv(SIGMA_OO) @ SIGMA_MO.T

print("MU_COND: \n", np.round(MU_COND,3))
print("SIGMA_COND: \n", np.round(SIGMA_COND,3))

#%%

# Conditional distribution from the SAEM

S_inv = np.linalg.inv(SIGMA)

Oi = np.linalg.inv(S_inv[np.ix_(missing_idx, missing_idx)])

mi = MU[missing_idx]

COND_MU = mi - (X_obs[observed_idx] - MU[observed_idx]) @ S_inv[np.ix_(observed_idx, missing_idx)] @ Oi
COND_SIGMA = Oi

print("COND_MU: \n", np.round(COND_MU,3))
print("COND_SIGMA: \n", np.round(COND_SIGMA,3))