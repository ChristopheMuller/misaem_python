
# %%

import numpy as np
import pandas as pd
import random 
import matplotlib.pyplot as plt

import os
if os.getcwd().endswith("test"):
    os.chdir(os.path.join(os.getcwd(), ".."))

from src.miss_glm import MissGLM

# %%


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

# %%

n = 300
d = 3
corr = 0.54

beta = np.random.normal(size=d + 1)

X = generate_X(n, d, corr)
M = generate_M(n, d, 0.2)

X_obs = X.copy()
X_obs[M == 1] = np.nan

Y_logits = X @ beta[1:] + beta[0]
Y_probs = 1 / (1 + np.exp(-Y_logits))
Y = np.random.binomial(n=1, p=Y_probs)

# %%

miss_glm = MissGLM(ll_obs_cal=False,
                   nmcmc=2,
                   seed=1234)
miss_glm.fit(X_obs, Y,
             save_trace=True)

# %%


print(miss_glm.mu_ - np.zeros(d))
print(miss_glm.sigma_ - toep_matrix(d, corr))

print(miss_glm.coef_ - beta)

coefs = miss_glm.coef_
std_err = miss_glm.std_err

plt.errorbar(range(len(coefs)), coefs, yerr=std_err, fmt='o', capsize=5)
plt.xticks(range(len(coefs)), [f'X{i+1}' for i in
range(len(coefs))], rotation=45)
plt.xlabel('Variables')
plt.ylabel('Coefficient')
plt.title('Coefficients with Standard Errors')
plt.scatter(range(len(coefs)), beta, color='red', label='True Coefficients', marker='x')

