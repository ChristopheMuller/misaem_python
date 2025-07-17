
# %%
import numpy as np
import pandas as pd
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.special import expit
from scipy.optimize import curve_fit, minimize_scalar
import matplotlib.pyplot as plt


def sigma(x):
    return 1 / (1 + np.exp(-x))

def sigma_inv(x):
    return np.log(x / (1 - x))

# %%

np.random.seed(0)

X1s = np.random.normal(1.5, 5, 1000)
X1s = np.sort(X1s)


# %% NORMAL: search for worse scale

# def compute_mse_for_scale(scale, n=10000):
#     mean_probs_list = []

#     for mu in X1s:
#         var = np.random.normal(0, scale, n)
#         logits = mu + var
#         probs = expit(logits)
#         mean_probs_list.append(np.mean(probs))

#     mean_probs_array = np.array(mean_probs_list)

#     # Fit sigmoid with both slope and intercept
#     def scaled_sigmoid(x1, a, b):
#         return expit(a * x1 + b)

#     popt, _ = curve_fit(scaled_sigmoid, X1s, mean_probs_array, p0=[1, 0])
#     a_opt, b_opt = popt

#     predicted_probs = expit(a_opt * X1s + b_opt)
#     mse = np.mean((mean_probs_array - predicted_probs) ** 2)
#     return mse * 100000

# # print(compute_mse_for_scale(1.5)) # test

# result = minimize_scalar(
#     lambda s: -compute_mse_for_scale(s, n=25000),
#     bounds=(0.01, 5),  # or wider/narrower depending on your use case
#     method='bounded'
# )

# best_scale = result.x
# max_mse = -result.fun

# print(f"Best scale (maximizing MSE): {best_scale:.4f}")
# print(f"Maximum MSE: {max_mse:.6f}")


# %% NORMAL: run the simulation of bayes logits

scale = 3.8312 # from optimization, = best_scale

results = pd.DataFrame(columns=["mu", "mean_probs", "sigma_mean"], index=X1s)

for mu in X1s:

    var = np.random.normal(0,scale,200000)

    logits = mu + np.array(var).reshape(-1,1)
    probs = sigma(logits)

    mean_probs = np.mean(probs)

    sigma_mean = sigma(mu)

    results.loc[mu] = [mu, mean_probs, sigma_mean]

results_NORMAL = results.copy()

# %% EXPONENTIAL: search for worse scale

# def compute_mse_for_scale(scale, n=10000):
#     mean_probs_list = []

#     for mu in X1s:
#         var = np.random.exponential(scale, n) - scale
#         logits = mu + var
#         probs = expit(logits)
#         mean_probs_list.append(np.mean(probs))

#     mean_probs_array = np.array(mean_probs_list)

#     # Fit sigmoid with both slope and intercept
#     def scaled_sigmoid(mu, a, b):
#         return expit(a * mu + b)

#     popt, _ = curve_fit(scaled_sigmoid, X1s, mean_probs_array, p0=[1, 0])
#     a_opt, b_opt = popt

#     predicted_probs = expit(a_opt * X1s + b_opt)
#     mse = np.mean((mean_probs_array - predicted_probs) ** 2)
#     return mse * 100000

# # compute_mse_for_scale(1.5) #test

# result = minimize_scalar(
#     lambda s: -compute_mse_for_scale(s, n=25000),
#     bounds=(0.01, 15),
#     method='bounded'
# )

# best_scale = result.x
# max_mse = -result.fun

# print(f"Best scale (maximizing MSE): {best_scale:.4f}")
# print(f"Maximum MSE: {max_mse:.6f}")


# %% EXPONENTIAL: run the simulation of bayes logits

scale = 7.6275 # from optimization, = best_scale

results = pd.DataFrame(columns=["mu", "mean_probs", "sigma_mean"], index=X1s)

for mu in X1s:

    draws = np.random.exponential(scale, 100000) - scale 
    var = draws - np.mean(draws)

    logits = mu + np.array(var).reshape(-1,1)
    probs = sigma(logits)

    mean_probs = np.mean(probs)

    sigma_mean = sigma(mu)

    results.loc[mu] = [mu, mean_probs, sigma_mean]

results_EXPONENTIAL = results.copy()

# %% Plot 1: In logits space

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# First plot: Normal

results_NORMAL["logits_mean_probs"] = np.log(results_NORMAL["mean_probs"].astype(float) / (1 - results_NORMAL["mean_probs"].astype(float)))

lr = LinearRegression()
lr.fit(results_NORMAL["mu"].values.reshape(-1, 1), results_NORMAL["logits_mean_probs"])
preds_probs_NORMAL = lr.predict(results_NORMAL["mu"].values.reshape(-1, 1))

axes[0].plot(results_NORMAL["mu"], results_NORMAL["logits_mean_probs"], label=r"$\sigma^{-1}(E[\sigma(x_1+X_2)])$", color="#FFC482")
axes[0].plot(results["mu"], preds_probs_NORMAL, color="#FFC482", linestyle="--", alpha=0.5)
axes[0].legend()
axes[0].set_title("Normal Covariate")

# Second plot: Exponential

results_EXPONENTIAL["logits_mean_probs"] = np.log(results_EXPONENTIAL["mean_probs"].astype(float) / (1 - results_EXPONENTIAL["mean_probs"].astype(float)))

lr = LinearRegression()
lr.fit(results_EXPONENTIAL["mu"].values.reshape(-1, 1), results_EXPONENTIAL["logits_mean_probs"])
preds_probs_EXPONENTIAL = lr.predict(results_EXPONENTIAL["mu"].values.reshape(-1, 1))

axes[1].plot(results_EXPONENTIAL["mu"], results_EXPONENTIAL["logits_mean_probs"], label=r"$\sigma^{-1}(E[\sigma(x_1+X_2)])$", color="#FFC482")
axes[1].plot(results["mu"], preds_probs_EXPONENTIAL, color="#FFC482", linestyle="--", alpha=0.5)
axes[1].legend()
axes[1].set_title("Exponential Covariate")

plt.tight_layout()
plt.savefig("plots_scripts/plots/logits.pdf")
plt.show()

# %% Second plot: In probability space

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# First plot: Normal

def scaled_sigmoid(mu, a, b):
    return expit(a * mu + b)

popt, _ = curve_fit(scaled_sigmoid, results_NORMAL["mu"].astype(float), results_NORMAL["mean_probs"].astype(float), p0=[1,0])
a_opt, b_opt = popt[0], popt[1]
probs_LR_NORMAL = expit(a_opt * results_NORMAL["mu"].astype(float) + b_opt)


axes[0].plot(results_NORMAL["mu"], results_NORMAL["mean_probs"], label=r"$E[\sigma(x_1+X_2)]$", color="#AF7595")
axes[0].plot(results_NORMAL["mu"], results_NORMAL["sigma_mean"], label=r"$\sigma(E[x_1+X_2])$", color="#FFC482")
axes[0].plot(results_NORMAL["mu"], probs_LR_NORMAL, label=r"Closest $\sigma(\alpha + \beta . x_1)$", linestyle="--", color="blue")
axes[0].set_ylim(-0.1, 1.1)
axes[0].set_xlim(1.5-13,1.5+13)
axes[0].hlines(0, 1.5-13,1.5+13, color='gray', linestyle='--')
axes[0].hlines(1, 1.5-13,1.5+13, color='gray', linestyle='--')
axes[0].legend()
axes[0].set_title("Normal Covariate")

# Second plot: Exponential
popt, _ = curve_fit(scaled_sigmoid, results_EXPONENTIAL["mu"].astype(float), results_EXPONENTIAL["mean_probs"].astype(float), p0=[1,0])
a_opt, b_opt = popt[0], popt[1]
probs_LR_EXPONENTIAL = expit(a_opt * results_EXPONENTIAL["mu"].astype(float) + b_opt)

axes[1].plot(results_EXPONENTIAL["mu"], results_EXPONENTIAL["mean_probs"], label=r"$E[\sigma(x_1+X_2)]$", color="#AF7595")
axes[1].plot(results_EXPONENTIAL["mu"], results_EXPONENTIAL["sigma_mean"], label=r"$\sigma(E[x_1+X_2])$", color="#FFC482")
axes[1].plot(results_EXPONENTIAL["mu"], probs_LR_EXPONENTIAL, label=r"Closest $\sigma(\alpha + \beta . x_1)$", linestyle="--", color="blue")
axes[1].set_ylim(-0.1, 1.1)
axes[1].set_xlim(1.5-13,1.5+13)
axes[1].hlines(0, 1.5-13,1.5+13, color='gray', linestyle='--')
axes[1].hlines(1, 1.5-13,1.5+13, color='gray', linestyle='--')
axes[1].legend()
axes[1].set_title("Exponential Covariate")

plt.tight_layout()
plt.savefig("plots_scripts/plots/probs_bayes.pdf")
plt.show()

# %%
