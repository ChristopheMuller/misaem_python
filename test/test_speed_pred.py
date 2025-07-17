
# %%

import numpy as np
import pandas as pd
import os

if os.getcwd().endswith("test"):
    os.chdir(os.path.join(os.getcwd(), ".."))
    print("Changed directory to parent folder.")


# %%

path_data = os.path.join("logistic_with_NAs", "data", "MCAR_5d_0corr", "test_data", "MCAR_5d_0corr_rep0_n115000_d5_corr0_NA25.npz")
path_data_bayes = os.path.join("logistic_with_NAs", "data", "MCAR_5d_0corr", "bayes_data", "MCAR_5d_0corr_rep0_n115000_d5_corr0_NA25.npz")
data = np.load(path_data)
X = data["X_obs"]
y = data["y"]
data_bayes = np.load(path_data_bayes)
y_bayes = data_bayes["y_probs_bayes"]

n_test = 10000

X_test = X[:n_test, :]
y_test = y[:n_test]
y_test_bayes = y_bayes[:n_test]

n_train = 2000
X_train = X[n_test:n_test+n_train, :]
y_train = y[n_test:n_test+n_train]
y_train_bayes = y_bayes[n_test:n_test+n_train]


# %%

# from src.miss_glm import MissGLM
# from src.miss_glm_fast import MissGLM_fast as MissGLM
# from src.miss_glm_parallel import MissGLM_parallel as MissGLM
from src.miss_glm_parallel_fast import MissGLM_parallel_fast as MissGLM

# TRAIN
tic = pd.Timestamp.now()
miss_glm = MissGLM()
miss_glm.fit(X_train, y_train)
toc = pd.Timestamp.now()
print(f"Training time: {toc - tic}")

# PREDICT
tic = pd.Timestamp.now()
y_pred = miss_glm.predict_proba(X_test)[:,1]
toc = pd.Timestamp.now()
print(f"Prediction time: {toc - tic}")

# EVALUATE
mae = np.mean(np.abs(y_test_bayes - y_pred))
print(f"MAE: {mae}")


