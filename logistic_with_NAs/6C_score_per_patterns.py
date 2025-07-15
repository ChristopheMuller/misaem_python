#####
#
# This script builds extends the matrix of scores for the simulation study,
# using a score per pattern instead of overall score.
#
#####

# %%

import os
import numpy as np
import pandas as pd
from utils import *

# %%

exp = "SimMCAR"
set_up = pd.read_csv(os.path.join("data", exp, "set_up.csv"))
simulation = pd.read_csv(os.path.join("data", exp, "simulation.csv"))

score_matrix = pd.read_csv(os.path.join("data", exp, "score_matrix.csv"))
# %%

metrics = {
    "mae_bayes": lambda y, pred_probs, bayes_probs: np.mean(np.abs(bayes_probs - pred_probs))
}

methods = [
"MICE.1.IMP","MICE.1.Y.IMP","MICE.1.M.IMP","MICE.1.Y.M.IMP",
"MICE.1.IMP.M","MICE.1.Y.IMP.M","MICE.1.M.IMP.M","MICE.1.Y.M.IMP.M",
"MICE.10.IMP","MICE.10.Y.IMP","MICE.10.M.IMP","MICE.10.Y.M.IMP",
"MICE.10.IMP.M","MICE.10.Y.IMP.M","MICE.10.M.IMP.M","MICE.10.Y.M.IMP.M",
"SAEM",
"Mean.IMP","Mean.IMP.M","05.IMP","05.IMP.M",
"PbP.Fixed",#"CC",
"MICE.RF.10.IMP","MICE.RF.10.Y.IMP","MICE.RF.10.M.IMP","MICE.RF.10.Y.M.IMP",
"MICE.RF.10.IMP.M","MICE.RF.10.Y.IMP.M","MICE.RF.10.M.IMP.M","MICE.RF.10.Y.M.IMP.M",
"MICE.100.IMP","MICE.100.Y.IMP","MICE.100.M.IMP","MICE.100.Y.M.IMP",
"MICE.100.IMP.M","MICE.100.Y.IMP.M","MICE.100.M.IMP.M","MICE.100.Y.M.IMP.M",
]

# patterns = [
#     0,
#     1,
#     2,
#     3,
#     4
# ]

patterns = [
    [1,0,0,0,0],
    [0,1,0,0,0],
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1],
]

# %%

def build_score_matrix_pattern(pattern, exp, set_up, simulation, metrics, methods, existing_matrix=None):

    n_set_ups = len(set_up)
    n_metrics = len(metrics)

    if existing_matrix is not None:
        score_matrix = existing_matrix.copy()
    else:  
        score_matrix = pd.DataFrame(
            columns = ["exp", "set_up", "method", "n_train", "bayes_adj", "metric", "score", "filter"]
        )

    simulation = simulation.copy()
    simulation = simulation[simulation["method"].isin(methods)]

    for i in range(n_set_ups):

        setup = set_up.iloc[i]["set_up"]
        print(f"Set up {i+1}/{n_set_ups} - {setup}")

        simulation_setup = simulation[simulation["set_up"] == setup]

        M = np.load(os.path.join("data", exp, "test_data", f"{setup}.npz"))["M"]
        idx = get_index_pattern(pattern, M)
        print(f"\tPattern {pattern}: {len(idx)}")

        true_y = np.load(os.path.join("data", exp, "test_data", f"{setup}.npz"))["y"][idx]
        bayes_probs = np.load(os.path.join("data", exp, "bayes_data", f"{setup}.npz"))["y_probs_bayes"][idx]
        
        for j in range(len(simulation_setup)):
            method = simulation_setup.iloc[j]["method"]
            ntrain = np.round(simulation_setup.iloc[j]["n_train"], 0).astype(int)

            ####

            pred_probs = np.load(os.path.join("data", exp, "pred_data", f"{setup}_{method}_{ntrain}.npz"))["y_probs_pred"].ravel()[idx]

            for k in range(n_metrics):

                metric = list(metrics.keys())[k]
                score = metrics[metric](true_y, pred_probs, bayes_probs)
                score_bayes = metrics[metric](true_y, bayes_probs, bayes_probs)
                score_bayes_adj = score - score_bayes

                score_matrix = pd.concat([
                    score_matrix,
                    pd.DataFrame({
                        "exp": [exp],
                        "set_up": [setup],
                        "method": [method],
                        "n_train": [ntrain],
                        "bayes_adj": False,
                        "metric": [metric],
                        "score": [score],
                        "filter": [pattern]
                    })
                ], ignore_index=True)

                score_matrix = pd.concat([
                    score_matrix,
                    pd.DataFrame({
                        "exp": [exp],
                        "set_up": [setup],
                        "method": [method],
                        "n_train": [ntrain],
                        "bayes_adj": True,
                        "metric": [metric],
                        "score": [score_bayes_adj],
                        "filter": [pattern]
                    })
                ], ignore_index=True)

    score_matrix = score_matrix.reset_index(drop=True)
    return score_matrix

def score_matrix_per_pattern(exp, set_up, simulation, metrics, methods, patterns, score_matrix):

    for pattern in patterns:
        print(f"Processing pattern {pattern}")
        score_matrix = build_score_matrix_pattern(pattern, exp, set_up, simulation, metrics, methods, score_matrix)
    
    return score_matrix

new_matrix = score_matrix_per_pattern( exp, set_up, simulation, metrics, methods, patterns, score_matrix)

# %%

new_matrix.to_csv(os.path.join("data", exp, "score_matrix.csv"), index=False)