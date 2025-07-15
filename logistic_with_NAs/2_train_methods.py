import numpy as np
import pandas as pd
import os
import time

from .methods import *

exp = "MCAR_5d_0corr"


methods_list = [
    pySAEM(name="py.SAEM"),
    pySAEM_fast(name="py.SAEM.fast"),
    pySAEM_parallel(name="py.SAEM.parallel")
]

training_size = np.array([100, 500, 1000, 5000, 10000])

df_set_up = pd.read_csv(os.path.join("logistic_with_NAs", "data",exp,"set_up.csv"))

def simulation_exists(set_up, method_name, n_train, simulations_df):
    """Check if simulation exists in the simulations.csv file"""
    if simulations_df.empty:
        return False
    
    mask = (simulations_df['set_up'] == set_up) & \
           (simulations_df['method'] == method_name) & \
           (simulations_df['n_train'] == n_train)
    
    return mask.any()

def prediction_file_exists(set_up, method_name, n_train, exp):
    """Check if prediction file exists in pred_data folder"""
    save_name = f"{set_up}_{method_name}_{n_train}"
    file_path = os.path.join("logistic_with_NAs", "data", exp, "pred_data", f"{save_name}.npz")
    return os.path.exists(file_path)

# Load or create simulations DataFrame
if os.path.exists(os.path.join("logistic_with_NAs", "data", exp, "simulation.csv")):
    simulations_df = pd.read_csv(os.path.join("logistic_with_NAs", "data", exp, "simulation.csv"))
else:
    simulations_df = pd.DataFrame({
        "set_up": [],
        "method": [],
        "n_train": [],
        "estimated_beta": [],
        "file_name": [],
        "running_time_train": [],
        "running_time_predict": []
    })


for i in range(df_set_up.shape[0]):

    print(f"Running set up {i+1} out of {df_set_up.shape[0]}: {df_set_up['set_up'][i]}")

    # load as npz
    data = np.load(os.path.join("logistic_with_NAs", "data", exp, "original_data", f"{df_set_up['set_up'][i]}.npz"))
    X_obs = data["X_obs"]
    M = data["M"]
    y = data["y"]
    y_probs = data["y_probs"]
    X_full = data["X_full"]

    data_test = np.load(os.path.join("logistic_with_NAs", "data", exp, "test_data", f"{df_set_up['set_up'][i]}.npz"))
    X_test = data_test["X_obs"]
    M_test = data_test["M"]
    y_probs_test = data_test["y_probs"]
    y_test = data_test["y"]

    for j in range(len(training_size)):

        print("\tTraining size: ", training_size[j])

        X_train = X_obs[:training_size[j]]
        M_train = M[:training_size[j]]
        y_train = y[:training_size[j]]

        for met in methods_list:
            # Check if simulation should be skipped
            if simulation_exists(df_set_up['set_up'][i], met.name, training_size[j], simulations_df) and \
               (not met.can_predict or prediction_file_exists(df_set_up['set_up'][i], met.name, training_size[j], exp)):
                print(f"\t\tSkipping {met.name} - already completed")
                continue
            
            tic = time.time()
            met.fit(X_train, M_train, y_train)
            toc = time.time()
            running_time_fit = toc - tic

            if met.can_predict:
                tic = time.time()
                y_probs_pred = met.predict_probs(X_test, M_test)
                toc = time.time()
                running_time_predict = toc - tic
                to_save = {
                    "y_probs_pred": y_probs_pred,
                }

                save_name = f"{df_set_up['set_up'][i]}_{met.name}_{training_size[j]}"
                np.savez(os.path.join("logistic_with_NAs", "data", exp, "pred_data", f"{save_name}.npz"), **to_save)

            else:
                save_name = np.nan

            if met.return_beta:
                estimated_beta = met.return_params()

            else:
                estimated_beta = None

            new_row_sim = {
                "set_up": df_set_up['set_up'][i],
                "method": met.name,
                "n_train": training_size[j],
                "estimated_beta": str(estimated_beta),
                "file_name": save_name,
                "running_time_train": running_time_fit,
                "running_time_predict": running_time_predict if met.can_predict else None
            }
            simulations_df = pd.concat([simulations_df, pd.DataFrame(new_row_sim, index=[0])])
            simulations_df.to_csv(os.path.join("logistic_with_NAs", "data", exp, "simulation.csv"), index=False)
