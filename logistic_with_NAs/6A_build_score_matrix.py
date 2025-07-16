#####
#
# This script builds a score matrix for the simulation study,
# instead of re-computing the scores for each plot.
#
# Determine:
# - If the score matrix already exists,
# - What scores to add,
# - What methods to add
#
#####


# %% load packages

import os
import numpy as np
import pandas as pd

name_score_matrix = "score_matrix.csv"

# %% set up

exp = "MCAR_5d_0corr"

all_methods_to_process = [
    "SAEM",
    "py.SAEM",
    "py.SAEM.fast",
    "py.SAEM.parallel"
]


# %% Load data

set_up_df = pd.read_csv(os.path.join("data", exp, "set_up.csv"))
simulation_df = pd.read_csv(os.path.join("data", exp, "simulation.csv"))
simulation_df = simulation_df[simulation_df["method"].isin(all_methods_to_process)]

simulation_results_df = pd.read_csv(os.path.join("data", exp, "simulation_set_up.csv"))
simulation_results_df = simulation_results_df[simulation_results_df["method"].isin(all_methods_to_process)]

path_score_matrix = os.path.join("data", exp, name_score_matrix)
if os.path.exists(path_score_matrix):
    original_score_matrix = pd.read_csv(path_score_matrix)
    print("Score matrix found, loading it.")
else:
    original_score_matrix = None
    print("No score matrix found, building a new one.")


# %% Define prediction metrics

prediction_metrics = {
    "brier": lambda y, pred_probs, bayes_probs: np.mean((y - pred_probs)**2),
    "misclassification": lambda y, pred_probs, bayes_probs: 1 - np.mean(y == (pred_probs >= 0.5)),
    "mae_bayes": lambda y, pred_probs, bayes_probs: np.mean(np.abs(bayes_probs - pred_probs))
}

estimation_metrics_names = [
    "angular_error",
    "mse_error",
    "angular_error_with_intercept", 
    "mse_error_with_intercept",
    "running_time_train",
    "running_time_pred"
]


# %% Function to build score matrix for prediction metrics

def calculate_prediction_scores(exp, simulation_runs_df, metrics_dict, existing_matrix=None):
    """
    Calculates prediction scores and appends to existing matrix if provided.
    """
    new_scores_list = []

    unique_runs = simulation_runs_df[['set_up', 'method', 'n_train']].drop_duplicates()

    for index, row in unique_runs.iterrows():
        setup = row["set_up"]
        method = row["method"]
        ntrain = np.round(row["n_train"], 0).astype(int)

        if existing_matrix is not None:

            existing_rows_count = existing_matrix[
                (existing_matrix["exp"] == exp) &
                (existing_matrix["set_up"] == setup) &
                (existing_matrix["method"] == method) &
                (existing_matrix["n_train"] == ntrain) &
                (existing_matrix["metric"].isin(list(metrics_dict.keys())))
            ].shape[0]
            
            # Since each metric produces two rows (bayes_adj=True/False), check for double the count
            if existing_rows_count >= (len(metrics_dict) * 2):
                print(f"Skipping existing prediction data for {setup} - {method} - {ntrain}")
                continue


        print(f"Processing prediction data for {setup} - {method} - {ntrain}")

        # Load data once per unique run
        try:
            true_y = np.load(os.path.join("data", exp, "test_data", f"{setup}.npz"))["y"]
            pred_probs = np.load(os.path.join("data", exp, "pred_data", f"{setup}_{method}_{ntrain}.npz"))["y_probs_pred"].ravel()
            bayes_probs = np.load(os.path.join("data", exp, "bayes_data", f"{setup}.npz"))["y_probs_bayes"]
        except FileNotFoundError as e:
            print(f"Warning: Missing data file for {setup}_{method}_{ntrain}. Skipping. Error: {e}")
            continue

        for metric_name, metric_func in metrics_dict.items():
            score = metric_func(true_y, pred_probs, bayes_probs)
            score_bayes = metric_func(true_y, bayes_probs, bayes_probs)
            score_bayes_adj = score - score_bayes

            new_scores_list.append({
                "exp": exp, "set_up": setup, "method": method, "n_train": ntrain,
                "bayes_adj": False, "metric": metric_name, "score": score, "filter": "all"
            })
            new_scores_list.append({
                "exp": exp, "set_up": setup, "method": method, "n_train": ntrain,
                "bayes_adj": True, "metric": metric_name, "score": score_bayes_adj, "filter": "all"
            })

    if new_scores_list:
        new_scores_df = pd.DataFrame(new_scores_list)
        if existing_matrix is not None:
            combined_df = pd.concat([existing_matrix, new_scores_df], ignore_index=True)
            return combined_df.drop_duplicates(subset=["exp", "set_up", "method", "n_train", "metric", "bayes_adj"])
        else:
            return new_scores_df
    else:
        return existing_matrix if existing_matrix is not None else pd.DataFrame(columns = ["exp", "set_up", "method", "n_train", "bayes_adj", "metric", "score", "filter"])


# %% Calculate prediction scores
score_matrix_pred = calculate_prediction_scores(exp, simulation_df, prediction_metrics, original_score_matrix)


# %% Function to add estimation metrics (angular, MSE, running_time)

def add_estimation_scores(score_matrix, results_df, metrics_to_add):
    """
    Adds estimation-based scores (angular error, MSE error, running time)
    to the score matrix.
    """
    new_estimation_scores_list = []

    # Iterate through each row of the simulation_results_df
    for index, row in results_df.iterrows():
        setup = row["set_up"]
        method = row["method"]
        ntrain = np.round(row["n_train"], 0).astype(int)

        is_already_present = (score_matrix["exp"] == exp) & \
                             (score_matrix["set_up"] == setup) & \
                             (score_matrix["method"] == method) & \
                             (score_matrix["n_train"] == ntrain) & \
                             (score_matrix["metric"].isin(metrics_to_add))
        
        if score_matrix[is_already_present].shape[0] >= len(metrics_to_add):
            continue
            
        for metric_name in metrics_to_add:
            # Ensure the metric exists in the results_df
            score = row[metric_name]
            if pd.isna(score):
                print(f"Skipping NaN score for {setup} - {method} - {ntrain} - {metric_name}")
                continue
            new_estimation_scores_list.append({
                "exp": exp, "set_up": setup, "method": method, "n_train": ntrain,
                "bayes_adj": False, "metric": metric_name, "score": score, "filter": "all"
            })

    if new_estimation_scores_list:
        new_estimation_df = pd.DataFrame(new_estimation_scores_list)

        final_score_matrix = pd.concat([score_matrix, new_estimation_df], ignore_index=True)
        return final_score_matrix.drop_duplicates(subset=["exp", "set_up", "method", "n_train", "metric", "bayes_adj"])
    else:
        return score_matrix


# %% Add estimation scores
final_score_matrix = add_estimation_scores(score_matrix_pred, simulation_results_df, estimation_metrics_names)


# %% Save the final score matrix

final_score_matrix.to_csv(path_score_matrix, index=False)
print(f"Score matrix saved to {path_score_matrix}")