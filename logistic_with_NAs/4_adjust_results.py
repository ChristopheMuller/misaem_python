
# %%

import numpy as np
import pandas as pd
import os

# %%

exp = "SimMCAR"

df_set_up = pd.read_csv(os.path.join("data",exp,"set_up.csv"))
df_simulations = pd.read_csv(os.path.join("data",exp,"simulation.csv"))


# %%

# some cols of set_up not in simulations
if not np.all([df_set_up.columns[i] in df_simulations.columns for i in range(len(df_set_up.columns))]):
    df_simulations_enlarged = pd.merge(left=df_simulations, right=df_set_up, on="set_up")

# %%

import ast
import ast

def deal_with_estimated_beta(beta_str, method, d=5):
    if not isinstance(beta_str, str):
        return None

    # Replace 'NA' with 'None' for proper parsing
    processed_beta_str = beta_str.replace("NA", "None")
    beta_list_of_lists = ast.literal_eval(processed_beta_str)


    # Ensure the input structure is as expected: a list of two sublists
    if not (isinstance(beta_list_of_lists, list) and len(beta_list_of_lists) == 2 and
            isinstance(beta_list_of_lists[0], list) and
            isinstance(beta_list_of_lists[1], list) and len(beta_list_of_lists[1]) == 1):
        raise ValueError("Input structure is not [[beta_values], [intercept_value]].")

    beta_values = beta_list_of_lists[0]

    if len(beta_values) == d:
        return beta_values
    elif len(beta_values) == 2 * d:
        return beta_values[:d]
    else:
        raise ValueError(f"The length of the beta_values list is not {d} or {2*d} -- Method {method}")    
def deal_with_estimated_intercept(beta_str):
    if not isinstance(beta_str, str):
        return None

    # Replace 'NA' with 'None' for proper parsing by ast.literal_eval
    processed_beta_str = beta_str.replace("NA", "None")

    beta_list = ast.literal_eval(processed_beta_str)

    # Ensure the input structure is as expected: a list of two sublists
    # and the second sublist (intercept) is a single-element list.
    if not (isinstance(beta_list, list) and len(beta_list) == 2 and
            isinstance(beta_list[1], list) and len(beta_list[1]) == 1):
        raise ValueError("Input structure for intercept extraction is not [[...], [intercept_value]].")

    intercept_value = beta_list[1][0]

    # If the intercept value was 'NA' and got converted to None, return None
    if intercept_value is None:
        return None
    else:
        return intercept_value


df_simulations_enlarged["pred_beta"] = df_simulations_enlarged.apply(lambda x: deal_with_estimated_beta(x["estimated_beta"], x["method"], d=np.round(x["d"]).astype(int)), axis=1)
df_simulations_enlarged["pred_intercept"] = df_simulations_enlarged.apply(lambda x: deal_with_estimated_intercept(x["estimated_beta"]), axis=1)
df_simulations_enlarged.to_csv(os.path.join("data",exp,"simulation_set_up.csv"), index=False)

# %%

# fix your results if necessary

def filter_simulations(method, n_train, exp):

    df_simulations_temp = pd.read_csv(os.path.join("data",exp,"simulation.csv"))
    df_simulations_copy = df_simulations_temp.copy()
    all_n_train = df_simulations_temp["n_train"].unique()
    if n_train is None:
        n_train = all_n_train

    df_simulations_temp = df_simulations_temp[df_simulations_temp["n_train"].isin(n_train)]
    df_simulations_temp = df_simulations_temp[df_simulations_temp["method"].isin(method)]

    for i, row in df_simulations_temp.iterrows():
        file_name = row["file_name"]

        # delete the file
        if os.path.exists(os.path.join("data",exp,"pred_data", f"{file_name}.npz")):
            os.remove(os.path.join("data",exp,"pred_data", f"{file_name}.npz"))

        # delete row in the dataframe
        df_simulations_copy = df_simulations_copy.drop(i)

        print(f"{i}..Deleted {file_name}")

    df_simulations_copy.to_csv(os.path.join("data",exp,"simulation.csv"), index=False)

import os
import pandas as pd

# make a copy of all files with characterisitcs
method = ["MICE.100.Y.IMP"]
n_train = None

exp = "ExpA"
folder_goal = os.path.join("data",exp,"temp2")

if not os.path.exists(folder_goal):
    os.makedirs(folder_goal)

def copy_files(method, n_train, exp, folder_goal):

    df_simulations_temp = pd.read_csv(os.path.join("data",exp,"simulation.csv"))
    all_n_train = df_simulations_temp["n_train"].unique()
    if n_train is None:
        n_train = all_n_train

    df_simulations_temp = df_simulations_temp[df_simulations_temp["n_train"].isin(n_train)]
    df_simulations_temp = df_simulations_temp[df_simulations_temp["method"].isin(method)]

    for i, row in df_simulations_temp.iterrows():
        file_name = row["file_name"]

        # copy the file
        if os.path.exists(os.path.join("data",exp,"temp", f"{file_name}.npz")):
            os.system(f"cp {os.path.join('data',exp,'temp', f'{file_name}.npz')} {os.path.join(folder_goal, f'{file_name}.npz')}")

        print(f"{i}..Copied {file_name}")

