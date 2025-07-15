###
# Investigate if / how many simulations failed
###

# %%

import os
if os.getcwd().endswith("plots_scripts"):
    os.chdir(os.path.join(os.getcwd(), ".."))


# %% load packages

import numpy as np
import pandas as pd

exp = "SimMCAR"

simulation_df = pd.read_csv(os.path.join("data", exp, "simulation.csv"))

temp = simulation_df.groupby(["method", "n_train"]).agg({"file_name": "count"})

# what methods did not run for all n_train?

temp = temp.reset_index()
temp.sort_values(by=["file_name"], inplace=True)
temp[:10]