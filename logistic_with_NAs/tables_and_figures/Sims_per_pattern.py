#####
#
# Plot results per pattern (1 metrics)
#
#####

# %%

# if current working directory is "/plots_scripts", change it to the parent directory
import os
if os.getcwd().endswith("plots_scripts"):
    os.chdir(os.path.join(os.getcwd(), ".."))

# %% load packages

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from setups_design import metrics_config, methods_config

# %% set up

exp = "SimMCAR"
score_matrix = pd.read_csv(os.path.join("data", exp, "score_matrix.csv"))
score_matrix = score_matrix[score_matrix["exp"] == exp]


metrics_config


# %%

metric_sel = "mae_bayes"
patterns_sel = [[0,0,1,0],
    [0,1,1,0],
    [1,0,0,0],
    [1,1,0,0]]
patterns_sel = [str(pattern) for pattern in patterns_sel]

pattern_names = patterns_sel.copy()

methods_sel = [
# "MICE.1.IMP","MICE.1.Y.IMP",
# "MICE.1.M.IMP","MICE.1.Y.M.IMP",
# "MICE.1.IMP.M","MICE.1.Y.IMP.M",
# "MICE.1.M.IMP.M","MICE.1.Y.M.IMP.M",
# "MICE.10.IMP","MICE.10.Y.IMP",
# "MICE.10.M.IMP","MICE.10.Y.M.IMP",
# "MICE.10.IMP.M","MICE.10.Y.IMP.M",
# "MICE.10.M.IMP.M","MICE.10.Y.M.IMP.M",
# "MICE.100.IMP","MICE.100.Y.IMP",
# "MICE.100.M.IMP","MICE.100.Y.M.IMP",
# "MICE.100.IMP.M","MICE.100.Y.IMP.M",
# "MICE.100.M.IMP.M","MICE.100.Y.M.IMP.M",
# "SAEM",
"Mean.IMP",
"Mean.IMP.M",
# "05.IMP",
# "05.IMP.M",
# "PbP","CC",
"PbP.Fixed",
# "MICE.RF.10.IMP","MICE.RF.10.Y.IMP","MICE.RF.10.M.IMP","MICE.RF.10.Y.M.IMP",
# "MICE.RF.10.IMP.M","MICE.RF.10.Y.IMP.M","MICE.RF.10.M.IMP.M","MICE.RF.10.Y.M.IMP.M"
]


score_matrix_sel = score_matrix[score_matrix["metric"] == metric_sel]
score_matrix_sel = score_matrix_sel[score_matrix_sel["method"].isin(methods_sel)]
score_matrix_sel = score_matrix_sel[score_matrix_sel["exp"] == exp]

score_matrix_sel = score_matrix_sel[score_matrix_sel["filter"].isin(patterns_sel)]

score_matrix_sel["score"] = score_matrix_sel["score"].astype(float)
score_matrix_sel = score_matrix_sel.groupby(["filter", "method", "n_train"]).agg({"score": ["mean", "std", "count"]}).reset_index()
score_matrix_sel.columns = ["filter", "method", "n_train", "mean", "sd", "count"]

score_matrix_sel["se"] = score_matrix_sel["sd"] / np.sqrt(score_matrix_sel["count"])
fig, axes = plt.subplots(1, len(patterns_sel), figsize=(4 * len(patterns_sel), 5))

for i, pattern in enumerate(patterns_sel):

    print(i, pattern)

    # filter the score
    score_matrix_pattern = score_matrix_sel[score_matrix_sel["filter"] == pattern]

    # plot the mean and se
    for method in methods_sel:
        method_config = methods_config[method]

        score_matrix_method = score_matrix_pattern[score_matrix_pattern["method"] == method]
        axes[i].plot(score_matrix_method["n_train"], score_matrix_method["mean"], label=method_config["label"], 
                     color=method_config["color"], linestyle=method_config["linestyle"],
                     marker=method_config["marker"], markersize=5)
        axes[i].fill_between(score_matrix_method["n_train"], score_matrix_method["mean"] - score_matrix_method["se"],
                              score_matrix_method["mean"] + score_matrix_method["se"], alpha=0.2, 
                              color=method_config["color"], linestyle=method_config["linestyle"])
    
    axes[i].set_xscale("log")
    axes[i].set_xlabel("Number of training samples")
    axes[i].set_ylabel(metrics_config[metric_sel]["label"])
    axes[i].set_title(f"{pattern_names[i]}")   

    if i == 0:
        axes[i].legend()
    # axes[i].grid()

    axes[i].set_ylim(-0.01, 0.2)
    # line at
    axes[i].axhline(0, color="black", linestyle="--", linewidth=0.5)

plt.tight_layout()
# plt.savefig(os.path.join("plots_scripts", exp, f"perPattern_{metric_sel}_{selection_name}.pdf"))
plt.show()
