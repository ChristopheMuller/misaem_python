#####
#
# Plot main results (4 metrics)
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

from utils import calculate_ymin_for_R_proportion
score_matrix = score_matrix[score_matrix["filter"] == "all"]

# methods_sel = [
# "MICE.1.IMP","MICE.1.Y.IMP","MICE.1.M.IMP","MICE.1.Y.M.IMP",
# "MICE.1.IMP.M","MICE.1.Y.IMP.M","MICE.1.M.IMP.M","MICE.1.Y.M.IMP.M",
# ]
# selection_name = "MICE1_with_or_without_M"

methods_sel = [
"MICE.10.IMP","MICE.10.Y.IMP","MICE.10.M.IMP","MICE.10.Y.M.IMP",
"MICE.10.IMP.M",
"MICE.10.Y.IMP.M",
"MICE.10.M.IMP.M",
"MICE.10.Y.M.IMP.M",
]
selection_name = "MICE10_with_or_without_M"


# methods_sel = [
# "MICE.RF.10.IMP","MICE.RF.10.Y.IMP","MICE.RF.10.M.IMP","MICE.RF.10.Y.M.IMP",
# "MICE.RF.10.IMP.M",
# "MICE.RF.10.Y.IMP.M",
# "MICE.RF.10.M.IMP.M",
# "MICE.RF.10.Y.M.IMP.M",
# ]
# selection_name = "MICE10RF_with_or_without_M"


# methods_sel = [
# "MICE.10.IMP","MICE.10.Y.IMP","MICE.10.M.IMP","MICE.10.Y.M.IMP",
# "MICE.RF.10.IMP","MICE.RF.10.Y.IMP","MICE.RF.10.M.IMP","MICE.RF.10.Y.M.IMP",
# ]
# selection_name = "MICE10_with_RF"



# methods_sel = [
# "MICE.1.IMP","MICE.1.Y.IMP","MICE.1.M.IMP","MICE.1.Y.M.IMP",
# "Mean.IMP","Mean.IMP.M","05.IMP","05.IMP.M",
# ]
# selection_name = "Single_Imputation"


methods_sel = [
        "CC",
        "PbP.Fixed",
        "Mean.IMP.M",
        "SAEM",
        "MICE.100.Y.IMP",
        "MICE.100.IMP",
        "MICE.RF.10.Y.IMP",
        "MICE.RF.10.IMP"
        # "MICE.100.Y.M.IMP.M",
        # "MICE.100.Y.IMP.M",
        # "MICE.RF.10.Y.M.IMP.M",
        # "MICE.RF.10.Y.IMP.M"
    ]
selection_name = "Selected_Procedures"


# methods_sel = [
# "MICE.10.IMP","MICE.10.Y.IMP",
# "MICE.100.IMP","MICE.100.Y.IMP",
# "MICE.RF.10.IMP","MICE.RF.10.Y.IMP",
# "SAEM",
# "Mean.IMP.M",
# "PbP.Fixed", "CC",
# ]
# selection_name = "RF_selected"

# methods_sel = [
# "MICE.RF.10.IMP","MICE.RF.10.Y.IMP","MICE.RF.10.M.IMP","MICE.RF.10.Y.M.IMP",
# "MICE.RF.10.IMP.M","MICE.RF.10.Y.IMP.M","MICE.RF.10.M.IMP.M","MICE.RF.10.Y.M.IMP.M"
# ]
# selection_name = "RF_all"

# methods_sel = [
# "MICE.1.IMP","MICE.1.Y.IMP","MICE.1.M.IMP","MICE.1.Y.M.IMP",
# "MICE.10.IMP","MICE.10.Y.IMP","MICE.10.M.IMP","MICE.10.Y.M.IMP",
# "MICE.100.IMP","MICE.100.Y.IMP","MICE.100.M.IMP","MICE.100.Y.M.IMP",
# ]
# selection_name = "MICE_multiple"

# methods_sel = [
#     "PbP.Fixed", "PbP.Fixed.MinObs"
# ]
# selection_name = "PbP.Fixed_selected"

# scores_sel = ["misclassification", "calibration", "mse_error", "mae_bayes"]
# metrics_name = "4_metrics"
# filter_bayes = [True, True, False, False]
# ylimsmax = [0.07, 0.01, 0.85, 0.16]

scores_sel = ["brier", "angular_error"]
metrics_name = "2_other_metrics"
filter_bayes = [True, False]
ylimsmax = [0.05, 0.5]

ntrains = [100, 500, 1000, 5000, 10000, 50000]



ylimsmin = calculate_ymin_for_R_proportion(0.03, ylimsmax)
ylims = [(ylimsmin[i], ylimsmax[i]) for i in range(len(ylimsmax))] 

fig, axes = plt.subplots(1, len(scores_sel), figsize=(4 * len(scores_sel), 5)) # default is 4 * len, 5
if len(scores_sel) == 1:
    axes = [axes]

for i, score in enumerate(scores_sel):

    print(i, score)

    # filter the score
    score_matrix_sel = score_matrix[score_matrix["metric"] == score]
    score_matrix_sel = score_matrix_sel[score_matrix_sel["method"].isin(methods_sel)]
    score_matrix_sel = score_matrix_sel[score_matrix_sel["exp"] == exp]
    score_matrix_sel = score_matrix_sel[score_matrix_sel["n_train"].isin(ntrains)]
    score_matrix_sel = score_matrix_sel[score_matrix_sel["bayes_adj"] == filter_bayes[i]]

    # group by method and n_train
    score_matrix_sel["score"] = score_matrix_sel["score"].replace("[nan nan nan nan nan]", np.nan)
    score_matrix_sel["score"] = score_matrix_sel["score"].astype(float)
    score_matrix_sel = score_matrix_sel.groupby(["method", "n_train"]).agg({"score": ["mean", "std", "count"]}).reset_index()
    score_matrix_sel.columns = ["method", "n_train", "mean", "sd", "count"]
    score_matrix_sel["se"] = score_matrix_sel["sd"] / np.sqrt(score_matrix_sel["count"])

    # plot the mean and se
    for method in methods_sel:
        method_config = methods_config[method]

        score_matrix_method = score_matrix_sel[score_matrix_sel["method"] == method]
        axes[i].plot(score_matrix_method["n_train"], score_matrix_method["mean"], label=method_config["label"], 
                     color=method_config["color"], linestyle=method_config["linestyle"],
                     marker=method_config["marker"], markersize=5)
        axes[i].fill_between(score_matrix_method["n_train"], score_matrix_method["mean"] - score_matrix_method["se"],
                              score_matrix_method["mean"] + score_matrix_method["se"], alpha=0.2, 
                              color=method_config["color"], linestyle=method_config["linestyle"])
    
    axes[i].set_xscale("log")
    axes[i].set_xlabel("Number of training samples")
    axes[i].set_ylabel(metrics_config[score]["label"])
    axes[i].set_title(metrics_config[score]["label"])   
    if i == 0:
    # if i == 1:
        axes[i].legend()
    # axes[i].grid()
    axes[i].set_ylim(ylims[i])

    # line at 
    axes[i].axhline(0, color="black", linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join("plots_scripts", exp, f"{metrics_name}_{selection_name}.pdf"))
plt.show()
    

# %%
