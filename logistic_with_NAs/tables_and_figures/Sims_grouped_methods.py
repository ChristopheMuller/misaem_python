###
# Plot grid of results per simulation
###

#%%  if current working directory is "/plots_scripts", change it to the parent directory
import os
if os.getcwd().endswith("plots_scripts"):
    os.chdir(os.path.join(os.getcwd(), ".."))

# %% load packages

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

# Assuming these are defined in your project
from setups_design import metrics_config, methods_config
from utils import calculate_ymin_for_R_proportion

# %% set up

exp = "SimMCAR"
score_matrix = pd.read_csv(os.path.join("data", exp, "score_matrix.csv"))
score_matrix = score_matrix[score_matrix["exp"] == exp]

# Ensure score_matrix is filtered as in your original script
score_matrix = score_matrix[score_matrix["filter"] == "all"]

# Define your method groups and their display names
method_groups = {
    "Constant Imputation": [
        "Mean.IMP", "Mean.IMP.M", "05.IMP", "05.IMP.M"
    ],
    "MICE 1 Imputation": [
        "MICE.1.IMP", "MICE.1.Y.IMP", "MICE.1.M.IMP", "MICE.1.Y.M.IMP",
        "MICE.1.IMP.M", "MICE.1.Y.IMP.M", "MICE.1.M.IMP.M", "MICE.1.Y.M.IMP.M",
    ],
    "MICE 10 Imputations": [
        "MICE.10.IMP", "MICE.10.Y.IMP", "MICE.10.M.IMP", "MICE.10.Y.M.IMP",
        "MICE.10.IMP.M", "MICE.10.Y.IMP.M", "MICE.10.M.IMP.M", "MICE.10.Y.M.IMP.M",
    ],
    "MICE 100 Imputations": [
        "MICE.100.IMP", "MICE.100.Y.IMP", "MICE.100.M.IMP", "MICE.100.Y.M.IMP",
        "MICE.100.IMP.M", "MICE.100.Y.IMP.M", "MICE.100.M.IMP.M", "MICE.100.Y.M.IMP.M",
    ],
    "MICE.RF 10 Imputations": [
        "MICE.RF.10.IMP", "MICE.RF.10.Y.IMP", "MICE.RF.10.M.IMP", "MICE.RF.10.Y.M.IMP",
        "MICE.RF.10.IMP.M", "MICE.RF.10.Y.IMP.M", "MICE.RF.10.M.IMP.M", "MICE.RF.10.Y.M.IMP.M",
    ],
    "Selected Methods": [
        "CC",
        "PbP.Fixed",
        "Mean.IMP.M",
        "SAEM",
        # "MICE.100.Y.IMP",
        # "MICE.100.IMP",
        # "MICE.RF.10.Y.IMP",
        # "MICE.RF.10.IMP"
        "MICE.100.Y.M.IMP.M",
        "MICE.100.Y.IMP.M",
        "MICE.RF.10.Y.M.IMP.M",
        "MICE.RF.10.Y.IMP.M"
    ]
}

# Choose which groups to plot. You can modify this list to select specific groups.
selected_method_groups = [
    "Constant Imputation",
    "MICE 1 Imputation",
    "MICE 10 Imputations",
    "MICE 100 Imputations",
    "MICE.RF 10 Imputations",
    "Selected Methods"
]

# Define metrics
scores_sel = ["misclassification", "mae_bayes", "calibration", "mse_error"]
metrics_name = "4_metrics_grid_with_legends" # Updated filename
filter_bayes = [True, True, True, False]
ylimsmax = [0.1, 0.2, 0.009, 0.6]
ntrains = [100, 500, 1000, 5000, 10000, 50000]

ylimsmin = calculate_ymin_for_R_proportion(0.03, ylimsmax)
ylims = [(ylimsmin[i], ylimsmax[i]) for i in range(len(ylimsmax))]

num_rows = len(selected_method_groups)
num_cols = len(scores_sel) + 1 # +1 for the legend column

fig = plt.figure(figsize=(5 * len(scores_sel) + 1.5, 4.75 * num_rows))
gs = gridspec.GridSpec(num_rows, num_cols, figure=fig,
                         width_ratios=[1] * len(scores_sel) + [0.3],
                         )

for r_idx, group_name in enumerate(selected_method_groups):
    methods_in_group = method_groups[group_name]

    # Row title (group name) - Move closer to plots
    row_title_ax = fig.add_subplot(gs[r_idx, 0])
    row_title_ax.text(-0.15, 0.5, group_name,
                      transform=row_title_ax.transAxes,
                      fontsize=12, va='center', ha='right', rotation=90)
    row_title_ax.set_axis_off()

    # Calculate character for the label: 'a' + r_idx
    row_char_label = '(' + chr(ord('a') + r_idx) + ')'
    row_title_ax.text(-0.25, 0.5, row_char_label,
                      transform=row_title_ax.transAxes,
                      fontsize=14, va='center', ha='right')


    # Plot data for each metric column
    for c_idx, score in enumerate(scores_sel):
        ax = fig.add_subplot(gs[r_idx, c_idx])

        # Filter and group data
        score_matrix_sel_metric = score_matrix[score_matrix["metric"] == score].copy()
        score_matrix_sel_metric = score_matrix_sel_metric[score_matrix_sel_metric["method"].isin(methods_in_group)]
        score_matrix_sel_metric = score_matrix_sel_metric[score_matrix_sel_metric["exp"] == exp]
        score_matrix_sel_metric = score_matrix_sel_metric[score_matrix_sel_metric["n_train"].isin(ntrains)]
        score_matrix_sel_metric = score_matrix_sel_metric[score_matrix_sel_metric["bayes_adj"] == filter_bayes[c_idx]]

        score_matrix_sel_metric["score"] = score_matrix_sel_metric["score"].astype(float)
        score_matrix_grouped = score_matrix_sel_metric.groupby(["method", "n_train"]).agg({"score": ["mean", "std", "count"]}).reset_index()
        score_matrix_grouped.columns = ["method", "n_train", "mean", "sd", "count"]
        score_matrix_grouped["se"] = score_matrix_grouped["sd"] / np.sqrt(score_matrix_grouped["count"])

        # Plot lines and error bands
        for method in methods_in_group:
            if method in methods_config:
                method_config = methods_config[method]
                score_matrix_method = score_matrix_grouped[score_matrix_grouped["method"] == method]

                ax.plot(score_matrix_method["n_train"], score_matrix_method["mean"], label=method_config["label"],
                        color=method_config["color"], linestyle=method_config["linestyle"],
                        marker=method_config["marker"], markersize=5)
                ax.fill_between(score_matrix_method["n_train"],
                                score_matrix_method["mean"] - score_matrix_method["se"],
                                score_matrix_method["mean"] + score_matrix_method["se"],
                                alpha=0.2, color=method_config["color"])

        # Set common x-axis properties
        ax.set_xscale("log")
        ax.set_ylim(ylims[c_idx])

        # Set titles and labels
        if r_idx == 0: 
            ax.set_title(metrics_config[score]["label"], fontsize=12)

        if r_idx == num_rows - 1:
            ax.set_xlabel("Number of training samples", fontsize=10)
        else:
            ax.set_xticklabels([])

        # set up smaller y-axis labels
        ax.set_ylabel("", fontsize=10)
        ax.tick_params(axis='y', which='both', labelleft=True, labelsize=7)

        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)

    # Add legend to the last column for the current row's methods
    legend_ax = fig.add_subplot(gs[r_idx, num_cols - 1])
    handles, labels = [], []
    for method in methods_in_group:
        if method in methods_config:
            method_config = methods_config[method]
            handles.append(plt.Line2D([0], [0], color=method_config["color"],
                                      linestyle=method_config["linestyle"],
                                      marker=method_config["marker"], markersize=5))
            labels.append(method_config["label"])


    sorted_labels, sorted_handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

    legend_ax.legend(sorted_handles, sorted_labels, loc='center left',
                     bbox_to_anchor=(-0.1, 0.5), frameon=False, fontsize=10)
    legend_ax.set_axis_off()

plt.subplots_adjust(left=0.08, top=0.9, right=0.85, bottom=0.05, wspace=0.21, hspace=0.1)

plt.savefig(os.path.join("plots_scripts", exp, f"{metrics_name}_grid.pdf"), bbox_inches='tight', pad_inches=0.1)
plt.show()