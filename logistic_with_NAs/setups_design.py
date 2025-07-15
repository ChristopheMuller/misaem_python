import numpy as np
import matplotlib.pyplot as plt

metrics_config = {
    "brier": {"label": "Brier"},
    "misclassification": {"label": "Classification"},
    "mae_bayes": {"label": "Probability Estimation"},
    "calibration": {"label": "Calibration"},

    "angular_error": {"label": "Angular"},
    "mse_error": {"label": "Inference"},
    "mse_error_with_intercept": {"label": "Inference"},
    "angular_error_with_intercept": {"label": "Angular"},
}

import seaborn as sns
color_palette = sns.color_palette()

methods_config = {

    "05.IMP": {"label": "05.IMP", "color": color_palette[8], "linestyle": "-", "marker":"o"},
    "05.IMP.M": {"label": "05.IMP.M", "color": color_palette[8], "linestyle": "--", "marker":"v"}, # Changed marker

    "CC": {"label": "CC", "color": color_palette[9], "linestyle": "-", "marker":"s"}, # Changed marker

    "PbP.Fixed": {"label": "PbP", "color": color_palette[0], "linestyle": "-", "marker":"o"},
    "PbP.MinObs": {"label": "PbP.MinObs", "color": color_palette[7], "linestyle": "--", "marker":"x"},

    "SAEM": {"label": "SAEM", "color": color_palette[5], "linestyle": "-", "marker":"D"}, # Changed marker

    "Mean.IMP": {"label": "Mean.IMP", "color": color_palette[3], "linestyle": "-", "marker":"o"},
    "Mean.IMP.M": {"label": "Mean.IMP.M", "color": color_palette[3], "linestyle": "--", "marker":"v"}, # Changed marker

    # MICE 1 Imputation Group - Solid lines
    "MICE.1.IMP": {"label": "MICE.1.IMP", "color": color_palette[4], "linestyle": "-", "marker":"o"},
    "MICE.1.Y.IMP": {"label": "MICE.1.Y.IMP", "color": color_palette[2], "linestyle": "-", "marker":"o"},
    "MICE.1.M.IMP": {"label": "MICE.1.M.IMP", "color": color_palette[1], "linestyle": "-", "marker":"o"},
    "MICE.1.Y.M.IMP": {"label": "MICE.1.Y.M.IMP", "color": color_palette[6], "linestyle": "-", "marker":"o"},

    # MICE 1 Imputation Group with .M suffix - Dashed lines
    "MICE.1.IMP.M": {"label": "MICE.1.IMP.M", "color": color_palette[4], "linestyle": "--", "marker":"x"},
    "MICE.1.Y.IMP.M": {"label": "MICE.1.Y.IMP.M", "color": color_palette[2], "linestyle": "--", "marker":"x"},
    "MICE.1.M.IMP.M": {"label": "MICE.1.M.IMP.M", "color": color_palette[1], "linestyle": "--", "marker":"x"},
    "MICE.1.Y.M.IMP.M": {"label": "MICE.1.Y.M.IMP.M", "color": color_palette[6], "linestyle": "--", "marker":"x"},

    # MICE 10 Imputations Group - Dashdot lines
    "MICE.10.IMP": {"label": "MICE.10.IMP", "color": color_palette[4], "linestyle": "-", "marker":"o"},
    "MICE.10.Y.IMP": {"label": "MICE.10.Y.IMP", "color": color_palette[2], "linestyle": "-", "marker":"o"},
    "MICE.10.M.IMP": {"label": "MICE.10.M.IMP", "color": color_palette[1], "linestyle": "-", "marker":"o"},
    "MICE.10.Y.M.IMP": {"label": "MICE.10.Y.M.IMP", "color": color_palette[6], "linestyle": "-", "marker":"o"},

    # MICE 10 Imputation Group with .M suffix - Dotted lines
    "MICE.10.IMP.M": {"label": "MICE.10.IMP.M", "color": color_palette[4], "linestyle": "--", "marker":"x"},
    "MICE.10.Y.IMP.M": {"label": "MICE.10.Y.IMP.M", "color": color_palette[2], "linestyle": "--", "marker":"x"},
    "MICE.10.M.IMP.M": {"label": "MICE.10.M.IMP.M", "color": color_palette[1], "linestyle": "--", "marker":"x"},
    "MICE.10.Y.M.IMP.M": {"label": "MICE.10.Y.M.IMP.M", "color": color_palette[6], "linestyle": "--", "marker":"x"},

    # MICE 100 Imputations Group - Dotted lines
    # Using more distinct markers here for differentiation from MICE.10.IMP.M if on same plot
    "MICE.100.IMP": {"label": "MICE.100.IMP", "color": color_palette[4], "linestyle": "-", "marker":"o"},
    "MICE.100.Y.IMP": {"label": "MICE.100.Y.IMP", "color": color_palette[2], "linestyle": "-", "marker":"o"},
    "MICE.100.M.IMP": {"label": "MICE.100.M.IMP", "color": color_palette[1], "linestyle": "-", "marker":"o"},
    "MICE.100.Y.M.IMP": {"label": "MICE.100.Y.M.IMP", "color": color_palette[6], "linestyle": "-", "marker":"o"},

    # MICE 100 Imputation Group with .M suffix - Loosely dashed lines
    "MICE.100.IMP.M": {"label": "MICE.100.IMP.M", "color": color_palette[4], "linestyle": "--", "marker":"x"},
    "MICE.100.Y.IMP.M": {"label": "MICE.100.Y.IMP.M", "color": color_palette[2], "linestyle": "--", "marker":"x"},
    "MICE.100.M.IMP.M": {"label": "MICE.100.M.IMP.M", "color": color_palette[1], "linestyle": "--", "marker":"x"},
    "MICE.100.Y.M.IMP.M": {"label": "MICE.100.Y.M.IMP.M", "color": color_palette[6], "linestyle": "--", "marker":"x"},

    # MICE.RF 10 Imputations Group - Solid lines (reusing patterns but with RF colors)
    "MICE.RF.10.IMP": {"label": "MICE.RF.10.IMP", "color": color_palette[4], "linestyle": ":", "marker":"D"},
    "MICE.RF.10.Y.IMP": {"label": "MICE.RF.10.Y.IMP", "color": color_palette[2], "linestyle": ":", "marker":"D"},
    "MICE.RF.10.M.IMP": {"label": "MICE.RF.10.M.IMP", "color": color_palette[1], "linestyle": ":", "marker":"D"},
    "MICE.RF.10.Y.M.IMP": {"label": "MICE.RF.10.Y.M.IMP", "color": color_palette[6], "linestyle": ":", "marker":"D"},

    # MICE.RF 10 Imputation Group with .M suffix - Dashed lines (reusing patterns)
    "MICE.RF.10.IMP.M": {"label": "MICE.RF.10.IMP.M", "color": color_palette[4], "linestyle": "dashdot", "marker":"P"},
    "MICE.RF.10.Y.IMP.M": {"label": "MICE.RF.10.Y.IMP.M", "color": color_palette[2], "linestyle": "dashdot", "marker":"P"},
    "MICE.RF.10.M.IMP.M": {"label": "MICE.RF.10.M.IMP.M", "color": color_palette[1], "linestyle": "dashdot", "marker":"P"},
    "MICE.RF.10.Y.M.IMP.M": {"label": "MICE.RF.10.Y.M.IMP.M", "color": color_palette[6], "linestyle": "dashdot", "marker":"P"},
}
