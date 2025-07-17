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

    "running_time_train": {"label": "Training Time"},
    "running_time_pred": {"label": "Prediction Time"},
}

import seaborn as sns
color_palette = sns.color_palette()

methods_config = {

    "SAEM": {"label": "R.SAEM", "color": color_palette[5], "linestyle": "-", "marker":"D"}, # Changed marker
    "py.SAEM": {"label": "py.SAEM", "color": color_palette[1], "linestyle": "--", "marker":"D"},
    "py.SAEM.fast": {"label": "py.SAEM.fast", "color": color_palette[2], "linestyle": "-.", "marker":"D"},
    "py.SAEM.fast.fixed": {"label": "py.SAEM.fast.fixed", "color": color_palette[0], "linestyle": "-", "marker":"D"},
    "py.SAEM.parallel": {"label": "py.SAEM.parallel", "color": color_palette[3], "linestyle": ":", "marker":"D"},
    "py.SAEM.parallel.fast": {"label": "py.SAEM.parallel.fast", "color": color_palette[4], "linestyle": "-", "marker":"D"},
}
