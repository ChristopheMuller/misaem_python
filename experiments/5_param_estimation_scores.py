# %%

from utils import *
import numpy as np
import pandas as pd
import os

# %%

exp = "MAR_5d_050corr"

simulation_set_up = pd.read_csv(os.path.join("data", exp, "simulation_set_up.csv"))

# %%

import re

def parse_vector_from_string(s):
    """Parses a vector stored as a string in a CSV cell."""

    if pd.isnull(s):
        return None
    
    s = s.strip()
    
    # Remove brackets if present
    s = s.lstrip("[").rstrip("]")
    
    # Try parsing with commas first
    if "," in s:
        values = [float(x) for x in s.split(",")]
    else:
        # Use regex to extract numbers (handles spaces between numbers)
        values = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]
    
    return np.array(values)

# %%

def compute_beta_estimation_error(true_params, pred_params, error="mse"):
    """
    Computes estimation error for beta or concatenated (intercept, beta).
    Args:
        true_params (np.array or string): True parameters (beta or concatenated).
        pred_params (np.array or string): Predicted parameters (beta or concatenated).
        error (str): Type of error ("mse" or "angular").
    """
    # Ensure inputs are numpy arrays
    if isinstance(true_params, str):
        true_params = parse_vector_from_string(true_params)
    if isinstance(pred_params, str):
        pred_params = parse_vector_from_string(pred_params)

    if pred_params is None or true_params is None: 
        return None
    
    # Ensure they are numpy arrays if they weren't strings
    true_params = np.array(true_params)
    pred_params = np.array(pred_params)

    if error == "mse":
        return np.mean((true_params - pred_params) ** 2)
    
    if error == "angular":
        norm_true = np.linalg.norm(true_params)
        norm_pred = np.linalg.norm(pred_params)
        
        if norm_true == 0 or norm_pred == 0:
            if norm_true == 0 and norm_pred == 0:
                return 0.0
            else:
                return np.nan 
        
        cosine_angle = np.dot(true_params, pred_params) / (norm_true * norm_pred)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return np.arccos(cosine_angle)
    
    return None 

# %%
# Ensure 'true_intercept' exists and default to 0 if not
if "true_intercept" not in simulation_set_up.columns:
    simulation_set_up["true_intercept"] = 0.0

# %%
# Calculate errors for beta coefficients only
simulation_set_up["angular_error"] = simulation_set_up.apply(
    lambda x: compute_beta_estimation_error(x["true_beta"], x["pred_beta"], "angular"), 
    axis=1
)
simulation_set_up["mse_error"] = simulation_set_up.apply(
    lambda x: compute_beta_estimation_error(x["true_beta"], x["pred_beta"], "mse"), 
    axis=1
)

# %%
# Calculate errors for concatenated (intercept + beta) parameters
simulation_set_up["angular_error_with_intercept"] = simulation_set_up.apply(
    lambda x: compute_beta_estimation_error(
        np.concatenate([np.array([x["true_intercept"]]), parse_vector_from_string(x["true_beta"])])
        if parse_vector_from_string(x["true_beta"]) is not None else None,
        np.concatenate([np.array([x["pred_intercept"]]), parse_vector_from_string(x["pred_beta"])])
        if parse_vector_from_string(x["pred_beta"]) is not None else None,
        "angular"
    ),
    axis=1
)

simulation_set_up["mse_error_with_intercept"] = simulation_set_up.apply(
    lambda x: compute_beta_estimation_error(
        np.concatenate([np.array([x["true_intercept"]]), parse_vector_from_string(x["true_beta"])])
        if parse_vector_from_string(x["true_beta"]) is not None else None,
        np.concatenate([np.array([x["pred_intercept"]]), parse_vector_from_string(x["pred_beta"])])
        if parse_vector_from_string(x["pred_beta"]) is not None else None,
        "mse"
    ),
    axis=1
)

# %%

simulation_set_up.to_csv(os.path.join("data",exp,"simulation_set_up.csv"), index=False)

# %%
