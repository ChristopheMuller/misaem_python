import numpy as np
import os

def sigma(x):
    return 1 / (1 + np.exp(-x))

def toep(d, rate):
    return np.array([[(rate) ** abs(i - j) for i in range(d)] for j in range(d)])

def generate_X(d, corr_rate, n, prop=None, beta0=None, limit=0.005, max_iter=100, intercept=0):
    cov = toep(d, corr_rate)
    X = np.random.multivariate_normal(np.zeros(d), cov, size=n)

    center_X = np.zeros(d)

    if prop is not None:
        for _ in range(max_iter):
            y_attempt = sigma(X @ beta0 + intercept)
            y_drawn = np.random.binomial(1, y_attempt)
            current_prop = np.mean(y_drawn)

            if np.abs(current_prop - prop) < limit:
                break

            # Adaptive step size with Newton-like update
            step_size = np.clip((prop - current_prop) / (current_prop * (1 - current_prop) + 1e-6), -0.5, 0.5)
            X += step_size * beta0  # Faster convergence
            center_X += step_size * beta0
        
    return X, center_X


    
def get_y_prob_bayes(X_m, full_mu, full_cov, true_beta, n_mc=1000, intercept=0):
    # Group rows by their missingness pattern
    M = np.isnan(X_m)
    unique_patterns = np.unique(M, axis=0)
    
    # Preallocate result array
    prob_y_all = np.zeros((X_m.shape[0], n_mc))
    
    for pattern in unique_patterns:
        # Find indices of rows with this missingness pattern
        pattern_indices = np.all(M == pattern, axis=1)
        X_m_subset = X_m[pattern_indices]
        
        # Directly reuse get_y_prob_bayes_same_pattern for this subset
        prob_y_subset = get_y_prob_bayes_same_pattern(X_m_subset, full_mu, full_cov, true_beta, n_mc, intercept)
        
        # Update the results for this pattern
        prob_y_all[pattern_indices] = prob_y_subset
    
    return prob_y_all


def get_y_prob_bayes_same_pattern(X_m, full_mu, full_cov, true_beta, n_mc=1000, intercept=0):
    """
    Apply the conditional_probability function row-wise to X_m,
    optimized for a shared missingness pattern m across all rows.

    Parameters:
    - X_m: Data matrix with missing values (rows to process).
    - full_mu: Mean vector of the full data.
    - full_cov: Covariance matrix of the full data.
    - true_beta: Coefficients for the logistic model.
    - n_mc: Number of Monte Carlo samples.

    Returns:
    - prob_y_all: Array of probabilities for all rows in X_m.
    - X_mc_all: Generated imputed datasets for all rows in X_m.
    """
    # Determine the shared missingness pattern m
    m = np.isnan(X_m[0])
    
    # Precompute components that depend only on m
    observed_idx = ~m
    missing_idx = m

    mu_obs = full_mu[observed_idx]
    mu_mis = full_mu[missing_idx]

    cov_obs = full_cov[np.ix_(observed_idx, observed_idx)]
    cov_obs_inv = np.linalg.inv(cov_obs)

    cov_mis = full_cov[np.ix_(missing_idx, missing_idx)]
    cross_cov = full_cov[np.ix_(observed_idx, missing_idx)]

    # Precompute covariances
    cond_cov = cov_mis - cross_cov.T @ cov_obs_inv @ cross_cov

    # Initialize results
    prob_y_all = []
    X_mc_all = []

    for x_obs in X_m:
        x_obs_obs = x_obs[observed_idx]

        # Compute conditional mean
        cond_mu = mu_mis + cross_cov.T @ cov_obs_inv @ (x_obs_obs - mu_obs)

        # Generate Monte Carlo samples
        if len(cond_mu) == 0:
            X_mc = np.zeros((n_mc, 0))
        else:
            X_mc = np.random.multivariate_normal(cond_mu, cond_cov, size=n_mc)

        # Full imputed matrix
        X_full_mc = np.tile(x_obs, (n_mc, 1))
        X_full_mc[:, missing_idx] = X_mc

        # Compute probabilities
        logits_mc = X_full_mc @ true_beta + intercept
        prob_y_mc = sigma(logits_mc)

        # Append results
        prob_y_all.append(prob_y_mc)
        X_mc_all.append(X_mc)

    return np.array(prob_y_all)


def load_data(set_up, data_type, exp):

    dict_path = {
        "bayes": "bayes_data",
        "test": "test_data",
        "original": "original_data",
    }

    if type(set_up) == str:
        
        path = os.path.join("data", exp, dict_path[data_type], f"{set_up}.npz")
        data = np.load(path)

        return data
    
def get_full_pattern(pattern, d):

    m = np.array(pattern)
    if len(m) < d:
        m = np.concatenate([m, np.zeros(d - len(m))])
    if len(m) > d:
        m = m[:d]
    m = m.astype(int)

    return m

def get_index_pattern(pattern, M, remove_all_missing=True):
    if type(pattern) == int:
        # get idx of M where sum(row) == pattern
        temp = np.where(np.sum(M, axis=1) == pattern)[0]
    elif type(pattern) == list:
        full_pattern = get_full_pattern(pattern, M.shape[1])
        temp = np.where(np.all(M == full_pattern, axis=1))[0]
    elif pattern is None:
        temp = np.arange(M.shape[0])
    else:
        raise ValueError("Pattern type not recognized.", pattern)

    if remove_all_missing:
        id_all_missing = np.where(np.sum(M, axis=1) == M.shape[1])[0]
        temp = np.setdiff1d(temp, id_all_missing)

    return temp


def get_index_pattern_probs(pattern_probs=None, true_y=None, bayes=None):

    if pattern_probs is None:
        pattern_probs = {"bayes":[0,1], "true":[0,1]}
    
    all_idx = np.arange(len(true_y))
    if pattern_probs["bayes"] is not None:
        if bayes is None:
            raise ValueError("Bayes data is missing")
        else:
            from_bayes = pattern_probs["bayes"][0]
            to_bayes = pattern_probs["bayes"][1]
            all_idx = all_idx[np.where((bayes[all_idx] >= from_bayes) & (bayes[all_idx] < to_bayes))[0]]

    if pattern_probs["true"] is not None:
        if true_y is None:
            raise ValueError("True data is missing")
        from_true = pattern_probs["true"][0]
        to_true = pattern_probs["true"][1]
        all_idx = all_idx[np.where((true_y[all_idx] >= from_true) & (true_y[all_idx] < to_true))[0]]

    return all_idx


def generate_mask(n, d, prc):
    """Generate an MCAR missingness mask M, ensuring no row is fully missing."""
    M = np.random.binomial(n=1, p=prc, size=(n, d))

    # Identify rows that are fully missing (all 1s)
    all_ones = np.all(M == 1, axis=1)

    # Redraw only those rows until no row is fully missing
    while np.any(all_ones):
        M[all_ones] = np.random.binomial(n=1, p=prc, size=(all_ones.sum(), d))
        all_ones = np.all(M == 1, axis=1)  # Recheck after redrawing

    return M

def filter_data(df, **kwargs):
    for key, value in kwargs.items():
        if key in df.columns:
            df = df[df[key] == value]
        else:
            print(f"Warning: {key} not in columns")
    return df

def calculate_ymin_for_R_proportion(R, max_y_values):


    if not (0 < R < 1):
        raise ValueError("R (proportion) must be between 0 and 1 (exclusive of 1).")

    # Ensure max_y_values is iterable
    if isinstance(max_y_values, (int, float)):
        max_y_values = [max_y_values]

    calculated_ymin_values = []
    
    # The formula derived from R = (0 - y_min) / (y_max - y_min)
    # is: y_min = (R * y_max) / (R - 1)
    # Since R is a proportion, R-1 will be negative, making y_min negative as expected.
    denominator = R - 1 

    for y_max in max_y_values:
        y_min = (R * y_max) / denominator
        calculated_ymin_values.append(y_min)
        
    return calculated_ymin_values