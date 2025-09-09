import numpy as np


def louis_lr_saem(beta, mu, Sigma, Y, X_obs, pos_var=None, rindic=None, nmcmc=2):
    if pos_var is None:
        pos_var = np.arange(X_obs.shape[1])
    if rindic is None:
        rindic = np.isnan(X_obs).astype(int)

    n = X_obs.shape[0]
    p = len(pos_var)

    beta = beta[[0] + (pos_var + 1).tolist()]
    mu = mu[pos_var]
    Sigma = Sigma[np.ix_(pos_var, pos_var)]
    X_obs = X_obs[:, pos_var]
    rindic = rindic[:, pos_var]

    # Initialize X.mean, X.sim
    X_mean = np.copy(X_obs)
    for i in range(X_mean.shape[1]):
        nan_idx = np.isnan(X_mean[:, i])
        X_mean[nan_idx, i] = np.nanmean(X_mean[:, i])

    X_sim = np.copy(X_mean)
    G = D = I_obs = np.zeros((p + 1, p + 1))
    Delta = np.zeros((p + 1, 1))
    S_inv = np.linalg.inv(Sigma)

    for i in range(n):
        jna = np.where(np.isnan(X_obs[i, :]))[0]
        njna = len(jna)

        if njna == 0:
            x = np.concatenate([[1], X_sim[i, :]])
            exp_b = np.exp(beta @ x)
            d2l = -np.outer(x, x) * (exp_b / (1 + exp_b) ** 2)
            I_obs -= d2l

        if njna > 0:
            xi = X_sim[i, :]
            Oi = np.linalg.inv(S_inv[np.ix_(jna, jna)])
            mi = mu[jna]
            lobs = beta[0]

            if njna < p:
                jobs = np.setdiff1d(np.arange(p), jna)
                mi = mi - (xi[jobs] - mu[jobs]) @ S_inv[np.ix_(jobs, jna)] @ Oi
                lobs += np.sum(xi[jobs] * beta[jobs + 1])

            cobs = np.exp(lobs)
            xina = xi[jna]
            betana = beta[jna + 1]

            for m in range(1, nmcmc + 1):
                xina_c = mi + np.random.randn(njna) @ np.linalg.cholesky(Oi).T
                if Y[i] == 1:
                    alpha = (1 + np.exp(-np.sum(xina * betana)) / cobs) / (
                        1 + np.exp(-np.sum(xina_c * betana)) / cobs
                    )
                else:
                    alpha = (1 + np.exp(np.sum(xina * betana)) * cobs) / (
                        1 + np.exp(np.sum(xina_c * betana)) * cobs
                    )

                if np.random.rand() < alpha:
                    xina = xina_c

                X_sim[i, jna] = xina
                x = np.concatenate([[1], X_sim[i, :]])
                exp_b = np.exp(beta @ x)
                dl = x * (Y[i] - exp_b / (1 + exp_b))
                d2l = -np.outer(x, x) * (exp_b / (1 + exp_b) ** 2)

                D = D + (1 / m) * (d2l - D)
                G = G + (1 / m) * (dl[:, None] @ dl[None, :] - G)
                Delta = Delta + (1 / m) * (dl[:, None] - Delta)

            I_obs -= D + G - Delta @ Delta.T

    V_obs = np.linalg.inv(I_obs)
    return V_obs


def log_reg(y, x, beta, log=True):
    """
    Compute the (log-)likelihood of a logistic regression model.
    """
    res = y * (beta @ x) - np.log(1 + np.exp(beta @ x))
    if log:
        return res
    else:
        return np.exp(res)


def likelihood_saem(beta, mu, Sigma, Y, X_obs, rindic=None, nmcmc=2):

    n = X_obs.shape[0]
    p = X_obs.shape[1]

    if rindic is None:
        rindic = np.isnan(X_obs).astype(int)

    lh = 0

    for i in range(n):

        y = Y[i]
        x = X_obs[i, :]

        if np.sum(rindic[i, :]) == 0:
            lh += log_reg(y, np.concatenate([[1], x]), beta, log=True)
        else:

            miss_col = np.where(rindic[i, :])[0]
            x2 = np.delete(x, miss_col)
            mu1 = mu[miss_col]
            mu2 = mu[np.delete(np.arange(p), miss_col)]

            sigma11 = Sigma[np.ix_(miss_col, miss_col)]
            sigma12 = Sigma[np.ix_(miss_col, np.delete(np.arange(p), miss_col))]
            sigma22 = Sigma[
                np.ix_(
                    np.delete(np.arange(p), miss_col), np.delete(np.arange(p), miss_col)
                )
            ]
            sigma21 = sigma12.T

            mu_cond = mu1 + sigma12 @ np.linalg.inv(sigma22) @ (x2 - mu2)
            sigma_cond = sigma11 - sigma12 @ np.linalg.inv(sigma22) @ sigma21

            # generate missing values
            x1_all = np.zeros((nmcmc, len(miss_col)))
            for m in range(nmcmc):
                x1_all[m, :] = mu_cond + np.random.normal(
                    size=len(miss_col)
                ) @ np.linalg.cholesky(sigma_cond)

            lh_mis1 = 0
            for m in range(nmcmc):
                x[miss_col] = x1_all[m, :]
                lh_mis1 += log_reg(y, np.concatenate([[1], x]), beta, log=False)

            lr = np.log(lh_mis1 / nmcmc)
            lh += lr

    return lh


def combinations(p):
    if p < 20:
        comb = np.array([[1], [0]])  # Start with combinations of 1 variable
        for i in range(1, p):  # Iterate for each variable
            comb = np.vstack(
                [
                    np.hstack([np.ones((comb.shape[0], 1)), comb]),
                    np.hstack([np.zeros((comb.shape[0], 1)), comb]),
                ]
            )
        return comb
    else:
        raise ValueError(
            "Error: the dimension of dataset is too large to possibly block your computer. Better try with number of variables smaller than 20."
        )


def check_X_y(X=None, y=None, predict=False):
    if y is None and not predict:
        raise ValueError("y cannot be None when fitting.")
    if X is None:
        raise ValueError("X cannot be None.")

    X = np.asarray(X).copy()
    if y is not None:
        y = np.asarray(y).ravel().copy()

    if y is not None:
        if np.any(np.isnan(y)):
            raise ValueError("No missing data allowed in response variable y")
        
        unique_y = np.unique(y)
        if len(unique_y) != 2 or not np.array_equal(unique_y, [0, 1]):
            raise ValueError("y must be binary with values 0 and 1.")

    if np.all(np.isnan(X)):
        raise ValueError("X contains only NaN values.")

    complete_rows = ~np.all(np.isnan(X), axis=1)
    
    if np.any(~complete_rows):
        if y is not None:
            y = y[complete_rows]
        X = X[complete_rows]

    if not predict:
        if np.any(np.all(np.isnan(X), axis=0)):
            raise ValueError("X contains at least one column with only NaN values.")

    if y is not None:
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y do not match.")

    return X, y


def compute_conditional_mvn_params(
        sigma_inv, missing_idx, obs_idx, X_sim, rows_with_pattern, mu, beta
):
    Q_MM = sigma_inv[np.ix_(missing_idx, missing_idx)]
    Q_MO = sigma_inv[np.ix_(missing_idx, obs_idx)]

    sigma_cond_M = np.linalg.inv(Q_MM)

    X_O = X_sim[rows_with_pattern][:, obs_idx]

    delta_X_term = (X_O - mu[obs_idx]).T
    adjustment_term = (sigma_cond_M @ (Q_MO @ delta_X_term)).T
    mu_cond_M = mu[missing_idx] - adjustment_term

    lobs = beta[0] + X_O @ beta[obs_idx + 1]

    return mu_cond_M, sigma_cond_M, lobs