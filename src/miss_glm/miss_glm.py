from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y
from typing import Optional, List, Dict, Any
import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm
from .utils import louis_lr_saem, likelihood_saem


class MissGLM(BaseEstimator, ClassifierMixin):
    """Logistic regression model that handles missing data using SAEM algorithm.

    Parameters
    ----------
    maxruns : int, default=500
        Maximum number of SAEM iterations.
    tol_em : float, default=1e-7
        Convergence tolerance for SAEM algorithm.
    nmcmc : int, default=2
        Number of MCMC iterations per SAEM step.
    tau : float, default=1.0
        Learning rate decay parameter.
    k1 : int, default=50
        Number of initial iterations with step size 1.
    var_cal : bool, default=True
        Whether to calculate variance estimates.
    ll_obs_cal : bool, default=True
        Whether to calculate observed data likelihood.
    subsets : ArrayLike, optional
        Subset of features to use in model.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    coef_ : NDArray
        Model coefficients.
    mu_ : NDArray
        Estimated means of features.
    sigma_ : NDArray
        Estimated covariance matrix of features.
    ll_obs : float, optional
        Observed data likelihood.
    std_err : NDArray, optional
        Standard errors of coefficients.
    trace : Dict, optional
        Evolution of parameters during SAEM iterations.
    """

    def __init__(
        self,
        maxruns: int = 500,
        tol_em: float = 1e-7,
        nmcmc: int = 2,
        tau: float = 1.0,
        k1: int = 50,
        var_cal: bool = True,
        ll_obs_cal: bool = True,
        subsets: Optional[ArrayLike] = None,
        seed: Optional[int] = None,
    ):

        self.maxruns = maxruns
        self.tol_em = tol_em
        self.nmcmc = nmcmc
        self.tau = tau
        self.k1 = k1
        self.subsets = subsets
        self.seed = seed
        self.var_cal = var_cal
        self.ll_obs_cal = ll_obs_cal
        self.coef_ = None
        self.mu_ = None
        self.sigma_ = None
        self.ll_obs = None
        self.std_err = None
        self.trace: Dict[str, List[Any]] = {}

    def fit(self, X, y, save_trace=False, progress_bar=True):
        """Fit the model using SAEM algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target values.
        save_trace : bool, default=False
            Whether to save evolution of parameters.

        Returns
        -------
        self : object
            Returns self.
        """

        if np.any(np.isnan(y)):
            raise ValueError("No missing data allowed in response variable y")

        complete_rows = ~np.all(np.isnan(X), axis=1)
        X = X[complete_rows]
        y = y[complete_rows]

        n, p = X.shape

        if self.subsets is None:
            self.subsets = np.arange(p)
        if isinstance(self.subsets, list):
            self.subsets = np.array(self.subsets)

        if len(np.unique(self.subsets)) != len(self.subsets):
            raise ValueError("Subsets must be unique.")

        rindic = np.isnan(X)
        missing_cols = np.any(rindic, axis=0)
        num_missing_cols = np.sum(missing_cols)

        if self.seed is not None:
            np.random.seed(self.seed)

        if num_missing_cols > 0:
            X_sim = np.where(rindic, np.nanmean(X, axis=0), X)
            mu = np.mean(X_sim, axis=0)
            sigma = np.cov(X_sim, rowvar=False) * (n - 1) / n
            sigma_inv = np.linalg.inv(sigma)

            log_reg_model = LogisticRegression(
                solver="lbfgs", max_iter=1000, fit_intercept=True
            )
            log_reg_model.fit(X_sim[:, self.subsets], y)
            beta = np.zeros(p + 1)
            beta[np.hstack([0, self.subsets + 1])] = np.hstack(
                [log_reg_model.intercept_, log_reg_model.coef_.ravel()]
            )

            unique_patterns, pattern_indices = np.unique(
                rindic, axis=0, return_inverse=True
            )

            if save_trace:
                self.trace["beta"] = [beta]
                self.trace["mu"] = [mu]
                self.trace["sigma"] = [sigma]
                self.trace["accepted_X"] = []

            for k in tqdm(range(self.maxruns), disable=not progress_bar):
                beta_old = beta.copy()

                for pattern_idx, pattern in enumerate(unique_patterns):
                    if not np.any(pattern):
                        continue

                    rows_with_pattern = np.where(pattern_indices == pattern_idx)[0]
                    n_pattern = len(rows_with_pattern)

                    missing_idx = np.where(pattern)[0]
                    obs_idx = np.where(~pattern)[0]
                    n_missing = len(missing_idx)

                    if n_missing > 0:
                        Q_MM = sigma_inv[np.ix_(missing_idx, missing_idx)]
                        Q_MO = sigma_inv[np.ix_(missing_idx, obs_idx)]

                        sigma_cond_M = np.linalg.inv(Q_MM)

                        X_O = X_sim[rows_with_pattern][:, obs_idx]

                        delta_X_term = (X_O - mu[obs_idx]).T
                        adjustment_term = (sigma_cond_M @ (Q_MO @ delta_X_term)).T
                        mu_cond_M = mu[missing_idx] - adjustment_term

                        lobs = beta[0] + X_O @ beta[obs_idx + 1]

                    else:
                        sigma_cond_M = sigma.copy()
                        mu_cond_M = np.tile(mu, (n_pattern, 1))
                        lobs = beta[0]

                    cobs = np.exp(lobs)
                    xina = X_sim[rows_with_pattern][:, missing_idx]
                    betana = beta[missing_idx + 1]
                    y_pattern = y[rows_with_pattern]

                    chol_sigma_cond_M = np.linalg.cholesky(sigma_cond_M)

                    for m in range(self.nmcmc):
                        xina_c = (
                            mu_cond_M
                            + np.random.normal(size=(n_pattern, n_missing))
                            @ chol_sigma_cond_M
                        )

                        current_logit_contrib = np.sum(xina * betana, axis=1)
                        candidate_logit_contrib = np.sum(xina_c * betana, axis=1)

                        is_y1 = y_pattern == 1

                        ratio_y1 = (1 + np.exp(-current_logit_contrib) / cobs) / (
                            1 + np.exp(-candidate_logit_contrib) / cobs
                        )
                        ratio_y0 = (1 + np.exp(current_logit_contrib) * cobs) / (
                            1 + np.exp(candidate_logit_contrib) * cobs
                        )

                        alpha = np.where(is_y1, ratio_y1, ratio_y0)

                        accepted = np.random.uniform(size=n_pattern) < alpha
                        xina[accepted] = xina_c[accepted]

                    X_sim[np.ix_(rows_with_pattern, missing_idx)] = xina

                log_reg_model.fit(X_sim[:, self.subsets], y)
                beta_new = np.zeros(p + 1)
                beta_new[np.hstack([0, self.subsets + 1])] = np.hstack(
                    [log_reg_model.intercept_, log_reg_model.coef_.ravel()]
                )

                gamma = 1 if k < self.k1 else 1 / ((k - self.k1 + 1) ** self.tau)
                beta = (1 - gamma) * beta + gamma * beta_new
                mu = (1 - gamma) * mu + gamma * np.mean(X_sim, axis=0)
                sigma = (1 - gamma) * sigma + gamma * np.cov(
                    X_sim, rowvar=False, bias=True
                )
                sigma_inv = np.linalg.inv(sigma)

                if save_trace:
                    self.trace["beta"].append(beta.copy())
                    self.trace["mu"].append(mu.copy())
                    self.trace["sigma"].append(sigma.copy())

                if np.sum((beta - beta_old) ** 2) < self.tol_em:
                    if progress_bar:
                        print(f"...converged after {k+1} iterations.")
                    break

            var_obs = None
            ll = None
            std_obs = None

            if self.var_cal:
                var_obs = louis_lr_saem(
                    beta,
                    mu,
                    sigma,
                    y,
                    X,
                    pos_var=self.subsets,
                    rindic=rindic,
                    nmcmc=100,
                )
                std_obs = np.sqrt(np.diag(var_obs))
                self.std_err = std_obs

            if self.ll_obs_cal:
                ll = likelihood_saem(beta, mu, sigma, y, X, rindic=rindic, nmcmc=100)
                self.ll_obs = ll

        else:
            log_reg = LogisticRegression(solver="lbfgs", max_iter=1000)
            log_reg.fit(X, y)
            beta = np.hstack([log_reg.intercept_, log_reg.coef_.ravel()])
            mu = np.nanmean(X, axis=0)
            sigma = np.cov(X, rowvar=False) * (n - 1) / n
            if self.var_cal:
                X_design = np.hstack([np.ones((n, 1)), X])
                linear_pred = X_design @ beta
                P = 1 / (1 + np.exp(-linear_pred))
                W = np.diag(P * (1 - P))
                var_obs = np.linalg.inv(X_design.T @ W @ X_design)
                std_obs = np.sqrt(np.diag(var_obs))
                self.std_err = std_obs

            if self.ll_obs_cal:
                ll = likelihood_saem(beta, mu, sigma, y, X, rindic=rindic, nmcmc=100)
                self.ll_obs = ll

        self.coef_ = beta[np.hstack([0, self.subsets + 1])]
        self.mu_ = mu
        self.sigma_ = sigma
        return self

    def predict_proba(self, Xtest, method="map", nmcmc=500):

        if self.seed is not None:
            np.random.seed(self.seed)

        mu_saem = self.mu_
        sigma_saem = self.sigma_
        beta_saem = self.coef_

        n, p = Xtest.shape
        pr_saem = np.zeros(n)
        rindic = np.isnan(Xtest)

        unique_patterns, pattern_indices = np.unique(
            rindic, axis=0, return_inverse=True
        )

        for pattern_idx, pattern in enumerate(unique_patterns):

            rows_with_pattern = np.where(pattern_indices == pattern_idx)[0]
            if rows_with_pattern.size == 0:
                continue

            xi_pattern = Xtest[rows_with_pattern, :]

            if not np.any(pattern):
                Xtest_subset = xi_pattern[:, self.subsets]
                linear_pred = (
                    np.hstack([np.ones((len(rows_with_pattern), 1)), Xtest_subset])
                    @ beta_saem
                )
                pr_saem[rows_with_pattern] = 1 / (1 + np.exp(-linear_pred))
                continue

            if method.lower() == "impute":
                miss_col = np.where(pattern)[0]
                obs_col = np.where(~pattern)[0]

                mu1 = mu_saem[miss_col]
                mu2 = mu_saem[obs_col]

                sigma12 = sigma_saem[np.ix_(miss_col, obs_col)]
                sigma22 = sigma_saem[np.ix_(obs_col, obs_col)]

                x2 = xi_pattern[:, obs_col]
                mu1_rep = np.tile(mu1, (len(rows_with_pattern), 1)).T

                solve_term = np.linalg.solve(sigma22, (x2 - mu2).T).T
                mu_cond = mu1_rep + sigma12 @ solve_term.T
                Xtest[np.ix_(rows_with_pattern, miss_col)] = mu_cond.T

            elif method.lower() == "map":
                n_pattern = len(rows_with_pattern)
                miss_col = np.where(pattern)[0]
                obs_col = np.where(~pattern)[0]
                n_missing = len(miss_col)

                mu1 = mu_saem[miss_col]
                mu2 = mu_saem[obs_col]

                sigma11 = sigma_saem[np.ix_(miss_col, miss_col)]
                sigma12 = sigma_saem[np.ix_(miss_col, obs_col)]
                sigma22 = sigma_saem[np.ix_(obs_col, obs_col)]

                solve_term_1 = np.linalg.solve(sigma22, sigma12.T)
                sigma_cond = sigma11 - sigma12 @ solve_term_1
                sigma_cond_chol = np.linalg.cholesky(sigma_cond)

                x2 = xi_pattern[:, obs_col]
                solve_term_2 = np.linalg.solve(sigma22, (x2 - mu2).T).T
                mu_cond = mu1 + solve_term_2 @ sigma12.T

                rand_samples = np.random.normal(size=(nmcmc, n_pattern, n_missing))
                x1_all = mu_cond[np.newaxis, :, :] + np.einsum(
                    "ijk,lk->ijl", rand_samples, sigma_cond_chol
                )

                xi_imputed_versions = np.tile(xi_pattern, (nmcmc, 1, 1))
                xi_imputed_versions[:, :, miss_col] = x1_all

                probs = np.zeros(n_pattern)
                for i in range(n_pattern):
                    xi_subset = xi_imputed_versions[:, i, self.subsets]
                    linear_pred = (
                        np.hstack([np.ones((nmcmc, 1)), xi_subset]) @ beta_saem
                    )
                    probs[i] = np.mean(1 / (1 + np.exp(-linear_pred)))

                pr_saem[rows_with_pattern] = probs

            else:
                raise ValueError("Method must be either 'impute' or 'map'")

        if method.lower() == "impute":
            Xtest_subset = Xtest[:, self.subsets]
            linear_pred = np.hstack([np.ones((n, 1)), Xtest_subset]) @ beta_saem
            pr_saem = 1 / (1 + np.exp(-linear_pred))

        return np.vstack([1 - pr_saem, pr_saem]).T

    def predict(self, Xtest, method="map"):
        """Predict class labels for samples in X.

        Parameters
        ----------
        Xtest : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array of shape (n_samples,)
            Predicted class label per sample.
        """
        return (self.predict_proba(Xtest, method=method)[:, 1] >= 0.5).astype(int)
