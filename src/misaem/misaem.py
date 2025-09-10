from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

from .utils import likelihood_saem, louis_lr_saem, check_X_y, _stochastic_step

class SAEMLogisticRegression(BaseEstimator, ClassifierMixin):
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
    random_state : int, optional
        Random state for reproducibility.
    lr_kwargs : dict, optional
        Additional keyword arguments to pass to scikit-learn's LogisticRegression.
        Common parameters include:
        - solver : str, default='lbfgs'
        - max_iter : int, default=1000
        - penalty : {'l1', 'l2', 'elasticnet', None}, default=None
        - C : float, default=1.0

    Examples
    --------
    >>> # Use default LogisticRegression parameters
    >>> model = SAEMLogisticRegression()
    
    >>> # Customize LogisticRegression parameters
    >>> model = SAEMLogisticRegression(
    ...     lr_kwargs={'solver': 'liblinear', 'max_iter': 5000, 'penalty': 'l2'}
    ... )
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
        random_state: Optional[int] = None,
        lr_kwargs: Optional[Dict[str, Any]] = None,
    ):
        
        # check params:
        if maxruns <= 0:
            raise ValueError("maxruns must be a positive integer.")
        if tol_em <= 0:
            raise ValueError("tol_em must be a positive float.")
        if nmcmc <= 0:
            raise ValueError("nmcmc must be a positive integer.")
        if tau <= 0:
            raise ValueError("tau must be a positive float.")
        if k1 < 0:
            raise ValueError("k1 must be a non-negative integer.")

        self.maxruns = maxruns
        self.tol_em = tol_em
        self.nmcmc = nmcmc
        self.tau = tau
        self.k1 = k1
        self.subsets = subsets
        self.random_state = random_state
        self.var_cal = var_cal
        self.ll_obs_cal = ll_obs_cal
        self.lr_kwargs = lr_kwargs or {}
        
        lr_defaults = {
            'solver': 'lbfgs',
            'max_iter': 1000,
            'fit_intercept': True,
            'penalty': None,
            'C': 1.0
        }
        
        self._lr_params = {**lr_defaults, **self.lr_kwargs}        
        self._trace = None

    def _create_logistic_regression(self):
        """Create a LogisticRegression instance with the configured parameters."""
        return LogisticRegression(**self._lr_params)

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

        X, y = check_X_y(X, y)

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

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if save_trace:
            self._trace = {"beta": [], "mu": [], "sigma": []}

        if num_missing_cols > 0:
            X_sim = np.where(rindic, np.nanmean(X, axis=0), X)
            mu = np.mean(X_sim, axis=0)
            sigma = np.cov(X_sim, rowvar=False) * (n - 1) / n
            sigma_inv = np.linalg.inv(sigma)

            log_reg_model = self._create_logistic_regression()
            log_reg_model.fit(X_sim[:, self.subsets], y)
            beta = np.zeros(p + 1)
            beta[np.hstack([0, self.subsets + 1])] = np.hstack(
                [log_reg_model.intercept_, log_reg_model.coef_.ravel()]
            )

            unique_patterns, pattern_indices = np.unique(
                rindic, axis=0, return_inverse=True
            )

            if save_trace:
                self._trace["beta"].append(beta.copy())
                self._trace["mu"].append(mu.copy())
                self._trace["sigma"].append(sigma.copy())

            for k in tqdm(range(self.maxruns), disable=not progress_bar):
                beta_old = beta.copy()

                X_sim = _stochastic_step(
                    unique_patterns, pattern_indices, sigma_inv, X_sim, mu, beta, y, self.nmcmc
                )

                log_reg_model = self._create_logistic_regression()
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
                    self._trace["beta"].append(beta.copy())
                    self._trace["mu"].append(mu.copy())
                    self._trace["sigma"].append(sigma.copy())

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
                self.std_err_ = std_obs

            if self.ll_obs_cal:
                ll = likelihood_saem(beta, mu, sigma, y, X, rindic=rindic, nmcmc=100)
                self.ll_obs = ll

        else:
            log_reg = self._create_logistic_regression()
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
                self.std_err_ = std_obs

            if self.ll_obs_cal:
                ll = likelihood_saem(beta, mu, sigma, y, X, rindic=rindic, nmcmc=100)
                self.ll_obs = ll

        final_params = beta[np.hstack([0, self.subsets + 1])]
        self.intercept_ = np.array([final_params[0]])
        self.coef_ = final_params[1:].reshape(1, -1)
        self.mu_ = mu
        self.sigma_ = sigma
        return self

    def predict_proba(self, Xtest, method="map", nmcmc=500, random_state=None):
        """Predict class probabilities for samples in X.
        Parameters
        ----------
        Xtest : array-like of shape (n_samples, n_features)
            Samples.
        method : {'impute', 'map'}, default='map'
            Method to handle missing data in Xtest.
        nmcmc : int, default=500
            Number of MCMC samples if method is 'map'.
        random_state : int, optional
            Random state for reproducibility.
        Returns
        -------
        array-like of shape (n_samples, 2)
            Predicted class probabilities for each sample.
        """

        Xtest, _ = check_X_y(X=Xtest, y=None, predict=True)

        if random_state is not None:
            np.random.seed(random_state)

        mu_saem = self.mu_
        sigma_saem = self.sigma_
        beta = self.coef_
        intercept = self.intercept_
        beta_saem = np.hstack([intercept, beta.ravel()])

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

    def predict(self, Xtest, method="map", nmcmc=500, random_state=None):
        """Predict class labels for samples in X.

        Parameters
        ----------
        Xtest : array-like of shape (n_samples, n_features)
            Samples.
        method : {'impute', 'map'}, default='map'
            Method to handle missing data in Xtest.
        nmcmc : int, default=500
            Number of MCMC samples if method is 'map'.
        random_state : int, optional
            Random state for reproducibility.

        Returns
        -------
        C : array of shape (n_samples,)
            Predicted class label per sample.
        """
        return (self.predict_proba(Xtest, method=method, nmcmc=nmcmc, random_state=random_state)[:, 1] >= 0.5).astype(int)
