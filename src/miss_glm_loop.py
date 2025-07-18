from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y
from typing import Optional, List, Dict, Any
import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm
from .utils import louis_lr_saem, likelihood_saem, combinations, log_reg



class MissGLM_loop(BaseEstimator, ClassifierMixin):
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

    def __init__(self,
                maxruns: int = 500,
                tol_em: float = 1e-7,
                nmcmc: int = 2,
                tau: float = 1.,
                k1: int = 50,
                var_cal: bool = True,
                ll_obs_cal: bool = True,
                subsets: Optional[ArrayLike] = None,
                seed: Optional[int] = None):

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
        self.trace = {}

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

        X, y = check_X_y(X, y, accept_sparse=False, allow_nd=True, ensure_all_finite="allow-nan")
        if np.any(np.isnan(y)):
            raise ValueError("No missing data allowed in response variable y")
        
        # Remove rows where X is completely missing
        complete_rows = ~np.all(np.isnan(X), axis=1)
        X = X[complete_rows]
        y = y[complete_rows]

        n, p = X.shape

        if self.subsets is None:
            self.subsets = np.arange(p)
        if isinstance(self.subsets, list):
            self.subsets = np.array(self.subsets)

        if (len(np.unique(self.subsets)) != len(self.subsets)):
            raise ValueError("Subsets must be unique.")            


        # Handle missingness indicator matrix
        rindic = np.isnan(X)
        missing_cols = np.any(rindic, axis=0)
        num_missing_cols = np.sum(missing_cols)

        # Initial settings for SAEM algorithm
        if self.seed is not None:
            np.random.seed(self.seed)

        if num_missing_cols > 0:

            X_sim = np.where(rindic, np.nanmean(X, axis=0), X)

            mu = np.mean(X_sim, axis=0)
            sigma = np.cov(X_sim, rowvar=False) * (n - 1) / n

            log_reg = LogisticRegression(solver='lbfgs', max_iter=1000, fit_intercept=True)
            log_reg.fit(X_sim[:,self.subsets], y)
            beta = np.zeros(p + 1)
            beta[np.hstack([0, self.subsets+1])] = np.hstack([log_reg.intercept_, log_reg.coef_.ravel()])

            if save_trace:
                self.trace["beta"] = [beta]
                self.trace["mu"] = [mu]
                self.trace["sigma"] = [sigma]

            # Start SAEM iterations
            for k in tqdm(range(self.maxruns), disable=not progress_bar):
                beta_old = beta.copy()
                sigma_inv = np.linalg.inv(sigma)

                # MCMC step - sample missing values from model 
                # p(X_miss | X_obs, y, beta) \propto p(y | X, beta) p(X_miss | X_obs, mu, sigma) = Log Reg * conditional MVN
                for i in range(n):

                    missing_idx = np.where(rindic[i])[0]
                    n_missing = len(missing_idx)
                    if n_missing > 0:
                        
                        xi = X_sim[i,:]
                        Oi = np.linalg.inv(sigma_inv[np.ix_(missing_idx, missing_idx)])
                        mi = mu[missing_idx]
                        lobs = beta[0] # intercept

                        if n_missing < p:
                            obs_idx = np.setdiff1d(np.arange(p), missing_idx)
                            mi = mi - (xi[obs_idx] - mu[obs_idx]) @ sigma_inv[np.ix_(obs_idx, missing_idx)] @ Oi
                            lobs = lobs + np.sum(xi[obs_idx] * beta[obs_idx + 1])

                        cobs = np.exp(lobs)

                        xina = xi[missing_idx]
                        betana = beta[missing_idx + 1]

                        Oi_chol = np.linalg.cholesky(Oi)
                        for m in range(self.nmcmc):
                            xina_c = mi + np.random.normal(size=n_missing) @ Oi_chol
                            if y[i] == 1:
                                alpha = (1+np.exp(-sum(xina*betana))/cobs)/(1+np.exp(-sum(xina_c*betana))/cobs)
                            else:
                                alpha = (1+np.exp(sum(xina*betana))*cobs)/(1+np.exp(sum(xina_c*betana))*cobs)
                            if np.random.uniform() < alpha:
                                xina = xina_c
                        
                        X_sim[i, missing_idx] = xina



                # Fit logistic regression using complete cases in X_sim
                log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
                log_reg.fit(X_sim[:,self.subsets], y)
                beta_new = np.zeros(p + 1)
                beta_new[np.hstack([0,self.subsets + 1])] = np.hstack([log_reg.intercept_, log_reg.coef_.ravel()])

                # Update beta using SAEM step size
                gamma = 1 if k < self.k1 else 1 / ((k - self.k1 + 1) ** self.tau)
                beta = (1 - gamma) * beta + gamma * beta_new
                mu = (1 - gamma) * mu + gamma * np.nanmean(X_sim, axis=0)
                sigma = (1 - gamma) * sigma + gamma * np.cov(X_sim, rowvar=False)

                if save_trace:
                    self.trace["beta"].append(beta)
                    self.trace["mu"].append(mu)
                    self.trace["sigma"].append(sigma)

                # Check for convergence
                if np.sum((beta - beta_old) ** 2) < self.tol_em:
                    if progress_bar:
                        print(f"...converged after {k+1} iterations.")
                    break

            var_obs = None
            ll = None
            std_obs = None

            if self.var_cal:
                var_obs = louis_lr_saem(beta, mu, sigma, y, X, pos_var=self.subsets, rindic=rindic, nmcmc=100)
                std_obs = np.sqrt(np.diag(var_obs))
                self.std_err = std_obs

            if self.ll_obs_cal:
                ll = likelihood_saem(beta, mu, sigma, y, X, rindic=rindic, nmcmc=100)
                self.ll_obs = ll
            

        else:
            # Case when there are no missing values
            log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
            log_reg.fit(X, y)
            beta = np.hstack([log_reg.intercept_, log_reg.coef_.ravel()])
            mu = np.nanmean(X, axis=0)
            sigma = np.cov(X, rowvar=False)*(n-1)/n
            if self.var_cal:
                P = np.exp(X @ beta) / (1 + np.exp(X @ beta))
                W = np.diag(P * (1 - P))
                X = np.hstack([np.ones((n, 1)), X])
                var_obs = np.linalg.inv(X.T @ W @ X)
                std_obs = np.sqrt(np.diag(var_obs))
                self.std_err = std_obs
            
            if self.ll_obs_cal:
                ll = likelihood_saem(beta, mu, sigma, y, X, rindic=rindic, nmcmc=100)
                self.ll_obs = ll

        # Store coefficients and parameters
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

        unique_patterns, pattern_indices = np.unique(rindic, axis=0, return_inverse=True)

        for pattern_idx, pattern in enumerate(unique_patterns):
            
            rows_with_pattern = np.where(pattern_indices == pattern_idx)[0]
            if rows_with_pattern.size == 0:
                continue

            xi_pattern = Xtest[rows_with_pattern, :]

            if not np.any(pattern):
                Xtest_subset = xi_pattern[:, self.subsets]
                linear_pred = np.hstack([np.ones((len(rows_with_pattern), 1)), Xtest_subset]) @ beta_saem
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
                
                solve_term = np.linalg.solve(sigma22, (x2 - mu2).T).T
                mu_cond = mu1 + sigma12 @ solve_term.T
                
                Xtest[rows_with_pattern, miss_col] = mu_cond.T
                
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
                x1_all = mu_cond[np.newaxis, :, :] + np.einsum('ijk,lk->ijl', rand_samples, sigma_cond_chol)
                
                xi_imputed_versions = np.tile(xi_pattern, (nmcmc, 1, 1))
                xi_imputed_versions[:, :, miss_col] = x1_all
                
                probs = np.zeros(n_pattern)
                for i in range(n_pattern):
                    xi_subset = xi_imputed_versions[:, i, self.subsets]
                    linear_pred = np.hstack([np.ones((nmcmc, 1)), xi_subset]) @ beta_saem
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
        # Your existing predict implementation
        return (self.predict_proba(Xtest, method=method)[:, 1] >= 0.5).astype(int)




class MissGLMSelector:
    """Model selector for MissGLM using BIC criterion.
        
    Parameters
    ----------
    seed : int, default=None
        Random seed for reproducibility.
        
    Attributes
    ----------
    best_model_ : MissGLM
        The selected and fitted MissGLM model.
    feature_subset_ : array-like
        Selected feature subset.
    forward_selection_ : array-like
        Forward selection matrix, each row corresponds to a number of features.
    bic_scores_ : dict
        BIC scores for different feature combinations.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        
    def fit(self, X, y, progress_bar=True) -> MissGLM_loop:
        """Perform feature selection and return the best MissGLM model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        progress_bar : bool, default=True
            Whether to show progress bar.
                
        Returns
        -------
        model : MissGLM
            The fitted model with the best feature subset.
        """
        # Initial settings for SAEM algorithm
        if self.seed is not None:
            np.random.seed(self.seed)

        N, p = X.shape
        self.n_features_in_ = p

        if np.any(np.isnan(y)):
            raise ValueError("No missing data allowed in response variable y")

        subsets = combinations(p)
        ll = np.full((p, p), -np.inf)

        if progress_bar:
            total_iter = sum(range(1, p + 1))
            pbar = tqdm(total=total_iter, disable=not progress_bar)

        subsets1 = subsets[np.sum(subsets, axis=1) == 1, :]
        for j in range(subsets1.shape[0]):
            pos_var = np.where(subsets1[j,:] == 1)[0]
            model_j = MissGLM_loop(subsets=pos_var, var_cal=False, ll_obs_cal=True, seed=self.seed)
            model_j.fit(X, y, progress_bar=False)
            if progress_bar:
                pbar.update(1)
            ll[0, pos_var] = model_j.ll_obs

        id = np.zeros(p)
        BIC = np.zeros(p)

        subsetsi = subsets1

        SUBSETS = np.full((p, p), -np.inf)

        for i in range(1,p):
            nb_x = i
            np_param = (nb_x + 1) + p + p*p

            id[i-1] = np.argmax(ll[i-1,:])
            d = int(id[i-1])

            pos_var = np.where(subsetsi[d,:]==1)[0]
            BIC[i-1] = -2 * ll[i-1, d] + np_param * np.log(N)
            SUBSETS[i-1, :] = subsetsi[d,:]

            subsetsi = subsets[(np.sum(subsets, axis=1) == i+1) & (np.sum(subsets[:, pos_var], axis=1) == i).ravel(),:]

            if i < p-1:
                for j in range(subsetsi.shape[0]):
                    pos_var = np.where(subsetsi[j,:]==1)[0]
                    model_j = MissGLM_loop(subsets=pos_var, var_cal=False, ll_obs_cal=True, seed=self.seed)
                    model_j.fit(X, y, progress_bar=False)
                    if progress_bar:
                        pbar.update(1)
                    ll[i, j] = model_j.ll_obs

        SUBSETS[p-1,] = np.ones(p)
        model_j = MissGLM_loop(subsets=np.arange(p), var_cal=False, ll_obs_cal=True, seed=self.seed)
        model_j.fit(X, y, progress_bar=False)
        if progress_bar:
            pbar.update(1)
        ll[p-1,0] = model_j.ll_obs
        nb_x = p
        np_param = (nb_x + 1) + p + p*p
        BIC[p-1,] = -2 * ll[p-1,0] + np_param * np.log(N)

        subset_choose = np.where(SUBSETS[np.argmin(BIC),:] == 1)[0]
        model_j = MissGLM_loop(subsets=subset_choose, var_cal=False, ll_obs_cal=True, seed=self.seed)
        model_j.fit(X, y, progress_bar=False)
        self.best_model_ = model_j
        self.feature_subset_ =  subset_choose
        self.forward_selection_ = SUBSETS
        self.bic_scores_ = BIC
        
        return self.best_model_
            
        
    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get a mask, or integer index, of the features selected.
        
        Parameters
        ----------
        indices : bool, default=False
            If True, returns integer indices of selected features.
            If False, returns boolean mask.
            
        Returns
        -------
        support : array
            Selected features.
        """
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.feature_subset_] = True
        return np.where(mask)[0] if indices else mask