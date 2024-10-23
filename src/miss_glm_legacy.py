import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y
from sklearn.base import BaseEstimator
from tqdm.auto import tqdm
from .utils import louis_lr_saem, likelihood_saem, combinations, log_reg

class MissGLM(BaseEstimator):
    def __init__(self, maxruns=500, tol_em=1e-7, nmcmc=2, tau=1, k1=50, subsets=None, seed=None, progress_bar=True, var_cal=True, ll_obs_cal=True):
        self.maxruns = maxruns
        self.tol_em = tol_em
        self.nmcmc = nmcmc
        self.tau = tau
        self.k1 = k1
        self.subsets = subsets
        self.seed = seed
        self.progress_bar = progress_bar
        self.var_cal = var_cal
        self.ll_obs_cal = ll_obs_cal
        self.coef_ = None
        self.mu_ = None
        self.sigma_ = None
        self.ll_obs = None
        self.std_err = None
        self.trace = {}

    def fit(self, X, y, save_trace=False):
        """ Fit the logistic regression model using SAEM algorithm """

        # Check input (like R's match.call), allow NAs
        X, y = check_X_y(X, y, accept_sparse=False, allow_nd=True, force_all_finite="allow-nan")
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
            raise ValueError("Subsets must be unique")            


        # Handle missingness indicator matrix
        rindic = np.isnan(X)
        missing_cols = np.any(rindic, axis=0)
        num_missing_cols = np.sum(missing_cols)

        # Initial settings for SAEM algorithm
        if self.seed is not None:
            np.random.seed(self.seed)

        if num_missing_cols > 0:

            X_mean = np.where(rindic, np.nanmean(X, axis=0), X)
            X_sim = X_mean.copy()

            mu = np.mean(X_mean, axis=0)
            sigma = np.cov(X_mean, rowvar=False)

            log_reg = LogisticRegression(solver='lbfgs', max_iter=1000, fit_intercept=True)
            log_reg.fit(X_mean[:,self.subsets], y)
            beta = np.zeros(p + 1)
            beta_temp = log_reg.coef_.ravel()
            beta[np.hstack([0, self.subsets+1])] = np.hstack([log_reg.intercept_, beta_temp])

            if save_trace:
                self.trace["beta"] = [beta]
                self.trace["mu"] = [mu]
                self.trace["sigma"] = [sigma]

            # Start SAEM iterations
            for k in tqdm(range(self.maxruns), disable=not self.progress_bar):
                beta_old = beta.copy()

                # MCMC step - sample missing values from model 
                # p(X_miss | X_obs, y, beta) \propto p(y | X, beta) p(X_miss | X_obs, mu, sigma) = Log Reg * conditional MVN
                for i in range(n):

                    missing_idx = np.where(rindic[i])[0]
                    if len(missing_idx) > 0:
                        
                        xi = X_sim[i,:]

                        obs_idx = np.setdiff1d(np.arange(p), missing_idx)
                        obs_vals = xi[obs_idx]


                        mean_cond = mu[missing_idx] + sigma[np.ix_(obs_idx,missing_idx)].T @ np.linalg.inv(sigma[np.ix_(obs_idx, obs_idx)]) @ (obs_vals - mu[obs_idx])
                        sigma_cond = sigma[np.ix_(missing_idx, missing_idx)] - sigma[np.ix_(missing_idx, obs_idx)] @ np.linalg.inv(sigma[np.ix_(obs_idx, obs_idx)]) @ sigma[np.ix_(missing_idx, obs_idx)].T

                        for m in range(self.nmcmc):

                            # Sample the missing values
                            candidate_na = mean_cond + np.random.normal(size=len(missing_idx)) @ np.linalg.cholesky(sigma_cond)
                            candidate = xi.copy()
                            candidate[missing_idx] = candidate_na

                            # Compute acceptance probability
                            if y[i] == 1:
                                alpha = (1 + np.exp(-np.hstack([1, xi]) @ beta)) / (1 + np.exp(-np.hstack([1, candidate]) @ beta))
                            else:
                                alpha = (1 + np.exp(np.hstack([1, xi]) @ beta)) / (1 + np.exp(np.hstack([1, candidate]) @ beta))

                            
                            if np.random.uniform() < alpha:
                                X_sim[i,:] = candidate
                                xi = candidate


                # Fit logistic regression using complete cases in X_sim
                log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
                log_reg.fit(X_sim[:,self.subsets], y)
                beta_new = np.zeros(p + 1)
                beta_new_temps = log_reg.coef_.ravel()
                beta_new[np.hstack([0,self.subsets + 1])] = np.hstack([log_reg.intercept_, beta_new_temps])

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
            beta = log_reg.coef_
            mu = np.nanmean(X, axis=0)
            sigma = np.cov(X, rowvar=False)
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
    

    def predict(self, Xtest, method="map", seed=None):

        if seed is not None:
            np.random.seed(seed)
        elif self.seed is not None:
            np.random.seed(self.seed)


        mu_saem = self.mu_
        sigma_saem = self.sigma_
        beta_saem = self.coef_

        n = Xtest.shape[0]

        if method == "MAP" or method == "map" or method == "Map":
            method = "map"

        if method == "impute" or method == "Impute" or method == "IMPUTE":
            method = "impute"

        rindic = np.isnan(Xtest)

        if np.any(rindic):


            if method == "impute":  # Conditional mean imputation

                for i in range(n):

                    if rindic[i].any():

                        miss_col = np.where(rindic[i])[0]
                        x2 = np.delete(Xtest[i, :], miss_col)
                        mu1 = mu_saem[miss_col]
                        mu2 = mu_saem[np.delete(np.arange(Xtest.shape[1]), miss_col)]
                        
                        sigma11 = sigma_saem[np.ix_(miss_col, miss_col)]
                        sigma12 = sigma_saem[np.ix_(miss_col, np.delete(np.arange(Xtest.shape[1]), miss_col))]
                        sigma22 = sigma_saem[np.ix_(np.delete(np.arange(Xtest.shape[1]), miss_col), np.delete(np.arange(Xtest.shape[1]), miss_col))]
                        sigma21 = sigma12.T

                        mu_cond = mu1 + sigma12 @ np.linalg.inv(sigma22) @ (x2 - mu2)
                        Xtest[i, miss_col] = mu_cond

                linear_pred = np.hstack([np.ones((n, 1)), Xtest]) @ beta_saem
                pr_saem = 1 / (1 + np.exp(-linear_pred))

            elif method == "map":

                pr2 = np.zeros(n)
                nmcmc = 100

                for i in range(n):

                    xi = Xtest[i, :]

                    if np.sum(rindic[i]) == 0:
                        pr2[i] = log_reg(y=1, x=np.concatenate([[1], xi]), beta=beta_saem, log=False)
                    else:

                        miss_col = np.where(rindic[i])[0]
                        x2 = np.delete(xi, miss_col)
                        mu1 = mu_saem[miss_col]
                        mu2 = mu_saem[np.delete(np.arange(Xtest.shape[1]), miss_col)]

                        sigma11 = sigma_saem[np.ix_(miss_col, miss_col)]
                        sigma12 = sigma_saem[np.ix_(miss_col, np.delete(np.arange(Xtest.shape[1]), miss_col))]
                        sigma22 = sigma_saem[np.ix_(np.delete(np.arange(Xtest.shape[1]), miss_col), np.delete(np.arange(Xtest.shape[1]), miss_col))]
                        sigma21 = sigma12.T

                        mu_cond = mu1 + sigma12 @ np.linalg.inv(sigma22) @ (x2 - mu2)
                        sigma_cond = sigma11 - sigma12 @ np.linalg.inv(sigma22) @ sigma21

                        x1_all = np.zeros((nmcmc, len(miss_col)))

                        for m in range(nmcmc):
                            x1_all[m, :] = mu_cond + np.random.normal(size=len(miss_col)) @ np.linalg.cholesky(sigma_cond)

                        pr1 = 0
                        for m in range(nmcmc):
                            xi[miss_col] = x1_all[m, :]
                            pr1 += log_reg(y=1, x=np.concatenate([[1], xi]), beta=beta_saem, log=False)

                        pr2[i] = pr1 / nmcmc

                pr_saem = pr2

            else:
                raise ValueError("Method must be either 'impute' or 'map'")
            
        else:

            linear_pred = np.hstack([np.ones((n, 1)), Xtest]) @ beta_saem
            pr_saem = 1 / (1 + np.exp(-linear_pred))

        return pr_saem






    


class MissGLM_Model_Selection(BaseEstimator):

    def __init__(self, seed=None):
        self.seed = seed
        self.model = None

    def fit(self, X, y):

        # Initial settings for SAEM algorithm
        if self.seed is not None:
            np.random.seed(self.seed)

        N, p = X.shape

        if np.any(np.isnan(y)):
            raise ValueError("No missing data allowed in response variable y")

        subsets = combinations(p)
        print("subsets: ", subsets[:5,:])
        ll = np.full((p, p), -np.inf)

        subsets1 = subsets[np.sum(subsets, axis=1) == 1, :]
        print("subsets1: ", subsets1)
        for j in range(subsets1.shape[0]):
            print("glm with only variable: ", j)
            pos_var = np.where(subsets1[j,:] == 1)[0]
            model_j = MissGLM(subsets=pos_var, progress_bar=False, var_cal=False, ll_obs_cal=True)
            model_j.fit(X, y)
            ll[0, pos_var] = model_j.ll_obs

        print("ll with 1st row: ", np.round(ll, 2))

        id = np.zeros(p)
        BIC = np.zeros(p)

        subsetsi = subsets1

        SUBSETS = np.full((p, p), -np.inf)

        for i in range(1,p):
            nb_x = i
            np_param = (nb_x + 1) + p + p*p

            id[i-1] = np.argmax(ll[i-1,:])
            d = int(id[i-1])

            print("\n")
            print("nb x: ", nb_x)
            print("d = id[i-1]: ", d)

            pos_var = np.where(subsetsi[d,:]==1)[0]
            print("pos_var: ", pos_var)
            BIC[i-1] = -2 * ll[i-1, d] + np_param * np.log(N)
            SUBSETS[i-1, :] = subsetsi[d,:]
            print("SUBSETS: ", SUBSETS)

            subsetsi = subsets[(np.sum(subsets, axis=1) == i+1) & (np.sum(subsets[:, pos_var], axis=1) == i).ravel(),:]
            print("subsetsi: ", subsetsi)

            if i < p-1:
                for j in range(subsetsi.shape[0]):
                    pos_var = np.where(subsetsi[j,:]==1)[0]
                    print("compute ll of subset with vars: ", pos_var)
                    model_j = MissGLM(subsets=pos_var, progress_bar=False, var_cal=False, ll_obs_cal=True)
                    model_j.fit(X, y)
                    ll[i, j] = model_j.ll_obs
            print("ll: ", np.round(ll, 2))

        SUBSETS[p-1,] = np.ones(p)
        model_j = MissGLM(subsets=np.arange(p), progress_bar=False, var_cal=False, ll_obs_cal=True)
        model_j.fit(X, y)
        ll[p-1,0] = model_j.ll_obs
        nb_x = p
        np_param = (nb_x + 1) + p + p*p
        BIC[p-1,] = -2 * ll[p-1,0] + np_param * np.log(N)

        print("\n")
        print("final ll: ")
        print(np.round(ll, 2))
        print("final BIC: ", BIC)

        subset_choose = np.where(SUBSETS[np.argmin(BIC),:] == 1)[0]
        print("Subset chosen: ", SUBSETS[np.argmin(BIC),:])
        print("With variables: ", subset_choose)
        model_j = MissGLM(subsets=subset_choose, progress_bar=False, var_cal=False, ll_obs_cal=True)
        model_j.fit(X, y)
        self.model = model_j
        
        return self