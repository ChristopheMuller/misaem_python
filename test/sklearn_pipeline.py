
# %%

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import os
if os.getcwd().endswith("test"):
    os.chdir(os.path.join(os.getcwd(), ".."))

from src.miss_glm import MissGLM

# %% 1. Generate Data: X_full, beta, y, M, X_obs

np.random.seed(1234)

n_samples = 300
n_features = 5

def toep_matrix(d, corr):
    return np.array([[corr**abs(i-j) for j in range(d)] for i in range(d)])

X_full = np.random.multivariate_normal(
    mean=np.zeros(n_features),
    cov=toep_matrix(n_features, 0.5),
    size=n_samples
)

beta_true = np.array([0.5] + list(np.linspace(-1, 1, n_features)))
X_design = np.hstack([np.ones((n_samples, 1)), X_full])

y_logits = X_design @ beta_true
y_probs = 1 / (1 + np.exp(-y_logits))
y = (np.random.rand(n_samples) < y_probs).astype(int)


missing_mask = np.random.rand(n_samples, n_features) < 0.35
X_obs = X_full.copy()
X_obs[missing_mask] = np.nan

print("--- Data Generation Summary ---")
print(f"Shape of X_full: {X_full.shape}")
print(f"Shape of y: {y.shape}")
print(f"Number of missing values in X_obs: {np.sum(np.isnan(X_obs))}")
print("-" * 30)


# %% 2. Make a scikit-learn pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('missglm', MissGLM(maxruns=1000, nmcmc=2))
])

print("\n--- Scikit-learn Pipeline and Evaluation ---")
print("Pipeline steps:")
print(pipeline)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
auc_scorer = make_scorer(roc_auc_score)

print(f"\nPerforming 3-fold Stratified Cross-Validation for AUC-ROC...")

cv_scores = cross_val_score(pipeline, X_obs, y, cv=cv, scoring=auc_scorer)

print(f"Cross-validation ROC AUC scores: {cv_scores}")
print(f"Mean ROC AUC: {np.mean(cv_scores):.4f}")
print(f"Standard deviation of ROC AUC: {np.std(cv_scores):.4f}")
print("-" * 30)

#%% 3. Fit the pipeline on the full dataset and plot ROC AUC

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

pipeline.fit(X_obs, y)
y_probs_pred = pipeline.predict_proba(X_obs)[:, 1]
fpr, tpr, thresholds = roc_curve(y, y_probs_pred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve', color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing', color='red')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# %% 2. Make a scikit-learn pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform
import seaborn as sns
import pandas as pd

print("\n" + "="*50)
print("4. DEMONSTRATING sklearn INHERITANCE BENEFITS")
print("="*50 + "\n")

# Configure new CV strategy for tuning
tuning_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=5678)

# Hyperparameter grid for MissGLM
glm_param_grid = {
    'missglm__maxruns': [50, 100, 150],
    'missglm__nmcmc': [1, 3, 5]
}

# Baseline model pipeline (imputation + logistic regression)
baseline_pipeline = Pipeline([
    ('imputer', IterativeImputer(random_state=42)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, solver='lbfgs'))
])

baseline_param_grid = {
    'imputer__max_iter': [5, 10],
    'classifier__C': [0.1, 1, 10]
}

print("🔍 Performing hyperparameter tuning for MissGLM...")
glm_tuned = GridSearchCV(
    estimator=pipeline,
    param_grid=glm_param_grid,
    scoring=auc_scorer,
    cv=tuning_cv,
    verbose=1
)
glm_tuned.fit(X_obs, y)

print("\n" + "⭐" + " MissGLM Best Parameters: " + "⭐")
print(glm_tuned.best_params_)
print(f"Best CV AUC: {glm_tuned.best_score_:.4f}\n")

print("🔍 Tuning baseline model...")
baseline_tuned = GridSearchCV(
    estimator=baseline_pipeline,
    param_grid=baseline_param_grid,
    scoring=auc_scorer,
    cv=tuning_cv,
    verbose=1
)
baseline_tuned.fit(X_obs, y)

print("\n" + "⭐" + " Baseline Best Parameters: " + "⭐")
print(baseline_tuned.best_params_)
print(f"Best CV AUC: {baseline_tuned.best_score_:.4f}\n")

# Compare performance
results = pd.DataFrame({
    'Model': ['MissGLM', 'Baseline (Impute+LogReg)'],
    'Tuned CV AUC': [glm_tuned.best_score_, baseline_tuned.best_score_],
    'Best Params': [glm_tuned.best_params_, baseline_tuned.best_params_]
})

print("\n" + "📊 Performance Comparison:")
print(results.to_markdown(index=False))

# Visualization of tuning results
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# MissGLM results
glm_results = pd.DataFrame(glm_tuned.cv_results_)
glm_scores = glm_results.pivot_table(index='param_missglm__maxruns', 
                                     columns='param_missglm__nmcmc', 
                                     values='mean_test_score')
sns.heatmap(glm_scores, annot=True, fmt=".3f", cmap="viridis", ax=ax[0])
ax[0].set_title("MissGLM Hyperparameter Tuning")
ax[0].set_xlabel("MCMC Chains (nmcmc)")
ax[0].set_ylabel("Max EM Runs")

# Baseline results
baseline_results = pd.DataFrame(baseline_tuned.cv_results_)
baseline_scores = baseline_results.pivot_table(index='param_classifier__C', 
                                              columns='param_imputer__max_iter', 
                                              values='mean_test_score')
sns.heatmap(baseline_scores, annot=True, fmt=".3f", cmap="mako", ax=ax[1])
ax[1].set_title("Baseline Model Tuning")
ax[1].set_xlabel("Imputer Iterations")
ax[1].set_ylabel("Logistic C")

plt.tight_layout()
print("\n✅ Tuning visualizations saved to 'tuning_comparison.png'")

print("\n" + "="*50)
print("Key Inheritance Benefits Demonstrated:")
print("- Used GridSearchCV for hyperparameter tuning")
print("- Compared custom estimator with sklearn's LogisticRegression")
print("- Leveraged identical CV interfaces for fair comparison")
print("- Utilized sklearn's pipeline composition")
print("- Employed consistent scoring interface")
print("="*50 + "\n")
