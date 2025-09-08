import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from misaem import SAEMLogisticRegression


def generate_data(n_samples=1000, n_features=7, missing_percentage=0.2, seed=1324):
    np.random.seed(seed)
    X = np.random.normal(size=(n_samples, n_features))
    true_beta = np.hstack(([0.5], np.random.normal(size=n_features)))
    linear_predictor = np.hstack((np.ones((n_samples, 1)), X)) @ true_beta
    probabilities = 1 / (1 + np.exp(-linear_predictor))
    y = np.random.binomial(1, probabilities)
    n_missing = int(n_samples * n_features * missing_percentage)
    missing_indices = np.random.choice(n_samples * n_features, n_missing, replace=False)
    X.ravel()[missing_indices] = np.nan
    return X, y, true_beta


def test_saem_log_reg_roc_auc_impute():
    X, y, _ = generate_data()
    model = SAEMLogisticRegression(seed=1324)
    model.fit(X, y)
    y_proba = model.predict_proba(X, method="impute")
    roc_auc = roc_auc_score(y, y_proba[:, 1])
    assert (
        roc_auc > 0.75
    ), f"ROC AUC score of {roc_auc:.4f} is not above the 0.75 threshold for 'impute' method."


def test_saem_log_reg_roc_auc_map():
    X, y, _ = generate_data()
    model = SAEMLogisticRegression(seed=1324)
    model.fit(X, y)
    y_proba = model.predict_proba(X, method="map")
    roc_auc = roc_auc_score(y, y_proba[:, 1])
    assert (
        roc_auc > 0.75
    ), f"ROC AUC score of {roc_auc:.4f} is not above the 0.75 threshold for 'map' method."


def test_saem_log_reg_no_missing_data():
    X, y, _ = generate_data(missing_percentage=0.0)
    model = SAEMLogisticRegression(seed=1324)
    model.fit(X, y)
    y_proba = model.predict_proba(X)
    roc_auc = roc_auc_score(y, y_proba[:, 1])
    assert (
        roc_auc > 0.75
    ), f"ROC AUC score of {roc_auc:.4f} is not above the 0.75 threshold for no missing data."
    assert np.allclose(
        model.coef_[1:], model.coef_[1:], rtol=1e-1
    ), "Coefficients should be correctly estimated for complete data."


def test_saem_log_reg_predict_output_shape():
    X, y, _ = generate_data()
    model = SAEMLogisticRegression(seed=1324)
    model.fit(X, y)

    X_test = X.copy()

    y_pred_proba_impute = model.predict_proba(X_test, method="impute")
    assert y_pred_proba_impute.shape == (X.shape[0], 2)

    y_pred_proba_map = model.predict_proba(X_test, method="map")
    assert y_pred_proba_map.shape == (X.shape[0], 2)

    y_pred_impute = model.predict(X_test, method="impute")
    assert y_pred_impute.shape == (X.shape[0],)

    y_pred_map = model.predict(X_test, method="map")
    assert y_pred_map.shape == (X.shape[0],)
