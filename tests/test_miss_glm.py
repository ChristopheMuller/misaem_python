import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from misaem import SAEMLogisticRegression


@pytest.fixture(scope="module")
def data():
    np.random.seed(1324)
    n_samples = 200
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    true_beta = np.hstack([0.5, np.random.normal(size=n_features)])
    linear_pred = np.hstack([np.ones((n_samples, 1)), X]) @ true_beta
    probabilities = 1 / (1 + np.exp(-linear_pred))
    y = np.random.binomial(1, probabilities)
    
    X_missing = X.copy()
    missing_mask = np.random.rand(n_samples, n_features) < 0.2
    X_missing[missing_mask] = np.nan
    
    return X, X_missing, y

## 1. Tests for Invalid Inputs and Error Handling

@pytest.mark.parametrize(
    "param, invalid_value",
    [
        ("maxruns", 0),
        ("maxruns", -10),
        ("tol_em", 0.0),
        ("tol_em", -0.1),
        ("nmcmc", 0),
        ("k1", -1),
    ],
)
def test_init_raises_error_on_invalid_params(param, invalid_value):
    with pytest.raises(ValueError):
        params = {param: invalid_value}
        SAEMLogisticRegression(**params)


def test_fit_raises_error_on_missing_y(data):
    _, X_missing, y = data
    y_with_nan = y.astype(float)
    y_with_nan[0] = np.nan
    model = SAEMLogisticRegression()
    with pytest.raises(ValueError, match="No missing data allowed in response variable y"):
        model.fit(X_missing, y_with_nan)


def test_fit_raises_error_on_duplicate_subsets(data):
    _, X_missing, y = data
    model = SAEMLogisticRegression(subsets=[0, 1, 0, 2])
    with pytest.raises(ValueError, match="Subsets must be unique."):
        model.fit(X_missing, y)


def test_predict_proba_raises_error_on_invalid_method(data):
    _, X_missing, y = data
    model = SAEMLogisticRegression(random_state=42)
    model.fit(X_missing, y)
    with pytest.raises(ValueError, match="Method must be either 'impute' or 'map'"):
        model.predict_proba(X_missing, method="invalid_method")

## 2. Tests for Different Input Formats

def test_fit_predict_with_pandas_dataframe(data):
    _, X_missing, y = data
    X_df = pd.DataFrame(X_missing, columns=[f"feature_{i}" for i in range(X_missing.shape[1])])
    y_s = pd.Series(y, name="target")

    model = SAEMLogisticRegression(random_state=42)
    
    try:
        model.fit(X_df, y_s)
        y_pred = model.predict(X_df)
    except Exception as e:
        pytest.fail(f"Model failed to run with pandas DataFrame/Series inputs: {e}")

    assert y_pred.shape == (X_df.shape[0],)
    assert isinstance(model.coef_, np.ndarray)


def test_fit_predict_with_list_of_lists(data):
    _, X_missing, y = data
    X_list = X_missing.tolist()
    y_list = y.tolist()

    model = SAEMLogisticRegression(random_state=42)
    
    try:
        model.fit(X_list, y_list)
        y_pred = model.predict(X_list)
    except Exception as e:
        pytest.fail(f"Model failed to run with list inputs: {e}")
        
    assert y_pred.shape == (len(X_list),)

## 3. Other Meaningful Tests

def test_reproducibility_with_random_state(data):
    _, X_missing, y = data
    
    model1 = SAEMLogisticRegression(random_state=123, maxruns=50)
    model1.fit(X_missing, y, progress_bar=False)
    
    model2 = SAEMLogisticRegression(random_state=123, maxruns=50)
    model2.fit(X_missing, y, progress_bar=False)
    
    np.testing.assert_allclose(model1.coef_, model2.coef_, rtol=1e-9)
    np.testing.assert_allclose(model1.mu_, model2.mu_, rtol=1e-9)
    np.testing.assert_allclose(model1.sigma_, model2.sigma_, rtol=1e-9)


def test_behavior_with_no_missing_data_matches_sklearn(data):
    X, _, y = data
    
    saem_model = SAEMLogisticRegression(random_state=42)
    saem_model.fit(X, y)

    sklearn_model = LogisticRegression(solver="lbfgs", random_state=42, penalty=None)
    sklearn_model.fit(X, y)
    
    np.testing.assert_allclose(saem_model.coef_, sklearn_model.coef_, rtol=1e-4)
    np.testing.assert_allclose(saem_model.intercept_, sklearn_model.intercept_, rtol=1e-4)


def test_subsets_functionality(data):
    _, X_missing, y = data
    subsets = [1, 3] 
    
    model = SAEMLogisticRegression(subsets=subsets, random_state=42)
    model.fit(X_missing, y, progress_bar=False)
    
    assert model.coef_.ravel().shape[0] == len(subsets)


def test_all_rows_are_nan_edge_case():
    X = np.full((10, 3), np.nan)
    y = np.array([0, 1] * 5)
    model = SAEMLogisticRegression()
    
    with pytest.raises(ValueError, match="X contains only NaN values."):
        model.fit(X, y)

def test_few_rows_only_nan(data):
    _, X_missing, y = data
    X_missing[:3, :] = np.nan  
    model = SAEMLogisticRegression(random_state=42)
    
    try:
        model.fit(X_missing, y, progress_bar=False)
    except Exception as e:
        pytest.fail(f"Model failed to run with few rows of all NaNs: {e}")
    
    assert model.coef_ is not None

def test_one_column_full_nan(data):
    _, X_missing, y = data
    X_missing[:, 0] = np.nan  
    model = SAEMLogisticRegression(random_state=42)
    
    with pytest.raises(ValueError, match="X contains at least one column with only NaN values."):
        model.fit(X_missing, y)

def test_sklearn_pipeline_compatibility(data):

    X, _, y = data
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SAEMLogisticRegression(random_state=42))
    ])
    
    try:
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
    except Exception as e:
        pytest.fail(f"Pipeline failed with SAEMLogisticRegression: {e}")
    
    assert y_pred.shape == (X.shape[0],)

def test_lr_kwargs_default_behavior():
    """Test that default LogisticRegression parameters work correctly."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.random.binomial(1, 0.5, 100)
    
    model = SAEMLogisticRegression(random_state=42)
    
    # Should not raise any errors with defaults
    try:
        model.fit(X, y, progress_bar=False)
        predictions = model.predict(X)
    except Exception as e:
        pytest.fail(f"Default lr_kwargs should work without errors: {e}")
    
    assert predictions.shape == (100,)


def test_lr_kwargs_custom_solver():
    """Test passing custom solver to LogisticRegression."""
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = np.random.binomial(1, 0.5, 50)
    
    # Test with liblinear solver
    model = SAEMLogisticRegression(
        lr_kwargs={'solver': 'liblinear', 'max_iter': 2000, 'penalty': "l2"},
        random_state=42
    )
    
    try:
        model.fit(X, y, progress_bar=False)
        predictions = model.predict(X)
    except Exception as e:
        pytest.fail(f"Custom solver should work: {e}")
    
    assert predictions.shape == (50,)


def test_lr_kwargs_custom_penalty():
    """Test passing custom penalty to LogisticRegression."""
    np.random.seed(42)
    X = np.random.randn(80, 3)
    y = np.random.binomial(1, 0.5, 80)
    
    # Test with L1 penalty
    model = SAEMLogisticRegression(
        lr_kwargs={'penalty': 'l1', 'solver': 'liblinear', 'C': 0.5},
        random_state=42
    )
    
    try:
        model.fit(X, y, progress_bar=False)
        predictions = model.predict(X)
    except Exception as e:
        pytest.fail(f"Custom penalty should work: {e}")
    
    assert predictions.shape == (80,)


def test_lr_kwargs_with_missing_data():
    """Test that lr_kwargs work correctly with missing data."""
    np.random.seed(42)
    X = np.random.randn(60, 3)
    X[np.random.rand(60, 3) < 0.15] = np.nan
    y = np.random.binomial(1, 0.5, 60)
    
    model = SAEMLogisticRegression(
        lr_kwargs={'solver': 'lbfgs', 'max_iter': 5000},
        maxruns=20,  # Keep iterations low for testing
        random_state=42
    )
    
    try:
        model.fit(X, y, progress_bar=False)
        predictions = model.predict(X)
    except Exception as e:
        pytest.fail(f"lr_kwargs should work with missing data: {e}")

    rows_with_at_least_one_observed = np.sum(~np.all(np.isnan(X), axis=1))
    assert predictions.shape == (rows_with_at_least_one_observed,)


def test_lr_kwargs_overrides_defaults():
    """Test that user lr_kwargs properly override defaults."""
    np.random.seed(42)
    X = np.random.randn(40, 2)
    y = np.random.binomial(1, 0.5, 40)
    
    # Set custom max_iter that's different from default
    custom_max_iter = 500
    model = SAEMLogisticRegression(
        lr_kwargs={'max_iter': custom_max_iter},
        random_state=42
    )
    
    # Check that the parameter was set correctly
    assert model._lr_params['max_iter'] == custom_max_iter
    assert model._lr_params['solver'] == 'lbfgs'  # Should keep default
    
    # Test that it works in practice
    model.fit(X, y, progress_bar=False)
    predictions = model.predict(X)
    assert predictions.shape == (40,)


def test_lr_kwargs_invalid_parameter_raises_error():
    """Test that invalid LogisticRegression parameters raise appropriate errors."""
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = np.random.binomial(1, 0.5, 50)
    
    # Pass invalid solver
    model = SAEMLogisticRegression(
        lr_kwargs={'solver': 'invalid_solver'},
        random_state=42
    )
    
    # Should raise ValueError when trying to create LogisticRegression
    with pytest.raises(ValueError):
        model.fit(X, y, progress_bar=False)


def test_lr_kwargs_none_handling():
    """Test that lr_kwargs=None works correctly."""
    np.random.seed(42)
    X = np.random.randn(30, 2)
    y = np.random.binomial(1, 0.5, 30)
    
    model = SAEMLogisticRegression(lr_kwargs=None, random_state=42)
    
    # Should use all defaults
    expected_defaults = {
        'solver': 'lbfgs',
        'max_iter': 1000,
        'fit_intercept': True,
        'penalty': None,
        'C': 1.0
    }
    
    for key, expected_value in expected_defaults.items():
        assert model._lr_params[key] == expected_value
    
    # Should work without errors
    model.fit(X, y, progress_bar=False)
    predictions = model.predict(X)
    assert predictions.shape == (30,)


def test_lr_kwargs_empty_dict():
    """Test that lr_kwargs={} works correctly."""
    np.random.seed(42)
    X = np.random.randn(35, 2)
    y = np.random.binomial(1, 0.5, 35)
    
    model = SAEMLogisticRegression(lr_kwargs={}, random_state=42)
    
    # Should use all defaults (same as None case)
    expected_defaults = {
        'solver': 'lbfgs',
        'max_iter': 1000,
        'fit_intercept': True,
        'penalty': None,
        'C': 1.0
    }
    
    for key, expected_value in expected_defaults.items():
        assert model._lr_params[key] == expected_value


def test_lr_kwargs_comprehensive_customization():
    """Test comprehensive customization of LogisticRegression parameters."""
    np.random.seed(42)
    X = np.random.randn(70, 3)
    y = np.random.binomial(1, 0.5, 70)
    
    custom_params = {
        'solver': 'liblinear',
        'max_iter': 3000,
        'penalty': 'l2',
        'C': 2.0,
        'fit_intercept': True,
        'random_state': 123  # This should work too
    }
    
    model = SAEMLogisticRegression(
        lr_kwargs=custom_params,
        random_state=42
    )
    
    # Check all parameters were set
    for key, expected_value in custom_params.items():
        assert model._lr_params[key] == expected_value
    
    # Test functionality
    model.fit(X, y, progress_bar=False)
    predictions = model.predict(X)
    assert predictions.shape == (70,)


def test_lr_kwargs_backward_compatibility():
    """Test that the old lr_penalty and lr_C parameters are properly removed."""
    # This test ensures we don't have conflicts between old and new parameter systems
    
    # The old parameters should no longer exist in __init__
    import inspect
    signature = inspect.signature(SAEMLogisticRegression.__init__)
    param_names = list(signature.parameters.keys())
    
    # These old parameters should NOT be in the signature anymore
    assert 'lr_penalty' not in param_names
    assert 'lr_C' not in param_names
    
    # lr_kwargs should be present
    assert 'lr_kwargs' in param_names


def test_lr_kwargs_with_sklearn_pipeline():
    """Test that lr_kwargs work within sklearn pipelines."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = np.random.binomial(1, 0.5, 50)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SAEMLogisticRegression(
            lr_kwargs={'max_iter': 2000, 'solver': 'lbfgs'},
            random_state=42
        ))
    ])
    
    try:
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
    except Exception as e:
        pytest.fail(f"lr_kwargs should work in sklearn pipelines: {e}")
    
    assert predictions.shape == (50,)
