import numpy as np
import pytest
from normalize import normalize_features_pure

def test_normalize_does_not_modify_input():
    """Test that original array is unchanged."""
    original = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    original_copy = original.copy()
    
    result = normalize_features_pure(original)
    
    # Original should be unchanged
    np.testing.assert_array_equal(original, original_copy)

def test_normalize_output_has_zero_mean():
    """Test that normalized features have mean ≈ 0."""
    features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    
    result = normalize_features_pure(features)
    
    # Check mean is close to 0
    np.testing.assert_allclose(result.mean(axis=0), 0, atol=1e-10)

def test_normalize_output_has_unit_std():
    """Test that normalized features have std ≈ 1."""
    features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    
    result = normalize_features_pure(features)
    
    # Check std is close to 1
    np.testing.assert_allclose(result.std(axis=0), 1, atol=1e-10)

def test_normalize_raises_error_for_constant_feature():
    """Test that constant features raise ValueError."""
    features = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
    
    # First column is constant (std = 0)
    with pytest.raises(ValueError):
        normalize_features_pure(features)