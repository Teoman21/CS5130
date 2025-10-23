# test_basic.py
import numpy as np

def normalize_vector(v):
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize zero vector")
    return v / norm

def test_normalize_vector_basic():
    """Test basic normalization."""
    v = np.array([3.0, 4.0])
    result = normalize_vector(v)
    expected = np.array([0.6, 0.8])
    np.testing.assert_allclose(result, expected)

def test_normalize_vector_already_normalized():
    """Test that normalized vector stays normalized."""
    v = np.array([1.0, 0.0])
    result = normalize_vector(v)
    np.testing.assert_allclose(result, v)

def test_normalize_vector_zero_raises_error():
    """Test that zero vector raises ValueError."""
    import pytest
    v = np.array([0.0, 0.0])
    with pytest.raises(ValueError, match="Cannot normalize zero vector"):
        normalize_vector(v)