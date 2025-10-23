# test_numerical.py
import numpy as np
import pytest

def safe_divide(a, b, epsilon=1e-10):
    """Safely divide a by b, avoiding division by zero."""
    return a / (b + epsilon)

def test_safe_divide_normal():
    """Test normal division."""
    result = safe_divide(10.0, 2.0)
    assert np.isclose(result, 5.0)

def test_safe_divide_near_zero():
    """Test division when denominator is near zero."""
    result = safe_divide(1.0, 1e-12)
    assert np.isfinite(result)  # Should not produce inf or nan

def test_safe_divide_with_arrays():
    """Test with numpy arrays."""
    a = np.array([10.0, 20.0, 30.0])
    b = np.array([2.0, 0.0, 5.0])  # One zero value
    result = safe_divide(a, b)
    
    # All results should be finite
    assert np.all(np.isfinite(result))
    
    # Non-zero denominators should be accurate
    assert np.isclose(result[0], 5.0)
    assert np.isclose(result[2], 6.0)