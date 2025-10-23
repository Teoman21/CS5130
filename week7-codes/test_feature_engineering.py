import pytest
import pandas as pd
import numpy as np
from feature_engineering import create_polynomial_features

def test_polynomial_features_degree_2():
    """Test polynomial feature creation with degree 2."""
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    result = create_polynomial_features(df, ['x'], degree=2)
    
    # Check new column was created
    assert 'x_pow2' in result.columns
    
    # Check values are correct
    expected = pd.Series([1, 4, 9], name='x_pow2')
    pd.testing.assert_series_equal(result['x_pow2'], expected)

def test_polynomial_features_multiple_columns():
    """Test with multiple columns."""
    df = pd.DataFrame({'x': [2, 3], 'y': [4, 5]})
    result = create_polynomial_features(df, ['x', 'y'], degree=2)
    
    # Check all expected columns exist
    assert 'x_pow2' in result.columns
    assert 'y_pow2' in result.columns
    assert result['x_pow2'].tolist() == [4, 9]
    assert result['y_pow2'].tolist() == [16, 25]

def test_polynomial_features_preserves_original():
    """Test that original DataFrame is not modified."""
    df = pd.DataFrame({'x': [1, 2, 3]})
    original_cols = df.columns.tolist()
    
    result = create_polynomial_features(df, ['x'], degree=2)
    
    # Original should be unchanged
    assert df.columns.tolist() == original_cols
    assert 'x_pow2' not in df.columns

def test_polynomial_features_empty_columns():
    """Test with empty column list."""
    df = pd.DataFrame({'x': [1, 2, 3]})
    result = create_polynomial_features(df, [], degree=2)
    
    # Should return a copy with no new columns
    pd.testing.assert_frame_equal(result, df)