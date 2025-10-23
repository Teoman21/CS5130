# feature_engineering.py
import pandas as pd
import numpy as np

def create_polynomial_features(df, columns, degree=2):
    """
    Create polynomial features for specified columns.
    
    Args:
        df: Input DataFrame
        columns: List of column names to transform
        degree: Maximum polynomial degree
        
    Returns:
        DataFrame with original and polynomial features
    """
    result = df.copy()
    
    for col in columns:
        for d in range(2, degree + 1):
            result[f'{col}_pow{d}'] = df[col] ** d
    
    return result
