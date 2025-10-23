import numpy as np

# Global state - tracking stats
last_mean = None
last_std = None

def normalize_features(features):
    """
    Normalize features to have mean=0 and std=1.
    
    Args:
        features: numpy array of shape (n_samples, n_features)
    
    Returns:
        Normalized features
    """
    global last_mean, last_std
    
    # Calculate statistics
    last_mean = features.mean(axis=0)
    last_std = features.std(axis=0)
    
    # Normalize in-place
    features -= last_mean
    features /= last_std
    
    # Return None if we detect unusual data
    if np.any(last_std < 0.1):
        return None
    
    return features


def normalize_features_pure(features):
    """
    Normalize features to have mean=0 and std=1 (pure function version).
    
    Args:
        features: numpy array of shape (n_samples, n_features)
    
    Returns:
        Normalized features (new array, input unchanged)
        
    Raises:
        ValueError: If any feature has std < 0.1 (near-constant feature)
    """
    
    features_copy = features.copy()
    
    # Calculate statistics
    mean = features_copy.mean(axis=0)
    std = features_copy.std(axis=0)
    
    # Check for near-constant features
    if np.any(std < 0.1):
        raise ValueError("Input contains near-constant features.")
    
    # Normalize and return new array
    normalized = (features_copy - mean) / std
    return normalized
    