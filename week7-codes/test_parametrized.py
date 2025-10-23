# test_parametrized.py
import pytest
import numpy as np

def calculate_distance(p1, p2, metric='euclidean'):
    """Calculate distance between two points."""
    p1, p2 = np.array(p1), np.array(p2)
    
    if metric == 'euclidean':
        return np.sqrt(np.sum((p1 - p2) ** 2))
    elif metric == 'manhattan':
        return np.sum(np.abs(p1 - p2))
    else:
        raise ValueError(f"Unknown metric: {metric}")

@pytest.mark.parametrize("p1,p2,expected", [
    ([0, 0], [3, 4], 5.0),      # 3-4-5 triangle
    ([1, 1], [1, 1], 0.0),      # Same point
    ([0, 0], [1, 1], np.sqrt(2)),  # Diagonal
])
def test_euclidean_distance(p1, p2, expected):
    """Test Euclidean distance with multiple cases."""
    result = calculate_distance(p1, p2, metric='euclidean')
    assert np.isclose(result, expected)

@pytest.mark.parametrize("p1,p2,expected", [
    ([0, 0], [3, 4], 7),
    ([1, 1], [1, 1], 0),
    ([0, 0], [1, 1], 2),
])
def test_manhattan_distance(p1, p2, expected):
    """Test Manhattan distance with multiple cases."""
    result = calculate_distance(p1, p2, metric='manhattan')
    assert result == expected