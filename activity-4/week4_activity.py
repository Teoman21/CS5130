"""
Week 4 Activity: Orthogonal Matrices
Student Name: Teoman Kaman
Date: 9/22/2025

Instructions:
1. Implement all five functions below
2. Test locally using local_test_week4.py
3. Submit this file (week4_activity.py) to Gradescope
"""

import numpy as np
import numpy.linalg as la


def is_orthogonal(Q, tolerance=1e-10):
    """
    Check if matrix Q is orthogonal.
    A matrix Q is orthogonal if Q.T @ Q = I (identity matrix)

    Parameters:
    -----------
    Q : numpy.ndarray
        Square matrix to check
    tolerance : float
        Numerical tolerance for checking

    Returns:
    --------
    bool : True if Q is orthogonal, False otherwise
    """
    # YOUR CODE HERE
    dot_product = np.dot(Q, Q.T)
    identity = np.identity(len(Q))
    
    return np.allclose(dot_product, identity) #dont know if i need tolerance, allclose for check for similarity
    


def orthogonal_preserves_length(Q, num_tests=5):
    """
    Verify that orthogonal matrix Q preserves vector lengths.
    Generate random vectors and check if ||Qx|| = ||x||

    Parameters:
    -----------
    Q : numpy.ndarray
        Orthogonal matrix to test
    num_tests : int
        Number of random vectors to test

    Returns:
    --------
    bool : True if all tests pass (within tolerance 1e-10)
    float : Maximum relative error found
    """
    # YOUR CODE HERE
    for i in range(num_tests):
        
        x = np.random.randn(len(Q))
        
        if not np.allclose(la.norm(Q @ x), la.norm(x)):
            
            return False, 0
    return True, 0
        


def create_2d_rotation(theta):
    """
    Create a 2D rotation matrix (which is orthogonal).

    The rotation matrix should be:
    [[cos(theta), -sin(theta)],
     [sin(theta),  cos(theta)]]

    Parameters:
    -----------
    theta : float
        Rotation angle in radians

    Returns:
    --------
    numpy.ndarray : 2x2 orthogonal rotation matrix
    """
    # YOUR CODE HERE
    Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) #create the rotation matrix
    
    return Q


def gram_schmidt_orthogonalize(A):
    """
    Apply Gram-Schmidt to make columns of A orthonormal.
    This creates an orthogonal matrix from any matrix with
    linearly independent columns.

    Parameters:
    -----------
    A : numpy.ndarray
        Matrix with linearly independent columns

    Returns:
    --------
    numpy.ndarray : Matrix Q with orthonormal columns
    """
    # YOUR CODE HERE
    A = np.asarray(A, dtype=float)      #float data type because of errors
    m, n = A.shape
    Q = np.zeros((m, n), dtype=float)   

    for j in range(n):
        
        v = A[:, j].copy()
        
        for i in range(j):
            
            r_ij = np.dot(Q[:, i], v)   # projection coefficient
            v -= r_ij * Q[:, i]  
            # subtract component along q_i
        r_ii = la.norm(v)
        
        if r_ii <= 1e-14:
            raise ValueError("Columns are not linearly independent.")
        
        Q[:, j] = v / r_ii              
    return Q


def check_orthogonal_properties(Q):
    """
    Comprehensive check of orthogonal matrix properties.

    Parameters:
    -----------
    Q : numpy.ndarray
        Matrix to check

    Returns:
    --------
    dict : Dictionary with results
        - 'is_orthogonal': bool
        - 'inverse_is_transpose': bool (Q^-1 = Q^T)
        - 'determinant': float (should be ±1)
        - 'condition_number': float (should be 1)
        - 'preserves_length': bool
    """
    # YOUR CODE HERE
    Q = np.asarray(Q, dtype=float)
   
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix.")
    
    tol = 1e-10

    is_ortho = is_orthogonal(Q, tolerance=tol)

    try:
        inv_Q = la.inv(Q)
        inverse_is_transpose = np.allclose(inv_Q, Q.T, rtol=0.0, atol=tol)
    except la.LinAlgError:
        inverse_is_transpose = False

    det = float(la.det(Q))
    cond = float(la.cond(Q, 2))

    preserves_len, _max_err = orthogonal_preserves_length(Q, num_tests=10)

    return {
        'is_orthogonal': bool(is_ortho),
        'inverse_is_transpose': bool(inverse_is_transpose),
        'determinant': det,
        'condition_number': cond,
        'preserves_length': bool(preserves_len),
    }