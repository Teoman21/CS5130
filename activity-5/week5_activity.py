import numpy as np
import warnings
warnings.filterwarnings('ignore')  # Suppress ill-conditioned matrix warnings

def create_ill_conditioned_problem(n=50, kappa=1e8, seed=42):
    """
    Creates an ill-conditioned least squares problem with known solution.
    
    Args:
        n: Size of the problem
        kappa: Target condition number
        seed: Random seed for reproducibility
    
    Returns:
        A: Matrix with condition number ≈ kappa
        b: Right-hand side
        x_true: True solution (all ones)
    """
    np.random.seed(seed)
    
    # Create matrix with specific condition number
    U, _ = np.linalg.qr(np.random.randn(n, n))
    V, _ = np.linalg.qr(np.random.randn(n, n))
    
    # Create singular values from 1 to 1/kappa
    singular_values = np.logspace(0, -np.log10(kappa), n)
    
    A = U @ np.diag(singular_values) @ V.T
    
    # True solution
    x_true = np.ones(n)
    
    # Create right-hand side with small noise
    b_clean = A @ x_true
    noise = np.random.randn(n) * 1e-10
    b = b_clean + noise
    
    return A, b, x_true


# PROBLEM 1: Normal Equations (3 points)
def solve_normal_equations(A, b):
    """
    Solve the least squares problem using normal equations.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side
    
    Returns:
        x: Solution vector
    """
    # YOUR CODE HERE (2-3 lines)
    # Step 1: Compute A^T A and A^T b
    # Step 2: Solve (A^T A) x = A^T b
    
    AtA = A.T @ A
    Atb = A.T @ b
    x = np.linalg.solve(AtA, Atb) 
    
    return x


# PROBLEM 2: QR Decomposition (3 points)
def solve_qr(A, b):
    """
    Solve the least squares problem using QR decomposition.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side
    
    Returns:
        x: Solution vector
    """
    # YOUR CODE HERE (2-3 lines)
    # Step 1: Compute QR decomposition of A
    # Step 2: Solve R x = Q^T b
    
    Q, R = np.linalg.qr(A)
    Qtb = Q.T @ b
    
    x = np.linalg.solve(R, Qtb)
    return x


# PROBLEM 3: SVD with Truncation (4 points)
def solve_svd(A, b, tol=1e-10):
    """
    Solve the least squares problem using SVD with truncation.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side
        tol: Truncation tolerance for small singular values
    
    Returns:
        x: Solution vector
    """
    # YOUR CODE HERE (4-5 lines)
    # Step 1: Compute SVD of A
    # Step 2: Compute pseudoinverse, truncating singular values < tol
    # Step 3: Compute x = V @ (S_inv @ (U^T @ b))
    # Hint: Use np.where to handle truncation
    
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_inv = np.where(S > tol, 1/S, 0) 
    x = Vt.T @ (S_inv * (U.T @ b))
    return x


def test_methods():
    """Test all three methods on increasingly ill-conditioned problems."""
    
    condition_numbers = [1e2, 1e4, 1e6, 1e8, 1e10]
    results = []
    
    print("="*60)
    print("STABILITY COMPARISON: Normal Equations vs QR vs SVD")
    print("="*60)
    
    for kappa in condition_numbers:
        # Create problem
        A, b, x_true = create_ill_conditioned_problem(n=50, kappa=kappa)
        
        # Solve with each method
        try:
            x_normal = solve_normal_equations(A, b)
            error_normal = np.linalg.norm(x_normal - x_true) / np.linalg.norm(x_true)
        except:
            error_normal = np.inf
            
        try:
            x_qr = solve_qr(A, b)
            error_qr = np.linalg.norm(x_qr - x_true) / np.linalg.norm(x_true)
        except:
            error_qr = np.inf
            
        try:
            x_svd = solve_svd(A, b)
            error_svd = np.linalg.norm(x_svd - x_true) / np.linalg.norm(x_true)
        except:
            error_svd = np.inf
        
        results.append({
            'kappa': kappa,
            'error_normal': error_normal,
            'error_qr': error_qr,
            'error_svd': error_svd
        })
        
        print(f"\nCondition number κ = {kappa:.0e}")
        print(f"  Normal equations error: {error_normal:.2e}")
        print(f"  QR decomposition error: {error_qr:.2e}")
        print(f"  SVD with truncation error: {error_svd:.2e}")
        
        # Interpretation
        if error_normal > 1e-6:
            print("  → Normal equations: ❌ FAILED (large error)")
        if error_qr < error_normal / 10:
            print("  → QR is {:.0f}x more accurate than normal equations".format(
                error_normal / error_qr))
    
    return results


# Run the test when file is executed
if __name__ == "__main__":
    results = test_methods()
    print("\n" + "="*60)
    print("CONCLUSION: QR and SVD maintain stability even for")
    print("ill-conditioned problems where normal equations fail!")
    print("="*60)