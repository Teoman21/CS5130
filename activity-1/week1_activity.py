import numpy as np
import time

def dot_product_loop(vec1, vec2):
    """
    Calculate dot product using a for loop.
    
    Parameters:
    vec1 (list): First vector as a Python list
    vec2 (list): Second vector as a Python list
    
    Returns:
    float: The dot product of vec1 and vec2
    """
     # Your implementation here
    n = len(vec1)
    result = 0.0
    
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length")
    
    for i in range(n):
        result += vec1[i] * vec2[i]
        
    
    return result
         
    
def dot_product_numpy(vec1, vec2):
    """
    Calculate dot product using NumPy.
    
    Parameters:
    vec1 (np.ndarray): First vector as NumPy array
    vec2 (np.ndarray): Second vector as NumPy array
    
    Returns:
    float: The dot product of vec1 and vec2
    """
    # Your implementation here
    
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must be of the same length")
    
    #np.dot returns int so i converted it for float
    res = float(np.dot(vec1, vec2)) 
    
    return res

def vector_norm_loop(vec):
    """
    Calculate L2 norm using a for loop.
    
    Parameters:
    vec (list): Input vector as a Python list
    
    Returns:
    float: The L2 norm of the vector
    """
    
    #NOTE : NO CHECK FOR EMPTY VECTOR 
    # Your implementation here
    
    n = len(vec)
    res = 0.0
    
    for i in range(n):
        res += vec[i] ** 2
        
    return res ** 0.5
    

def vector_norm_numpy(vec):
    """
    Calculate L2 norm using NumPy.
    
    Parameters:
    vec (np.ndarray): Input vector as NumPy array
    
    Returns:
    float: The L2 norm of the vector
    """
    # Your implementation here
    
    l2 = np.linalg.norm(vec)
    
    return l2

def matrix_multiply_loop(mat1, mat2):
    """
    Multiply two matrices using nested for loops.
    
    Parameters:
    mat1 (list of lists): First matrix as nested Python lists
    mat2 (list of lists): Second matrix as nested Python lists
    
    Returns:
    list of lists: Result of matrix multiplication
    """
    # Your implementation here
    
    rows_mat1 = len(mat1) #to get the length of rows for the loop
    cols_mat1 = len(mat1[0]) #to get the length of cols for the loop 
    rows_mat2 = len(mat2) #to get the length of rows for the loop
    cols_mat2 = len(mat2[0]) #to get the length of cols for the loop
    
    #need to check mat1's cols and mat2's rows are equal
    if cols_mat1 != rows_mat2:
       raise ValueError("Number of columns in mat1 must equal number of rows in mat2")
        #used aactualy error instead of print message
    
    res = []
    
    for i in range(rows_mat1):
        new_row = []
        
        for j in range(cols_mat2):
            summ = 0
            
            for k in range(cols_mat1):
                summ += mat1[i][k] * mat2[k][j]
            
            new_row.append(summ)
            
        res.append(new_row)
        
    return res

def matrix_multiply_numpy(mat1, mat2):
    """
    Multiply two matrices using NumPy.
    
    Parameters:
    mat1 (np.ndarray): First matrix as NumPy array
    mat2 (np.ndarray): Second matrix as NumPy array
    
    Returns:
    np.ndarray: Result of matrix multiplication
    """
    # Your implementation here
    
    res = mat1 @ mat2
    
    return res

def matrix_transpose_loop(mat):
    """
    Transpose a matrix using for loops.
    
    Parameters:
    mat (list of lists): Input matrix as nested Python lists
    
    Returns:
    list of lists: Transposed matrix
    """
    # Your implementation here
    rows = len(mat) #to get the length of rows for the loop
    cols = len(mat[0]) #to get the length of cols for the loop 
    
    res = []
    
    for i in range(cols):
        new_row = []
        
        for j in range(rows):
            
            new_row.append(mat[j][i])
        res.append(new_row)
        
    return res
        
    
    

def matrix_transpose_numpy(mat):
    """
    Transpose a matrix using NumPy.
    
    Parameters:
    mat (np.ndarray): Input matrix as NumPy array
    
    Returns:
    np.ndarray: Transposed matrix
    """
    # Your implementation here
    res = mat.T
    
    return res


def performance_comparison(size=1000):
    """
    Compare execution time between loop-based and NumPy implementations.
    
    Parameters:
    size (int): Size of vectors/matrices to test
    
    Returns:
    dict: Dictionary containing timing results with keys:
          - 'dot_product_loop_time'
          - 'dot_product_numpy_time'
          - 'matrix_multiply_loop_time'
          - 'matrix_multiply_numpy_time'
          - 'speedup_dot_product' (loop_time / numpy_time) 
          - 'speedup_matrix_multiply' (loop_time / numpy_time)
    """
    results = {}
    
    # Generate random test data
    vec1_list = [np.random.random() for _ in range(size)]
    vec2_list = [np.random.random() for _ in range(size)]
    vec1_np = np.array(vec1_list)
    vec2_np = np.array(vec2_list)
    
    # For matrix multiplication, use smaller size to avoid timeout
    mat_size = min(100, size // 10)
    mat1_list = [[np.random.random() for _ in range(mat_size)] 
                 for _ in range(mat_size)]
    mat2_list = [[np.random.random() for _ in range(mat_size)] 
                 for _ in range(mat_size)]
    mat1_np = np.array(mat1_list)
    mat2_np = np.array(mat2_list)
    
    # Time dot product operations
    # Your timing implementation here
    
    
    #NOTE: DONT USE TIME.START INSTEAD USE TIME PERFCOUNTER
    #timing for loop implementation
    start = time.perf_counter()
    dot_product_loop(vec1_list, vec2_list)
    end = time.perf_counter()
    results['dot_product_loop_time'] = end - start
    
    
    
    #timing for numpy implementation
    start = time.perf_counter()
    dot_product_numpy(vec1_np, vec2_np)
    end = time.perf_counter()
    
    results['dot_product_numpy_time'] = end - start
    
    # Time matrix multiplication operations
    # Your timing implementation here
    
    #loop implementation timing
    start= time.perf_counter()
    matrix_multiply_loop(mat1_list, mat2_list)
    end = time.perf_counter()
    
    results['matrix_multiply_loop_time'] = end - start
    
    #numpy implementation timing
    start = time.perf_counter()
    matrix_multiply_numpy(mat1_np, mat2_np)
    end = time.perf_counter()
    
    results['matrix_multiply_numpy_time'] = end - start
    
    
    # Calculate speedup ratios
    # Your calculation here
    
    #spedup of for dot product
    results['speedup_matrix_multiply'] = (
        
    results['matrix_multiply_loop_time'] / results['matrix_multiply_numpy_time']
    if results['matrix_multiply_numpy_time'] > 0 else float('inf')
    )

    #speedup of for matrix multiplication
    results['speedup_dot_product'] = (
        
        results['dot_product_loop_time'] / results['dot_product_numpy_time']
        if results['dot_product_numpy_time'] > 0 else float('inf')
    )
        
    return results
    
    

def main():
    """
    Main function to demonstrate all implementations and print results.
    """
    print("=" * 50)
    print("NumPy vs For Loops Performance Comparison")
    print("=" * 50)
    
    # Test with small examples for correctness
    vec1 = [1, 2, 3]
    vec2 = [4, 5, 6]
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    print("\n--- Correctness Check ---")
    print(f"Dot Product (Loop): {dot_product_loop(vec1, vec2)}")
    print(f"Dot Product (NumPy): {dot_product_numpy(vec1_np, vec2_np)}")
    
    mat = [[1, 2], [3, 4]]
    mat_np = np.array(mat)
    print(f"\nOriginal Matrix: {mat}")
    print(f"Transpose (Loop): {matrix_transpose_loop(mat)}")
    print(f"Transpose (NumPy): {matrix_transpose_numpy(mat_np).tolist()}")
    
    # Performance comparison
    print("\n--- Performance Analysis ---")
    results = performance_comparison(size=1000)
    
    print(f"\nDot Product:")
    print(f"  Loop Time: {results['dot_product_loop_time']:.6f} seconds")
    print(f"  NumPy Time: {results['dot_product_numpy_time']:.6f} seconds")
    print(f"  NumPy Speedup: {results['speedup_dot_product']:.2f}x faster")
    
    print(f"\nMatrix Multiplication:")
    print(f"  Loop Time: {results['matrix_multiply_loop_time']:.6f} seconds")
    print(f"  NumPy Time: {results['matrix_multiply_numpy_time']:.6f} seconds")
    print(f"  NumPy Speedup: {results['speedup_matrix_multiply']:.2f}x faster")
    
    return results

if __name__ == "__main__":
    main()
