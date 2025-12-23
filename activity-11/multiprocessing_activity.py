"""
Activity: Your First Multiprocessing Program
Student Name: YOUR NAME HERE
"""

import multiprocessing as mp
import time


def compute_cube_root(n):
    """
    Compute the cube root of n (repeated 1,000,000 times for CPU intensity).
    
    Args:
        n: A positive number
        
    Returns:
        The cube root of n
    """
    result = 0
    for _ in range(1000000):
        result = n ** (1/3)
    return result


def process_serial(numbers):
    """
    Process numbers serially (one at a time).
    
    Args:
        numbers: List of numbers to process
        
    Returns:
        Tuple of (results list, elapsed time in seconds)
    """
    start_time = time.time()
    
    # TODO: Process each number using compute_cube_root
    results = []
    for n in numbers:
        results.append(compute_cube_root(n))
    
    elapsed = time.time() - start_time
    return results, elapsed


def process_parallel(numbers):
    """
    Process numbers in parallel using multiprocessing.
    
    Args:
        numbers: List of numbers to process
        
    Returns:
        Tuple of (results list, elapsed time in seconds)
    """
    start_time = time.time()
    
    # TODO: Use multiprocessing.Pool() and pool.map()
    results = []
    with mp.Pool() as pool:
        results = pool.map(compute_cube_root, numbers)
    
    elapsed = time.time() - start_time
    return results, elapsed


if __name__ == "__main__":
    # Test with 100 numbers
    test_numbers = list(range(1, 101))
    
    # Run serial version
    serial_results, serial_time = process_serial(test_numbers)
    
    # Run parallel version
    parallel_results, parallel_time = process_parallel(test_numbers)
    
    # Calculate speedup
    speedup = serial_time / parallel_time
    
    # Print results (required for autograder)
    print(f"Serial time: {serial_time:.4f} seconds")
    print(f"Parallel time: {parallel_time:.4f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Results match: {serial_results == parallel_results}")