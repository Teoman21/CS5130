"""
Week 4 Activity - Local Test Script
Run this before submitting to Gradescope!

Usage: python3 local_test_week4.py
"""

import numpy as np
from week4_activity import *


def run_local_tests():
    print("=" * 60)
    print("WEEK 4: ORTHOGONAL MATRICES - LOCAL TESTS")
    print("=" * 60)

    total_passed = 0
    total_tests = 5

    # Test 1: is_orthogonal
    print("\nTest 1: is_orthogonal function")
    print("-" * 40)
    try:
        I = np.eye(3)
        assert is_orthogonal(I) == True, "Identity should be orthogonal"

        S = np.array([[2, 0], [0, 2]])
        assert is_orthogonal(S) == False, "Scaled matrix should not be orthogonal"

        print("✓ PASSED")
        total_passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 2: orthogonal_preserves_length
    print("\nTest 2: orthogonal_preserves_length function")
    print("-" * 40)
    try:
        Q = np.array([[0, -1], [1, 0]])  # 90-degree rotation
        passed, error = orthogonal_preserves_length(Q)
        assert passed, f"Rotation should preserve length, error: {error}"
        print("✓ PASSED")
        total_passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 3: create_2d_rotation
    print("\nTest 3: create_2d_rotation function")
    print("-" * 40)
    try:
        R = create_2d_rotation(np.pi / 4)
        assert is_orthogonal(R), "Rotation matrix should be orthogonal"

        # Check if it rotates correctly
        v = np.array([1, 0])
        v_rot = R @ v
        expected = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
        assert np.allclose(v_rot, expected), "Rotation incorrect"

        print("✓ PASSED")
        total_passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 4: gram_schmidt_orthogonalize
    print("\nTest 4: gram_schmidt_orthogonalize function")
    print("-" * 40)
    try:
        A = np.array([[1, 1], [0, 1]])
        Q = gram_schmidt_orthogonalize(A)
        assert is_orthogonal(Q), "Gram-Schmidt should produce orthogonal matrix"
        print("✓ PASSED")
        total_passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 5: check_orthogonal_properties
    print("\nTest 5: check_orthogonal_properties function")
    print("-" * 40)
    try:
        R = create_2d_rotation(np.pi / 6)
        props = check_orthogonal_properties(R)

        assert props['is_orthogonal'], "Should be orthogonal"
        assert props['inverse_is_transpose'], "Inverse should equal transpose"
        assert abs(abs(props['determinant']) - 1) < 1e-10, "Det should be ±1"
        assert abs(props['condition_number'] - 1) < 1e-10, "Condition number should be 1"
        assert props['preserves_length'], "Should preserve length"

        print("✓ PASSED")
        total_passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {total_passed}/{total_tests} tests passed")
    if total_passed == total_tests:
        print("🎉 All tests passed! Ready to submit week4_activity.py to Gradescope.")
    else:
        print("⚠️  Some tests failed. Review your code before submitting.")
    print("=" * 60)


if __name__ == "__main__":
    run_local_tests()