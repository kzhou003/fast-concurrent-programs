"""
PROBLEM: Implement Matrix Multiplication in Triton

Build an optimized matrix multiplication kernel using Triton.
This is fundamental to understanding GPU programming and kernel optimization.

REQUIREMENTS:
- Implement C = A @ B for matrices (use tiling/blocking)
- Optimize for GPU memory hierarchy (global -> shared -> registers)
- Support different block sizes and tile configurations
- Implement proper synchronization
- Handle non-square matrices
- Benchmark and compare with baseline

PERFORMANCE NOTES:
- Should achieve >80% of theoretical peak FLOPS
- Should use block sizes of 64-128 for typical GPUs
- Tile size should match GPU cache line (typically 128 bytes)
- Memory access patterns should be coalesced

TEST CASE EXPECTATIONS:
- Output should match PyTorch's matrix multiplication
- Should handle different matrix shapes
- Performance should scale with matrix size
- Larger tiles should generally be faster (with memory limits)
- Different block configurations should be comparable in performance
"""

"""
Note: This is pseudocode for Triton. Actual implementation requires Triton library.
Structure demonstrates key concepts you'll implement:
"""


# Pseudocode for Triton implementation
TRITON_MATMUL_KERNEL = """
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # TODO: Implement matrix multiplication kernel
    # 1. Get block indices
    # 2. Load blocks of A and B from global memory
    # 3. Compute partial dot products
    # 4. Accumulate results
    # 5. Store result to C

    # Get block IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # TODO: Compute offsets for A, B, C blocks
    # TODO: Loop over K dimension in blocks
    # TODO: Load tiles of A and B
    # TODO: Compute tile multiply
    # TODO: Store result
    pass
"""

import numpy as np
import torch

# Fallback numpy implementation for testing without Triton
def matmul_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Naive matrix multiplication for testing.

    Args:
        A: Matrix of shape (M, K)
        B: Matrix of shape (K, N)

    Returns:
        Result of shape (M, N)
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    # TODO: Implement naive O(n^3) matrix multiplication
    # C[i,j] = sum_k A[i,k] * B[k,j]
    C = np.zeros((M, N), dtype=A.dtype)

    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]

    return C


def matmul_blocked(
    A: np.ndarray, B: np.ndarray, block_size: int = 64
) -> np.ndarray:
    """
    Block-based matrix multiplication (simulates tiling on GPU).

    Args:
        A: Matrix of shape (M, K)
        B: Matrix of shape (K, N)
        block_size: Tile/block size

    Returns:
        Result of shape (M, N)
    """
    # TODO: Implement blocked matrix multiplication
    # This simulates how Triton would process blocks
    # 1. Process A and B in blocks of block_size x block_size
    # 2. Accumulate partial products
    # 3. This improves cache locality

    M, K = A.shape
    _, N = B.shape

    C = np.zeros((M, N), dtype=A.dtype)

    # Block over M and N dimensions
    for i in range(0, M, block_size):
        for j in range(0, N, block_size):
            # Block for K dimension
            for k in range(0, K, block_size):
                # Get blocks
                i_end = min(i + block_size, M)
                j_end = min(j + block_size, N)
                k_end = min(k + block_size, K)

                A_block = A[i:i_end, k:k_end]
                B_block = B[k:k_end, j:j_end]

                # Compute block multiplication
                C[i:i_end, j:j_end] += A_block @ B_block

    return C


def test_matmul_correctness():
    """Test that multiplication produces correct results."""
    M, K, N = 128, 256, 64

    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    # Reference
    C_ref = A @ B

    # Our implementation
    C_naive = matmul_numpy(A, B)

    np.testing.assert_allclose(C_naive, C_ref, rtol=1e-5)

    print(f"✓ Matmul correctness test passed ({M}x{K} @ {K}x{N})")


def test_matmul_blocked_correctness():
    """Test blocked implementation correctness."""
    M, K, N = 128, 256, 64

    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    C_ref = A @ B
    C_blocked = matmul_blocked(A, B, block_size=32)

    np.testing.assert_allclose(C_blocked, C_ref, rtol=1e-5)

    print(f"✓ Blocked matmul correctness test passed")


def test_different_shapes():
    """Test with various matrix shapes."""
    shapes = [
        (32, 64, 32),
        (64, 128, 64),
        (128, 256, 128),
        (256, 512, 256),
        (100, 150, 75),  # Non-multiples of block size
    ]

    for M, K, N in shapes:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        C_ref = A @ B
        C_blocked = matmul_blocked(A, B, block_size=32)

        np.testing.assert_allclose(C_blocked, C_ref, rtol=1e-5)

    print(f"✓ Different shapes test passed ({len(shapes)} configurations)")


def test_block_size_impact():
    """Test performance with different block sizes."""
    import time

    M, K, N = 512, 512, 512

    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    block_sizes = [16, 32, 64, 128]

    for block_size in block_sizes:
        start = time.time()
        C = matmul_blocked(A, B, block_size=block_size)
        elapsed = time.time() - start

        # Check correctness
        C_ref = A @ B
        np.testing.assert_allclose(C, C_ref, rtol=1e-5)

        print(f"  Block size {block_size:3d}: {elapsed*1000:.2f}ms")

    print(f"✓ Block size impact test passed")


def test_rectangular_matrices():
    """Test with non-square matrices."""
    test_cases = [
        (64, 128, 32),   # Wide A
        (32, 128, 64),   # Tall A
        (256, 64, 256),  # Wide B
        (128, 256, 32),  # Tall B
    ]

    for M, K, N in test_cases:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        C_ref = A @ B
        C_blocked = matmul_blocked(A, B, block_size=32)

        np.testing.assert_allclose(C_blocked, C_ref, rtol=1e-5)

    print(f"✓ Rectangular matrices test passed")


def test_single_element():
    """Test edge case of 1x1 matrices."""
    A = np.array([[2.0]], dtype=np.float32)
    B = np.array([[3.0]], dtype=np.float32)

    C_ref = A @ B
    C_blocked = matmul_blocked(A, B, block_size=32)

    np.testing.assert_allclose(C_blocked, C_ref, rtol=1e-5)

    print(f"✓ Single element test passed")


def test_large_matrices():
    """Test with large matrices (simulating realistic GPUs)."""
    M, K, N = 2048, 4096, 2048

    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    C_ref = A @ B
    C_blocked = matmul_blocked(A, B, block_size=64)

    np.testing.assert_allclose(C_blocked, C_ref, rtol=1e-4)

    print(f"✓ Large matrices test passed ({M}x{K}x{N})")


def benchmark_operations():
    """Benchmark different implementations."""
    import time

    M, K, N = 1024, 1024, 1024

    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    # Reference (NumPy @)
    start = time.time()
    C_ref = A @ B
    t_ref = time.time() - start

    # Blocked implementation
    start = time.time()
    C_blocked = matmul_blocked(A, B, block_size=64)
    t_blocked = time.time() - start

    print(f"\nBenchmark results ({M}x{K}x{N}):")
    print(f"  NumPy @: {t_ref*1000:.2f}ms")
    print(f"  Blocked: {t_blocked*1000:.2f}ms")
    print(f"  Speedup: {t_ref/t_blocked:.2f}x")


if __name__ == "__main__":
    print("Running Matrix Multiplication tests...\n")

    test_matmul_correctness()
    test_matmul_blocked_correctness()
    test_different_shapes()
    test_rectangular_matrices()
    test_single_element()
    test_large_matrices()
    test_block_size_impact()

    benchmark_operations()

    print("\n✓ All tests passed!")
