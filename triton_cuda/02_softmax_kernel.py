"""
PROBLEM: Implement Softmax Kernel in Triton

Build an optimized softmax kernel that processes matrices row-wise.
This is critical for attention mechanisms and classification layers.

REQUIREMENTS:
- Implement numerically stable softmax: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))
- Process entire rows efficiently (critical for attention)
- Support different block sizes for parallelization
- Implement both forward and backward passes
- Handle different data types (float32, float16)
- Avoid numerical overflow/underflow

PERFORMANCE NOTES:
- Should achieve high memory bandwidth utilization
- Row-wise processing should be well-parallelized
- Should minimize synchronization overhead
- Memory coalescing is critical

TEST CASE EXPECTATIONS:
- Output should match PyTorch's softmax (within numerical precision)
- Output should sum to 1 per row
- Should be numerically stable even with large inputs
- Backward pass should have correct gradients
- Should work with different matrix shapes
"""

import numpy as np
from typing import Tuple


def softmax_numpy(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax.

    Args:
        x: Input array
        axis: Axis to normalize over

    Returns:
        Softmax output
    """
    # TODO: Implement numerically stable softmax
    # 1. Subtract max for numerical stability
    # 2. Compute exp
    # 3. Divide by sum

    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_blocked(
    x: np.ndarray, block_size: int = 1024, axis: int = -1
) -> np.ndarray:
    """
    Blocked softmax computation (simulates GPU processing).

    Args:
        x: Input array of shape (batch, seq_len)
        block_size: Number of elements to process per block
        axis: Axis to normalize

    Returns:
        Softmax output
    """
    # TODO: Implement blocked softmax
    # Process each row (or batch) in blocks
    # For large sequences, this reduces memory pressure

    batch_size, seq_len = x.shape
    output = np.zeros_like(x, dtype=x.dtype)

    for b in range(batch_size):
        row = x[b, :]

        # TODO: Compute in blocks for large sequences
        # 1. Find max in blocks
        # 2. Compute exp of (x - global_max)
        # 3. Compute block sums
        # 4. Final normalization

        # For simplicity, can use stable softmax
        output[b, :] = softmax_numpy(row[None, :], axis=1)[0]

    return output


def softmax_backward(
    dout: np.ndarray, softmax_out: np.ndarray, axis: int = -1
) -> np.ndarray:
    """
    Backward pass through softmax.

    Args:
        dout: Gradient w.r.t. output
        softmax_out: Output from forward pass
        axis: Axis that was normalized

    Returns:
        Gradient w.r.t. input
    """
    # TODO: Implement softmax backward
    # d(softmax)/dx = softmax * (dout - (softmax * dout).sum(axis))

    s = softmax_out
    ds = dout

    # Compute gradient
    # Using chain rule: d/dx softmax(x) = softmax(x) * (d - (softmax * d).sum())
    grad_sum = np.sum(ds * s, axis=axis, keepdims=True)
    return s * (ds - grad_sum)


def log_softmax_numpy(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable log softmax.

    Args:
        x: Input array
        axis: Axis to normalize

    Returns:
        Log softmax output
    """
    # TODO: Implement numerically stable log softmax
    # log(softmax(x)) = x - log(sum(exp(x))) more stable than log(softmax)
    # = x - (max(x) + log(sum(exp(x - max(x)))))

    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    log_sum_exp = np.log(np.sum(exp_x, axis=axis, keepdims=True))
    return x - x_max - log_sum_exp


def test_softmax_correctness():
    """Test softmax produces correct results."""
    batch_size, seq_len = 4, 10

    x = np.random.randn(batch_size, seq_len).astype(np.float32)

    # Reference
    y_ref = softmax_numpy(x, axis=1)

    # Check sum to 1
    sums = y_ref.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-5)

    # Check all positive
    assert np.all(y_ref >= 0)

    print(f"✓ Softmax correctness test passed")


def test_softmax_numerical_stability():
    """Test numerical stability with large inputs."""
    x_large = np.random.randn(4, 100).astype(np.float32) * 1000

    y = softmax_numpy(x_large, axis=1)

    # Should not have NaN or Inf
    assert np.all(np.isfinite(y))

    # Should still sum to 1
    sums = y.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-5)

    print(f"✓ Numerical stability test passed")


def test_softmax_gradient():
    """Test softmax gradient computation."""
    batch_size, seq_len = 4, 10

    x = np.random.randn(batch_size, seq_len).astype(np.float32)
    y = softmax_numpy(x, axis=1)

    # Numerical gradient check
    dout = np.random.randn(batch_size, seq_len).astype(np.float32)
    dx_analytical = softmax_backward(dout, y, axis=1)

    # Compute numerical gradient
    eps = 1e-4
    dx_numerical = np.zeros_like(x)

    for i in range(batch_size):
        for j in range(seq_len):
            x[i, j] += eps
            y_plus = softmax_numpy(x, axis=1)
            loss_plus = np.sum(y_plus * dout)

            x[i, j] -= 2 * eps
            y_minus = softmax_numpy(x, axis=1)
            loss_minus = np.sum(y_minus * dout)

            x[i, j] += eps

            dx_numerical[i, j] = (loss_plus - loss_minus) / (2 * eps)

    # Compare
    error = np.abs(dx_analytical - dx_numerical) / (np.abs(dx_analytical) + np.abs(dx_numerical) + 1e-8)
    assert np.max(error) < 1e-2

    print(f"✓ Softmax gradient test passed (max error: {np.max(error):.2e})")


def test_softmax_blocked():
    """Test blocked softmax implementation."""
    batch_size, seq_len = 4, 100

    x = np.random.randn(batch_size, seq_len).astype(np.float32)

    y_ref = softmax_numpy(x, axis=1)
    y_blocked = softmax_blocked(x, block_size=32, axis=1)

    np.testing.assert_allclose(y_blocked, y_ref, rtol=1e-5)

    print(f"✓ Blocked softmax test passed")


def test_log_softmax():
    """Test log softmax."""
    batch_size, seq_len = 4, 10

    x = np.random.randn(batch_size, seq_len).astype(np.float32)

    log_y = log_softmax_numpy(x, axis=1)

    # Verify: exp(log_softmax) = softmax
    y_from_log = np.exp(log_y)
    y_direct = softmax_numpy(x, axis=1)

    np.testing.assert_allclose(y_from_log, y_direct, rtol=1e-5)

    print(f"✓ Log softmax test passed")


def test_different_axis():
    """Test softmax along different axes."""
    x = np.random.randn(4, 8, 6).astype(np.float32)

    # Softmax along last axis
    y_axis2 = softmax_numpy(x, axis=2)
    sums2 = y_axis2.sum(axis=2)
    np.testing.assert_allclose(sums2, 1.0, rtol=1e-5)

    # Softmax along axis 1
    y_axis1 = softmax_numpy(x, axis=1)
    sums1 = y_axis1.sum(axis=1)
    np.testing.assert_allclose(sums1, 1.0, rtol=1e-5)

    print(f"✓ Different axis test passed")


def test_very_large_inputs():
    """Test with very large input values."""
    x = np.ones((4, 100), dtype=np.float32) * 1e6

    y = softmax_numpy(x, axis=1)

    # Should have no NaN/Inf
    assert np.all(np.isfinite(y))

    # Should sum to 1
    sums = y.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-5)

    print(f"✓ Very large inputs test passed")


def test_very_small_inputs():
    """Test with very small input values."""
    x = np.ones((4, 100), dtype=np.float32) * 1e-10

    y = softmax_numpy(x, axis=1)

    # Should have no NaN/Inf
    assert np.all(np.isfinite(y))

    # All outputs should be similar (uniform-ish)
    expected = 1.0 / 100
    np.testing.assert_allclose(y, expected, rtol=0.1)

    print(f"✓ Very small inputs test passed")


def test_attention_use_case():
    """Test softmax as used in attention (batch x seq_len x seq_len)."""
    batch_size, seq_len = 4, 8

    # Attention scores
    scores = np.random.randn(batch_size, seq_len, seq_len).astype(np.float32)

    # Apply softmax per query (along key dimension)
    attention_weights = softmax_numpy(scores, axis=2)

    # Each query's attention should sum to 1
    sums = attention_weights.sum(axis=2)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-5)

    # All values should be in [0, 1]
    assert np.all(attention_weights >= 0) and np.all(attention_weights <= 1)

    print(f"✓ Attention use case test passed")


def test_softmax_temperature():
    """Test softmax with temperature scaling."""
    x = np.random.randn(4, 10).astype(np.float32)

    # Higher temperature -> more uniform distribution
    y_hot = softmax_numpy(x / 10.0, axis=1)  # Temperature 10
    y_cold = softmax_numpy(x / 0.1, axis=1)  # Temperature 0.1

    # Cold should be more peaked (higher max)
    assert np.max(y_cold) > np.max(y_hot)

    print(f"✓ Softmax temperature test passed")


if __name__ == "__main__":
    print("Running Softmax Kernel tests...\n")

    test_softmax_correctness()
    test_softmax_numerical_stability()
    test_softmax_gradient()
    test_softmax_blocked()
    test_log_softmax()
    test_different_axis()
    test_very_large_inputs()
    test_very_small_inputs()
    test_attention_use_case()
    test_softmax_temperature()

    print("\n✓ All tests passed!")
