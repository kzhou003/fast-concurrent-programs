"""
PROBLEM: Implement Flash Attention (IO-Aware Attention)

Build an optimized attention implementation that reduces memory I/O.
Flash Attention is a key innovation that dramatically improves attention efficiency
by changing the computational order to reduce data movement.

REQUIREMENTS:
- Implement forward pass with tiling strategy
- Split attention computation into blocks (flash blocks)
- Minimize global memory accesses
- Implement online softmax (incremental max/sum tracking)
- Support variable sequence lengths
- Compute gradients efficiently

PERFORMANCE NOTES:
- Should achieve 2-4x speedup over naive attention
- Memory access pattern should be optimal for GPU cache
- Should handle long sequences (4000+ tokens) efficiently
- Numerical stability is critical

TEST CASE EXPECTATIONS:
- Output should match standard attention (within precision)
- Should work with different sequence lengths
- Gradient computation should be correct
- Performance should improve with longer sequences
- Should handle different d_k values
"""

import numpy as np
from typing import Tuple, Optional


def attention_naive(
    Q: np.ndarray,  # (batch, seq_len, d_k)
    K: np.ndarray,  # (batch, seq_len, d_k)
    V: np.ndarray,  # (batch, seq_len, d_v)
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Naive attention: softmax(QK^T / sqrt(d_k)) @ V

    Args:
        Q, K, V: Query, Key, Value matrices
        mask: Optional attention mask

    Returns:
        Output and attention weights
    """
    batch_size, seq_len, d_k = Q.shape
    _, _, d_v = V.shape

    # Compute scores
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)

    # Apply mask
    if mask is not None:
        scores = scores + mask

    # Softmax
    exp_scores = np.exp(scores - np.max(scores, axis=2, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=2, keepdims=True)

    # Apply to values
    output = weights @ V

    return output, weights


def attention_flash(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    block_size: int = 128,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flash Attention: IO-aware attention with tiling.

    The key insight is to compute attention in blocks to maximize cache usage.

    Args:
        Q, K, V: Query, Key, Value matrices
        block_size: Block size for tiling
        mask: Optional attention mask

    Returns:
        Output and attention weights
    """
    batch_size, seq_len, d_k = Q.shape
    _, _, d_v = V.shape

    output = np.zeros((batch_size, seq_len, d_v), dtype=Q.dtype)
    weights = np.zeros((batch_size, seq_len, seq_len), dtype=Q.dtype)

    # TODO: Implement flash attention
    # Key idea: Instead of computing full attention matrix, process in blocks
    #
    # For each block of queries:
    #   - Initialize running max and sum for softmax
    #   - For each block of keys/values:
    #     - Compute block scores
    #     - Update running max/sum (online softmax)
    #     - Accumulate weighted values
    #   - Final normalization using running stats
    #
    # This reduces the amount of data kept in GPU memory

    for b in range(batch_size):
        # Process query blocks
        for q_idx in range(0, seq_len, block_size):
            q_end = min(q_idx + block_size, seq_len)
            Q_block = Q[b, q_idx:q_end, :]

            # Initialize accumulators
            O_block = np.zeros((q_end - q_idx, d_v), dtype=Q.dtype)
            m_block = np.full((q_end - q_idx, 1), -np.inf, dtype=Q.dtype)
            l_block = np.zeros((q_end - q_idx, 1), dtype=Q.dtype)
            w_block = np.zeros((q_end - q_idx, seq_len), dtype=Q.dtype)

            # Process key/value blocks
            for k_idx in range(0, seq_len, block_size):
                k_end = min(k_idx + block_size, seq_len)
                K_block = K[b, k_idx:k_end, :]
                V_block = V[b, k_idx:k_end, :]

                # TODO: Compute attention for this block pair
                # 1. Compute scores: (q_len, k_len)
                scores_block = Q_block @ K_block.T / np.sqrt(d_k)

                # 2. Apply mask if provided
                if mask is not None:
                    scores_block = scores_block + mask[q_idx:q_end, k_idx:k_end]

                # 3. Online softmax update
                # Track max and running sum for numerical stability
                m_block_new = np.maximum(m_block, np.max(scores_block, axis=1, keepdims=True))

                # 4. Compute attention weights for this block
                exp_scores = np.exp(scores_block - m_block_new)
                l_block_new = l_block * np.exp(m_block - m_block_new) + np.sum(exp_scores, axis=1, keepdims=True)

                # 5. Update output with rescaled previous values
                O_block = O_block * (np.exp(m_block - m_block_new) * l_block / l_block_new)
                O_block = O_block + (exp_scores / l_block_new) @ V_block

                # 6. Store weights for reference
                w_block[:, k_idx:k_end] = exp_scores / l_block_new.squeeze(1)[:, None]

                # Update max and sum
                m_block = m_block_new
                l_block = l_block_new

            output[b, q_idx:q_end, :] = O_block
            weights[b, q_idx:q_end, :] = w_block

    return output, weights


def test_flash_attention_correctness():
    """Test that flash attention matches naive implementation."""
    batch_size, seq_len, d_k, d_v = 2, 64, 32, 32

    Q = np.random.randn(batch_size, seq_len, d_k).astype(np.float32) / np.sqrt(d_k)
    K = np.random.randn(batch_size, seq_len, d_k).astype(np.float32) / np.sqrt(d_k)
    V = np.random.randn(batch_size, seq_len, d_v).astype(np.float32)

    # Compute both
    out_naive, weights_naive = attention_naive(Q, K, V)
    out_flash, weights_flash = attention_flash(Q, K, V, block_size=32)

    # Compare
    np.testing.assert_allclose(out_flash, out_naive, rtol=1e-4)
    np.testing.assert_allclose(weights_flash, weights_naive, rtol=1e-4)

    print(f"✓ Flash attention correctness test passed")


def test_flash_attention_long_sequence():
    """Test flash attention with long sequences."""
    batch_size, seq_len, d_k, d_v = 2, 512, 64, 64

    Q = np.random.randn(batch_size, seq_len, d_k).astype(np.float32) / np.sqrt(d_k)
    K = np.random.randn(batch_size, seq_len, d_k).astype(np.float32) / np.sqrt(d_k)
    V = np.random.randn(batch_size, seq_len, d_v).astype(np.float32)

    out_naive, _ = attention_naive(Q, K, V)
    out_flash, _ = attention_flash(Q, K, V, block_size=128)

    np.testing.assert_allclose(out_flash, out_naive, rtol=1e-3, atol=1e-5)

    print(f"✓ Long sequence test passed ({seq_len} tokens)")


def test_flash_attention_with_mask():
    """Test flash attention with causal mask."""
    batch_size, seq_len, d_k, d_v = 2, 32, 16, 16

    Q = np.random.randn(batch_size, seq_len, d_k).astype(np.float32) / np.sqrt(d_k)
    K = np.random.randn(batch_size, seq_len, d_k).astype(np.float32) / np.sqrt(d_k)
    V = np.random.randn(batch_size, seq_len, d_v).astype(np.float32)

    # Create causal mask
    mask = np.tril(np.zeros((seq_len, seq_len))) - np.triu(
        np.ones((seq_len, seq_len)) * np.inf, k=1
    )

    out_naive, weights_naive = attention_naive(Q, K, V, mask=mask)
    out_flash, weights_flash = attention_flash(Q, K, V, block_size=16, mask=mask)

    np.testing.assert_allclose(out_flash, out_naive, rtol=1e-3)

    # Verify causality
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert np.allclose(weights_flash[:, i, j], 0.0), f"Position {i} attends to future {j}"

    print(f"✓ Causal mask test passed")


def test_different_block_sizes():
    """Test flash attention with different block sizes."""
    batch_size, seq_len, d_k, d_v = 2, 128, 32, 32

    Q = np.random.randn(batch_size, seq_len, d_k).astype(np.float32) / np.sqrt(d_k)
    K = np.random.randn(batch_size, seq_len, d_k).astype(np.float32) / np.sqrt(d_k)
    V = np.random.randn(batch_size, seq_len, d_v).astype(np.float32)

    out_naive, _ = attention_naive(Q, K, V)

    # Try different block sizes
    for block_size in [16, 32, 64, 128]:
        out_flash, _ = attention_flash(Q, K, V, block_size=block_size)
        np.testing.assert_allclose(out_flash, out_naive, rtol=1e-3)

    print(f"✓ Different block sizes test passed")


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    batch_size, seq_len, d_k, d_v = 2, 32, 16, 16

    # Large values
    Q_large = np.random.randn(batch_size, seq_len, d_k).astype(np.float32) * 1000
    K_large = np.random.randn(batch_size, seq_len, d_k).astype(np.float32) * 1000
    V_large = np.random.randn(batch_size, seq_len, d_v).astype(np.float32)

    out_flash, _ = attention_flash(Q_large, K_large, V_large, block_size=16)

    # Should not have NaN or Inf
    assert np.all(np.isfinite(out_flash))

    print(f"✓ Numerical stability test passed")


def test_multi_head_attention():
    """Test flash attention for multi-head setup."""
    batch_size, seq_len, num_heads, d_k, d_v = 2, 64, 8, 8, 8

    Q = np.random.randn(batch_size, num_heads, seq_len, d_k).astype(np.float32) / np.sqrt(d_k)
    K = np.random.randn(batch_size, num_heads, seq_len, d_k).astype(np.float32) / np.sqrt(d_k)
    V = np.random.randn(batch_size, num_heads, seq_len, d_v).astype(np.float32)

    # Process each head
    output = np.zeros((batch_size, num_heads, seq_len, d_v), dtype=Q.dtype)

    for h in range(num_heads):
        out_h, _ = attention_flash(
            Q[:, h, :, :],
            K[:, h, :, :],
            V[:, h, :, :],
            block_size=32,
        )
        output[:, h, :, :] = out_h

    assert output.shape == (batch_size, num_heads, seq_len, d_v)
    print(f"✓ Multi-head attention test passed")


def test_attention_output_properties():
    """Test properties of attention output."""
    batch_size, seq_len, d_k, d_v = 2, 32, 16, 16

    Q = np.random.randn(batch_size, seq_len, d_k).astype(np.float32) / np.sqrt(d_k)
    K = np.random.randn(batch_size, seq_len, d_k).astype(np.float32) / np.sqrt(d_k)
    V = np.random.randn(batch_size, seq_len, d_v).astype(np.float32)

    out, weights = attention_flash(Q, K, V, block_size=16)

    # Check weight properties
    assert np.all(weights >= 0)  # Weights should be non-negative
    sums = np.sum(weights, axis=2)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-5)  # Should sum to 1 per query

    # Output shape should match
    assert out.shape == (batch_size, seq_len, d_v)

    print(f"✓ Attention output properties test passed")


def benchmark_flash_vs_naive():
    """Benchmark flash attention vs naive implementation."""
    import time

    seq_lengths = [128, 256, 512]
    d_k = 64
    d_v = 64

    print("\nBenchmark (Flash vs Naive):")
    print(f"{'Seq Len':<10} {'Naive (ms)':<15} {'Flash (ms)':<15} {'Speedup':<10}")
    print("-" * 50)

    for seq_len in seq_lengths:
        batch_size = 2

        Q = np.random.randn(batch_size, seq_len, d_k).astype(np.float32) / np.sqrt(d_k)
        K = np.random.randn(batch_size, seq_len, d_k).astype(np.float32) / np.sqrt(d_k)
        V = np.random.randn(batch_size, seq_len, d_v).astype(np.float32)

        # Naive
        start = time.time()
        for _ in range(3):
            attention_naive(Q, K, V)
        t_naive = (time.time() - start) / 3

        # Flash
        start = time.time()
        for _ in range(3):
            attention_flash(Q, K, V, block_size=128)
        t_flash = (time.time() - start) / 3

        speedup = t_naive / t_flash if t_flash > 0 else 0

        print(f"{seq_len:<10} {t_naive*1000:<15.2f} {t_flash*1000:<15.2f} {speedup:<10.2f}x")


if __name__ == "__main__":
    print("Running Flash Attention tests...\n")

    test_flash_attention_correctness()
    test_flash_attention_long_sequence()
    test_flash_attention_with_mask()
    test_different_block_sizes()
    test_numerical_stability()
    test_multi_head_attention()
    test_attention_output_properties()

    benchmark_flash_vs_naive()

    print("\n✓ All tests passed!")
