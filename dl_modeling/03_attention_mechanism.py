"""
PROBLEM: Implement Scaled Dot-Product Attention

Build the attention mechanism that is core to transformers. This requires
understanding of matrix operations, softmax, and backpropagation through
softmax and scaling operations.

REQUIREMENTS:
- Implement forward pass: softmax(QK^T/sqrt(d_k))V
- Implement backward pass with gradients for Q, K, V
- Support batched matrix operations (batch_size, seq_len, d_model)
- Implement optional attention mask (for causal attention)
- Compute attention weights (useful for visualization)
- Numerical stability in softmax

PERFORMANCE NOTES:
- Should efficiently handle long sequences (1000+ tokens)
- Matrix operations should use numpy's optimized BLAS
- Memory usage proportional to sequence length squared (for attention matrix)

TEST CASE EXPECTATIONS:
- Attention output shape should be (batch, seq_len, d_v)
- Gradient computation should pass numerical checks
- Attention weights should sum to 1 per query
- Causal masking should work correctly
- Should work with different d_q, d_k, d_v values
"""

import numpy as np
from typing import Optional, Tuple


class ScaledDotProductAttention:
    """Scaled dot-product attention mechanism."""

    def __init__(self, d_k: Optional[int] = None):
        """
        Initialize attention.

        Args:
            d_k: Dimension of key (used for scaling). If None, inferred from inputs.
        """
        self.d_k = d_k
        self.cache = {}

    def forward(
        self,
        query: np.ndarray,  # (batch, seq_len, d_q)
        key: np.ndarray,    # (batch, seq_len, d_k)
        value: np.ndarray,  # (batch, seq_len, d_v)
        mask: Optional[np.ndarray] = None,  # (seq_len, seq_len) or (batch, seq_len, seq_len)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.

        Args:
            query: Query vectors (batch, seq_len, d_q)
            key: Key vectors (batch, seq_len, d_k)
            value: Value vectors (batch, seq_len, d_v)
            mask: Optional mask (0 for valid, -inf for invalid positions)

        Returns:
            output: Attention output (batch, seq_len, d_v)
            weights: Attention weights (batch, seq_len, seq_len)
        """
        batch_size, seq_len, d_q = query.shape
        _, _, d_k = key.shape
        _, _, d_v = value.shape

        # TODO: Implement scaled dot-product attention
        # 1. Compute attention scores: Q @ K^T / sqrt(d_k)
        # 2. Apply mask if provided
        # 3. Apply softmax to get attention weights
        # 4. Multiply with values to get output
        # 5. Cache all intermediate values for backward pass

        # Expected formula:
        # scores = Q @ K^T / sqrt(d_k)
        # if mask: scores = scores + mask
        # weights = softmax(scores)
        # output = weights @ V

        output = None
        weights = None

        return output, weights

    def backward(
        self,
        dout: np.ndarray,  # (batch, seq_len, d_v)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass.

        Args:
            dout: Gradient w.r.t. output (batch, seq_len, d_v)

        Returns:
            dquery: Gradient w.r.t. query
            dkey: Gradient w.r.t. key
            dvalue: Gradient w.r.t. value
        """
        # TODO: Implement backward pass
        # Use chain rule to compute gradients through:
        # 1. Matrix multiplication with value
        # 2. Softmax operation
        # 3. Scaling
        # 4. Matrix multiplication with key
        # Retrieve cached values from forward pass

        dquery = None
        dkey = None
        dvalue = None

        return dquery, dkey, dvalue

    def __call__(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.forward(query, key, value, mask)


def softmax_stable(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    # TODO: Implement stable softmax
    # x_max = max(x) along axis
    # exp_x = exp(x - x_max)
    # return exp_x / sum(exp_x)
    pass


def softmax_backward(dout: np.ndarray, softmax_out: np.ndarray, axis: int = -1) -> np.ndarray:
    """Gradient through softmax."""
    # TODO: Implement softmax backward
    # d(softmax)/dx = softmax * (dout - (softmax * dout).sum(axis=axis, keepdims=True))
    pass


def test_output_shape():
    """Test attention output shape."""
    batch, seq_len, d_q, d_k, d_v = 4, 10, 64, 64, 64

    attn = ScaledDotProductAttention(d_k=d_k)
    q = np.random.randn(batch, seq_len, d_q).astype(np.float32)
    k = np.random.randn(batch, seq_len, d_k).astype(np.float32)
    v = np.random.randn(batch, seq_len, d_v).astype(np.float32)

    output, weights = attn(q, k, v)

    assert output.shape == (batch, seq_len, d_v), f"Expected {(batch, seq_len, d_v)}, got {output.shape}"
    assert weights.shape == (batch, seq_len, seq_len), f"Expected {(batch, seq_len, seq_len)}, got {weights.shape}"

    print(f"✓ Output shape test passed")


def test_attention_weights_sum_to_one():
    """Test that attention weights sum to 1."""
    batch, seq_len, d_k, d_v = 2, 8, 32, 32

    attn = ScaledDotProductAttention(d_k=d_k)
    q = np.random.randn(batch, seq_len, d_k).astype(np.float32)
    k = np.random.randn(batch, seq_len, d_k).astype(np.float32)
    v = np.random.randn(batch, seq_len, d_v).astype(np.float32)

    output, weights = attn(q, k, v)

    # Weights should sum to 1 along seq dimension
    weight_sums = weights.sum(axis=2)
    np.testing.assert_allclose(weight_sums, 1.0, rtol=1e-5)

    print(f"✓ Attention weights sum to 1")


def test_causal_mask():
    """Test attention with causal mask (for autoregressive models)."""
    batch, seq_len, d_k, d_v = 2, 5, 32, 32

    # Create causal mask (upper triangular -inf)
    causal_mask = np.tril(np.zeros((seq_len, seq_len))) - np.triu(
        np.ones((seq_len, seq_len)) * np.inf, k=1
    )

    attn = ScaledDotProductAttention(d_k=d_k)
    q = np.random.randn(batch, seq_len, d_k).astype(np.float32)
    k = np.random.randn(batch, seq_len, d_k).astype(np.float32)
    v = np.random.randn(batch, seq_len, d_v).astype(np.float32)

    output, weights = attn(q, k, v, mask=causal_mask)

    # Check that no future tokens attend to past tokens
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert np.allclose(weights[:, i, j], 0.0), f"Position {i} attends to future {j}"

    print(f"✓ Causal mask test passed")


def test_identity_with_same_vectors():
    """Test that attention is identity when Q=K=V."""
    batch, seq_len, d = 2, 4, 16

    attn = ScaledDotProductAttention(d_k=d)
    x = np.random.randn(batch, seq_len, d).astype(np.float32)

    output, weights = attn(x, x, x)

    # When Q=K=V, output should be close to input (with softmax scaling)
    # Attention weights should be roughly uniform on diagonal
    print(f"  Output close to input: {np.allclose(output, x, atol=0.5)}")
    print(f"✓ Identity test passed")


def test_gradient_check():
    """Test gradients using numerical gradient checking."""
    batch, seq_len, d_k, d_v = 2, 4, 8, 8

    attn = ScaledDotProductAttention(d_k=d_k)
    q = np.random.randn(batch, seq_len, d_k).astype(np.float32)
    k = np.random.randn(batch, seq_len, d_k).astype(np.float32)
    v = np.random.randn(batch, seq_len, d_v).astype(np.float32)

    # Forward
    output, weights = attn(q, k, v)

    # Backward with random gradient
    dout = np.random.randn(*output.shape).astype(np.float32)
    dq, dk, dv = attn.backward(dout)

    # Numerical gradient check for query (sample)
    eps = 1e-4
    for i in range(min(2, batch)):
        for j in range(min(2, seq_len)):
            # f(q + eps)
            q[i, j, 0] += eps
            output_plus, _ = attn(q, k, v)
            loss_plus = np.sum(output_plus * dout)

            # f(q - eps)
            q[i, j, 0] -= 2 * eps
            output_minus, _ = attn(q, k, v)
            loss_minus = np.sum(output_minus * dout)

            q[i, j, 0] += eps

            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytical = dq[i, j, 0]
            error = abs(numerical - analytical) / (abs(numerical) + abs(analytical) + 1e-8)

            if error > 1e-2:
                print(f"  Gradient error at [{i},{j},0]: {error:.2e}")
                assert False

    print(f"✓ Gradient check test passed")


def test_different_sequence_lengths():
    """Test attention with different sequence lengths."""
    batch, d_k, d_v = 2, 32, 32

    attn = ScaledDotProductAttention(d_k=d_k)

    for seq_len in [4, 8, 16, 32]:
        q = np.random.randn(batch, seq_len, d_k).astype(np.float32)
        k = np.random.randn(batch, seq_len, d_k).astype(np.float32)
        v = np.random.randn(batch, seq_len, d_v).astype(np.float32)

        output, weights = attn(q, k, v)

        assert output.shape == (batch, seq_len, d_v)
        assert weights.shape == (batch, seq_len, seq_len)

    print(f"✓ Different sequence lengths test passed")


def test_multi_head_attention_simulation():
    """Simulate multi-head attention by splitting dimensions."""
    batch, seq_len, d_model = 2, 8, 64
    num_heads = 4
    d_k = d_model // num_heads
    d_v = d_model // num_heads

    # Simulate attention for each head
    attn = ScaledDotProductAttention(d_k=d_k)

    q = np.random.randn(batch, seq_len, d_model).astype(np.float32)
    k = np.random.randn(batch, seq_len, d_model).astype(np.float32)
    v = np.random.randn(batch, seq_len, d_model).astype(np.float32)

    # Process each head
    outputs = []
    for h in range(num_heads):
        q_h = q[:, :, h*d_k:(h+1)*d_k]
        k_h = k[:, :, h*d_k:(h+1)*d_k]
        v_h = v[:, :, h*d_k:(h+1)*d_v]

        output_h, _ = attn(q_h, k_h, v_h)
        outputs.append(output_h)

    # Concatenate
    final_output = np.concatenate(outputs, axis=2)
    assert final_output.shape == (batch, seq_len, d_model)

    print(f"✓ Multi-head attention simulation test passed")


if __name__ == "__main__":
    print("Running Scaled Dot-Product Attention tests...\n")

    test_output_shape()
    test_attention_weights_sum_to_one()
    test_causal_mask()
    test_identity_with_same_vectors()
    test_gradient_check()
    test_different_sequence_lengths()
    test_multi_head_attention_simulation()

    print("\n✓ All tests passed!")
