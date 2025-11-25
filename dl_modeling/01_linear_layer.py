"""
PROBLEM: Implement a Linear Layer from Scratch

Build a Linear (fully-connected) layer that mimics PyTorch's nn.Linear.
This is a fundamental building block for neural networks.

REQUIREMENTS:
- Implement forward pass: Y = X @ W.T + b
- Implement backward pass with gradient computation
- Support batched operations (batch_size, in_features) -> (batch_size, out_features)
- Initialize weights properly (Xavier/Kaiming initialization)
- Track gradients for both weight and bias
- Support in-place operations where applicable

PERFORMANCE NOTES:
- Forward pass should use efficient matrix multiplication
- Backward pass should compute all gradients correctly
- Should handle large batch sizes (1024+) efficiently
- Memory usage should be reasonable for intermediate activations

TEST CASE EXPECTATIONS:
- Forward pass should match PyTorch's output (within numerical precision)
- Gradient computation should pass numerical gradient checking
- Training loop should converge on simple task (e.g., linear regression)
- Backward pass should work correctly with chain rule
"""

import numpy as np
from typing import Tuple, Optional


class Linear:
    """Linear layer: Y = X @ W.T + b"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: np.dtype = np.float32,
    ):
        """
        Initialize linear layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to include bias term
            dtype: Data type for weights
        """
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.dtype = dtype

        # TODO: Initialize weights with proper initialization
        # Use Xavier uniform: U(-sqrt(k), sqrt(k)) where k = 1/(in_features)
        self.weight = None  # Shape: (out_features, in_features)
        self.bias_param = None  # Shape: (out_features,)

        # Gradients
        self.grad_weight = None
        self.grad_bias = None

        # Cache for backward pass
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # TODO: Implement forward pass
        # Y = X @ W.T + b
        # Cache X for backward pass
        pass

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            dout: Gradient of loss w.r.t. output (batch_size, out_features)

        Returns:
            Gradient of loss w.r.t. input (batch_size, in_features)
        """
        # TODO: Implement backward pass
        # Compute gradients w.r.t. weight, bias, and input
        # dL/dW = dout.T @ X
        # dL/db = sum(dout, axis=0)
        # dL/dX = dout @ W
        pass

    def update(self, learning_rate: float):
        """Update weights and biases using computed gradients."""
        # TODO: Implement gradient descent update
        # W -= lr * dL/dW
        # b -= lr * dL/db
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


def test_forward_shape():
    """Test forward pass output shape."""
    batch_size, in_features, out_features = 32, 10, 5

    layer = Linear(in_features, out_features)
    x = np.random.randn(batch_size, in_features).astype(np.float32)
    y = layer(x)

    assert y.shape == (batch_size, out_features), f"Expected {(batch_size, out_features)}, got {y.shape}"
    print(f"✓ Forward shape test passed")


def test_forward_values():
    """Test forward pass computation against manual calculation."""
    in_features, out_features = 3, 2
    batch_size = 2

    layer = Linear(in_features, out_features, bias=True)

    # Set known weights for testing
    layer.weight = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    layer.bias_param = np.array([0.1, 0.2], dtype=np.float32)

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    y = layer(x)

    # Manual calculation
    expected = x @ layer.weight.T + layer.bias_param

    np.testing.assert_allclose(y, expected, rtol=1e-5)
    print(f"✓ Forward values test passed")


def test_backward_shape():
    """Test backward pass output shape."""
    batch_size, in_features, out_features = 32, 10, 5

    layer = Linear(in_features, out_features)
    x = np.random.randn(batch_size, in_features).astype(np.float32)
    y = layer(x)

    dout = np.random.randn(batch_size, out_features).astype(np.float32)
    dx = layer.backward(dout)

    assert dx.shape == x.shape, f"Expected {x.shape}, got {dx.shape}"
    assert layer.grad_weight.shape == layer.weight.shape
    assert layer.grad_bias.shape == layer.bias_param.shape

    print(f"✓ Backward shape test passed")


def test_gradient_check():
    """Test gradients using numerical gradient checking."""
    in_features, out_features, batch_size = 5, 3, 4

    layer = Linear(in_features, out_features)
    x = np.random.randn(batch_size, in_features).astype(np.float32)

    # Forward and backward
    y = layer(x)
    dout = np.random.randn(batch_size, out_features).astype(np.float32)
    layer.backward(dout)

    # Numerical gradient check for weights
    eps = 1e-5
    numerical_grad = np.zeros_like(layer.weight)

    for i in range(layer.weight.shape[0]):
        for j in range(layer.weight.shape[1]):
            # f(w + eps)
            layer.weight[i, j] += eps
            y_plus = layer(x)
            loss_plus = np.sum(y_plus * dout)

            # f(w - eps)
            layer.weight[i, j] -= 2 * eps
            y_minus = layer(x)
            loss_minus = np.sum(y_minus * dout)

            # Restore
            layer.weight[i, j] += eps

            numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * eps)

    # Compare analytical and numerical gradients
    relative_error = np.abs(layer.grad_weight - numerical_grad) / (
        np.abs(layer.grad_weight) + np.abs(numerical_grad) + 1e-8
    )

    assert np.allclose(relative_error, 0, atol=1e-3), f"Gradient check failed: max error = {np.max(relative_error)}"
    print(f"✓ Gradient check test passed (max relative error: {np.max(relative_error):.2e})")


def test_linear_regression():
    """Test that the layer can learn a simple linear relationship."""
    np.random.seed(42)

    in_features, out_features = 5, 1
    layer = Linear(in_features, out_features)

    # Generate synthetic data: y = 2*x1 + 3*x2 - 1*x3 + 0.5
    n_samples = 1000
    x = np.random.randn(n_samples, in_features).astype(np.float32)
    true_weights = np.array([2.0, 3.0, -1.0, 0.5, 0.2], dtype=np.float32)
    true_bias = 0.5
    y_true = x @ true_weights + true_bias

    # Train for a few iterations
    learning_rate = 0.01
    for epoch in range(100):
        # Forward
        y_pred = layer(x)

        # Loss
        loss = np.mean((y_pred - y_true.reshape(-1, 1)) ** 2)

        # Backward
        dout = 2 * (y_pred - y_true.reshape(-1, 1)) / n_samples
        layer.backward(dout)

        # Update
        layer.update(learning_rate)

    # Check final loss
    y_pred_final = layer(x)
    final_loss = np.mean((y_pred_final - y_true.reshape(-1, 1)) ** 2)

    assert final_loss < 0.1, f"Final loss too high: {final_loss}"
    print(f"✓ Linear regression test passed (final loss: {final_loss:.4f})")


def test_batch_independence():
    """Test that batches are processed independently."""
    in_features, out_features, batch_size = 3, 2, 4

    layer = Linear(in_features, out_features)
    x = np.random.randn(batch_size, in_features).astype(np.float32)

    # Process batch
    y_batch = layer(x)

    # Process individual samples
    y_individual = []
    for i in range(batch_size):
        y_i = layer(x[i:i+1])
        y_individual.append(y_i)

    y_individual = np.concatenate(y_individual, axis=0)

    np.testing.assert_allclose(y_batch, y_individual, rtol=1e-5)
    print(f"✓ Batch independence test passed")


if __name__ == "__main__":
    print("Running Linear Layer tests...\n")

    test_forward_shape()
    test_forward_values()
    test_backward_shape()
    test_gradient_check()
    test_linear_regression()
    test_batch_independence()

    print("\n✓ All tests passed!")
