"""
PROBLEM: Implement Batch Normalization

Build batch normalization from scratch, understanding running statistics,
training vs inference modes, and backpropagation through normalization.

REQUIREMENTS:
- Implement forward pass with batch statistics
- Maintain running statistics (exponential moving average)
- Separate behavior for train vs eval modes
- Implement backward pass with gradient computation
- Support learnable scale (gamma) and shift (beta) parameters
- Handle numerical stability

PERFORMANCE NOTES:
- Should efficiently compute batch statistics
- Running average update should be efficient
- Should handle large batch sizes (1024+)

TEST CASE EXPECTATIONS:
- Output should be normalized (mean ~0, std ~1) in training
- Running statistics should converge after many batches
- Eval mode should use running statistics
- Gradient computation should pass numerical checks
- Should reduce internal covariate shift
"""

import numpy as np
from typing import Tuple, Optional


class BatchNorm1D:
    """Batch normalization for 1D inputs (fully-connected networks)."""

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.9,
        epsilon: float = 1e-5,
        dtype: np.dtype = np.float32,
    ):
        """
        Initialize batch norm.

        Args:
            num_features: Number of features
            momentum: Momentum for running statistics (default PyTorch value)
            epsilon: Small constant for numerical stability
            dtype: Data type for weights
        """
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon
        self.dtype = dtype

        # Learnable parameters
        self.gamma = np.ones(num_features, dtype=dtype)  # scale
        self.beta = np.zeros(num_features, dtype=dtype)  # shift

        # Gradients
        self.grad_gamma = None
        self.grad_beta = None

        # Running statistics (for eval)
        self.running_mean = np.zeros(num_features, dtype=dtype)
        self.running_var = np.ones(num_features, dtype=dtype)

        # Training flag
        self.is_training = True

        # Cache for backward
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, num_features)

        Returns:
            Normalized output (batch_size, num_features)
        """
        # TODO: Implement batch normalization forward pass
        # Training mode:
        #   1. Compute mean and variance of batch
        #   2. Normalize: (x - mean) / sqrt(var + eps)
        #   3. Scale and shift: gamma * normalized + beta
        #   4. Update running statistics
        # Eval mode:
        #   1. Normalize using running mean/var: (x - running_mean) / sqrt(running_var + eps)
        #   2. Scale and shift: gamma * normalized + beta
        # Cache intermediate values for backward
        pass

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            dout: Gradient w.r.t. output (batch_size, num_features)

        Returns:
            Gradient w.r.t. input (batch_size, num_features)
        """
        # TODO: Implement batch norm backward pass
        # Compute:
        # dL/dgamma = sum(dout * normalized, axis=0)
        # dL/dbeta = sum(dout, axis=0)
        # dL/dx = chain rule through normalization
        #
        # The backward through normalization is complex:
        # Need to account for mean and variance computation in the backward pass
        pass

    def set_train(self):
        """Set to training mode."""
        self.is_training = True

    def set_eval(self):
        """Set to evaluation mode."""
        self.is_training = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class BatchNorm2D:
    """Batch normalization for 2D inputs (convolutional networks)."""

    def __init__(
        self,
        num_channels: int,
        momentum: float = 0.9,
        epsilon: float = 1e-5,
        dtype: np.dtype = np.float32,
    ):
        """
        Initialize batch norm for 2D.

        Args:
            num_channels: Number of channels
            momentum: Momentum for running statistics
            epsilon: Small constant for stability
            dtype: Data type
        """
        self.num_channels = num_channels
        self.momentum = momentum
        self.epsilon = epsilon
        self.dtype = dtype

        # Learnable parameters
        self.gamma = np.ones(num_channels, dtype=dtype)
        self.beta = np.zeros(num_channels, dtype=dtype)

        # Gradients
        self.grad_gamma = None
        self.grad_beta = None

        # Running statistics
        self.running_mean = np.zeros(num_channels, dtype=dtype)
        self.running_var = np.ones(num_channels, dtype=dtype)

        self.is_training = True
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input (batch_size, height, width, channels)

        Returns:
            Normalized output
        """
        # TODO: Implement batch norm for 2D
        # Normalize across batch, height, width dimensions
        # Keep per-channel statistics
        # Similar to 1D but compute statistics differently
        pass

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass."""
        # TODO: Implement backward for 2D
        pass

    def set_train(self):
        self.is_training = True

    def set_eval(self):
        self.is_training = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


def test_output_shape():
    """Test output shape matches input."""
    batch_size, num_features = 32, 10

    bn = BatchNorm1D(num_features)
    x = np.random.randn(batch_size, num_features).astype(np.float32)
    y = bn(x)

    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
    print(f"✓ Output shape test passed")


def test_normalization_in_training():
    """Test that training mode normalizes correctly."""
    batch_size, num_features = 64, 20

    bn = BatchNorm1D(num_features)
    bn.set_train()

    x = np.random.randn(batch_size, num_features).astype(np.float32) * 10 + 5  # Large scale/shift
    y = bn(x)

    # Output should have mean ~0 and std ~1 per feature
    output_mean = y.mean(axis=0)
    output_std = y.std(axis=0)

    np.testing.assert_allclose(output_mean, 0, atol=0.1)
    np.testing.assert_allclose(output_std, 1.0, atol=0.1)

    print(f"✓ Normalization in training test passed (mean: {output_mean.mean():.3f}, std: {output_std.mean():.3f})")


def test_scale_and_shift():
    """Test that gamma and beta are applied correctly."""
    batch_size, num_features = 32, 10

    bn = BatchNorm1D(num_features)
    bn.gamma = np.array([2.0] * num_features, dtype=np.float32)
    bn.beta = np.array([3.0] * num_features, dtype=np.float32)

    x = np.random.randn(batch_size, num_features).astype(np.float32)
    y = bn(x)

    # With gamma=2 and beta=3, output = 2*(x-mean)/std + 3
    # Mean of output should be ~3, std should be ~2
    output_mean = y.mean(axis=0)
    output_std = y.std(axis=0)

    np.testing.assert_allclose(output_mean, 3.0, atol=0.2)
    np.testing.assert_allclose(output_std, 2.0, atol=0.2)

    print(f"✓ Scale and shift test passed")


def test_running_statistics():
    """Test that running statistics are updated correctly."""
    batch_size, num_features = 32, 10

    bn = BatchNorm1D(num_features, momentum=0.9)
    bn.set_train()

    # First batch
    x1 = np.random.randn(batch_size, num_features).astype(np.float32) + 10
    y1 = bn(x1)

    # Running stats should be updated
    mean_after_first = bn.running_mean.copy()
    var_after_first = bn.running_var.copy()

    # Second batch with different distribution
    x2 = np.random.randn(batch_size, num_features).astype(np.float32) - 5
    y2 = bn(x2)

    mean_after_second = bn.running_mean.copy()
    var_after_second = bn.running_var.copy()

    # Running statistics should have changed
    assert not np.allclose(mean_after_first, mean_after_second)
    assert not np.allclose(var_after_first, var_after_second)

    print(f"✓ Running statistics test passed")


def test_eval_mode():
    """Test evaluation mode uses running statistics."""
    batch_size, num_features = 32, 10

    bn = BatchNorm1D(num_features)

    # Train on some batches to build running stats
    bn.set_train()
    for _ in range(10):
        x = np.random.randn(batch_size, num_features).astype(np.float32) + 5
        _ = bn(x)

    # Save running stats
    running_mean = bn.running_mean.copy()
    running_var = bn.running_var.copy()

    # Eval mode
    bn.set_eval()
    x_eval = np.random.randn(batch_size, num_features).astype(np.float32)
    y_eval = bn(x_eval)

    # In eval, output should use running stats regardless of input distribution
    # Apply same transformation manually
    normalized = (x_eval - running_mean) / np.sqrt(running_var + bn.epsilon)
    expected = bn.gamma * normalized + bn.beta

    np.testing.assert_allclose(y_eval, expected, rtol=1e-5)

    print(f"✓ Eval mode test passed")


def test_backward_shape():
    """Test backward pass output shape."""
    batch_size, num_features = 32, 10

    bn = BatchNorm1D(num_features)
    bn.set_train()

    x = np.random.randn(batch_size, num_features).astype(np.float32)
    y = bn(x)

    dout = np.random.randn(batch_size, num_features).astype(np.float32)
    dx = bn.backward(dout)

    assert dx.shape == x.shape
    assert bn.grad_gamma.shape == (num_features,)
    assert bn.grad_beta.shape == (num_features,)

    print(f"✓ Backward shape test passed")


def test_gradient_check():
    """Test gradients using numerical gradient checking."""
    batch_size, num_features = 8, 4

    bn = BatchNorm1D(num_features)
    bn.set_train()

    x = np.random.randn(batch_size, num_features).astype(np.float32)

    # Forward and backward
    y = bn(x)
    dout = np.ones_like(y)
    dx = bn.backward(dout)

    # Numerical gradient check for gamma
    eps = 1e-4
    numerical_grad = np.zeros_like(bn.gamma)

    for i in range(num_features):
        # f(gamma + eps)
        bn.gamma[i] += eps
        y_plus = bn(x)
        loss_plus = np.sum(y_plus * dout)

        # f(gamma - eps)
        bn.gamma[i] -= 2 * eps
        y_minus = bn(x)
        loss_minus = np.sum(y_minus * dout)

        bn.gamma[i] += eps
        numerical_grad[i] = (loss_plus - loss_minus) / (2 * eps)

    relative_error = np.abs(bn.grad_gamma - numerical_grad) / (
        np.abs(bn.grad_gamma) + np.abs(numerical_grad) + 1e-8
    )

    assert np.allclose(relative_error, 0, atol=1e-2), f"Gradient check failed: {np.max(relative_error)}"
    print(f"✓ Gradient check test passed")


def test_2d_batch_norm():
    """Test batch norm for 2D inputs."""
    batch_size, height, width, num_channels = 4, 8, 8, 3

    bn = BatchNorm2D(num_channels)
    bn.set_train()

    x = np.random.randn(batch_size, height, width, num_channels).astype(np.float32)
    y = bn(x)

    assert y.shape == x.shape
    print(f"✓ 2D batch norm test passed")


if __name__ == "__main__":
    print("Running Batch Normalization tests...\n")

    test_output_shape()
    test_normalization_in_training()
    test_scale_and_shift()
    test_running_statistics()
    test_eval_mode()
    test_backward_shape()
    test_gradient_check()
    test_2d_batch_norm()

    print("\n✓ All tests passed!")
