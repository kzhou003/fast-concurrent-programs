"""
PROBLEM: Implement a 2D Convolution Layer

Build a Conv2D layer from scratch, implementing the core convolution operation
used extensively in CNNs. This requires understanding of tensor operations and
the convolution mathematics.

REQUIREMENTS:
- Implement forward pass with convolution operation
- Support different kernel sizes, strides, and padding
- Implement backward pass for all learnable parameters
- Support multiple input/output channels
- Handle batched inputs: (batch, height, width, channels)
- Proper weight initialization

PERFORMANCE NOTES:
- Should efficiently use numpy broadcasting where possible
- Memory usage for im2col transformation should be reasonable
- Should handle standard layer sizes (3x3, 5x5 kernels, etc.)

TEST CASE EXPECTATIONS:
- Output shape should match expected dimensions
- Gradient computation should pass numerical gradient checks
- Should work with different kernel sizes and strides
- Training on simple task should converge
"""

import numpy as np
from typing import Tuple, Optional


class Conv2D:
    """2D Convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dtype: np.dtype = np.float32,
    ):
        """
        Initialize Conv2D layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel (square)
            stride: Stride of convolution
            padding: Zero padding to add
            dtype: Data type for weights
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dtype = dtype

        # TODO: Initialize weights (out_channels, in_channels, kernel_size, kernel_size)
        # Use He initialization: N(0, sqrt(2 / (in_channels * kernel_size^2)))
        self.weight = None
        self.bias = None

        # Gradients
        self.grad_weight = None
        self.grad_bias = None

        # Cache for backward
        self.cache = {}

    def _pad_input(self, x: np.ndarray) -> np.ndarray:
        """Add zero padding to input."""
        # TODO: Implement zero padding
        # x shape: (batch, height, width, channels)
        # Return padded x
        pass

    def _get_output_shape(self, x: np.ndarray) -> Tuple[int, int]:
        """Calculate output height and width."""
        batch, height, width, _ = x.shape
        h_out = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        return h_out, w_out

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, height, width, in_channels)

        Returns:
            Output tensor (batch, out_height, out_width, out_channels)
        """
        # TODO: Implement convolution forward pass
        # 1. Add padding
        # 2. For each output position, perform element-wise multiply and sum
        # 3. Add bias
        # 4. Cache inputs for backward pass
        #
        # Naive implementation is acceptable for learning:
        # For each batch, h, w:
        #   For each output channel:
        #     For each kernel position (kh, kw):
        #       output[batch, h, w, oc] += input[batch, h:h+k, w:w+k, :] * weight[oc, :, kh, kw]
        pass

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            dout: Gradient w.r.t. output (batch, out_h, out_w, out_channels)

        Returns:
            Gradient w.r.t. input (batch, in_h, in_w, in_channels)
        """
        # TODO: Implement convolution backward pass
        # Compute:
        # - dL/dW: gradient w.r.t. weights
        # - dL/db: gradient w.r.t. bias
        # - dL/dX: gradient w.r.t. input (via transposed convolution)
        pass

    def update(self, learning_rate: float):
        """Update weights and bias."""
        # TODO: Implement gradient descent
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


def test_forward_shape():
    """Test output shape with various configurations."""
    batch, in_h, in_w, in_c = 2, 28, 28, 3
    out_c, k, s, p = 16, 3, 1, 0

    layer = Conv2D(in_c, out_c, kernel_size=k, stride=s, padding=p)
    x = np.random.randn(batch, in_h, in_w, in_c).astype(np.float32)
    y = layer(x)

    expected_h = (in_h + 2*p - k) // s + 1
    expected_w = (in_w + 2*p - k) // s + 1

    assert y.shape == (batch, expected_h, expected_w, out_c), \
        f"Expected {(batch, expected_h, expected_w, out_c)}, got {y.shape}"
    print(f"✓ Forward shape test passed: {x.shape} -> {y.shape}")


def test_forward_with_padding_stride():
    """Test forward pass with different padding and stride values."""
    configs = [
        # (kernel_size, stride, padding)
        (3, 1, 0),
        (3, 1, 1),
        (3, 2, 1),
        (5, 2, 2),
    ]

    batch, h, w, in_c, out_c = 1, 28, 28, 3, 16

    for k, s, p in configs:
        layer = Conv2D(in_c, out_c, kernel_size=k, stride=s, padding=p)
        x = np.random.randn(batch, h, w, in_c).astype(np.float32)
        y = layer(x)

        expected_h = (h + 2*p - k) // s + 1
        expected_w = (w + 2*p - k) // s + 1

        assert y.shape[1] == expected_h and y.shape[2] == expected_w
        print(f"  k={k}, s={s}, p={p}: {(h,w)} -> {(expected_h, expected_w)} ✓")

    print(f"✓ Padding and stride test passed")


def test_backward_shape():
    """Test backward pass output shape."""
    batch, in_h, in_w, in_c = 4, 16, 16, 3
    out_c, k, s, p = 8, 3, 1, 1

    layer = Conv2D(in_c, out_c, kernel_size=k, stride=s, padding=p)
    x = np.random.randn(batch, in_h, in_w, in_c).astype(np.float32)
    y = layer(x)

    dout = np.random.randn(*y.shape).astype(np.float32)
    dx = layer.backward(dout)

    assert dx.shape == x.shape, f"Expected {x.shape}, got {dx.shape}"
    assert layer.grad_weight.shape == layer.weight.shape
    assert layer.grad_bias.shape == (out_c,)

    print(f"✓ Backward shape test passed")


def test_single_channel_convolution():
    """Test convolution with single channel for easy verification."""
    # Create simple 1x1 conv on single input
    in_c, out_c, k, s, p = 1, 1, 3, 1, 0

    layer = Conv2D(in_c, out_c, kernel_size=k, stride=s, padding=p)

    # Set known weights
    layer.weight = np.ones((1, 1, 3, 3), dtype=np.float32)
    layer.bias = np.zeros(1, dtype=np.float32)

    # Simple input
    x = np.ones((1, 5, 5, 1), dtype=np.float32)
    y = layer(x)

    # Output should be all 9s (sum of 3x3 kernel of ones)
    expected = np.full((1, 3, 3, 1), 9.0, dtype=np.float32)
    np.testing.assert_allclose(y, expected, rtol=1e-5)

    print(f"✓ Single channel convolution test passed")


def test_gradient_check():
    """Test gradients using numerical gradient checking."""
    in_c, out_c, k, s, p = 2, 2, 3, 1, 0
    batch, h, w = 1, 5, 5

    layer = Conv2D(in_c, out_c, kernel_size=k, stride=s, padding=p)
    x = np.random.randn(batch, h, w, in_c).astype(np.float32)

    # Forward and backward
    y = layer(x)
    dout = np.random.randn(*y.shape).astype(np.float32)
    layer.backward(dout)

    # Numerical gradient check (sample a few weights)
    eps = 1e-4
    max_error = 0

    for i in range(min(2, layer.weight.shape[0])):
        for j in range(min(2, layer.weight.shape[1])):
            # f(w + eps)
            layer.weight[i, j, 0, 0] += eps
            y_plus = layer(x)
            loss_plus = np.sum(y_plus * dout)

            # f(w - eps)
            layer.weight[i, j, 0, 0] -= 2 * eps
            y_minus = layer(x)
            loss_minus = np.sum(y_minus * dout)

            layer.weight[i, j, 0, 0] += eps

            numerical_grad = (loss_plus - loss_minus) / (2 * eps)
            analytical_grad = layer.grad_weight[i, j, 0, 0]
            error = abs(numerical_grad - analytical_grad) / (abs(numerical_grad) + abs(analytical_grad) + 1e-8)
            max_error = max(max_error, error)

    assert max_error < 1e-2, f"Gradient check failed: max error = {max_error}"
    print(f"✓ Gradient check test passed (max error: {max_error:.2e})")


def test_conv_followed_by_pooling():
    """Test that conv can be used as part of a pipeline."""
    batch, h, w, in_c, out_c = 2, 28, 28, 3, 16

    conv = Conv2D(in_c, out_c, kernel_size=3, stride=1, padding=1)
    x = np.random.randn(batch, h, w, in_c).astype(np.float32)

    # Forward
    y = conv(x)
    assert y.shape == (batch, h, w, out_c)

    # Simulate pooling (just for shape check)
    y_pooled = y[:, ::2, ::2, :]
    assert y_pooled.shape == (batch, h//2, w//2, out_c)

    print(f"✓ Conv + pooling pipeline test passed")


if __name__ == "__main__":
    print("Running Conv2D Layer tests...\n")

    test_forward_shape()
    test_forward_with_padding_stride()
    test_backward_shape()
    test_single_channel_convolution()
    test_gradient_check()
    test_conv_followed_by_pooling()

    print("\n✓ All tests passed!")
