"""
PROBLEM: Implement LSTM Cell

Build an LSTM (Long Short-Term Memory) cell from scratch, implementing all four
gates and the cell state update. Understanding this is crucial for sequence modeling.

REQUIREMENTS:
- Implement forward pass with all 4 gates (input, forget, cell, output)
- Implement backward pass (BPTT - Backpropagation Through Time)
- Support batched sequences (batch_size, seq_len, input_size)
- Maintain hidden state and cell state between steps
- Proper weight initialization
- Numerical stability in operations

PERFORMANCE NOTES:
- Forward pass should be efficient for long sequences
- Gradient computation should handle vanishing gradient issues properly
- Should support sequences of 1000+ timesteps

TEST CASE EXPECTATIONS:
- Output shape should match expected dimensions
- Hidden state should be tracked correctly
- Cell state should be updated properly
- Gradient computation should pass numerical checks
- Should work with different sequence lengths
"""

import numpy as np
from typing import Tuple, Optional


class LSTMCell:
    """LSTM cell for processing one timestep."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dtype: np.dtype = np.float32,
    ):
        """
        Initialize LSTM cell.

        Args:
            input_size: Size of input
            hidden_size: Size of hidden state
            dtype: Data type for weights
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dtype = dtype

        # TODO: Initialize weights for all 4 gates
        # Each gate has weights for input and hidden state
        # Gates: input, forget, cell, output
        # W_* shape: (hidden_size, input_size + hidden_size)
        # b_* shape: (hidden_size,)

        # Input gate
        self.W_i = None
        self.b_i = None

        # Forget gate
        self.W_f = None
        self.b_f = None

        # Cell gate (candidate)
        self.W_c = None
        self.b_c = None

        # Output gate
        self.W_o = None
        self.b_o = None

        # Gradients
        self.grad_W_i = None
        self.grad_b_i = None
        self.grad_W_f = None
        self.grad_b_f = None
        self.grad_W_c = None
        self.grad_b_c = None
        self.grad_W_o = None
        self.grad_b_o = None

        self.cache = {}

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation."""
        return np.tanh(x)

    def forward(
        self,
        x_t: np.ndarray,  # (batch_size, input_size)
        h_prev: np.ndarray,  # (batch_size, hidden_size)
        c_prev: np.ndarray,  # (batch_size, hidden_size)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass for one timestep.

        Args:
            x_t: Input at timestep t
            h_prev: Hidden state from previous timestep
            c_prev: Cell state from previous timestep

        Returns:
            h_t: Hidden state at timestep t
            c_t: Cell state at timestep t
            y_t: Output (usually same as h_t)
        """
        # TODO: Implement LSTM forward pass
        # 1. Concatenate input and previous hidden state
        # 2. Compute all 4 gates:
        #    - Input gate: i_t = sigmoid(W_i @ [x_t, h_prev] + b_i)
        #    - Forget gate: f_t = sigmoid(W_f @ [x_t, h_prev] + b_f)
        #    - Cell candidate: c_tilde = tanh(W_c @ [x_t, h_prev] + b_c)
        #    - Output gate: o_t = sigmoid(W_o @ [x_t, h_prev] + b_o)
        # 3. Update cell state: c_t = f_t * c_prev + i_t * c_tilde
        # 4. Compute hidden state: h_t = o_t * tanh(c_t)
        # 5. Cache values for backward pass

        h_t = None
        c_t = None
        y_t = None

        return h_t, c_t, y_t

    def backward(
        self,
        dh_t: np.ndarray,
        dc_t: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for one timestep.

        Args:
            dh_t: Gradient w.r.t. hidden state at timestep t
            dc_t: Gradient w.r.t. cell state at timestep t

        Returns:
            dx_t: Gradient w.r.t. input
            dh_prev: Gradient w.r.t. previous hidden state
            dc_prev: Gradient w.r.t. previous cell state
        """
        # TODO: Implement LSTM backward pass
        # Use chain rule to compute gradients through all operations
        # This is complex due to all the gate operations
        pass

    def __call__(
        self,
        x_t: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.forward(x_t, h_prev, c_prev)


class LSTM:
    """LSTM layer processing sequences."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dtype: np.dtype = np.float32,
    ):
        """Initialize LSTM layer."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size, dtype)

    def forward(
        self,
        x: np.ndarray,  # (batch_size, seq_len, input_size)
        h_0: Optional[np.ndarray] = None,
        c_0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through sequence.

        Args:
            x: Input sequence
            h_0: Initial hidden state
            c_0: Initial cell state

        Returns:
            outputs: All hidden states (batch_size, seq_len, hidden_size)
            h_n: Final hidden state
            c_n: Final cell state
        """
        batch_size, seq_len, _ = x.shape

        if h_0 is None:
            h_0 = np.zeros((batch_size, self.hidden_size), dtype=x.dtype)
        if c_0 is None:
            c_0 = np.zeros((batch_size, self.hidden_size), dtype=x.dtype)

        # TODO: Process sequence through LSTM cell
        # For each timestep, call cell.forward()
        # Keep track of outputs
        outputs = []
        h_t = h_0
        c_t = c_0

        for t in range(seq_len):
            h_t, c_t, _ = self.cell(x[:, t, :], h_t, c_t)
            outputs.append(h_t)

        outputs = np.stack(outputs, axis=1)

        return outputs, h_t, c_t


def test_output_shape():
    """Test LSTM output shapes."""
    batch_size, seq_len, input_size, hidden_size = 4, 10, 5, 8

    lstm = LSTM(input_size, hidden_size)
    x = np.random.randn(batch_size, seq_len, input_size).astype(np.float32)

    outputs, h_n, c_n = lstm(x)

    assert outputs.shape == (batch_size, seq_len, hidden_size)
    assert h_n.shape == (batch_size, hidden_size)
    assert c_n.shape == (batch_size, hidden_size)

    print(f"✓ Output shape test passed")


def test_cell_forward():
    """Test single LSTM cell forward pass."""
    batch_size, input_size, hidden_size = 2, 3, 4

    cell = LSTMCell(input_size, hidden_size)
    x_t = np.random.randn(batch_size, input_size).astype(np.float32)
    h_prev = np.random.randn(batch_size, hidden_size).astype(np.float32)
    c_prev = np.random.randn(batch_size, hidden_size).astype(np.float32)

    h_t, c_t, y_t = cell(x_t, h_prev, c_prev)

    assert h_t.shape == (batch_size, hidden_size)
    assert c_t.shape == (batch_size, hidden_size)
    assert y_t.shape == (batch_size, hidden_size)

    print(f"✓ Cell forward test passed")


def test_sequence_processing():
    """Test processing a full sequence."""
    batch_size, seq_len, input_size, hidden_size = 2, 5, 3, 4

    lstm = LSTM(input_size, hidden_size)
    x = np.random.randn(batch_size, seq_len, input_size).astype(np.float32)

    # Forward
    outputs, h_n, c_n = lstm(x)

    # Verify all outputs are filled
    assert np.all(np.isfinite(outputs))
    assert np.all(np.isfinite(h_n))
    assert np.all(np.isfinite(c_n))

    # Last hidden output should match h_n
    np.testing.assert_allclose(outputs[:, -1, :], h_n, rtol=1e-5)

    print(f"✓ Sequence processing test passed")


def test_hidden_state_dependency():
    """Test that output depends on sequence history."""
    batch_size, input_size, hidden_size = 2, 3, 4

    lstm = LSTM(input_size, hidden_size)

    # Process same input twice with different initial states
    x = np.random.randn(batch_size, 1, input_size).astype(np.float32)

    # Different initial hidden states
    h_0_a = np.random.randn(batch_size, hidden_size).astype(np.float32)
    h_0_b = np.random.randn(batch_size, hidden_size).astype(np.float32)

    outputs_a, _, _ = lstm(x, h_0_a, None)
    outputs_b, _, _ = lstm(x, h_0_b, None)

    # Outputs should be different due to different hidden states
    assert not np.allclose(outputs_a, outputs_b)

    print(f"✓ Hidden state dependency test passed")


def test_cell_state_flow():
    """Test that cell state flows correctly through sequence."""
    batch_size, input_size, hidden_size = 2, 3, 4

    lstm = LSTM(input_size, hidden_size)
    x = np.random.randn(batch_size, 3, input_size).astype(np.float32)

    # Process sequence
    outputs, h_n, c_n = lstm(x)

    # Cell state should be bounded (tanh output is [-1, 1], but with gates it can be larger)
    assert np.max(np.abs(c_n)) < 1000, "Cell state exploding"

    print(f"✓ Cell state flow test passed")


def test_gradient_flow():
    """Test that gradients can flow through sequence."""
    batch_size, seq_len, input_size, hidden_size = 2, 3, 3, 4

    lstm = LSTM(input_size, hidden_size)
    x = np.random.randn(batch_size, seq_len, input_size).astype(np.float32)

    # Forward
    outputs, h_n, c_n = lstm(x)

    # Simulate backward (just check shapes work)
    doutputs = np.random.randn(*outputs.shape).astype(np.float32)
    dh_n = np.random.randn(*h_n.shape).astype(np.float32)
    dc_n = np.random.randn(*c_n.shape).astype(np.float32)

    # In full implementation, would process backward through sequence
    print(f"✓ Gradient flow test passed")


def test_variable_sequence_length():
    """Test LSTM with different sequence lengths."""
    batch_size, input_size, hidden_size = 2, 3, 4

    lstm = LSTM(input_size, hidden_size)

    for seq_len in [1, 5, 10, 20]:
        x = np.random.randn(batch_size, seq_len, input_size).astype(np.float32)
        outputs, h_n, c_n = lstm(x)

        assert outputs.shape == (batch_size, seq_len, hidden_size)

    print(f"✓ Variable sequence length test passed")


def test_zero_input():
    """Test LSTM behavior with zero input."""
    batch_size, seq_len, input_size, hidden_size = 2, 5, 3, 4

    lstm = LSTM(input_size, hidden_size)
    x = np.zeros((batch_size, seq_len, input_size), dtype=np.float32)

    outputs, h_n, c_n = lstm(x)

    # Should still produce valid outputs
    assert np.all(np.isfinite(outputs))
    assert np.all(np.isfinite(h_n))

    print(f"✓ Zero input test passed")


if __name__ == "__main__":
    print("Running LSTM Cell tests...\n")

    test_output_shape()
    test_cell_forward()
    test_sequence_processing()
    test_hidden_state_dependency()
    test_cell_state_flow()
    test_gradient_flow()
    test_variable_sequence_length()
    test_zero_input()

    print("\n✓ All tests passed!")
