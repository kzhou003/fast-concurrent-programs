"""
PROBLEM: Implement Dynamic Batching

Build a dynamic batching system that groups requests into efficient batches
for LLM inference. This is critical for maximizing GPU utilization while
respecting latency constraints.

REQUIREMENTS:
- Queue incoming requests with different input lengths
- Group requests into batches to minimize padding waste
- Support different batching strategies (FCFS, priority, SJF)
- Implement request dequeuing based on max_batch_size and max_wait_time
- Track request-to-batch mapping for response routing
- Support variable sequence lengths efficiently

PERFORMANCE NOTES:
- Should achieve >85% GPU utilization in typical scenarios
- Request latency should be bounded by max_wait_time
- Should minimize padding overhead (wasted compute)
- Throughput should scale with batch size efficiency

TEST CASE EXPECTATIONS:
- Requests should be grouped into batches
- Batches should not exceed max_batch_size
- Batches should be formed within max_wait_time
- Responses should be correctly routed to original requests
- Different batching strategies should be comparable
"""

import time
import heapq
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


@dataclass
class Request:
    """Represents an inference request."""

    request_id: int
    input_tokens: int  # Number of input tokens
    arrival_time: float = field(default_factory=time.time)
    processed: bool = False
    result: Optional[np.ndarray] = None
    batch_id: Optional[int] = None
    output_tokens: int = 128  # Expected output tokens

    @property
    def age(self) -> float:
        """Age of request in seconds."""
        return time.time() - self.arrival_time


@dataclass
class Batch:
    """Represents a batch of requests."""

    batch_id: int
    requests: List[Request] = field(default_factory=list)
    max_tokens: int = 0  # Maximum sequence length in batch
    total_tokens: int = 0  # Sum of all tokens (with padding)
    creation_time: float = field(default_factory=time.time)

    @property
    def size(self) -> int:
        """Number of requests in batch."""
        return len(self.requests)

    def add_request(self, request: Request):
        """Add request to batch."""
        self.requests.append(request)
        request.batch_id = self.batch_id
        self.max_tokens = max(self.max_tokens, request.input_tokens)
        self._recompute_total()

    def _recompute_total(self):
        """Recompute total tokens (with padding)."""
        if self.requests:
            self.total_tokens = self.max_tokens * len(self.requests)
        else:
            self.total_tokens = 0


class DynamicBatcher:
    """Dynamic batching system for LLM inference."""

    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_time: float = 0.1,  # seconds
        strategy: str = "fcfs",  # fcfs, sjf, priority
    ):
        """
        Initialize dynamic batcher.

        Args:
            max_batch_size: Maximum number of requests per batch
            max_wait_time: Maximum time to wait before forming a batch
            strategy: Batching strategy (fcfs, sjf, priority)
        """
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.strategy = strategy

        self.request_queue: List[Request] = []
        self.batches: Dict[int, Batch] = {}
        self.request_to_batch: Dict[int, int] = {}

        self.next_batch_id = 0
        self.next_request_id = 0

    def add_request(self, input_tokens: int) -> int:
        """
        Add a request to the queue.

        Args:
            input_tokens: Number of input tokens

        Returns:
            request_id for tracking
        """
        # TODO: Create request and add to queue
        request = Request(request_id=self.next_request_id, input_tokens=input_tokens)
        self.next_request_id += 1
        self.request_queue.append(request)
        return request.request_id

    def _should_form_batch(self, current_size: int, oldest_age: float) -> bool:
        """
        Determine if batch should be formed.

        Batch is formed if:
        - Reached max_batch_size, OR
        - Have requests AND oldest request exceeds max_wait_time
        """
        return current_size >= self.max_batch_size or (current_size > 0 and oldest_age >= self.max_wait_time)

    def _get_next_batch_requests(self) -> List[Request]:
        """
        Get next batch of requests based on strategy.

        TODO: Implement different batching strategies:
        - fcfs: First-Come-First-Served (just take first N)
        - sjf: Shortest-Job-First (sort by input_tokens)
        - priority: Oldest first (maximize fairness)
        """
        if not self.request_queue:
            return []

        # TODO: Sort or select based on strategy
        if self.strategy == "fcfs":
            # Take first max_batch_size requests
            batch_requests = self.request_queue[:self.max_batch_size]
        elif self.strategy == "sjf":
            # Sort by input tokens (shortest first)
            self.request_queue.sort(key=lambda r: r.input_tokens)
            batch_requests = self.request_queue[:self.max_batch_size]
        elif self.strategy == "priority":
            # Sort by age (oldest first)
            self.request_queue.sort(key=lambda r: r.arrival_time)
            batch_requests = self.request_queue[:self.max_batch_size]
        else:
            batch_requests = self.request_queue[:self.max_batch_size]

        # Remove from queue
        for req in batch_requests:
            self.request_queue.remove(req)

        return batch_requests

    def get_batch(self) -> Optional[Batch]:
        """
        Get next batch if conditions are met.

        Returns:
            Batch object or None if no batch ready
        """
        if not self.request_queue:
            return None

        # Check if should form batch
        oldest_request = min(self.request_queue, key=lambda r: r.arrival_time)
        oldest_age = oldest_request.age

        if not self._should_form_batch(len(self.request_queue), oldest_age):
            return None

        # TODO: Get requests for batch
        batch_requests = self._get_next_batch_requests()

        if not batch_requests:
            return None

        # TODO: Create batch
        batch = Batch(batch_id=self.next_batch_id)
        self.next_batch_id += 1

        for request in batch_requests:
            batch.add_request(request)
            self.request_to_batch[request.request_id] = batch.batch_id

        self.batches[batch.batch_id] = batch

        return batch

    def queue_size(self) -> int:
        """Return number of pending requests."""
        return len(self.request_queue)

    def pending_batches(self) -> int:
        """Return number of formed batches waiting for processing."""
        return len(self.batches)


class BatchProcessor:
    """Simulates LLM model inference."""

    @staticmethod
    def process_batch(batch: Batch) -> Dict[int, np.ndarray]:
        """
        Simulate processing a batch.

        Args:
            batch: Batch to process

        Returns:
            Dictionary mapping request_id to output
        """
        # TODO: Simulate inference on batch
        # Return dummy outputs for each request in batch
        results = {}
        for request in batch.requests:
            # Simulate output generation
            output = np.random.randn(request.output_tokens, 4096).astype(np.float32)
            results[request.request_id] = output
        return results


def test_basic_batching():
    """Test basic batching functionality."""
    batcher = DynamicBatcher(max_batch_size=4, max_wait_time=0.01)

    # Add 10 requests
    for i in range(10):
        batcher.add_request(input_tokens=100 + i * 10)

    # Get first batch
    batch = batcher.get_batch()

    assert batch is not None
    assert batch.size <= batcher.max_batch_size
    assert batcher.queue_size() == 6  # 10 - 4

    print(f"✓ Basic batching test passed (batch size: {batch.size})")


def test_max_batch_size_respected():
    """Test that max_batch_size is not exceeded."""
    batcher = DynamicBatcher(max_batch_size=8, max_wait_time=10.0)

    # Add many requests
    for i in range(100):
        batcher.add_request(input_tokens=100)

    # Get multiple batches
    batches = []
    while True:
        batch = batcher.get_batch()
        if batch is None:
            break
        batches.append(batch)

    # All batches should respect max_batch_size
    for batch in batches:
        assert batch.size <= batcher.max_batch_size, f"Batch size {batch.size} exceeds max {batcher.max_batch_size}"

    # Total requests should match
    total_requests = sum(b.size for b in batches)
    assert total_requests == 100

    print(f"✓ Max batch size test passed ({len(batches)} batches)")


def test_wait_time_respected():
    """Test that requests don't wait longer than max_wait_time."""
    batcher = DynamicBatcher(max_batch_size=100, max_wait_time=0.05)

    # Add one request
    batcher.add_request(input_tokens=100)

    # Wait a bit but not max_wait_time
    time.sleep(0.02)
    batch = batcher.get_batch()
    assert batch is None, "Batch formed too early"

    # Wait until max_wait_time exceeded
    time.sleep(0.04)
    batch = batcher.get_batch()
    assert batch is not None, "Batch not formed after max_wait_time"
    assert batch.size == 1

    print(f"✓ Wait time test passed")


def test_request_routing():
    """Test that responses are correctly routed to requests."""
    batcher = DynamicBatcher(max_batch_size=4, max_wait_time=10.0)

    # Add requests
    request_ids = []
    for i in range(10):
        rid = batcher.add_request(input_tokens=100)
        request_ids.append(rid)

    # Get batch and process
    batch = batcher.get_batch()
    assert batch is not None

    results = BatchProcessor.process_batch(batch)

    # Verify all requests in batch got results
    for request in batch.requests:
        assert request.request_id in results
        assert results[request.request_id].shape[0] == request.output_tokens

    print(f"✓ Request routing test passed ({batch.size} requests)")


def test_variable_length_batching():
    """Test batching with variable length inputs."""
    batcher = DynamicBatcher(max_batch_size=4, max_wait_time=10.0)

    # Add requests with different lengths
    lengths = [50, 100, 150, 200, 75, 125, 175]
    for length in lengths:
        batcher.add_request(input_tokens=length)

    # Get first batch
    batch = batcher.get_batch()

    # Check that max_tokens is set correctly
    expected_max = max([r.input_tokens for r in batch.requests])
    assert batch.max_tokens == expected_max

    # Check total_tokens includes padding
    expected_total = batch.max_tokens * batch.size
    assert batch.total_tokens == expected_total

    print(f"✓ Variable length batching test passed (max_tokens: {batch.max_tokens})")


def test_fcfs_strategy():
    """Test FCFS batching strategy."""
    batcher = DynamicBatcher(max_batch_size=3, max_wait_time=10.0, strategy="fcfs")

    # Add requests
    for i in range(10):
        batcher.add_request(input_tokens=100 + i * 50)

    # Get batch
    batch = batcher.get_batch()

    # In FCFS, should get first 3 requests
    assert batch.size == 3
    assert batch.requests[0].input_tokens == 100
    assert batch.requests[1].input_tokens == 150

    print(f"✓ FCFS strategy test passed")


def test_sjf_strategy():
    """Test Shortest-Job-First batching strategy."""
    batcher = DynamicBatcher(max_batch_size=3, max_wait_time=10.0, strategy="sjf")

    # Add requests in random order
    lengths = [200, 50, 150, 100, 75]
    for length in lengths:
        batcher.add_request(input_tokens=length)

    # Get batch
    batch = batcher.get_batch()

    # In SJF, should get shortest jobs: 50, 75, 100
    batch_lengths = sorted([r.input_tokens for r in batch.requests])
    expected_lengths = [50, 75, 100]
    assert batch_lengths == expected_lengths, f"Expected {expected_lengths}, got {batch_lengths}"

    print(f"✓ SJF strategy test passed")


def test_throughput():
    """Test throughput of batching system."""
    batcher = DynamicBatcher(max_batch_size=32, max_wait_time=0.01)

    # Add many requests
    num_requests = 1000
    start = time.time()

    for i in range(num_requests):
        batcher.add_request(input_tokens=100)

    # Process all batches
    total_processed = 0
    while True:
        batch = batcher.get_batch()
        if batch is None:
            if batcher.queue_size() > 0:
                time.sleep(0.005)
                continue
            break
        total_processed += batch.size

    elapsed = time.time() - start

    assert total_processed == num_requests
    print(f"✓ Throughput test passed ({num_requests} requests in {elapsed:.3f}s)")


def test_empty_queue():
    """Test behavior with empty queue."""
    batcher = DynamicBatcher(max_batch_size=4)

    assert batcher.queue_size() == 0
    assert batcher.get_batch() is None

    print(f"✓ Empty queue test passed")


if __name__ == "__main__":
    print("Running Dynamic Batching tests...\n")

    test_basic_batching()
    test_max_batch_size_respected()
    test_wait_time_respected()
    test_request_routing()
    test_variable_length_batching()
    test_fcfs_strategy()
    test_sjf_strategy()
    test_throughput()
    test_empty_queue()

    print("\n✓ All tests passed!")
