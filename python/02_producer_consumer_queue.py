"""
PROBLEM: Producer-Consumer Queue

Implement a thread-safe queue where producer threads add items and consumer threads
process them. This is a classic concurrent programming pattern.

REQUIREMENTS:
- Implement a bounded queue with a maximum size
- Producers should block when queue is full
- Consumers should block when queue is empty
- Must handle proper signaling between producers and consumers
- Should use threading.Condition or similar synchronization primitives

PERFORMANCE NOTES:
- Should handle 100+ items per second throughput
- Producers and consumers should not busy-wait
- Should minimize lock contention

TEST CASE EXPECTATIONS:
- Single producer adding 100 items, single consumer processing all should complete correctly
- Multiple producers and consumers should maintain correct ordering/counts
- Queue of size 5 with 3 producers and 3 consumers should not overflow
"""

import threading
import time
from typing import Any, Optional


class ThreadSafeQueue:
    """A thread-safe queue for producer-consumer patterns."""

    def __init__(self, max_size: int = 10):
        """
        Initialize the queue.

        Args:
            max_size: Maximum number of items the queue can hold
        """
        self.max_size = max_size
        # TODO: Initialize queue, locks, and condition variables
        self.items = []

    def put(self, item: Any, timeout: Optional[float] = None) -> bool:
        """
        Add an item to the queue. Blocks if queue is full.

        Args:
            item: Item to add to the queue
            timeout: Maximum time to wait (None = wait forever)

        Returns:
            True if item was added, False if timeout occurred
        """
        # TODO: Implement with proper synchronization
        # Block if queue is full
        # Signal waiting consumers that an item is available
        pass

    def get(self, timeout: Optional[float] = None) -> tuple[bool, Any]:
        """
        Remove and return an item from the queue. Blocks if queue is empty.

        Args:
            timeout: Maximum time to wait (None = wait forever)

        Returns:
            Tuple of (success, item) - success is False if timeout occurred
        """
        # TODO: Implement with proper synchronization
        # Block if queue is empty
        # Signal waiting producers that space is available
        pass

    def size(self) -> int:
        """Get current number of items in queue."""
        # TODO: Return size safely
        pass

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        # TODO: Implement thread-safely
        pass

    def is_full(self) -> bool:
        """Check if queue is full."""
        # TODO: Implement thread-safely
        pass


def test_single_producer_consumer():
    """Test single producer and consumer."""
    queue = ThreadSafeQueue(max_size=5)
    produced = []
    consumed = []

    def producer():
        for i in range(20):
            queue.put(i)
            produced.append(i)
            time.sleep(0.01)

    def consumer():
        for _ in range(20):
            success, item = queue.get()
            if success:
                consumed.append(item)
                time.sleep(0.015)

    prod_thread = threading.Thread(target=producer)
    cons_thread = threading.Thread(target=consumer)

    prod_thread.start()
    cons_thread.start()

    prod_thread.join()
    cons_thread.join()

    assert produced == consumed, f"Mismatch: produced {produced}, consumed {consumed}"
    print(f"✓ Single producer-consumer test passed ({len(consumed)} items)")


def test_multiple_producers_consumers():
    """Test multiple producers and consumers."""
    queue = ThreadSafeQueue(max_size=10)
    num_producers = 3
    num_consumers = 2
    items_per_producer = 30
    all_produced = set()
    all_consumed = []
    lock = threading.Lock()

    def producer(producer_id):
        for i in range(items_per_producer):
            item = f"producer_{producer_id}_item_{i}"
            queue.put(item)
            with lock:
                all_produced.add(item)

    def consumer(consumer_id):
        for _ in range(items_per_producer):
            success, item = queue.get()
            if success:
                with lock:
                    all_consumed.append(item)

    # Create and start threads
    producers = [
        threading.Thread(target=producer, args=(i,)) for i in range(num_producers)
    ]
    consumers = [
        threading.Thread(target=consumer, args=(i,)) for i in range(num_consumers)
    ]

    for thread in producers:
        thread.start()

    for thread in consumers:
        thread.start()

    for thread in producers:
        thread.join()

    for thread in consumers:
        thread.join()

    assert len(all_produced) == num_producers * items_per_producer
    assert len(all_consumed) == num_producers * items_per_producer
    assert set(all_consumed) == all_produced

    print(f"✓ Multiple producers-consumers test passed ({len(all_consumed)} items)")


def test_queue_blocking():
    """Test that queue properly blocks when full/empty."""
    queue = ThreadSafeQueue(max_size=3)

    # Fill the queue
    for i in range(3):
        success = queue.put(i, timeout=1.0)
        assert success, "Should be able to put items in empty queue"

    # Try to put with timeout (should timeout since queue is full)
    success = queue.put(99, timeout=0.5)
    assert not success, "Should timeout when queue is full"

    # Get an item to make space
    success, item = queue.get(timeout=1.0)
    assert success and item == 0, "Should be able to get from non-empty queue"

    # Now we should be able to put
    success = queue.put(99, timeout=1.0)
    assert success, "Should be able to put after making space"

    print("✓ Queue blocking test passed")


def test_queue_empty_get():
    """Test that get blocks/times out on empty queue."""
    queue = ThreadSafeQueue(max_size=5)

    success, item = queue.get(timeout=0.5)
    assert not success, "Should timeout on empty queue"

    print("✓ Empty queue get test passed")


if __name__ == "__main__":
    test_single_producer_consumer()
    test_multiple_producers_consumers()
    test_queue_blocking()
    test_queue_empty_get()
    print("\n✓ All tests passed!")
