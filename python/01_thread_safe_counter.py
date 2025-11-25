"""
PROBLEM: Thread-Safe Counter

Implement a thread-safe counter that multiple threads can safely increment and decrement.

REQUIREMENTS:
- Must handle concurrent access from multiple threads without race conditions
- Provide increment() and decrement() methods
- Provide get() method to read current value
- Should use proper synchronization primitives (Lock, RLock, etc.)

PERFORMANCE NOTES:
- Should be able to handle 1000+ increments from 10+ threads without issues
- Each operation should complete in < 1ms on average

TEST CASE EXPECTATIONS:
- Starting with counter = 0
- 10 threads each incrementing 100 times should result in 1000
- 10 threads each decrementing 100 times should result in -1000
- Mixed increments and decrements should give correct results
"""

import threading
from typing import Optional


class ThreadSafeCounter:
    """A counter that can be safely accessed by multiple threads."""

    def __init__(self, initial_value: int = 0):
        """
        Initialize the counter.

        Args:
            initial_value: Starting value for the counter
        """
        # TODO: Implement initialization with proper synchronization
        self.value = initial_value

    def increment(self) -> int:
        """
        Increment the counter by 1.

        Returns:
            The new value after incrementing
        """
        # TODO: Implement with thread safety
        pass

    def decrement(self) -> int:
        """
        Decrement the counter by 1.

        Returns:
            The new value after decrementing
        """
        # TODO: Implement with thread safety
        pass

    def get(self) -> int:
        """
        Get the current value of the counter.

        Returns:
            Current counter value
        """
        # TODO: Implement with thread safety
        pass

    def add(self, value: int) -> int:
        """
        Add a value to the counter.

        Args:
            value: Amount to add (can be negative)

        Returns:
            The new value after adding
        """
        # TODO: Implement with thread safety
        pass


def test_basic_operations():
    """Test basic counter operations."""
    counter = ThreadSafeCounter(0)

    assert counter.increment() == 1
    assert counter.increment() == 2
    assert counter.decrement() == 1
    assert counter.get() == 1
    assert counter.add(5) == 6
    assert counter.get() == 6

    print("✓ Basic operations test passed")


def test_concurrent_increments():
    """Test counter with concurrent increments."""
    counter = ThreadSafeCounter(0)
    num_threads = 10
    increments_per_thread = 100

    def worker():
        for _ in range(increments_per_thread):
            counter.increment()

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    expected = num_threads * increments_per_thread
    actual = counter.get()

    assert actual == expected, f"Expected {expected}, got {actual}"
    print(f"✓ Concurrent increments test passed ({expected} increments)")


def test_concurrent_mixed_operations():
    """Test counter with mixed increments and decrements."""
    counter = ThreadSafeCounter(0)
    num_threads = 10
    operations_per_thread = 100

    def worker_increment():
        for _ in range(operations_per_thread):
            counter.increment()

    def worker_decrement():
        for _ in range(operations_per_thread):
            counter.decrement()

    threads = []
    for i in range(num_threads):
        if i % 2 == 0:
            threads.append(threading.Thread(target=worker_increment))
        else:
            threads.append(threading.Thread(target=worker_decrement))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # 5 threads incrementing, 5 threads decrementing should give 0
    expected = 0
    actual = counter.get()

    assert actual == expected, f"Expected {expected}, got {actual}"
    print(f"✓ Concurrent mixed operations test passed (final value: {actual})")


if __name__ == "__main__":
    test_basic_operations()
    test_concurrent_increments()
    test_concurrent_mixed_operations()
    print("\n✓ All tests passed!")
