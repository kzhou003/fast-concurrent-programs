"""
PROBLEM: Thread Pool Executor

Implement a simple thread pool that manages a fixed number of worker threads and
distributes tasks from a queue to these workers. Workers should process tasks
and return results.

REQUIREMENTS:
- Create a fixed pool of N worker threads
- Implement a task queue for distributing work
- Worker threads should process tasks from queue
- Provide a submit() method to add tasks
- Provide a shutdown() method for graceful termination
- Track task results/completion

PERFORMANCE NOTES:
- Should efficiently distribute work among threads
- Threads should not busy-wait
- Shutdown should wait for all tasks to complete
- Should handle 1000+ tasks efficiently

TEST CASE EXPECTATIONS:
- Submit 100 tasks to pool with 4 workers, all should complete
- Tasks should execute in parallel (measure with timing)
- Results should be collectable in any order
- Shutdown should block until all tasks complete
"""

import threading
import time
from typing import Any, Callable, Optional
from queue import Queue, Empty


class Task:
    """Represents a task to be executed."""

    def __init__(self, task_id: int, func: Callable, args: tuple = (), kwargs: dict = None):
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.result = None
        self.exception = None
        self.completed = False
        self.event = threading.Event()

    def execute(self):
        """Execute the task and store result."""
        try:
            self.result = self.func(*self.args, **self.kwargs)
        except Exception as e:
            self.exception = e
        finally:
            self.completed = True
            self.event.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for task to complete."""
        return self.event.wait(timeout)

    def get_result(self, timeout: Optional[float] = None) -> Any:
        """Get the result of the task."""
        if not self.wait(timeout):
            raise TimeoutError(f"Task {self.task_id} did not complete in time")
        if self.exception:
            raise self.exception
        return self.result


class ThreadPool:
    """A simple thread pool executor."""

    def __init__(self, num_workers: int = 4):
        """
        Initialize the thread pool.

        Args:
            num_workers: Number of worker threads to create
        """
        self.num_workers = num_workers
        self.task_queue = Queue()
        # TODO: Initialize workers and other necessary state
        self.workers = []
        self.running = True
        self.task_counter = 0
        self.counter_lock = threading.Lock()

    def _worker(self):
        """Worker thread main loop."""
        # TODO: Implement worker thread logic
        # - Get tasks from queue
        # - Execute them
        # - Handle shutdown signal
        pass

    def start(self):
        """Start all worker threads."""
        # TODO: Create and start worker threads
        pass

    def submit(self, func: Callable, *args, **kwargs) -> Task:
        """
        Submit a task to the pool.

        Args:
            func: Callable to execute
            args: Positional arguments for the callable
            kwargs: Keyword arguments for the callable

        Returns:
            Task object that can be used to get results
        """
        # TODO: Create task, add to queue, return task object
        pass

    def shutdown(self, wait: bool = True):
        """
        Shut down the thread pool.

        Args:
            wait: If True, wait for all tasks to complete before returning
        """
        # TODO: Signal workers to stop and optionally wait for completion
        pass


def simple_task(x: int, y: int = 2) -> int:
    """A simple task that returns x^y."""
    time.sleep(0.01)  # Simulate work
    return x ** y


def failing_task():
    """A task that raises an exception."""
    raise ValueError("This task always fails")


def test_basic_submit():
    """Test submitting basic tasks."""
    pool = ThreadPool(num_workers=2)
    pool.start()

    tasks = []
    for i in range(5):
        task = pool.submit(simple_task, i, y=2)
        tasks.append(task)

    results = [task.get_result() for task in tasks]
    expected = [i ** 2 for i in range(5)]

    assert results == expected, f"Expected {expected}, got {results}"
    pool.shutdown(wait=True)

    print(f"✓ Basic submit test passed ({len(results)} tasks)")


def test_parallel_execution():
    """Test that tasks execute in parallel."""
    pool = ThreadPool(num_workers=4)
    pool.start()

    num_tasks = 10
    start_time = time.time()

    tasks = []
    for i in range(num_tasks):
        task = pool.submit(simple_task, i)
        tasks.append(task)

    # Wait for all to complete
    for task in tasks:
        task.get_result()

    elapsed = time.time() - start_time
    pool.shutdown(wait=False)

    # With 4 workers and 10 tasks at 0.01s each, should take ~0.03s (not 0.1s)
    # Allow some overhead
    expected_sequential = num_tasks * 0.01
    assert elapsed < expected_sequential * 0.5, (
        f"Tasks don't seem to run in parallel. "
        f"Took {elapsed:.3f}s, expected ~{expected_sequential * 0.25:.3f}s"
    )

    print(f"✓ Parallel execution test passed (10 tasks in {elapsed:.3f}s)")


def test_exception_handling():
    """Test handling of exceptions in tasks."""
    pool = ThreadPool(num_workers=2)
    pool.start()

    task = pool.submit(failing_task)

    try:
        result = task.get_result()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "This task always fails"

    pool.shutdown(wait=True)
    print("✓ Exception handling test passed")


def test_multiple_tasks():
    """Test submitting many tasks."""
    pool = ThreadPool(num_workers=4)
    pool.start()

    num_tasks = 100
    tasks = []

    for i in range(num_tasks):
        task = pool.submit(simple_task, i % 10)
        tasks.append(task)

    results = [task.get_result() for task in tasks]

    assert len(results) == num_tasks
    assert all(isinstance(r, int) for r in results)

    pool.shutdown(wait=True)
    print(f"✓ Multiple tasks test passed ({num_tasks} tasks)")


def test_shutdown_wait():
    """Test that shutdown waits for tasks."""
    pool = ThreadPool(num_workers=2)
    pool.start()

    slow_task_completed = []

    def slow_task(duration):
        time.sleep(duration)
        slow_task_completed.append(True)

    tasks = [pool.submit(slow_task, 0.1) for _ in range(5)]

    start = time.time()
    pool.shutdown(wait=True)
    elapsed = time.time() - start

    assert len(slow_task_completed) == 5, "Not all tasks completed"
    assert elapsed >= 0.1, "Shutdown returned too quickly"

    print(f"✓ Shutdown wait test passed (waited {elapsed:.3f}s)")


if __name__ == "__main__":
    test_basic_submit()
    test_parallel_execution()
    test_exception_handling()
    test_multiple_tasks()
    test_shutdown_wait()
    print("\n✓ All tests passed!")
