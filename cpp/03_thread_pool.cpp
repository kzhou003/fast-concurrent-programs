/*
PROBLEM: Thread Pool Implementation

Implement a reusable thread pool that manages a fixed number of worker threads
and distributes tasks to them. This is a fundamental building block for
concurrent applications.

REQUIREMENTS:
- Create a fixed pool of N worker threads
- Implement thread-safe task queue
- Provide submit() method that returns a future for result
- Support exception handling through futures
- Provide shutdown() method for graceful termination
- Use std::future and std::promise for result handling

PERFORMANCE NOTES:
- Should efficiently distribute work among threads
- Threads should not busy-wait (use condition variables)
- Should handle 10000+ tasks efficiently
- Minimal overhead per task submission

TEST CASE EXPECTATIONS:
- Submit 1000 tasks to pool with 4 workers, all should complete
- Tasks should execute in parallel (measure with timing)
- Futures should return results correctly
- Exceptions in tasks should propagate through futures
- Shutdown should wait for all tasks to complete
*/

#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <vector>
#include <cassert>
#include <chrono>
#include <memory>

class ThreadPool {
public:
    ThreadPool(size_t num_workers = 4) : num_workers_(num_workers), shutdown_(false) {
        // TODO: Create and start worker threads
    }

    ~ThreadPool() {
        // TODO: Ensure proper shutdown
    }

    // Submit a task to the pool and return a future for the result
    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using return_type = std::invoke_result_t<F, Args...>;

        // TODO: Create a packaged_task with the function
        // TODO: Get a future from the packaged_task
        // TODO: Wrap the task for the queue
        // TODO: Add to task queue
        // TODO: Notify worker
        // TODO: Return future

        // Placeholder return
        std::promise<return_type> p;
        return p.get_future();
    }

    // Shutdown the thread pool
    // wait: if true, wait for all tasks to complete
    void shutdown(bool wait = true) {
        // TODO: Signal workers to stop
        // TODO: If wait, join all threads
        // TODO: If not wait, detach threads or force exit
    }

    size_t pending_tasks() const {
        // TODO: Return number of queued tasks
        return 0;
    }

private:
    void worker_loop() {
        // TODO: Implement worker thread main loop
        // - Get tasks from queue
        // - Execute them
        // - Handle shutdown signal
        // - Exit when shutdown and queue is empty
    }

    size_t num_workers_;
    bool shutdown_;
    std::queue<std::function<void()>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::vector<std::thread> workers_;
};


int simple_task(int x, int y = 2) {
    // Simulate some work
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return x * y;
}

void test_basic_submit() {
    ThreadPool pool(2);

    std::vector<std::future<int>> futures;

    for (int i = 0; i < 5; ++i) {
        auto future = pool.submit(simple_task, i, 2);
        futures.push_back(std::move(future));
    }

    std::vector<int> results;
    for (auto& future : futures) {
        results.push_back(future.get());
    }

    std::vector<int> expected = {0, 2, 4, 6, 8};
    assert(results == expected);

    pool.shutdown(true);
    std::cout << "✓ Basic submit test passed (" << results.size() << " tasks)\n";
}


void test_parallel_execution() {
    ThreadPool pool(4);

    std::vector<std::future<int>> futures;
    const int num_tasks = 10;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_tasks; ++i) {
        auto future = pool.submit(simple_task, i);
        futures.push_back(std::move(future));
    }

    // Wait for all to complete
    for (auto& future : futures) {
        future.get();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    pool.shutdown(false);

    // With 4 workers and 10 tasks at 10ms each, should take ~30ms (not 100ms)
    // Allow for overhead
    int expected_sequential_ms = num_tasks * 10;
    assert(duration.count() < expected_sequential_ms / 3);

    std::cout << "✓ Parallel execution test passed (10 tasks in " << duration.count() << "ms)\n";
}


void test_exception_handling() {
    ThreadPool pool(2);

    auto failing_task = []() -> int {
        throw std::runtime_error("Task failed");
    };

    auto future = pool.submit(failing_task);

    try {
        future.get();
        assert(false); // Should have thrown
    } catch (const std::runtime_error& e) {
        assert(std::string(e.what()) == "Task failed");
    }

    pool.shutdown(true);
    std::cout << "✓ Exception handling test passed\n";
}


void test_multiple_tasks() {
    ThreadPool pool(4);

    std::vector<std::future<int>> futures;
    const int num_tasks = 100;

    for (int i = 0; i < num_tasks; ++i) {
        auto future = pool.submit(simple_task, i % 10);
        futures.push_back(std::move(future));
    }

    std::vector<int> results;
    for (auto& future : futures) {
        results.push_back(future.get());
    }

    assert(results.size() == num_tasks);

    pool.shutdown(true);
    std::cout << "✓ Multiple tasks test passed (" << results.size() << " tasks)\n";
}


void test_shutdown_wait() {
    ThreadPool pool(2);

    std::vector<std::future<int>> futures;

    auto slow_task = []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return 42;
    };

    for (int i = 0; i < 5; ++i) {
        auto future = pool.submit(slow_task);
        futures.push_back(std::move(future));
    }

    auto start = std::chrono::high_resolution_clock::now();
    pool.shutdown(true);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start
    );

    // Shutdown should have waited
    assert(elapsed.count() >= 100);

    // All futures should be ready
    for (auto& future : futures) {
        assert(future.get() == 42);
    }

    std::cout << "✓ Shutdown wait test passed (waited " << elapsed.count() << "ms)\n";
}


int main() {
    std::cout << "Running ThreadPool tests...\n\n";

    test_basic_submit();
    test_parallel_execution();
    test_exception_handling();
    test_multiple_tasks();
    test_shutdown_wait();

    std::cout << "\n✓ All tests passed!\n";

    return 0;
}
