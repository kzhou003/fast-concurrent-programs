/*
PROBLEM: Thread-Safe Counter

Implement a thread-safe counter that multiple threads can safely increment and decrement
using C++ synchronization primitives.

REQUIREMENTS:
- Must handle concurrent access from multiple threads without race conditions
- Provide increment() and decrement() methods
- Provide get() method to read current value
- Should use std::mutex and/or std::atomic for synchronization
- Should use RAII patterns (lock_guard, unique_lock)

PERFORMANCE NOTES:
- Should be able to handle 1000+ increments from 10+ threads without issues
- Each operation should complete in < 1us on average
- Consider using std::atomic<int> for maximum performance

TEST CASE EXPECTATIONS:
- Starting with counter = 0
- 10 threads each incrementing 10000 times should result in 100000
- 10 threads each decrementing 10000 times should result in -100000
- Mixed increments and decrements should give correct results
*/

#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <cassert>
#include <chrono>

class ThreadSafeCounter {
public:
    ThreadSafeCounter(int initial_value = 0) : value(initial_value) {
        // TODO: Initialize with proper synchronization
    }

    // Increment the counter by 1, return new value
    int increment() {
        // TODO: Implement with thread safety
        return 0;
    }

    // Decrement the counter by 1, return new value
    int decrement() {
        // TODO: Implement with thread safety
        return 0;
    }

    // Get the current value
    int get() const {
        // TODO: Implement with thread safety
        return 0;
    }

    // Add a value to the counter, return new value
    int add(int amount) {
        // TODO: Implement with thread safety
        return 0;
    }

private:
    int value;
    // TODO: Add synchronization primitive (mutex or atomic)
};


void test_basic_operations() {
    ThreadSafeCounter counter(0);

    assert(counter.increment() == 1);
    assert(counter.increment() == 2);
    assert(counter.decrement() == 1);
    assert(counter.get() == 1);
    assert(counter.add(5) == 6);
    assert(counter.get() == 6);

    std::cout << "✓ Basic operations test passed\n";
}


void test_concurrent_increments() {
    ThreadSafeCounter counter(0);
    const int num_threads = 10;
    const int increments_per_thread = 10000;

    auto worker = [&counter]() {
        for (int i = 0; i < 10000; ++i) {
            counter.increment();
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    int expected = num_threads * increments_per_thread;
    int actual = counter.get();

    assert(actual == expected);
    std::cout << "✓ Concurrent increments test passed (" << expected << " increments)\n";
}


void test_concurrent_mixed_operations() {
    ThreadSafeCounter counter(0);
    const int num_threads = 10;
    const int operations_per_thread = 10000;

    auto increment_worker = [&counter]() {
        for (int i = 0; i < 10000; ++i) {
            counter.increment();
        }
    };

    auto decrement_worker = [&counter]() {
        for (int i = 0; i < 10000; ++i) {
            counter.decrement();
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        if (i % 2 == 0) {
            threads.emplace_back(increment_worker);
        } else {
            threads.emplace_back(decrement_worker);
        }
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // 5 threads incrementing, 5 threads decrementing should give 0
    int expected = 0;
    int actual = counter.get();

    assert(actual == expected);
    std::cout << "✓ Concurrent mixed operations test passed (final value: " << actual << ")\n";
}


void benchmark_performance() {
    ThreadSafeCounter counter(0);
    const int num_operations = 1000000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_operations; ++i) {
        counter.increment();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avg_time_us = duration.count() / static_cast<double>(num_operations);
    std::cout << "✓ Performance benchmark: " << avg_time_us << " microseconds per operation\n";

    // Should be < 1 microsecond per operation (unless using mutex, then < 10 microseconds)
    assert(avg_time_us < 100.0);
}


int main() {
    std::cout << "Running ThreadSafeCounter tests...\n\n";

    test_basic_operations();
    test_concurrent_increments();
    test_concurrent_mixed_operations();
    benchmark_performance();

    std::cout << "\n✓ All tests passed!\n";

    return 0;
}
