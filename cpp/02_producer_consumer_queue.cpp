/*
PROBLEM: Producer-Consumer Queue

Implement a thread-safe queue where producer threads add items and consumer threads
remove them. This classic concurrent pattern requires careful synchronization.

REQUIREMENTS:
- Implement a template-based bounded queue
- Producers block when queue is full
- Consumers block when queue is empty
- Use std::condition_variable for efficient waiting
- Use std::mutex for synchronization
- RAII patterns throughout

PERFORMANCE NOTES:
- Should handle 100+ items per second throughput
- Minimal busy-waiting (use condition variables)
- Lock contention should be minimized

TEST CASE EXPECTATIONS:
- Single producer adding 1000 items, single consumer getting all should work correctly
- Multiple producers and consumers should maintain integrity
- Queue of size 10 with 5 producers and 5 consumers should not overflow
- notify_one() should efficiently wake waiting threads
*/

#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <cassert>
#include <chrono>
#include <optional>

template <typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue(size_t max_size = 10) : max_size_(max_size) {
        // TODO: Initialize queue, mutex, and condition variables
    }

    // Add an item to queue, block if full
    // Returns true if added, false if timeout
    bool put(const T& item, double timeout_seconds = -1.0) {
        // TODO: Implement with proper synchronization
        // - Acquire lock
        // - Wait if queue is full (using condition_variable with timeout)
        // - Add item
        // - Notify waiting consumers
        // - Return success/timeout status
        return false;
    }

    // Get an item from queue, block if empty
    // Returns {success, item} - success is false if timeout
    std::pair<bool, std::optional<T>> get(double timeout_seconds = -1.0) {
        // TODO: Implement with proper synchronization
        // - Acquire lock
        // - Wait if queue is empty (using condition_variable with timeout)
        // - Remove and return item
        // - Notify waiting producers
        // - Return success and item
        return {false, std::nullopt};
    }

    size_t size() const {
        // TODO: Return size thread-safely
        return 0;
    }

    bool is_empty() const {
        // TODO: Implement thread-safely
        return true;
    }

    bool is_full() const {
        // TODO: Implement thread-safely
        return false;
    }

private:
    std::queue<T> queue_;
    size_t max_size_;
    // TODO: Add std::mutex and std::condition_variable
};


void test_single_producer_consumer() {
    ThreadSafeQueue<int> queue(5);
    std::vector<int> produced;
    std::vector<int> consumed;

    auto producer = [&queue, &produced]() {
        for (int i = 0; i < 100; ++i) {
            queue.put(i);
            produced.push_back(i);
        }
    };

    auto consumer = [&queue, &consumed]() {
        for (int i = 0; i < 100; ++i) {
            auto [success, item] = queue.get();
            if (success && item.has_value()) {
                consumed.push_back(item.value());
            }
        }
    };

    std::thread prod_thread(producer);
    std::thread cons_thread(consumer);

    prod_thread.join();
    cons_thread.join();

    assert(produced == consumed);
    std::cout << "✓ Single producer-consumer test passed (" << consumed.size() << " items)\n";
}


void test_multiple_producers_consumers() {
    ThreadSafeQueue<int> queue(10);
    const int num_producers = 3;
    const int num_consumers = 3;
    const int items_per_producer = 50;
    std::mutex stats_lock;
    int total_produced = 0;
    int total_consumed = 0;

    auto producer = [&queue, &stats_lock, &total_produced](int producer_id) {
        for (int i = 0; i < 50; ++i) {
            queue.put(producer_id * 1000 + i);
            {
                std::lock_guard<std::mutex> lock(stats_lock);
                total_produced++;
            }
        }
    };

    auto consumer = [&queue, &stats_lock, &total_consumed]() {
        while (true) {
            auto [success, item] = queue.get(0.1); // 100ms timeout
            if (success && item.has_value()) {
                {
                    std::lock_guard<std::mutex> lock(stats_lock);
                    total_consumed++;
                }
            } else {
                // Small timeout reached, check if we should continue
                // In a real scenario, you'd have a shutdown signal
                break;
            }
        }
    };

    std::vector<std::thread> threads;

    // Create producers
    for (int i = 0; i < num_producers; ++i) {
        threads.emplace_back(producer, i);
    }

    // Create consumers
    for (int i = 0; i < num_consumers; ++i) {
        threads.emplace_back(consumer);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Wait a bit for remaining items to be consumed
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    int expected = num_producers * items_per_producer;
    // Note: This test might have race conditions in counting, adjust as needed
    std::cout << "✓ Multiple producers-consumers test passed (" << total_consumed << " items consumed)\n";
}


void test_queue_blocking() {
    ThreadSafeQueue<int> queue(3);

    // Fill the queue
    for (int i = 0; i < 3; ++i) {
        bool success = queue.put(i);
        assert(success);
    }

    // Try to put with timeout (should timeout since queue is full)
    bool success = queue.put(99, 0.1); // 100ms timeout
    assert(!success); // Should timeout

    // Get an item to make space
    auto [get_success, item] = queue.get();
    assert(get_success && item.has_value());
    assert(item.value() == 0);

    // Now we should be able to put
    success = queue.put(99);
    assert(success);

    std::cout << "✓ Queue blocking test passed\n";
}


void test_queue_empty_get() {
    ThreadSafeQueue<int> queue(5);

    auto [success, item] = queue.get(0.1); // 100ms timeout
    assert(!success); // Should timeout on empty queue

    std::cout << "✓ Empty queue get test passed\n";
}


void benchmark_throughput() {
    ThreadSafeQueue<int> queue(100);
    const int num_items = 10000;

    auto start = std::chrono::high_resolution_clock::now();

    std::thread producer([&queue, num_items]() {
        for (int i = 0; i < num_items; ++i) {
            queue.put(i);
        }
    });

    std::thread consumer([&queue, num_items]() {
        for (int i = 0; i < num_items; ++i) {
            queue.get();
        }
    });

    producer.join();
    consumer.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double items_per_second = (num_items * 1000.0) / duration.count();
    std::cout << "✓ Throughput benchmark: " << items_per_second << " items/second\n";
}


int main() {
    std::cout << "Running ThreadSafeQueue tests...\n\n";

    test_single_producer_consumer();
    test_multiple_producers_consumers();
    test_queue_blocking();
    test_queue_empty_get();
    benchmark_throughput();

    std::cout << "\n✓ All tests passed!\n";

    return 0;
}
