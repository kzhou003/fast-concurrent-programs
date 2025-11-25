/*
PROBLEM: Lock-Free Queue

Implement a lock-free queue using atomic operations and CAS (Compare-And-Swap).
Lock-free data structures provide better performance and scalability than
mutex-based implementations, but are more complex to implement correctly.

REQUIREMENTS:
- Use std::atomic operations for synchronization
- Implement compare_exchange_strong/weak for CAS operations
- Memory ordering considerations (acquire, release, relaxed)
- Handle ABA problem (optional: use tagged pointers)
- Single-consumer or multi-consumer variant

PERFORMANCE NOTES:
- Should be significantly faster than mutex-based queue for high contention
- Should scale linearly with number of threads
- Memory ordering should be optimized for performance

TEST CASE EXPECTATIONS:
- Producer-consumer test should complete correctly
- Multiple producers and consumers should work without locks
- Performance should exceed mutex-based queue by 2-5x under high contention
- No memory leaks or undefined behavior
*/

#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <cassert>
#include <memory>
#include <chrono>
#include <optional>

template <typename T>
class LockFreeQueue {
private:
    struct Node {
        T value;
        std::atomic<Node*> next{nullptr};

        Node(const T& val) : value(val) {}
    };

public:
    LockFreeQueue() {
        // Create sentinel node
        Node* sentinel = new Node(T());
        head_.store(sentinel);
        tail_.store(sentinel);
    }

    ~LockFreeQueue() {
        // TODO: Clean up remaining nodes
    }

    // Enqueue an item
    void enqueue(const T& value) {
        // TODO: Implement lock-free enqueue
        // 1. Create new node
        // 2. Use CAS to append to tail
        // 3. Update tail to point to new node
        // Considerations:
        // - Handle concurrent enqueues
        // - Update tail pointer after node insertion
        // - Use appropriate memory ordering
    }

    // Try to dequeue an item
    std::optional<T> dequeue() {
        // TODO: Implement lock-free dequeue
        // 1. Check if queue is empty
        // 2. Load first node after sentinel
        // 3. Try to move head forward
        // 4. Extract and return value
        // Considerations:
        // - Handle empty queue
        // - Use CAS for head update
        // - Proper memory ordering
        // - Consider ABA problem
        return std::nullopt;
    }

    // Check if queue is empty (note: may be racy)
    bool is_empty() const {
        return head_.load() == tail_.load();
    }

private:
    std::atomic<Node*> head_;
    std::atomic<Node*> tail_;
};


void test_basic_operations() {
    LockFreeQueue<int> queue;

    queue.enqueue(1);
    queue.enqueue(2);
    queue.enqueue(3);

    assert(queue.dequeue() == 1);
    assert(queue.dequeue() == 2);
    assert(queue.dequeue() == 3);
    assert(!queue.dequeue()); // Empty

    std::cout << "✓ Basic operations test passed\n";
}


void test_single_producer_consumer() {
    LockFreeQueue<int> queue;
    const int num_items = 1000;

    auto producer = [&queue]() {
        for (int i = 0; i < 1000; ++i) {
            queue.enqueue(i);
        }
    };

    auto consumer = [&queue]() {
        for (int i = 0; i < 1000; ++i) {
            while (!queue.dequeue()) {
                // Spin until item is available
            }
        }
    };

    std::thread prod(producer);
    std::thread cons(consumer);

    prod.join();
    cons.join();

    assert(!queue.dequeue());
    std::cout << "✓ Single producer-consumer test passed (" << num_items << " items)\n";
}


void test_multiple_producers_consumers() {
    LockFreeQueue<int> queue;
    const int num_producers = 4;
    const int num_consumers = 4;
    const int items_per_producer = 250;
    std::atomic<int> consumed_count(0);

    auto producer = [&queue](int producer_id) {
        for (int i = 0; i < 250; ++i) {
            queue.enqueue(producer_id * 1000 + i);
        }
    };

    auto consumer = [&queue, &consumed_count]() {
        int local_count = 0;
        while (local_count < 250) { // Expect to consume items
            auto item = queue.dequeue();
            if (item) {
                local_count++;
                consumed_count++;
            } else {
                // Spin or yield
                std::this_thread::yield();
            }
        }
    };

    std::vector<std::thread> threads;

    auto start = std::chrono::high_resolution_clock::now();

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

    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start
    );

    int expected = num_producers * items_per_producer;
    std::cout << "✓ Multiple producers-consumers test passed (" << consumed_count.load()
              << " items in " << elapsed.count() << " microseconds)\n";
}


void test_concurrent_enqueues() {
    LockFreeQueue<int> queue;
    const int num_threads = 10;
    const int enqueues_per_thread = 100;

    auto worker = [&queue]() {
        for (int i = 0; i < 100; ++i) {
            queue.enqueue(i);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // All items should be dequeue-able
    int count = 0;
    while (auto item = queue.dequeue()) {
        count++;
    }

    int expected = num_threads * enqueues_per_thread;
    assert(count == expected);
    std::cout << "✓ Concurrent enqueues test passed (" << count << " items)\n";
}


void benchmark_throughput() {
    const int num_items = 100000;

    // Lock-free queue
    LockFreeQueue<int> queue;

    auto start = std::chrono::high_resolution_clock::now();

    std::thread producer([&queue, num_items]() {
        for (int i = 0; i < num_items; ++i) {
            queue.enqueue(i);
        }
    });

    std::thread consumer([&queue, num_items]() {
        int consumed = 0;
        while (consumed < num_items) {
            if (auto item = queue.dequeue()) {
                consumed++;
            }
        }
    });

    producer.join();
    consumer.join();

    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start
    );

    double throughput = (num_items * 1000000.0) / elapsed.count();
    std::cout << "✓ Throughput benchmark: " << throughput << " ops/second\n";
}


int main() {
    std::cout << "Running LockFreeQueue tests...\n\n";

    test_basic_operations();
    test_single_producer_consumer();
    test_multiple_producers_consumers();
    test_concurrent_enqueues();
    benchmark_throughput();

    std::cout << "\n✓ All tests passed!\n";

    return 0;
}
