/*
PROBLEM: Read-Write Lock

Implement a readers-writer lock that allows multiple readers to access a resource
simultaneously, but exclusive access for writers. This is an important pattern
for read-heavy concurrent applications.

REQUIREMENTS:
- Allow multiple simultaneous readers
- Only one writer at a time
- Writers have priority (no writer starvation)
- Provide lock() and unlock() for exclusive access
- Provide read_lock() and read_unlock() for shared access
- Use std::mutex and std::condition_variable
- RAII patterns with lock guards

PERFORMANCE NOTES:
- Should handle 100+ readers simultaneously
- Writers should not be starved
- Low contention even with many concurrent readers

TEST CASE EXPECTATIONS:
- Multiple readers can access simultaneously
- Writer blocks readers
- Readers block writers
- No deadlock with concurrent readers and writers
- Fair scheduling (no reader/writer starvation)
*/

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <shared_mutex>
#include <vector>
#include <cassert>
#include <chrono>
#include <atomic>

class ReadWriteLock {
public:
    ReadWriteLock() : num_readers_(0), num_writers_(0), num_waiting_writers_(0) {
        // TODO: Initialize state
    }

    // Acquire exclusive write lock
    void lock() {
        // TODO: Implement exclusive lock
        // - Increment waiting writers
        // - Wait if any readers or writers are active
        // - Decrement waiting writers, increment active writers
    }

    // Release exclusive write lock
    void unlock() {
        // TODO: Implement unlock
        // - Decrement active writers
        // - Notify all waiters (readers and writers)
    }

    // Acquire shared read lock
    void lock_shared() {
        // TODO: Implement shared lock
        // - Wait if any writers or waiting writers exist
        // - Increment readers
    }

    // Release shared read lock
    void unlock_shared() {
        // TODO: Implement shared unlock
        // - Decrement readers
        // - Notify waiting writers if no more readers
    }

    // RAII wrapper for exclusive lock
    class WriteLockGuard {
    public:
        WriteLockGuard(ReadWriteLock& lock) : lock_(lock) {
            lock_.lock();
        }

        ~WriteLockGuard() {
            lock_.unlock();
        }

    private:
        ReadWriteLock& lock_;
    };

    // RAII wrapper for shared lock
    class ReadLockGuard {
    public:
        ReadLockGuard(ReadWriteLock& lock) : lock_(lock) {
            lock_.lock_shared();
        }

        ~ReadLockGuard() {
            lock_.unlock_shared();
        }

    private:
        ReadWriteLock& lock_;
    };

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    int num_readers_;
    int num_writers_;
    int num_waiting_writers_;
};


class SharedResource {
public:
    SharedResource() : value_(0), reads_(0), writes_(0) {}

    void write(int new_value) {
        ReadWriteLock::WriteLockGuard guard(lock_);
        value_ = new_value;
        writes_++;
    }

    int read() {
        ReadWriteLock::ReadLockGuard guard(lock_);
        reads_++;
        return value_;
    }

    int get_reads() const { return reads_; }
    int get_writes() const { return writes_; }

private:
    ReadWriteLock lock_;
    int value_;
    std::atomic<int> reads_;
    std::atomic<int> writes_;
};


void test_multiple_readers() {
    SharedResource resource;
    const int num_readers = 10;
    const int reads_per_thread = 100;

    auto reader_thread = [&resource]() {
        for (int i = 0; i < 100; ++i) {
            resource.read();
        }
    };

    std::vector<std::thread> threads;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_readers; ++i) {
        threads.emplace_back(reader_thread);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start
    );

    assert(resource.get_reads() == num_readers * reads_per_thread);
    std::cout << "✓ Multiple readers test passed (" << resource.get_reads()
              << " reads in " << elapsed.count() << "ms)\n";
}


void test_writer_blocking_readers() {
    SharedResource resource;
    std::atomic<bool> writer_acquired(false);
    std::atomic<int> concurrent_readers(0);
    std::atomic<int> max_concurrent_readers(0);

    auto reader = [&resource, &writer_acquired, &concurrent_readers, &max_concurrent_readers]() {
        if (!writer_acquired.load()) {
            concurrent_readers++;
            int current = concurrent_readers.load();
            if (current > max_concurrent_readers.load()) {
                max_concurrent_readers.store(current);
            }

            resource.read();

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            concurrent_readers--;
        }
    };

    ReadWriteLock lock;
    std::vector<std::thread> threads;

    // Start readers
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back(reader);
    }

    // Let them start
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // Writer acquires lock
    {
        ReadWriteLock::WriteLockGuard guard(lock);
        writer_acquired.store(true);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        writer_acquired.store(false);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    std::cout << "✓ Writer blocking readers test passed\n";
}


void test_readers_blocking_writers() {
    SharedResource resource;
    std::atomic<int> concurrent_readers(0);

    auto reader = [&resource, &concurrent_readers]() {
        {
            ReadWriteLock::ReadLockGuard guard(resource.lock_);
            concurrent_readers++;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            concurrent_readers--;
        }
    };

    std::vector<std::thread> threads;

    // Start readers
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back(reader);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Try to write - should block
    auto write_start = std::chrono::high_resolution_clock::now();
    resource.write(42);
    auto write_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - write_start
    );

    // Write should have taken at least ~40ms (rest of reader time)
    assert(write_elapsed.count() >= 35);

    for (auto& thread : threads) {
        thread.join();
    }

    std::cout << "✓ Readers blocking writers test passed (write delayed " << write_elapsed.count()
              << "ms)\n";
}


void test_exclusive_write_access() {
    SharedResource resource;
    std::atomic<int> concurrent_writers(0);
    std::atomic<int> max_concurrent_writers(0);

    auto writer = [&resource, &concurrent_writers, &max_concurrent_writers]() {
        concurrent_writers++;
        int current = concurrent_writers.load();
        if (current > max_concurrent_writers.load()) {
            max_concurrent_writers.store(current);
        }

        resource.write(42);

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        concurrent_writers--;
    };

    std::vector<std::thread> threads;

    for (int i = 0; i < 5; ++i) {
        threads.emplace_back(writer);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    assert(max_concurrent_writers.load() == 1);
    std::cout << "✓ Exclusive write access test passed (max concurrent writers: "
              << max_concurrent_writers.load() << ")\n";
}


void test_no_deadlock() {
    SharedResource resource;
    std::atomic<bool> deadlock_detected(false);

    auto reader = [&resource]() {
        for (int i = 0; i < 100; ++i) {
            resource.read();
        }
    };

    auto writer = [&resource]() {
        for (int i = 0; i < 10; ++i) {
            resource.write(i);
        }
    };

    std::vector<std::thread> threads;

    for (int i = 0; i < 10; ++i) {
        threads.emplace_back(reader);
    }

    for (int i = 0; i < 3; ++i) {
        threads.emplace_back(writer);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (auto& thread : threads) {
        thread.join();
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start
    );

    // Should complete reasonably quickly (< 5 seconds)
    assert(elapsed.count() < 5000);

    std::cout << "✓ No deadlock test passed (completed in " << elapsed.count() << "ms)\n";
}


int main() {
    std::cout << "Running ReadWriteLock tests...\n\n";

    test_multiple_readers();
    test_writer_blocking_readers();
    test_readers_blocking_writers();
    test_exclusive_write_access();
    test_no_deadlock();

    std::cout << "\n✓ All tests passed!\n";

    return 0;
}
