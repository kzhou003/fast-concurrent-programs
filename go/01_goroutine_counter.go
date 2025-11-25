/*
PROBLEM: Goroutine Counter with Mutex

Implement a thread-safe counter that multiple goroutines can safely increment and decrement
using Go's sync.Mutex primitive.

REQUIREMENTS:
- Use sync.Mutex for synchronization
- Provide Increment() and Decrement() methods
- Provide Get() method for reading value
- Handle concurrent access from multiple goroutines safely

PERFORMANCE NOTES:
- Should handle 1000+ increments from 100+ goroutines
- Each operation should be very fast (< 1 microsecond)
- Consider using sync.atomic for better performance

TEST CASE EXPECTATIONS:
- 100 goroutines each incrementing 1000 times should result in 100000
- 100 goroutines with mixed operations should maintain correctness
- Benchmark should show good performance characteristics
*/

package main

import (
	"fmt"
	"sync"
	"time"
)

// Counter is a thread-safe counter
type Counter struct {
	// TODO: Add mutex field
	// TODO: Add value field
}

// NewCounter creates a new counter with initial value
func NewCounter(initial int) *Counter {
	// TODO: Initialize counter with mutex and value
	return &Counter{}
}

// Increment increases the counter by 1 and returns the new value
func (c *Counter) Increment() int {
	// TODO: Implement with mutex protection
	return 0
}

// Decrement decreases the counter by 1 and returns the new value
func (c *Counter) Decrement() int {
	// TODO: Implement with mutex protection
	return 0
}

// Get returns the current value
func (c *Counter) Get() int {
	// TODO: Implement with mutex protection
	return 0
}

// Add increases the counter by the given amount and returns the new value
func (c *Counter) Add(amount int) int {
	// TODO: Implement with mutex protection
	return 0
}

func TestBasicOperations() {
	counter := NewCounter(0)

	if counter.Increment() != 1 {
		panic("Increment failed")
	}
	if counter.Increment() != 2 {
		panic("Increment failed")
	}
	if counter.Decrement() != 1 {
		panic("Decrement failed")
	}
	if counter.Get() != 1 {
		panic("Get failed")
	}
	if counter.Add(5) != 6 {
		panic("Add failed")
	}

	fmt.Println("✓ Basic operations test passed")
}

func TestConcurrentIncrements() {
	counter := NewCounter(0)
	numGoroutines := 100
	incrementsPerGoroutine := 1000

	var wg sync.WaitGroup
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < incrementsPerGoroutine; j++ {
				counter.Increment()
			}
		}()
	}

	wg.Wait()

	expected := numGoroutines * incrementsPerGoroutine
	actual := counter.Get()

	if actual != expected {
		panic(fmt.Sprintf("Expected %d, got %d", expected, actual))
	}

	fmt.Printf("✓ Concurrent increments test passed (%d increments)\n", expected)
}

func TestConcurrentMixedOperations() {
	counter := NewCounter(0)
	numGoroutines := 100
	operationsPerGoroutine := 1000

	var wg sync.WaitGroup
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < operationsPerGoroutine; j++ {
				if id%2 == 0 {
					counter.Increment()
				} else {
					counter.Decrement()
				}
			}
		}(i)
	}

	wg.Wait()

	// 50 goroutines incrementing, 50 decrementing should result in 0
	expected := 0
	actual := counter.Get()

	if actual != expected {
		panic(fmt.Sprintf("Expected %d, got %d", expected, actual))
	}

	fmt.Printf("✓ Concurrent mixed operations test passed (final value: %d)\n", actual)
}

func BenchmarkCounter() {
	counter := NewCounter(0)
	numOperations := 1000000

	start := time.Now()
	for i := 0; i < numOperations; i++ {
		counter.Increment()
	}
	elapsed := time.Since(start)

	avgTime := float64(elapsed.Microseconds()) / float64(numOperations)
	fmt.Printf("✓ Performance benchmark: %.3f microseconds per operation\n", avgTime)

	if avgTime > 1.0 {
		fmt.Printf("  Warning: Performance is slower than expected (> 1µs per op)\n")
	}
}

func main() {
	fmt.Println("Running Counter tests...\n")

	TestBasicOperations()
	TestConcurrentIncrements()
	TestConcurrentMixedOperations()
	BenchmarkCounter()

	fmt.Println("\n✓ All tests passed!")
}
