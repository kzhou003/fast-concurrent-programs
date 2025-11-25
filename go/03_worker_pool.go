/*
PROBLEM: Worker Pool Pattern

Implement a worker pool that distributes tasks to a fixed number of worker goroutines.
This is an essential pattern for managing concurrent work in Go.

REQUIREMENTS:
- Create a fixed number of worker goroutines
- Distribute tasks from a work channel to workers
- Support result collection via result channel
- Graceful shutdown when no more work
- Use idiomatic Go patterns (channels, select)

PERFORMANCE NOTES:
- Should efficiently handle 10000+ tasks
- Workers should be fully utilized
- Minimal idle time

TEST CASE EXPECTATIONS:
- 1000 tasks distributed to 10 workers should complete correctly
- Results should be collectible in any order
- All tasks should be processed exactly once
- No goroutine leaks
*/

package main

import (
	"fmt"
	"sync"
	"time"
)

// Task represents a unit of work
type Task struct {
	ID    int
	Value int
}

// Result represents the result of a task
type Result struct {
	TaskID int
	Value  int
	Error  error
}

// WorkerPool manages a pool of worker goroutines
type WorkerPool struct {
	workers    int
	workChan   chan Task
	resultChan chan Result
	wg         sync.WaitGroup
}

// NewWorkerPool creates a new worker pool
func NewWorkerPool(numWorkers int) *WorkerPool {
	// TODO: Initialize WorkerPool
	// TODO: Create work and result channels
	// TODO: Start worker goroutines
	// TODO: Return pool
	return &WorkerPool{}
}

// Submit adds a task to the work queue
func (wp *WorkerPool) Submit(task Task) error {
	// TODO: Send task to work channel
	// Return error if pool is closed
	return nil
}

// Results returns the result channel
func (wp *WorkerPool) Results() <-chan Result {
	// TODO: Return read-only channel to results
	return wp.resultChan
}

// Shutdown gracefully shuts down the pool
func (wp *WorkerPool) Shutdown() {
	// TODO: Close work channel
	// TODO: Wait for all workers to finish
}

// worker is the main loop for a worker goroutine
// TODO: Implement worker function that:
//   - Reads tasks from work channel
//   - Processes them (apply some function)
//   - Sends results to result channel
//   - Exits when work channel is closed

func processTask(task Task) (int, error) {
	// Simulate work
	time.Sleep(time.Millisecond)
	// Simple processing: square the value
	return task.Value * task.Value, nil
}

func TestBasicWorkerPool() {
	pool := NewWorkerPool(4)
	numTasks := 20

	// Submit tasks
	go func() {
		for i := 0; i < numTasks; i++ {
			pool.Submit(Task{ID: i, Value: i})
		}
		pool.Shutdown()
	}()

	// Collect results
	results := make([]Result, 0, numTasks)
	for result := range pool.Results() {
		results = append(results, result)
	}

	if len(results) != numTasks {
		panic(fmt.Sprintf("Expected %d results, got %d", numTasks, len(results)))
	}

	// Verify results
	for _, result := range results {
		expected := result.TaskID * result.TaskID
		if result.Value != expected {
			panic(fmt.Sprintf("Task %d: expected %d, got %d",
				result.TaskID, expected, result.Value))
		}
	}

	fmt.Printf("✓ Basic worker pool test passed (%d tasks)\n", numTasks)
}

func TestMultipleWorkers() {
	pool := NewWorkerPool(8)
	numTasks := 100

	// Submit tasks
	go func() {
		for i := 0; i < numTasks; i++ {
			pool.Submit(Task{ID: i, Value: i})
		}
		pool.Shutdown()
	}()

	// Collect results with timing
	start := time.Now()
	count := 0
	for range pool.Results() {
		count++
	}
	elapsed := time.Since(start)

	if count != numTasks {
		panic(fmt.Sprintf("Expected %d results, got %d", numTasks, count))
	}

	fmt.Printf("✓ Multiple workers test passed (%d tasks in %dms)\n",
		numTasks, elapsed.Milliseconds())
}

func TestWorkerPoolEfficiency() {
	// Test that workers actually run in parallel
	const numWorkers = 4
	const numTasks = 100
	const taskDuration = 10 // milliseconds

	pool := NewWorkerPool(numWorkers)

	go func() {
		for i := 0; i < numTasks; i++ {
			pool.Submit(Task{ID: i, Value: i})
		}
		pool.Shutdown()
	}()

	// Measure time
	start := time.Now()
	count := 0
	for range pool.Results() {
		count++
	}
	elapsed := time.Since(start)

	// Sequential time would be numTasks * taskDuration
	// Parallel should be roughly (numTasks / numWorkers) * taskDuration
	sequentialTime := time.Duration(numTasks*taskDuration) * time.Millisecond
	parallelEstimate := time.Duration((numTasks/numWorkers)*taskDuration) * time.Millisecond

	fmt.Printf("✓ Efficiency test: processed %d tasks in %dms\n", count, elapsed.Milliseconds())
	fmt.Printf("  Sequential would take ~%dms, parallel estimate ~%dms\n",
		sequentialTime.Milliseconds(), parallelEstimate.Milliseconds())

	if elapsed > sequentialTime {
		fmt.Printf("  Warning: Took longer than sequential time\n")
	}
}

func TestLargeWorkload() {
	pool := NewWorkerPool(10)
	numTasks := 10000

	// Submit all tasks
	go func() {
		for i := 0; i < numTasks; i++ {
			pool.Submit(Task{ID: i, Value: i})
		}
		pool.Shutdown()
	}()

	// Collect results
	start := time.Now()
	count := 0
	for range pool.Results() {
		count++
	}
	elapsed := time.Since(start)

	if count != numTasks {
		panic(fmt.Sprintf("Expected %d results, got %d", numTasks, count))
	}

	throughput := float64(numTasks) / elapsed.Seconds()
	fmt.Printf("✓ Large workload test passed (%d tasks in %dms, %.0f tasks/sec)\n",
		numTasks, elapsed.Milliseconds(), throughput)
}

func main() {
	fmt.Println("Running Worker Pool tests...\n")

	TestBasicWorkerPool()
	TestMultipleWorkers()
	TestWorkerPoolEfficiency()
	TestLargeWorkload()

	fmt.Println("\n✓ All tests passed!")
}
