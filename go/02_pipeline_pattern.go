/*
PROBLEM: Pipeline Pattern

Implement a multi-stage pipeline where data flows through multiple goroutines.
This demonstrates Go's idiomatic channel-based concurrency pattern.

REQUIREMENTS:
- Implement a 3-stage pipeline (source -> transform -> sink)
- Use channels to pass data between stages
- Each stage should be a separate goroutine
- Handle graceful shutdown when done
- No data loss or deadlocks

PERFORMANCE NOTES:
- Should efficiently process large volumes of data
- Minimal latency between stages
- Should scale well with multiple pipeline copies

TEST CASE EXPECTATIONS:
- Process 10000 integers through pipeline correctly
- Each stage should execute in parallel
- Verify output matches expected transformation
- No goroutine leaks or deadlocks
*/

package main

import (
	"fmt"
	"sync"
	"time"
)

// Stage represents a single stage in the pipeline
// It reads from input channel, processes, and sends to output channel

// Source generates numbers and sends them to output channel
func Source(count int) <-chan int {
	// TODO: Create output channel
	// TODO: Start goroutine to generate numbers
	// TODO: Close channel when done
	// TODO: Return channel
	return make(chan int)
}

// Transform reads from input, applies function, sends to output
func Transform(input <-chan int, fn func(int) int) <-chan int {
	// TODO: Create output channel
	// TODO: Start goroutine to:
	//   - Read from input channel
	//   - Apply fn to each value
	//   - Send to output channel
	//   - Close output when input closes
	// TODO: Return output channel
	return make(chan int)
}

// Sink reads from input and returns all values as a slice
func Sink(input <-chan int) []int {
	// TODO: Read all values from input channel
	// TODO: Return as slice
	return []int{}
}

// Pipeline3Stage connects three stages together
// Example of chaining pipeline stages
func Pipeline3Stage(
	count int,
	fn1 func(int) int,
	fn2 func(int) int,
) []int {
	// TODO: Connect source -> transform1 -> transform2 -> sink
	return []int{}
}

func TestBasicPipeline() {
	// Create a simple pipeline that squares numbers
	source := Source(10)
	square := Transform(source, func(x int) int { return x * x })
	results := Sink(square)

	expected := []int{0, 1, 4, 9, 16, 25, 36, 49, 64, 81}

	if len(results) != len(expected) {
		panic(fmt.Sprintf("Expected %d results, got %d", len(expected), len(results)))
	}

	for i, val := range results {
		if val != expected[i] {
			panic(fmt.Sprintf("At index %d: expected %d, got %d", i, expected[i], val))
		}
	}

	fmt.Println("✓ Basic pipeline test passed")
}

func TestMultiStageTransform() {
	// Test a 3-stage pipeline: numbers -> square -> add 1
	results := Pipeline3Stage(10,
		func(x int) int { return x * x },    // square
		func(x int) int { return x + 1 },    // add 1
	)

	expected := []int{1, 2, 5, 10, 17, 26, 37, 50, 65, 82}

	if len(results) != len(expected) {
		panic(fmt.Sprintf("Expected %d results, got %d", len(expected), len(results)))
	}

	for i, val := range results {
		if val != expected[i] {
			panic(fmt.Sprintf("At index %d: expected %d, got %d", i, expected[i], val))
		}
	}

	fmt.Println("✓ Multi-stage transform test passed")
}

func TestParallelPipelines() {
	// Run multiple pipelines in parallel
	const numPipelines = 5
	const itemsPerPipeline = 1000

	var wg sync.WaitGroup
	results := make([][]int, numPipelines)

	start := time.Now()

	for i := 0; i < numPipelines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			source := Source(itemsPerPipeline)
			square := Transform(source, func(x int) int { return x * x })
			results[idx] = Sink(square)
		}(i)
	}

	wg.Wait()
	elapsed := time.Since(start)

	totalItems := numPipelines * itemsPerPipeline
	fmt.Printf("✓ Parallel pipelines test passed (%d items in %dms)\n",
		totalItems, elapsed.Milliseconds())

	// Verify results
	for i := 0; i < numPipelines; i++ {
		if len(results[i]) != itemsPerPipeline {
			panic(fmt.Sprintf("Pipeline %d: expected %d items, got %d",
				i, itemsPerPipeline, len(results[i])))
		}
	}
}

func TestLargePipeline() {
	// Test pipeline with large number of items
	const numItems = 10000

	source := Source(numItems)
	square := Transform(source, func(x int) int { return x * x })
	addOne := Transform(square, func(x int) int { return x + 1 })
	results := Sink(addOne)

	if len(results) != numItems {
		panic(fmt.Sprintf("Expected %d results, got %d", numItems, len(results)))
	}

	// Verify a few values
	expected := []int{1, 2, 5, 10, 17} // (0^2)+1, (1^2)+1, (2^2)+1, ...
	for i := 0; i < len(expected); i++ {
		if results[i] != expected[i] {
			panic(fmt.Sprintf("At index %d: expected %d, got %d", i, expected[i], results[i]))
		}
	}

	fmt.Printf("✓ Large pipeline test passed (%d items)\n", numItems)
}

func main() {
	fmt.Println("Running Pipeline Pattern tests...\n")

	TestBasicPipeline()
	TestMultiStageTransform()
	TestParallelPipelines()
	TestLargePipeline()

	fmt.Println("\n✓ All tests passed!")
}
