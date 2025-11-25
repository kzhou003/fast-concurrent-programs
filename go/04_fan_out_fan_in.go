/*
PROBLEM: Fan-Out / Fan-In Pattern

Implement the fan-out/fan-in pattern where work is distributed to multiple
worker goroutines (fan-out) and results are aggregated (fan-in).
This is a powerful pattern for parallel processing in Go.

REQUIREMENTS:
- Distribute work from single input to multiple workers
- Each worker processes independently
- Merge results from all workers back into single stream
- Handle proper channel closing and synchronization
- No data loss or deadlocks

PERFORMANCE NOTES:
- Should process items concurrently across multiple workers
- Aggregation should be efficient
- Should scale with number of workers

TEST CASE EXPECTATIONS:
- Fan out 100 items to 5 workers should process all
- Results should be collectable regardless of order
- Proper channel closure prevents goroutine leaks
- Output should contain all processed items
*/

package main

import (
	"fmt"
	"sync"
	"time"
)

// FanOut distributes items from input to multiple output channels
func FanOut(input <-chan int, numWorkers int) []<-chan int {
	// TODO: Create numWorkers output channels
	// TODO: Start goroutine that:
	//   - Reads from input
	//   - Distributes to each output channel
	//   - Closes all outputs when input is exhausted
	// TODO: Return slice of output channels
	return nil
}

// Worker processes items from input and sends to output
// (used in fan-out stage)
func Worker(id int, input <-chan int) <-chan int {
	// TODO: Create output channel
	// TODO: Start goroutine that:
	//   - Reads from input
	//   - Processes items (simple transformation)
	//   - Sends to output
	//   - Closes output when input closes
	// TODO: Return output channel
	return make(chan int)
}

// FanIn merges multiple input channels into a single output channel
func FanIn(inputs ...<-chan int) <-chan int {
	// TODO: Create output channel
	// TODO: Use sync.WaitGroup to track when all inputs are exhausted
	// TODO: Start goroutine that:
	//   - Reads from all input channels using select
	//   - Sends to output channel
	//   - Closes output when all inputs are exhausted
	// TODO: Return output channel
	return make(chan int)
}

func TestBasicFanOut() {
	// Create a source
	source := make(chan int)

	go func() {
		for i := 0; i < 20; i++ {
			source <- i
		}
		close(source)
	}()

	// Fan out to 4 workers
	outputs := FanOut(source, 4)

	// Collect all results
	var wg sync.WaitGroup
	results := make([]int, 0, 20)
	var mu sync.Mutex

	for _, output := range outputs {
		wg.Add(1)
		go func(ch <-chan int) {
			defer wg.Done()
			for val := range ch {
				mu.Lock()
				results = append(results, val)
				mu.Unlock()
			}
		}(output)
	}

	wg.Wait()

	if len(results) != 20 {
		panic(fmt.Sprintf("Expected 20 results, got %d", len(results)))
	}

	fmt.Printf("✓ Basic fan-out test passed (%d items distributed)\n", len(results))
}

func TestBasicFanIn() {
	// Create multiple input sources
	const numSources = 5
	const itemsPerSource = 10

	inputs := make([]<-chan int, numSources)

	for i := 0; i < numSources; i++ {
		ch := make(chan int)
		inputs[i] = ch

		go func(source int, c chan<- int) {
			for j := 0; j < itemsPerSource; j++ {
				c <- source*1000 + j
			}
			close(c)
		}(i, ch)
	}

	// Fan in to single output
	output := FanIn(inputs...)

	// Collect results
	results := make([]int, 0, numSources*itemsPerSource)
	for val := range output {
		results = append(results, val)
	}

	if len(results) != numSources*itemsPerSource {
		panic(fmt.Sprintf("Expected %d results, got %d",
			numSources*itemsPerSource, len(results)))
	}

	fmt.Printf("✓ Basic fan-in test passed (%d items merged)\n", len(results))
}

func TestFanOutFanIn() {
	// Complete pipeline: source -> fan out -> workers -> fan in -> sink

	// Create source
	source := make(chan int)

	go func() {
		for i := 0; i < 100; i++ {
			source <- i
		}
		close(source)
	}()

	// Fan out to workers
	const numWorkers = 4
	workers := make([]<-chan int, numWorkers)

	for i := 0; i < numWorkers; i++ {
		workers[i] = Worker(i, source) // Note: this needs refactoring
	}

	// Alternative: properly fan out and process
	workerOutputs := FanOut(source, numWorkers)

	// Apply transformation to each worker output
	processedOutputs := make([]<-chan int, len(workerOutputs))
	for i, output := range workerOutputs {
		ch := make(chan int)
		processedOutputs[i] = ch

		go func(in <-chan int, out chan<- int) {
			for val := range in {
				// Simple transformation: square
				out <- val * val
			}
			close(out)
		}(output, ch)
	}

	// Fan in results
	finalOutput := FanIn(processedOutputs...)

	// Collect all results
	results := make([]int, 0, 100)
	for val := range finalOutput {
		results = append(results, val)
	}

	if len(results) != 100 {
		panic(fmt.Sprintf("Expected 100 results, got %d", len(results)))
	}

	fmt.Printf("✓ Fan-out/Fan-in pipeline test passed (%d items processed)\n", len(results))
}

func TestFanOutFanInWithTiming() {
	// Test with timing to verify parallelism

	source := make(chan int)
	const numItems = 50
	const numWorkers = 5

	go func() {
		for i := 0; i < numItems; i++ {
			source <- i
		}
		close(source)
	}()

	start := time.Now()

	// Fan out
	outputs := FanOut(source, numWorkers)

	// Process with delay to simulate work
	processedOutputs := make([]<-chan int, len(outputs))
	for i, output := range outputs {
		ch := make(chan int)
		processedOutputs[i] = ch

		go func(in <-chan int, out chan<- int) {
			for val := range in {
				time.Sleep(time.Millisecond) // Simulate work
				out <- val * 2
			}
			close(out)
		}(output, ch)
	}

	// Fan in
	finalOutput := FanIn(processedOutputs...)

	// Collect results
	count := 0
	for range finalOutput {
		count++
	}

	elapsed := time.Since(start)

	if count != numItems {
		panic(fmt.Sprintf("Expected %d results, got %d", numItems, count))
	}

	fmt.Printf("✓ Fan-out/Fan-in timing test passed (%d items in %dms)\n",
		numItems, elapsed.Milliseconds())

	// With parallelism, should be roughly (numItems/numWorkers) * 1ms
	// Sequential would be numItems * 1ms
	expectedParallel := time.Duration((numItems/numWorkers)*1) * time.Millisecond
	if elapsed > expectedParallel*2 {
		fmt.Printf("  Warning: Slower than expected parallel execution\n")
	}
}

func TestComplexFanPattern() {
	// Test with multiple stages of fan-out/fan-in

	// Stage 1: Generate numbers
	source := make(chan int)
	go func() {
		for i := 0; i < 100; i++ {
			source <- i
		}
		close(source)
	}()

	// Stage 2: Fan out to 4 workers, each squares
	squares := FanOut(source, 4)
	squaredOutputs := make([]<-chan int, len(squares))
	for i, output := range squares {
		ch := make(chan int)
		squaredOutputs[i] = ch
		go func(in <-chan int, out chan<- int) {
			for val := range in {
				out <- val * val
			}
			close(out)
		}(output, ch)
	}

	// Stage 3: Fan in results, then fan out again to 3 workers for another op
	merged := FanIn(squaredOutputs...)
	final := FanOut(merged, 3)

	// Stage 4: Fan in final results
	finalMerged := FanIn(final...)

	// Collect results
	results := make([]int, 0, 100)
	for val := range finalMerged {
		results = append(results, val)
	}

	if len(results) != 100 {
		panic(fmt.Sprintf("Expected 100 results, got %d", len(results)))
	}

	fmt.Printf("✓ Complex fan pattern test passed (%d items through multi-stage pipeline)\n",
		len(results))
}

func main() {
	fmt.Println("Running Fan-Out/Fan-In Pattern tests...\n")

	TestBasicFanOut()
	TestBasicFanIn()
	TestFanOutFanIn()
	TestFanOutFanInWithTiming()
	TestComplexFanPattern()

	fmt.Println("\n✓ All tests passed!")
}
