/*
PROBLEM: Rate Limiter

Implement a rate limiter using Go channels and goroutines. This is useful for
controlling throughput in concurrent systems.

REQUIREMENTS:
- Implement token bucket rate limiting
- Support configurable rate (e.g., 5 requests per second)
- Block or return error when rate limit exceeded
- Use channels and goroutines idiomatically
- Handle burst capacity

PERFORMANCE NOTES:
- Should accurately enforce configured rate
- Should use minimal CPU when idle
- Should handle 1000+ requests efficiently

TEST CASE EXPECTATIONS:
- 100 requests at 10 req/sec should take ~10 seconds
- Burst requests should be possible within capacity
- Rate limiting should be enforced consistently
*/

package main

import (
	"fmt"
	"sync"
	"time"
)

// RateLimiter implements token bucket rate limiting
type RateLimiter struct {
	tokens chan int
	ticker *time.Ticker
}

// NewRateLimiter creates a rate limiter with specified rate
// rate: requests per second
// burst: maximum tokens to accumulate
func NewRateLimiter(rate float64, burst int) *RateLimiter {
	// TODO: Create token channel with initial burst capacity
	// TODO: Start ticker goroutine that:
	//   - Fires at specified rate
	//   - Adds tokens to channel
	//   - Keeps tokens at maximum of burst
	// TODO: Return limiter
	return &RateLimiter{}
}

// Wait blocks until a token is available
func (rl *RateLimiter) Wait() {
	// TODO: Block until token received from channel
}

// TryAcquire attempts to acquire a token without blocking
// Returns true if token acquired, false if none available
func (rl *RateLimiter) TryAcquire() bool {
	// TODO: Try non-blocking receive from token channel
	return false
}

// Stop stops the rate limiter
func (rl *RateLimiter) Stop() {
	// TODO: Stop ticker and close channels
}

// Limiter is a simpler rate limiter based on time.Ticker
type Limiter struct {
	limiter chan struct{}
	ticker  *time.Ticker
	done    chan bool
}

// NewLimiter creates a new rate limiter
// rps: requests per second
func NewLimiter(rps int) *Limiter {
	// TODO: Implement simpler version using time.Ticker
	// Emit token every 1/rps seconds
	return &Limiter{}
}

// Allow checks if request is allowed
// Returns true if allowed, false if rate limit exceeded
func (l *Limiter) Allow() bool {
	// TODO: Try to receive from limiter channel
	return false
}

// Stop closes the limiter
func (l *Limiter) Stop() {
	// TODO: Signal done and clean up
}

func TestBasicRateLimit() {
	limiter := NewLimiter(10) // 10 requests per second
	defer limiter.Stop()

	// Make 20 requests with rate limiting
	start := time.Now()
	count := 0

	for i := 0; i < 20; i++ {
		if limiter.Allow() {
			count++
		}
	}

	elapsed := time.Since(start)

	// 20 requests at 10 req/sec should take ~2 seconds
	// Allow some variance
	expectedTime := time.Duration(count) * time.Second / 10
	if elapsed < expectedTime*8/10 {
		fmt.Printf("Warning: Rate limiting too fast (took %dms, expected ~%dms)\n",
			elapsed.Milliseconds(), expectedTime.Milliseconds())
	}

	fmt.Printf("✓ Basic rate limit test passed (%d requests in %dms)\n",
		count, elapsed.Milliseconds())
}

func TestBurstCapacity() {
	// Test that limiter allows burst up to configured capacity
	limiter := NewRateLimiter(5, 10) // 5 req/sec, burst of 10
	defer limiter.Stop()

	// Try to get 15 tokens
	burstCount := 0
	for i := 0; i < 15; i++ {
		if limiter.TryAcquire() {
			burstCount++
		}
	}

	// Should get at least the burst amount
	if burstCount < 8 {
		panic(fmt.Sprintf("Expected burst capacity, got only %d tokens", burstCount))
	}

	fmt.Printf("✓ Burst capacity test passed (%d tokens acquired\n", burstCount)
}

func TestConcurrentRequests() {
	limiter := NewLimiter(10) // 10 requests per second
	defer limiter.Stop()

	const numGoroutines = 5
	const requestsPerGoroutine = 20

	var wg sync.WaitGroup
	var mu sync.Mutex
	successCount := 0

	start := time.Now()

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < requestsPerGoroutine; j++ {
				limiter.Wait()
				mu.Lock()
				successCount++
				mu.Unlock()
			}
		}()
	}

	wg.Wait()
	elapsed := time.Since(start)

	totalRequests := numGoroutines * requestsPerGoroutine
	expectedTime := time.Duration(totalRequests) * time.Second / 10

	fmt.Printf("✓ Concurrent requests test passed (%d requests in %dms, expected ~%dms)\n",
		totalRequests, elapsed.Milliseconds(), expectedTime.Milliseconds())

	if successCount != totalRequests {
		panic(fmt.Sprintf("Expected %d successes, got %d", totalRequests, successCount))
	}
}

func TestRateLimitingUnderLoad() {
	limiter := NewLimiter(100) // 100 requests per second
	defer limiter.Stop()

	const duration = 2 * time.Second
	const expectedRate = 100

	start := time.Now()
	count := 0

	for time.Since(start) < duration {
		limiter.Wait()
		count++
	}

	elapsed := time.Since(start)
	actualRate := float64(count) / elapsed.Seconds()

	fmt.Printf("✓ Load test: %.0f requests in %.1fs (%.0f req/s)\n",
		float64(count), elapsed.Seconds(), actualRate)

	// Should be close to configured rate
	if actualRate < float64(expectedRate)*0.8 {
		fmt.Printf("Warning: Rate too low (%.0f req/s, expected ~%d)\n",
			actualRate, expectedRate)
	}
}

func TestMultipleLimiters() {
	// Test multiple independent limiters
	limiter1 := NewLimiter(10)
	limiter2 := NewLimiter(20)
	defer limiter1.Stop()
	defer limiter2.Stop()

	var wg sync.WaitGroup
	var mu sync.Mutex
	count1, count2 := 0, 0

	start := time.Now()

	// Goroutine using limiter1
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 20; i++ {
			limiter1.Wait()
			mu.Lock()
			count1++
			mu.Unlock()
		}
	}()

	// Goroutine using limiter2
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 20; i++ {
			limiter2.Wait()
			mu.Lock()
			count2++
			mu.Unlock()
		}
	}()

	wg.Wait()
	elapsed := time.Since(start)

	fmt.Printf("✓ Multiple limiters test passed\n")
	fmt.Printf("  Limiter1 (10 req/s): %d requests in %dms\n", count1, elapsed.Milliseconds())
	fmt.Printf("  Limiter2 (20 req/s): %d requests in %dms\n", count2, elapsed.Milliseconds())

	// Limiter2 should complete faster than Limiter1
	if count1 >= count2 {
		fmt.Printf("Warning: Rates don't seem independent\n")
	}
}

func main() {
	fmt.Println("Running Rate Limiter tests...\n")

	TestBasicRateLimit()
	TestBurstCapacity()
	TestConcurrentRequests()
	TestRateLimitingUnderLoad()
	TestMultipleLimiters()

	fmt.Println("\n✓ All tests passed!")
}
