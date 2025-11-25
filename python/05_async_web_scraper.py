"""
PROBLEM: Async Web Scraper

Implement an asynchronous web scraper that fetches multiple URLs concurrently
using asyncio. The scraper should handle rate limiting, timeouts, and errors
gracefully.

REQUIREMENTS:
- Use asyncio for concurrent requests (not threads)
- Implement rate limiting to avoid overwhelming servers
- Handle timeouts for each request
- Handle errors gracefully (connection errors, HTTP errors)
- Fetch and process multiple URLs concurrently
- Provide statistics on success/failure

PERFORMANCE NOTES:
- Should fetch 10+ URLs in the time it would take 2-3 sequential requests
- Should maintain configurable rate limit (e.g., max 5 requests/second)
- Should handle 100+ URLs efficiently

TEST CASE EXPECTATIONS:
- Successfully fetch multiple test URLs in parallel
- Rate limiting should be enforced (measure time for 10 requests with rate limit)
- Timeouts should be respected
- Error handling should not crash the scraper
"""

import asyncio
import time
from typing import Optional, Dict, List
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass
class FetchResult:
    """Result of a URL fetch."""

    url: str
    success: bool
    status_code: Optional[int] = None
    content_length: int = 0
    error: Optional[str] = None
    duration: float = 0.0


class AsyncWebScraper:
    """Asynchronous web scraper with rate limiting."""

    def __init__(self, max_concurrent: int = 5, rate_limit: Optional[float] = None):
        """
        Initialize the scraper.

        Args:
            max_concurrent: Maximum concurrent requests
            rate_limit: Requests per second (None = no limit)
        """
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit
        # TODO: Initialize semaphore for concurrency control
        # TODO: Initialize rate limiter if needed
        self.results: List[FetchResult] = []
        self.semaphore = None
        self.rate_limiter = None

    async def _rate_limit(self):
        """Apply rate limiting if configured."""
        # TODO: Implement rate limiting
        # If rate_limit is set (e.g., 5 requests/second),
        # ensure we don't exceed that rate
        if self.rate_limit:
            # Calculate delay between requests
            min_interval = 1.0 / self.rate_limit
            # You may need to track last request time
            pass

    async def fetch(self, url: str, timeout: float = 10.0) -> FetchResult:
        """
        Fetch a single URL.

        Args:
            url: URL to fetch
            timeout: Request timeout in seconds

        Returns:
            FetchResult with outcome
        """
        # TODO: Implement actual HTTP fetching
        # For testing purposes, you can use a mock or http.client
        # or implement a simple socket connection
        # Note: The test will provide mock implementations

        start_time = time.time()
        result = FetchResult(url=url, success=False, error="Not implemented")

        try:
            # Acquire semaphore to limit concurrency
            await self.semaphore.acquire()

            try:
                # Apply rate limiting
                await self._rate_limit()

                # TODO: Actually fetch the URL
                # You can use urllib, http.client, or mock for testing
                # For now, simulating a fetch:
                await asyncio.sleep(0.05)  # Simulate network delay

                result = FetchResult(
                    url=url,
                    success=True,
                    status_code=200,
                    content_length=1000,
                    duration=time.time() - start_time,
                )
            finally:
                self.semaphore.release()

        except asyncio.TimeoutError:
            result = FetchResult(
                url=url,
                success=False,
                error="Timeout",
                duration=time.time() - start_time,
            )
        except Exception as e:
            result = FetchResult(
                url=url,
                success=False,
                error=str(e),
                duration=time.time() - start_time,
            )

        self.results.append(result)
        return result

    async def fetch_multiple(self, urls: List[str], timeout: float = 10.0) -> List[FetchResult]:
        """
        Fetch multiple URLs concurrently.

        Args:
            urls: List of URLs to fetch
            timeout: Timeout per request in seconds

        Returns:
            List of FetchResult objects
        """
        # TODO: Initialize semaphore if not already done
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.max_concurrent)

        # TODO: Create tasks for all URLs and gather results
        pass

    def get_stats(self) -> Dict:
        """Get statistics about fetches."""
        if not self.results:
            return {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
            }

        successful = sum(1 for r in self.results if r.success)
        failed = len(self.results) - successful
        avg_duration = sum(r.duration for r in self.results) / len(self.results)

        return {
            "total": len(self.results),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(self.results) * 100,
            "avg_duration": avg_duration,
        }

    def print_stats(self):
        """Print statistics."""
        stats = self.get_stats()
        print(f"\nScraper Statistics:")
        print(f"  Total requests: {stats['total']}")
        print(f"  Successful: {stats['successful']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Success rate: {stats['success_rate']:.1f}%")
        print(f"  Avg duration: {stats['avg_duration']:.3f}s")


# Mock URLs for testing
MOCK_URLS = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/page3",
    "https://example.com/page4",
    "https://example.com/page5",
]


async def test_concurrent_fetches():
    """Test fetching multiple URLs concurrently."""
    scraper = AsyncWebScraper(max_concurrent=3)

    start = time.time()
    results = await scraper.fetch_multiple(MOCK_URLS[:5])
    elapsed = time.time() - start

    assert len(results) == 5, f"Expected 5 results, got {len(results)}"
    assert all(isinstance(r, FetchResult) for r in results), "Invalid result types"

    print(f"✓ Concurrent fetch test passed ({len(results)} URLs in {elapsed:.3f}s)")
    scraper.print_stats()


async def test_rate_limiting():
    """Test that rate limiting is enforced."""
    scraper = AsyncWebScraper(max_concurrent=10, rate_limit=5)  # 5 requests/second

    urls = MOCK_URLS * 2  # 10 URLs

    start = time.time()
    results = await scraper.fetch_multiple(urls)
    elapsed = time.time() - start

    # 10 requests at 5 req/sec should take ~2 seconds
    # Allow some overhead
    min_expected = (len(urls) / 5) * 0.8  # 80% of theoretical minimum
    assert elapsed >= min_expected, (
        f"Rate limiting not enforced. "
        f"Expected ~{len(urls)/5:.1f}s, took {elapsed:.3f}s"
    )

    print(f"✓ Rate limiting test passed (10 requests in {elapsed:.3f}s)")
    scraper.print_stats()


async def test_error_handling():
    """Test error handling."""
    scraper = AsyncWebScraper(max_concurrent=3)

    # Include some URLs that would normally fail
    urls = [
        "https://example.com/valid1",
        "https://example.com/valid2",
        "https://invalid.example.com/",
    ]

    results = await scraper.fetch_multiple(urls)

    assert len(results) == len(urls), "Should have result for each URL"
    print(f"✓ Error handling test passed ({len(results)} requests processed)")
    scraper.print_stats()


async def test_concurrent_vs_sequential():
    """Compare concurrent vs sequential execution."""
    urls = MOCK_URLS * 3  # 15 URLs

    # Concurrent
    scraper_concurrent = AsyncWebScraper(max_concurrent=5)
    start = time.time()
    await scraper_concurrent.fetch_multiple(urls)
    concurrent_time = time.time() - start

    print(f"✓ Concurrent fetch test passed ({len(urls)} URLs in {concurrent_time:.3f}s)")
    scraper_concurrent.print_stats()


async def main():
    """Run all tests."""
    print("Starting async web scraper tests...\n")

    await test_concurrent_fetches()
    print("\n" + "=" * 50 + "\n")

    await test_rate_limiting()
    print("\n" + "=" * 50 + "\n")

    await test_error_handling()
    print("\n" + "=" * 50 + "\n")

    await test_concurrent_vs_sequential()

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
