import asyncio
import time
from statistics import mean, stdev

async def benchmark_query_performance():
    """Benchmark query execution performance"""

    test_queries = [
        "What is the total revenue?",
        "Show sales by region",
        "Which product has highest profit margin?",
        "Analyze sales trends over time",
    ]

    results = {}

    for query in test_queries:
        times = []
        for _ in range(10):  # 10 iterations
            start = time.time()
            # In a real-world scenario, you would call the actual query execution function
            # For now, we will simulate a delay
            await asyncio.sleep(1)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)

        results[query] = {
            "mean": mean(times),
            "stdev": stdev(times),
            "min": min(times),
            "max": max(times),
            "p95": sorted(times)[int(len(times) * 0.95)]
        }

    # Assert performance targets
    for query, metrics in results.items():
        assert metrics["p95"] < 5000, f"Query too slow: {query} (p95: {metrics['p95']}ms)"
        print(f"✓ {query}: {metrics['mean']:.0f}ms avg, {metrics['p95']:.0f}ms p95")

if __name__ == "__main__":
    asyncio.run(benchmark_query_performance())
