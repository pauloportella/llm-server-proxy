#!/usr/bin/env python3
"""Test parallel processing with multiple workers."""

import requests
import time
import threading
from datetime import datetime

BASE_URL = "http://localhost:8888"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}")

def send_request(request_id, results):
    """Send a single request and record timing."""
    payload = {
        "model": "qwen3-30b-instruct",
        "messages": [{"role": "user", "content": f"Write a detailed technical explanation of how neural networks work, covering backpropagation, gradient descent, activation functions, and common architectures. Include examples. (Request {request_id})"}],
        "max_tokens": 500  # Force longer generation time (~10-15s per request)
    }

    start = time.time()
    log(f"Request {request_id}: SENT")

    try:
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=60
        )
        elapsed = time.time() - start

        if resp.status_code == 200:
            result = resp.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            log(f"Request {request_id}: COMPLETED in {elapsed:.2f}s - {content[:50]}")
            results[request_id] = {"status": "success", "time": elapsed}
        else:
            log(f"Request {request_id}: FAILED ({resp.status_code})")
            results[request_id] = {"status": "failed", "time": elapsed}
    except Exception as e:
        elapsed = time.time() - start
        log(f"Request {request_id}: ERROR after {elapsed:.2f}s - {e}")
        results[request_id] = {"status": "error", "time": elapsed}

def test_concurrent_requests(num_requests):
    """Send multiple concurrent requests."""
    log(f"\n{'='*70}")
    log(f"TEST: {num_requests} CONCURRENT REQUESTS TO SAME MODEL")
    log(f"{'='*70}")

    # Get initial metrics
    try:
        metrics_before = requests.get(f"{BASE_URL}/metrics", timeout=5).json()
        log(f"Workers configured: {metrics_before.get('num_workers', 'unknown')}")
        log(f"Active workers before: {metrics_before.get('active_workers', 0)}")
    except:
        log("Could not fetch initial metrics")

    results = {}
    threads = []

    # Start all requests at once
    overall_start = time.time()
    for i in range(num_requests):
        t = threading.Thread(target=send_request, args=(i+1, results))
        threads.append(t)
        t.start()
        time.sleep(0.05)  # Small stagger to avoid connection issues

    # Wait for all to complete
    for t in threads:
        t.join()

    overall_elapsed = time.time() - overall_start

    # Analyze results
    log(f"\n{'='*70}")
    log(f"RESULTS")
    log(f"{'='*70}")
    log(f"Total wall time: {overall_elapsed:.2f}s")

    successful = [r for r in results.values() if r["status"] == "success"]
    if successful:
        avg_time = sum(r["time"] for r in successful) / len(successful)
        max_time = max(r["time"] for r in successful)
        min_time = min(r["time"] for r in successful)

        log(f"Successful requests: {len(successful)}/{num_requests}")
        log(f"Individual request times: min={min_time:.2f}s, max={max_time:.2f}s, avg={avg_time:.2f}s")

        # Calculate theoretical speedup
        sequential_time = sum(r["time"] for r in successful)
        speedup = sequential_time / overall_elapsed if overall_elapsed > 0 else 0
        log(f"\nTheoretical sequential time: {sequential_time:.2f}s")
        log(f"Actual parallel time: {overall_elapsed:.2f}s")
        log(f"Speedup: {speedup:.2f}x")

        # Get final metrics
        try:
            metrics_after = requests.get(f"{BASE_URL}/metrics", timeout=5).json()
            log(f"\nPeak concurrent requests: {metrics_after.get('peak_concurrent_requests', 0)}")
            log(f"Active workers after: {metrics_after.get('active_workers', 0)}")
        except:
            pass

    return len(successful) == num_requests

if __name__ == "__main__":
    # Test with 4 concurrent requests (should use both workers)
    test_concurrent_requests(4)

    time.sleep(2)

    # Test with 2 concurrent requests (exactly matches worker count)
    test_concurrent_requests(2)
