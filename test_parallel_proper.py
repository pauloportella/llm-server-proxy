#!/usr/bin/env python3
"""Proper test: Sequential vs Parallel processing with unique requests."""

import requests
import time
import threading
from datetime import datetime

BASE_URL = "http://localhost:8888"

# Generate completely different prompts to avoid cache benefits
PROMPTS = [
    "Explain the history and evolution of the Roman Empire from 27 BC to 476 AD, including major emperors, military campaigns, and cultural achievements.",
    "Describe in detail the process of photosynthesis in plants, including light-dependent and light-independent reactions, and the role of chloroplasts.",
    "Write a comprehensive guide to quantum mechanics, covering wave-particle duality, the uncertainty principle, and the Schrödinger equation.",
    "Explain the architecture and inner workings of modern CPU processors, including instruction pipelines, cache hierarchies, and branch prediction.",
]

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}")

def send_request(request_id, prompt, results):
    """Send a single request and record timing."""
    payload = {
        "model": "qwen3-30b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400  # Long enough for ~8-12s per request
    }

    start = time.time()
    log(f"Request {request_id}: SENT - {prompt[:60]}...")

    try:
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=120
        )
        elapsed = time.time() - start

        if resp.status_code == 200:
            result = resp.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            tokens = len(content.split())
            log(f"Request {request_id}: COMPLETED in {elapsed:.2f}s ({tokens} tokens)")
            results[request_id] = {"status": "success", "time": elapsed, "tokens": tokens}
        else:
            log(f"Request {request_id}: FAILED ({resp.status_code})")
            results[request_id] = {"status": "failed", "time": elapsed}
    except Exception as e:
        elapsed = time.time() - start
        log(f"Request {request_id}: ERROR after {elapsed:.2f}s - {e}")
        results[request_id] = {"status": "error", "time": elapsed}

def test_sequential():
    """Send requests one by one sequentially."""
    log(f"\n{'='*80}")
    log(f"TEST 1: SEQUENTIAL PROCESSING (baseline)")
    log(f"{'='*80}")
    log(f"Sending {len(PROMPTS)} requests ONE AT A TIME...\n")

    results = {}
    overall_start = time.time()

    for i, prompt in enumerate(PROMPTS):
        send_request(i+1, prompt, results)

    overall_elapsed = time.time() - overall_start

    log(f"\n{'='*80}")
    log(f"SEQUENTIAL RESULTS")
    log(f"{'='*80}")
    log(f"Total time: {overall_elapsed:.2f}s")

    successful = [r for r in results.values() if r["status"] == "success"]
    if successful:
        individual_times = [r["time"] for r in successful]
        log(f"Successful: {len(successful)}/{len(PROMPTS)}")
        log(f"Individual times: {', '.join(f'{t:.2f}s' for t in individual_times)}")
        log(f"Average per request: {sum(individual_times)/len(individual_times):.2f}s")

    return overall_elapsed, results

def test_parallel():
    """Send all requests concurrently."""
    log(f"\n{'='*80}")
    log(f"TEST 2: PARALLEL PROCESSING (2 workers)")
    log(f"{'='*80}")
    log(f"Sending {len(PROMPTS)} requests CONCURRENTLY...\n")

    results = {}
    threads = []

    overall_start = time.time()

    # Launch all requests at once
    for i, prompt in enumerate(PROMPTS):
        t = threading.Thread(target=send_request, args=(i+1, prompt, results))
        threads.append(t)
        t.start()
        time.sleep(0.05)  # Tiny stagger to avoid connection issues

    # Wait for all to complete
    for t in threads:
        t.join()

    overall_elapsed = time.time() - overall_start

    log(f"\n{'='*80}")
    log(f"PARALLEL RESULTS")
    log(f"{'='*80}")
    log(f"Total time: {overall_elapsed:.2f}s")

    successful = [r for r in results.values() if r["status"] == "success"]
    if successful:
        individual_times = [r["time"] for r in successful]
        log(f"Successful: {len(successful)}/{len(PROMPTS)}")
        log(f"Individual times: {', '.join(f'{t:.2f}s' for t in individual_times)}")
        log(f"Min: {min(individual_times):.2f}s, Max: {max(individual_times):.2f}s, Avg: {sum(individual_times)/len(individual_times):.2f}s")

    # Get metrics
    try:
        metrics = requests.get(f"{BASE_URL}/metrics", timeout=5).json()
        log(f"\nPeak concurrent requests: {metrics.get('peak_concurrent_requests', 0)}")
        log(f"Num workers: {metrics.get('num_workers', 0)}")
    except:
        pass

    return overall_elapsed, results

if __name__ == "__main__":
    log("="*80)
    log("PROPER SEQUENTIAL vs PARALLEL COMPARISON TEST")
    log("="*80)
    log(f"Testing with {len(PROMPTS)} completely different prompts")
    log("Each prompt is unique to avoid KV cache benefits")
    print()

    # Test 1: Sequential
    seq_time, seq_results = test_sequential()

    time.sleep(3)  # Let queue settle

    # Test 2: Parallel
    par_time, par_results = test_parallel()

    # Final comparison
    log(f"\n{'='*80}")
    log(f"FINAL COMPARISON")
    log(f"{'='*80}")
    log(f"Sequential time: {seq_time:.2f}s")
    log(f"Parallel time:   {par_time:.2f}s")

    if par_time > 0:
        speedup = seq_time / par_time
        improvement = ((seq_time - par_time) / seq_time) * 100
        log(f"\nSpeedup: {speedup:.2f}x")
        log(f"Time saved: {seq_time - par_time:.2f}s ({improvement:.1f}% faster)")

        if speedup >= 1.5:
            log(f"\n✓ EXCELLENT: Parallel processing is {speedup:.2f}x faster!")
        elif speedup >= 1.2:
            log(f"\n✓ GOOD: Parallel processing is working ({speedup:.2f}x speedup)")
        else:
            log(f"\n⚠ WARNING: Limited parallelization benefit ({speedup:.2f}x speedup)")
