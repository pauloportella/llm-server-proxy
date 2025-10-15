#!/usr/bin/env python3
"""Integration tests for LLM Queue Proxy with model switching and queue validation."""

import requests
import json
import time
import sys
import subprocess
from datetime import datetime

BASE_URL = "http://localhost:8888"
TIMEOUT = 60

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def get_metrics():
    """Get current metrics."""
    try:
        resp = requests.get(f"{BASE_URL}/metrics", timeout=5)
        return resp.json()
    except Exception as e:
        log(f"Failed to get metrics: {e}")
        return None

def get_queue_status():
    """Get queue status from metrics."""
    metrics = get_metrics()
    if metrics:
        return metrics.get("queue_stats", {})
    return {}

def test_same_model_longer_request():
    """Test longer request to same model."""
    log("=" * 70)
    log("TEST 1: Longer request to qwen3-30b-instruct (higher token count)")
    log("=" * 70)

    queue_before = get_queue_status()
    log(f"Queue before: {queue_before}")

    payload = {
        "model": "qwen3-30b-instruct",
        "messages": [
            {"role": "user", "content": "Write a detailed explanation of how machine learning works, covering neural networks, training, and common algorithms."}
        ],
        "max_tokens": 200
    }

    try:
        log("Sending request (expecting ~5-10s response time)...")
        start = time.time()
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=TIMEOUT
        )
        elapsed = time.time() - start

        if resp.status_code == 200:
            result = resp.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            preview = content[:80] + "..." if len(content) > 80 else content
            log(f"✓ SUCCESS ({elapsed:.1f}s): {preview}")
            return True
        else:
            log(f"✗ FAILED ({resp.status_code}): {resp.text[:200]}")
            return False
    except requests.Timeout:
        log(f"✗ TIMEOUT after {TIMEOUT}s")
        return False
    except Exception as e:
        log(f"✗ ERROR: {e}")
        return False
    finally:
        queue_after = get_queue_status()
        log(f"Queue after: {queue_after}\n")

def test_model_switching():
    """Test switching between different models."""
    log("=" * 70)
    log("TEST 2: Model switching (qwen3 -> lfm2 -> qwen3)")
    log("=" * 70)

    models = [
        ("qwen3-30b-instruct", "Explain quantum computing in 100 words"),
        ("lfm2-8b", "What is machine learning?"),
        ("qwen3-30b-instruct", "Tell me about reinforcement learning")
    ]

    for i, (model, prompt) in enumerate(models, 1):
        log(f"\n[2.{i}] Switching to {model}")
        log(f"    Prompt: {prompt}")

        queue_before = get_queue_status()
        log(f"    Queue before: pending={queue_before.get('pending_jobs', 0)}, processing={queue_before.get('processing_jobs', 0)}")

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150
        }

        try:
            start = time.time()
            resp = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                json=payload,
                timeout=TIMEOUT
            )
            elapsed = time.time() - start

            if resp.status_code == 200:
                result = resp.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                preview = content[:70] + "..." if len(content) > 70 else content
                log(f"    ✓ Response ({elapsed:.1f}s): {preview}")
            else:
                log(f"    ✗ Failed ({resp.status_code})")
        except requests.Timeout:
            log(f"    ⏱ Timeout")
        except Exception as e:
            log(f"    ✗ Error: {e}")

        queue_after = get_queue_status()
        log(f"    Queue after: pending={queue_after.get('pending_jobs', 0)}, processing={queue_after.get('processing_jobs', 0)}")

        if i < len(models):
            time.sleep(1)

    print()

def test_rapid_sequential():
    """Test rapid sequential requests to stress queue."""
    log("=" * 70)
    log("TEST 3: Rapid sequential requests (5 requests)")
    log("=" * 70)

    requests_list = [
        ("qwen3-30b-instruct", "Explain photosynthesis"),
        ("qwen3-30b-instruct", "What is climate change?"),
        ("qwen3-30b-instruct", "Describe the water cycle"),
        ("qwen3-30b-instruct", "What is DNA?"),
        ("qwen3-30b-instruct", "Explain gravity"),
    ]

    for i, (model, prompt) in enumerate(requests_list, 1):
        log(f"[3.{i}] Sending: {prompt[:40]}...")

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100
        }

        try:
            start = time.time()
            resp = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                json=payload,
                timeout=TIMEOUT
            )
            elapsed = time.time() - start

            if resp.status_code == 200:
                result = resp.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                preview = content[:50] + "..." if len(content) > 50 else content
                log(f"     ✓ ({elapsed:.1f}s): {preview}")
            else:
                log(f"     ✗ Failed ({resp.status_code})")
        except requests.Timeout:
            log(f"     ⏱ Timeout")
        except Exception as e:
            log(f"     ✗ Error: {e}")

        if i < len(requests_list):
            time.sleep(0.5)

    print()

def check_final_state():
    """Check final queue state - should be EMPTY."""
    log("=" * 70)
    log("FINAL STATE CHECK - Queue should be CLEAN")
    log("=" * 70)

    time.sleep(2)  # Wait for queue to process

    metrics = get_metrics()
    if metrics:
        log(f"Total requests: {metrics.get('total_requests', 0)}")
        log(f"Successful: {metrics.get('successful_requests', 0)}")
        log(f"Failed: {metrics.get('failed_requests', 0)}")
        log(f"Cancelled: {metrics.get('cancelled_requests', 0)}")
        log(f"Success rate: {metrics.get('success_rate', 0):.1%}")

        qs = metrics.get("queue_stats", {})
        log(f"\nQueue depth (should all be 0):")
        log(f"  Pending jobs: {qs.get('pending_jobs', 0)}")
        log(f"  Processing jobs: {qs.get('processing_jobs', 0)}")
        log(f"  Dead-letter queue: {qs.get('dead_letter_queue', 0)}")
        log(f"  Total jobs: {qs.get('total_jobs', 0)}")

        # Check Redis directly
        log(f"\nDirect Redis inspection:")
        try:
            pending = subprocess.check_output(
                ["docker", "exec", "llm-queue-redis", "redis-cli", "-a", "redis_password", "--no-auth-warning", "ZCARD", "llm_requests"],
                text=True
            ).strip()
            processing = subprocess.check_output(
                ["docker", "exec", "llm-queue-redis", "redis-cli", "-a", "redis_password", "--no-auth-warning", "SCARD", "llm_processing_ids"],
                text=True
            ).strip()
            dlq = subprocess.check_output(
                ["docker", "exec", "llm-queue-redis", "redis-cli", "-a", "redis_password", "--no-auth-warning", "LLEN", "llm_dlq"],
                text=True
            ).strip()
            log(f"  Redis pending: {pending}")
            log(f"  Redis processing: {processing}")
            log(f"  Redis DLQ: {dlq}")

            if pending == "0" and processing == "0" and dlq == "0":
                log("\n✓✓✓ SUCCESS - All jobs processed and acknowledged correctly! ✓✓✓")
                return True
            else:
                log(f"\n⚠ WARNING - Jobs still in queue: pending={pending}, processing={processing}, dlq={dlq}")
                return False

        except Exception as e:
            log(f"  Could not check Redis: {e}")
            return False

    print()
    return False

if __name__ == "__main__":
    log("=" * 70)
    log("COMPREHENSIVE LLM QUEUE PROXY INTEGRATION TESTS")
    log("=" * 70)
    log(f"Target: {BASE_URL}")
    log("Testing: Queue integrity, model switching, job acknowledgement")
    print()

    # Run tests
    test_same_model_longer_request()
    time.sleep(2)

    test_model_switching()
    time.sleep(2)

    test_rapid_sequential()
    time.sleep(2)

    success = check_final_state()

    log("All tests complete!")
    sys.exit(0 if success else 1)
