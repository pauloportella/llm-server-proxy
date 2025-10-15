# Integration Tests

## Running Tests

```bash
# From project root
uv run python3 tests/test_integration.py
```

## What's Tested

1. **Longer inference requests** - Validates model responses with higher token counts
2. **Model switching** - Tests switching between `qwen3-30b-instruct` and `lfm2-8b`
3. **Rapid sequential requests** - Stresses the queue with 5 back-to-back requests
4. **Queue integrity** - Verifies all jobs are properly acknowledged (no orphans)

## Expected Output

```
✓✓✓ SUCCESS - All jobs processed and acknowledged correctly! ✓✓✓

Total requests: 9
Successful: 9
Failed: 0
Success rate: 100.0%

Queue depth (should all be 0):
  Pending jobs: 0
  Processing jobs: 0
  Dead-letter queue: 0
  Total jobs: 0
```

## Requirements

- Docker containers running (`docker-compose up -d`)
- Backend models available (`qwen3-30b-instruct`, `lfm2-8b`)
- Python dependencies installed (`requests`)

## Exit Codes

- `0` - All tests passed, queue clean
- `1` - Test failures or orphaned jobs detected
