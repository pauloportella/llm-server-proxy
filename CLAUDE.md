# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**LLM Queue Proxy** - OpenAI-compatible proxy server with automatic model swapping for single-GPU homelab deployments.

**Key Problem Solved:** Prevents GPU memory crashes by ensuring only ONE large language model is loaded at a time, while maintaining OpenAI API compatibility for tools like Open WebUI and n8n.

## Architecture

### Core Components

1. **app/main.py** - FastAPI application with:
   - Redis-based request queue (FIFO, persistent)
   - Multiple parallel queue workers (configurable)
   - Model switcher (Docker SDK start/stop)
   - Health polling (waits for backend readiness)
   - OpenAI-compatible endpoints (via LiteLLM integration)

2. **app/config.py** - YAML configuration loader
   - Loads model definitions from config.yml
   - Validates model configurations
   - Provides model lookup

3. **app/models.py** - Minimal Pydantic models (LiteLLM integration)
   - Minimal request validation (model name, messages, stream flag)
   - Model configuration schema
   - Health check responses
   - OpenAI compatibility delegated to LiteLLM

4. **config.yml** - Model definitions
   - Container names
   - Backend URLs
   - Health check URLs
   - Startup timeouts

### Design Decisions

**Why Docker start/stop instead of create/destroy?**
- Faster switching (no container recreation overhead)
- Container logs preserved
- Matches docker-compose patterns
- State persistence

**Why Redis queue (v2.0 upgrade)?**
- **Reliability:** Atomic dequeue-to-processing transfers prevent lost requests
- **Job Acknowledgement:** Failed jobs automatically retry (up to 3x) then move to DLQ
- **Persistence:** Queue survives proxy restarts
- **Observability:** Inspect queue state externally via redis-cli
- **Stalled Job Recovery:** Automatic recovery if worker crashes mid-processing
- **Client Cancellation:** Proper detection of client disconnects with no queue corruption

**Why LiteLLM integration (v2.1-litellm)?**
- **Zero OpenAI API maintenance:** LiteLLM handles all OpenAI schema validation and updates
- **Vision support:** Image input (URLs/base64) works automatically with LLaVA models
- **Function calling:** Tool/function calling supported out of the box
- **Streaming:** Simplified streaming implementation with proper SSE formatting
- **Cleaner codebase:** Removed 50+ lines of manual request forwarding and Pydantic models
- **Future-proof:** New OpenAI features automatically supported via LiteLLM updates
- **Separation of concerns:** Proxy focuses on GPU management, LiteLLM handles API compatibility

**Why health polling?**
- llama-server takes 90-180s to load models
- Prevents forwarding requests to unready backends
- Graceful handling of startup delays

**Why separate model_switch_lock and parallel workers? (v2.1)**
- **model_switch_lock**: Serializes Docker operations (container start/stop) only
- **Parallel workers**: Multiple workers can forward requests to same backend concurrently
- **Result**: When model is already loaded, requests process in parallel (~2x speedup)
- **Safety**: Model switching still serialized (no GPU memory conflicts)
- **Metrics lock**: Protects counter updates from race conditions

## Development Workflow

### Adding a New Model

1. Create the backend container:
   ```bash
   docker create --name your-model-server -p 8080:8080 \
     --device /dev/dri --device /dev/kfd \
     -v /mnt/ai_models:/models \
     kyuz0/amd-strix-halo-toolboxes:rocm-7rc \
     llama-server -m /models/your-model.gguf \
     --alias your-model -ngl 999 -c 32768 -ub 2048 --no-mmap \
     --flash-attn on \
     --host 0.0.0.0 --port 8080 --jinja
   ```

2. Add to `config.yml`:
   ```yaml
   models:
     your-model:
       container_name: your-model-server
       backend_url: http://localhost:8080/v1
       health_url: http://localhost:8080/health
       startup_timeout: 90
   ```

3. Restart proxy: `docker-compose restart`

### Migration to v2.1-litellm (LiteLLM Integration)

**What Changed:**
- **API forwarding:** Replaced manual aiohttp with `litellm.acompletion()`
- **Request validation:** Simplified from 60+ fields to 3 essential fields (model, messages, stream)
- **Code reduction:** `models.py` reduced from 119 to 61 lines (~50% smaller)
- **Vision support:** Image inputs (URLs/base64) now work automatically
- **Function calling:** Tool/function calling supported out of the box
- **Streaming:** Improved streaming with proper SSE formatting

**Upgrade Steps:**
1. Pull latest code from `feature/litellm-integration` branch
2. Run: `docker-compose up -d --build` (installs litellm==1.78.5)
3. Verify: `curl http://localhost:8888/health`
4. Test: Vision models and function calling now work without code changes

**Breaking Changes:**
- ❌ None - fully backward compatible
- ✅ All existing endpoints work identically
- ✅ All tests pass with 100% success rate

**New Capabilities:**
- ✅ Vision/multimodal inputs (e.g., LLaVA models with images)
- ✅ Function calling (OpenAI-style tools)
- ✅ Automatic OpenAI API updates via LiteLLM

### Migration from v1 (in-memory) to v2 (Redis)

**What Changed:**
- Queue now persists in Redis (survives container restarts)
- Failed jobs automatically retry up to 3 times
- Dead-letter queue stores permanently failed jobs
- Client cancellations no longer corrupt queue state
- New monitoring endpoints: `/metrics`, `/queue/dlq`

**Upgrade Steps:**
1. Pull latest code (includes new `app/redis_queue.py`)
2. Update dependencies: `pip install -r requirements.txt`
3. Run: `docker-compose up -d --build` (starts Redis + proxy)
4. Verify: `curl http://localhost:8888/health`

**Breaking Changes:**
- ❌ No more `asyncio.Queue` - all queue operations go through Redis
- ❌ Environment vars required: `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`
- ✅ Backward compatible: Same OpenAI API endpoints

### Testing Changes

```bash
# Rebuild and restart
docker-compose up -d --build

# View logs
docker logs -f llm-queue-proxy

# Run integration tests (recommended - tests queue integrity)
uv run python3 tests/test_integration.py

# Test health
curl http://localhost:8888/health

# Test model list
curl http://localhost:8888/v1/models

# Test chat completion
curl -X POST http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [{"role": "user", "content": "test"}]
  }'

# View metrics
curl http://localhost:8888/metrics | jq
```

### Monitoring & Debugging

**Check proxy logs:**
```bash
docker logs -f llm-queue-proxy
```

**View queue metrics:**
```bash
curl http://localhost:8888/metrics | jq
```

**Inspect queue status:**
```bash
curl http://localhost:8888/health | jq
```

**Check dead-letter queue:**
```bash
curl http://localhost:8888/queue/dlq | jq
```

**Clear dead-letter queue (admin):**
```bash
curl -X DELETE http://localhost:8888/queue/dlq | jq
```

**Check backend health manually:**
```bash
curl http://localhost:8080/health
```

**Inspect Docker socket access:**
```bash
docker exec llm-queue-proxy ls -la /var/run/docker.sock
```

**Direct Redis inspection:**
```bash
docker exec llm-queue-redis redis-cli -a redis_password
# Once in redis-cli:
> ZRANGE llm_requests 0 -1 WITHSCORES  # View pending jobs (sorted set)
> SMEMBERS llm_processing_ids          # View processing job IDs (set)
> HGETALL llm_jobs                     # View job metadata (hash)
> LRANGE llm_dlq 0 -1                  # View dead-letter jobs (list)
```

## Dependencies

**Core Framework:**
- **FastAPI 0.119.0** - Modern async web framework
- **uvicorn 0.37.0** - ASGI server
- **pydantic 2.12.2** - Minimal data validation (most validation delegated to LiteLLM)

**LLM Routing & API:**
- **litellm 1.78.5** - LLM proxy/routing library for OpenAI-compatible API handling
- **aiohttp 3.13.1** - Async HTTP client for health checks

**Infrastructure:**
- **docker 7.1.0** - Docker SDK for container management
- **pyyaml 6.0.3** - YAML config parsing
- **redis 6.4.0** - Redis async client for queue management

**Observability (Langfuse via OpenTelemetry):**
- **langfuse 3.8.0** - LLM observability and tracing (SDK v3)
- **opentelemetry-api 1.38.0** - OpenTelemetry API for instrumentation
- **opentelemetry-sdk 1.38.0** - OpenTelemetry SDK implementation
- **opentelemetry-exporter-otlp-proto-grpc 1.38.0** - OTLP gRPC exporter for Langfuse

## Configuration

### Environment Variables (LLM Proxy)

**Core Settings:**
- `PYTHONUNBUFFERED=1` - Enable real-time log output
- `NUM_WORKERS` - Number of parallel queue workers (default: 2)

**Redis Queue (Hardcoded in docker-compose.yml):**
- `REDIS_HOST` - Redis hostname (default: localhost)
- `REDIS_PORT` - Redis port (default: 6379)
- `REDIS_PASSWORD` - Redis password (default: redis_password) ⚠️ **CHANGE FOR PRODUCTION**
- `REDIS_DB` - Redis database number (default: 0)

> **⚠️ SECURITY WARNING**: The default Redis password `redis_password` is hardcoded in `docker-compose.yml` for development convenience. **You MUST change this** if exposing Redis to any network or using in production. Update both the Redis container command and the `REDIS_PASSWORD` environment variable in docker-compose.yml.

**Langfuse Observability (SDK v3 with Manual Instrumentation):**
- `LANGFUSE_PUBLIC_KEY` - Langfuse project public key (format: `pk-lf-...`) - Required
- `LANGFUSE_SECRET_KEY` - Langfuse project secret key (format: `sk-lf-...`) - Required
- `LANGFUSE_OTEL_HOST` - Custom OTEL endpoint (default: `https://us.cloud.langfuse.com`)
  - US Cloud: `https://us.cloud.langfuse.com` (default)
  - EU Cloud: `https://cloud.langfuse.com`
  - Self-hosted: Your custom endpoint

**Notes:**
- If `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are not set, observability is automatically disabled
- Set Langfuse variables in your `.env` file to enable tracing (see `.env.example`)
- NUM_WORKERS can be overridden in `.env` or via environment variable

### Docker Compose Settings

**LLM Proxy:**
- `network_mode: host` - Required for accessing localhost backends
- `/var/run/docker.sock` mount - Required for Docker API access
- `restart: unless-stopped` - Auto-restart on crashes
- `depends_on: redis` - Ensures Redis starts first with health check

**Redis:**
- `image: redis:7-alpine` - Lightweight Redis server
- `--appendonly yes` - Persist queue data to disk
- `--requirepass redis_password` - Password protection
- Health check ensures proxy only starts when Redis is ready

### Port Usage

- **8888** - Proxy server (this service)
- **6379** - Redis queue (internal, localhost only)
- **8080** - Backend LLM servers (gpt-oss, qwen3)
- **8081** - Backend LLM servers (dolphin)

## Related Projects

- **CACHYOS_SETUP.md** - Backend model configurations and Docker commands
- **amd-strix-halo-toolboxes** - Docker images with ROCm 7 RC support
- **llama.cpp** - Backend inference engine
- **benches/radv-vs-rocm7/** - Performance benchmarks showing ROCm 2x faster PP at large contexts

## Common Tasks

### Update Dependencies

```bash
# Edit requirements.txt
vim requirements.txt

# Rebuild
docker-compose up -d --build
```

### Change Queue Size

```bash
# Edit config.yml
vim config.yml

# Restart (no rebuild needed)
docker-compose restart
```

### Add OpenAI Feature Support

When adding new OpenAI endpoints (e.g., /v1/embeddings):

1. Add Pydantic models in `app/models.py`
2. Add endpoint in `app/main.py`
3. Add forwarding logic to backend
4. Update README.md with examples

## Production Considerations

**Current Design:** Production-ready with Redis queue (v2.0)

**Key Improvements (v2.0):**
- Redis queue for reliable job processing
- Automatic job acknowledgement and retry (up to 3x)
- Dead-letter queue for failed jobs
- Client cancellation detection
- Metrics endpoint for monitoring
- Stalled job recovery (auto-retry if worker crashes)
- JSON-serialized job payloads in Redis

**Recent Fixes (v2.0.1):**
- ✅ Fixed job acknowledgement bug (hash-based tracking instead of JSON string matching)
- ✅ Removed incorrect `EXPIRE` call on entire hash (TTL now via timestamp-based cleanup)
- ✅ Fixed orphaned job cleanup in `cleanup_stalled_jobs()`
- ✅ Improved error handling in `nack_job()` with explicit Redis connection errors
- ✅ Added constants for magic numbers (`JOB_PROCESSING_TIMEOUT_SECONDS = 300`, `DEFAULT_MAX_RETRIES = 3`)
- ✅ Comprehensive integration tests in `tests/test_integration.py`

**v2.1 - Parallel Workers (Current):**
- ✅ Split `processing_lock` into `model_switch_lock` (Docker ops) and `metrics_lock` (counters)
- ✅ Multiple parallel queue workers (configurable via `NUM_WORKERS`, default: 2)
- ✅ ~2x speedup for concurrent requests to same model
- ✅ Model switching still serialized for safety (no GPU conflicts)
- ✅ Worker ID in logs for debugging (`[Worker 0]`, `[Worker 1]`)
- ✅ New metrics: `active_workers`, `peak_concurrent_requests`, `num_workers`
- ✅ Client disconnection detection (checks every 0.5s)
- ✅ Race-free metrics updates with async locks

**Future Enhancements:**
- Authentication/API keys
- Rate limiting per user
- Multi-GPU support (load multiple models)
- Model preloading (keep N models warm)
- Prometheus metrics export
- Request priority levels (currently uniform)
- Dynamic worker scaling based on queue depth

## Notes

- This proxy is **NOT** for multi-user production use
- Designed for **stability over latency**
- Model switching is **intentionally slow** (safe)
- Queue rejections (503) are **expected** under load
- Request persistence: Redis AOF ensures minimal data loss on crashes (v2.0+)
