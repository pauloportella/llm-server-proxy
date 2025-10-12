# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**LLM Queue Proxy** - OpenAI-compatible proxy server with automatic model swapping for single-GPU homelab deployments.

**Key Problem Solved:** Prevents GPU memory crashes by ensuring only ONE large language model is loaded at a time, while maintaining OpenAI API compatibility for tools like Open WebUI and n8n.

## Architecture

### Core Components

1. **app/main.py** - FastAPI application with:
   - Request queue (asyncio.Queue, FIFO)
   - Model switcher (Docker SDK start/stop)
   - Health polling (waits for backend readiness)
   - OpenAI-compatible endpoints

2. **app/config.py** - YAML configuration loader
   - Loads model definitions from config.yml
   - Validates model configurations
   - Provides model lookup

3. **app/models.py** - Pydantic models for validation
   - OpenAI request/response formats
   - Model configuration schema
   - Health check responses

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

**Why in-memory queue instead of Redis?**
- Homelab deployment (single instance)
- Simplicity over distributed systems complexity
- Request loss on restart is acceptable

**Why health polling?**
- llama-server takes 90-180s to load models
- Prevents forwarding requests to unready backends
- Graceful handling of startup delays

**Why global processing lock?**
- Guarantees only 1 concurrent GPU request
- Prevents race conditions in model switching
- Critical for single-GPU stability

## Development Workflow

### Adding a New Model

1. Create the backend container:
   ```bash
   docker create --name your-model-server -p 8080:8080 \
     --device /dev/dri --device /dev/kfd \
     -v /mnt/ai_models:/models \
     kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
     llama-server -m /models/your-model.gguf \
     --alias your-model -ngl 999 -c 32768 \
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

### Testing Changes

```bash
# Rebuild and restart
docker-compose up -d --build

# View logs
docker logs -f llm-queue-proxy

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
```

### Debugging

**Check proxy logs:**
```bash
docker logs -f llm-queue-proxy
```

**Check backend health manually:**
```bash
curl http://localhost:8080/health
```

**Inspect Docker socket access:**
```bash
docker exec llm-queue-proxy ls -la /var/run/docker.sock
```

**Check queue status:**
```bash
curl http://localhost:8888/health | jq
```

## Dependencies

- **FastAPI 0.115.0** - Modern async web framework
- **uvicorn 0.32.0** - ASGI server
- **aiohttp 3.11.0** - Async HTTP client for backend forwarding
- **pyyaml 6.0.2** - YAML config parsing
- **docker 7.1.0** - Docker SDK for container management
- **pydantic 2.9.2** - Data validation and OpenAI schema

## Configuration

### Environment Variables

- `PYTHONUNBUFFERED=1` - Enable real-time log output

### Docker Compose Settings

- `network_mode: host` - Required for accessing localhost backends
- `/var/run/docker.sock` mount - Required for Docker API access
- `restart: unless-stopped` - Auto-restart on crashes

### Port Usage

- **8888** - Proxy server (this service)
- **8080** - Backend LLM servers (gpt-oss, qwen3)
- **8081** - Backend LLM servers (dolphin)

## Related Projects

- **CACHYOS_SETUP.md** - Backend model configurations and Docker commands
- **amd-strix-halo-toolboxes** - Docker images with Vulkan RADV support
- **llama.cpp** - Backend inference engine

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

**Current Design:** Optimized for single-user homelab

**Future Enhancements:**
- Authentication/API keys
- Rate limiting per user
- Multi-GPU support (load multiple models)
- Model preloading (keep N models warm)
- Streaming responses (SSE)
- Request priority queue
- Metrics/monitoring (Prometheus)

## Notes

- This proxy is **NOT** for multi-user production use
- Designed for **stability over latency**
- Model switching is **intentionally slow** (safe)
- Queue rejections (503) are **expected** under load
- Request loss on restart is **acceptable** (in-memory queue)
