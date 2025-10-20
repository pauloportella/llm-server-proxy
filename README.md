# LLM Queue Proxy

OpenAI-compatible proxy server with automatic model swapping and request queueing for single-GPU homelab deployments.

## Features

- **OpenAI API Compatible** - Works with Open WebUI, n8n, and any OpenAI client
- **Automatic Model Swapping** - Safely switches between models by stopping/starting Docker containers
- **Parallel Request Processing** - Multiple workers process requests concurrently (~2x speedup for same model)
- **Reliable Redis Queue** - Persistent FIFO queue with job acknowledgement and retry
- **Single GPU Safety** - Only one model loaded at a time, prevents GPU memory crashes
- **Health Monitoring** - Polls backend health endpoints before forwarding requests
- **Client Cancellation Detection** - Automatically detects and skips cancelled requests
- **Zero Latency** - No request forwarding until model is fully loaded and ready

## Architecture

```
Client Request → Redis Queue → Worker Pool (2 workers) → Backend LLM → Response
                                    ↓
                           [model_switch_lock]
                           [docker stop old]
                           [docker start new]
                           [poll /health]
```

**Key Design (v2.1):**
- **Parallel Processing**: 2 workers process requests concurrently when model is loaded
- **Safe Model Switching**: model_switch_lock serializes Docker operations (no GPU conflicts)
- **Redis Persistence**: Queue survives restarts, failed jobs auto-retry
- **Client Detection**: Cancelled requests automatically skipped
- **Stability > Latency** (homelab optimized)

## Prerequisites

**1. Create Backend Containers**

The proxy manages existing containers. Create them first (stopped state is fine):

```bash
# GPT-OSS-120B (ROCm 7 RC - 2x faster prompt processing at large contexts)
docker create --name gpt-oss-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:rocm-7rc \
  llama-server -m /models/huggingface/gpt-oss-120b-Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf \
  --alias gpt-oss-120b -ngl 999 -c 65536 -ub 2048 --no-mmap \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja

# Qwen3-Coder-30B (ROCm 7 RC)
docker create --name qwen-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:rocm-7rc \
  llama-server -m /models/huggingface/hub/models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF/snapshots/7ce945e58ed3f09f9cf9c33a2122d86ac979b457/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  --alias qwen3-coder-30b -ngl 999 -c 262144 -ub 2048 --no-mmap \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja

# Dolphin-Mistral-24B (Q6, ROCm 7 RC - Note: -ub 32 prevents GPU cleanup crashes)
docker create --name dolphin-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:rocm-7rc \
  llama-server -m /models/huggingface/dolphin-mistral-24b-venice/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-Q6_K_L.gguf \
  --alias dolphin-mistral-24b -ngl 999 -c 32768 -ub 32 -b 2048 --no-mmap \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja

# Dolphin-Mistral-24B-Fast (Q4, ROCm 7 RC - Note: -ub 32 prevents GPU cleanup crashes)
docker create --name dolphin-fast-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:rocm-7rc \
  llama-server -m /models/huggingface/dolphin-mistral-24b-venice/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-Q4_K_L.gguf \
  --alias dolphin-mistral-24b-fast -ngl 999 -c 32768 -ub 32 -b 2048 --no-mmap \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja

# LFM2-8B-A1B (Q8_0, ROCm 7 RC)
docker create --name lfm2-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:rocm-7rc \
  llama-server -m /models/huggingface/lfm2-8b-a1b/LFM2-8B-A1B-Q8_0.gguf \
  --alias lfm2-8b -ngl 999 -c 32768 -b 4096 -ub 2048 --no-mmap \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on --mlock \
  --host 0.0.0.0 --port 8080 --jinja

# GPT-OSS-20B (Q8_0, ROCm 7 RC)
docker create --name gpt-oss-20b-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:rocm-7rc \
  llama-server -m /models/huggingface/gpt-oss-20b/gpt-oss-20b-Q8_0.gguf \
  --alias gpt-oss-20b -ngl 999 -c 131072 -b 2048 -ub 2048 --no-mmap \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja

# Qwen3-30B-A3B-Thinking-2507 (Q8_0, ROCm 7 RC)
docker create --name qwen-thinking-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:rocm-7rc \
  llama-server -m /models/huggingface/qwen3-30b-thinking/Qwen3-30B-A3B-Thinking-2507-Q8_0.gguf \
  --alias qwen3-30b-thinking -ngl 999 -c 131072 -b 2048 -ub 2048 --no-mmap \
  --flash-attn on --reasoning-format deepseek \
  --host 0.0.0.0 --port 8080 --jinja

# AI21-Jamba-Reasoning-3B (F16, ROCm 7 RC)
docker create --name jamba-reasoning-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:rocm-7rc \
  llama-server -m /models/huggingface/jamba-reasoning-3b/jamba-reasoning-3b-F16.gguf \
  --alias jamba-reasoning-3b -ngl 999 -c 128000 -ub 2048 --no-mmap \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja

# Qwen3-30B-A3B-Instruct-2507 (Q8_0, ROCm 7 RC)
docker create --name qwen-instruct-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:rocm-7rc \
  llama-server -m /models/huggingface/qwen3-30b-instruct/Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf \
  --alias qwen3-30b-instruct -ngl 999 -c 131072 -b 2048 -ub 2048 --no-mmap \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja

# GPT-OSS-20B-NEOPlus-Uncensored (Q8_0 Imatrix, ROCm 7 RC)
docker create --name gpt-oss-20b-neoplus-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:rocm-7rc \
  llama-server -m /models/huggingface/gpt-oss-20b-neoplus/OpenAI-20B-NEOPlus-Uncensored-Q8_0.gguf \
  --alias gpt-oss-20b-neoplus -ngl 999 -c 131072 -b 2048 -ub 2048 --no-mmap \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja

# GPT-OSS-20B-NEO-CODE-DI-Uncensored (Q8_0 DI-Matrix, ROCm 7 RC)
docker create --name gpt-oss-20b-code-di-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:rocm-7rc \
  llama-server -m /models/huggingface/gpt-oss-20b-code-di/OpenAI-20B-NEO-CODE-DI-Uncensored-Q8_0.gguf \
  --alias gpt-oss-20b-code-di -ngl 999 -c 131072 -b 2048 -ub 2048 --no-mmap \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja

# Qwen3-Almost-Human-X3-6B (Q5_K_M, ROCm 7 RC)
docker create --name qwen3-6b-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:rocm-7rc \
  llama-server -m /models/huggingface/qwen3-almost-human-x3-6b/qwen3-almost-human-x3-1839-6b-q5_k_m.gguf \
  --alias qwen3-6b-almost-human -ngl 999 -c 32768 -b 2048 -ub 2048 --no-mmap \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja

# LFM2-1.2B-Tool (Q8_0, ROCm 7 RC - Tool-calling specialist)
docker create --name lfm2-1.2b-tool-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:rocm-7rc \
  llama-server -m /models/huggingface/lfm2-1.2b-tool/LFM2-1.2B-Tool-Q8_0.gguf \
  --alias lfm2-1.2b-tool -ngl 999 -c 32768 -b 4096 -ub 2048 --no-mmap \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on --mlock \
  --host 0.0.0.0 --port 8080 --jinja

# LFM2-1.2B-RAG (Q8_0, ROCm 7 RC - RAG specialist, temperature=0.0 recommended)
docker create --name lfm2-1.2b-rag-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:rocm-7rc \
  llama-server -m /models/huggingface/lfm2-1.2b-rag/LFM2-1.2B-RAG-Q8_0.gguf \
  --alias lfm2-1.2b-rag -ngl 999 -c 32768 -b 4096 -ub 2048 --no-mmap \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on --mlock \
  --host 0.0.0.0 --port 8080 --jinja

# LFM2-1.2B-Extract (Q8_0, ROCm 7 RC - Information extraction, JSON/XML output)
docker create --name lfm2-1.2b-extract-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:rocm-7rc \
  llama-server -m /models/huggingface/lfm2-1.2b-extract/LFM2-1.2B-Extract-Q8_0.gguf \
  --alias lfm2-1.2b-extract -ngl 999 -c 32768 -b 4096 -ub 2048 --no-mmap \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on --mlock \
  --host 0.0.0.0 --port 8080 --jinja

# Llama-3.2-3B-Instruct (Q6_K_L, ROCm 7 RC - 128K context, edge-optimized)
docker create --name llama-3.2-3b-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:rocm-7rc \
  llama-server -m /models/huggingface/llama-3.2-3b-instruct/Llama-3.2-3B-Instruct-Q6_K_L.gguf \
  --alias llama-3.2-3b -ngl 999 -c 131072 -b 2048 -ub 2048 --no-mmap \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja

# TODO: LFM2-VL-1.6B Vision Model (NOT YET CONFIGURED)
# Downloaded: /mnt/ai_models/huggingface/lfm2-vl-1.6b/LFM2-VL-1.6B-Q8_0.gguf (1.16GB)
# Backend: kyuz0/amd-strix-halo-toolboxes:vulkan-radv (NOT rocm-7rc - vision needs Vulkan)
# Key differences: Single-file model (NO --mmproj needed), 32K context, uses libmtmd
# Proxy limitation: Vision support NOT YET IMPLEMENTED - requires API updates for image input
# See research output above for full configuration when ready to implement

# Huihui-Qwen3-VL-30B (Q8_0, Vision-Language, port 8080) - EXPERIMENTAL
# ⚠️ WARNING: Vision support NOT YET IMPLEMENTED in proxy
docker create --name huihui-qwen3-vl-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
  llama-server -m /models/huggingface/huihui-qwen3-vl-30b/GGUF/ggml-model-q8_0.gguf \
  --mmproj /models/huggingface/huihui-qwen3-vl-30b/GGUF/mmproj-ggml-model-f16.gguf \
  --alias huihui-qwen3-vl-30b -ngl 999 -c 4096 \
  --host 0.0.0.0 --port 8080 --jinja
```

**2. Firewall Configuration**

```bash
# Open proxy port on Netbird interface
ufw allow in on wt0 to any port 8888 proto tcp comment 'LLM Queue Proxy'
```

## Installation

**1. Build and Start**

```bash
cd ~/Dev/AI/llm-server-proxy
docker-compose up -d --build
```

**2. View Logs**

```bash
docker logs -f llm-queue-proxy
```

**3. Check Health**

```bash
curl http://localhost:8888/health
```

## Configuration

Edit `config.yml` to add/modify models:

```yaml
queue_size: 50
request_timeout: 600
num_workers: 2  # Number of parallel queue workers

models:
  your-model-name:
    container_name: your-container
    backend_url: http://localhost:8080/v1
    health_url: http://localhost:8080/health
    startup_timeout: 90
```

**Environment Variables:**
- `NUM_WORKERS` - Override worker count (default: 2)

```bash
# Run with 4 workers for higher concurrency
NUM_WORKERS=4 docker-compose up -d

# Run with 1 worker for backward compatibility
NUM_WORKERS=1 docker-compose up -d
```

After changes, restart the proxy:

```bash
docker-compose restart
```

## Usage

**API Endpoint:** `http://100.78.198.217:8888`

### List Available Models

```bash
curl http://100.78.198.217:8888/v1/models
```

Response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-oss-120b",
      "object": "model",
      "created": 1697000000,
      "owned_by": "local"
    },
    {
      "id": "gpt-oss-20b",
      "object": "model",
      "created": 1697000000,
      "owned_by": "local"
    },
    {
      "id": "qwen3-coder-30b",
      "object": "model",
      "created": 1697000000,
      "owned_by": "local"
    },
    {
      "id": "dolphin-mistral-24b",
      "object": "model",
      "created": 1697000000,
      "owned_by": "local"
    },
    {
      "id": "dolphin-mistral-24b-fast",
      "object": "model",
      "created": 1697000000,
      "owned_by": "local"
    },
    {
      "id": "lfm2-8b",
      "object": "model",
      "created": 1697000000,
      "owned_by": "local"
    },
    {
      "id": "qwen3-30b-thinking",
      "object": "model",
      "created": 1697000000,
      "owned_by": "local"
    },
    {
      "id": "jamba-reasoning-3b",
      "object": "model",
      "created": 1697000000,
      "owned_by": "local"
    },
    {
      "id": "qwen3-30b-instruct",
      "object": "model",
      "created": 1697000000,
      "owned_by": "local"
    },
    {
      "id": "gpt-oss-20b-neoplus",
      "object": "model",
      "created": 1697000000,
      "owned_by": "local"
    },
    {
      "id": "gpt-oss-20b-code-di",
      "object": "model",
      "created": 1697000000,
      "owned_by": "local"
    },
    {
      "id": "huihui-qwen3-vl-30b",
      "object": "model",
      "created": 1697000000,
      "owned_by": "local"
    },
    {
      "id": "qwen3-6b-almost-human",
      "object": "model",
      "created": 1697000000,
      "owned_by": "local"
    },
    {
      "id": "lfm2-1.2b-tool",
      "object": "model",
      "created": 1697000000,
      "owned_by": "local"
    },
    {
      "id": "lfm2-1.2b-rag",
      "object": "model",
      "created": 1697000000,
      "owned_by": "local"
    },
    {
      "id": "lfm2-1.2b-extract",
      "object": "model",
      "created": 1697000000,
      "owned_by": "local"
    },
    {
      "id": "llama-3.2-3b",
      "object": "model",
      "created": 1697000000,
      "owned_by": "local"
    }
  ]
}
```

### Chat Completion

```bash
curl -X POST http://100.78.198.217:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "temperature": 0.7
  }'
```

**Model Switching Example:**

```bash
# First request uses GPT-OSS-120B (cold start: ~180s)
curl ... -d '{"model": "gpt-oss-120b", ...}'

# Second request also uses GPT-OSS-120B (instant, already loaded)
curl ... -d '{"model": "gpt-oss-120b", ...}'

# Third request switches to Dolphin-24B (wait ~90s for swap)
curl ... -d '{"model": "dolphin-mistral-24b", ...}'
```

### Health Check

```bash
curl http://100.78.198.217:8888/health
```

Response:
```json
{
  "status": "healthy",
  "current_model": "gpt-oss-120b",
  "queue_size": 2,
  "queue_capacity": 50,
  "message": "Proxy is operational"
}
```

### Metrics

```bash
curl http://100.78.198.217:8888/metrics
```

Response:
```json
{
  "uptime_seconds": 3600,
  "total_requests": 150,
  "successful_requests": 148,
  "failed_requests": 0,
  "cancelled_requests": 2,
  "active_workers": 1,
  "peak_concurrent_requests": 2,
  "num_workers": 2,
  "success_rate": 0.987,
  "queue_stats": {
    "pending_jobs": 0,
    "processing_jobs": 1,
    "dead_letter_queue": 0,
    "total_jobs": 1
  },
  "current_model": "qwen3-30b-instruct",
  "pending_futures": 3
}
```

## Integration Examples

### Open WebUI

1. Go to Settings → Connections
2. Add OpenAI Connection:
   - **API URL:** `http://100.78.198.217:8888/v1`
   - **API Key:** (leave empty or use any string)
3. Models will appear automatically

### n8n

Use the OpenAI node with:
- **Base URL:** `http://100.78.198.217:8888/v1`
- **Model:** Choose from dropdown (gpt-oss-120b, gpt-oss-20b, gpt-oss-20b-neoplus, gpt-oss-20b-code-di, qwen3-coder-30b, qwen3-30b-thinking, qwen3-30b-instruct, qwen3-6b-almost-human, dolphin-mistral-24b, dolphin-mistral-24b-fast, lfm2-8b, jamba-reasoning-3b, huihui-qwen3-vl-30b*)

*Note: huihui-qwen3-vl-30b is a vision model - image support not yet implemented in proxy

### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://100.78.198.217:8888/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Troubleshooting

### Container not found error

**Problem:** `Container not found: gpt-oss-server`

**Solution:** Create the container first using docker create commands above.

### Model not loading

**Problem:** Request times out after 180s

**Solutions:**
1. Check backend container logs: `docker logs gpt-oss-server`
2. Verify GPU access: `docker exec gpt-oss-server vulkaninfo | grep deviceName`
3. Increase `startup_timeout` in config.yml

### Queue full

**Problem:** `Queue full (50 requests). Try again later.`

**Solution:** Increase `queue_size` in config.yml or wait for pending requests to complete.

### Model switching stuck

**Problem:** Model switch takes longer than expected

**Checks:**
1. Verify old container stopped: `docker ps -a`
2. Check proxy logs: `docker logs -f llm-queue-proxy`
3. Manually test health: `curl http://localhost:8080/health`

## Performance Notes

**Startup Times (from CACHYOS_SETUP.md):**
- GPT-OSS-120B (Q4_K_M, 63GB): ~180s
- Qwen3-Coder-30B (Q4_K_M, 19GB, 262K context): ~120s
- Dolphin-Mistral-24B (Q6, 19.67GB): ~90s
- Dolphin-Mistral-24B-Fast (Q4, 14GB): ~60s
- Huihui-Qwen3-VL-30B (Q8_0, 33.6GB, vision model): ~30-45s
- Qwen3-30B-A3B-Thinking (Q8_0, 32.5GB, reasoning): ~30-45s
- Qwen3-30B-A3B-Instruct (Q8_0, 32.5GB, general): ~30-45s
- GPT-OSS-20B-NEOPlus-Uncensored (Q8_0, 22.1GB, uncensored): ~20-30s
- GPT-OSS-20B-NEO-CODE-DI-Uncensored (Q8_0, 22.1GB, code-focused): ~20-30s
- Qwen3-Almost-Human-X3-6B (Q5_K_M, 4.0GB): ~20-40s
- GPT-OSS-20B (Q8_0, 12.1GB, 128K context): ~20-30s
- AI21-Jamba-Reasoning-3B (F16, 6.4GB): ~5-10s
- Llama-3.2-3B-Instruct (Q6_K_L, 2.74GB, 128K context): ~5-8s
- LFM2-8B-A1B (Q8_0, 8.87GB): ~4-6s
- LFM2-1.2B-Tool (Q8_0, 1.2GB, tool-calling): ~2-5s
- LFM2-1.2B-RAG (Q8_0, 1.2GB, RAG specialist): ~2-5s
- LFM2-1.2B-Extract (Q8_0, 1.2GB, extraction): ~2-5s

**Recommendations:**
- Use same model for consecutive requests to avoid swapping
- Pre-load your most common model: `docker start gpt-oss-server`
- Monitor queue: `watch -n 1 'curl -s http://localhost:8888/health'`
- Send concurrent requests to maximize parallel processing
- Check metrics to tune worker count: `curl http://localhost:8888/metrics | jq`

## Project Structure

```
llm-server-proxy/
├── app/
│   ├── main.py         # FastAPI app, workers, Docker management
│   ├── redis_queue.py  # Redis-based reliable queue
│   ├── config.py       # YAML config loader
│   └── models.py       # Pydantic models
├── tests/
│   └── test_integration.py  # Integration tests
├── config.yml          # Model definitions
├── docker-compose.yml  # Proxy + Redis services
├── Dockerfile
├── requirements.txt
├── CLAUDE.md           # Technical documentation
└── README.md
```

## License

MIT

## Related Documentation

- [CACHYOS_SETUP.md](../../dotfiles/CACHYOS_SETUP.md) - Backend model setup
- [AMD Strix Halo Toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)
