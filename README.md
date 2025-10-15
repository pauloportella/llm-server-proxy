# LLM Queue Proxy

OpenAI-compatible proxy server with automatic model swapping and request queueing for single-GPU homelab deployments.

## Features

- **OpenAI API Compatible** - Works with Open WebUI, n8n, and any OpenAI client
- **Automatic Model Swapping** - Safely switches between models by stopping/starting Docker containers
- **Request Queueing** - FIFO queue with configurable capacity (default: 50 requests)
- **Single GPU Safety** - Only one model loaded at a time, prevents GPU memory crashes
- **Health Monitoring** - Polls backend health endpoints before forwarding requests
- **Zero Latency** - No request forwarding until model is fully loaded and ready

## Architecture

```
Client Request → Queue → Model Switcher → Backend LLM → Response
                   ↓
            [docker stop old]
            [docker start new]
            [poll /health]
```

**Key Design:**
- Only 1 concurrent GPU request at a time
- Requests queue while model is switching
- Health check polling prevents premature requests
- Stability > Latency (homelab optimized)

## Prerequisites

**1. Create Backend Containers**

The proxy manages existing containers. Create them first (stopped state is fine):

```bash
# GPT-OSS-120B
docker create --name gpt-oss-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
  llama-server -m /models/huggingface/gpt-oss-120b-Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf \
  --alias gpt-oss-120b -ngl 999 -c 65536 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --host 0.0.0.0 --port 8080 --jinja

# Qwen3-Coder-30B
docker create --name qwen-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
  llama-server -m /models/huggingface/hub/models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF/snapshots/7ce945e58ed3f09f9cf9c33a2122d86ac979b457/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  --alias qwen3-coder-30b -ngl 999 -c 262144 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --host 0.0.0.0 --port 8080 --jinja

# Dolphin-Mistral-24B (Q6, port 8080)
docker create --name dolphin-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
  llama-server -m /models/huggingface/dolphin-mistral-24b-venice/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-Q6_K_L.gguf \
  --alias dolphin-mistral-24b -ngl 999 -c 32768 -ub 32 -b 32 \
  --host 0.0.0.0 --port 8080 --jinja

# Dolphin-Mistral-24B-Fast (Q4, port 8080)
docker create --name dolphin-fast-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
  llama-server -m /models/huggingface/dolphin-mistral-24b-venice/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-Q4_K_L.gguf \
  --alias dolphin-mistral-24b-fast -ngl 999 -c 32768 -ub 32 -b 32 \
  --host 0.0.0.0 --port 8080 --jinja

# LFM2-8B-A1B (Q8_0, port 8080)
docker create --name lfm2-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
  llama-server -m /models/huggingface/lfm2-8b-a1b/LFM2-8B-A1B-Q8_0.gguf \
  --alias lfm2-8b -ngl 999 -c 32768 -b 4096 -ub 1024 \
  --cache-type-k f16 --cache-type-v f16 \
  --mlock --host 0.0.0.0 --port 8080 --jinja

# GPT-OSS-20B (Q8_0, port 8080)
docker create --name gpt-oss-20b-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
  llama-server -m /models/huggingface/gpt-oss-20b/gpt-oss-20b-Q8_0.gguf \
  --alias gpt-oss-20b -ngl 999 -c 131072 -b 2048 -ub 2048 \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja

# Qwen3-30B-A3B-Thinking-2507 (Q8_0, port 8080)
docker create --name qwen-thinking-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
  llama-server -m /models/huggingface/qwen3-30b-thinking/Qwen3-30B-A3B-Thinking-2507-Q8_0.gguf \
  --alias qwen3-30b-thinking -ngl 999 -c 131072 -b 2048 -ub 2048 \
  --reasoning-format deepseek \
  --host 0.0.0.0 --port 8080 --jinja

# AI21-Jamba-Reasoning-3B (F16, port 8080)
docker create --name jamba-reasoning-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
  llama-server -m /models/huggingface/jamba-reasoning-3b/jamba-reasoning-3b-F16.gguf \
  --alias jamba-reasoning-3b -ngl 999 -c 128000 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja

# Qwen3-30B-A3B-Instruct-2507 (Q8_0, port 8080)
docker create --name qwen-instruct-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
  llama-server -m /models/huggingface/qwen3-30b-instruct/Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf \
  --alias qwen3-30b-instruct -ngl 999 -c 131072 -b 2048 -ub 2048 \
  --host 0.0.0.0 --port 8080 --jinja

# GPT-OSS-20B-NEOPlus-Uncensored (Q8_0 Imatrix, port 8080)
docker create --name gpt-oss-20b-neoplus-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
  llama-server -m /models/huggingface/gpt-oss-20b-neoplus/OpenAI-20B-NEOPlus-Uncensored-Q8_0.gguf \
  --alias gpt-oss-20b-neoplus -ngl 999 -c 131072 -b 2048 -ub 2048 \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja

# GPT-OSS-20B-NEO-CODE-DI-Uncensored (Q8_0 DI-Matrix, port 8080)
docker create --name gpt-oss-20b-code-di-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v /mnt/ai_models:/models \
  kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
  llama-server -m /models/huggingface/gpt-oss-20b-code-di/OpenAI-20B-NEO-CODE-DI-Uncensored-Q8_0.gguf \
  --alias gpt-oss-20b-code-di -ngl 999 -c 131072 -b 2048 -ub 2048 \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja

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

models:
  your-model-name:
    container_name: your-container
    backend_url: http://localhost:8080/v1
    health_url: http://localhost:8080/health
    startup_timeout: 90
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
- **Model:** Choose from dropdown (gpt-oss-120b, gpt-oss-20b, gpt-oss-20b-neoplus, gpt-oss-20b-code-di, qwen3-coder-30b, qwen3-30b-thinking, qwen3-30b-instruct, dolphin-mistral-24b, dolphin-mistral-24b-fast, lfm2-8b, jamba-reasoning-3b, huihui-qwen3-vl-30b*)

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
- GPT-OSS-20B (Q8_0, 12.1GB, 128K context): ~20-30s
- AI21-Jamba-Reasoning-3B (F16, 6.4GB): ~5-10s
- LFM2-8B-A1B (Q8_0, 8.87GB): ~4-6s

**Recommendations:**
- Use same model for consecutive requests to avoid swapping
- Pre-load your most common model: `docker start gpt-oss-server`
- Monitor queue: `watch -n 1 'curl -s http://localhost:8888/health'`

## Project Structure

```
llm-server-proxy/
├── app/
│   ├── main.py       # FastAPI app, queue, Docker management
│   ├── config.py     # YAML config loader
│   └── models.py     # Pydantic models
├── config.yml        # Model definitions
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## License

MIT

## Related Documentation

- [CACHYOS_SETUP.md](../../dotfiles/CACHYOS_SETUP.md) - Backend model setup
- [AMD Strix Halo Toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)
