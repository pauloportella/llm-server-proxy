#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE="kyuz0/amd-strix-halo-toolboxes:vulkan-radv"
MODEL_MOUNT="/mnt/ai_models"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}LLM Model Containers Update Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Pull latest image
echo -e "${YELLOW}[1/3]${NC} Pulling latest Docker image: ${IMAGE}"
docker pull "${IMAGE}"
echo -e "${GREEN}✓ Image pulled successfully${NC}"
echo ""

# Step 2: Stop and remove existing containers
echo -e "${YELLOW}[2/3]${NC} Stopping and removing existing model containers..."

CONTAINERS=(
  "gpt-oss-server"
  "qwen-server"
  "dolphin-server"
  "dolphin-fast-server"
  "lfm2-server"
  "gpt-oss-20b-server"
  "qwen-thinking-server"
  "jamba-reasoning-server"
  "qwen-instruct-server"
  "gpt-oss-20b-neoplus-server"
  "gpt-oss-20b-code-di-server"
)

for container in "${CONTAINERS[@]}"; do
  if docker ps -a --format '{{.Names}}' | grep -q "^${container}$"; then
    echo -n "  Stopping ${container}... "
    docker stop "${container}" 2>/dev/null || true
    echo -n "Removing... "
    docker rm "${container}" 2>/dev/null || true
    echo -e "${GREEN}✓${NC}"
  fi
done
echo ""

# Step 3: Create new containers
echo -e "${YELLOW}[3/3]${NC} Creating new model containers..."
echo ""

# GPT-OSS-120B
echo -n "  Creating gpt-oss-server... "
docker create --name gpt-oss-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/gpt-oss-120b-Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf \
  --alias gpt-oss-120b -ngl 999 -c 65536 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# Qwen3-Coder-30B
echo -n "  Creating qwen-server... "
docker create --name qwen-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/hub/models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF/snapshots/7ce945e58ed3f09f9cf9c33a2122d86ac979b457/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  --alias qwen3-coder-30b -ngl 999 -c 262144 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# Dolphin-Mistral-24B (Q6)
echo -n "  Creating dolphin-server... "
docker create --name dolphin-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/dolphin-mistral-24b-venice/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-Q6_K_L.gguf \
  --alias dolphin-mistral-24b -ngl 999 -c 32768 -ub 32 -b 32 \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# Dolphin-Mistral-24B-Fast (Q4)
echo -n "  Creating dolphin-fast-server... "
docker create --name dolphin-fast-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/dolphin-mistral-24b-venice/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-Q4_K_L.gguf \
  --alias dolphin-mistral-24b-fast -ngl 999 -c 32768 -ub 32 -b 32 \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# LFM2-8B-A1B
echo -n "  Creating lfm2-server... "
docker create --name lfm2-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/lfm2-8b-a1b/LFM2-8B-A1B-Q8_0.gguf \
  --alias lfm2-8b -ngl 999 -c 32768 -b 4096 -ub 1024 \
  --cache-type-k f16 --cache-type-v f16 \
  --mlock --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# GPT-OSS-20B
echo -n "  Creating gpt-oss-20b-server... "
docker create --name gpt-oss-20b-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/gpt-oss-20b/gpt-oss-20b-Q8_0.gguf \
  --alias gpt-oss-20b -ngl 999 -c 131072 -b 2048 -ub 2048 \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# Qwen3-30B-A3B-Thinking
echo -n "  Creating qwen-thinking-server... "
docker create --name qwen-thinking-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/qwen3-30b-thinking/Qwen3-30B-A3B-Thinking-2507-Q8_0.gguf \
  --alias qwen3-30b-thinking -ngl 999 -c 131072 -b 2048 -ub 2048 \
  --reasoning-format deepseek \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# AI21-Jamba-Reasoning-3B
echo -n "  Creating jamba-reasoning-server... "
docker create --name jamba-reasoning-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/jamba-reasoning-3b/jamba-reasoning-3b-F16.gguf \
  --alias jamba-reasoning-3b -ngl 999 -c 128000 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# Qwen3-30B-A3B-Instruct
echo -n "  Creating qwen-instruct-server... "
docker create --name qwen-instruct-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/qwen3-30b-instruct/Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf \
  --alias qwen3-30b-instruct -ngl 999 -c 131072 -b 2048 -ub 2048 \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# GPT-OSS-20B-NEOPlus-Uncensored
echo -n "  Creating gpt-oss-20b-neoplus-server... "
docker create --name gpt-oss-20b-neoplus-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/gpt-oss-20b-neoplus/OpenAI-20B-NEOPlus-Uncensored-Q8_0.gguf \
  --alias gpt-oss-20b-neoplus -ngl 999 -c 131072 -b 2048 -ub 2048 \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# GPT-OSS-20B-NEO-CODE-DI-Uncensored
echo -n "  Creating gpt-oss-20b-code-di-server... "
docker create --name gpt-oss-20b-code-di-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/gpt-oss-20b-code-di/OpenAI-20B-NEO-CODE-DI-Uncensored-Q8_0.gguf \
  --alias gpt-oss-20b-code-di -ngl 999 -c 131072 -b 2048 -ub 2048 \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ All containers recreated successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Summary:"
echo "  • Image updated: ${IMAGE}"
echo "  • Containers created: 11"
echo "  • Model mount: ${MODEL_MOUNT}"
echo ""
echo "Not created (vision support not implemented):"
echo "  • huihui-qwen3-vl-server"
echo ""
echo "Next steps:"
echo "  1. Start a model container: docker start gpt-oss-server"
echo "  2. Restart the proxy: docker-compose restart llm-queue-proxy"
echo "  3. Check health: curl http://localhost:8888/health"
echo ""
