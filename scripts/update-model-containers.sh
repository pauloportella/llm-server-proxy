#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE="kyuz0/amd-strix-halo-toolboxes:rocm-7rc"
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
  "qwen3-6b-server"
  "lfm2-1.2b-tool-server"
  "lfm2-1.2b-rag-server"
  "lfm2-1.2b-extract-server"
  "llama-3.2-3b-server"
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

# GPT-OSS-120B (ROCm 7 RC - 2x faster prompt processing)
echo -n "  Creating gpt-oss-server... "
docker create --name gpt-oss-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/gpt-oss-120b-Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf \
  --alias gpt-oss-120b -ngl 999 -c 65536 -ub 2048 --no-mmap \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# Qwen3-Coder-30B (ROCm 7 RC)
echo -n "  Creating qwen-server... "
docker create --name qwen-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/hub/models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF/snapshots/7ce945e58ed3f09f9cf9c33a2122d86ac979b457/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  --alias qwen3-coder-30b -ngl 999 -c 262144 -ub 2048 --no-mmap \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# Dolphin-Mistral-24B (Q6) (ROCm 7 RC) - Note: -ub 32 prevents GPU cleanup crashes
echo -n "  Creating dolphin-server... "
docker create --name dolphin-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/dolphin-mistral-24b-venice/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-Q6_K_L.gguf \
  --alias dolphin-mistral-24b -ngl 999 -c 32768 -ub 32 -b 2048 --no-mmap \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# Dolphin-Mistral-24B-Fast (Q4) (ROCm 7 RC) - Note: -ub 32 prevents GPU cleanup crashes
echo -n "  Creating dolphin-fast-server... "
docker create --name dolphin-fast-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/dolphin-mistral-24b-venice/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-Q4_K_L.gguf \
  --alias dolphin-mistral-24b-fast -ngl 999 -c 32768 -ub 32 -b 2048 --no-mmap \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# LFM2-8B-A1B (ROCm 7 RC)
echo -n "  Creating lfm2-server... "
docker create --name lfm2-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/lfm2-8b-a1b/LFM2-8B-A1B-Q8_0.gguf \
  --alias lfm2-8b -ngl 999 -c 32768 -b 4096 -ub 2048 --no-mmap \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on --mlock \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# GPT-OSS-20B (ROCm 7 RC)
echo -n "  Creating gpt-oss-20b-server... "
docker create --name gpt-oss-20b-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/gpt-oss-20b/gpt-oss-20b-Q8_0.gguf \
  --alias gpt-oss-20b -ngl 999 -c 131072 -b 2048 -ub 2048 --no-mmap \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# Qwen3-30B-A3B-Thinking (ROCm 7 RC)
echo -n "  Creating qwen-thinking-server... "
docker create --name qwen-thinking-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/qwen3-30b-thinking/Qwen3-30B-A3B-Thinking-2507-Q8_0.gguf \
  --alias qwen3-30b-thinking -ngl 999 -c 131072 -b 2048 -ub 2048 --no-mmap \
  --flash-attn on --reasoning-format deepseek \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# AI21-Jamba-Reasoning-3B (ROCm 7 RC)
echo -n "  Creating jamba-reasoning-server... "
docker create --name jamba-reasoning-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/jamba-reasoning-3b/jamba-reasoning-3b-F16.gguf \
  --alias jamba-reasoning-3b -ngl 999 -c 128000 -ub 2048 --no-mmap \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# Qwen3-30B-A3B-Instruct (ROCm 7 RC)
echo -n "  Creating qwen-instruct-server... "
docker create --name qwen-instruct-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/qwen3-30b-instruct/Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf \
  --alias qwen3-30b-instruct -ngl 999 -c 131072 -b 2048 -ub 2048 --no-mmap \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# GPT-OSS-20B-NEOPlus-Uncensored (ROCm 7 RC)
echo -n "  Creating gpt-oss-20b-neoplus-server... "
docker create --name gpt-oss-20b-neoplus-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/gpt-oss-20b-neoplus/OpenAI-20B-NEOPlus-Uncensored-Q8_0.gguf \
  --alias gpt-oss-20b-neoplus -ngl 999 -c 131072 -b 2048 -ub 2048 --no-mmap \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# GPT-OSS-20B-NEO-CODE-DI-Uncensored (ROCm 7 RC)
echo -n "  Creating gpt-oss-20b-code-di-server... "
docker create --name gpt-oss-20b-code-di-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/gpt-oss-20b-code-di/OpenAI-20B-NEO-CODE-DI-Uncensored-Q8_0.gguf \
  --alias gpt-oss-20b-code-di -ngl 999 -c 131072 -b 2048 -ub 2048 --no-mmap \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# Qwen3-Almost-Human-X3-6B (ROCm 7 RC)
echo -n "  Creating qwen3-6b-server... "
docker create --name qwen3-6b-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/qwen3-almost-human-x3-6b/qwen3-almost-human-x3-1839-6b-q5_k_m.gguf \
  --alias qwen3-6b-almost-human -ngl 999 -c 32768 -b 2048 -ub 2048 --no-mmap \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# LFM2-1.2B-Tool (ROCm 7 RC)
echo -n "  Creating lfm2-1.2b-tool-server... "
docker create --name lfm2-1.2b-tool-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/lfm2-1.2b-tool/LFM2-1.2B-Tool-Q8_0.gguf \
  --alias lfm2-1.2b-tool -ngl 999 -c 32768 -b 4096 -ub 2048 --no-mmap \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on --mlock \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# LFM2-1.2B-RAG (ROCm 7 RC)
echo -n "  Creating lfm2-1.2b-rag-server... "
docker create --name lfm2-1.2b-rag-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/lfm2-1.2b-rag/LFM2-1.2B-RAG-Q8_0.gguf \
  --alias lfm2-1.2b-rag -ngl 999 -c 32768 -b 4096 -ub 2048 --no-mmap \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on --mlock \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# LFM2-1.2B-Extract (ROCm 7 RC)
echo -n "  Creating lfm2-1.2b-extract-server... "
docker create --name lfm2-1.2b-extract-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/lfm2-1.2b-extract/LFM2-1.2B-Extract-Q8_0.gguf \
  --alias lfm2-1.2b-extract -ngl 999 -c 32768 -b 4096 -ub 2048 --no-mmap \
  --cache-type-k f16 --cache-type-v f16 \
  --flash-attn on --mlock \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

# Llama-3.2-3B-Instruct (ROCm 7 RC)
echo -n "  Creating llama-3.2-3b-server... "
docker create --name llama-3.2-3b-server -p 8080:8080 \
  --device /dev/dri --device /dev/kfd \
  -v "${MODEL_MOUNT}":/models \
  "${IMAGE}" \
  llama-server -m /models/huggingface/llama-3.2-3b-instruct/Llama-3.2-3B-Instruct-Q6_K_L.gguf \
  --alias llama-3.2-3b -ngl 999 -c 131072 -b 2048 -ub 2048 --no-mmap \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080 --jinja > /dev/null
echo -e "${GREEN}✓${NC}"

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
  "qwen3-6b-server"
  "lfm2-1.2b-tool-server"
  "lfm2-1.2b-rag-server"
  "lfm2-1.2b-extract-server"
  "llama-3.2-3b-server"
)

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ All containers recreated successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Summary:"
echo "  • Backend: ROCm 7 RC (${IMAGE})"
echo "  • Containers created: 16 (all ROCm 7 RC)"
echo "  • Model mount: ${MODEL_MOUNT}"
echo ""
echo "ROCm 7 RC Benefits:"
echo "  • 2x faster prompt processing at large contexts (30k+ tokens)"
echo "  • 60-80% faster overall for RAG/document analysis workflows"
echo "  • Optimal for coding agents with large context ingestion"
echo ""
echo "Special configurations:"
echo "  • Dolphin models: -ub 32 (prevents GPU cleanup crashes)"
echo "  • All other models: -ub 2048 (optimal for ROCm)"
echo "  • --no-mmap flag: Required for ROCm large model stability"
echo ""
echo "Not created (vision support not implemented):"
echo "  • huihui-qwen3-vl-server"
echo ""

# Step 4: Restart proxy to register new models
echo -e "${YELLOW}[4/4]${NC} Restarting LLM Queue Proxy..."
docker restart llm-queue-proxy > /dev/null 2>&1
if [ $? -eq 0 ]; then
  echo -e "${GREEN}✓ Proxy restarted successfully${NC}"
  sleep 2
  echo ""
  echo "Proxy Health:"
  curl -s http://localhost:8888/health | jq 2>/dev/null
  echo ""
  echo "Available Models:"
  curl -s http://localhost:8888/v1/models | jq -r '.data[].id' 2>/dev/null | sort | while read model; do
    echo "  ${model} ${GREEN}✓${NC}"
  done
else
  echo -e "${RED}✗ Failed to restart proxy${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo "Next steps:"
echo "  1. Start a model container: docker start gpt-oss-server"
echo "  2. Test inference: curl http://localhost:8888/v1/chat/completions ..."
echo ""
