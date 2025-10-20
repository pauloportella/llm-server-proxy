# How to Run RADV vs ROCm 7 RC Benchmarks

This guide explains how to reproduce the benchmarks or test other models.

---

## Prerequisites

1. **LLM Queue Proxy running** with both RADV and ROCm model variants configured
2. **Raw Discord transcript** (included: `discord-transcript.txt`)
3. **Models available** in proxy's `config.yml`

---

## Quick Start

### Test Any Model

```bash
# 1. Create benchmark request
python3 << 'EOF'
import json

# Read raw Discord transcript
with open('benches/radv-vs-rocm7/discord-transcript.txt', 'r') as f:
    discord_content = f.read()

# Create request for RADV variant
radv_request = {
    "model": "YOUR-MODEL-NAME-RADV",
    "messages": [{
        "role": "user",
        "content": f"Analyze this raw Discord chat transcript from the Strix Halo Homelab #AI/llms channel. Summarize the 10 most important technical discussions and performance insights about running LLMs on AMD Strix Halo. Be concise.\n\n{discord_content}"
    }],
    "max_tokens": 1000,
    "temperature": 0.7
}

# Create request for ROCm variant
rocm_request = {
    "model": "YOUR-MODEL-NAME-ROCM",
    "messages": [{
        "role": "user",
        "content": f"Analyze this raw Discord chat transcript from the Strix Halo Homelab #AI/llms channel. Summarize the 10 most important technical discussions and performance insights about running LLMs on AMD Strix Halo. Be concise.\n\n{discord_content}"
    }],
    "max_tokens": 1000,
    "temperature": 0.7
}

# Save requests
with open('/tmp/radv_benchmark.json', 'w') as f:
    json.dump(radv_request, f)

with open('/tmp/rocm_benchmark.json', 'w') as f:
    json.dump(rocm_request, f)

print(f"✓ Benchmark requests created")
print(f"  • Context: {len(discord_content):,} chars (~30k tokens)")
EOF

# 2. Run RADV benchmark
curl -X POST http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @/tmp/radv_benchmark.json \
  2>&1 | tee /tmp/radv_result.txt

# 3. Run ROCm benchmark
curl -X POST http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @/tmp/rocm_benchmark.json \
  2>&1 | tee /tmp/rocm_result.txt

# 4. Extract and compare results
grep -o '"timings":{[^}]*}' /tmp/radv_result.txt | tail -1
echo "---"
grep -o '"timings":{[^}]*}' /tmp/rocm_result.txt | tail -1
```

---

## Model Name Mapping

### Available Models (config.yml)

**RADV Variants:**
- `gpt-oss-120b` - GPT-OSS-120B Q4_K_M
- `qwen3-30b-instruct` - Qwen3-30B-Instruct Q8_0
- `qwen3-coder-30b` - Qwen3-Coder-30B Q4_K_M
- `gpt-oss-20b` - GPT-OSS-20B Q8_0
- `dolphin-mistral-24b` - Dolphin-Mistral-24B Q6_K_L
- `dolphin-mistral-24b-fast` - Dolphin-Mistral-24B Q4_K_L
- `lfm2-8b` - LFM2-8B Q8_0

**ROCm 7 RC Variants:**
- `gpt-oss-120b-rocm` - GPT-OSS-120B Q4_K_M (ROCm)
- `qwen3-30b-instruct-rocm` - Qwen3-30B-Instruct Q8_0 (ROCm)
- `gpt-oss-20b-rocm` - GPT-OSS-20B Q8_0 (ROCm)

---

## Benchmark Scenarios

### 1. Small Context Test (1,573 tokens)

Use the curated knowledge base report (smaller, clean text):

```bash
# Read the knowledge base report
KB_CONTENT=$(cat /home/jrbaron/Dev/AI/datasets/discord/reports/2025-10-12_strix-halo-ai-llms.md)

# Create small context request
curl -X POST http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [{
      "role": "user",
      "content": "Analyze this Discord knowledge base about AMD Strix Halo LLM performance. List the 5 most important technical takeaways for optimizing local inference. Be concise.\n\n'"$KB_CONTENT"'"
    }],
    "max_tokens": 500,
    "temperature": 0.7
  }' 2>&1 | tee /tmp/small_radv.txt
```

**Expected context size:** ~1,573 tokens
**Expected time:** 10-20 seconds

### 2. Large Context Test (30k tokens)

Use the full raw Discord transcript (this benchmark):

```bash
# Use the included discord-transcript.txt
# Context size: ~29,861 tokens for gpt-oss, ~36,900 for qwen3
# Expected time: 80-260 seconds depending on model/backend
```

**Best for:**
- Testing PP performance at scale
- RAG/document processing benchmarks
- Codebase analysis simulation

### 3. Custom Context Test

Test with your own content:

```python
import json

with open('YOUR_LARGE_FILE.txt', 'r') as f:
    content = f.read()

request = {
    "model": "gpt-oss-120b",
    "messages": [{
        "role": "user",
        "content": f"Summarize the key points from this document:\n\n{content}"
    }],
    "max_tokens": 1000,
    "temperature": 0.7
}

with open('/tmp/custom_benchmark.json', 'w') as f:
    json.dump(request, f)
```

---

## Interpreting Results

### Timing Fields (JSON output)

```json
{
  "timings": {
    "prompt_n": 29861,              // Number of prompt tokens
    "prompt_ms": 109202.241,        // Time to process prompt (milliseconds)
    "prompt_per_second": 273.44,    // Prompt processing speed (t/s)
    "predicted_n": 1000,            // Number of generated tokens
    "predicted_ms": 25413.277,      // Time to generate tokens (milliseconds)
    "predicted_per_second": 39.35   // Token generation speed (t/s)
  }
}
```

### Key Metrics

**Prompt Processing Speed (PP):**
- **Higher is better**
- ROCm typically 2x faster at large contexts
- Most important for RAG/document analysis

**Token Generation Speed (TG):**
- **Higher is better**
- RADV typically 5-10% faster
- Most important for interactive chat

**Total Time:**
- `(prompt_ms + predicted_ms) / 1000` = seconds
- Overall workflow speed

---

## Analysis Script

Extract and compare results programmatically:

```bash
#!/bin/bash

# Extract timings from curl output
radv_pp=$(grep -o '"prompt_per_second":[^,]*' /tmp/radv_result.txt | tail -1 | cut -d: -f2)
radv_tg=$(grep -o '"predicted_per_second":[^}]*' /tmp/radv_result.txt | tail -1 | cut -d: -f2)

rocm_pp=$(grep -o '"prompt_per_second":[^,]*' /tmp/rocm_result.txt | tail -1 | cut -d: -f2)
rocm_tg=$(grep -o '"predicted_per_second":[^}]*' /tmp/rocm_result.txt | tail -1 | cut -d: -f2)

# Calculate speedup
echo "Prompt Processing:"
echo "  RADV: $radv_pp t/s"
echo "  ROCm: $rocm_pp t/s"
echo ""
echo "Token Generation:"
echo "  RADV: $radv_tg t/s"
echo "  ROCm: $rocm_tg t/s"
```

Or use Python:

```python
import json
import re

def extract_timings(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract last JSON response
    last_line = content.strip().split('\n')[-1]
    data = json.loads(last_line)

    return data['timings']

radv = extract_timings('/tmp/radv_result.txt')
rocm = extract_timings('/tmp/rocm_result.txt')

pp_speedup = (rocm['prompt_per_second'] - radv['prompt_per_second']) / radv['prompt_per_second'] * 100
tg_speedup = (rocm['predicted_per_second'] - radv['predicted_per_second']) / radv['predicted_per_second'] * 100

print(f"Prompt Processing: ROCm {pp_speedup:+.1f}% vs RADV")
print(f"Token Generation:  ROCm {tg_speedup:+.1f}% vs RADV")
```

---

## Tips for Accurate Benchmarking

### 1. Warm-up Run
Always do a warm-up request first to ensure containers are fully loaded:

```bash
curl -X POST http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-oss-120b", "messages": [{"role": "user", "content": "test"}], "max_tokens": 10}'
```

### 2. Sequential Testing
Don't run benchmarks in parallel - test RADV first, then ROCm (or vice versa).

### 3. Consistent Context
Use the **exact same prompt** for both variants to ensure fair comparison.

### 4. Multiple Runs
Run each benchmark 3 times and average results for stability:

```bash
for i in {1..3}; do
  echo "Run $i:"
  curl -X POST http://localhost:8888/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d @/tmp/radv_benchmark.json \
    2>&1 | grep prompt_per_second
done
```

### 5. Monitor System Load
Check that no other processes are consuming GPU/memory:

```bash
# Before benchmark
htop  # Check CPU/memory
docker stats  # Check container usage
```

---

## Troubleshooting

### Model Not Found
```json
{"error": {"message": "Model not found", "type": "invalid_request_error"}}
```

**Solution:** Check available models:
```bash
curl http://localhost:8888/v1/models | jq -r '.data[].id'
```

### Request Timeout
```
curl: (28) Operation timed out
```

**Solution:** Increase curl timeout:
```bash
curl --max-time 600 -X POST ...
```

### Container Not Running
```json
{"error": {"message": "Backend container not available"}}
```

**Solution:** Start the model container:
```bash
docker start gpt-oss-server-rocm
# Wait 2-3 minutes for model to load
curl http://localhost:8080/health
```

### Out of Memory
```
Error: Failed to allocate memory
```

**Solution:** Use smaller model or reduce context:
- Try `max_tokens: 500` instead of 1000
- Use Q4_K_M instead of Q8_0 quant
- Reduce context size

---

## Batch Testing Multiple Models

Test all available models automatically:

```bash
#!/bin/bash

MODELS=(
  "gpt-oss-120b:gpt-oss-120b-rocm"
  "qwen3-30b-instruct:qwen3-30b-instruct-rocm"
  "gpt-oss-20b:gpt-oss-20b-rocm"
)

for pair in "${MODELS[@]}"; do
  radv=${pair%:*}
  rocm=${pair#*:}

  echo "Testing $radv vs $rocm..."

  # Run benchmarks
  # ... (use curl commands from above)

  # Extract results
  # ... (use analysis script)
done
```

---

## Example Output

```
═══════════════════════════════════════════════════════════════════
RADV (gpt-oss-120b)
═══════════════════════════════════════════════════════════════════

PERFORMANCE:
• Prompt Processing: 109.2s (29861 tokens) = 273 t/s
• Token Generation:  25.4s (1000 tokens) = 39 t/s
• Total Time:        134.6s

═══════════════════════════════════════════════════════════════════
ROCm 7 RC (gpt-oss-120b-rocm)
═══════════════════════════════════════════════════════════════════

PERFORMANCE:
• Prompt Processing: 55.5s (29861 tokens) = 538 t/s
• Token Generation:  27.8s (1000 tokens) = 36 t/s
• Total Time:        83.3s

═══════════════════════════════════════════════════════════════════

📊 COMPARISON:
  • Prompt Processing: ROCm 97% FASTER (538 vs 273 t/s)
  • Token Generation:  RADV 9% FASTER (39 vs 36 t/s)
  • Overall:           ROCm 61% FASTER (83.3s vs 134.6s)
```

---

## Resources

- **Discord Report:** `/home/jrbaron/Dev/AI/datasets/discord/reports/2025-10-12_strix-halo-ai-llms.md`
- **Raw Transcript:** `benches/radv-vs-rocm7/discord-transcript.txt`
- **Results:** `benches/radv-vs-rocm7/RESULTS.md`
- **Container Script:** `scripts/update-model-containers.sh`
- **Proxy Config:** `config.yml`
