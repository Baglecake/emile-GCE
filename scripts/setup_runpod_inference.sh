#!/bin/bash
# Setup script for émile-GCE inference on RunPod
# Run this on each pod after starting it

set -e

echo "=== émile-GCE RunPod Setup ==="

# Detect which pod we're on based on argument
POD_ROLE=${1:-"performer"}  # "performer" or "coach"

if [ "$POD_ROLE" == "performer" ]; then
    echo "Setting up PERFORMER (Llama + LoRA)"
    BASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
    LORA_REPO="baglecake/emile-gce-llama-lora"
    LORA_NAME="emile-llama"
    LORA_DIR="./llama_lora"
elif [ "$POD_ROLE" == "coach" ]; then
    echo "Setting up COACH (Mistral + LoRA)"
    BASE_MODEL="unsloth/Mistral-Nemo-Base-2407"
    LORA_REPO="baglecake/emile-gce-mistral-lora"
    LORA_NAME="emile-mistral"
    LORA_DIR="./mistral_lora"
else
    echo "Usage: $0 [performer|coach]"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -q vllm huggingface_hub openai

# Download LoRA adapter from HuggingFace
echo "Downloading LoRA adapter: $LORA_REPO"
huggingface-cli download $LORA_REPO --local-dir $LORA_DIR

# Check what we downloaded
echo "LoRA adapter contents:"
ls -la $LORA_DIR

# Start vLLM with LoRA
echo ""
echo "=== Starting vLLM server ==="
echo "Base model: $BASE_MODEL"
echo "LoRA adapter: $LORA_NAME -> $LORA_DIR"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model $BASE_MODEL \
    --enable-lora \
    --lora-modules "${LORA_NAME}=${LORA_DIR}" \
    --port 8000 \
    --host 0.0.0.0 \
    --max-lora-rank 16 \
    --gpu-memory-utilization 0.9
