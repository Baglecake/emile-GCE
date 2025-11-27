#!/bin/bash
# Training data batch generation
# Runs 10 experiments per model (Llama + Mistral) in parallel

echo "Starting batch generation - 10 runs per model, 4 rounds each"
echo "=================================================="

for seed in 1 2 3 4 5 6 7 8 9 10; do
  echo "[Seed $seed] Starting both models..."

  python3 experiments/run_ces_experiment.py --provider vllm \
    --base-url "https://coaapc0tyag7h3-8000.proxy.runpod.net/v1" \
    --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --api-key "sk-1234" --rounds 4 --max-turns 12 \
    --experiment-id "batch_llama_s${seed}" \
    --real-ces --seed $seed --no-dual-llm -q &

  python3 experiments/run_ces_experiment.py --provider vllm \
    --base-url "https://9qrgc461yk73t4-8080.proxy.runpod.net/v1" \
    --model "mistralai/Mistral-Nemo-Instruct-2407" \
    --api-key "sk-1234" --rounds 4 --max-turns 12 \
    --experiment-id "batch_mistral_s${seed}" \
    --real-ces --seed $seed --no-dual-llm -q &

  wait
  echo "[Seed $seed] Complete!"
done

echo ""
echo "=================================================="
echo "Batch complete!"
echo ""
echo "Output directories:"
ls -d outputs/batch_* 2>/dev/null | wc -l
echo "experiments generated"
