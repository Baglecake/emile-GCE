# Experiment Execution Guide

## Pre-Flight Checklist

### 1. Verify GPU Endpoints Are Running

Before starting any experiment, test that both endpoints respond:

```bash
# Test 14B Performer endpoint
curl -s -X POST "https://9qrgc461yk73t4-8080.proxy.runpod.net/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{"model": "Qwen/Qwen2.5-14B-Instruct", "messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 10}'

# Test 7B Coach endpoint
curl -s -X POST "https://coaapc0tyag7h3-8000.proxy.runpod.net/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 10}'
```

**Expected**: Both return JSON with `choices[0].message.content` containing a response.

**If endpoints fail**: Check `GPU_info.txt` for current pod IDs and update URLs accordingly.

### 2. Current GPU Configuration (from GPU_info.txt)

| Role | Pod ID | Model | Port | URL Pattern |
|------|--------|-------|------|-------------|
| Performer (14B) | 9qrgc461yk73t4 | Qwen/Qwen2.5-14B-Instruct | 8080 | `https://{pod_id}-8080.proxy.runpod.net/v1` |
| Coach (7B) | coaapc0tyag7h3 | Qwen/Qwen2.5-7B-Instruct | 8000 | `https://{pod_id}-8000.proxy.runpod.net/v1` |

**API Key**: `sk-1234`

---

## Running Experiments

### TRUE Dual-LLM Experiment (Recommended)

Uses separate 14B Performer and 7B Coach on distinct GPUs:

```bash
# Create timestamped experiment ID
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_ID="experiments/${TIMESTAMP}_<description>"

python3 experiments/run_ces_experiment.py \
  --performer-url "https://9qrgc461yk73t4-8080.proxy.runpod.net/v1" \
  --performer-model "Qwen/Qwen2.5-14B-Instruct" \
  --coach-url "https://coaapc0tyag7h3-8000.proxy.runpod.net/v1" \
  --coach-model "Qwen/Qwen2.5-7B-Instruct" \
  --api-key "sk-1234" \
  --condition G \
  --seed 42 \
  --rounds 10 \
  --experiment-id "$EXP_ID"
```

### Single-LLM Experiment (Testing/Fallback)

Uses single endpoint for both performer and coach:

```bash
python3 experiments/run_ces_experiment.py \
  --provider vllm \
  --base-url "https://coaapc0tyag7h3-8000.proxy.runpod.net/v1" \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --api-key "sk-1234" \
  --condition G \
  --seed 42 \
  --rounds 10 \
  --experiment-id "experiments/<name>"
```

### WorldState Experiment

Enables identity-grounded environmental context injection. Agents experience shared world events differentially based on their 7D identity vectors:

```bash
python3 experiments/run_ces_experiment.py \
  --performer-url "https://9qrgc461yk73t4-8080.proxy.runpod.net/v1" \
  --performer-model "Qwen/Qwen2.5-14B-Instruct" \
  --coach-url "https://coaapc0tyag7h3-8000.proxy.runpod.net/v1" \
  --coach-model "Qwen/Qwen2.5-7B-Instruct" \
  --api-key "sk-1234" \
  --condition G \
  --seed 42 \
  --rounds 10 \
  --world-state \
  --experiment-id "worldstate_experiment"
```

The `--world-state` flag activates the WorldStateEngine, which:
- Injects world events with identity-specific salience, valence, and stakes
- Rotates discussion topics with position seeds based on agent profile
- Tracks per-agent frustration, fatigue, and entrenchment

---

## Output Directory Structure

```
outputs/
├── archive/                    # Old/completed experiments
└── experiments/                # Active experiments
    └── YYYYMMDD_HHMMSS_<description>/
        ├── meta.json           # Experiment metadata (models, condition, seed)
        ├── ces_profiles.json   # Agent CES profiles
        ├── policy_state.json   # Final policy state
        ├── semiotic_state_log.json
        └── round{N}_social_rl.json  # Per-round data with identity_states
```

### Naming Convention

Use timestamped directories with descriptive suffix:
- `20251126_143000_te_validation_truedual`
- `20251126_150000_mortality_test_seed42`

---

## Verifying Experiment Output

### Check 7D Vectors Are Present

```python
import json
with open('outputs/experiments/<exp>/round1_social_rl.json') as f:
    data = json.load(f)

for agent, state in data['identity_states'].items():
    vec = state.get('vector', {})
    print(f'{agent}: {len(vec)} dims -> {list(vec.keys())}')
```

**Expected**: 7 dimensions: `['engagement', 'institutional_faith', 'ideology', 'partisanship', 'sociogeographic', 'social_friction', 'tie_to_place']`

### Check TE Data (Rounds 9+)

```python
# TE ratio only computed after 8 rounds (min_len=8)
with open('outputs/experiments/<exp>/round9_social_rl.json') as f:
    data = json.load(f)

for agent, state in data['identity_states'].items():
    te = state.get('te_ratio', 1.0)
    print(f'{agent}: TE={te:.3f} ({"AUTHENTIC" if te > 0.5 else "CONFORMIST"})')
```

### Check meta.json for TRUE Dual-LLM

```bash
cat outputs/experiments/<exp>/meta.json | python3 -m json.tool
```

Look for:
- `"true_dual_llm": true` (or model string in older runs)
- `"performer_model"`: should show 14B model
- `"coach_model"`: should show 7B model

---

## Troubleshooting

### "Connection refused" or timeout
- GPU pod may have stopped - check RunPod dashboard
- Pod ID may have changed - update `GPU_info.txt` and URLs

### TE ratio = 1.0 for all agents
- Normal for rounds 1-8 (insufficient history)
- Check `history_length` in identity_states - should be >= 8 for real TE

### ValueError: operands could not be broadcast together
- Old 3D traces mixed with new 7D vectors
- Fixed in commit `1168316` - ensure code is updated

### meta.json shows wrong model info
- Older experiments don't capture performer/coach separately
- Fixed in commit `df110c1` - run new experiment to get proper metadata

---

## Quick Reference

### Key Files
- `GPU_info.txt` - Current GPU pod IDs and connection info
- `experiments/run_ces_experiment.py` - Main experiment runner
- `agents/identity_core/core.py` - IdentityCore implementation
- `social_rl/world_state.py` - WorldStateEngine for identity-grounded environmental context
- `data/identity/identity_weights_2021.v1.json` - 7D CES to identity mapping
- `data/identity/identity_drift_priors.v1.json` - Per-dimension drift priors for tau calibration
- `analysis/identity/drift_prior_loader.py` - Loads drift priors with group modifiers

### Key Parameters
- `--condition G` - Optimal architecture (dual-LLM + adaptive + challenge)
- `--rounds 10` - Minimum for meaningful TE data (need 8+ for real TE)
- `--seed N` - Random seed for reproducibility
- `--world-state` - Enable WorldState engine for differential event/topic injection

### Identity Dimensions (Canonical Order)
1. engagement
2. institutional_faith
3. ideology
4. partisanship
5. sociogeographic
6. social_friction
7. tie_to_place
