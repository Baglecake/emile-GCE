# Experiment Outputs

This directory contains outputs from Social RL experiments. Each timestamped subdirectory represents a complete experimental run.

## Directory Structure

```
outputs/
├── social_rl_YYYY-MM-DD_HHMMSS/
│   ├── round1_social_rl.json
│   ├── round2_social_rl.json
│   ├── round3_social_rl.json
│   └── policy_state.json
└── README.md
```

## Output Format

### Round Files (`roundN_social_rl.json`)

Each round produces a structured JSON file:

```json
{
  "round_number": 1,
  "round_config": {
    "scenario": "...",
    "rules": "...",
    "tasks": "..."
  },
  "messages": [
    {
      "agent_id": "Worker+Alice",
      "role": "assistant",
      "content": "...",
      "turn_index": 0,
      "context_frame": {
        "concept_a_manifestation": "...",
        "concept_b_manifestation": "...",
        "prar_cue": "..."
      }
    }
  ],
  "feedback": {
    "Worker+Alice": {
      "engagement": 0.85,
      "alignment": 0.72,
      "contribution_value": 0.68
    }
  },
  "policy_adaptations": [...],
  "duration_seconds": 45.2
}
```

### Policy State (`policy_state.json`)

Captures the final policy configuration after all rounds:

```json
{
  "framework_option": "A",
  "policies": [
    {
      "role": "Worker",
      "cues_active": ["reflect_on_alienation", "connect_to_labor"],
      "feedback_snapshot": {...}
    }
  ],
  "timestamp": "2025-11-23T04:38:03",
  "source_run_id": "social_rl_2025-11-23_043136"
}
```

## Naming Convention

Directories follow the pattern: `social_rl_YYYY-MM-DD_HHMMSS`

- `YYYY-MM-DD`: Date of execution
- `HHMMSS`: Time of execution (24-hour format)

This ensures chronological sorting and unique identification.

## Relationship to Other Outputs

| Output Type | Location | Description |
|-------------|----------|-------------|
| PRAR outputs | `prar/outputs/` | Raw PRAR workflow artifacts (state.json, document.txt) |
| Social RL outputs | `outputs/` | Simulation transcripts and feedback data |
| Test outputs | `local_rcm/output/` | Local test artifacts (gitignored) |

Social RL experiments typically consume a PRAR output (state.json) as input and produce simulation transcripts as output. The `config.json` in each PRAR output directory records which state file was used.

## Usage

### Running New Experiments

```bash
# From repository root
python run_social_rl_local.py

# Outputs automatically saved to outputs/social_rl_YYYY-MM-DD_HHMMSS/
```

### Analyzing Outputs

```python
import json

# Load a round transcript
with open('outputs/social_rl_2025-11-23_043136/round1_social_rl.json') as f:
    round_data = json.load(f)

# Access messages
for msg in round_data['messages']:
    print(f"{msg['agent_id']}: {msg['content'][:100]}...")

# Access feedback
for agent_id, feedback in round_data['feedback'].items():
    print(f"{agent_id}: engagement={feedback['engagement']:.2f}")
```

## See Also

- [social_rl/README.md](../social_rl/README.md) - Social RL framework documentation
- [prar/README.md](../prar/README.md) - PRAR methodology and outputs
- [ROADMAP.md](../ROADMAP.md) - Development phases
