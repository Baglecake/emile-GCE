# Architecture Documentation

## System Overview

émile-GCE is a **Generative Computational Ethnography** platform for studying how architectural design embeds social conditions in multi-agent LLM systems.

### Design Principles

1. **CES-Grounded Agents**: All agents derive from real voter profiles (CES 2021, N=61,000)
2. **Dual-LLM Architecture**: Separate Performer (enacts) and Coach (validates) LLMs
3. **Semiotic Regime Detection**: System identifies emergent social patterns (e.g., ENGAGED_HARMONY)
4. **Identity-in-Place**: Agents have dynamic identity cores, not static demographic tags

## 2×2×2 Architectural Sweep

### Dimensions

**Challenge Mode** (coach intervention):
- `off`: Coach validates without challenging
- `always`: Coach challenges positions when consensus forms

**Context Type** (semiotic manifesting):
- `progressive`: Context descriptions accumulate linearly
- `adaptive`: Context responds to detected regimes

**LLM Architecture**:
- `dual`: Separate Performer + Coach LLMs
- `single`: Combined LLM (performer self-validates)

### 8 Conditions

| Condition | Challenge | Context | Dual-LLM | Description |
|-----------|-----------|---------|----------|-------------|
| A | off | progressive | ✓ | Baseline dual-LLM |
| B | off | progressive | ✗ | Baseline single-LLM |
| C | off | adaptive | ✓ | Regime-aware, no challenge |
| D | off | adaptive | ✗ | Regime-aware single-LLM |
| E | always | progressive | ✓ | Challenge + linear context |
| F | always | progressive | ✗ | Challenge single-LLM |
| **G** | **always** | **adaptive** | **✓** | **OPTIMAL** |
| H | always | adaptive | ✗ | Challenge single-LLM adaptive |

**Key Finding**: Condition G produces lowest hyper-enfranchisement (0.256 vs 0.667 in H).

## Phase Structure

### Phase 1: Architecture Optimization (COMPLETE ✅)

**Goal**: Identify optimal architectural configuration

**Method**:
- Run all 8 conditions (A-H) with multiple seeds (2-3 per condition)
- Extract identity vectors (engagement, institutional_faith, social_friction)
- Factorial analysis to isolate main effects and interactions

**Finding**:
- **G is optimal**: 2.6× better than H at preventing hyper-enfranchisement
- **Dual-LLM effect is context-dependent**: Helps with adaptive, hurts with progressive
- **Network topology effects**: Constraining low-salience agents affects all agents

**Deliverables**:
- [FACTORIAL_ANALYSIS.md](FACTORIAL_ANALYSIS.md): Full sweep results
- `outputs/`: All experimental data (A-H, seeds 2-3)
- `analysis/analyze_full_sweep.py`: Factorial analysis script

### Phase 2: Identity-Grounding (CURRENT 🔄)

**Goal**: Reduce G's residual +50% hyper-enfranchisement to CES-accurate levels (~0.17)

**Method**:
- Implement IdentityCore with émile QSE mechanics
- Per-round vector extraction (currently per-experiment only)
- Transfer entropy: TE(I→B) vs TE(others→I)
- Temperature modulation: T_base + k_r\*rupture + k_c\*(1-coherence) + k_n\*natality

**Components**:
- `agents/identity_core/`: IdentityCore class, coherence, emergent time
- `analysis/compute_transfer_entropy.py`: TE(I→B), TE(others→I)
- `social_rl/identity_core.py`: Integration with runner

**Deliverables** (planned):
- Grit v2: Calibrated constraints targeting +50% residual
- Identity salience experiments
- Per-round vector extraction + coherence trajectories

### Phase 3: Sociogeographic Embodiment (FUTURE ⏭️)

**Goal**: Full émile-style embodiment with mortality/natality

**Method**:
- SociogeographicBody: Tie-to-place, affordances
- Coach as convention field C(t): Strain = |I_t - C_t|
- Mortality: energy death, incoherence death, silencing death
- Natality: CES-grounded repopulation with perturbed vectors

**Components** (planned):
- `identity_core/life_cycle.py`: Mortality mechanics
- `identity_core/natality.py`: Relative z-score natality
- `identity_core/coach_field.py`: Social convention coupling

## Components

### `agents/`

**`ces_generators/`**: CES profile → agent configuration

- `identity_metrics.py`: Extract engagement, turnout, salience from CES
- `row_to_agent.py`: Convert CES row to agent canvas
- `needs_grit_constraint()`: Identify low-salience agents

**`identity_core/`** (Phase 2, planned):

- `core.py`: IdentityCore class with QSE mechanics
- `coherence.py`: cos(I_t, I_0) × TE ratio
- `temperature.py`: Dynamic T modulation

### `experiments/`

- `run_ces_experiment.py`: Main experiment runner
- `social_aesthetics_regimes.py`: Regime detection (ENGAGED_HARMONY, etc.)
- `analyze_sweep.py`: Per-condition analysis

### `social_rl/`

**Core social RL components**:

- `runner.py`: Multi-round deliberation orchestrator
- `feedback_extractor.py`: Compute engagement (references + responses + initiative)
- `context_injector.py`: Inject semiotic context (progressive or adaptive)
- `semiotic_coder.py`: Detect regimes from message patterns

**Identity integration** (Phase 2, planned):

- `identity_core.py`: IdentityCore integration with runner
- Per-agent temperature based on coherence/rupture

### `analysis/`

- `extract_identity_vectors.py`: Extract vectors from simulation logs
- `analyze_full_sweep.py`: Factorial analysis across conditions A-H
- `compute_transfer_entropy.py` (Phase 2, planned): TE(I→B), TE(others→I)

### `data/`

- `CES_2021.parquet`: Full CES dataset (N=61,000)
- `CES_2021_codebook.txt`: Variable definitions
- `DATA_DICTIONARY.md`: Identity vector mappings

### `outputs/`

Structure per experiment:

```
outputs/G_seed2_fixed/
├── config.json              # Experiment configuration
├── ces_profiles.json        # Agent CES profiles
├── meta.json                # Metadata
├── policy_state.json        # Final policy state
├── round1_social_rl.json    # Round 1 messages + feedback
├── round2_social_rl.json
├── round3_social_rl.json
└── semiotic_state_log.json  # Detected regimes
```

## Two-Layer LLM Architecture

### Computational Layer (Network Position)

**Measured by engagement formula** (`social_rl/feedback_extractor.py:361-371`):

```python
engagement = (reference_score * 0.4 +
              response_score * 0.4 +
              initiative_score * 0.2)
```

Where:
- `reference_score`: How often agent is mentioned by others
- `response_score`: How often agent receives responses
- `initiative_score`: How often agent initiates exchanges

**Architecture CAN modulate this**: G vs H shows 2.6× difference (0.256 vs 0.667).

### Affective Layer (Prose Style)

**Measured by**:
- Message length (chars)
- Politeness markers (qualitative)
- Helpfulness indicators

**RLHF "helpfulness gravity" resists architectural constraints**:
- Even with grit v1 (engagement→0), message length stayed ~2,200 chars
- Agents stayed polite and verbose despite constraints

**Phase 2 solution**: Dynamic temperature modulation based on identity coherence.

## Semiotic Regimes

Emergent patterns detected via `semiotic_coder.py`:

### ENGAGED_HARMONY

**Signature**:
- High mutual reference density
- Consensus on core concepts (e.g., "Healthcare," "Climate")
- Sustained across rounds

**Observed in**: Condition G (optimal architecture)

**Hypothesis**: G's low hyper-enfranchisement enables stable ENGAGED_HARMONY (agents don't all shout at once).

### PRODUCTIVE_DISSONANCE

**Signature**:
- High social friction (direct references + tension markers)
- Divergent issue positions maintained
- Mutual engagement without convergence

**Status**: Rarely observed (Phase 1)

**Phase 3 goal**: Achieve via identity-grounding (agents have "existential stakes" in positions).

## Factorial Effects

From Phase 1 analysis ([FACTORIAL_ANALYSIS.md](FACTORIAL_ANALYSIS.md)):

### Main Effects

1. **Dual-LLM effect is context-dependent**:
   - With adaptive: -0.067 (dual helps)
   - With progressive: +0.081 (dual hurts)

2. **Adaptive context helps**: -0.041 effect

3. **Challenge mode minimal**: +0.024 effect (affects semiotic style, not network position)

### Interaction Effect

G's combination (dual + adaptive + challenge) produces **synergistic optimal performance**.

### Network Topology Effects

When low-salience agents received grit constraints (G seed 6):

| Agent | Baseline | + Grit | Delta |
|-------|----------|--------|-------|
| Urban Progressive | 0.375 | 0.167 | -0.208 |
| Suburban Swing (GRIT) | 0.301 | 0.000 | -0.301 |
| Rural Conservative | 0.294 | 0.000 | -0.294 |
| Disengaged Renter (GRIT) | 0.321 | 0.000 | -0.321 |

**Interpretation**: High-salience agents (who did NOT receive grit) also withdrew. Agents are **relationally constituted** - constraining one reconfigures the entire social field.

## Weber's Positioned Associations

**Core concept**: Agents don't just have traits; they "form a part of and stand within" a social community.

**Evidence**:
- Network topology effects (above)
- G's dual-LLM allows differential validation (low-salience agents get permission to withdraw)
- Architecture shapes **social field**, not just individual behavior

**Phase 2-3 goal**: Make this explicit via Coach as convention field C(t) that agents couple to.

## Identity-in-Place Function (Phase 2)

Proposed formalization:

```python
I_i(τ) = f(identity_salience_i, tie_to_place_i, affordance_validation_i(τ))
```

Where:
- `identity_salience_i`: How much agent has invested in identity (surplus S)
- `tie_to_place_i`: Attachment to geographic/social context
- `affordance_validation_i(τ)`: Whether context confirms/disconfirms identity at emergent time τ

**Implementation** (Phase 2):

```python
class IdentityCore:
    vector: np.ndarray  # (engagement, faith, friction, tie_to_place, salience, ...)
    surplus: float      # QSE-style accumulated enactment
    sigma: float        # tension between I and enacted behavior
    tau_emergent: float # emergent time from |ΔI|
    coherence: float    # cos(I_t, I_0) × TE(I→B) / (TE(I→B)+TE(others→I))
    energy: float       # drains with dissonance, recovers with validation
    temperature: float  # T_base + k_r*rupture + k_c*(1-coherence) + k_n*natality
    history: deque      # for computing μ, σ for natality
```

## Running Experiments

### Basic Usage

```bash
# Run single experiment
python3 experiments/run_ces_experiment.py \
  --condition G \
  --seed 2 \
  --rounds 3 \
  --output outputs/my_experiment

# Extract vectors
python3 analysis/extract_identity_vectors.py outputs/my_experiment/
```

### Sweep Across Conditions

```bash
# Run all 8 conditions, seeds 2-3
for cond in A B C D E F G H; do
  for seed in 2 3; do
    python3 experiments/run_ces_experiment.py \
      --condition $cond \
      --seed $seed \
      --rounds 3
  done
done

# Analyze full sweep
python3 analysis/analyze_full_sweep.py
```

### Configuration Options

See `experiments/run_ces_experiment.py --help` for full options:

- `--condition`: A-H (architectural configuration)
- `--seed`: Random seed for agent sampling
- `--rounds`: Number of deliberation rounds (default: 3)
- `--n_agents`: Number of agents (default: 4)
- `--model_performer`: LLM for Performer (default: claude-3-7-sonnet-20250219)
- `--model_coach`: LLM for Coach (default: qwen/qwen-2.5-72b-instruct)

## Next Steps

### Immediate (Phase 2 start)

1. Implement per-round vector extraction (currently per-experiment)
2. Create stub `IdentityCore` class in `agents/identity_core/core.py`
3. Implement coherence calculation (cosine similarity I_t to I_0)

### Short-term (Phase 2 core)

1. Transfer entropy implementation (`analysis/compute_transfer_entropy.py`)
2. Temperature modulation based on coherence/rupture
3. Grit v2: Calibrated constraints + dynamic temperature

### Medium-term (Phase 2 complete)

1. Identity salience experiments (does high-salience prevent convergence?)
2. Tie-to-place metrics from CES
3. Qualitative analysis of G2 vs G6 messages (computational vs affective layers)

### Long-term (Phase 3)

1. Coach as convention field C(t)
2. Mortality mechanics (energy death, incoherence death, silencing death)
3. Natality/repopulation (CES-grounded children)
4. Multi-generation experiments

---

See [Research Roadmap](notes/research_roadmap.md) for detailed timeline.
