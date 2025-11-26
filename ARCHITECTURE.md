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

### Phase 2: Identity-Grounding (Stages 1-4 COMPLETE)

**Goal**: Reduce G's residual +50% hyper-enfranchisement to CES-accurate levels (~0.17)

**Implemented (Stages 1-4):**

1. **IdentityCore with QSE mechanics** (`agents/identity_core/core.py`, 791 lines)
   - Emergent time tau: `tau = logistic(|I_t - I_0|)`
   - Stateful natality with tau-based baseline
   - Qualitative surplus: `local_surplus = delta_I * f_tau * f_natality * f_recognition`
   - Coherence: `cos(I_t, I_0)` (TE ratio placeholder for Phase 2b)

2. **SurplusTrace buffer** (Stage 3)
   - Memory of enacted surplus events with decay/revalorization
   - Traces created on high-surplus + high-recognition events
   - Identity blending: `I_new = I_current + eta * T`

3. **Expression capacity** (`social_rl/runner.py`)
   - `soft_cap = base_cap * f_salience * f_natality`
   - Identity-grounded token limits (not crude punishment)

4. **TRUE dual-LLM** (`social_rl/dual_llm_client.py`)
   - Separate 14B Performer + 7B Coach on distinct GPU endpoints
   - `create_true_dual_llm()` factory function

5. **Grit v2** (`agents/ces_generators/grit_config.py`)
   - Tiered constraints: NONE/LIGHT/MODERATE/STRONG
   - CES-calibrated targets
   - Dynamic calibration based on engagement overshoot

**Remaining (Phase 2b):**
- Transfer entropy: TE(I->B) vs TE(others->I) for full coherence formula
- Multi-wave CES priors for empirical delta_mu/sigma per group
- Mortality mechanics (energy death, incoherence death, silencing death)

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

**`identity_core/`** (IMPLEMENTED):

- `core.py`: Full IdentityCore class (791 lines) with:
  - SurplusTrace dataclass (decay, revalorization, blending)
  - IdentityVector dataclass (3D: engagement, faith, friction)
  - Stateful natality with tau-based baseline
  - Qualitative surplus computation
  - Temperature modulation: `T = T_base + k_r*rupture + k_c*(1-coherence) + k_n*natality`
- `tau.py`: Emergent time from identity magnitude change
- `grit_config.py`: Tiered grit constraints (NONE/LIGHT/MODERATE/STRONG)

### `experiments/`

- `run_ces_experiment.py`: Main experiment runner
- `social_aesthetics_regimes.py`: Regime detection (ENGAGED_HARMONY, etc.)
- `analyze_sweep.py`: Per-condition analysis

### `social_rl/`

**Core social RL components**:

- `world_state.py`: WorldState engine providing identity-grounded environmental context (see below)
- `runner.py`: Multi-round deliberation orchestrator (1130+ lines) with:
  - Expression capacity: `soft_cap = base_cap * f_salience * f_natality`
  - SurplusTrace management (decay, creation, revalorization)
  - IdentityCore integration per agent
  - Per-agent temperature modulation
- `feedback_extractor.py`: Compute engagement + recognition scores
- `context_injector.py`: Inject semiotic context (progressive or adaptive)
- `semiotic_coder.py`: Detect regimes from message patterns
- `dual_llm_client.py`: TRUE dual-LLM architecture (IMPLEMENTED)
  - `create_true_dual_llm()`: Factory for separate Performer/Coach endpoints
  - Coach validates, Performer generates
  - Supports 14B Performer + 7B Coach on separate GPUs

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

## WorldState Engine

**Location**: `social_rl/world_state.py`

The WorldState engine provides agents with a dynamic environment that differentially affects them based on their 7D identity vectors. Rather than presenting identical context to all agents, the engine generates identity-grounded experiences of shared events.

### Core Components

**WorldEvent**: Events that occur in the world (e.g., housing price increases, factory closures, policy announcements). Each event defines:
- `salience_weights`: Which identity dimensions this event activates (and in which direction)
- `valence_map`: Emotional framing by group membership (e.g., "threatening" vs. "opportunity")
- `stakes_map`: What is at risk for different identity profiles

**DiscussionTopic**: Topics for deliberation with identity-specific stakes and position seeds. Topics rotate sequentially or randomly across rounds.

**AgentState**: Per-agent accumulating state including:
- `frustration`: Increases when agent contributions are not recognized
- `fatigue`: Accumulates over rounds, reduced by successful engagement
- `entrenchment`: Position hardening when repeatedly challenged without movement

### Differential Impact Mechanism

The engine computes per-agent impact through identity vector alignment:

```python
salience = sum(
    weight * (dim_value if weight > 0 else 1 - dim_value)
    for dim, weight in event.salience_weights.items()
)
```

This produces different salience scores for the same event based on agent identity. A housing price surge registers as "urgent" for an urban renter but "distant" for a rural homeowner.

### Integration

Enable WorldState via the `--world-state` flag:

```bash
python3 experiments/run_ces_experiment.py \
  --condition G \
  --seed 42 \
  --rounds 10 \
  --world-state \
  --experiment-id "worldstate_experiment"
```

The engine injects context into user messages (not system prompts) for better LLM attention:

```
=== WORLD CONTEXT ===
[RECENT NEWS: Housing prices surge 15% in major cities]
For you, this feels urgent. At stake: Whether you can afford to stay

[DISCUSSION: What does it mean to belong somewhere?]
Consider your connection to place, community, and identity.
For someone like you: Freedom to move matters more than staying
```

### Validation Results

Extended 10-round experiment (`world-output/worldstate_FIXED_extended`) demonstrated:
- Engagement trajectory: 0.48 (Round 1) to 0.80 (Round 10, EMA-smoothed)
- Regime trajectory: UNKNOWN to ACTIVE_CONTESTATION with ENGAGED_HARMONY phases
- No divergence events or collapse
- Agents addressing each other by name throughout discourse

## Conversation Engagement Protocol

**Location**: `social_rl/runner.py`, `_build_user_message()` method

A critical component ensuring authentic multi-agent discourse rather than parallel monologues.

### Problem Addressed

Initial experiments exhibited a "repetition bug" where agents produced identical messages within the same round, indicating failure to engage with prior speakers. Root cause analysis identified that the user message construction merely listed conversation history without explicit engagement requirements.

### Solution

The `_build_user_message()` function now enforces explicit engagement:

```python
engagement_prompt = (
    f"You are {agent_name}. You have just heard from: {recent_names}.\n"
    f"CRITICAL: You MUST directly respond to or acknowledge at least ONE specific point "
    f"from a previous speaker. Do NOT simply repeat your own position.\n"
    f"Engage with what others have said, then share your perspective."
)
```

Key elements:
1. **Turn numbering**: Each turn explicitly numbered for LLM awareness
2. **Speaker identification**: Recent speakers named explicitly
3. **Engagement mandate**: Explicit instruction to reference prior contributions
4. **Position warning**: Explicit prohibition against mere repetition

### Verification

The fix was validated across multiple 10-round experiments. Discourse logs confirm:
- Agents reference each other by name (e.g., "Rural Conservative makes a valid point...")
- Substantive engagement with prior arguments
- No repetition of identical content within rounds

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

## Identity-in-Place Function (IMPLEMENTED)

Formalization:

```python
I_i(tau) = f(identity_salience_i, natality_i, surplus_i, trace_history_i)
```

Where:
- `identity_salience_i`: From CES turnout * (1 - lr_distance) * engagement
- `natality_i`: Capacity for new beginnings (tau-based baseline, recognition-modulated)
- `surplus_i`: Qualitative capacity = delta_I * f_tau * f_natality * f_recognition
- `trace_history_i`: Memory of enacted surplus events

**Actual Implementation** (`agents/identity_core/core.py`):

```python
@dataclass
class SurplusTrace:
    """Memory of an enacted surplus event."""
    round_number: int
    turn_number: int
    semiotic_regime: str
    delta_I: np.ndarray      # 3D normalized direction
    tau_at_event: float
    natality_at_event: float
    recognition_score: float
    contribution_value: float
    engagement: float
    weight: float            # Decays unless revalorized

class IdentityCore:
    """Full identity mechanics for GCE agents."""

    # State
    vector: IdentityVector           # 3D: engagement, institutional_faith, social_friction
    initial_vector: IdentityVector   # I_0 for coherence
    natality_t: float                # Current natality (stateful)
    surplus: float                   # EMA-smoothed qualitative surplus
    traces: List[SurplusTrace]       # Trace buffer (max 10)
    _recognition_ema: float          # Smoothed recognition score

    def compute_tau(self) -> float:
        """Emergent time: tau = logistic(|I_t - I_0|)"""

    def update_natality(self, recognition_score: float, overshoot: float) -> float:
        """Stateful natality with recognition/overshoot modulation"""
        # Decay toward tau-based baseline
        # Boost when recognized, suppress when overshooting

    def update_surplus(self) -> float:
        """Qualitative capacity: local_surplus = delta_I * f_tau * f_nat * f_rec"""

    def maybe_create_trace(self, ...) -> Optional[SurplusTrace]:
        """Create trace on high-surplus + high-recognition events"""

    def apply_trace_blend(self, eta: float = 0.1) -> None:
        """I_new = I_current + eta * T (weighted trace direction)"""

    def get_temperature(self, base_temp: float) -> float:
        """T = T_base + k_r*rupture + k_c*(1-coherence) + k_n*natality"""
```

**Expression Capacity** (`social_rl/runner.py:444-472`):

```python
# Identity-grounded token limits
f_salience = 0.5 + 0.5 * identity_salience  # [0.5, 1.0]
f_natality = 0.5 + 0.5 * natality_t         # [0.5, 1.0]
soft_cap = int(base_cap * f_salience * f_natality)
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

### Completed (Phase 2a - Stages 1-4)

- [x] IdentityCore class with full QSE mechanics
- [x] Emergent time tau: `logistic(|I_t - I_0|)`
- [x] Stateful natality with tau-based baseline
- [x] Qualitative surplus with f_tau, f_natality, f_recognition modulation
- [x] SurplusTrace buffer (decay, revalorization, blending)
- [x] Expression capacity: `soft_cap = base_cap * f_salience * f_natality`
- [x] Temperature modulation: `T = T_base + k_r*rupture + k_c*(1-coherence) + k_n*natality`
- [x] TRUE dual-LLM with separate 14B/7B endpoints
- [x] Grit v2: Tiered constraints (NONE/LIGHT/MODERATE/STRONG)
- [x] Per-round identity state logging

### Phase 2b (Current)

1. Transfer entropy: TE(I->B) vs TE(others->I) for coherence formula
2. Multi-wave CES priors for empirical delta_mu/sigma per group
3. Tie-to-place metrics integration from CES riding data

### Phase 3 (Future)

1. Coach as convention field C(t) (inner/outer thought separation)
2. Mortality mechanics (energy death, incoherence death, silencing death)
3. Field vector F_t from discourse (alignment = cos(I_t, F_t))
4. Natality/repopulation (CES-grounded children)
5. Multi-generation experiments

---

See [Research Roadmap](notes/research_roadmap.md) for detailed planning.
