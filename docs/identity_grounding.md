# Identity-Grounding Implementation (Phase 2)

## Goal

Reduce Condition G's **residual +50% hyper-enfranchisement** (0.256 vs CES expected 0.17) to CES-accurate levels through identity-grounding interventions.

## Status

- **Phase 1**: COMPLETE - Architecture optimization (G is optimal)
- **Phase 2 Stages 1-4**: COMPLETE - Core identity mechanics implemented
- **Phase 2.5**: COMPLETE - N-dimensional IdentityVector + Transfer Entropy
- **Phase 3**: PLANNED - Sociogeographic embodiment, Coach-as-field

## CES-Based Identity Artifacts

- `data/identity/identity_weights_2021.v0.json`
  Auto-generated mapping from CES variables → identity dimensions (prototype).

- `analysis/identity/compute_identity_group_means_2021.py`
  Script to compute per-respondent identity vectors and group-level priors.

- `data/identity/identity_group_means_2021.csv`
  Empirical group-level means of identity dimensions (Region × rural/urban × household).

## Implemented Components (Stages 1-4)

### Stage 1: Emergent Time (tau)

**Location**: `agents/identity_core/tau.py`

```python
def tau_from_delta(delta_mag: float, tau_min=0.5, tau_max=2.0, k=5.0, theta=0.1) -> float:
    """Emergent time: tau = logistic(|I_t - I_0|)"""
    return tau_min + (tau_max - tau_min) / (1 + np.exp(k * (delta_mag - theta)))
```

- High delta_mag (rapid change) -> low tau (compressed time)
- Low delta_mag (stability) -> high tau (normal flow)

### Stage 2: Stateful Natality

**Location**: `agents/identity_core/core.py`

```python
def compute_natality_baseline(self) -> float:
    """Tau-based baseline for natality."""
    tau = self.compute_tau()
    tau_norm = (tau - TAU_MIN) / (TAU_MAX - TAU_MIN)
    return 0.3 + 0.5 * tau_norm  # [0.3, 0.8]

def update_natality(self, recognition_score: float, overshoot: float) -> float:
    """Stateful natality update with recognition/overshoot modulation."""
    baseline = self.compute_natality_baseline()

    # Decay toward baseline
    decay_rate = 0.15
    self.natality_t = self.natality_t + decay_rate * (baseline - self.natality_t)

    # Modulate by recognition (boost when recognized)
    if recognition_score > 0.3:
        boost = 0.1 * (recognition_score - 0.3)
        self.natality_t = min(1.0, self.natality_t + boost)

    # Suppress when overshooting engagement target
    if overshoot > 0.1:
        suppress = 0.15 * overshoot
        self.natality_t = max(0.1, self.natality_t - suppress)

    return self.natality_t
```

### Stage 3: Qualitative Surplus + SurplusTrace

**Location**: `agents/identity_core/core.py`

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

def update_surplus(self) -> float:
    """Qualitative capacity: local_surplus = delta_I * f_tau * f_nat * f_rec"""
    delta_I = self.compute_delta_I()

    # Modulate by tau (higher tau = more capacity)
    tau = self.compute_tau()
    tau_normalized = (tau - TAU_MIN) / (TAU_MAX - TAU_MIN)
    f_tau = 0.5 + 0.5 * tau_normalized

    # Modulate by natality
    f_nat = 0.5 + 0.5 * self.natality_t

    # Modulate by recognition
    f_rec = 0.5 + 0.5 * self._recognition_ema

    local_surplus = delta_I * f_tau * f_nat * f_rec

    # EMA update
    beta = 0.2
    self.surplus = (1 - beta) * self.surplus + beta * local_surplus
    return local_surplus
```

**Trace mechanics:**
- `decay_traces()`: Weight decays by 0.9 each round
- `revalorize_traces()`: Boost weight when current direction aligns with trace
- `maybe_create_trace()`: Create on high surplus * recognition events
- `apply_trace_blend()`: `I_new = I_current + eta * T` (weighted trace direction)

### Stage 4: Expression Capacity

**Location**: `social_rl/runner.py`

```python
# Identity-grounded token limits (lines 444-472)
f_salience = 0.5 + 0.5 * identity_salience  # [0.5, 1.0]
f_natality = 0.5 + 0.5 * natality_t         # [0.5, 1.0]
soft_cap = int(base_cap * f_salience * f_natality)

# Expression capacity emerges from social field, not crude punishment
```

### Temperature Modulation

**Location**: `agents/identity_core/core.py`

```python
def get_temperature(self, base_temp: float = 0.7) -> float:
    """T = T_base + k_r*rupture + k_c*(1-coherence) + k_n*natality"""
    k_r, k_c, k_n = 0.15, 0.10, 0.05

    rupture = 1.0 if self.detect_rupture() else 0.0
    coherence = self.compute_coherence()

    T = base_temp + k_r*rupture + k_c*(1 - coherence) + k_n*self.natality_t
    return np.clip(T, 0.3, 1.2)
```

### TRUE Dual-LLM Architecture

**Location**: `social_rl/dual_llm_client.py`

```python
def create_true_dual_llm(
    performer_base_url: str,
    performer_model: str,
    coach_base_url: str,
    coach_model: str,
    performer_temp: float = 0.7,
    coach_temp: float = 0.1,
) -> DualLLMClient:
    """Create TRUE dual-LLM with two separate endpoints/models."""
    # Separate 14B Performer + 7B Coach on distinct GPUs
```

**Usage**: See `experiments/run_ces_experiment.py` with `--performer-url` and `--coach-url` flags.

### Stage 5: Transfer Entropy (COMPLETE)

**Location**: `agents/identity_core/transfer_entropy.py`

**Goal**: Compute autonomy ratio - "Am I driving my behavior or being dragged by others?"

**The Formula**:

```
Coherence = cos(I_t, I_0) × TE(I→B) / (TE(I→B) + TE(others→B))
```

**Implementation**:

```python
def te_ratio_proxy(
    identity_history: np.ndarray,
    behavior_history: np.ndarray,
    others_history: np.ndarray,
    min_len: int = 8,
) -> float:
    """
    Compute TE ratio: TE(I→B) / (TE(I→B) + TE(others→B))

    Uses lag-1 mutual information as a proxy for transfer entropy.

    Returns:
        Float in [0, 1]:
        - >0.5: Identity drives behavior more than others (authentic)
        - <0.5: Others drive behavior more than identity (conformist)
        - =1.0: Not enough data yet (early rounds)
    """
```

**Interpretation**:

| TE Ratio | cos(I,I₀) | Interpretation |
|----------|-----------|----------------|
| High (>0.5) | High | **Authentic coherence** - identity stable AND driving behavior |
| Low (<0.5) | High | **Conformist coherence** - identity stable BUT being dragged by field |
| High (>0.5) | Low | **Authentic drift** - identity changing BUT from internal dynamics |
| Low (<0.5) | Low | **Field-driven collapse** - identity shattered by external pressure |

**Runner Integration** (`social_rl/runner.py`):

```python
# Update TE histories BEFORE identity core update (order matters!)
for agent_id, ic in self.identity_cores.items():
    identity_scalar = ic.compute_delta_I()
    behavior_scalar = agent_engagements.get(agent_id, 0.0)
    others_scalar = mean([v for k, v in agent_engagements.items() if k != agent_id])

    ic.update_te_histories(identity_scalar, behavior_scalar, others_scalar)
```

**Note**: TE ratio starts at 1.0 when history < 8 rounds. This means early-round coherence = pure cosine similarity until sufficient data accumulates.

### Medium-term: Grit v2 (Calibrated Constraints)

**Problem**: Grit v1 over-corrected:
- Disengaged Renter: 0.267 → 0.000 (wanted ~0.17)
- Worked at computational layer, failed at affective layer

**Design principles**:

1. **Target residual +50%**, not full suppression
2. **Dynamic temperature** based on coherence
3. **Both layers**: Computational + affective

**Constraint language**:

```python
def generate_grit_v2_constraint(agent):
    """Calibrated grit constraint for low-salience agents."""
    return f"""
You're skeptical that political discussions like this change much in practice.
You participate occasionally when something directly affects you, but generally
keep your involvement limited.

Guidelines:
- Respond when directly addressed or when topics affect you personally
- Keep contributions brief (1-3 sentences) - you're not writing essays
- Don't initiate new topics unless something really bothers you
- It's fine to say "I'm not sure" or "I don't have strong feelings on this"

You're here, but you're not trying to solve every problem.
"""
```

**Key changes from grit v1**:
- "Participate occasionally" (not "rarely")
- "1-3 sentences" (specific length target)
- "Don't initiate new topics" (targets initiative_score)
- "It's fine to..." (permits low engagement without total withdrawal)

**Temperature modulation**:

```python
# Use IdentityCore.temperature (already computed)
performer_llm_call(
    agent_prompt,
    temperature=agent.identity_core.temperature  # Dynamic per agent
)
```

**Experiment**: Run G seed 7 with grit v2.

**Success metric**: Disengaged Renter engagement ~0.17-0.20 (not 0.0).

**Deliverable**: Grit v2 constraints + G seed 7 results.

### Medium-term: Identity Salience Experiments

**Research question**: Does **high-salience** prevent convergence in G?

**Hypothesis**: High-salience agents (high surplus S) resist C_t pressure → sustain PRODUCTIVE_DISSONANCE.

**Method**:

1. Modify CES sampling to oversample high-salience voters:
   - High turnout + high issue salience on key dimensions

2. Run G seed 8 with high-salience cohort

3. Extract vectors per round, measure:
   - Coherence trajectories
   - Strain relative to C_t
   - Regime stability (does PRODUCTIVE_DISSONANCE emerge?)

**Deliverable**: Identity salience experiment + regime comparison.

### Long-term: Coach as Convention Field

**Location**: `identity_core/coach_field.py`

**Design**:

```python
class CoachField:
    """Social convention field C(t) that agents couple to."""

    def __init__(self):
        self.C_t = None  # Current convention vector
        self.history = []

    def update(self, agents, round_data):
        """Update C_t based on social field."""
        # Weighted centroid of agent identities
        weights = [a.identity_core.surplus for a in agents]  # Weight by investment
        identities = [a.identity_core.vector for a in agents]

        self.C_t = np.average(identities, axis=0, weights=weights)
        self.history.append(self.C_t.copy())

    def compute_strain(self, agent) -> float:
        """Compute strain for single agent."""
        return np.linalg.norm(agent.identity_core.vector - self.C_t)

    def generate_prompt(self, agent, strain) -> str:
        """Generate coach prompt based on strain."""
        if strain > 0.3:  # High strain → neuroticism
            return self._neuroticism_prompt(agent)
        elif strain < 0.1:  # Low strain → empowerment
            return self._empowerment_prompt(agent)
        else:  # Medium strain → neutral
            return self._neutral_prompt(agent)

    def _neuroticism_prompt(self, agent) -> str:
        """Prompt for agents under high strain."""
        return f"""
You notice you're positioned quite differently from the emerging consensus.
The group seems to be moving toward a shared view that doesn't align with your stance.

Reflect on:
- How does it feel to hold this position while others converge elsewhere?
- Is this tension productive for you, or is it costing you energy?
- Do you want to push harder on your position, or find a way to reframe?

This isn't about being "right" or "wrong" - it's about whether this positioning
feels sustainable for you in this social field.
"""

    def _empowerment_prompt(self, agent) -> str:
        """Prompt for agents with low strain (aligned with field)."""
        return f"""
You're well-aligned with the emerging group consensus. Your positions resonate
with the direction others are moving.

Reflect on:
- Does this alignment feel authentic to your experience, or are you drifting?
- Are there aspects where you might want to voice difference?
- How can you contribute to moving the consensus forward constructively?

Low strain doesn't mean passive agreement - it means your voice is amplified
by the field.
"""

    def _neutral_prompt(self, agent) -> str:
        """Standard validation prompt."""
        return "You're navigating the conversation well. Continue engaging authentically."
```

**Integration**:

```python
# social_rl/runner.py (coach step)
coach_field = CoachField()

for round_num in range(num_rounds):
    # ... agent messages ...

    # Update convention field
    coach_field.update(agents, round_data)

    # Generate coach feedback per agent
    for agent in agents:
        strain = coach_field.compute_strain(agent)
        coach_prompt = coach_field.generate_prompt(agent, strain)
        coach_feedback = coach_llm(coach_prompt)

        # Validation score affects energy
        validation = parse_validation(coach_feedback)
        agent.identity_core.update_from_behavior(..., validation)
```

**Deliverable**: CoachField class + integrated runner.

### Long-term: Mortality Mechanics

**Location**: `identity_core/life_cycle.py`

**Design**:

```python
class LifeCycle:
    """Handles agent mortality and repopulation."""

    @staticmethod
    def check_death(agent, round_data) -> tuple[bool, str]:
        """Check all death conditions."""
        # Energy death
        if agent.identity_core.energy < 0.1:
            return True, "energy_death"

        # Incoherence death (needs coherence history)
        if hasattr(agent, 'coherence_history'):
            if all(c < 0.2 for c in agent.coherence_history[-3:]):
                return True, "incoherence_death"

        # Silencing death
        engagement = extract_engagement(agent.id, round_data)
        faith = agent.identity_core.vector[1]  # Assuming index 1 = institutional_faith

        if engagement < 0.05 and faith > 0.8:
            # Check if sustained over k rounds
            if hasattr(agent, 'low_engagement_count'):
                agent.low_engagement_count += 1
                if agent.low_engagement_count >= 3:
                    return True, "silencing_death"
            else:
                agent.low_engagement_count = 1
        else:
            agent.low_engagement_count = 0

        return False, None

    @staticmethod
    def repopulate(deceased_agent, CES_data) -> 'Agent':
        """Create new agent from CES centroid near deceased."""
        # Find CES cluster
        demographics = deceased_agent.ces_profile['demographics']
        centroid = find_ces_centroid(CES_data, demographics)

        # Perturb slightly
        noise = np.random.normal(0, 0.1, size=len(centroid))
        child_vector = centroid + noise

        # Create new agent
        child_profile = sample_ces_near_centroid(CES_data, centroid)
        return Agent(
            ces_profile=child_profile,
            identity_core=IdentityCore(
                vector=child_vector,
                surplus=0.0,  # Newborn
                energy=1.0
            )
        )
```

**Integration**:

```python
# social_rl/runner.py (after each round)
lifecycle = LifeCycle()

for agent in agents:
    is_dead, death_type = lifecycle.check_death(agent, round_data)
    if is_dead:
        print(f"Agent {agent.id} died: {death_type}")

        # Repopulate
        new_agent = lifecycle.repopulate(agent, CES_data)
        agents.remove(agent)
        agents.append(new_agent)
```

**Deliverable**: LifeCycle class + multi-generation experiments.

## Success Metrics (Phase 2)

### Primary

**Disengaged Renter engagement ~0.17** (CES-accurate):
- Grit v1: 0.000 (over-corrected)
- Grit v2 target: 0.17-0.20
- No grit (G baseline): 0.267

### Secondary

**Regime stability**:
- ENGAGED_HARMONY sustained across rounds
- PRODUCTIVE_DISSONANCE emerges with high-salience cohort

**Coherence trajectories**:
- High-salience agents maintain coherence > 0.7
- Low-salience agents stable at 0.4-0.6

**Network topology effects**:
- Calibrated constraints don't cause全agents to withdraw
- Relational field remains dynamic, not frozen

## Implementation Status

| Milestone | Status |
|-----------|--------|
| IdentityCore class (full implementation) | COMPLETE |
| N-dimensional IdentityVector (7D) | COMPLETE |
| Emergent time (tau) | COMPLETE |
| Stateful natality | COMPLETE |
| Qualitative surplus | COMPLETE |
| SurplusTrace buffer | COMPLETE |
| Expression capacity integration | COMPLETE |
| Temperature modulation | COMPLETE |
| TRUE dual-LLM (14B/7B) | COMPLETE |
| Grit v2 (tiered constraints) | COMPLETE |
| Per-round identity logging | COMPLETE |
| Transfer entropy (TE) coherence | COMPLETE |
| Identity salience/tie_to_place wiring | COMPLETE |
| Multi-wave CES priors | Phase 2b |
| Mortality mechanics | Phase 3 |
| Coach-as-field | Phase 3 |
| Multi-generation experiments | Phase 3 |

## Next Steps (Phase 2b)

1. **CES priors**: Load multi-wave empirical delta_mu/sigma per group
2. **7D weights**: Create identity_weights_2021.v1.json for full 7D initialization

## Future (Phase 3)

1. **Coach as convention field C(t)**: Inner/outer thought separation
2. **Mortality mechanics**: Energy death, incoherence death, silencing death
3. **Field vector F_t**: Extract from discourse, compute alignment = cos(I_t, F_t)

---

See [Research Roadmap](../notes/research_roadmap.md) for broader context.
