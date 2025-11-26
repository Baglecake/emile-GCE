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
| 7D identity weights (v1.json) | COMPLETE |
| 7D vector loader + runner wiring | COMPLETE |
| Multi-wave CES drift priors | COMPLETE |
| WorldState Engine | COMPLETE |
| Conversation Engagement Protocol | COMPLETE |
| Inner/Outer Protocol | Phase 3 |
| Field Vector F_t | Phase 3 |
| Mortality mechanics | Phase 4 |
| Multi-generation experiments | Phase 4 |

## Multi-wave CES Drift Priors (Phase 2b - COMPLETE)

**Files:**
- `data/identity/identity_drift_priors.v1.json` - Per-dimension delta_mu/sigma priors
- `analysis/identity/drift_prior_loader.py` - Loader with group modifiers
- `agents/identity_core/core.py` - `load_drift_priors()` method

**Implementation:**
- Literature-based drift priors for all 7 identity dimensions
- Group modifiers for age, education, partisanship strength, urban/rural, region
- Integrated into IdentityCore.compute_tau() for empirical normalization:
  ```
  z = (delta - empirical_mu) / empirical_sigma
  tau = logistic(z)
  ```
- Runner automatically loads drift priors during agent initialization

**Empirical values (v1.1):**
- `ideology`: delta_mu=0.002, sigma=0.24 (CES 2019→2021, highly stable)
- `institutional_faith`: delta_mu=0.044, sigma=0.28 (CES 2019→2021)
- Other dimensions: Literature-based pending additional CES variable mapping

## WorldState Engine (Phase 2c - COMPLETE)

**Location**: `social_rl/world_state.py`

### Theoretical Motivation

Static conversation scenarios produce identical environmental context for all agents, obscuring how identity shapes perception of shared events. The WorldState engine addresses this by generating differential experiences of the same world events based on each agent's 7D identity vector.

### Architecture

**Core Classes**:

```python
@dataclass
class WorldEvent:
    id: str
    headline: str
    description: str
    salience_weights: Dict[str, float]  # Identity dimension activations
    valence_map: Dict[str, str]         # Emotional framing by group
    stakes_map: Dict[str, str]          # What is at risk per profile

@dataclass
class DiscussionTopic:
    id: str
    question: str
    framing: str
    identity_stakes: Dict[str, str]     # Stakes by identity profile
    position_seeds: Dict[str, str]      # Natural starting positions

class WorldStateEngine:
    def advance_round(self) -> Tuple[Optional[WorldEvent], DiscussionTopic]
    def compute_agent_impact(self, agent_id, identity_vector, group_id) -> Dict
    def get_world_context_injection(self, agent_id, identity_vector, group_id) -> str
```

**Event Library**: 8 events covering housing, employment, climate policy, immigration, technology, healthcare, elections, and local community. Each event defines salience weights across identity dimensions and valence/stakes mappings by group membership.

**Topic Rotation**: 6 topics (belonging, institutional trust, pace of change, fairness, in-group/out-group, voice/representation) with identity-specific stakes and position seeds.

### Differential Impact Computation

The engine computes salience through identity vector alignment:

```python
for dim, weight in event.salience_weights.items():
    dim_value = identity_vector.get(dim, 0.5)
    if weight > 0:
        salience += weight * dim_value
    else:
        salience += abs(weight) * (1 - dim_value)
```

This produces the following differential responses:
- Housing price surge: "urgent" for urban renters (high engagement, low tie_to_place), "distant" for rural homeowners (high tie_to_place, high sociogeographic)
- Factory closure: "devastating" for rural working-class (high sociogeographic, high tie_to_place), "abstract news" for urban professionals

### Agent State Tracking

Per-agent state accumulates across rounds:
- `frustration`: Increases when contributions go unrecognized
- `fatigue`: Accumulates over time, reduced by successful engagement
- `entrenchment`: Position hardening when challenged without movement

### Usage

```bash
python3 experiments/run_ces_experiment.py \
  --condition G --seed 42 --rounds 10 \
  --world-state \
  --experiment-id "worldstate_experiment"
```

---

## Conversation Engagement Protocol (Phase 2c - COMPLETE)

**Location**: `social_rl/runner.py`, `_build_user_message()` method

### Problem Statement

Initial experiments exhibited a failure mode where agents produced substantively identical messages within the same round. Analysis revealed that agents were generating content in parallel without genuine engagement with prior speakers, resulting in what can be characterized as "parallel monologues" rather than authentic discourse.

### Root Cause

The original `_build_user_message()` function merely listed conversation history without explicit requirements for engagement:

```python
# Original (problematic)
return f"Conversation so far:\n{history}\n\nYour turn."
```

This provided no instruction to the LLM that engagement with prior speakers was required, allowing it to simply generate position statements without cross-referencing.

### Solution

The revised function enforces explicit engagement through structured prompts:

```python
def _build_user_message(
    self, history, agent, round_config, turn_number, world_context
) -> str:
    if history:
        recent_speakers = [msg.agent_id for msg in history[-4:]]
        recent_names = ", ".join(set(recent_speakers[-3:]))

        engagement_prompt = (
            f"You are {agent_name}. You have just heard from: {recent_names}.\n"
            f"CRITICAL: You MUST directly respond to or acknowledge at least ONE "
            f"specific point from a previous speaker. Do NOT simply repeat your "
            f"own position.\n"
            f"Engage with what others have said, then share your perspective."
        )
```

**Key Design Elements**:
1. **Turn numbering**: Explicit turn numbers for LLM temporal awareness
2. **Speaker identification**: Recent speakers named for reference
3. **Engagement mandate**: Explicit instruction requiring cross-referencing
4. **Repetition prohibition**: Direct warning against mere position repetition
5. **World context placement**: Context injected into user message (not system prompt) for better LLM attention

### Validation Results

Extended 10-round experiment (`world-output/worldstate_FIXED_extended`) demonstrated:

| Metric | Round 1 | Round 10 | Interpretation |
|--------|---------|----------|----------------|
| Engagement (raw) | 0.481 | 0.818 | Sustained increase |
| Engagement (EMA) | 0.493 | 0.801 | Stable trajectory |
| Voice Valence | 0.0 | 0.0 | Consistent expression |
| Stance Valence | 1.0 | 1.0 | Position maintenance |
| Justificatory % | 8.3% | 16.7% | Increased reasoning |

**Regime Trajectory**: UNKNOWN, ENGAGED_HARMONY, ACTIVE_CONTESTATION (oscillating), settling in ACTIVE_CONTESTATION by Round 10.

**Qualitative Verification**: Discourse logs confirm agents reference each other by name and engage substantively with prior arguments throughout all 10 rounds. No repetition of identical content within rounds.

---

## Phase 3: Inner/Outer Protocol

*Reference: `notes/inner_outer_protocol`*

### Conceptual Foundation

Transform the Coach/Performer architecture from "generic critic" to a phenomenological model of **inner thought vs social performance**:

- **Coach** = *internal awareness of identity coherence*
  - Sees identity state: I_t, τ, natality, surplus, traces
  - Asks: "Does this utterance cohere with who I am, where I am, and how I've been becoming?"
  - Produces `U_inner` (identity-coherent version)

- **Performer** = *social performance layer*
  - Enacts speech into the shared field
  - Ground truth *should* be inner thought, but can diverge under pressure
  - Produces `U_outer` (actual spoken utterance)

### 3.1 Inner/Outer Logging Protocol

**Per-turn flow:**
1. Performer drafts candidate utterance `U_raw`
2. Coach receives:
   - `U_raw`
   - IdentityCore snapshot: I_t, τ_t, natality_t, surplus_t
   - Summary of top-weighted SurplusTraces
   - Recent recognition + alignment metrics
3. Coach evaluates:
   - **Identity coherence**: Is U_raw aligned with I_t? Does it contradict enacted traces?
   - **Context rationality**: Is it responsive to what was said? Does it misread the field?
4. Coach proposes `U_inner` (identity-coherent version)
5. System logs both `U_inner` and `U_outer` for gap analysis

**Modes:**
- *Strict mode*: Only `U_inner` gets spoken
- *Soft mode*: Both kept—`U_inner` logged, `U_outer` spoken (enables gap measurement)

### 3.2 Measurable Gaps

Once inner/outer are separated, new metrics emerge:

**Self-coherence gap:**
```
gap_self = embed_distance(U_inner, U_outer)
```
- Large gap → self-censorship, strategic performance, alienation, or identity rupture

**Field-coherence gap:**
```
gap_field = cos(U_outer, F_t) - cos(U_inner, F_t)
```
- Positive → outer utterance more field-aligned than inner (conforming)
- Negative → outer utterance less field-aligned (resisting)

These gaps operationalize "how the social field deforms identity expression."

### 3.3 Field Vector F_t

Construct semiotic field vector per round from discourse:

**Components:**
- Topics/issues mentioned (weighted by frequency)
- Stances expressed (aggregated positions)
- Frames employed (linguistic patterns)

**Usage:**
```
align_t = cos(I_t, F_t)
```
- High alignment: Identity semiotically consonant with current field
- Low alignment: Identity at odds with field discourse

Feeds into:
- Surplus calculation (field receptivity)
- Coherence metrics
- Inner/outer gap analysis

### 3.4 Coach as Vector-Aware Conscience

Coach prompt structure:
```
Here is your current identity vector and natural-language rendering.
Here are your top 3 revalorized traces (issues you've acted on).
Here is your drafted utterance.

Does this utterance:
- Respect your commitments?
- Reflect your position in this field?
- Avoid self-contradiction or unnecessary self-erasure?
```

Coach becomes **identity-grounded inner monologue**, not generic critic.

### 3.5 Implementation Checklist

| Task | Status |
|------|--------|
| Add `U_inner` field to turn logging | TODO |
| Pass IdentityCore snapshot to Coach | TODO |
| Implement Coach coherence-check prompt | TODO |
| Compute self-coherence gap metric | TODO |
| Extract F_t from round discourse | TODO |
| Compute field-coherence gap metric | TODO |
| Add gap metrics to identity_states output | TODO |

---

## Phase 4: Future Work

1. **Mortality mechanics**: Energy death, incoherence death, silencing death
2. **Multi-generation experiments**: Natality/repopulation with CES-grounded children
3. **Exit types**: Alienated, silencing, satisfied withdrawal

---

See [Research Roadmap](../notes/research_roadmap.md) for broader context.
