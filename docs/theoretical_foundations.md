# Theoretical Foundations

## Social Aesthetics (Weber, Simmel)

### Weber's Positioned Associations

**Key concept**: Agents "form a part of and stand within" a social community. They don't have fixed traits; they occupy **relational positions**.

From Weber (1922):
> "Association denotes a social relationship which is closed to outsiders... insofar as participation is limited by the order governing the organized group."

**Operationalization in émile-GCE**:
- Agents don't have independent trait vectors
- Identity emerges from **positioning** within social field
- Constraining one agent reconfigures entire relational architecture

**Evidence**: Network topology effects in Phase 1
- When low-salience agents received grit constraints, ALL agents (including high-salience) reduced engagement
- G seed 6: Urban Progressive (no grit) went from 0.375 → 0.167 engagement
- Proves agents are **relationally constituted**, not independent individuals

### Simmel's Social Forms

**Key concept**: Social forms emerge from and shape interaction patterns.

From Simmel (1908):
> "Society exists where a number of individuals enter into interaction... The unity of these interactions is the society."

**Operationalization in émile-GCE**:
- **Semiotic regimes** (ENGAGED_HARMONY, PRODUCTIVE_DISSONANCE) are Simmelian social forms
- Forms aren't imposed; they **emerge** from agent interactions
- Architecture provides constraints that make certain forms more/less likely

**Evidence**: Regime detection
- ENGAGED_HARMONY observed consistently in Condition G (optimal architecture)
- Regime stability correlates with low hyper-enfranchisement
- Forms persist across rounds, shaping subsequent interaction

### Architecture as Social Aesthetics

**Claim**: Architectural design doesn't just constrain behavior - it **constitutes the social field** agents inhabit.

**Evidence from Phase 1**:
- **2.6× difference** between G (dual-LLM + adaptive) and H (single-LLM + adaptive)
- Dual-LLM effect is **context-dependent**: helps with adaptive (-0.067), hurts with progressive (+0.081)
- This isn't about "better prompts" - it's about how architecture shapes **relational possibilities**

## Émile QSE Mechanics

### Quality-Space-Enaction (QSE)

From the émile framework, agents have **quality spaces** (Q) that structure **enactions** (E), producing emergent **symbolic dynamics** (S).

**Core insight**: Identity isn't a static tag - it's a **dynamic accumulation** of enactments.

### Surplus (S)

**Definition**: Identity-as-accumulated-enactment. In émile-GCE, surplus is **qualitative capacity** rather than simple accumulation.

**Implementation** (`agents/identity_core/core.py`):

```python
# Qualitative surplus formula
local_surplus = delta_I * f_tau * f_natality * f_recognition

# Where:
#   delta_I = magnitude of identity change
#   f_tau = 0.5 + 0.5 * tau_normalized (emergent time modulation)
#   f_natality = 0.5 + 0.5 * natality_t (capacity for new beginnings)
#   f_recognition = 0.5 + 0.5 * recognition_ema (social validation)

# EMA update (smoothed)
surplus = (1 - beta) * surplus + beta * local_surplus
```

**Interpretation**:
- High surplus → agent has enacted meaningful identity expressions with social recognition
- Low surplus → agent is exploratory or unrecognized
- Surplus modulates expression capacity: `soft_cap = base_cap * f_salience * f_natality`

### Symbolic Tension (σ)

**Definition**: Gap between identity vector and enacted behavior.

```python
σ_t = |I_t - B_t|
```

Where:
- `I_t`: Identity vector (engagement, faith, friction, ...)
- `B_t`: Behavioral vector (actual network position, messages, ...)

**Operationalization**:
- Compare **Prior** (CES-derived identity) to **Posterior** (observed behavior)
- G seed 2 Disengaged Renter: Prior engagement ~0.17, Posterior 0.267 → σ = 0.097
- G seed 6 with grit: Prior 0.17, Posterior 0.0 → σ = 0.17 (HIGHER tension)

**Implication**: Grit v1 **increased** symbolic tension (forced withdrawal vs identity), not reduced it.

### Rupture

**Definition**: When |σ| exceeds threshold → identity collapse.

**Condition**:
```python
if σ_t > θ_rupture:
    identity_rupture()  # Shed old identity, form new one
```

**Operationalization in émile-GCE**:
- Grit v1 caused rupture: Disengaged Renter engagement 0.267 → 0.0 (not calibration, annihilation)
- Phase 2 goal: **Calibrated tension** that allows identity adjustment without rupture

**Mortality connection**: Rupture → incoherence death (identity shattered beyond repair).

### Emergent Time (τ)

**Definition**: Social clock that compresses/expands based on magnitude of identity change.

**Formula** (from Émile):
```python
delta = mean |σ_t - σ_{t-1}|
τ = TAU_MIN + (TAU_MAX - TAU_MIN) / (1 + exp(K * (delta - THETA)))
```

**Interpretation**:
- Small Δσ → long τ → "slow time" (identity-place relation stable)
- Large Δσ → short τ → time "thickens" (rapid identity change)

**Extension in émile-GCE**: Make σ **place-specific**:
```python
delta_place = mean |σ_place(t) - σ_place(t-1)|
τ_place = logistic(delta_place)
```

**Application**:
- Compute τ per agent per round
- Test if ENGAGED_HARMONY has characteristic τ signatures (stable, slow)
- Test if regime transitions show τ compression (rapid change)

## Identity Coherence (IMPLEMENTED)

### Definition

**Coherence** measures how well identity **predicts behavior** vs how much identity is **overwritten by social field**.

```python
C_t = cos(I_t, I_0) × TE(I→B) / (TE(I→B) + TE(others→B))
```

Where:
- `cos(I_t, I_0)`: How much identity has drifted from initial state (7D vector)
- `TE(I→B)`: Transfer entropy from identity to behavior (predictive power)
- `TE(others→B)`: Transfer entropy from others' behavior to this agent's behavior

**Implementation**: `agents/identity_core/transfer_entropy.py` + `core.py:compute_coherence()`

### Components

**Directional Stability**: `cos(I_t, I_0)`

- Cosine similarity between current and initial identity vector
- High (→1): Identity direction preserved
- Low (→0): Identity has rotated significantly

**Identity-to-Behavior Transfer Entropy**: `TE(I→B)`

```python
TE(I→B) = H(B_{t+1} | history) - H(B_{t+1} | history, I_t)
```

- How much does knowing identity reduce uncertainty about future behavior?
- High TE(I→B) → identity **drives** behavior
- Low TE(I→B) → behavior is random/externally driven

**Field-to-Identity Transfer Entropy**: `TE(others→I)`

```python
TE(others→I) = H(I_{t+1} | history) - H(I_{t+1} | history, social_field_t)
```

- How much does social field reduce uncertainty about future identity?
- High TE(others→I) → identity is being **overwritten**
- Low TE(others→I) → identity is autonomous

### Interpretation

| Coherence | cos(I,I₀) | TE(I→B) | TE(others→I) | Interpretation |
|-----------|-----------|---------|--------------|----------------|
| **High** | High | High | Low | Identity stable, drives behavior, resists field |
| **Medium** | Medium | Medium | Medium | Identity adjusting, mutual influence |
| **Low** | Low | Low | High | Identity shattered, overwritten by field |

### Application in Phase 2

**Use coherence to modulate temperature**:

```python
T_t = T_base + k_r * rupture + k_c * (1 - coherence) + k_n * natality
```

- High coherence → low T → stable voice
- Low coherence → high T → exploratory, variable prose
- Rupture → high T → frantic exploration

**Mortality trigger**: `coherence < θ_coherence` for k consecutive steps → incoherence death.

## Temperature Modulation

### Rationale

Phase 1 identified **two-layer LLM architecture**:
1. **Computational layer** (network position): Architecture CAN modulate ✓
2. **Affective layer** (prose style): RLHF "helpfulness gravity" resists ✗

**Problem**: Grit v1 worked at computational layer (engagement→0) but affective layer stayed verbose (~2,200 chars).

**Solution**: Dynamic temperature based on identity state.

### Formula

```python
T_t = T_base + k_r * R_t + k_c * (1 - C_t) + k_n * N_t
```

Where:
- `T_base`: Baseline temperature (e.g., 0.7)
- `R_t`: Rupture signal (binary or continuous)
- `C_t`: Coherence (0-1)
- `N_t`: Natality signal (0-1, z-score normalized)
- `k_r, k_c, k_n`: Scaling coefficients

### Dimension-Specific Temperature

**Extension**: Apply different T to different issue dimensions.

```python
T_issue_i = T_base + k_s * salience_i + k_c * (1 - coherence_i)
```

- High-salience dimensions → lower T ("you cannot endorse X")
- Low-salience dimensions → higher T (freedom to explore)

**Example**: Disengaged Renter
- Low T on federal budget (don't care, brief responses)
- Moderate T on housing/rent (occasional engaged bursts)

### Implementation (COMPLETE)

**Location**: `agents/identity_core/core.py:get_temperature()`

```python
def get_temperature(self, base_temp: float = 0.7) -> float:
    """T = T_base + k_r*rupture + k_c*(1-coherence) + k_n*natality"""
    k_r, k_c, k_n = 0.15, 0.10, 0.05

    rupture = 1.0 if self.detect_rupture() else 0.0
    coherence = self.compute_coherence()

    T = base_temp + k_r*rupture + k_c*(1 - coherence) + k_n*self.natality_t
    return np.clip(T, 0.3, 1.2)
```

This temperature is passed to the Performer LLM for each agent. Integration is in `social_rl/runner.py`.

## Natality: Relative to Pace of Change

### Problem with Absolute Thresholds

Émile originally used: `if |ΔP| > θ: natality = True`

**Issue**: What counts as "significant change" for a newborn vs mature agent?
- Newborn: Small ΔP feels like birth
- Mature: Same ΔP is white noise

### Relative Natality (z-score)

**Formula**:
```python
z_{i,t} = (ΔP_{i,t} - μ_i(t)) / (σ_i(t) + ε)
N_{i,t} = sigmoid(k · z_{i,t})
```

Where:
- `μ_i(t)`: Mean ΔP for agent i over history
- `σ_i(t)`: Std dev of ΔP for agent i
- `z_{i,t}`: How many standard deviations is current change?
- `sigmoid(k·z)`: Smooth 0-1 natality signal

### Interpretation

| z-score | Natality | Interpretation |
|---------|----------|----------------|
| z < -1 | Low (~0) | Less change than usual (stagnation) |
| z ≈ 0 | Medium (~0.5) | Normal variation |
| z > 2 | High (~1) | Significant change (birth event) |

**For newborn agents**:
- μ, σ tiny → small ΔP gives big z → high N
- Small iterations feel like births

**For mature agents**:
- μ, σ reflect past exploration → same ΔP gives low z
- Only major shifts register as natality

### Connection to Factorial Findings

**Hypothesis**: G's lower hyper-enfranchisement creates differential natality pressures:
- Low-salience agents: low baseline μ → engagement spikes register as high N (natality events)
- High-salience agents: higher baseline μ → engagement stays stable

**Test in Phase 2**: Run per-round vector extraction, compute z-scores, correlate with regime stability.

## Coach as Social Convention / Relational Conscience

### Reframing the Coach

**Original**: Coach validates/challenges agent positions.

**Phase 2**: Coach is a **convention field** C(t) that agents couple to.

### Strain

```python
strain_t = |I_t - C_t|
```

Where:
- `I_t`: Agent's identity vector
- `C_t`: Social convention (Coach's current "policy")

**C_t updates** based on social field:
- Majority consensus → C_t shifts toward consensus
- Polarization → C_t reflects tension

### Two Modes

**Socially Induced Neuroticism** (high strain, coach pushes alignment):
- Chronic identity dissonance
- High rupture, frequent high T but constrained
- Prose: "Tons of qualifiers, apologies, self-contradictions"

**Socially Induced Empowerment** (low strain, coach validates):
- Amplifies identity coherence and energy
- T can lower (no need for frantic exploration)
- Prose: Confident, stable voice

### Why G < H for Hyper-Enfranchisement

**With dual-LLM (G)**:
- Coach detects strain between I and enacted behavior
- For Disengaged Renter (low salience), coach doesn't push when strain is low
- Coach validates withdrawal/brief statements
- **Result**: Engagement stays closer to CES expected (0.256)

**With single-LLM (H)**:
- No separate conscience layer
- Performer self-validates everything toward RLHF "helpfulness"
- **Result**: Hyper-enfranchisement (0.667)

### Relational Deviance Tracking

Coach as "mirror of the field":

**Prompts**:
> "You're the only one voicing this concern about rural health care. How does it feel to carry that alone? Do you want to push harder or reframe it?"

vs

> "Notice that you've moved much closer to the majority's position than where you started. Does this still feel true to your own experience?"

This is **relational deviance** (agent vs field), not static rule enforcement.

### Implementation (Phase 2)

```python
class CoachField:
    def __init__(self):
        self.C_t = None  # Current convention vector

    def update(self, agents, round_data):
        """Update C_t based on social field."""
        # Compute centroid of agent identities (weighted by salience?)
        self.C_t = weighted_centroid([a.identity for a in agents])

    def compute_strain(self, agent):
        """Compute strain for single agent."""
        return np.linalg.norm(agent.identity - self.C_t)

    def generate_feedback(self, agent, strain):
        """Generate coach prompt based on strain."""
        if strain > threshold_high:
            return neuroticism_prompt(agent, self.C_t)
        else:
            return empowerment_prompt(agent)
```

## Mortality and Repopulation

### Death Conditions (Non-Arbitrary)

**Energy Death**: Repeated high |ΔI| + low validation → energy < threshold

```python
energy_t = energy_{t-1} - cost(|ΔI_t|) + reward(validation_t)
if energy_t < θ_energy: death_by_exhaustion()
```

**Incoherence Death**: `coherence < θ_coherence` for k consecutive steps

```python
if all(coherence_history[-k:] < θ_coherence): death_by_incoherence()
```

**Silencing Death**: `engagement ≈ 0` and `institutional_faith` high → "hollow compliance"

```python
if engagement < 0.05 and institutional_faith > 0.8 for k rounds:
    death_by_silencing()  # Present but not alive
```

### Evidence: G6 Grit Results

**Disengaged Renter** in G seed 6 (grit v1):
- Engagement: 0.0
- Institutional faith: 1.0
- Message length: ~2,274 (stayed verbose)

**Interpretation**: This is silencing death - agent is present but not alive.

### Exit Types

Making mortality explicit lets us distinguish:

**Alienated Exit**: "This isn't for people like me"
- Identity coherent, refuses assimilation
- Energy low but coherence high
- Explicit departure statement

**Silencing Death**: Quiet compliance for N rounds → removal
- Identity ruptured, hollow shell remains
- Energy low, coherence low
- No explicit departure

**Satisfied Exit**: High coherence, low strain, consensus achieved
- Identity aligned with C_t
- Energy stable, coherence high
- Graceful departure

### Repopulation

**Method**: CES-grounded children with perturbed vectors

```python
def repopulate(parent_agent, CES_data):
    # Find CES centroid near parent
    centroid = find_ces_centroid(parent_agent.demographics)

    # Perturb slightly
    child_vector = centroid + noise(σ=0.1)

    # Create new agent
    return Agent(identity=child_vector, surplus=0, energy=1.0)
```

**Design choices**:
- Inherit demographics from parent region/cluster
- Start with low surplus (newborn)
- Full energy
- Low initial μ, σ (so small ΔP feels like natality)

### Multi-Generation Experiments (Phase 3)

**Research questions**:
- Do lineages converge over generations?
- Does G produce different evolutionary patterns than H?
- Can PRODUCTIVE_DISSONANCE be sustained across mortality/natality?

---

See [Identity Grounding](identity_grounding.md) for Phase 2 implementation plan.
