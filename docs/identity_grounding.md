# Identity-Grounding Implementation Plan (Phase 2)

## Goal

Reduce Condition G's **residual +50% hyper-enfranchisement** (0.256 vs CES expected 0.17) to CES-accurate levels through identity-grounding interventions.

## Status

- **Phase 1**: COMPLETE ✅ - Architecture optimization (G is optimal)
- **Phase 2**: IN PROGRESS 🔄 - Identity-grounding
- **Phase 3**: PLANNED ⏭️ - Sociogeographic embodiment

## CES-Based Identity Artifacts

- `data/identity/identity_weights_2021.v0.json`
  Auto-generated mapping from CES variables → identity dimensions (prototype).

- `analysis/identity/compute_identity_group_means_2021.py`
  Script to compute per-respondent identity vectors and group-level priors.

- `data/identity/identity_group_means_2021.csv`
  Empirical group-level means of identity dimensions (Region × rural/urban × household).

## Phase 2 Roadmap

### Immediate: Per-Round Vector Extraction

**Current limitation**: `extract_identity_vectors.py` works per-experiment (avg across rounds).

**Need**: Per-round extraction to compute:
- `ΔI` between rounds
- Emergent time `τ` from magnitude of change
- Coherence trajectories `cos(I_t, I_0)`
- Natality intensity `z_{i,t}`

**Implementation**:

```python
# extract_identity_vectors.py (modified)
def extract_vectors_per_round(exp_dir: Path) -> Dict:
    """Extract identity vectors for each round separately."""
    rounds = load_rounds(exp_dir)

    vectors_by_round = {}
    for r, round_data in enumerate(rounds):
        vectors_by_round[r] = {
            agent_id: {
                'engagement': calc_engagement(agent_id, round_data),
                'institutional_faith': calc_faith(agent_id, round_data),
                'social_friction': calc_friction(agent_id, round_data),
            }
            for agent_id in get_agent_ids(round_data)
        }

    return vectors_by_round
```

**Deliverable**: `outputs/G_seed2_fixed/identity_vectors_per_round.json`

### Short-term: IdentityCore Class

**Location**: `agents/identity_core/core.py`

**Design**:

```python
from dataclasses import dataclass
from collections import deque
import numpy as np

@dataclass
class IdentityCore:
    """Dynamic identity core with émile QSE mechanics."""

    # Core identity vector
    vector: np.ndarray  # (engagement, faith, friction, tie_to_place, salience, ...)

    # QSE mechanics
    surplus: float = 0.0           # Accumulated enactment
    sigma: float = 0.0             # Symbolic tension |I - B|
    tau_emergent: float = 1.0      # Emergent time (default: normal pace)

    # Coherence tracking
    initial_vector: np.ndarray = None  # I_0 for cos(I_t, I_0)
    coherence: float = 1.0         # cos(I_t, I_0) × TE ratio

    # Energy / lifecycle
    energy: float = 1.0            # Drains with dissonance, recovers with validation

    # Temperature modulation
    temperature: float = 0.7       # T_base + k_r*rupture + k_c*(1-coherence) + k_n*natality

    # History for natality z-scores
    history: deque = None          # Recent ΔI values

    def __post_init__(self):
        if self.initial_vector is None:
            self.initial_vector = self.vector.copy()
        if self.history is None:
            self.history = deque(maxlen=10)  # Keep last 10 rounds

    def update_from_behavior(self, behavior_vector: np.ndarray, validation: float):
        """Update identity based on enacted behavior and validation received."""
        # 1. Compute symbolic tension
        self.sigma = np.linalg.norm(self.vector - behavior_vector)

        # 2. Update surplus (accumulated enactment)
        self.surplus += 0.1 * self.sigma  # More tension → more enactment

        # 3. Update identity vector (drift toward behavior, modulated by tension)
        drift_rate = 0.05 if self.sigma < 0.15 else 0.02  # Less drift when ruptured
        self.vector += drift_rate * (behavior_vector - self.vector)

        # 4. Update energy
        dissonance_cost = 0.1 * self.sigma
        validation_reward = 0.15 * validation
        self.energy = max(0, self.energy - dissonance_cost + validation_reward)

        # 5. Track ΔI for natality
        delta_I = np.linalg.norm(self.vector - self.initial_vector)
        self.history.append(delta_I)

        # 6. Update emergent time (τ)
        self.tau_emergent = self._compute_emergent_time()

        # 7. Update coherence (cos + TE ratio, requires round_data)
        # NOTE: TE computation needs full round context, done separately
        self.coherence = self._compute_directional_stability()

        # 8. Update temperature
        self.temperature = self._compute_temperature()

    def _compute_emergent_time(self) -> float:
        """Emergent time from magnitude of recent identity change."""
        if len(self.history) < 2:
            return 1.0

        delta = np.mean(np.abs(np.diff(list(self.history))))
        TAU_MIN, TAU_MAX = 0.5, 2.0
        K, THETA = 5.0, 0.1

        tau = TAU_MIN + (TAU_MAX - TAU_MIN) / (1 + np.exp(K * (delta - THETA)))
        return tau

    def _compute_directional_stability(self) -> float:
        """Cosine similarity between I_t and I_0."""
        cos_sim = np.dot(self.vector, self.initial_vector) / (
            np.linalg.norm(self.vector) * np.linalg.norm(self.initial_vector) + 1e-6
        )
        return max(0, cos_sim)  # Clamp to [0, 1]

    def _compute_temperature(self) -> float:
        """Dynamic temperature based on rupture, coherence, natality."""
        T_base = 0.7
        k_r, k_c, k_n = 0.3, 0.2, 0.1

        rupture = 1.0 if self.sigma > 0.15 else 0.0
        natality = self._compute_natality_z_score()

        T = T_base + k_r*rupture + k_c*(1 - self.coherence) + k_n*natality
        return np.clip(T, 0.2, 1.2)

    def _compute_natality_z_score(self) -> float:
        """Natality signal from z-score of recent change."""
        if len(self.history) < 3:
            return 0.5  # Default for new agents

        recent_deltas = np.diff(list(self.history))
        mu = np.mean(recent_deltas)
        sigma = np.std(recent_deltas) + 1e-6

        current_delta = self.history[-1] - self.history[-2] if len(self.history) >= 2 else 0
        z = (current_delta - mu) / sigma

        # Sigmoid activation
        natality = 1 / (1 + np.exp(-2 * z))
        return natality

    def detect_rupture(self, threshold: float = 0.15) -> bool:
        """Check if identity has ruptured."""
        return self.sigma > threshold

    def detect_death(self) -> tuple[bool, str]:
        """Check death conditions. Returns (is_dead, death_type)."""
        # Energy death
        if self.energy < 0.1:
            return True, "energy_death"

        # Incoherence death (coherence low for k consecutive steps)
        # NOTE: Needs coherence history tracking
        if self.coherence < 0.2:
            return True, "incoherence_death"

        # Silencing death (checked externally: engagement ≈ 0 + high faith)

        return False, None
```

**Integration with runner**:

```python
# social_rl/runner.py (modified)
class Agent:
    def __init__(self, ces_profile, ...):
        self.ces_profile = ces_profile

        # Initialize identity core
        self.identity_core = IdentityCore(
            vector=extract_ces_identity_vector(ces_profile),
            surplus=0.0,
            energy=1.0
        )

    def after_round(self, round_data, validation):
        """Update identity core after each round."""
        behavior_vector = extract_behavior_vector(self.id, round_data)
        self.identity_core.update_from_behavior(behavior_vector, validation)

        # Check death
        is_dead, death_type = self.identity_core.detect_death()
        if is_dead:
            self.handle_death(death_type)
```

**Deliverable**: Working IdentityCore class integrated with runner.

### Medium-term: Transfer Entropy

**Location**: `analysis/compute_transfer_entropy.py`

**Goal**: Compute `TE(I→B)` and `TE(others→I)` for coherence formula.

**Formulas**:

```
TE(I→B) = H(B_{t+1} | history) - H(B_{t+1} | history, I_t)
TE(others→I) = H(I_{t+1} | history) - H(I_{t+1} | history, social_field_t)
```

**Simplifications for Phase 2**:

Use **mutual information** as proxy:

```python
def compute_TE_proxy(identity_series, behavior_series):
    """Simplified TE using mutual information."""
    # Discretize continuous vectors
    I_binned = discretize(identity_series, bins=5)
    B_binned = discretize(behavior_series, bins=5)

    # Mutual information I(I_t; B_{t+1})
    MI = mutual_info_score(I_binned[:-1], B_binned[1:])

    return MI
```

**Full implementation** (Phase 3):

Use `jpype` + `JIDT` (Java Information Dynamics Toolkit):

```python
import jpype
from jpype import JPackage

def compute_TE_full(source, target, k=1, tau=1):
    """Full TE using JIDT."""
    teCalc = JPackage('infodynamics.measures.discrete').TransferEntropyCalculatorDiscrete(256, k)
    teCalc.initialise()
    teCalc.addObservations(source, target)
    te = teCalc.computeAverageLocalOfObservations()
    return te
```

**Deliverable**: `compute_transfer_entropy.py` with proxy and full implementations.

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

## Timeline

| Milestone | Estimated Effort | Status |
|-----------|------------------|--------|
| Per-round vector extraction | 1-2 days | ⏭️ Next |
| IdentityCore class (stub) | 1 day | ⏭️ Next |
| Transfer entropy (proxy) | 2-3 days | 🔄 Planned |
| Grit v2 + G seed 7 | 2-3 days | 🔄 Planned |
| Identity salience experiments | 3-4 days | 🔄 Planned |
| CoachField class | 4-5 days | ⏭️ Future |
| Mortality mechanics | 5-7 days | ⏭️ Future |
| Multi-generation experiments | 7-10 days | ⏭️ Future |

**Total Phase 2 estimate**: 4-6 weeks (intermittent work)

## Next Steps

1. ✅ **DONE**: Migration to emile-gce repo
2. **IMMEDIATE**: Implement per-round vector extraction
3. **THIS WEEK**: Stub IdentityCore class + integration
4. **NEXT WEEK**: Grit v2 design + G seed 7 experiment

---

See [Research Roadmap](../notes/research_roadmap.md) for broader context.
