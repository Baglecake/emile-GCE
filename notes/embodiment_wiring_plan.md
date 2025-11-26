# Embodiment Wiring Plan

## Current State: What Exists

### 1. IdentityCore Energy System
**Location**: `agents/identity_core/core.py`

| Component | Line | Status |
|-----------|------|--------|
| `energy: float = 1.0` | 224 | ✅ Initialized |
| `self.energy -= 0.1` on rupture | 350 | ✅ Decrements |
| `'energy': round(self.energy, 4)` | 958 | ✅ Logged |

**NOT WIRED**:
- No energy recovery mechanism
- No mortality check (energy ≤ 0 → agent dies)
- Energy doesn't affect expression capacity
- Energy doesn't affect temperature

### 2. WorldState Embodied Scalars
**Location**: `social_rl/world_state.py`

| Scalar | Lines | Update Logic | Status |
|--------|-------|--------------|--------|
| `frustration` | 619-626 | EMA: +0.8α on no recognition, →0 on recognition | ✅ Updated |
| `fatigue` | 628-631 | +0.05/round, -0.1 if recognition > 0.5 | ✅ Updated |
| `entrenchment` | 659-661 | +0.1 when challenged | ✅ Updated |

**Runner calls** (`runner.py:751-756`):
```python
if self.world_engine:
    was_recognized = recognition_score > 0.3
    self.world_engine.update_agent_recognition(
        agent_id, was_recognized, recognition_score
    )
```

**NOT WIRED**:
- WorldState scalars NOT read back by runner
- fatigue/frustration/entrenchment don't affect:
  - Expression capacity
  - Temperature
  - Prompt injection

### 3. Expression Capacity Formula
**Location**: `runner.py:559-572`

**Current formula**:
```python
soft_cap = base_cap * f_salience * f_natality * f_temperature
```

Where:
- `f_salience = 0.5 + 0.5 * identity_salience`  # [0.5, 1.0]
- `f_natality = 0.5 + 0.5 * natality_t`         # [0.5, 1.0]
- `f_temperature = identity_core.compute_temperature_capacity_factor()` # [τ, 1.0]

**MISSING factors**:
- `f_energy` - energy depletion → reduced capacity
- `f_fatigue` - fatigue → reduced capacity
- `f_frustration` - frustration → ??? (could increase OR decrease)

---

## Wiring Plan

### Phase A: Energy Recovery + Mortality

#### A1. Add energy recovery on recognition
**File**: `agents/identity_core/core.py`

Add method:
```python
def recover_energy(self, recognition_score: float, recovery_rate: float = 0.05) -> None:
    """
    Recover energy based on recognition.
    Recognition = social existence = metabolic sustenance.
    """
    recovery = recovery_rate * recognition_score
    self.energy = min(1.0, self.energy + recovery)
```

#### A2. Wire energy recovery in runner
**File**: `runner.py` after line 726 (revalorize_traces block)

Add:
```python
# Energy recovery from recognition
core.recover_energy(recognition_score)
```

#### A3. Add mortality check
**File**: `agents/identity_core/core.py`

Add method:
```python
def is_dead(self) -> bool:
    """Check if agent has died from energy depletion."""
    return self.energy <= 0.0
```

#### A4. Wire mortality in runner
**File**: `runner.py` in `_process_agent_turn()`, after identity update

Add check:
```python
if core.is_dead():
    # Mark agent as dead, skip future turns
    self.dead_agents.add(agent_id)
    return None  # or handle death gracefully
```

### Phase B: Energy → Expression Capacity

#### B1. Add energy capacity factor to IdentityCore
**File**: `agents/identity_core/core.py`

Add method:
```python
def compute_energy_capacity_factor(self) -> float:
    """
    Compute expression capacity multiplier from energy.

    Low energy → constrained expression (survival mode).
    f_E = 0.3 + 0.7 * energy  # [0.3, 1.0]

    Floor at 0.3: even depleted agents can speak (death throes).
    """
    return 0.3 + 0.7 * self.energy
```

#### B2. Wire in runner expression capacity
**File**: `runner.py:567-572`

Change:
```python
if agent_id in self.identity_cores:
    f_temperature = self.identity_cores[agent_id].compute_temperature_capacity_factor()
    f_energy = self.identity_cores[agent_id].compute_energy_capacity_factor()
else:
    f_temperature = 0.5
    f_energy = 1.0

soft_cap = int(base_cap * f_salience * f_natality * f_temperature * f_energy)
```

### Phase C: WorldState Scalars → Expression Capacity

#### C1. Read fatigue/frustration from WorldState
**File**: `runner.py` in expression capacity block (~line 560)

Add:
```python
# Get embodied scalars from WorldState
if self.world_engine and agent_id in self.world_engine.agent_states:
    ws_state = self.world_engine.agent_states[agent_id]
    fatigue = ws_state.fatigue
    frustration = ws_state.frustration
else:
    fatigue = 0.0
    frustration = 0.0
```

#### C2. Define fatigue factor
```python
# Fatigue reduces capacity (tired = fewer words)
f_fatigue = 1.0 - 0.3 * fatigue  # [0.7, 1.0]
```

#### C3. Define frustration factor
```python
# Frustration is complex:
# - Low frustration: normal
# - Moderate frustration: MORE words (venting)
# - High frustration: withdrawal (fewer words)
if frustration < 0.3:
    f_frustration = 1.0
elif frustration < 0.7:
    f_frustration = 1.0 + 0.2 * (frustration - 0.3)  # [1.0, 1.08] venting
else:
    f_frustration = 1.08 - 0.4 * (frustration - 0.7)  # [1.08, 0.96] withdrawal
```

#### C4. Update formula
```python
soft_cap = int(base_cap * f_salience * f_natality * f_temperature * f_energy * f_fatigue * f_frustration)
```

### Phase D: Embodied Scalars → Temperature

#### D1. Add frustration to temperature formula
**File**: `agents/identity_core/core.py`, `compute_temperature()` method

Current:
```python
T = T_base + k_rupture*rupture + k_coherence*(1-coherence) + k_natality*natality
```

Add frustration coefficient:
```python
T = T_base + k_rupture*rupture + k_coherence*(1-coherence) + k_natality*natality + k_frustration*frustration
```

Where `k_frustration ≈ 0.15`: frustrated agents are more erratic.

**Problem**: IdentityCore doesn't have access to WorldState frustration.

**Solution**: Pass frustration to `compute_temperature()` or store in IdentityCore.

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           RUNNER.PY                                  │
│                                                                      │
│  ┌──────────────────┐     ┌──────────────────┐                      │
│  │  IdentityCore    │     │   WorldState     │                      │
│  │  per agent       │     │   Engine         │                      │
│  │                  │     │                  │                      │
│  │  energy ────────────────> frustration     │                      │
│  │  surplus         │     │  fatigue         │                      │
│  │  natality        │     │  entrenchment    │                      │
│  │  temperature     │     │                  │                      │
│  │  τ (emergent)    │     │  recognition_hist│                      │
│  └────────┬─────────┘     └────────┬─────────┘                      │
│           │                        │                                 │
│           ▼                        ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │              EXPRESSION CAPACITY FORMULA                 │        │
│  │                                                          │        │
│  │  soft_cap = base_cap                                     │        │
│  │           * f_salience    (identity_salience)            │        │
│  │           * f_natality    (IdentityCore.natality_t)      │        │
│  │           * f_temperature (IdentityCore.compute_temp())  │        │
│  │           * f_energy      (IdentityCore.energy) [NEW]    │        │
│  │           * f_fatigue     (WorldState.fatigue)  [NEW]    │        │
│  │           * f_frustration (WorldState.frustration) [NEW] │        │
│  └─────────────────────────────────────────────────────────┘        │
│                          │                                           │
│                          ▼                                           │
│                    max_tokens                                        │
│                          │                                           │
│                          ▼                                           │
│              ┌───────────────────┐                                   │
│              │  LLM Generation   │                                   │
│              │  (with temp + cap)│                                   │
│              └───────────────────┘                                   │
│                          │                                           │
│                          ▼                                           │
│              ┌───────────────────┐                                   │
│              │  Feedback Extract │                                   │
│              │  recognition_score│                                   │
│              └─────────┬─────────┘                                   │
│                        │                                             │
│           ┌────────────┴────────────┐                                │
│           ▼                         ▼                                │
│  IdentityCore.update()    WorldState.update_recognition()            │
│  - energy recovery [NEW]  - frustration update ✓                     │
│  - rupture check          - fatigue update ✓                         │
│  - surplus/natality       - entrenchment update                      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Order

1. **A1-A2**: Energy recovery (IdentityCore method + runner wiring)
2. **B1-B2**: Energy → capacity factor
3. **C1-C4**: WorldState scalars → capacity factors
4. **A3-A4**: Mortality check (can defer to Phase 3)
5. **D1**: Frustration → temperature (requires passing WorldState data to IdentityCore)

---

## Files to Modify

| File | Changes |
|------|---------|
| `agents/identity_core/core.py` | Add `recover_energy()`, `compute_energy_capacity_factor()`, `is_dead()` |
| `social_rl/runner.py` | Wire energy recovery, read WorldState scalars, update formula |
| `ARCHITECTURE.md` | Update expression capacity documentation |

---

## Questions for User

1. Should frustration increase OR decrease expression capacity? (Plan assumes non-monotonic: venting then withdrawal)
2. What's the mortality threshold? (Plan assumes `energy ≤ 0`)
3. Should dead agents be removed or replaced (natality)?
4. Should entrenchment affect anything? (Currently only tracked, not used)
