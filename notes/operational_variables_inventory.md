# Operational Variables Inventory

Complete inventory of all operational variables in the emile-gce codebase, their wiring status, and data flow.

---

## Summary

| Category | Total | Wired | Orphaned |
|----------|-------|-------|----------|
| Embodiment | 4 | 3 | 1 |
| Identity | 7 | 6 | 1 |
| Temperature | 5 | 5 | 0 |
| Capacity Factors | 6 | 6 | 0 |
| Traces | 4 | 4 | 0 |
| Semiotic | 4 | 3 | 1 |
| CES Profile | 4 | 2 | 2 |
| Config | 5 | 5 | 0 |
| **TOTAL** | **39** | **34** | **5** |

---

## 1. EMBODIMENT VARIABLES

### ✅ energy (WIRED)
- **Location**: `agents/identity_core/core.py:224`
- **Type**: float [0, 1]
- **Updated by**:
  - `-0.1` on rupture (`core.py:350`)
  - `+0.05 × recognition_score` via `recover_energy()` (`core.py:937-951`)
- **Consumed by**:
  - `compute_energy_capacity_factor()` → f_energy (`core.py:953-965`)
  - `is_dead()` → mortality check (`core.py:967-976`)
- **Affects**: Expression capacity (soft_cap), agent mortality

### ✅ frustration (WIRED)
- **Location**: `social_rl/world_state.py:359`
- **Type**: float [0, 1]
- **Updated by**: `WorldState.update_agent_recognition()` - EMA with α=0.3
  - `+0.8α` on no recognition
  - `→0` on recognition
- **Consumed by**:
  - Runner expression capacity → f_frustration (non-monotonic)
  - `IdentityCore.compute_temperature(frustration)` → T
- **Affects**: Expression capacity, LLM temperature

### ✅ fatigue (WIRED)
- **Location**: `social_rl/world_state.py:362`
- **Type**: float [0, 1]
- **Updated by**: `WorldState.update_agent_recognition()`
  - `+0.05` per round
  - `-0.1` if recognition > 0.5
- **Consumed by**: Runner expression capacity → f_fatigue
- **Affects**: Expression capacity (soft_cap)

### ❌ entrenchment (ORPHANED)
- **Location**: `social_rl/world_state.py:365`
- **Type**: float [0, 1]
- **Updated by**: `WorldState.update_agent_recognition()`
  - `+0.1` when challenged (low recognition + high stakes)
- **Consumed by**: NOTHING
- **Affects**: NOTHING (logged only)
- **Potential wiring**: Temperature (entrenched = less exploratory), capacity (entrenched = more verbose)

---

## 2. IDENTITY VARIABLES

### ✅ vector (7D IdentityVector) (WIRED)
- **Location**: `agents/identity_core/core.py:214-279`
- **Dimensions**: care, fairness, loyalty, authority, purity, liberty, efficiency
- **Updated by**: `update_identity()` via recognized content extraction
- **Consumed by**:
  - `compute_coherence()` → temperature
  - `compute_delta_I()` → surplus
  - `compute_tau()` → capacity baseline
  - WorldState event impact calculations
- **Affects**: Temperature, surplus, natality, traces, world events

### ✅ surplus (WIRED)
- **Location**: `agents/identity_core/core.py:223`
- **Type**: float
- **Updated by**: `update_surplus()` = delta_I × f_tau × f_nat × f_rec
- **Consumed by**: `maybe_create_trace()` when surplus > threshold
- **Affects**: Trace creation

### ✅ natality_t (WIRED)
- **Location**: `agents/identity_core/core.py:228`
- **Type**: float [0, 1]
- **Updated by**: `update_natality()` based on recognition/overshoot
- **Consumed by**:
  - Runner expression capacity → f_natality
  - `compute_temperature()` → T
- **Affects**: Expression capacity, temperature

### ✅ rupture_active (WIRED)
- **Location**: `agents/identity_core/core.py:266`
- **Type**: bool
- **Updated by**: Set when delta_I > rupture_threshold
- **Consumed by**:
  - `compute_temperature()` → +k_rupture when active
  - Energy depletion: `-0.1` on rupture
- **Affects**: Temperature, energy

### ✅ _recognition_ema (WIRED)
- **Location**: `agents/identity_core/core.py:251`
- **Type**: float [0, 1]
- **Updated by**: EMA of recognition scores
- **Consumed by**: `update_natality()` for recognition factor
- **Affects**: Natality updates

### ✅ identity_salience (WIRED)
- **Location**: `agents/identity_core/core.py:233`
- **Type**: float [0, 1]
- **Updated by**: Set from CES profile at init
- **Consumed by**: Runner expression capacity → f_salience
- **Affects**: Expression capacity

### ❌ tie_to_place (ORPHANED)
- **Location**: `agents/identity_core/core.py:234`
- **Type**: float [0, 1]
- **Updated by**: Set from CES profile at init
- **Consumed by**: NOTHING
- **Affects**: NOTHING (logged only)
- **Potential wiring**: Response to local events, identity stability

---

## 3. TEMPERATURE VARIABLES

### ✅ T_base (WIRED)
- **Location**: `agents/identity_core/core.py:270`
- **Default**: 0.7
- **Consumed by**: `compute_temperature()` baseline

### ✅ k_rupture (WIRED)
- **Location**: `agents/identity_core/core.py:271`
- **Default**: 0.3
- **Formula**: `T += k_rupture × rupture_signal`

### ✅ k_coherence (WIRED)
- **Location**: `agents/identity_core/core.py:272`
- **Default**: 0.2
- **Formula**: `T += k_coherence × (1 - coherence)`

### ✅ k_natality (WIRED)
- **Location**: `agents/identity_core/core.py:273`
- **Default**: 0.1
- **Formula**: `T += k_natality × natality_t`

### ✅ k_frustration (WIRED)
- **Location**: `agents/identity_core/core.py:274`
- **Default**: 0.15
- **Formula**: `T += k_frustration × frustration`

**Temperature Formula** (complete):
```
T = T_base + k_r*rupture + k_c*(1-coherence) + k_n*natality + k_f*frustration
T = clamp(T, 0.2, 1.2)
```

---

## 4. CAPACITY FACTOR VARIABLES

All computed in `runner.py:589-617`:

### ✅ f_salience
- **Formula**: `0.5 + 0.5 × identity_salience`
- **Range**: [0.5, 1.0]

### ✅ f_natality
- **Formula**: `0.5 + 0.5 × natality_t`
- **Range**: [0.5, 1.0]

### ✅ f_temperature
- **Source**: `IdentityCore.compute_temperature_capacity_factor()`
- **Formula**: `τ + (1-τ) × t_normalized`
- **Range**: [τ, 1.0]

### ✅ f_energy
- **Source**: `IdentityCore.compute_energy_capacity_factor()`
- **Formula**: `0.3 + 0.7 × energy`
- **Range**: [0.3, 1.0]

### ✅ f_fatigue
- **Formula**: `1.0 - 0.3 × fatigue`
- **Range**: [0.7, 1.0]

### ✅ f_frustration (non-monotonic)
- **Formula**:
  - frustration < 0.3: `1.0`
  - 0.3 ≤ frustration < 0.7: `1.0 + 0.2 × (frustration - 0.3)` (venting)
  - frustration ≥ 0.7: `1.08 - 0.4 × (frustration - 0.7)` (withdrawal)
- **Range**: [0.96, 1.08]

**Expression Capacity Formula** (complete):
```
soft_cap = base_cap × f_salience × f_natality × f_temperature × f_energy × f_fatigue × f_frustration
```

---

## 5. TRACE SYSTEM VARIABLES

### ✅ trace_creation_threshold
- **Location**: `agents/identity_core/core.py:259`
- **Default**: 0.1
- **Purpose**: Surplus threshold for trace creation

### ✅ trace_decay_lambda
- **Location**: `agents/identity_core/core.py:258`
- **Default**: 0.05
- **Purpose**: Trace salience decay rate per round

### ✅ trace_revalorize_rho
- **Location**: `agents/identity_core/core.py:260`
- **Default**: 0.2
- **Purpose**: Salience boost on recognition

### ✅ trace_blend_eta
- **Location**: `agents/identity_core/core.py:262`
- **Default**: 0.1
- **Purpose**: Learning rate for trace → identity blending

---

## 6. SEMIOTIC TRACKER VARIABLES

Located in `social_rl/context_injector.py`:

### ✅ _engagement_ema (WIRED)
- **Updated by**: EMA of engagement scores from feedback
- **Consumed by**: Regime detection (harmony_collapse threshold)

### ✅ _stance_ema (WIRED)
- **Updated by**: EMA of stance valence
- **Consumed by**: Regime detection (harmony_collapse threshold)

### ✅ _justification_ema (WIRED)
- **Updated by**: EMA of justification ratio
- **Consumed by**: Regime detection (harmony_collapse, retreat thresholds)

### ⚠️ _voice_ema (PARTIALLY WIRED)
- **Updated by**: EMA of voice valence
- **Consumed by**: Regime detection (retreat threshold)
- **Note**: Used for regime detection but doesn't affect LLM parameters directly

---

## 7. CES PROFILE VARIABLES

### ✅ group_id (WIRED)
- **Location**: `agents/identity_core/core.py:213`
- **Format**: `"{location}_{lean}"` e.g., "urban_left", "rural_right"
- **Consumed by**: `WorldState.compute_agent_impact()` for event valence
- **Affects**: How world events impact agent

### ✅ identity_vector_7d (WIRED → initial_vector)
- **Location**: CES config → IdentityCore.initial_vector
- **Consumed by**: All identity dynamics

### ❌ local_minority_pct (ORPHANED)
- **Location**: `agents/ces_generators/row_to_agent.py:160`
- **Consumed by**: NOTHING
- **Potential wiring**: Frustration accumulation rate

### ❌ riding_competitiveness (ORPHANED)
- **Location**: `agents/ces_generators/row_to_agent.py:159`
- **Consumed by**: NOTHING
- **Potential wiring**: Stakes calculation in WorldState

---

## 8. CONFIG VARIABLES

All in `social_rl/runner.py`:

### ✅ verbosity_penalty_alpha (line 120)
- **Default**: 0.1
- **Purpose**: Rate of verbosity penalty increase

### ✅ verbosity_max_penalty (line 121)
- **Default**: 0.5
- **Purpose**: Maximum verbosity penalty cap

### ✅ token_throttle_factor (line 126)
- **Default**: 1.0
- **Purpose**: Base token throttle multiplier

### ✅ grit_smoothing (line 113)
- **Default**: 0.5
- **Purpose**: Grit word limit smoothing factor

### ✅ world_event_probability (line 132)
- **Default**: 0.3
- **Purpose**: Probability of world events per round

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RECOGNITION SCORE                               │
│                       (extracted from coach feedback)                        │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│   IDENTITYCORE    │    │   IDENTITYCORE    │    │    WORLDSTATE     │
│  update_identity()│    │  recover_energy() │    │ update_agent_rec()│
└─────────┬─────────┘    └─────────┬─────────┘    └─────────┬─────────┘
          │                        │                        │
          ▼                        ▼                        ▼
   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
   │   surplus    │         │    energy    │         │  frustration │
   │   natality_t │         │              │         │   fatigue    │
   │rupture_active│         │              │         │ entrenchment │
   │  coherence   │         │              │         │     (❌)     │
   └──────┬───────┘         └──────┬───────┘         └──────┬───────┘
          │                        │                        │
          └────────────────────────┴────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXPRESSION CAPACITY FORMULA                         │
│                                                                              │
│  soft_cap = base_cap × f_salience × f_natality × f_temperature              │
│                      × f_energy × f_fatigue × f_frustration                 │
│                                                                              │
│  Where:                                                                      │
│    f_salience    = 0.5 + 0.5 × identity_salience           [0.5, 1.0]       │
│    f_natality    = 0.5 + 0.5 × natality_t                  [0.5, 1.0]       │
│    f_temperature = τ + (1-τ) × T_normalized                [τ, 1.0]         │
│    f_energy      = 0.3 + 0.7 × energy                      [0.3, 1.0]       │
│    f_fatigue     = 1.0 - 0.3 × fatigue                     [0.7, 1.0]       │
│    f_frustration = non-monotonic(frustration)              [0.96, 1.08]     │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TEMPERATURE FORMULA                                │
│                                                                              │
│  T = T_base + k_rupture×rupture + k_coherence×(1-coherence)                 │
│            + k_natality×natality + k_frustration×frustration                │
│                                                                              │
│  T = clamp(T, 0.2, 1.2)                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                          ┌────────────────┐
                          │ LLM GENERATION │
                          │  (max_tokens   │
                          │   temperature) │
                          └────────────────┘
```

---

## Orphaned Variables Summary

| Variable | Location | Issue | Potential Wiring |
|----------|----------|-------|------------------|
| entrenchment | WorldState | Updated but never read | Temperature, capacity |
| tie_to_place | IdentityCore | Set but never used | Local event response, identity stability |
| local_minority_pct | CESAgent | Computed but unused | Frustration rate |
| riding_competitiveness | CESAgent | Computed but unused | Stakes calculation |
| _te_ratio | IdentityCore | Computed, logged only | Prompt injection (influence indicator) |

---

## Verification Checklist

- [x] Energy recovery wired in runner
- [x] Energy → capacity factor wired
- [x] Fatigue → capacity factor wired
- [x] Frustration → capacity factor wired
- [x] Frustration → temperature wired
- [x] Mortality check wired
- [x] group_id → WorldState impact wired
- [x] identity_vector_7d → initial_vector wired
- [ ] entrenchment → ??? NOT WIRED
- [ ] tie_to_place → ??? NOT WIRED
- [ ] local_minority_pct → ??? NOT WIRED
- [ ] riding_competitiveness → ??? NOT WIRED

---

*Generated: 2025-11-26*
*Last verified against codebase: commit 4b6c2c2*
