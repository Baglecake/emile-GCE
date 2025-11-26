# Research Roadmap

## Completed ✅

### Phase 1: Architecture Optimization

**Goal**: Identify optimal architectural configuration for minimizing hyper-enfranchisement.

**Completed work**:

1. **2×2×2 Factorial Sweep** (2025-11-22 to 2025-11-24)
   - Ran all 8 conditions (A-H) with multiple seeds (2-3 per condition)
   - 705 experimental output files (configs, round data, regime logs)
   - Full sweep analysis across ~50 experiments

2. **Empirical Vector Extraction** (2025-11-24)
   - Implemented `extract_identity_vectors.py` based on CES profiles
   - Extracts: engagement, institutional_faith, social_friction
   - Per-experiment and comparative modes (e.g., G2 vs G6)

3. **Factorial Analysis** (2025-11-24)
   - Created `analyze_full_sweep.py` for full sweep analysis
   - **MAJOR FINDING**: Condition G is optimal (rank #1/8)
     - G: 0.256 avg engagement (+50% deviation from CES expected 0.17)
     - H: 0.667 avg engagement (+292% deviation) - WORST
     - **2.6× difference** proves dual-LLM + adaptive context critical

4. **Grit Constraint Experiments** (2025-11-23)
   - Implemented identity-aware grit constraints for low-salience agents
   - G seed 6 with grit: Disengaged Renter 0.000 engagement (over-corrected)
   - Discovered network topology effects (all agents withdrew)

5. **Comprehensive Documentation** (2025-11-24)
   - Created `FACTORIAL_ANALYSIS_HYPER_ENFRANCHISEMENT.md` (12 sections)
   - Documented regime patterns (ENGAGED_HARMONY stability in G)
   - Identified two-layer LLM architecture (computational vs affective)

**Key findings**:

- **G is optimal architecture**: Dual-LLM + adaptive context + challenge mode
- **Dual-LLM effect is context-dependent**: Helps with adaptive (-0.067), hurts with progressive (+0.081)
- **Network topology effects**: Weber's "positioned associations" operationalized
- **Architectural limits**: Even optimal G shows +50% residual → identity-grounding needed

**Deliverables**:

- [FACTORIAL_ANALYSIS.md](../FACTORIAL_ANALYSIS.md): Full sweep results
- `outputs/`: All experimental data (A-H, seeds 2-3)
- `analysis/extract_identity_vectors.py`: Vector extraction tool
- `analysis/analyze_full_sweep.py`: Factorial analysis tool
- `outputs/G_series_vector_analysis.md`: G seeds 2-6 comprehensive analysis
- `outputs/vector_comparison_G2_vs_G6.json`: Grit effectiveness comparison

---

## Completed (Phase 2a)

### Phase 2: Identity-Grounding (Stages 1-4 Complete)

**Goal**: Reduce G's residual +50% hyper-enfranchisement to CES-accurate levels (~0.17) through identity-grounding interventions.

**Completed work**:

1. **IdentityCore Class** (COMPLETE)
   - Location: `agents/identity_core/core.py` (791 lines)
   - Full QSE mechanics: surplus, sigma, tau, coherence, natality
   - Dynamic temperature: T_base + k_r*rupture + k_c*(1-coherence) + k_n*natality

2. **Emergent Time (tau)** (COMPLETE)
   - Location: `agents/identity_core/tau.py`
   - `tau = logistic(|I_t - I_0|)`
   - High delta -> compressed time, low delta -> normal flow

3. **Stateful Natality** (COMPLETE)
   - Tau-based baseline: `baseline = 0.3 + 0.5 * tau_normalized`
   - Recognition-driven modulation (boost when recognized)
   - Overshoot suppression (suppress when above engagement target)

4. **Qualitative Surplus + SurplusTrace** (COMPLETE)
   - `local_surplus = delta_I * f_tau * f_natality * f_recognition`
   - SurplusTrace dataclass with decay/revalorization
   - Identity blending: `I_new = I_current + eta * T`

5. **Expression Capacity** (COMPLETE)
   - Location: `social_rl/runner.py`
   - `soft_cap = base_cap * f_salience * f_natality`
   - Identity-grounded token limits

6. **TRUE Dual-LLM** (COMPLETE)
   - Location: `social_rl/dual_llm_client.py`
   - `create_true_dual_llm()` for separate 14B Performer + 7B Coach
   - Distinct GPU endpoints

7. **Grit v2: Tiered Constraints** (COMPLETE)
   - Location: `agents/ces_generators/grit_config.py`
   - Levels: NONE/LIGHT/MODERATE/STRONG
   - CES-calibrated targets
   - Dynamic calibration based on overshoot

8. **Per-Round Identity Logging** (COMPLETE)
   - Integrated with runner.py
   - Logs natality, surplus, coherence, traces per turn

**Deliverables**:
- Full IdentityCore implementation with all QSE mechanics
- Expression capacity formula integrated with runner
- TRUE dual-LLM with separate model endpoints
- Grit v2 tiered constraint system

---

## Current (Phase 2b)

### Phase 2b: Remaining Identity-Grounding

**In progress**:

1. **Transfer Entropy Implementation**
   - Location: `analysis/compute_transfer_entropy.py` (planned)
   - Compute TE(I->B): How much identity predicts behavior
   - Compute TE(others->I): How much social field overwrites identity
   - Full coherence formula: cos(I_t, I_0) × TE(I->B) / (TE(I->B)+TE(others->I))
   - **Status**: Phase 2b (MI proxy to start)

2. **Multi-Wave CES Priors**
   - Load empirical delta_mu/sigma per demographic group
   - Enable z-score natality relative to CES baselines
   - **Status**: Phase 2b

3. **Tie-to-Place Integration**
   - Integrate riding-level geographic attachment
   - Place-specific tau computation
   - **Status**: Phase 2b

---

## Future ⏭️

### Phase 3: Sociogeographic Embodiment

**Goal**: Full émile-style embodiment with mortality/natality, tie-to-place, and dynamic roster.

**Components**:

1. **Coach as Convention Field**
   - Location: `identity_core/coach_field.py`
   - C(t): Social convention agents couple to
   - Strain: |I_t - C_t|
   - Two modes: Socially induced neuroticism (high strain) vs empowerment (low strain)
   - Relational deviance tracking ("you vs others" not "right vs wrong")

2. **Mortality Mechanics**
   - Location: `identity_core/life_cycle.py`
   - **Energy death**: Repeated high |ΔI| + low validation → energy < threshold
   - **Incoherence death**: coherence < threshold for k consecutive steps
   - **Silencing death**: engagement ≈ 0 + high institutional_faith → hollow compliance
   - Exit types: Alienated, silencing, satisfied

3. **Natality / Repopulation**
   - Relative z-score natality: (ΔP - μ) / σ (not arbitrary thresholds)
   - CES-grounded children: Sample near deceased agent's demographic cluster
   - Perturb vectors slightly (σ=0.1)
   - Newborns: surplus=0, energy=1.0, low μ/σ (small ΔP feels like birth)

4. **SociogeographicBody**
   - Tie-to-place metrics from CES (region, urbanicity, tenure)
   - Affordance validation: Context confirms/disconfirms identity
   - Place-specific emergent time: τ_place from |Δσ_place|

5. **Dynamic Roster**
   - Exit mechanisms: Alienated ("not for people like me"), satisfied (consensus achieved)
   - Entry mechanisms: Repopulate on death, add agents when convergence too high
   - Multi-generation experiments

**Planned experiments**:

1. **Multi-Generation Lineages**
   - Run 10-round experiment with mortality/natality enabled
   - Track lineage evolution (do descendants converge?)
   - Compare G vs H evolutionary patterns

2. **PRODUCTIVE_DISSONANCE Sustainability**
   - Test if identity-grounding sustains dissonance across generations
   - High-salience agents with existential stakes in positions
   - Coach as relational deviance tracker (not rule enforcer)

3. **Place-Based Identity**
   - Rural vs urban agents with tie-to-place
   - Test if place-grounding prevents convergence
   - Affordance validation (context confirms/disconfirms identity)

**Deliverables** (planned):

- CoachField class integrated with runner
- LifeCycle class (mortality + repopulation)
- SociogeographicBody implementation
- Multi-generation experimental results
- Publication: "Generative Computational Ethnography: Phase 3"

**Timeline**: 8-12 weeks (after Phase 2 complete)

---

## Publication Strategy

### Paper I: Architecture Shapes Positioning

**Status**: Ready to draft (Phase 1 complete)

**Title**: "Architecture Shapes Positioning: A 2×2×2 Factorial Analysis of Multi-Agent LLM Systems"

**Claims**:
1. Architecture matters: 2.6× difference between conditions (G vs H)
2. Dual-LLM effect is context-dependent (interaction effect)
3. Agents are relationally constituted (network topology effects)
4. Architectural limits exist (even optimal G shows +50% residual)

**Evidence**:
- Full factorial sweep (8 conditions, multiple seeds)
- Empirical identity vectors grounded in CES
- Weber's positioned associations operationalized
- Two-layer LLM architecture (computational vs affective)

**Target venues**: CHI, FAccT, Computational Social Science

### Paper II: Identity-Grounding via QSE

**Status**: Phase 2 in progress

**Title**: "Identity-Grounding in Multi-Agent Systems: Émile QSE Mechanics for CES-Accurate Positioning"

**Claims**:
1. Identity cores with surplus, tension, rupture enable CES accuracy
2. Transfer entropy measures identity-behavior coupling
3. Temperature modulation targets affective layer
4. Calibrated constraints achieve CES-accurate engagement

**Evidence** (planned):
- Grit v2 experiments (G seed 7 achieving ~0.17)
- Identity salience preventing convergence
- Coherence trajectories showing stability
- Transfer entropy validating identity→behavior causality

**Target venues**: NeurIPS, ICLR (agent foundations track), AAMAS

### Paper III: Sociogeographic Embodiment

**Status**: Phase 3 planned

**Title**: "Generative Computational Ethnography: Multi-Generation Dynamics in Embodied Agent Systems"

**Claims**:
1. Mortality/natality creates living demographic-semiotic ecology
2. Coach as convention field induces neuroticism/empowerment
3. Tie-to-place sustains PRODUCTIVE_DISSONANCE
4. Lineage evolution shows emergent social patterns

**Evidence** (planned):
- Multi-generation experiments
- Lineage convergence analysis
- PRODUCTIVE_DISSONANCE sustainability
- Place-based identity validation

**Target venues**: Sociological Methods & Research, Computational Culture, ICWSM

---

## Theoretical Integration

### Weber's Social Aesthetics

**Core concept**: Agents "form a part of and stand within" social community.

**Operationalization**:
- Phase 1: Network topology effects (constraining one agent reconfigures all)
- Phase 2: Identity cores (agents as positioned associations, not individuals)
- Phase 3: Coach as convention field (C_t that agents couple to)

### Émile QSE Mechanics

**Core concept**: Identity as accumulated enactment with surplus, tension, rupture.

**Operationalization**:
- Phase 1: Vector gap (Prior vs Posterior)
- Phase 2: IdentityCore class (S, σ, τ, coherence)
- Phase 3: Full embodiment (mortality, natality, place-grounding)

### Generative Computational Ethnography

**Core concept**: Study emergent social patterns through multi-agent LLM systems.

**Operationalization**:
- Phase 1: Architecture shapes social field (not just individual behavior)
- Phase 2: Identity-grounding enables CES accuracy
- Phase 3: Living demographic-semiotic ecology (not frozen archetypes)

**Distinction from social simulation**:
- Not testing hypotheses about individuals
- Studying how architectural constraints shape **relational possibilities**
- Emergent regimes (ENGAGED_HARMONY) are Simmelian social forms

---

## Open Questions

### Phase 2

1. Can grit v2 achieve CES-accurate engagement (~0.17) without over-correcting?
2. Does high identity salience prevent convergence → PRODUCTIVE_DISSONANCE?
3. Is transfer entropy proxy (MI) sufficient or do we need full JIDT?
4. How much does temperature modulation affect affective layer (prose length)?

### Phase 3

1. Do lineages converge over generations, or sustain diversity?
2. Does G produce different evolutionary patterns than H?
3. Can PRODUCTIVE_DISSONANCE be sustained with mortality/natality?
4. How does tie-to-place interact with identity salience?

### Methodological

1. What is the right level of CES fidelity? (Individual vs cluster sampling)
2. Should we use GPT-4, Claude, or open-source models for generalizability?
3. How do we validate that regimes (ENGAGED_HARMONY) aren't artifacts?
4. What counts as "CES-accurate" when CES itself has measurement error?

---

## Next Actions

### Completed (Phase 2a)

- [x] Implement per-round vector extraction
- [x] Create IdentityCore class (full implementation, 791 lines)
- [x] Integrate IdentityCore with runner
- [x] Implement emergent time (tau)
- [x] Implement stateful natality with tau-based baseline
- [x] Implement qualitative surplus + SurplusTrace buffer
- [x] Implement expression capacity formula
- [x] Implement temperature modulation
- [x] Implement TRUE dual-LLM (14B/7B)
- [x] Implement Grit v2 tiered constraints
- [x] Per-round identity state logging

### Phase 2b (Current)

- [ ] Implement transfer entropy (MI proxy)
- [ ] Load multi-wave CES priors for delta_mu/sigma
- [ ] Integrate tie-to-place metrics

### Phase 3 (Future)

- [ ] Coach as convention field C(t)
- [ ] Mortality mechanics
- [ ] Field vector F_t extraction
- [ ] Multi-generation experiments
- [ ] Draft Paper I outline

---

See [Identity Grounding](../docs/identity_grounding.md) for detailed implementation documentation.
