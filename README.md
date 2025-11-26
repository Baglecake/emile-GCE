# émile-GCE: Architecting Social Aesthetics in Agentic Systems

**Generative Computational Ethnography for studying emergent social patterns in multi-agent LLM systems.**

Part of the [émile](https://github.com/delcoburn/emile) series (emergent-interactive learner).

## Core Finding

**Architecture matters**: In a 2×2×2 factorial analysis of architectural configurations (challenge mode × context type × LLM architecture), we found a **2.6× difference** in agent participation levels between best (Condition G: dual-LLM + adaptive context) and worst (Condition H: single-LLM + adaptive context) configurations.

When low-salience agents were architecturally constrained, **all agents** reduced engagement - including high-salience agents who received no constraints. This demonstrates **Weber's positioned associations**: agents don't just have traits, they "form a part of and stand within" a social community. Architecture reconfigures the entire relational field.

## What is GCE?

**Generative Computational Ethnography** is a methodological framework for studying social systems through multi-agent LLM simulations:

- **Generative**: Agent behavior emerges from identity cores and social positioning, not hand-coded scripts
- **Computational**: Quantifiable metrics (engagement, coherence, transfer entropy) grounded in survey data
- **Ethnography**: Focus on emergent patterns, relational dynamics, and regime formation

Unlike traditional social simulation (which tests hypotheses about individuals), GCE studies how architectural constraints shape the **social field** agents inhabit.

## Research Program

### Phase 1: Architecture Optimization (Complete ✅)

- Implemented 2×2×2 architectural sweep (8 conditions, multiple seeds)
- Discovered **hyper-enfranchisement**: Low-salience agents showing engagement 4-5× higher than CES baseline
- **Key finding**: Condition G (dual-LLM + adaptive context + challenge mode) is optimal
- See: [FACTORIAL_ANALYSIS.md](FACTORIAL_ANALYSIS.md)

### Phase 2: Identity-Grounding (Stages 1-4 Complete)

**Implemented:**
- **IdentityCore** with full QSE mechanics (surplus, rupture, emergent time tau)
- **Stateful natality**: tau-based baseline with recognition-driven modulation
- **Qualitative surplus**: `local_surplus = delta_I * f_tau * f_natality * f_recognition` (EMA-smoothed)
- **SurplusTrace buffer**: Memory of enacted surplus events with decay/revalorization
- **Identity blending**: `I_new = I_current + eta * T` (weighted trace direction)
- **Expression capacity**: `soft_cap = base_cap * f_salience * f_natality` (identity-grounded)
- **Temperature modulation**: `T = T_base + k_r*rupture + k_c*(1-coherence) + k_n*natality`
- **TRUE dual-LLM**: Separate 14B Performer + 7B Coach on distinct GPU endpoints

**Remaining (Phase 2b):**
- Transfer entropy: TE(I->B) vs TE(others->I) for coherence formula
- Multi-wave CES priors for empirical delta_mu/sigma
- Mortality mechanics (energy death, incoherence death, silencing death)

### Phase 3: Sociogeographic Embodiment (Future ⏭️)

- Tie-to-place via SociogeographicBody
- Affordance validation (context confirms/disconfirms identity)
- Coach as social convention field (C_t) agents couple to
- Dynamic roster: exit mechanisms (alienated, satisfied) and new entrants

## Theoretical Framework

### Social Aesthetics (Weber, Simmel)

How social forms emerge from and shape interaction patterns. Architecture doesn't just constrain behavior - it **constitutes the social field** agents inhabit.

Key concept: **Positioned associations** - agents "form a part of and stand within" a social community. They don't have fixed traits; they occupy relational positions.

### Émile QSE Mechanics

Quality-space-enaction patterns for identity coherence:

- **Surplus (S)**: Identity as accumulated enactment
- **Symbolic Tension (σ)**: Gap between identity vector and behavior
- **Rupture**: When |σ| exceeds threshold → identity collapse
- **Emergent Time (τ)**: Social clock from magnitude of change

### Two-Layer LLM Architecture

- **Computational Layer** (network position): References + responses + initiative
- **Affective Layer** (prose style): Message length, politeness, helpfulness markers
- Architecture CAN modulate computational layer; affective layer resists via RLHF "helpfulness gravity"

## Data: Canadian Election Survey (CES) 2021

All agents are grounded in real voter profiles from the [CES 2021](https://[cces.gov.harvard.edu/](http://www.ces-eec.ca/2021-canadian-election-study/)) survey (N=61,000):

- **Identity vectors**: Engagement, institutional faith, social friction (derived from turnout, salience, trust measures)
- **Issue positions**: 12 policy dimensions from survey responses
- **Demographics**: Age, education, region, party ID

This grounding prevents "strawman agents" and ensures findings reflect real political diversity.

## Quick Start

### Installation

```bash
git clone https://github.com/delcoburn/emile-gce.git
cd emile-gce
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running Experiments

```bash
# Run with TRUE dual-LLM (14B Performer + 7B Coach on separate GPUs)
python3 experiments/run_ces_experiment.py \
  --performer-url "https://your-14b-endpoint/v1" \
  --performer-model "Qwen/Qwen2.5-14B-Instruct" \
  --coach-url "https://your-7b-endpoint/v1" \
  --coach-model "Qwen/Qwen2.5-7B-Instruct" \
  --condition G --seed 2 --rounds 3

# Run Condition G with single vLLM endpoint
python3 experiments/run_ces_experiment.py \
  --provider vllm \
  --base-url "https://your-endpoint/v1" \
  --model "Qwen/Qwen2.5-14B-Instruct" \
  --condition G --seed 2 --rounds 3

# Extract identity vectors from results
python3 analysis/extract_identity_vectors.py outputs/G_seed2_fixed/

# Full factorial analysis across all conditions
python3 analysis/analyze_full_sweep.py
```

### Configuration

Create a `.env` file with your endpoints:

```bash
# RunPod API key
RUNPOD_API_KEY=your-key-here

# Performer endpoint (14B model)
PERFORMER_BASE_URL=https://your-14b-endpoint/v1
PERFORMER_MODEL=Qwen/Qwen2.5-14B-Instruct

# Coach endpoint (7B model)
COACH_BASE_URL=https://your-7b-endpoint/v1
COACH_MODEL=Qwen/Qwen2.5-7B-Instruct
```

## Key Results

**Hyper-Enfranchisement**: Low-salience agents (expected engagement ~0.17) show 2-5× higher participation across most architectures.

**Optimal Architecture**: Condition G (dual-LLM + adaptive context + challenge mode) minimizes this effect:

- G: 0.256 avg engagement (+50% deviation)
- H: 0.667 avg engagement (+292% deviation)

**Network Topology Effects**: When low-salience agents were constrained (grit v1), ALL agents reduced engagement - proving relational constitution.

See [FACTORIAL_ANALYSIS.md](FACTORIAL_ANALYSIS.md) for complete findings.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for system design, 2×2×2 sweep details, and component documentation.

## Documentation

- [Theoretical Foundations](docs/theoretical_foundations.md): Weber, Simmel, Social Aesthetics, QSE mechanics
- [Identity Grounding](docs/identity_grounding.md): Phase 2 implementation plan
- [Research Roadmap](notes/research_roadmap.md): Completed work and next steps

## Citation

```bibtex
@software{emile-gce,
  title = {émile-GCE: Architecting Social Aesthetics in Agentic Systems},
  author = {Coburn, Delmar},
  year = {2025},
  url = {https://github.com/delcoburn/emile-gce}
}
```

## Related Work

- **émile-core**: [github.com/Baglecake/emile-mini](https://github.com/Baglecake/emile-mini) - Emergent-interactive learner framework
- **Social Aesthetics**: Weber (1922), Simmel (1908) - Social forms as emergent from interaction
- **Computational Ethnography**: Boellstorff (2015), Geiger (2017) - Studying digital social systems

## License

MIT License - See LICENSE file

## Contact

Delmar Coburn - [GitHub](https://github.com/delcoburn)

---

**Status**: Phase 1 complete. Phase 2 Stages 1-4 complete (IdentityCore, natality, surplus, traces, expression capacity). Phase 2b (transfer entropy, mortality) in progress.
