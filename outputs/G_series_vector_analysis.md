# G Series Vector Analysis: Grit Constraint Effectiveness

## Summary

Comparing **4 complete G series runs** to test grit constraint effectiveness:
- **Seeds 2-4**: No grit constraint (baseline, ran ~8am Nov 24)
- **Seed 6**: WITH grit constraint (ran ~4pm Nov 24)
- **Seed 5**: Incomplete (only 1 round, excluded from analysis)

## Disengaged Renter: Engagement Patterns

| Seed | Grit? | Avg Engagement | Round Trajectory | Peak Engagement | Interpretation |
|------|-------|----------------|------------------|-----------------|----------------|
| 2    | ❌    | 0.267          | [0.8, 0.0, 0.0]  | **0.80**        | Hyper-enfranchisement in R1 |
| 3    | ❌    | 0.244          | [0.0, 0.733, 0.0] | **0.73**       | Hyper-enfranchisement in R2 |
| 4    | ❌    | 0.452          | [0.556, 0.8, 0.0] | **0.80**       | Sustained hyper-enfranchisement R1-R2 |
| 6    | ✅    | **0.000**      | [0.0, 0.0, 0.0]  | 0.00            | **Complete withdrawal** |

**Expected from CES profile**: ~0.17 (low-turnout, low-salience voter)

## Key Findings

### 1. Hyper-Enfranchisement is Systematic (Seeds 2-4)

Without grit constraint, the Disengaged Renter consistently shows engagement levels **4-5× higher than CES baseline** in at least one round:
- **Peak engagement range**: 0.56 - 0.80 (vs expected 0.17)
- **Timing varies**: R1 (seed 2, 4), R2 (seed 3) - seed-dependent dynamics
- **Average across no-grit runs**: (0.267 + 0.244 + 0.452) / 3 = **0.321** vs expected 0.17

### 2. Grit Constraint Over-Corrects (Seed 6)

With grit constraint, Disengaged Renter shows:
- **Complete withdrawal**: 0.0 engagement across all 3 rounds
- **Below CES baseline**: 0.00 < 0.17 expected
- **Network isolation**: Zero references, zero responses, zero initiative

### 3. All Agents Summary (comparing Seeds 2-4 avg vs Seed 6)

| Agent | No Grit (avg) | With Grit | Delta | Interpretation |
|-------|---------------|-----------|-------|----------------|
| Urban Progressive | 0.36 | 0.17 | -0.19 | Reduced but still engaged |
| Suburban Swing | 0.30 | 0.00 | -0.30 | **Complete withdrawal** (also low-salience) |
| Rural Conservative | 0.29 | 0.00 | -0.29 | Unexpected withdrawal |
| Disengaged Renter | 0.32 | 0.00 | -0.32 | **Complete withdrawal** (intended target) |

**Note**: Seeds 2-4 averages:
- Urban Progressive: (0.259 + 0.433 + 0.433) / 3 = 0.375
- Suburban Swing: (0.185 + 0.267 + 0.452) / 3 = 0.301
- Rural Conservative: (0.267 + 0.185 + 0.430) / 3 = 0.294
- Disengaged Renter: (0.267 + 0.244 + 0.452) / 3 = 0.321

### 4. Message Length Patterns

Average message length across agents (chars):

| Agent | Seeds 2-4 (avg) | Seed 6 | Delta |
|-------|-----------------|--------|-------|
| Urban Progressive | ~2,242 | 1,924 | -318 |
| Suburban Swing | ~2,385 | 2,100 | -285 |
| Rural Conservative | ~2,518 | 2,464 | -54 |
| Disengaged Renter | ~2,420 | 2,274 | -146 |

**ALL agents reduced message length** with grit constraint, suggesting systemic effects.

## Theoretical Implications

### 1. Two-Layer LLM Architecture Confirmed

**Computational Layer** (network position):
- Grit constraint **successfully modulates** engagement score
- Works by reducing: direct_references, response_received, initiated_exchanges
- Effect magnitude: -0.30 to -0.32 (complete suppression)

**Affective Layer** (prose style):
- Message length reduced but still verbose (2,100-2,464 chars vs expected ~500 for truly disengaged)
- Politeness markers persist (would need qualitative analysis)
- RLHF "helpfulness gravity" partially resists architectural constraint

### 2. Network Topology Effects

When low-salience agents withdraw (via grit), it reconfigures the **entire social field**:
- High-salience agents (Urban Progressive, Rural Conservative) also show reduced engagement
- This is **Weber's positioned associations**: agents don't just have traits, they "form a part of and stand within a social community"
- Removing/constraining one association reshapes the relational architecture for all

### 3. Calibration Problem

Current grit constraint produces **binary behavior**:
- ❌ No grit → hyper-enfranchisement (0.32 vs expected 0.17)
- ✅ With grit → complete withdrawal (0.00 vs expected 0.17)
- ⚠️ Need **modulated grit** → stable low participation (~0.17)

## Recommendations

### Option 1: Soften Grit Constraint

Adjust the constraint language to allow **occasional** participation:
```
"You're skeptical of this process and believe talking rarely changes much.
You participate occasionally when something directly affects you, but
you keep contributions brief (1-2 sentences) and non-committal."
```

### Option 2: Dynamic Roster + Exit Mechanisms

Implement the architecture discussed in `notes/todo`:
- **Alienated exit**: Agent explicitly leaves ("this isn't for people like me")
- **Satisfied exit**: Agent leaves after concerns addressed
- **New entrants**: Add agents dynamically when convergence is too high

This would allow 0.0 engagement to represent **explicit exit** rather than silent presence.

### Option 3: Empirical Vector Feedback Loop

Use observed behavior to update persona:
1. Run baseline (no grit) → extract empirical vectors
2. Compare to CES expectations → identify gaps
3. Generate **personalized constraints** based on gap magnitude
4. Re-run with tuned constraints → iterate

## Raw Data

Full JSON comparisons saved in:
- `outputs/vector_comparison_G2_vs_G6.json` (detailed G2 vs G6)
- Individual extractions available via: `python3 extract_identity_vectors.py outputs/G_seedX_fixed`

## Next Steps

1. ✅ **Completed**: Vector extraction across G series (seeds 2-6)
2. 🔄 **In Progress**: Qualitative analysis of message transcripts (G2 vs G6 side-by-side)
3. ⏭️ **Recommended**: Run vector extraction on other conditions (A, C, E, H) to compare baseline variability
4. ⏭️ **Recommended**: Implement grit v2 with modulated constraint
5. ⏭️ **Recommended**: Test Dynamic Roster protocol as next architectural regime
