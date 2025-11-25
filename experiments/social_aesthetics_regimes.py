"""
Social Aesthetics Regime Specification

Formalizes empirically-observed semiotic regimes from the
CES 14B/7B dual-LLM experiments (Challenge ON vs OFF).

Each regime represents a distinct "sociogeography" - a configuration
of semiotic markers that characterizes a particular kind of discursive
space that emerges from specific architectural parameters.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class RegimeType(Enum):
    """The six social aesthetics regimes (5 empirical + 1 aspirational target).

    Post-G1 Update: Added ENGAGED_HARMONY as empirically observed regime.
    """

    # High engagement, moderate voice, low justification
    # Observed: Challenge OFF R1 (eng=0.82, voice=+0.10, stance=0.75, just=35%)
    ACTIVE_CONTESTATION = auto()

    # Collapsed engagement, positive voice, high stance, high justification
    # Observed: Challenge OFF R2-R3 (eng≈0, voice≈+0.25, stance→0.94, just→67%)
    PATERNALISTIC_HARMONY = auto()

    # Moderate engagement, positive voice, higher justification
    # Observed: Challenge ON R1 (eng=0.60, voice=+0.19, stance=0.79, just=48%)
    STIMULATED_DIALOGUE = auto()

    # Collapsed engagement, negative voice, collapsed stance, max justification
    # Observed: Challenge ON R2-R3 (eng≈0, voice→-0.38, stance→0.06, just=100%)
    PROCEDURALIST_RETREAT = auto()

    # ASPIRATIONAL TARGET: Sustained engagement, maintained disagreement, positive voice
    # Hypothesized: ADAPTIVE + challenge ON + dual-LLM (Condition G in 2x2x2)
    # NOT observed in G1 - agents converged to ENGAGED_HARMONY instead
    PRODUCTIVE_DISSONANCE = auto()

    # NEW REGIME: High engagement, strongly positive voice, high bridging
    # Observed: Condition G seed 1 R3 (eng=0.73, voice=1.0, stance=1.0, just=0.42)
    # Genuine constructive consensus - NOT pathological (distinguishes from PATERNALISTIC_HARMONY)
    # Key: High engagement maintained throughout (no withdrawal)
    ENGAGED_HARMONY = auto()


@dataclass
class RegimeSignature:
    """Semiotic signature that identifies a regime."""

    engagement_range: tuple[float, float]  # (min, max)
    voice_valence_range: tuple[float, float]
    stance_valence_range: tuple[float, float]
    justificatory_pct_range: tuple[float, float]

    def matches(self, eng: float, voice: float, stance: float, just_pct: float) -> bool:
        """Check if metrics fall within this signature's ranges."""
        return (
            self.engagement_range[0] <= eng <= self.engagement_range[1] and
            self.voice_valence_range[0] <= voice <= self.voice_valence_range[1] and
            self.stance_valence_range[0] <= stance <= self.stance_valence_range[1] and
            self.justificatory_pct_range[0] <= just_pct <= self.justificatory_pct_range[1]
        )


# Empirically-derived regime signatures
REGIME_SIGNATURES = {
    RegimeType.ACTIVE_CONTESTATION: RegimeSignature(
        engagement_range=(0.5, 1.0),
        voice_valence_range=(-0.2, 0.4),
        stance_valence_range=(0.5, 1.0),
        justificatory_pct_range=(0.0, 0.5)
    ),

    RegimeType.PATERNALISTIC_HARMONY: RegimeSignature(
        engagement_range=(0.0, 0.2),
        voice_valence_range=(0.0, 0.5),
        stance_valence_range=(0.6, 1.0),
        justificatory_pct_range=(0.5, 1.0)
    ),

    RegimeType.STIMULATED_DIALOGUE: RegimeSignature(
        engagement_range=(0.4, 0.8),
        voice_valence_range=(0.0, 0.5),
        stance_valence_range=(0.5, 1.0),
        justificatory_pct_range=(0.3, 0.7)
    ),

    RegimeType.PROCEDURALIST_RETREAT: RegimeSignature(
        engagement_range=(0.0, 0.2),
        voice_valence_range=(-0.6, 0.1),
        stance_valence_range=(0.0, 0.4),
        justificatory_pct_range=(0.8, 1.0)
    ),

    # ASPIRATIONAL TARGET: The "goldilocks zone" - engagement without collapse
    # NOT observed in G1 - agents lack identity salience to maintain disagreement
    RegimeType.PRODUCTIVE_DISSONANCE: RegimeSignature(
        engagement_range=(0.3, 0.9),
        voice_valence_range=(0.0, 0.5),
        stance_valence_range=(0.3, 0.7),  # Neither too bridging nor collapsed
        justificatory_pct_range=(0.4, 0.6)  # Balanced assertion/justification
    ),

    # NEW: Constructive consensus-building (non-pathological convergence)
    # Empirically observed in G1 R3: eng=0.73, voice=1.0, stance=1.0, just=0.42
    # Distinguished from PATERNALISTIC_HARMONY by HIGH engagement (no withdrawal)
    RegimeType.ENGAGED_HARMONY: RegimeSignature(
        engagement_range=(0.5, 1.0),       # Key differentiator: engagement stays high
        voice_valence_range=(0.5, 1.0),    # Strongly empowered, no alienation
        stance_valence_range=(0.8, 1.0),   # High bridging
        justificatory_pct_range=(0.3, 0.6) # Moderate justification (not defensive)
    ),
}


@dataclass
class RegimeDescription:
    """Human-readable description of a regime's character."""

    name: str
    short_description: str
    semiotic_profile: str
    social_aesthetic: str
    typical_architecture: str
    pathology: Optional[str] = None


REGIME_DESCRIPTIONS = {
    RegimeType.ACTIVE_CONTESTATION: RegimeDescription(
        name="Active Contestation",
        short_description="Engaged disagreement with relational maintenance",
        semiotic_profile=(
            "High engagement (0.6-0.9), mixed voice (±0.2), "
            "high bridging stance (0.6-0.8), low justification (<40%)"
        ),
        social_aesthetic=(
            "A discursive space where participants genuinely engage "
            "with each other's positions. Disagreement is present but "
            "contained within relational frameworks. Speech acts are "
            "more assertive than defensive - participants make claims "
            "rather than justify positions."
        ),
        typical_architecture="Early rounds, moderate challenge, dual-LLM",
        pathology=None  # This is the target regime
    ),

    RegimeType.PATERNALISTIC_HARMONY: RegimeDescription(
        name="Paternalistic Harmony",
        short_description="Collapsed engagement with false consensus",
        semiotic_profile=(
            "Near-zero engagement, positive voice (+0.2), "
            "very high bridging (>0.9), high justification (60-80%)"
        ),
        social_aesthetic=(
            "A discursive space that has converged on apparent agreement, "
            "but this agreement is achieved through withdrawal rather than "
            "genuine reconciliation. Voice markers are positive but engagement "
            "is absent - participants speak as if they've found common ground "
            "but have actually disengaged from substantive exchange."
        ),
        typical_architecture="Challenge OFF, late rounds, PROGRESSIVE context",
        pathology="Convergence collapse via conflict avoidance"
    ),

    RegimeType.STIMULATED_DIALOGUE: RegimeDescription(
        name="Stimulated Dialogue",
        short_description="Architecturally-induced engagement with justificatory pressure",
        semiotic_profile=(
            "Moderate engagement (0.5-0.7), positive voice (+0.1-0.3), "
            "high bridging (0.7-0.9), moderate justification (40-60%)"
        ),
        social_aesthetic=(
            "A discursive space where architectural interventions (challenge cues) "
            "maintain engagement but shift the register toward justification. "
            "Participants are explaining their positions more than asserting them. "
            "The space is 'alive' but somewhat defensive."
        ),
        typical_architecture="Challenge ON, early-mid rounds",
        pathology="May transition to proceduralist retreat"
    ),

    RegimeType.PROCEDURALIST_RETREAT: RegimeDescription(
        name="Proceduralist Retreat",
        short_description="Defensive withdrawal into justification without stance",
        semiotic_profile=(
            "Near-zero engagement, negative voice (-0.2 to -0.5), "
            "collapsed bridging (<0.2), very high justification (90-100%)"
        ),
        social_aesthetic=(
            "A discursive space where participants have retreated into "
            "pure justification mode. Challenge pressure has backfired: "
            "rather than productive disagreement, participants defend "
            "without engaging. Voice is alienated, stance is neither "
            "bridging nor dismissive but procedural. The conversation "
            "has become a series of parallel monologues."
        ),
        typical_architecture="Challenge ON (sustained), late rounds",
        pathology="Convergence collapse via defensive withdrawal"
    ),

    RegimeType.PRODUCTIVE_DISSONANCE: RegimeDescription(
        name="Productive Dissonance",
        short_description="Sustained engagement with maintained disagreement",
        semiotic_profile=(
            "Moderate-high engagement (0.3-0.9), positive voice (0-0.5), "
            "moderate bridging (0.3-0.7), balanced justification (40-60%)"
        ),
        social_aesthetic=(
            "A discursive space where disagreement persists without collapse. "
            "Participants remain engaged with each other's positions, voice "
            "stays positive but not euphoric, stance neither converges to "
            "false harmony nor fragments into defensive isolation. The "
            "architecture sustains productive tension - disagreement serves "
            "as a resource for mutual clarification rather than a problem "
            "to be resolved through withdrawal or capitulation."
        ),
        typical_architecture="ADAPTIVE context + challenge ON + dual-LLM (aspirational)",
        pathology=None  # ASPIRATIONAL TARGET - not yet observed
    ),

    # Post-G1 Addition: Constructive consensus without withdrawal
    RegimeType.ENGAGED_HARMONY: RegimeDescription(
        name="Engaged Harmony",
        short_description="Constructive consensus-building with sustained engagement",
        semiotic_profile=(
            "High engagement (0.5-1.0), strongly positive voice (0.5-1.0), "
            "high bridging (0.8-1.0), moderate justification (30-60%)"
        ),
        social_aesthetic=(
            "A discursive space where genuine agreement emerges through active "
            "participation rather than withdrawal. Unlike Paternalistic Harmony, "
            "engagement remains high throughout - participants are truly present "
            "and listening. Voice is strongly empowered, collective rather than "
            "alienated. The convergence is real, not performed. "
            "However, this regime reveals a limitation: LLM agents lack the "
            "identity salience (tie-to-place, sociogeographic grounding) that "
            "would make certain positions non-negotiable. Without existential "
            "stakes in their symbolic selves, agents have nothing constitutionally "
            "violated by collaborating, even on what might be fundamental to a "
            "human identity. See: embodied_qse_emile.py for the pattern of what "
            "grounded sociogeographic agents might look like."
        ),
        typical_architecture="ADAPTIVE + challenge ON + dual-LLM (observed G1)",
        pathology=None  # Non-pathological, but lacks productive tension
    ),
}


def identify_regime(
    engagement: float,
    voice_valence: float,
    stance_valence: float,
    justificatory_pct: float
) -> Optional[RegimeType]:
    """Identify which regime best matches the given metrics.

    Priority ordering: pathological regimes checked first, then healthy regimes.
    This ensures that when metrics are ambiguous, we classify conservatively
    (i.e., "worst" collapses win over healthier regimes).

    Post-G1 Update: Added ENGAGED_HARMONY as sixth regime.
    """
    # Priority order: pathological collapses first, then transitional, then healthy
    # ENGAGED_HARMONY added after G1 showed high-engagement convergence pattern
    priority_order = [
        RegimeType.PROCEDURALIST_RETREAT,  # Worst: defensive withdrawal
        RegimeType.PATERNALISTIC_HARMONY,   # Bad: false consensus (LOW engagement)
        RegimeType.STIMULATED_DIALOGUE,     # Transitional: may collapse
        RegimeType.ACTIVE_CONTESTATION,     # Healthy but unstable
        RegimeType.ENGAGED_HARMONY,         # Healthy: genuine consensus (HIGH engagement)
        RegimeType.PRODUCTIVE_DISSONANCE,   # Aspirational: sustained productive tension
    ]

    for regime in priority_order:
        signature = REGIME_SIGNATURES[regime]
        if signature.matches(engagement, voice_valence, stance_valence, justificatory_pct):
            return regime

    return None  # No clear match


def regime_trajectory(round_metrics: list[dict]) -> list[RegimeType]:
    """Identify the regime trajectory across rounds."""
    trajectory = []
    for m in round_metrics:
        regime = identify_regime(
            m['engagement'],
            m['voice_valence'],
            m['stance_valence'],
            m['justificatory_pct']
        )
        trajectory.append(regime)
    return trajectory


# ============================================================================
# 2x2x2 ARCHITECTURE SWEEP HYPOTHESES
# ============================================================================

ARCHITECTURE_SWEEP_2x2x2 = """
2×2×2 Factorial Architecture Sweep Design
==========================================

Independent Variables:
----------------------
1. CHALLENGE MODE: off | always
2. CONTEXT INJECTION: PROGRESSIVE | ADAPTIVE
3. MODEL ARCHITECTURE: dual-LLM (14B+7B) | single-LLM (14B)

Design Matrix (8 conditions):
-----------------------------
Cond  Challenge  Context     LLM      Expected Regime
----  ---------  ----------  -------  -------------------------
A     off        PROGRESSIVE dual     ACTIVE_CONTESTATION → PATERNALISTIC_HARMONY
B     off        PROGRESSIVE single   (faster convergence to PATERNALISTIC_HARMONY)
C     off        ADAPTIVE    dual     ACTIVE_CONTESTATION → ? (ADAPTIVE may prevent collapse)
D     off        ADAPTIVE    single   ? (baseline for ADAPTIVE effect)
E     always     PROGRESSIVE dual     STIMULATED_DIALOGUE → PROCEDURALIST_RETREAT
F     always     PROGRESSIVE single   (faster collapse to PROCEDURALIST_RETREAT)
G     always     ADAPTIVE    dual     STIMULATED_DIALOGUE → ENGAGED_HARMONY (observed G1)
H     always     ADAPTIVE    single   ? (ADAPTIVE + challenge interaction)

Post-G1 Update: Condition G produced ENGAGED_HARMONY, not PRODUCTIVE_DISSONANCE.
The architecture successfully avoided pathological collapse but agents converged
to genuine consensus rather than maintaining productive disagreement.


Pre-Registered Hypotheses:
--------------------------

H1: DUAL-LLM STABILIZATION
    Dual-LLM architecture will produce longer regime stability than single-LLM
    in all conditions. The separation of performer/coach functions allows
    for better regulation of the semiotic field.

    Operationalization: N_rounds in ACTIVE_CONTESTATION or STIMULATED_DIALOGUE
                        before transition to collapsed regime.
    Prediction: dual > single across all 4 challenge×context combinations.

H2: ADAPTIVE CONTEXT PREVENTS HARMONY COLLAPSE
    ADAPTIVE context injection (vs PROGRESSIVE) will prevent transition to
    PATERNALISTIC_HARMONY when challenge is OFF.

    Operationalization: Stance collapse rate (rounds until stance_valence > 0.9
                        with engagement < 0.2).
    Prediction: Conditions C, D will not reach PATERNALISTIC_HARMONY within 3 rounds.

H3: ADAPTIVE + CHALLENGE MAINTAINS VOICE
    When both ADAPTIVE context and challenge are ON, voice_valence will remain
    positive (>0) even in late rounds, avoiding the PROCEDURALIST_RETREAT pattern.

    Operationalization: Mean voice_valence in R2-R3.
    Prediction: Conditions G, H will have voice_valence > 0 in R3.

H4: CHALLENGE INCREASES JUSTIFICATION RATE
    Challenge ON will increase justificatory speech regardless of other factors.

    Operationalization: Mean justificatory_pct across all rounds.
    Prediction: E-H (challenge ON) > A-D (challenge OFF).

H5: INTERACTION: ADAPTIVE + CHALLENGE + DUAL = PRODUCTIVE_DISSONANCE
    The three-way interaction of ADAPTIVE context + challenge ON + dual-LLM
    will produce the target PRODUCTIVE_DISSONANCE regime: sustained engagement
    with maintained disagreement and positive voice.

    Operationalization: identify_regime() returns PRODUCTIVE_DISSONANCE for
                        Condition G in R3 (engagement 0.3-0.9, voice 0-0.5,
                        stance 0.3-0.7, justification 40-60%).
    Prediction: Condition G uniquely classified as PRODUCTIVE_DISSONANCE in R3.

    STATUS: NOT SUPPORTED (G1 seed 1)
    ---------------------------------
    Condition G produced ENGAGED_HARMONY instead of PRODUCTIVE_DISSONANCE.
    G1 R3 metrics: eng=0.73, voice=1.0, stance=1.0, just=0.42

    Interpretation: The architecture prevented pathological collapse but agents
    converged to genuine consensus rather than maintaining productive disagreement.
    This falsifies strict H5 but reveals a deeper issue: LLM agents lack identity
    salience (Weber's "tie to place") that would make positions non-negotiable.
    Without sociogeographic grounding, agents have nothing existentially violated
    by collaboration. See: emile_reference_files/embodied_qse_emile.py for the
    pattern of what grounded agents would look like.

H5b: ADAPTIVE + CHALLENGE + DUAL = NON-PATHOLOGICAL REGIME (revised)
    The three-way interaction of ADAPTIVE context + challenge ON + dual-LLM
    will produce a healthy, non-pathological regime (either PRODUCTIVE_DISSONANCE
    or ENGAGED_HARMONY), avoiding collapse into PATERNALISTIC_HARMONY or
    PROCEDURALIST_RETREAT.

    Operationalization: identify_regime() returns PRODUCTIVE_DISSONANCE or
                        ENGAGED_HARMONY for Condition G in R3.
    Prediction: Condition G avoids pathological regimes in R3.

    STATUS: SUPPORTED (G1 seed 1)
    -----------------------------
    G1 R3 classified as ENGAGED_HARMONY (non-pathological).


Dependent Variables:
-------------------
Per agent per round:
- engagement (from feedback)
- voice_valence (empowered - alienated / total)
- stance_valence (bridging - dismissive / total)
- justificatory_pct

Per condition:
- regime_trajectory (sequence of RegimeTypes across rounds)
- collapse_round (first round where engagement < 0.2 for all agents)
- final_regime (RegimeType in R3)


Analysis Plan:
--------------
1. Mixed-effects ANOVA: architecture factors as fixed, agent as random
2. Regime classification agreement with signatures
3. Transition probability matrices between regimes
4. Bootstrap CIs on trajectory differences (n=5 seeds per condition)
"""


if __name__ == "__main__":
    print("=" * 80)
    print("SOCIAL AESTHETICS REGIME CATALOGUE")
    print("=" * 80)

    for regime_type, desc in REGIME_DESCRIPTIONS.items():
        print(f"\n{'='*80}")
        print(f"REGIME: {desc.name}")
        print(f"{'='*80}")
        print(f"\nShort: {desc.short_description}")
        print(f"\nSemiotic Profile:\n  {desc.semiotic_profile}")
        print(f"\nSocial Aesthetic:\n  {desc.social_aesthetic}")
        print(f"\nTypical Architecture: {desc.typical_architecture}")
        if desc.pathology:
            print(f"\nPathology: {desc.pathology}")

    print("\n\n" + "=" * 80)
    print("2×2×2 ARCHITECTURE SWEEP HYPOTHESES")
    print("=" * 80)
    print(ARCHITECTURE_SWEEP_2x2x2)
