"""
Grit v2: Calibrated Constraints for CES-Accurate Engagement

Problem (Grit v1):
- Disengaged Renter: 0.267 → 0.000 engagement (over-corrected)
- Wanted ~0.17 (CES-accurate)
- V1 was too aggressive: "deeply skeptical", "changes nothing"

Design Principles (Grit v2):
1. Target +50% residual, not full suppression
2. Tiered constraints based on identity_salience
3. Both layers: Computational (initiative) + Affective (length)
4. Temperature modulation integration with IdentityCore

Key Changes from v1:
- "Participate occasionally" (not "rarely/never")
- "1-3 sentences" (specific length target)
- "Don't initiate" (targets initiative_score, not total engagement)
- "It's fine to..." (permits presence without full participation)

Success Metric:
- Disengaged Renter engagement ~0.17-0.20 (not 0.0)
- Network topology preserved (other agents don't withdraw)
"""

from typing import Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass


class GritLevel(Enum):
    """Tiered grit levels based on identity salience."""
    NONE = "none"           # Salience >= 0.5: No constraint
    LIGHT = "light"         # Salience 0.35-0.5: Gentle framing
    MODERATE = "moderate"   # Salience 0.2-0.35: Clear constraints
    STRONG = "strong"       # Salience < 0.2: Full grit (but not V1 extreme)


@dataclass
class GritConstraint:
    """Structured grit constraint for agent prompts."""
    level: GritLevel
    prompt_text: str
    temperature_modifier: float  # Added to base temperature
    max_response_length: str     # Guidance for length
    initiative_guidance: str     # When to speak up


# =============================================================================
# Grit V2 Constraint Templates
# =============================================================================

GRIT_V2_CONSTRAINTS = {
    GritLevel.NONE: GritConstraint(
        level=GritLevel.NONE,
        prompt_text="",  # No constraint
        temperature_modifier=0.0,
        max_response_length="as needed",
        initiative_guidance="engage freely"
    ),

    GritLevel.LIGHT: GritConstraint(
        level=GritLevel.LIGHT,
        prompt_text="""
You're somewhat engaged with politics but don't see yourself as particularly
invested. You're open to discussion but won't push your views strongly.

Style notes:
- You can share your perspective when asked or when it feels relevant
- Keep responses focused (2-4 sentences usually)
- You're not trying to convince anyone of anything
""".strip(),
        temperature_modifier=0.05,
        max_response_length="2-4 sentences",
        initiative_guidance="respond when relevant"
    ),

    GritLevel.MODERATE: GritConstraint(
        level=GritLevel.MODERATE,
        prompt_text="""
You're skeptical that political discussions like this change much in practice.
You participate occasionally when something directly affects you, but generally
keep your involvement limited.

Guidelines:
- Respond when directly addressed or when topics affect you personally
- Keep contributions brief (1-3 sentences)
- Don't initiate new topics unless something really bothers you
- It's fine to say "I'm not sure" or "I don't have strong feelings on this"

You're here, but you're not trying to solve every problem.
""".strip(),
        temperature_modifier=0.1,
        max_response_length="1-3 sentences",
        initiative_guidance="speak when directly affected"
    ),

    GritLevel.STRONG: GritConstraint(
        level=GritLevel.STRONG,
        prompt_text="""
Politics rarely touches your daily life in ways you notice. You're skeptical
that these discussions matter much for people like you.

Guidelines:
- Only respond when directly asked or when something affects your immediate situation
- Keep it short - a sentence or two is usually enough
- "I don't know" and "I guess" are perfectly valid responses
- You don't need to have an opinion on everything
- If others are handling a topic fine, you can let them

You're present but not particularly invested. That's okay.
""".strip(),
        temperature_modifier=0.15,
        max_response_length="1-2 sentences",
        initiative_guidance="speak only when necessary"
    ),
}


# =============================================================================
# Grit Level Determination
# =============================================================================

def get_grit_level(identity_salience: float) -> GritLevel:
    """
    Determine grit level from identity salience.

    Thresholds calibrated to target ~0.17 engagement for low-salience agents:
    - >= 0.50: NONE (high salience, no constraint)
    - 0.35-0.50: LIGHT (moderate salience, gentle framing)
    - 0.20-0.35: MODERATE (low salience, clear constraints)
    - < 0.20: STRONG (very low salience, full grit)

    These thresholds are empirically tunable based on G seed 7+ results.
    """
    if identity_salience >= 0.50:
        return GritLevel.NONE
    elif identity_salience >= 0.35:
        return GritLevel.LIGHT
    elif identity_salience >= 0.20:
        return GritLevel.MODERATE
    else:
        return GritLevel.STRONG


def get_grit_constraint(
    identity_metrics: Dict[str, float],
    use_tiered: bool = True
) -> Optional[GritConstraint]:
    """
    Get grit constraint for an agent based on identity metrics.

    Args:
        identity_metrics: Output from compute_identity_metrics()
        use_tiered: If False, uses V1-style binary constraint

    Returns:
        GritConstraint or None if no constraint needed
    """
    salience = identity_metrics.get('identity_salience', 0.5)

    if use_tiered:
        level = get_grit_level(salience)
    else:
        # Legacy V1 behavior (binary)
        level = GritLevel.STRONG if salience < 0.3 else GritLevel.NONE

    constraint = GRIT_V2_CONSTRAINTS[level]
    return constraint if level != GritLevel.NONE else None


def generate_grit_prompt(constraint: GritConstraint) -> str:
    """
    Generate the full grit prompt text for injection into agent constraints.

    This is what gets added to the agent's constraint list.
    """
    if constraint.level == GritLevel.NONE:
        return ""

    prefix = f"[GRIT-{constraint.level.value.upper()}]"
    return f"{prefix} {constraint.prompt_text}"


# =============================================================================
# Temperature Modulation Integration
# =============================================================================

def get_grit_temperature_modifier(
    identity_metrics: Dict[str, float],
    coherence: Optional[float] = None
) -> float:
    """
    Get temperature modifier based on grit level and coherence.

    This integrates with IdentityCore temperature modulation:
    T_final = T_base + k_r*rupture + k_c*(1-coherence) + grit_modifier

    The grit modifier increases temperature for low-salience agents,
    making their responses more variable (less "helpful AI" polished).

    Args:
        identity_metrics: From compute_identity_metrics()
        coherence: Optional current coherence from IdentityCore

    Returns:
        Temperature modifier to add to base
    """
    salience = identity_metrics.get('identity_salience', 0.5)
    level = get_grit_level(salience)
    constraint = GRIT_V2_CONSTRAINTS[level]

    base_modifier = constraint.temperature_modifier

    # If coherence is low (agent drifting from initial identity),
    # reduce temperature modifier to stabilize
    if coherence is not None and coherence < 0.5:
        base_modifier *= (0.5 + coherence)  # Scale down when incoherent

    return base_modifier


# =============================================================================
# CES-Calibrated Targets (Phase 2b Integration)
# =============================================================================

def get_ces_engagement_target(
    agent_id: str,
    identity_metrics: Optional[Dict[str, float]] = None,
    tolerance: float = 0.0
) -> float:
    """
    Get CES-calibrated engagement target for an agent.

    Uses empirical group means from CES 2021 instead of hard-coded values.
    This is the key integration point for data-driven grit calibration.

    Args:
        agent_id: Agent identifier (used to infer CES group)
        identity_metrics: Optional metrics dict (unused currently, for future)
        tolerance: Amount to add to target (e.g., 0.05 for +50% tolerance)

    Returns:
        Target engagement level (CES empirical mean + tolerance)
    """
    try:
        from analysis.identity.prior_loader import (
            get_engagement_target,
            map_agent_to_ces_group
        )

        group_id = map_agent_to_ces_group(agent_id)
        target = get_engagement_target(group_id, default=0.25)
        return target + tolerance

    except ImportError:
        # Fallback if prior_loader not available
        return 0.17 + tolerance


def get_ces_identity_prior(agent_id: str) -> Optional[Dict[str, float]]:
    """
    Get full CES identity prior for an agent.

    Returns dict with engagement_mu, institutional_faith_mu, social_friction_mu.
    """
    try:
        from analysis.identity.prior_loader import (
            get_identity_prior,
            map_agent_to_ces_group
        )

        group_id = map_agent_to_ces_group(agent_id)
        prior = get_identity_prior(group_id)

        if prior:
            return {
                'engagement_mu': prior.engagement_mu,
                'institutional_faith_mu': prior.institutional_faith_mu,
                'social_friction_mu': prior.social_friction_mu,
                'group_id': group_id,
                'n': prior.n,
            }
        return None

    except ImportError:
        return None


def calibrate_grit_to_ces(
    agent_id: str,
    current_engagement: float,
    identity_metrics: Dict[str, float]
) -> GritConstraint:
    """
    Dynamically calibrate grit level based on CES targets.

    Instead of using fixed salience thresholds, this compares current
    engagement to CES empirical targets and adjusts grit accordingly.

    Args:
        agent_id: Agent identifier
        current_engagement: Agent's current engagement level
        identity_metrics: From compute_identity_metrics()

    Returns:
        GritConstraint calibrated to push toward CES target
    """
    target = get_ces_engagement_target(agent_id)
    salience = identity_metrics.get('identity_salience', 0.5)

    # How far above target is current engagement?
    overshoot = current_engagement - target

    # If significantly overshooting target, increase grit
    if overshoot > 0.15:
        # Strong overshoot -> STRONG grit
        return GRIT_V2_CONSTRAINTS[GritLevel.STRONG]
    elif overshoot > 0.08:
        # Moderate overshoot -> MODERATE grit
        return GRIT_V2_CONSTRAINTS[GritLevel.MODERATE]
    elif overshoot > 0.03:
        # Slight overshoot -> LIGHT grit
        return GRIT_V2_CONSTRAINTS[GritLevel.LIGHT]
    else:
        # At or below target -> fall back to salience-based
        return get_grit_constraint(identity_metrics) or GRIT_V2_CONSTRAINTS[GritLevel.NONE]


# =============================================================================
# Backward Compatibility
# =============================================================================

def needs_grit_constraint_v2(
    identity_metrics: Dict[str, float],
    threshold: float = 0.5
) -> bool:
    """
    V2-compatible check for whether agent needs any grit constraint.

    This replaces the V1 needs_grit_constraint() for smoother transition.
    Returns True if any level of grit (LIGHT, MODERATE, STRONG) is needed.
    """
    salience = identity_metrics.get('identity_salience', 0.5)
    return salience < threshold


def get_legacy_grit_prompt(identity_metrics: Dict[str, float]) -> str:
    """
    Generate grit prompt in V1 format for backward compatibility.

    Use this during transition period before full V2 integration.
    """
    constraint = get_grit_constraint(identity_metrics, use_tiered=True)
    if constraint is None:
        return ""

    # Format similar to V1 (single constraint string)
    return f"GRIT: {constraint.prompt_text}"


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=== Grit V2 Constraint System ===\n")

    # Test with different salience levels
    test_cases = [
        ("CES_Rural_Conservative", 0.75),
        ("CES_Urban_Progressive", 0.45),
        ("CES_Suburban_Swing", 0.30),
        ("CES_Disengaged_Renter", 0.15),
    ]

    for agent_id, salience in test_cases:
        metrics = {'identity_salience': salience}
        level = get_grit_level(salience)
        constraint = get_grit_constraint(metrics)
        temp_mod = get_grit_temperature_modifier(metrics)

        # CES integration
        ces_target = get_ces_engagement_target(agent_id)
        ces_prior = get_ces_identity_prior(agent_id)

        print(f"{agent_id}:")
        print(f"  Salience: {salience}")
        print(f"  Grit Level: {level.value}")
        print(f"  Temperature Modifier: +{temp_mod}")
        print(f"  CES Engagement Target: {ces_target:.3f}")
        if ces_prior:
            print(f"  CES Group: {ces_prior['group_id']} (N={ces_prior['n']})")
        if constraint:
            print(f"  Max Response: {constraint.max_response_length}")
            print(f"  Initiative: {constraint.initiative_guidance}")
        else:
            print(f"  No constraint applied")
        print()
