"""
Emergent Time τ Computation

From émile QSE mechanics:
- τ = social clock that compresses/expands based on magnitude of identity change
- Small Δσ → long τ → "slow time" (identity-place relation stable)
- Large Δσ → short τ → time "thickens" (rapid identity change)

Phase 2a: Fixed logistic transform with global parameters.
Phase 2b: Group-specific μ, σ priors from multi-wave CES.
"""

from typing import Optional
import numpy as np


# Default τ parameters (tunable)
TAU_MIN = 0.1   # Minimum τ (fastest time)
TAU_MAX = 2.0   # Maximum τ (slowest time)
K = 5.0         # Steepness of logistic
THETA = 0.2    # Midpoint of logistic (where τ = (TAU_MIN + TAU_MAX) / 2)


def tau_from_delta(
    delta: float,
    empirical_mu: Optional[float] = None,
    empirical_sigma: Optional[float] = None,
) -> float:
    """
    Compute emergent time τ from identity change magnitude.

    Args:
        delta: Magnitude of identity change |I_t - I_0|
        empirical_mu: (Phase 2b) Group-specific mean ΔI from CES waves
        empirical_sigma: (Phase 2b) Group-specific std ΔI from CES waves

    Returns:
        τ: Emergent time value (TAU_MIN to TAU_MAX)

    Formula:
        τ = TAU_MIN + (TAU_MAX - TAU_MIN) / (1 + exp(K * (delta - theta)))

    Phase 2b extension:
        If empirical priors provided, normalize delta by group:
        z = (delta - empirical_mu) / empirical_sigma
        Then use z in logistic instead of raw delta.
    """
    if empirical_mu is not None and empirical_sigma is not None:
        # Phase 2b: Normalize by empirical distribution
        z = (delta - empirical_mu) / (empirical_sigma + 1e-8)
        # Use z as input to logistic (centered at 0)
        tau = TAU_MIN + (TAU_MAX - TAU_MIN) / (1 + np.exp(K * z))
    else:
        # Phase 2a: Use raw delta with fixed threshold
        tau = TAU_MIN + (TAU_MAX - TAU_MIN) / (1 + np.exp(K * (delta - THETA)))

    return float(tau)


def compute_tau(
    delta_history: list,
    empirical_mu: Optional[float] = None,
    empirical_sigma: Optional[float] = None,
) -> float:
    """
    Compute τ from a history of deltas (mean |σ_t - σ_{t-1}|).

    This version uses the average rate of change, not cumulative change.
    More appropriate for tracking "pace" of identity evolution.

    Args:
        delta_history: List of round-to-round ΔI values
        empirical_mu: (Phase 2b) Group-specific mean
        empirical_sigma: (Phase 2b) Group-specific std

    Returns:
        τ: Emergent time value
    """
    if not delta_history:
        return TAU_MAX  # No change history = slow time

    mean_delta = float(np.mean(delta_history))
    return tau_from_delta(mean_delta, empirical_mu, empirical_sigma)
