"""
Transfer Entropy Proxy for Identity-Behavior Coupling.

Implements a mutual-information based proxy for transfer entropy:
- TE(I->B): How much does identity history predict current behavior?
- TE(others->B): How much does others' behavior predict this agent's behavior?

This is the "antifinite autonomy ratio" - am I living my own surplus or being dragged?

The Formula (Coherence with TE):
    Coherence = cos(It, I0) x TE(I->B) / (TE(I->B) + TE(others->B))

Interpretation:
- TE(I->B) high, TE(others->B) low: Agent is steering its own surplus = authentic coherence
- TE(others->B) dominates: Agent is being dragged by the field = conformist coherence

Based on:
- Social Aesthetics paper Section 4 (identity as process)
- Gemini's "Vector Gap" analysis (authentic vs conformist patterns)
- Information-theoretic approach to causal influence detection
"""

import numpy as np
from typing import Optional


def _discretize(series: np.ndarray, bins: int = 5) -> np.ndarray:
    """Simple equal-width binning for MI approximation."""
    if len(series) == 0 or series.max() == series.min():
        return np.zeros_like(series, dtype=int)
    edges = np.linspace(series.min(), series.max(), bins + 1)
    return np.digitize(series, edges[:-1], right=True)


def mutual_info(x: np.ndarray, y: np.ndarray, bins: int = 5) -> float:
    """
    Simple mutual information approximation via histogram.

    Can be replaced with more sophisticated estimators later (e.g., JIDT).

    Args:
        x: First variable (discrete or continuous)
        y: Second variable (discrete or continuous)
        bins: Number of bins for discretization

    Returns:
        Mutual information estimate in nats
    """
    if len(x) != len(y) or len(x) < 5:
        return 0.0

    x_b = _discretize(x, bins)
    y_b = _discretize(y, bins)

    # Joint histogram
    joint, _, _ = np.histogram2d(x_b, y_b, bins=(bins, bins))

    if joint.sum() == 0:
        return 0.0

    pxy = joint / joint.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    # MI = sum p(x,y) log(p(x,y) / (p(x)p(y)))
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))

    return float(max(0.0, mi))


def te_ratio_proxy(
    identity_history: np.ndarray,
    behavior_history: np.ndarray,
    others_history: np.ndarray,
    min_len: int = 8,
) -> float:
    """
    Compute TE ratio: TE(I->B) / (TE(I->B) + TE(others->B))

    Uses lag-1 mutual information as a proxy for transfer entropy:
    - TE(I->B) approx I(Bt; It-1)
    - TE(others->B) approx I(Bt; Ot-1)

    Args:
        identity_history: Time series of identity scalar (e.g., ||It - I0||)
        behavior_history: Time series of behavior scalar (e.g., engagement)
        others_history: Time series of mean others' behavior
        min_len: Minimum history length before computing (return 1.0 if insufficient)

    Returns:
        Float in [0, 1]:
        - >0.5: Identity drives behavior more than others (authentic)
        - <0.5: Others drive behavior more than identity (conformist)
        - =1.0: Not enough data yet (early rounds)
    """
    if (len(identity_history) < min_len or
        len(behavior_history) < min_len or
        len(others_history) < min_len):
        return 1.0  # Neutral until we have data

    # Ensure arrays
    I = np.asarray(identity_history, dtype=float)
    B = np.asarray(behavior_history, dtype=float)
    O = np.asarray(others_history, dtype=float)

    # Lag-1 series: predict current behavior from previous state
    I_prev = I[:-1]
    B_curr = B[1:]
    O_prev = O[:-1]

    # Compute MI proxies
    te_I = mutual_info(I_prev, B_curr)
    te_O = mutual_info(O_prev, B_curr)

    denom = te_I + te_O
    if denom <= 1e-8:
        return 1.0  # No signal yet

    return float(te_I / denom)


def compute_te_valid(history_len: int, min_len: int = 8) -> bool:
    """
    Check if TE ratio is based on real data vs default.

    Args:
        history_len: Current length of behavior history
        min_len: Minimum required for valid TE computation

    Returns:
        True if TE ratio is computed from actual data, False if defaulting to 1.0
    """
    return history_len >= min_len
