"""
Identity Core Module (Phase 2a)

Per-agent identity tracking with QSE mechanics:
- Identity vectors (engagement, institutional_faith, social_friction)
- Coherence tracking (directional stability)
- Emergent time τ (magnitude-based time compression)
- Temperature modulation (T = T_base + k_r*rupture + k_c*(1-coherence) + k_n*natality)

Designed for Phase 2b integration:
- group_id for CES strata mapping
- Placeholder hooks for empirical ΔI priors (multi-wave CES)
- Temporal compression config
"""

from .core import IdentityCore, IdentityVector, RoundFeedback
from .tau import compute_tau, tau_from_delta

__all__ = ['IdentityCore', 'IdentityVector', 'RoundFeedback', 'compute_tau', 'tau_from_delta']
