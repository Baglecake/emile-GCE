"""
IdentityCore: Per-agent identity tracking with QSE mechanics.

Phase 2a implementation following dev-notes design:
- group_id for CES strata mapping
- sim_time tracking in history
- Placeholder hooks for empirical ΔI priors
- Modular compute_delta_I() and compute_tau()

See docs/theoretical_foundations.md for theoretical grounding.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class IdentityVector:
    """
    Three-dimensional identity vector grounded in CES.

    Dimensions:
        engagement: Network centrality / participation level (0-1)
        institutional_faith: Trust in institutions, 1 - critical_concepts_ratio (0-1)
        social_friction: Direct references / conflict level (0+)
    """
    engagement: float
    institutional_faith: float
    social_friction: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for math operations."""
        return np.array([self.engagement, self.institutional_faith, self.social_friction])

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'IdentityVector':
        """Create from dictionary (e.g., from vector extraction)."""
        return cls(
            engagement=d.get('engagement', 0.0),
            institutional_faith=d.get('institutional_faith', 1.0),
            social_friction=d.get('social_friction', 0.0)
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            'engagement': round(self.engagement, 4),
            'institutional_faith': round(self.institutional_faith, 4),
            'social_friction': round(self.social_friction, 4),
        }

    def __sub__(self, other: 'IdentityVector') -> 'IdentityVector':
        """Vector subtraction for ΔI computation."""
        return IdentityVector(
            engagement=self.engagement - other.engagement,
            institutional_faith=self.institutional_faith - other.institutional_faith,
            social_friction=self.social_friction - other.social_friction,
        )

    def norm(self) -> float:
        """L2 norm of vector."""
        return float(np.linalg.norm(self.to_array()))


@dataclass
class IdentityCore:
    """
    Per-agent identity tracking module.

    Tracks identity vectors over simulation time, computes:
    - ΔI: Magnitude of identity change from initial
    - Coherence: Directional stability (cos similarity to initial)
    - Emergent time τ: Time compression based on change magnitude
    - Temperature T: For LLM generation, based on rupture/coherence/natality

    Designed for Phase 2b integration:
    - group_id maps to CES sociogeographic strata
    - empirical_delta_mu/sigma for multi-wave CES priors
    """

    # Core identity
    agent_id: str
    group_id: str  # CES strata: 'urban_renter_left', 'rural_owner_right', etc.
    initial_vector: IdentityVector

    # Current state
    vector: IdentityVector = field(default=None)

    # History: List of (sim_time, IdentityVector) tuples
    history: List[Tuple[float, IdentityVector]] = field(default_factory=list)

    # QSE state
    surplus: float = 0.0  # Accumulated enactment (S)
    energy: float = 1.0   # Identity energy (for mortality)

    # Phase 2b hooks: empirical priors from multi-wave CES
    empirical_delta_mu: Optional[float] = None
    empirical_delta_sigma: Optional[float] = None

    # Observed stats from this simulation (for natality z-score)
    _delta_history: List[float] = field(default_factory=list)

    # Rupture state
    rupture_active: bool = False
    rupture_threshold: float = 0.15

    # Temperature config
    T_base: float = 0.7
    k_rupture: float = 0.3
    k_coherence: float = 0.2
    k_natality: float = 0.1

    def __post_init__(self):
        """Initialize vector and history if not provided."""
        if self.vector is None:
            self.vector = self.initial_vector
        if not self.history:
            self.history = [(0.0, self.initial_vector)]

    # =========================================================================
    # Core Update Methods
    # =========================================================================

    def update(self, new_vector: IdentityVector, sim_time: float) -> None:
        """
        Update identity vector and record history.

        Called after each round with observed behavior vector.
        """
        # Record delta from previous for natality computation
        if self.history:
            prev_vec = self.history[-1][1]
            delta_from_prev = self.compute_delta_from(prev_vec, new_vector)
            self._delta_history.append(delta_from_prev)

        # Update state
        self.history.append((sim_time, new_vector))
        self.vector = new_vector

        # Accumulate surplus (identity-as-accumulated-enactment)
        delta_I = self.compute_delta_I()
        self.surplus += 0.1 * delta_I  # α = 0.1

        # Check rupture
        sigma = self.compute_sigma()
        if sigma > self.rupture_threshold:
            self.rupture_active = True
            self.energy -= 0.1  # Energy cost for rupture
        else:
            self.rupture_active = False

    # =========================================================================
    # ΔI and Coherence Computation
    # =========================================================================

    def compute_delta_I(self) -> float:
        """
        Compute magnitude of identity change: |I_t - I_0|

        Phase 2a: Simple norm over current vs initial.
        Phase 2b: Will compare against empirical_delta_mu/sigma.
        """
        return self.compute_delta_from(self.initial_vector, self.vector)

    @staticmethod
    def compute_delta_from(v0: IdentityVector, vt: IdentityVector) -> float:
        """Compute |vt - v0|."""
        diff = vt - v0
        return diff.norm()

    def compute_sigma(self) -> float:
        """
        Compute symbolic tension σ: gap between identity and behavior.

        σ_t = |I_t - I_0| (simplified for Phase 2a)
        """
        return self.compute_delta_I()

    def compute_coherence(self) -> float:
        """
        Compute directional coherence: cos(I_t, I_0)

        High (→1): Identity direction preserved
        Low (→0): Identity has rotated significantly

        Full formula (Phase 2b):
            C_t = cos(I_t, I_0) × TE(I→B) / (TE(I→B) + TE(others→I))

        Phase 2a: Just cos similarity (TE ratio = 1.0 placeholder)
        """
        arr0 = self.initial_vector.to_array()
        arrt = self.vector.to_array()

        norm0 = np.linalg.norm(arr0)
        normt = np.linalg.norm(arrt)

        if norm0 < 1e-8 or normt < 1e-8:
            return 1.0  # No movement = coherent

        cos_sim = float(np.dot(arr0, arrt) / (norm0 * normt))

        # Phase 2a: TE ratio placeholder = 1.0
        te_ratio = 1.0

        return cos_sim * te_ratio

    # =========================================================================
    # Emergent Time τ
    # =========================================================================

    def compute_tau(self) -> float:
        """
        Compute emergent time τ from identity change magnitude.

        Small ΔI → long τ → "slow time" (identity stable)
        Large ΔI → short τ → time "thickens" (rapid change)

        Phase 2a: Uses fixed logistic transform.
        Phase 2b: Will use group-specific empirical priors.
        """
        from .tau import tau_from_delta
        delta = self.compute_delta_I()
        return tau_from_delta(delta, self.empirical_delta_mu, self.empirical_delta_sigma)

    # =========================================================================
    # Natality (z-score of ΔP)
    # =========================================================================

    def compute_natality(self) -> float:
        """
        Compute natality as history-normalized novelty.

        z_{i,t} = (ΔP_{i,t} - μ_i(t)) / (σ_i(t) + ε)
        N_{i,t} = sigmoid(k · z_{i,t})

        High z → high natality (significant change relative to history)
        """
        if len(self._delta_history) < 2:
            return 0.5  # Neutral natality for newborns

        # Use observed μ, σ from this agent's history
        mu = np.mean(self._delta_history)
        sigma = np.std(self._delta_history) if len(self._delta_history) > 1 else 0.1

        current_delta = self._delta_history[-1] if self._delta_history else 0.0

        z = (current_delta - mu) / (sigma + 1e-8)

        # Sigmoid with k=1.0
        natality = 1.0 / (1.0 + np.exp(-z))
        return float(natality)

    # =========================================================================
    # Temperature Modulation
    # =========================================================================

    def compute_temperature(self) -> float:
        """
        Compute dynamic temperature for LLM generation.

        T_t = T_base + k_r * rupture + k_c * (1 - coherence) + k_n * natality

        High coherence → low T → stable voice
        Low coherence → high T → exploratory, variable prose
        Rupture → high T → frantic exploration
        """
        rupture_signal = 1.0 if self.rupture_active else 0.0
        coherence = self.compute_coherence()
        natality = self.compute_natality()

        T = (self.T_base
             + self.k_rupture * rupture_signal
             + self.k_coherence * (1 - coherence)
             + self.k_natality * natality)

        # Clamp to reasonable range
        return float(np.clip(T, 0.2, 1.2))

    # =========================================================================
    # State Export
    # =========================================================================

    def get_state(self) -> Dict[str, Any]:
        """Export current state for logging/analysis."""
        return {
            'agent_id': self.agent_id,
            'group_id': self.group_id,
            'vector': self.vector.to_dict(),
            'initial_vector': self.initial_vector.to_dict(),
            'delta_I': round(self.compute_delta_I(), 4),
            'sigma': round(self.compute_sigma(), 4),
            'coherence': round(self.compute_coherence(), 4),
            'tau': round(self.compute_tau(), 4),
            'natality': round(self.compute_natality(), 4),
            'temperature': round(self.compute_temperature(), 4),
            'surplus': round(self.surplus, 4),
            'energy': round(self.energy, 4),
            'rupture_active': self.rupture_active,
            'history_length': len(self.history),
        }

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Export full trajectory for analysis."""
        trajectory = []
        for i, (sim_time, vec) in enumerate(self.history):
            delta_I = self.compute_delta_from(self.initial_vector, vec)

            # Coherence at this point
            arr0 = self.initial_vector.to_array()
            arrt = vec.to_array()
            norm0, normt = np.linalg.norm(arr0), np.linalg.norm(arrt)
            if norm0 > 1e-8 and normt > 1e-8:
                coherence = float(np.dot(arr0, arrt) / (norm0 * normt))
            else:
                coherence = 1.0

            trajectory.append({
                'round': i + 1,
                'sim_time': sim_time,
                'vector': vec.to_dict(),
                'delta_I': round(delta_I, 4),
                'coherence': round(coherence, 4),
            })
        return trajectory
