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
class SurplusTrace:
    """
    Memory of an enacted surplus event with decay/revalorization.

    Traces encode "has this line of becoming actually been enacted
    and held in the field before?" - the missing middle between
    continuous identity process and explicit memory.

    From inner_outer_protocol:
    - Traces store direction of becoming (ΔI), field response, and weight
    - Weights decay unless revalorized by similar enactments
    - Identity blends toward weighted trace directions
    """
    # When/where
    round_number: int
    turn_number: int
    semiotic_regime: str  # 'ACTIVE_CONTESTATION', 'CONSENSUS', etc.

    # Direction of becoming (ΔI vector, normalized)
    delta_I: np.ndarray  # 3D array: [engagement, inst_faith, social_friction]

    # State at event
    tau_at_event: float
    natality_at_event: float

    # Field response
    recognition_score: float
    contribution_value: float
    engagement: float

    # Weight (decays unless revalorized)
    weight: float

    def to_dict(self) -> Dict[str, Any]:
        """Export trace for logging."""
        return {
            'round': self.round_number,
            'turn': self.turn_number,
            'regime': self.semiotic_regime,
            'delta_I': [round(x, 4) for x in self.delta_I.tolist()],
            'tau': round(self.tau_at_event, 4),
            'natality': round(self.natality_at_event, 4),
            'recognition': round(self.recognition_score, 4),
            'contribution': round(self.contribution_value, 4),
            'engagement': round(self.engagement, 4),
            'weight': round(self.weight, 4),
        }


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

    # STATEFUL natality - updated by recognition/overshoot, decays toward τ-baseline
    # This implements the Émile-cogito pattern: capacity degrades unless revalorized
    natality_t: Optional[float] = None  # Initialized in __post_init__ from τ-baseline

    # Recognition tracking (EMA for smoothing)
    _recognition_ema: float = 0.5

    # SurplusTrace buffer (Stage 3)
    # Memory of enacted surplus events with decay/revalorization
    surplus_traces: List['SurplusTrace'] = field(default_factory=list)

    # Trace config
    trace_decay_lambda: float = 0.95  # Per-step decay: w ← λw
    trace_creation_threshold: float = 0.3  # Min surplus × recognition for trace
    trace_revalorize_rho: float = 0.2  # Revalorization coefficient
    trace_similarity_threshold: float = 0.7  # Min cosine sim for revalorization
    trace_blend_eta: float = 0.05  # Identity blending coefficient
    trace_max_count: int = 10  # Max traces to keep per agent

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
        # Initialize natality_t from τ-based baseline
        if self.natality_t is None:
            self.natality_t = self.compute_natality_baseline()

    # =========================================================================
    # Core Update Methods
    # =========================================================================

    def update(self, new_vector: IdentityVector, sim_time: float) -> None:
        """
        Update identity vector and record history.

        Called after each round with observed behavior vector.

        Includes Stage 4 trace blending: identity is gently pulled toward
        weighted trace directions (frequently revalorized enactments).
        """
        # Record delta from previous for natality computation
        if self.history:
            prev_vec = self.history[-1][1]
            delta_from_prev = self.compute_delta_from(prev_vec, new_vector)
            self._delta_history.append(delta_from_prev)

        # Update state
        self.history.append((sim_time, new_vector))
        self.vector = new_vector

        # Stage 4: Apply trace blend (gentle pull toward weighted trace directions)
        # This provides a recursive correction based on enacted surplus without
        # swamping CES priors or immediate feedback.
        if len(self.surplus_traces) > 0:
            blended_vector = self.apply_trace_blend()
            self.vector = blended_vector

        # Update surplus as qualitative capacity (not crude accumulation)
        self.update_surplus()

        # Check rupture
        sigma = self.compute_sigma()
        if sigma > self.rupture_threshold:
            self.rupture_active = True
            self.energy -= 0.1  # Energy cost for rupture
        else:
            self.rupture_active = False

    def update_surplus(self) -> float:
        """
        Update surplus as qualitative capacity, not crude accumulation.

        Surplus = how enactable this identity is in this field at this time,
        relative to place (τ), natality, and recognition.

        From inner_outer_protocol:
        - τ and natality → "relative to place" and capacity
        - recognition_ema → semiotic alignment / field receptivity
        - EMA → not pure accumulation, but a rolling, qualitative index

        Returns:
            The computed local_surplus for this step (useful for trace creation)
        """
        from .tau import TAU_MIN, TAU_MAX

        delta_I = self.compute_delta_I()

        # 1) Base magnitude from identity change
        base = delta_I

        # 2) Modulate by emergent time τ (normalized to [0,1])
        tau = self.compute_tau()
        tau_normalized = (tau - TAU_MIN) / (TAU_MAX - TAU_MIN)
        tau_normalized = max(0.0, min(1.0, tau_normalized))
        f_tau = 0.5 + 0.5 * tau_normalized  # [0.5, 1.0]

        # 3) Modulate by natality (stateful, already in [0,1])
        f_nat = 0.5 + 0.5 * self.natality_t  # [0.5, 1.0]

        # 4) Modulate by recognition (field actually making space)
        f_rec = 0.5 + 0.5 * self._recognition_ema  # [0.5, 1.0]

        # Local surplus: qualitative capacity at this moment
        local_surplus = base * f_tau * f_nat * f_rec

        # 5) EMA update instead of raw accumulation
        beta = 0.2  # Smoothing factor
        self.surplus = (1 - beta) * self.surplus + beta * local_surplus

        return local_surplus

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
    # Natality (z-score of ΔP with τ-based baseline)
    # =========================================================================

    def compute_natality_baseline(self) -> float:
        """
        Compute τ-derived natality baseline.

        Baseline natality emerges from emergent time τ, not a flat constant.
        - High τ (slow time, stable identity) → higher baseline natality
        - Low τ (thick time, rapid change) → lower baseline natality

        This implements the Émile-cogito pattern where weights degrade
        unless revalorized by calling them into action.
        """
        from .tau import TAU_MIN, TAU_MAX

        tau = self.compute_tau()

        # Normalize τ to [0, 1]
        x = (tau - TAU_MIN) / (TAU_MAX - TAU_MIN)
        x = max(0.0, min(1.0, x))  # Clamp to valid range

        # Map to natality baseline band [0.2, 0.8]
        # Slower time (high τ) → higher baseline (more latent capacity)
        min_base, max_base = 0.2, 0.8
        return min_base + (max_base - min_base) * x

    def compute_natality(self) -> float:
        """
        Compute natality as history-normalized novelty with τ-based baseline decay.

        z_{i,t} = (ΔP_{i,t} - μ_i(t)) / (σ_i(t) + ε)
        novelty_signal = sigmoid(k · z_{i,t})
        natality = (1 - decay) * novelty_signal + decay * τ_baseline

        High z → high natality (significant change relative to history)
        Natality decays toward τ-derived baseline when not revalorized.

        This implements the Émile-cogito pattern: capacity for new beginnings
        emerges from emergent time and is sustained by field recognition.
        """
        # For newborns, use τ-based baseline (not flat 0.5)
        if len(self._delta_history) < 2:
            return self.compute_natality_baseline()

        # Use observed μ, σ from this agent's history
        mu = np.mean(self._delta_history)
        sigma = np.std(self._delta_history) if len(self._delta_history) > 1 else 0.1

        current_delta = self._delta_history[-1] if self._delta_history else 0.0

        z = (current_delta - mu) / (sigma + 1e-8)

        # Novelty signal via sigmoid (k=1.0)
        novelty_signal = 1.0 / (1.0 + np.exp(-z))

        # Decay toward τ-based baseline (not flat 0.5)
        baseline = self.compute_natality_baseline()
        decay = 0.02  # Small decay rate

        # Blend novelty signal with baseline
        natality = (1 - decay) * novelty_signal + decay * baseline

        return float(np.clip(natality, 0.0, 1.0))

    def update_natality(
        self,
        recognition_score: float,
        overshoot_ratio: float = 0.0,
        recognition_low_threshold: float = 0.3,
        alpha: float = 0.05,
        beta: float = 0.05,
        gamma: float = 0.05,
        decay: float = 0.02
    ) -> float:
        """
        Update stateful natality based on field recognition and overshoot.

        This is the core identity mechanic that implements:
        - Recognition from field → natality rises (being affirmed)
        - Overshoot + low recognition → natality drains ("talked over")
        - Overshoot + high recognition → natality rises ("held and amplified")
        - Decay toward τ-based baseline when idle

        Args:
            recognition_score: How much the field is recognizing this agent (0-1)
            overshoot_ratio: How much agent exceeded word limit (0 if within limit)
            recognition_low_threshold: Below this, recognition is considered "low"
            alpha: Base recognition effect coefficient
            beta: Overshoot penalty coefficient when recognition is low
            gamma: Overshoot bonus coefficient when recognition is high
            decay: Rate of decay toward τ-baseline

        Returns:
            Updated natality_t value

        Implements the Émile-cogito pattern:
        - Capacity for new beginnings emerges from the social field
        - Weights degrade unless revalorized by field recognition
        """
        # Update recognition EMA for smoothing
        ema_alpha = 0.3
        self._recognition_ema = (1 - ema_alpha) * self._recognition_ema + ema_alpha * recognition_score

        # Get novelty signal from ΔI history (the computed part)
        novelty_signal = self.compute_natality()

        # Base recognition delta: above 0.5 recognition → gain, below → loss
        base_delta = alpha * (recognition_score - 0.5)

        # Overshoot modulation
        overshoot_delta = 0.0
        if overshoot_ratio > 0:
            if recognition_score < recognition_low_threshold:
                # "Talked over": overshoot + low recognition → natality drains
                overshoot_delta = -beta * overshoot_ratio
            else:
                # "Held and amplified": overshoot + high recognition → natality rises
                overshoot_delta = gamma * recognition_score

        # Compute activation delta
        activation_delta = base_delta + overshoot_delta

        # Apply update with decay toward τ-baseline
        baseline = self.compute_natality_baseline()

        # Blend: novelty signal + activation + decay toward baseline
        # The formula: (1 - decay) * (current + activation) + decay * baseline
        new_natality = (1 - decay) * (self.natality_t + activation_delta) + decay * baseline

        # Clamp and store
        self.natality_t = float(np.clip(new_natality, 0.0, 1.0))

        return self.natality_t

    def get_natality(self) -> float:
        """
        Get current natality state.

        Returns the stateful natality_t which is updated by recognition/overshoot,
        not the computed novelty signal.
        """
        return self.natality_t

    # =========================================================================
    # SurplusTrace Management (Stage 3)
    # =========================================================================

    def decay_traces(self) -> None:
        """
        Apply per-step decay to all trace weights.

        w ← λw where λ in [0.90, 0.99]

        This implements "channels lose efficacy unless re-cohered" from Émile.
        """
        for trace in self.surplus_traces:
            trace.weight *= self.trace_decay_lambda

        # Prune traces with negligible weight
        self.surplus_traces = [
            t for t in self.surplus_traces if t.weight > 0.01
        ]

    def maybe_create_trace(
        self,
        round_number: int,
        turn_number: int,
        semiotic_regime: str,
        recognition_score: float,
        contribution_value: float,
        engagement: float,
    ) -> Optional[SurplusTrace]:
        """
        Create a trace on "high-surplus, high-recognition" events.

        Trace creation condition: surplus × recognition > threshold

        Args:
            round_number: Current round
            turn_number: Current turn within round
            semiotic_regime: Current regime ('ACTIVE_CONTESTATION', etc.)
            recognition_score: Recognition from field (0-1)
            contribution_value: Agent's contribution value
            engagement: Agent's engagement level

        Returns:
            Created trace or None if threshold not met
        """
        # Compute creation score
        creation_score = self.surplus * recognition_score

        if creation_score < self.trace_creation_threshold:
            return None

        # Compute ΔI direction (current vs previous)
        if len(self.history) < 2:
            return None  # Need history to compute direction

        prev_vec = self.history[-2][1] if len(self.history) >= 2 else self.initial_vector
        curr_vec = self.vector

        delta_I_vec = curr_vec - prev_vec
        delta_I_arr = delta_I_vec.to_array()

        # Normalize to unit direction (or zero if no change)
        norm = np.linalg.norm(delta_I_arr)
        if norm > 1e-8:
            delta_I_normalized = delta_I_arr / norm
        else:
            delta_I_normalized = delta_I_arr

        # Initial weight: proportional to surplus × recognition
        initial_weight = creation_score

        trace = SurplusTrace(
            round_number=round_number,
            turn_number=turn_number,
            semiotic_regime=semiotic_regime,
            delta_I=delta_I_normalized,
            tau_at_event=self.compute_tau(),
            natality_at_event=self.natality_t,
            recognition_score=recognition_score,
            contribution_value=contribution_value,
            engagement=engagement,
            weight=initial_weight,
        )

        self.surplus_traces.append(trace)

        # Enforce max trace count (keep highest weight)
        if len(self.surplus_traces) > self.trace_max_count:
            self.surplus_traces.sort(key=lambda t: t.weight, reverse=True)
            self.surplus_traces = self.surplus_traces[:self.trace_max_count]

        return trace

    def revalorize_traces(self, current_delta_I: np.ndarray, recognition_score: float) -> int:
        """
        Revalorize traces when new event has high cosine similarity with trace's ΔI.

        w ← w + ρ × similarity × recognition_score_new

        This implements "weights are revalorized by calling them into action" from Émile.

        Args:
            current_delta_I: Current ΔI direction (normalized)
            recognition_score: Current recognition from field

        Returns:
            Number of traces revalorized
        """
        if len(self.surplus_traces) == 0:
            return 0

        norm_current = np.linalg.norm(current_delta_I)
        if norm_current < 1e-8:
            return 0

        current_unit = current_delta_I / norm_current
        count = 0

        for trace in self.surplus_traces:
            # Compute cosine similarity between current and trace ΔI
            norm_trace = np.linalg.norm(trace.delta_I)
            if norm_trace < 1e-8:
                continue

            trace_unit = trace.delta_I / norm_trace
            similarity = float(np.dot(current_unit, trace_unit))

            if similarity > self.trace_similarity_threshold:
                # Revalorize: bump weight
                trace.weight += self.trace_revalorize_rho * similarity * recognition_score
                count += 1

        return count

    def compute_trace_direction(self) -> Optional[np.ndarray]:
        """
        Compute weighted sum of ΔI directions from traces.

        T = Σ_j w_j × ΔÎ_j / Σ_j w_j

        Returns:
            Weighted trace direction or None if no traces
        """
        if len(self.surplus_traces) == 0:
            return None

        total_weight = sum(t.weight for t in self.surplus_traces)
        if total_weight < 1e-8:
            return None

        weighted_sum = np.zeros(3)
        for trace in self.surplus_traces:
            weighted_sum += trace.weight * trace.delta_I

        return weighted_sum / total_weight

    def apply_trace_blend(self) -> IdentityVector:
        """
        Blend identity toward weighted trace direction.

        I_new = I_current + η × T

        Where T is the weighted trace direction and η ~ 0.05-0.1.
        This provides a "gentle pull" toward frequently revalorized enactments
        without swamping CES priors or immediate feedback.

        Returns:
            New identity vector after blending
        """
        T = self.compute_trace_direction()
        if T is None:
            return self.vector

        # Apply blend
        current_arr = self.vector.to_array()
        blended_arr = current_arr + self.trace_blend_eta * T

        # Convert back to IdentityVector
        return IdentityVector(
            engagement=float(np.clip(blended_arr[0], 0.0, 1.0)),
            institutional_faith=float(np.clip(blended_arr[1], 0.0, 1.0)),
            social_friction=float(max(0.0, blended_arr[2])),  # No upper bound
        )

    def get_trace_summary(self) -> Dict[str, Any]:
        """
        Get compact summary of traces for logging.

        Returns:
            Dict with num_traces, trace_mass, top trace info
        """
        if len(self.surplus_traces) == 0:
            return {
                'num_traces': 0,
                'trace_mass': 0.0,
                'top_trace': None,
            }

        trace_mass = sum(t.weight for t in self.surplus_traces)

        # Top trace by weight
        top_trace = max(self.surplus_traces, key=lambda t: t.weight)

        return {
            'num_traces': len(self.surplus_traces),
            'trace_mass': round(trace_mass, 4),
            'top_trace': {
                'round': top_trace.round_number,
                'weight': round(top_trace.weight, 4),
                'regime': top_trace.semiotic_regime,
            },
        }

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

        Uses stateful natality_t (updated by recognition/overshoot) not computed natality.
        """
        rupture_signal = 1.0 if self.rupture_active else 0.0
        coherence = self.compute_coherence()
        # Use stateful natality_t, not computed
        natality = self.natality_t

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
        trace_summary = self.get_trace_summary()
        return {
            'agent_id': self.agent_id,
            'group_id': self.group_id,
            'vector': self.vector.to_dict(),
            'initial_vector': self.initial_vector.to_dict(),
            'delta_I': round(self.compute_delta_I(), 4),
            'sigma': round(self.compute_sigma(), 4),
            'coherence': round(self.compute_coherence(), 4),
            'tau': round(self.compute_tau(), 4),
            'natality': round(self.natality_t, 4),  # Stateful natality
            'natality_baseline': round(self.compute_natality_baseline(), 4),
            'natality_novelty': round(self.compute_natality(), 4),  # Computed novelty signal
            'recognition_ema': round(self._recognition_ema, 4),
            'temperature': round(self.compute_temperature(), 4),
            'surplus': round(self.surplus, 4),
            'energy': round(self.energy, 4),
            'rupture_active': self.rupture_active,
            'history_length': len(self.history),
            # Stage 3: SurplusTrace summary
            'num_traces': trace_summary['num_traces'],
            'trace_mass': trace_summary['trace_mass'],
            'top_trace': trace_summary['top_trace'],
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
