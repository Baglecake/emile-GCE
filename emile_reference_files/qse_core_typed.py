"""
Core Quantum Surplus Emergence engine with type hints:
 - Symbolic curvature kernels
 - Surplus field update
 - Emergent time calculation
 - Schrödinger evolution via split-step FFT
 - Quantum→surplus feedback
"""
import numpy as np
import numpy.typing as npt
from scipy.fft import fft, ifft, fftfreq
from typing import Tuple, Dict, List, Any, Optional
from .config import QSEConfig, CONFIG

# Type aliases for clarity
FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


# -- Symbolic Curvature (Theorem 1) --
def calculate_symbolic_fields(
    S: FloatArray,
    cfg: QSEConfig = CONFIG
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """
    Compute Psi, Phi, and Sigma fields from surplus S.

    Args:
        S: Surplus field array
        cfg: Configuration object

    Returns:
        Tuple of (psi, phi, sigma) arrays

    Mathematical definition:
        Psi = sigmoid(K_PSI*(S - THETA_PSI))
        Phi = max(0, K_PHI*(S - THETA_PHI))
        Sigma = Psi - Phi
    """
    psi: FloatArray = 1.0 / (1.0 + np.exp(-cfg.K_PSI * (S - cfg.THETA_PSI)))
    phi: FloatArray = np.maximum(0.0, cfg.K_PHI * (S - cfg.THETA_PHI))
    sigma: FloatArray = psi - phi
    return psi, phi, sigma


# -- Emergent Time (Theorem 3) --
def calculate_emergent_time(
    sigma: FloatArray,
    sigma_prev: Optional[FloatArray],
    cfg: QSEConfig = CONFIG
) -> float:
    """
    Compute emergent time tau based on change in Sigma.

    Args:
        sigma: Current sigma field
        sigma_prev: Previous sigma field (None for first step)
        cfg: Configuration object

    Returns:
        Emergent time value in range [TAU_MIN, TAU_MAX]

    Mathematical definition:
        Tau = TAU_MIN + (TAU_MAX-TAU_MIN)/(1+exp(K*(delta-THETA)))
        where delta = mean(|sigma - sigma_prev|)
    """
    if sigma_prev is None:
        return cfg.TAU_MAX

    delta: float = float(np.mean(np.abs(sigma - sigma_prev)))
    raw: float = cfg.TAU_MIN + (cfg.TAU_MAX - cfg.TAU_MIN) / (
        1.0 + np.exp(cfg.TAU_K * (delta - cfg.TAU_THETA))
    )
    return float(np.clip(raw, cfg.TAU_MIN, cfg.TAU_MAX))


# -- Surplus Dynamics (Theorem 2) --
def update_surplus(
    S: FloatArray,
    sigma: FloatArray,
    dt: float,
    cfg: QSEConfig = CONFIG
) -> FloatArray:
    """
    Update surplus field S according to QSE dynamics.

    Args:
        S: Current surplus field
        sigma: Symbolic curvature field
        dt: Time step
        cfg: Configuration object

    Returns:
        Updated surplus field

    Dynamics equation:
        S_new = (1+gamma*dt)*S + beta*dt*sigma - expel + tension_coupling*laplacian - damping*S + noise
        where expel activates when |sigma| > theta_rupture
    """
    g: float = cfg.S_GAMMA * dt
    b: float = cfg.S_BETA * dt
    e: float = cfg.S_EPSILON * dt
    t: float = cfg.S_TENSION * dt
    c: float = cfg.S_COUPLING * dt
    d: float = cfg.S_DAMPING * dt

    # Rupture expulsion
    expel: FloatArray = np.where(
        np.abs(sigma) > cfg.S_THETA_RUPTURE,
        e * S,
        0.0
    )

    # Basic growth + coupling
    S_new: FloatArray = (1.0 + g) * S + b * sigma - expel

    # Laplacian coupling
    lap: FloatArray = np.roll(S, 1) - 2.0 * S + np.roll(S, -1)
    S_new += t * c * lap

    # Damping
    S_new -= d * S

    # Stochastic noise
    S_new += 0.01 * np.sqrt(dt) * np.random.randn(*S.shape)

    # Clamp to [0,1]
    return np.clip(S_new, 0.0, 1.0)


# -- Potential generation --
def create_double_well_potential(x: FloatArray) -> FloatArray:
    """
    Create static double-well potential.

    Args:
        x: Spatial grid

    Returns:
        Potential field V(x)
    """
    width: float = (x.max() - x.min()) / 8.0
    wells: FloatArray = -np.exp(-((x + 2 * width) ** 2) / (2 * width ** 2))
    wells += -np.exp(-((x - 2 * width) ** 2) / (2 * width ** 2))
    barrier: FloatArray = 0.5 * np.exp(-(x ** 2) / (width ** 2 / 2.0))
    V: FloatArray = wells + barrier
    V = V - V.min()
    return 0.2 * V


def create_dynamic_potential(
    x: FloatArray,
    sigma: FloatArray,
    cfg: QSEConfig = CONFIG,
    t: float = 0.0
) -> FloatArray:
    """
    Build time-varying potential with sigma coupling.

    Args:
        x: Spatial grid
        sigma: Symbolic curvature field
        cfg: Configuration object
        t: Current time

    Returns:
        Dynamic potential V(x, sigma, t)
    """
    base: FloatArray = create_double_well_potential(x)
    barrier: FloatArray = (0.3 + 0.2 * np.sin(t / 5.0)) * np.exp(
        -(x ** 2) / ((len(x) / 8.0) ** 2)
    )
    pot: FloatArray = base + barrier + cfg.INPUT_COUPLING * sigma
    # Don't normalize - we need sigma coupling to affect the potential energy scale
    return pot


# -- Schrödinger step (Split-step FFT) --
def schrodinger_step(
    psi: ComplexArray,
    V: FloatArray,
    x: FloatArray,
    dt: float,
    cfg: QSEConfig = CONFIG
) -> ComplexArray:
    """
    Evolve wavefunction using split-step Fourier method.

    Args:
        psi: Current wavefunction
        V: Potential field
        x: Spatial grid
        dt: Time step
        cfg: Configuration object

    Returns:
        Evolved wavefunction

    Method: Split-operator approximation
        exp(-iHdt) ≈ exp(-iT*dt/2) * exp(-iV*dt) * exp(-iT*dt/2)
        where T is kinetic energy operator, V is potential
    """
    N: int = psi.size
    dx: float = x[1] - x[0]

    # k-space frequencies
    k: FloatArray = fftfreq(N, d=dx) * 2.0 * np.pi

    # Half-step kinetic
    psi_k: ComplexArray = fft(psi)
    psi = ifft(np.exp(-1j * cfg.HBAR * k ** 2 / (2 * cfg.MASS) * dt / 2.0) * psi_k)

    # Potential step
    psi = np.exp(-1j * V * dt / cfg.HBAR) * psi

    # Half-step kinetic
    psi_k = fft(psi)
    psi = ifft(np.exp(-1j * cfg.HBAR * k ** 2 / (2 * cfg.MASS) * dt / 2.0) * psi_k)

    return psi


# -- QSE Engine --
class QSEEngine:
    """
    Encapsulates the QSE loop: surplus update, quantum evolution, feedback.
    """

    def __init__(self, cfg: QSEConfig = CONFIG):
        """
        Initialize QSE engine.

        Args:
            cfg: Configuration object
        """
        self.cfg: QSEConfig = cfg

        # Surplus field
        self.S: FloatArray = 0.1 + 0.05 * np.random.rand(cfg.GRID_SIZE)
        self.sigma_prev: Optional[FloatArray] = None

        # Spatial grid for quantum
        self.x: FloatArray = np.linspace(-1.0, 1.0, cfg.GRID_SIZE)
        dx: float = self.x[1] - self.x[0]

        # Initial Gaussian wavepacket
        psi0: ComplexArray = np.exp(-(self.x ** 2) / (2.0 * (0.2) ** 2))
        norm: float = float(np.sqrt((np.abs(psi0) ** 2).sum() * dx))
        self.psi: ComplexArray = psi0 / norm

        # Time tracker
        self.time: float = 0.0

        # History
        self.history: List[Dict[str, Any]] = []

    def step(self, sigma: FloatArray, dt: float = 0.01) -> Dict[str, Any]:
        """
        Perform one QSE step.

        Args:
            sigma: Symbolic curvature field
            dt: Time step

        Returns:
            Dictionary of metrics for this step

        Process:
            1) Update surplus using sigma
            2) Build potential & evolve quantum state
            3) Feed quantum back into surplus
            4) Record metrics
        """
        # 1) Surplus update
        self.S = update_surplus(self.S, sigma, dt, self.cfg)

        # 2) Quantum evolution
        V: FloatArray = create_dynamic_potential(self.x, sigma, self.cfg, self.time)
        self.psi = schrodinger_step(self.psi, V, self.x, dt, self.cfg)

        # 3) Quantum->Surplus feedback
        prob: FloatArray = np.abs(self.psi) ** 2
        alpha: float = self.cfg.QUANTUM_COUPLING
        self.S = (1.0 - alpha) * self.S + alpha * prob
        # Ensure surplus stays in [0, 1] after feedback
        self.S = np.clip(self.S, 0.0, 1.0)

        # 4) Record and advance
        metrics: Dict[str, Any] = {
            'time': self.time,
            'surplus_mean': float(self.S.mean()),
            'sigma_mean': float(sigma.mean()),
            'prob_density': prob.copy(),
        }

        self.history.append(metrics)
        self.time += dt

        return metrics
