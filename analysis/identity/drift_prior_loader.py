"""
Drift Prior Loader - Loads temporal drift priors for IdentityCore calibration.

Uses identity_drift_priors.v1.json to provide:
- Per-dimension delta_mu (mean change over election cycle)
- Per-dimension sigma (volatility of change)
- Group modifiers for demographic adjustment

These priors are used by tau.py to normalize identity change:
    z = (delta - empirical_mu) / empirical_sigma

Literature-based initial values; to be updated when CES panel linkage available.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple

# Canonical dimension order (must match identity_core/core.py)
IDENTITY_DIMS = [
    "engagement", "institutional_faith", "ideology",
    "partisanship", "sociogeographic", "social_friction", "tie_to_place"
]


@dataclass
class DimensionDriftPrior:
    """Drift prior for a single identity dimension."""
    delta_mu: float
    sigma: float
    stability_estimate: float


@dataclass
class AgentDriftPriors:
    """Complete drift priors for an agent (all 7 dimensions)."""
    agent_id: str
    dimension_priors: Dict[str, DimensionDriftPrior]
    aggregate_delta_mu: float
    aggregate_sigma: float

    def get_dimension_prior(self, dim: str) -> Tuple[float, float]:
        """Get (delta_mu, sigma) for a specific dimension."""
        if dim in self.dimension_priors:
            p = self.dimension_priors[dim]
            return (p.delta_mu, p.sigma)
        return (self.aggregate_delta_mu, self.aggregate_sigma)

    def get_aggregate(self) -> Tuple[float, float]:
        """Get aggregate (delta_mu, sigma) across all dimensions."""
        return (self.aggregate_delta_mu, self.aggregate_sigma)


# Cache for loaded priors
_PRIORS_CACHE: Optional[Dict[str, Any]] = None


def _find_priors_file(version: str = "v1") -> Path:
    """Find the drift priors JSON file."""
    candidates = [
        Path(__file__).parent.parent.parent / "data" / "identity" / f"identity_drift_priors.{version}.json",
        Path(f"data/identity/identity_drift_priors.{version}.json"),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find identity_drift_priors.{version}.json. Tried: {[str(p) for p in candidates]}"
    )


def load_drift_priors(version: str = "v1", force_reload: bool = False) -> Dict[str, Any]:
    """
    Load raw drift priors from JSON.

    Returns the full JSON structure including dimension_priors and group_modifiers.
    """
    global _PRIORS_CACHE

    cache_key = f"drift_{version}"
    if _PRIORS_CACHE is not None and cache_key in _PRIORS_CACHE and not force_reload:
        return _PRIORS_CACHE[cache_key]

    if _PRIORS_CACHE is None:
        _PRIORS_CACHE = {}

    priors_file = _find_priors_file(version)
    with open(priors_file, 'r') as f:
        data = json.load(f)

    _PRIORS_CACHE[cache_key] = data
    return data


def _apply_group_modifiers(
    base_delta_mu: float,
    base_sigma: float,
    modifiers: Dict[str, Any],
    profile: Dict[str, Any]
) -> Tuple[float, float]:
    """
    Apply demographic group modifiers to base delta_mu and sigma.

    Args:
        base_delta_mu: Base dimension delta_mu
        base_sigma: Base dimension sigma
        modifiers: Group modifier config from JSON
        profile: Agent's CES profile or demographic dict

    Returns:
        Adjusted (delta_mu, sigma) tuple
    """
    delta_mu_mult = 1.0
    sigma_mult = 1.0

    # Age modifier
    if "age" in modifiers:
        age_val = profile.get("cps21_age") or profile.get("age")
        if age_val is not None:
            try:
                age = float(age_val)
                if age < 30:
                    mod = modifiers["age"].get("young", {})
                elif age > 55:
                    mod = modifiers["age"].get("older", {})
                else:
                    mod = modifiers["age"].get("middle", {})
                delta_mu_mult *= mod.get("delta_mu_mult", 1.0)
                sigma_mult *= mod.get("sigma_mult", 1.0)
            except (ValueError, TypeError):
                pass

    # Education modifier
    if "education" in modifiers:
        edu_val = profile.get("cps21_education") or profile.get("education")
        if edu_val is not None:
            try:
                edu = float(edu_val)
                if edu <= 3:  # High school or less
                    mod = modifiers["education"].get("low", {})
                elif edu <= 6:
                    mod = modifiers["education"].get("medium", {})
                else:
                    mod = modifiers["education"].get("high", {})
                delta_mu_mult *= mod.get("delta_mu_mult", 1.0)
                sigma_mult *= mod.get("sigma_mult", 1.0)
            except (ValueError, TypeError):
                pass

    # Partisanship strength modifier
    if "partisanship_strength" in modifiers:
        pid_strength = profile.get("cps21_pid_strength") or profile.get("partisanship_strength")
        if pid_strength is not None:
            try:
                strength = float(pid_strength)
                if strength >= 3:  # Strong partisan
                    mod = modifiers["partisanship_strength"].get("strong", {})
                elif strength >= 1:
                    mod = modifiers["partisanship_strength"].get("weak", {})
                else:
                    mod = modifiers["partisanship_strength"].get("none", {})
                delta_mu_mult *= mod.get("delta_mu_mult", 1.0)
                sigma_mult *= mod.get("sigma_mult", 1.0)
            except (ValueError, TypeError):
                pass

    # Urban/rural modifier
    if "urban_rural" in modifiers:
        ur_val = profile.get("cps21_urban_rural") or profile.get("urban_rural")
        if ur_val is not None:
            try:
                ur = float(ur_val)
                if ur <= 1:  # Urban
                    mod = modifiers["urban_rural"].get("urban", {})
                elif ur <= 2:  # Suburban
                    mod = modifiers["urban_rural"].get("suburban", {})
                else:  # Rural
                    mod = modifiers["urban_rural"].get("rural", {})
                delta_mu_mult *= mod.get("delta_mu_mult", 1.0)
                sigma_mult *= mod.get("sigma_mult", 1.0)
            except (ValueError, TypeError):
                pass

    # Region modifier
    if "region" in modifiers:
        region = profile.get("Region") or profile.get("region")
        if region:
            region_str = str(region)
            if "Quebec" in region_str or "quebec" in region_str.lower():
                mod = modifiers["region"].get("Quebec", {})
            elif "West" in region_str or "BC" in region_str or "Alberta" in region_str:
                mod = modifiers["region"].get("West", {})
            elif "Atlantic" in region_str:
                mod = modifiers["region"].get("Atlantic", {})
            else:  # Default to Ontario
                mod = modifiers["region"].get("Ontario", {})
            delta_mu_mult *= mod.get("delta_mu_mult", 1.0)
            sigma_mult *= mod.get("sigma_mult", 1.0)

    return (base_delta_mu * delta_mu_mult, base_sigma * sigma_mult)


def compute_agent_drift_priors(
    agent_id: str,
    profile: Optional[Dict[str, Any]] = None,
    version: str = "v1"
) -> AgentDriftPriors:
    """
    Compute drift priors for a specific agent.

    Args:
        agent_id: Agent identifier
        profile: CES profile dict with demographic variables (optional)
        version: Priors version to load

    Returns:
        AgentDriftPriors with per-dimension and aggregate values
    """
    data = load_drift_priors(version)
    dim_priors_raw = data.get("dimension_priors", {})
    modifiers = data.get("group_modifiers", {})
    aggregate = data.get("aggregate_priors", {})

    if profile is None:
        profile = {}

    dimension_priors = {}
    delta_mus = []
    sigmas = []

    for dim in IDENTITY_DIMS:
        if dim in dim_priors_raw:
            base_mu = dim_priors_raw[dim].get("delta_mu", 0.07)
            base_sigma = dim_priors_raw[dim].get("sigma", 0.09)
            stability = dim_priors_raw[dim].get("stability_estimate", 0.65)

            # Apply group modifiers if profile provided
            adj_mu, adj_sigma = _apply_group_modifiers(
                base_mu, base_sigma, modifiers, profile
            )

            dimension_priors[dim] = DimensionDriftPrior(
                delta_mu=adj_mu,
                sigma=adj_sigma,
                stability_estimate=stability
            )
            delta_mus.append(adj_mu)
            sigmas.append(adj_sigma)
        else:
            # Default for missing dimensions
            dimension_priors[dim] = DimensionDriftPrior(
                delta_mu=0.07,
                sigma=0.09,
                stability_estimate=0.65
            )
            delta_mus.append(0.07)
            sigmas.append(0.09)

    # Compute aggregate as mean of dimensions
    agg_mu = sum(delta_mus) / len(delta_mus) if delta_mus else aggregate.get("overall_delta_mu", 0.069)
    agg_sigma = sum(sigmas) / len(sigmas) if sigmas else aggregate.get("overall_sigma", 0.090)

    return AgentDriftPriors(
        agent_id=agent_id,
        dimension_priors=dimension_priors,
        aggregate_delta_mu=agg_mu,
        aggregate_sigma=agg_sigma
    )


def get_dimension_drift(
    dim: str,
    profile: Optional[Dict[str, Any]] = None,
    version: str = "v1"
) -> Tuple[float, float]:
    """
    Get (delta_mu, sigma) for a single dimension.

    Convenience function for quick lookup without full AgentDriftPriors.

    Args:
        dim: Identity dimension name
        profile: Optional CES profile for group modifiers
        version: Priors version

    Returns:
        (delta_mu, sigma) tuple
    """
    data = load_drift_priors(version)
    dim_priors = data.get("dimension_priors", {})
    modifiers = data.get("group_modifiers", {})

    if dim not in dim_priors:
        return (0.07, 0.09)

    base_mu = dim_priors[dim].get("delta_mu", 0.07)
    base_sigma = dim_priors[dim].get("sigma", 0.09)

    if profile:
        return _apply_group_modifiers(base_mu, base_sigma, modifiers, profile)

    return (base_mu, base_sigma)


def get_aggregate_drift(
    profile: Optional[Dict[str, Any]] = None,
    version: str = "v1"
) -> Tuple[float, float]:
    """
    Get aggregate (delta_mu, sigma) across all dimensions.

    Args:
        profile: Optional CES profile for group modifiers
        version: Priors version

    Returns:
        (delta_mu, sigma) tuple
    """
    priors = compute_agent_drift_priors("aggregate", profile, version)
    return priors.get_aggregate()


# =============================================================================
# Summary/Debug Functions
# =============================================================================

def print_drift_summary(version: str = "v1"):
    """Print summary of drift priors for debugging."""
    data = load_drift_priors(version)
    dim_priors = data.get("dimension_priors", {})
    aggregate = data.get("aggregate_priors", {})

    print(f"Identity Drift Priors (v{version})")
    print("=" * 60)
    print(f"\nPer-Dimension Priors:")
    print(f"{'Dimension':<20} {'delta_mu':<10} {'sigma':<10} {'stability':<10}")
    print("-" * 50)

    for dim in IDENTITY_DIMS:
        if dim in dim_priors:
            p = dim_priors[dim]
            print(f"{dim:<20} {p['delta_mu']:<10.3f} {p['sigma']:<10.3f} {p['stability_estimate']:<10.2f}")

    print(f"\nAggregate: delta_mu={aggregate.get('overall_delta_mu', 'N/A'):.3f}, "
          f"sigma={aggregate.get('overall_sigma', 'N/A'):.3f}")

    # Show group modifier effects
    print(f"\nGroup Modifier Effects (multiplicative):")
    modifiers = data.get("group_modifiers", {})

    # Example: young vs older
    if "age" in modifiers:
        young = modifiers["age"].get("young", {})
        older = modifiers["age"].get("older", {})
        print(f"  Age: young={young.get('delta_mu_mult', 1.0):.2f}x, "
              f"older={older.get('delta_mu_mult', 1.0):.2f}x")

    if "partisanship_strength" in modifiers:
        strong = modifiers["partisanship_strength"].get("strong", {})
        none = modifiers["partisanship_strength"].get("none", {})
        print(f"  Partisanship: strong={strong.get('delta_mu_mult', 1.0):.2f}x, "
              f"none/independent={none.get('delta_mu_mult', 1.0):.2f}x")


if __name__ == "__main__":
    print_drift_summary()

    # Test with sample profile
    print("\n" + "=" * 60)
    print("Sample Agent Priors:")

    sample_profile = {
        "cps21_age": 25,  # Young
        "Region": "Quebec",
        "cps21_urban_rural": 1,  # Urban
        "cps21_pid_strength": 1,  # Weak partisan
    }

    priors = compute_agent_drift_priors("test_agent", sample_profile)
    print(f"\nAgent: {priors.agent_id}")
    print(f"Profile: Young (25), Quebec, Urban, Weak partisan")
    print(f"Aggregate: delta_mu={priors.aggregate_delta_mu:.4f}, sigma={priors.aggregate_sigma:.4f}")

    print(f"\nPer-dimension (with group modifiers applied):")
    for dim, dp in priors.dimension_priors.items():
        print(f"  {dim}: delta_mu={dp.delta_mu:.4f}, sigma={dp.sigma:.4f}")
