"""
CES Identity Prior Loader

Loads empirical group-level identity priors from CES 2021 data.
Used by:
- IdentityCore: To set empirical_delta_mu/sigma for emergent time calibration
- Grit v2: To set CES-accurate engagement targets per group

Data source: data/identity/identity_group_means_2021.csv
"""

import csv
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class IdentityPrior:
    """Empirical identity prior for a sociogeographic group."""
    group_id: str
    engagement_mu: float
    institutional_faith_mu: float
    social_friction_mu: float
    n: int  # Sample size for this group


# Cache for loaded priors
_PRIORS_CACHE: Optional[Dict[str, IdentityPrior]] = None


def _find_priors_file() -> Path:
    """Find the identity group means CSV file."""
    # Try multiple possible locations
    candidates = [
        Path(__file__).parent.parent.parent / "data" / "identity" / "identity_group_means_2021.csv",
        Path("data/identity/identity_group_means_2021.csv"),
        Path("identity_group_means_2021.csv"),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find identity_group_means_2021.csv. Tried: {[str(p) for p in candidates]}"
    )


def load_identity_priors(force_reload: bool = False) -> Dict[str, IdentityPrior]:
    """
    Load CES identity priors from CSV.

    Returns dict mapping group_id -> IdentityPrior.
    Group IDs are constructed from: Region_rural_urban_household

    Example group_ids:
    - "Atlantic_1_1" (Atlantic, urban code 1, household code 1)
    - "Ontario_3_2" (Ontario, rural code 3, household code 2)
    """
    global _PRIORS_CACHE

    if _PRIORS_CACHE is not None and not force_reload:
        return _PRIORS_CACHE

    priors = {}
    csv_path = _find_priors_file()

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Construct group_id from Region, rural_urban, household
            region = row.get('Region', 'Unknown')
            rural_urban = row.get('pes21_rural_urban', '0')
            household = row.get('cps21_household', '0')

            group_id = f"{region}_{rural_urban}_{household}"

            # Parse values
            try:
                prior = IdentityPrior(
                    group_id=group_id,
                    engagement_mu=float(row.get('mean_engagement', 0.5)),
                    institutional_faith_mu=float(row.get('mean_institutional_faith', 0.5)),
                    social_friction_mu=float(row.get('mean_social_friction', 0.3)),
                    n=int(float(row.get('N', 0))),
                )
                priors[group_id] = prior
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not parse row for {group_id}: {e}")
                continue

    _PRIORS_CACHE = priors
    return priors


def get_engagement_target(group_id: str, default: float = 0.25) -> float:
    """
    Get CES-calibrated engagement target for a group.

    Used by grit v2 to set data-driven targets instead of hard-coded values.

    Args:
        group_id: Sociogeographic group identifier
        default: Fallback if group not found

    Returns:
        Target engagement level (0-1)
    """
    priors = load_identity_priors()

    # Try exact match first
    if group_id in priors:
        return priors[group_id].engagement_mu

    # Try fuzzy matching on region
    for key, prior in priors.items():
        if group_id.lower() in key.lower() or key.lower() in group_id.lower():
            return prior.engagement_mu

    return default


def get_identity_prior(group_id: str) -> Optional[IdentityPrior]:
    """Get full identity prior for a group."""
    priors = load_identity_priors()
    return priors.get(group_id)


def get_empirical_bounds() -> Dict[str, Tuple[float, float]]:
    """
    Get empirical min/max for each identity dimension across all groups.

    Useful for understanding the range of "normal" values.

    Returns:
        Dict with keys 'engagement', 'institutional_faith', 'social_friction'
        and values (min, max).
    """
    priors = load_identity_priors()

    if not priors:
        return {
            'engagement': (0.0, 1.0),
            'institutional_faith': (0.0, 1.0),
            'social_friction': (0.0, 1.0),
        }

    engagements = [p.engagement_mu for p in priors.values()]
    faiths = [p.institutional_faith_mu for p in priors.values()]
    frictions = [p.social_friction_mu for p in priors.values()]

    return {
        'engagement': (min(engagements), max(engagements)),
        'institutional_faith': (min(faiths), max(faiths)),
        'social_friction': (min(frictions), max(frictions)),
    }


def map_agent_to_ces_group(agent_id: str) -> str:
    """
    Map an agent identifier to a CES group_id.

    This is a heuristic mapping from agent names like "CES_Urban_Progressive"
    to CES group codes. In practice, agents should carry their CES group_id
    directly from sampling.

    Args:
        agent_id: Agent identifier string

    Returns:
        Best-guess CES group_id
    """
    aid_lower = agent_id.lower()

    # Default region (Ontario is most common)
    region = "Ontario"
    if 'atlantic' in aid_lower or 'nova' in aid_lower or 'newfoundland' in aid_lower:
        region = "Atlantic"
    elif 'quebec' in aid_lower:
        region = "Quebec"
    elif 'prairie' in aid_lower or 'alberta' in aid_lower or 'saskatchewan' in aid_lower:
        region = "West"
    elif 'bc' in aid_lower or 'british columbia' in aid_lower:
        region = "West"

    # Rural/urban (codes 1-5 in CES)
    # 1 = large urban, 5 = rural
    if 'rural' in aid_lower:
        rural_urban = "5.0"
    elif 'suburban' in aid_lower:
        rural_urban = "3.0"
    else:  # default urban
        rural_urban = "1.0"

    # Household (simplified - using single/couple as proxy)
    # In CES: 1=single, 2=couple no kids, 3-4=couple with kids, etc.
    if 'single' in aid_lower or 'renter' in aid_lower:
        household = "1.0"
    else:
        household = "2.0"

    return f"{region}_{rural_urban}_{household}"


# =============================================================================
# Summary Statistics
# =============================================================================

def print_prior_summary():
    """Print summary of loaded priors for debugging."""
    priors = load_identity_priors()
    bounds = get_empirical_bounds()

    print(f"Loaded {len(priors)} CES identity priors")
    print(f"\nEmpirical bounds:")
    for dim, (lo, hi) in bounds.items():
        print(f"  {dim}: [{lo:.3f}, {hi:.3f}]")

    print(f"\nSample groups:")
    for i, (group_id, prior) in enumerate(list(priors.items())[:5]):
        print(f"  {group_id}: eng={prior.engagement_mu:.3f}, faith={prior.institutional_faith_mu:.3f}, N={prior.n}")

    # Find lowest and highest engagement groups
    sorted_by_eng = sorted(priors.values(), key=lambda p: p.engagement_mu)

    print(f"\nLowest engagement groups:")
    for p in sorted_by_eng[:3]:
        print(f"  {p.group_id}: {p.engagement_mu:.3f} (N={p.n})")

    print(f"\nHighest engagement groups:")
    for p in sorted_by_eng[-3:]:
        print(f"  {p.group_id}: {p.engagement_mu:.3f} (N={p.n})")


if __name__ == "__main__":
    print_prior_summary()
