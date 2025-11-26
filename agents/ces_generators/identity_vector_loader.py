"""
Identity Vector Loader - Computes 7D identity vectors from CES profiles.

Uses identity_weights_2021.v1.json to map CES variables to canonical IDENTITY_DIMS.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

# Canonical dimension order (must match identity_core/core.py IDENTITY_DIMS)
IDENTITY_DIMS = [
    "engagement", "institutional_faith", "ideology",
    "partisanship", "sociogeographic", "social_friction", "tie_to_place"
]


def load_weights(version: str = "v1") -> Dict[str, Any]:
    """Load identity weights file."""
    weights_dir = Path(__file__).parent.parent.parent / "data" / "identity"
    weights_file = weights_dir / f"identity_weights_2021.{version}.json"

    if not weights_file.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_file}")

    with open(weights_file, "r") as f:
        return json.load(f)


def _normalize_value(raw_value: Any, transform: str, max_val: float = 10.0) -> float:
    """Apply transform to normalize a CES variable value to 0-1."""
    if raw_value is None:
        return 0.5  # Default neutral

    try:
        val = float(raw_value)
    except (ValueError, TypeError):
        return 0.5

    if transform == "binary" or transform == "binary_high":
        return 1.0 if val == 1 else 0.0
    elif transform == "normalize_10":
        return min(1.0, max(0.0, val / 10.0))
    elif transform == "normalize_4":
        return min(1.0, max(0.0, val / 4.0))
    elif transform == "normalize":
        return min(1.0, max(0.0, val / max_val))
    elif transform == "inverse":
        return 1.0 - min(1.0, max(0.0, val / max_val))
    elif transform == "partisan_strength":
        # Party ID: 2,3 = strong (1.0), 1,4,5,6 = moderate (0.6), 7,8 = weak (0.2)
        if val in [2, 3]:
            return 1.0
        elif val in [1, 4, 5, 6]:
            return 0.6
        else:
            return 0.2
    elif transform == "urban_rural_scale":
        # 1=Urban (0.3), 2=Suburban (0.6), 3=Rural (1.0)
        return {1: 0.3, 2: 0.6, 3: 1.0}.get(int(val), 0.5)
    elif transform == "rural_bonus":
        # Same as urban_rural_scale for tie_to_place
        return {1: 0.4, 2: 0.6, 3: 1.0}.get(int(val), 0.5)
    elif transform == "age_stability":
        # Age from year of birth (assuming 2021 survey)
        age = 2021 - val if val > 1900 else val
        if age >= 55:
            return 1.0
        elif age >= 40:
            return 0.7
        elif age >= 30:
            return 0.5
        else:
            return 0.3
    elif transform == "middle_class_stability":
        # Income categories: higher = more stable
        if val >= 6:
            return 0.8
        elif val >= 4:
            return 0.6
        else:
            return 0.3
    elif transform == "regional_weight":
        # Province as regional identity (all provinces have identity)
        return 0.7  # Default moderate provincial identity
    elif transform == "binary_french":
        return 1.0 if val == 2 else 0.0
    elif transform == "generational":
        # Parents born in Canada = higher tie
        return 1.0 if val == 0 else 0.5
    elif transform == "consistency":
        # Vote choice consistency (presence of choice = engaged)
        return 0.8 if val is not None and val > 0 else 0.3
    elif transform == "competitiveness":
        # Riding competitiveness (placeholder - would need riding data)
        return 0.5
    elif transform == "categorical":
        # Categorical presence indicator
        return 0.7 if val is not None else 0.3
    elif transform == "local_education":
        # Education level (lower = more local ties typically)
        if val <= 3:  # High school or less
            return 0.7
        elif val <= 5:  # Some college/trade
            return 0.5
        else:  # University+
            return 0.3
    elif transform == "regional_identity":
        # Province code to regional identity
        return 0.6  # Moderate default
    else:
        # Unknown transform - return normalized value
        return min(1.0, max(0.0, val / max_val))


def compute_identity_vector(
    profile: Dict[str, Any],
    weights: Optional[Dict[str, Any]] = None,
    version: str = "v1"
) -> Dict[str, float]:
    """
    Compute 7D identity vector from CES profile using weights.

    Args:
        profile: CES profile dict with variable values
        weights: Pre-loaded weights dict (optional, will load if None)
        version: Weights version to use if loading

    Returns:
        Dict with all 7 canonical dimensions (0-1 values)
    """
    if weights is None:
        try:
            weights = load_weights(version)
        except FileNotFoundError:
            # Fallback to defaults if weights not found
            return {dim: 0.5 for dim in IDENTITY_DIMS}

    result = {}

    for dim in IDENTITY_DIMS:
        if dim not in weights:
            result[dim] = 0.5
            continue

        dim_config = weights[dim]
        variables = dim_config.get("variables", [])

        if not variables:
            result[dim] = 0.5
            continue

        weighted_sum = 0.0
        total_weight = 0.0

        for var_spec in variables:
            var_name = var_spec.get("variable")
            weight = var_spec.get("weight", 0.1)
            transform = var_spec.get("transform", "normalize")

            if var_name in profile:
                raw_value = profile.get(var_name)
                normalized = _normalize_value(raw_value, transform)
                weighted_sum += normalized * weight
                total_weight += weight

        if total_weight > 0:
            result[dim] = round(weighted_sum / total_weight, 4)
        else:
            result[dim] = 0.5

    return result


def profile_to_identity_dict(profile: Dict[str, Any], version: str = "v1") -> Dict[str, float]:
    """
    Convenience function to compute 7D identity dict from CES profile.

    This is the main entry point for row_to_agent integration.
    """
    return compute_identity_vector(profile, version=version)


# Example usage
if __name__ == "__main__":
    # Test with sample CES profile
    sample_profile = {
        "cps21_turnout": 1,
        "cps21_lr_scale": 3.0,  # Left-leaning
        "cps21_pid_party": 3,   # NDP
        "cps21_urban_rural": 1,  # Urban
        "cps21_demsat": 3,
        "cps21_bornin_canada": 1,
        "cps21_yob": 1990,
        "cps21_income_cat": 4,
    }

    identity = compute_identity_vector(sample_profile)
    print("Sample CES Profile -> 7D Identity Vector:")
    for dim, val in identity.items():
        print(f"  {dim}: {val:.3f}")
