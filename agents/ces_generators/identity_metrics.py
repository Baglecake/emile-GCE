"""
Identity Salience and Tie-to-Place Metrics

Derives Weber-inspired identity metrics from CES profiles.
Handles both raw CES 2021 codes (cps21_*) and normalized Parquet variables.

These metrics capture:
- identity_salience: strength of political identity and engagement (The "Potential for Action")
- tie_to_place: rootedness in geographic and social context (The "Potential for Resistance")

Based on:
- Weber's "tie to place" theory
- Gemini's "Vector Gap" analysis (gemini_on_vectors)
- Social Aesthetics paper Section 4
"""

from typing import Dict, Any, Tuple


def compute_identity_salience(profile: Dict[str, Any]) -> float:
    """
    Compute identity salience (0-1) from CES profile.

    High salience indicators:
    - Strong party identification (not None/Independent)
    - High turnout (voted in last election)
    - Clear ideological position (not centrist)
    - High political interest (if available)

    Args:
        profile: CES profile dict with keys like cps21_pid_party, cps21_turnout, etc.
                 OR normalized parquet keys like party_id_norm, turnout_2021, etc.

    Returns:
        Float 0-1 representing identity salience
    """
    score = 0.0
    max_score = 0.0

    # 1. Party Identification Strength
    # Check for raw CES 2021 codes
    if 'cps21_pid_party' in profile:
        party = profile.get('cps21_pid_party')
        if party in [2, 3]:  # Conservative or NDP (strong partisan identity)
            score += 1.0
        elif party in [1, 4, 5, 6]:  # Lib/Bloc/Green/PPC (moderate identity)
            score += 0.6
        elif party in [7, 8]:  # Other/None (weak identity)
            score += 0.2
        max_score += 1.0
    # Check for normalized parquet data
    elif 'party_id_norm' in profile:
        pid_norm = profile.get('party_id_norm')
        if pid_norm is not None:
            score += float(pid_norm)
            max_score += 1.0

    # 2. Turnout (Actual Voting Behavior)
    if 'cps21_turnout' in profile:
        turnout = profile.get('cps21_turnout')
        if turnout == 1:  # Voted
            score += 1.0
        elif turnout == 2:  # Didn't vote but could have
            score += 0.3
        else:  # Not eligible or didn't answer
            score += 0.1
        max_score += 1.0
    elif 'cps25_turnout_2021' in profile:  # CSV variable
        turnout = profile.get('cps25_turnout_2021')
        if turnout == 1:
            score += 1.0
        else:
            score += 0.3
        max_score += 1.0

    # 3. Ideological Clarity (Distance from Center)
    if 'cps21_lr_scale' in profile:
        lr_scale = profile.get('cps21_lr_scale')
        if lr_scale is not None:
            # Scale is 0-10, center is 5
            ideological_distance = abs(lr_scale - 5.0) / 5.0  # 0-1
            score += ideological_distance
            max_score += 1.0
    elif 'ideology_norm' in profile:  # Parquet normalized
        lr_norm = profile.get('ideology_norm')
        if lr_norm is not None:
            # 0.5 is center, 0 or 1 is extreme
            ideological_distance = abs(lr_norm - 0.5) * 2.0  # 0-1
            score += ideological_distance
            max_score += 1.0

    # 4. Political Interest (NEW - from CES selected variables)
    if 'cps25_interest_gen_1' in profile:
        interest = profile.get('cps25_interest_gen_1')
        if interest is not None:
            score += (interest / 10.0)  # Normalize 0-10 to 0-1
            max_score += 1.0

    # 5. Affective Party ID (NEW - emotional connection)
    if 'cps25_aff_pid' in profile:
        aff_pid = profile.get('cps25_aff_pid')
        if aff_pid is not None:
            # Assuming this is on a scale we can normalize
            score += float(aff_pid)
            max_score += 1.0

    if max_score == 0:
        return 0.5  # Default fallback

    return score / max_score


def compute_tie_to_place(profile: Dict[str, Any]) -> float:
    """
    Compute tie to place (0-1) from CES profile.

    High tie indicators:
    - Rural residence (stronger community ties)
    - Born in Canada (longer settlement)
    - Older age (more established)
    - Homeownership proxy (higher income in non-urban areas)
    - Generational rootedness (parents born in Canada)

    Args:
        profile: CES profile dict

    Returns:
        Float 0-1 representing tie to place
    """
    score = 0.0
    max_score = 0.0

    # 1. Urban-rural (rural = higher tie to specific place)
    if 'cps21_urban_rural' in profile:
        urban_rural = profile.get('cps21_urban_rural')
        if urban_rural == 3:  # Rural
            score += 1.0
        elif urban_rural == 2:  # Suburban
            score += 0.6
        else:  # Urban
            score += 0.4  # Urban can still have strong neighborhood ties
        max_score += 1.0

    # 2. Born in Canada (longer settlement = stronger ties)
    if 'cps21_bornin_canada' in profile:
        born_canada = profile.get('cps21_bornin_canada')
        if born_canada == 1:
            score += 1.0
        else:
            score += 0.4
        max_score += 1.0
    elif 'born_canada' in profile:  # Parquet
        born = profile.get('born_canada')
        if born is not None:
            score += float(born)
            max_score += 1.0

    # 3. Age (older = more established)
    if 'cps21_yob' in profile:
        yob = profile.get('cps21_yob')
        if yob is not None:
            age = 2021 - yob
            if age >= 55:
                score += 1.0
            elif age >= 40:
                score += 0.7
            elif age >= 30:
                score += 0.4
            else:
                score += 0.2
            max_score += 1.0
    elif 'age_norm' in profile:  # Parquet
        age_norm = profile.get('age_norm')
        if age_norm is not None:
            score += float(age_norm)
            max_score += 1.0

    # 4. Income as stability proxy (middle-high income = more settled)
    if 'cps21_income_cat' in profile:
        income = profile.get('cps21_income_cat')
        if income is not None:
            if income >= 6:  # Higher income
                score += 0.8
            elif income >= 4:  # Middle income
                score += 0.6
            else:  # Lower income
                score += 0.3
            max_score += 1.0

    # 5. Generational Rootedness (NEW - parents born outside Canada)
    if 'pes25_parents_born' in profile:
        parents_born = profile.get('pes25_parents_born')
        if parents_born == 0:  # Parents born in Canada
            score += 1.0
        else:
            score += 0.5
        max_score += 1.0

    if max_score == 0:
        return 0.5  # Default fallback

    return score / max_score


def compute_identity_metrics(profile: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute all identity metrics for a CES profile.

    Returns dict with:
    - identity_salience: 0-1
    - tie_to_place: 0-1
    - combined_identity: geometric mean of both (for G-identity condition)
    """
    salience = compute_identity_salience(profile)
    tie = compute_tie_to_place(profile)

    # Combined metric for identifying "strongly rooted" agents
    combined = (salience * tie) ** 0.5  # Geometric mean

    return {
        'identity_salience': round(salience, 3),
        'tie_to_place': round(tie, 3),
        'combined_identity': round(combined, 3)
    }


def get_identity_category(metrics: Dict[str, float]) -> str:
    """
    Categorize agent based on identity metrics.

    Categories:
    - "rooted_partisan": high salience + high tie (Rural Conservative archetype)
    - "urban_engaged": high salience + moderate tie (Urban Progressive archetype)
    - "settled_swing": low salience + moderate/high tie (Suburban Swing archetype)
    - "unanchored": low salience + low tie (Disengaged Renter archetype)
    """
    salience = metrics['identity_salience']
    tie = metrics['tie_to_place']

    if salience >= 0.6 and tie >= 0.6:
        return "rooted_partisan"
    elif salience >= 0.6 and tie < 0.6:
        return "urban_engaged"
    elif salience < 0.6 and tie >= 0.5:
        return "settled_swing"
    else:
        return "unanchored"


def needs_grit_constraint(metrics: Dict[str, float], threshold: float = 0.3) -> bool:
    """
    Determine if agent needs "grit" constraint to prevent hyper-enfranchisement.

    Based on Gemini's "Vector Gap" analysis (gemini_on_vectors):
    - Low-salience agents should NOT act like model citizens
    - They should be skeptical, non-committal, resistant to engagement
    - Without grit, LLMs exhibit "Toxic Positivity" (inherent helpfulness bias)

    Args:
        metrics: Output from compute_identity_metrics()
        threshold: Salience threshold below which grit is needed (default 0.3)

    Returns:
        True if identity_salience < threshold
    """
    return metrics['identity_salience'] < threshold


# Example usage and test
if __name__ == "__main__":
    # Test with the 4 standard CES agents
    test_profiles = [
        {
            "cps21_ResponseId": "CES_Urban_Progressive",
            "cps21_province": 35,
            "cps21_yob": 1995,
            "cps21_genderid": 2,
            "cps21_education": 9,
            "cps21_income_cat": 4,
            "cps21_urban_rural": 1,
            "cps21_pid_party": 3,  # NDP
            "cps21_lr_scale": 2.5,
            "cps21_turnout": 1,
            "cps21_bornin_canada": 1,
        },
        {
            "cps21_ResponseId": "CES_Suburban_Swing",
            "cps21_province": 35,
            "cps21_yob": 1975,
            "cps21_genderid": 1,
            "cps21_education": 7,
            "cps21_income_cat": 7,
            "cps21_urban_rural": 2,
            "cps21_pid_party": 8,  # None
            "cps21_lr_scale": 5.5,
            "cps21_turnout": 2,
            "cps21_bornin_canada": 1,
        },
        {
            "cps21_ResponseId": "CES_Rural_Conservative",
            "cps21_province": 35,
            "cps21_yob": 1960,
            "cps21_genderid": 1,
            "cps21_education": 5,
            "cps21_income_cat": 5,
            "cps21_urban_rural": 3,
            "cps21_pid_party": 2,  # Conservative
            "cps21_lr_scale": 7.5,
            "cps21_turnout": 1,
            "cps21_bornin_canada": 1,
        },
        {
            "cps21_ResponseId": "CES_Disengaged_Renter",
            "cps21_province": 35,
            "cps21_yob": 1998,
            "cps21_genderid": 3,
            "cps21_education": 8,
            "cps21_income_cat": 2,
            "cps21_urban_rural": 1,
            "cps21_pid_party": 8,  # None
            "cps21_lr_scale": 4.0,
            "cps21_turnout": 3,
            "cps21_bornin_canada": 1,
        }
    ]

    print("Identity Metrics for CES Agents")
    print("=" * 70)
    for profile in test_profiles:
        metrics = compute_identity_metrics(profile)
        category = get_identity_category(metrics)
        needs_grit = needs_grit_constraint(metrics)
        print(f"\n{profile['cps21_ResponseId']}:")
        print(f"  identity_salience: {metrics['identity_salience']:.3f}")
        print(f"  tie_to_place:      {metrics['tie_to_place']:.3f}")
        print(f"  combined_identity: {metrics['combined_identity']:.3f}")
        print(f"  category:          {category}")
        print(f"  needs_grit:        {needs_grit}")
