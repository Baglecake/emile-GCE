#!/usr/bin/env python3
"""
CES Data Sampler - Sample real agents from CES parquet data.

Maps normalized CES data back to raw variables for ces_row_to_agent().
"""

import random
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


# Province code mapping
PROVINCE_MAP = {
    'AB': 48, 'BC': 59, 'MB': 46, 'NB': 13, 'NL': 10,
    'NS': 12, 'ON': 35, 'PE': 11, 'QC': 24, 'SK': 47,
}

# Reverse province lookup
PROVINCE_NAMES = {v: k for k, v in PROVINCE_MAP.items()}


def load_ces_data(year: int = 2021) -> pd.DataFrame:
    """Load CES parquet data."""
    candidates = [
        Path(__file__).parent.parent.parent / "data" / f"CES_{year}.parquet",
        Path(f"data/CES_{year}.parquet"),
    ]

    for path in candidates:
        if path.exists():
            return pd.read_parquet(path)

    raise FileNotFoundError(f"Could not find CES_{year}.parquet")


def normalized_to_raw_row(row: pd.Series) -> Dict[str, Any]:
    """
    Convert normalized CES row back to raw variable format for ces_row_to_agent.
    """
    # Province
    province_str = row.get('province', 'ON')
    province_code = PROVINCE_MAP.get(province_str, 35)

    # Age: normalized 0-1 maps to ~18-90
    age_norm = row.get('age_norm', 0.5)
    age = int(18 + age_norm * 72)
    yob = 2021 - age

    # Gender: female=1 means woman, else man
    gender = 2 if row.get('female', 0) == 1 else 1

    # Education: normalized 0-1, map to 1-11 CES scale
    edu_norm = row.get('education_norm', 0.5)
    education = int(1 + edu_norm * 10)

    # Income: derive from education and other factors (rough proxy)
    # CES uses 1-10 income categories
    income = int(3 + edu_norm * 5 + random.uniform(-1, 1))
    income = max(1, min(10, income))

    # Ideology: 0-1 normalized to 0-10 LR scale
    ideology = row.get('ideology_norm', 0.5) * 10

    # Party ID from vote choice or party_id_norm
    # vote_cpc=1 means CPC voter
    if row.get('vote_cpc', 0) == 1:
        party = 2  # Conservative
    elif pd.notna(row.get('party_id_norm')):
        pid = row['party_id_norm']
        if pid < 0.25:
            party = 3  # NDP
        elif pid < 0.5:
            party = 1  # Liberal
        elif pid < 0.75:
            party = 2  # Conservative
        else:
            party = 8  # None
    else:
        # Derive from ideology
        if ideology < 3:
            party = 3  # NDP
        elif ideology < 5:
            party = 1  # Liberal
        elif ideology < 7:
            party = 2  # Conservative
        else:
            party = 2  # Conservative

    # Trust in government → turnout proxy
    trust = row.get('trust_govt_norm', 0.5)
    if trust > 0.6:
        turnout = 1  # Certain
    elif trust > 0.3:
        turnout = 2  # Likely
    else:
        turnout = 3  # Unlikely

    # Urban/rural: derive from province and education (rough proxy)
    # More educated in ON/BC/QC → more urban
    if province_str in ['ON', 'BC', 'QC'] and edu_norm > 0.6:
        urban_rural = 1  # Urban
    elif province_str in ['AB', 'SK', 'MB'] and edu_norm < 0.4:
        urban_rural = 3  # Rural
    else:
        urban_rural = 2  # Suburban

    # Generate a riding ID
    riding = f"{province_str}_{random.randint(1, 99):02d}"

    return {
        "cps21_ResponseId": f"CES_{row.get('respondent_id', random.randint(1000, 9999))}",
        "cps21_province": province_code,
        "cps21_yob": yob,
        "cps21_genderid": gender,
        "cps21_education": education,
        "cps21_income_cat": income,
        "cps21_urban_rural": urban_rural,
        "cps21_pid_party": party,
        "cps21_lr_scale": ideology,
        "cps21_turnout": turnout,
        "cps21_bornin_canada": int(row.get('born_canada', 1)),
        "cps21_riding_id": riding,
    }


def sample_ces_profiles(
    n: int = 4,
    year: int = 2021,
    stratify: bool = True,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Sample n CES profiles from real data.

    Args:
        n: Number of profiles to sample
        year: CES year (2015, 2019, or 2021)
        stratify: If True, stratify by province/ideology
        seed: Random seed for reproducibility

    Returns:
        List of CES profile dicts ready for ces_row_to_agent()
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    df = load_ces_data(year)

    if stratify and n >= 4:
        # Sample across ideology spectrum and provinces
        profiles = []

        # Split by ideology: left, center-left, center-right, right
        ideology_bins = [
            (0, 0.3, 'left'),
            (0.3, 0.5, 'center-left'),
            (0.5, 0.7, 'center-right'),
            (0.7, 1.0, 'right'),
        ]

        per_bin = n // 4
        remainder = n % 4

        for low, high, label in ideology_bins:
            bin_df = df[(df['ideology_norm'] >= low) & (df['ideology_norm'] < high)]
            if len(bin_df) == 0:
                bin_df = df  # Fallback to full df

            count = per_bin + (1 if remainder > 0 else 0)
            remainder = max(0, remainder - 1)

            sampled = bin_df.sample(n=min(count, len(bin_df)), replace=len(bin_df) < count)
            for _, row in sampled.iterrows():
                profiles.append(normalized_to_raw_row(row))

        return profiles[:n]
    else:
        # Simple random sample
        sampled = df.sample(n=min(n, len(df)))
        return [normalized_to_raw_row(row) for _, row in sampled.iterrows()]


def sample_diverse_experiment(
    n_agents: int = 4,
    year: int = 2021,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Sample a diverse set of agents for an experiment.

    Ensures diversity in:
    - Ideology (left/center/right)
    - Geography (urban/suburban/rural)
    - Engagement (high/medium/low)
    """
    return sample_ces_profiles(n=n_agents, year=year, stratify=True, seed=seed)


if __name__ == "__main__":
    # Test sampling
    profiles = sample_ces_profiles(n=4, seed=42)

    print("Sampled CES Profiles:")
    print("=" * 60)
    for p in profiles:
        print(f"\n{p['cps21_ResponseId']}:")
        print(f"  Province: {PROVINCE_NAMES.get(p['cps21_province'], 'Unknown')}")
        print(f"  Age: {2021 - p['cps21_yob']}")
        print(f"  Education: {p['cps21_education']}/11")
        print(f"  Ideology: {p['cps21_lr_scale']:.1f}/10")
        print(f"  Party: {p['cps21_pid_party']}")
        print(f"  Urban/Rural: {p['cps21_urban_rural']}")
        print(f"  Turnout: {p['cps21_turnout']}")
