"""
CES Row-to-Agent Transformer

Dynamically generates agent configurations from CES 2021 survey data.
This is NOT a static persona library - agents are constructed on-the-fly
from empirical survey responses.

Key CES 2021 Variables Used:
- Demographics: cps21_province, cps21_age, cps21_genderid, cps21_education, cps21_income_cat
- Identity: cps21_religion, cps21_bornin_canada, cps21_language
- Political: cps21_pid_party, cps21_lr_scale, cps21_votechoice, cps21_turnout
- Attitudes: cps21_most_important_issue, pes21_groups thermometers
- Geographic: cps21_riding_id, cps21_urban_rural

Usage:
    from agents.ces_generators import ces_row_to_agent, CESVariableMapper

    mapper = CESVariableMapper()
    agent = ces_row_to_agent(ces_row, mapper)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import json

# Import identity metrics for grit constraint detection
from .identity_metrics import compute_identity_metrics, needs_grit_constraint


# =============================================================================
# CES Variable Mappings
# =============================================================================

class CESProvince(Enum):
    """Province codes from CES 2021."""
    NL = 10
    PE = 11
    NS = 12
    NB = 13
    QC = 24
    ON = 35
    MB = 46
    SK = 47
    AB = 48
    BC = 59
    YT = 60
    NT = 61
    NU = 62


class CESParty(Enum):
    """Party identification codes from CES 2021."""
    LIBERAL = 1
    CONSERVATIVE = 2
    NDP = 3
    BLOC = 4
    GREEN = 5
    PPC = 6
    OTHER = 7
    NONE = 8


PROVINCE_NAMES = {
    10: "Newfoundland and Labrador",
    11: "Prince Edward Island",
    12: "Nova Scotia",
    13: "New Brunswick",
    24: "Quebec",
    35: "Ontario",
    46: "Manitoba",
    47: "Saskatchewan",
    48: "Alberta",
    59: "British Columbia",
    60: "Yukon",
    61: "Northwest Territories",
    62: "Nunavut"
}

PARTY_NAMES = {
    1: "Liberal",
    2: "Conservative",
    3: "NDP",
    4: "Bloc Quebecois",
    5: "Green",
    6: "People's Party",
    7: "Other",
    8: "None/Independent"
}

EDUCATION_LEVELS = {
    1: "No schooling",
    2: "Some elementary",
    3: "Completed elementary",
    4: "Some secondary/high school",
    5: "Completed secondary/high school",
    6: "Some technical/community college",
    7: "Completed technical/community college",
    8: "Some university",
    9: "Bachelor's degree",
    10: "Master's degree",
    11: "Professional degree or doctorate"
}


# =============================================================================
# CES Agent Configuration
# =============================================================================

@dataclass
class CESAgentConfig:
    """
    Agent configuration derived from CES survey data.

    This is the bridge between raw CES variables and the Social RL canvas format.
    It preserves both:
    1. Original CES variables (for validation against actual distributions)
    2. Generated persona/constraints (for simulation)
    """
    # Source identification
    source_type: str  # "respondent" or "cluster"
    source_id: str    # CES respondent ID or cluster ID

    # Demographics (from CES)
    province: int
    province_name: str
    age_group: str
    gender: str
    education: int
    education_label: str
    income_quintile: int
    urban_rural: str

    # Identity markers (from CES)
    religion: Optional[str] = None
    born_in_canada: bool = True
    visible_minority: bool = False
    language: str = "English"

    # Political disposition (from CES)
    party_id: int = 8  # Default: None
    party_name: str = "None/Independent"
    ideology_lr: float = 5.0  # 0-10 scale, 5 = center
    turnout_likelihood: float = 0.5

    # Mutable state (can change during simulation)
    current_vote_intention: Optional[int] = None
    issue_salience: Dict[str, float] = field(default_factory=dict)

    # Socio-geographic context (populated from riding data)
    riding_id: Optional[str] = None
    riding_competitiveness: float = 0.5
    local_minority_pct: float = 0.0

    # Generated persona (compiled from above)
    persona_description: str = ""
    constraints: List[str] = field(default_factory=list)

    def to_canvas_agent(self) -> Dict[str, Any]:
        """Convert to Social RL canvas agent format."""
        return {
            "identifier": f"CES_{self.source_id}",
            "source": {
                "type": self.source_type,
                "id": self.source_id,
                "dataset": "CES_2021"
            },
            "attributes": {
                "province": self.province,
                "province_name": self.province_name,
                "age_group": self.age_group,
                "gender": self.gender,
                "education": self.education,
                "education_label": self.education_label,
                "income_quintile": self.income_quintile,
                "urban_rural": self.urban_rural,
                "ideology_lr": self.ideology_lr,
                "party_id": self.party_id,
                "party_name": self.party_name,
                "turnout_likelihood": self.turnout_likelihood
            },
            "persona": self.persona_description,
            "goal": self._generate_goal(),
            "constraints": self.constraints,
            "mutable_state": {
                "current_vote_intention": self.current_vote_intention,
                "issue_salience": self.issue_salience
            }
        }

    def _generate_goal(self) -> str:
        """Generate contextual goal based on agent attributes."""
        goals = []

        if self.turnout_likelihood < 0.4:
            goals.append("deciding whether voting is worth my time")
        elif self.turnout_likelihood > 0.8:
            goals.append("ensuring my voice is heard in this election")

        if self.party_id == 8:  # No party ID
            goals.append("figuring out which party, if any, represents my interests")

        if self.income_quintile <= 2:
            goals.append("finding a party that addresses economic pressures I face")

        return "; ".join(goals) if goals else "participating meaningfully in democratic discourse"


# =============================================================================
# CES Variable Mapper
# =============================================================================

class CESVariableMapper:
    """
    Maps CES 2021 variable codes to meaningful values.

    This handles the translation between raw CES codes and
    human-interpretable attributes for agent generation.
    """

    def __init__(self, codebook_path: Optional[str] = None):
        """
        Initialize the mapper.

        Args:
            codebook_path: Path to CES codebook (for extended mappings)
        """
        self.codebook_path = codebook_path
        self._custom_mappings: Dict[str, Callable] = {}

    def map_province(self, value: int) -> tuple:
        """Map province code to (code, name)."""
        return value, PROVINCE_NAMES.get(value, "Unknown")

    def map_party(self, value: int) -> tuple:
        """Map party ID code to (code, name)."""
        return value, PARTY_NAMES.get(value, "Unknown")

    def map_education(self, value: int) -> tuple:
        """Map education code to (code, label)."""
        return value, EDUCATION_LEVELS.get(value, "Unknown")

    def map_age_to_group(self, year_of_birth: int, survey_year: int = 2021) -> str:
        """Convert year of birth to age group."""
        age = survey_year - year_of_birth
        if age < 25:
            return "18-24"
        elif age < 35:
            return "25-34"
        elif age < 45:
            return "35-44"
        elif age < 55:
            return "45-54"
        elif age < 65:
            return "55-64"
        else:
            return "65+"

    def map_gender(self, value: int) -> str:
        """Map gender code."""
        mapping = {1: "Man", 2: "Woman", 3: "Non-binary"}
        return mapping.get(value, "Not specified")

    def map_lr_scale(self, value: float) -> str:
        """Map left-right scale to label."""
        if value is None or value < 0:
            return "Not placed"
        elif value < 3:
            return "Left"
        elif value < 4:
            return "Centre-left"
        elif value < 6:
            return "Centre"
        elif value < 7:
            return "Centre-right"
        else:
            return "Right"

    def map_urban_rural(self, value: int) -> str:
        """Map urban/rural code."""
        mapping = {1: "Urban", 2: "Suburban", 3: "Rural"}
        return mapping.get(value, "Unknown")

    def map_income_to_quintile(self, income_cat: int) -> int:
        """Map income category to quintile (1-5)."""
        if income_cat <= 2:
            return 1
        elif income_cat <= 4:
            return 2
        elif income_cat <= 6:
            return 3
        elif income_cat <= 8:
            return 4
        else:
            return 5

    def register_custom_mapping(self, variable: str, mapper: Callable):
        """Register a custom mapping function for a variable."""
        self._custom_mappings[variable] = mapper

    def apply_custom(self, variable: str, value: Any) -> Any:
        """Apply custom mapping if registered."""
        if variable in self._custom_mappings:
            return self._custom_mappings[variable](value)
        return value


# =============================================================================
# Dynamic Agent Generation Functions
# =============================================================================

def ces_row_to_agent(
    row: Dict[str, Any],
    mapper: CESVariableMapper,
    include_persona: bool = True
) -> CESAgentConfig:
    """
    Transform a CES survey row into an agent configuration.

    This is the core transformation function. It:
    1. Extracts relevant variables from the CES row
    2. Maps codes to meaningful values
    3. Generates persona description and constraints

    Args:
        row: Dictionary with CES variable names as keys
        mapper: CESVariableMapper instance
        include_persona: Whether to generate persona description

    Returns:
        CESAgentConfig ready for Social RL canvas
    """
    # Extract and map demographics
    province_code, province_name = mapper.map_province(
        row.get("cps21_province", 35)  # Default Ontario
    )

    party_code, party_name = mapper.map_party(
        row.get("cps21_pid_party", 8)
    )

    edu_code, edu_label = mapper.map_education(
        row.get("cps21_education", 5)
    )

    # Build agent config
    config = CESAgentConfig(
        source_type="respondent",
        source_id=str(row.get("cps21_ResponseId", "unknown")),
        province=province_code,
        province_name=province_name,
        age_group=mapper.map_age_to_group(row.get("cps21_yob", 1980)),
        gender=mapper.map_gender(row.get("cps21_genderid", 0)),
        education=edu_code,
        education_label=edu_label,
        income_quintile=mapper.map_income_to_quintile(row.get("cps21_income_cat", 5)),
        urban_rural=mapper.map_urban_rural(row.get("cps21_urban_rural", 1)),
        party_id=party_code,
        party_name=party_name,
        ideology_lr=float(row.get("cps21_lr_scale", 5.0) or 5.0),
        turnout_likelihood=_estimate_turnout_likelihood(row),
        born_in_canada=row.get("cps21_bornin_canada", 1) == 1,
        language="French" if row.get("cps21_language", 1) == 2 else "English",
        riding_id=row.get("cps21_riding_id"),
        current_vote_intention=row.get("cps21_votechoice"),
    )

    # Generate persona if requested
    if include_persona:
        config.persona_description = _generate_persona_description(config, row)
        config.constraints = _generate_constraints(config, row)

    return config


def ces_cluster_to_prototype(
    cluster_stats: Dict[str, Any],
    cluster_id: str,
    mapper: CESVariableMapper
) -> CESAgentConfig:
    """
    Create a prototype agent from cluster statistics.

    Instead of individual respondents, this creates an agent representing
    a cluster of similar respondents (e.g., from k-means clustering).

    Args:
        cluster_stats: Dictionary with mean/mode values for cluster
        cluster_id: Identifier for the cluster
        mapper: CESVariableMapper instance

    Returns:
        CESAgentConfig representing the cluster prototype
    """
    # Use mode/mean values from cluster
    province_code, province_name = mapper.map_province(
        int(cluster_stats.get("province_mode", 35))
    )

    party_code, party_name = mapper.map_party(
        int(cluster_stats.get("party_mode", 8))
    )

    config = CESAgentConfig(
        source_type="cluster",
        source_id=cluster_id,
        province=province_code,
        province_name=province_name,
        age_group=cluster_stats.get("age_group_mode", "35-44"),
        gender=cluster_stats.get("gender_mode", "Not specified"),
        education=int(cluster_stats.get("education_mode", 5)),
        education_label=mapper.map_education(int(cluster_stats.get("education_mode", 5)))[1],
        income_quintile=int(cluster_stats.get("income_quintile_mean", 3)),
        urban_rural=cluster_stats.get("urban_rural_mode", "Urban"),
        party_id=party_code,
        party_name=party_name,
        ideology_lr=float(cluster_stats.get("ideology_lr_mean", 5.0)),
        turnout_likelihood=float(cluster_stats.get("turnout_mean", 0.5)),
    )

    # Generate cluster-specific persona
    config.persona_description = _generate_cluster_persona(config, cluster_stats)
    config.constraints = _generate_constraints(config, {})

    return config


# =============================================================================
# Persona and Constraint Generation
# =============================================================================

def _generate_persona_description(config: CESAgentConfig, row: Dict[str, Any]) -> str:
    """
    Generate a persona description from CES attributes.

    This creates a natural language description that the LLM can use
    to inhabit the agent's perspective authentically.
    """
    parts = []

    # Geographic identity
    parts.append(f"A {config.age_group} year old {config.gender.lower()} from {config.province_name}")

    # Socioeconomic
    if config.urban_rural:
        parts.append(f"living in a {config.urban_rural.lower()} area")

    edu_simplified = "university-educated" if config.education >= 9 else \
                    "college-educated" if config.education >= 7 else \
                    "high school educated"
    parts.append(edu_simplified)

    # Political orientation
    lr_label = CESVariableMapper().map_lr_scale(config.ideology_lr)
    if config.party_id != 8:
        parts.append(f"who leans {config.party_name} ({lr_label})")
    else:
        parts.append(f"politically {lr_label.lower()}, not attached to any party")

    # Engagement level
    if config.turnout_likelihood > 0.8:
        parts.append("and reliably votes in elections")
    elif config.turnout_likelihood < 0.4:
        parts.append("but often doesn't vote")

    return ", ".join(parts) + "."


def _generate_cluster_persona(config: CESAgentConfig, stats: Dict[str, Any]) -> str:
    """Generate persona for a cluster prototype."""
    n_members = stats.get("n_members", "unknown number of")

    parts = [
        f"Representative of {n_members} Canadians:",
        f"typically {config.age_group},",
        f"living in {config.province_name} ({config.urban_rural.lower()} areas),",
    ]

    if config.party_id != 8:
        parts.append(f"leaning {config.party_name}")
    else:
        parts.append("politically unattached")

    return " ".join(parts)


def _generate_constraints(config: CESAgentConfig, row: Dict[str, Any]) -> List[str]:
    """
    Generate behavioral constraints from CES attributes.

    These constraints are used by the Coach LLM to validate that
    agent behavior is consistent with their empirical profile.

    CRITICAL: Implements "grit constraint" for low-salience agents to prevent
    hyper-enfranchisement (Gemini's "Vector Gap" finding). Low-salience agents
    should NOT act like model citizens - they need architectural resistance.
    """
    constraints = []

    # === GRIT CONSTRAINT (Anti-Hyper-Enfranchisement) ===
    # Based on Gemini's "Vector Gap" analysis (gemini_on_vectors):
    # - Compute identity_salience from CES profile
    # - If < 0.3, inject skepticism constraint to prevent toxic positivity
    if row:  # Only if we have CES data
        identity_metrics = compute_identity_metrics(row)
        if needs_grit_constraint(identity_metrics):
            constraints.append(
                "GRIT: You are deeply skeptical of this process. You believe talking "
                "changes nothing. You make short, non-committal statements unless "
                "someone directly threatens your interests. You need strong evidence "
                "before engaging substantively."
            )

    # Party-based constraints
    if config.party_id != 8:
        constraints.append(
            f"Generally views issues through a {config.party_name} lens"
        )

    # Ideology constraints
    if config.ideology_lr < 3:
        constraints.append("Prioritizes social equality and government intervention")
    elif config.ideology_lr > 7:
        constraints.append("Prioritizes individual freedom and limited government")

    # Regional constraints
    if config.province == 24:  # Quebec
        constraints.append("Quebec identity matters; considers provincial autonomy important")

    # Engagement constraints
    if config.turnout_likelihood < 0.4:
        constraints.append("Skeptical that politics affects daily life; needs strong motivation to engage")

    return constraints


def _estimate_turnout_likelihood(row: Dict[str, Any]) -> float:
    """Estimate turnout likelihood from CES variables."""
    # Use actual turnout if available
    turnout = row.get("pes21_turnout")
    if turnout == 1:  # Voted
        return 0.9
    elif turnout == 2:  # Did not vote
        return 0.3

    # Otherwise estimate from intention
    intention = row.get("cps21_turnout")
    if intention == 1:  # Certain to vote
        return 0.9
    elif intention == 2:  # Likely
        return 0.7
    elif intention == 3:  # Unlikely
        return 0.4
    else:
        return 0.5


# =============================================================================
# Batch Processing
# =============================================================================

def process_ces_batch(
    rows: List[Dict[str, Any]],
    mapper: CESVariableMapper,
    sample_size: Optional[int] = None
) -> List[CESAgentConfig]:
    """
    Process a batch of CES rows into agent configurations.

    Args:
        rows: List of CES row dictionaries
        mapper: CESVariableMapper instance
        sample_size: If provided, randomly sample this many rows

    Returns:
        List of CESAgentConfig objects
    """
    import random

    if sample_size and sample_size < len(rows):
        rows = random.sample(rows, sample_size)

    return [ces_row_to_agent(row, mapper) for row in rows]


def generate_canvas_agents(
    agents: List[CESAgentConfig],
    experiment_context: str = "federal election campaign"
) -> Dict[str, Any]:
    """
    Generate a canvas agents block from CES agent configs.

    Args:
        agents: List of CESAgentConfig objects
        experiment_context: Context string for the experiment

    Returns:
        Dictionary in canvas format
    """
    return {
        "agents": [agent.to_canvas_agent() for agent in agents],
        "metadata": {
            "source": "CES_2021",
            "n_agents": len(agents),
            "context": experiment_context,
            "generated_at": __import__("datetime").datetime.now().isoformat()
        }
    }


if __name__ == "__main__":
    # Test the generator
    print("=== CES Agent Generator Test ===\n")

    # Simulate a CES row
    test_row = {
        "cps21_ResponseId": "R_12345",
        "cps21_province": 35,  # Ontario
        "cps21_yob": 1985,
        "cps21_genderid": 2,  # Woman
        "cps21_education": 9,  # Bachelor's
        "cps21_income_cat": 6,
        "cps21_urban_rural": 1,  # Urban
        "cps21_pid_party": 3,  # NDP
        "cps21_lr_scale": 3.0,  # Left-leaning
        "cps21_turnout": 1,  # Certain to vote
        "cps21_bornin_canada": 1,
    }

    mapper = CESVariableMapper()
    agent = ces_row_to_agent(test_row, mapper)

    print(f"Source: {agent.source_type} / {agent.source_id}")
    print(f"Location: {agent.province_name} ({agent.urban_rural})")
    print(f"Demographics: {agent.age_group}, {agent.gender}, {agent.education_label}")
    print(f"Politics: {agent.party_name} ({agent.ideology_lr})")
    print(f"Turnout likelihood: {agent.turnout_likelihood}")
    print(f"\nPersona: {agent.persona_description}")
    print(f"\nConstraints:")
    for c in agent.constraints:
        print(f"  - {c}")

    print("\n=== Canvas Format ===")
    canvas_agent = agent.to_canvas_agent()
    print(json.dumps(canvas_agent, indent=2))
