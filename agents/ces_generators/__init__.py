"""
CES Agent Generators

Transforms CES (Canadian Election Study) survey data into dynamically-configured
agents for Social RL simulations.

Key principle: Agents are NOT static personas. They are dynamically generated
from CES variables, with the ContextInjector providing turn-by-turn adaptation.
"""

from .row_to_agent import (
    CESAgentConfig,
    ces_row_to_agent,
    ces_cluster_to_prototype,
    CESVariableMapper,
)

from .identity_metrics import (
    compute_identity_salience,
    compute_tie_to_place,
    compute_identity_metrics,
    get_identity_category,
    needs_grit_constraint,
)

from .grit_config import (
    GritLevel,
    GritConstraint,
    get_grit_level,
    get_grit_constraint,
    generate_grit_prompt,
    get_grit_temperature_modifier,
    get_ces_engagement_target,
    get_ces_identity_prior,
    calibrate_grit_to_ces,
    needs_grit_constraint_v2,
)

__all__ = [
    "CESAgentConfig",
    "ces_row_to_agent",
    "ces_cluster_to_prototype",
    "CESVariableMapper",
    # Identity metrics (Weber's "tie to place")
    "compute_identity_salience",
    "compute_tie_to_place",
    "compute_identity_metrics",
    "get_identity_category",
    "needs_grit_constraint",
    # Grit v2 (calibrated constraints)
    "GritLevel",
    "GritConstraint",
    "get_grit_level",
    "get_grit_constraint",
    "generate_grit_prompt",
    "get_grit_temperature_modifier",
    "get_ces_engagement_target",
    "get_ces_identity_prior",
    "calibrate_grit_to_ces",
    "needs_grit_constraint_v2",
]
