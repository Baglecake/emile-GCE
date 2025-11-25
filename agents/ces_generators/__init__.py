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
]
