"""
Agent System - Canvas-to-agent transformation and execution.

This package provides the infrastructure for transforming PRAR canvas
definitions into executable simulation participants.
"""

from .agent_config import AgentConfig, AgentResponse, RoundConfig
from .agent_factory import AgentFactory

__all__ = [
    "AgentConfig",
    "AgentResponse",
    "RoundConfig",
    "AgentFactory",
]
