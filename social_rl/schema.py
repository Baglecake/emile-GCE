"""
Social RL Schema Definitions

TypedDict definitions for Social RL data structures, ensuring consistent
serialization and enabling static type checking.

These schemas define the contract between:
- SocialRLRunner (producer)
- Analysis tools (consumer)
- Test suite (validator)
"""

from typing import TypedDict, List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json


# =============================================================================
# Core Message Schema
# =============================================================================

class ContextFrame(TypedDict, total=False):
    """Dynamic context injected per turn."""
    base_scenario: str
    concept_a_manifestation: str
    concept_b_manifestation: str
    experiential_cue: str
    social_feedback_summary: str
    prar_cue: str


class SocialRLMessageDict(TypedDict):
    """Schema for individual agent messages."""
    agent_id: str
    content: str
    round_number: int
    turn_number: int
    timestamp: float
    prar_cue_used: Optional[str]
    feedback_snapshot: Optional[Dict[str, Any]]
    validation_metadata: Optional[Dict[str, Any]]


class SocialRLMessageWithContext(SocialRLMessageDict, total=False):
    """Extended message with optional context frame."""
    context_frame: ContextFrame
    role: str  # "assistant", "system", "user"
    internal: bool  # True for coach-only messages


# =============================================================================
# Feedback Schema
# =============================================================================

class FeedbackVector(TypedDict):
    """Social feedback metrics for a single agent."""
    agent_id: str
    round_number: int
    engagement: float  # 0.0-1.0: participation level
    theoretical_alignment: float  # 0.0-1.0: adherence to framework
    contribution_value: float  # 0.0-1.0: substantive participation
    direct_references: int  # Count of references to this agent
    response_received: int  # Count of responses received
    concepts_embodied: List[str]  # Detected concept expressions
    analyst_mentions: int  # Analyst function references
    synthesis_inclusion: float  # Inclusion in round synthesis


class FeedbackVectorMinimal(TypedDict):
    """Minimal feedback vector for compact serialization."""
    engagement: float
    alignment: float
    contribution_value: float


# =============================================================================
# Policy Adaptation Schema
# =============================================================================

class PolicyAdaptation(TypedDict):
    """Record of a policy change during execution."""
    agent_id: str
    adaptation_type: str  # "activate_cue", "deactivate_cue", "adjust_intensity"
    cue: Optional[str]  # The specific cue affected
    reason: str  # Human-readable explanation
    round_number: int
    feedback_trigger: Optional[Dict[str, float]]  # Feedback values that triggered this


# =============================================================================
# Round Result Schema
# =============================================================================

class ExperimentMeta(TypedDict):
    """Metadata block for experiment traceability."""
    experiment_id: str  # e.g., "social_rl_2025-11-23_175825"
    prar_run_id: str  # e.g., "2025-11-23_baseline_full_qwen"
    framework: str  # e.g., "Alienation vs Non-Domination"
    framework_option: str  # e.g., "A"
    model: str  # e.g., "Qwen/Qwen2.5-7B-Instruct"
    performer_temperature: float
    coach_temperature: float
    social_rl_version: str
    timestamp: str  # ISO format


class SocialRLRoundResult(TypedDict):
    """Complete result for a single round."""
    round_number: int
    messages: List[SocialRLMessageDict]
    feedback: Dict[str, FeedbackVector]
    policy_adaptations: List[PolicyAdaptation]
    synthesis: str
    duration_seconds: float


class SocialRLRoundResultWithMeta(SocialRLRoundResult, total=False):
    """Round result with optional metadata block."""
    meta: ExperimentMeta
    round_config: Dict[str, Any]


# =============================================================================
# Experiment-Level Schema
# =============================================================================

class ExperimentConfig(TypedDict):
    """Configuration for a Social RL experiment."""
    manifestation_mode: str  # "static", "progressive", "reactive", "adaptive"
    extract_feedback_per_turn: bool
    adapt_policies_per_round: bool
    use_prar_cues: bool
    use_coach_validation: bool
    coach_temperature: float
    performer_temperature: float
    max_turns_per_round: int
    prar_state_path: str


class ExperimentSummary(TypedDict):
    """Summary of a complete experiment."""
    experiment_id: str
    config: ExperimentConfig
    rounds_completed: int
    total_messages: int
    total_duration_seconds: float
    agents: List[str]
    framework: str


# =============================================================================
# Dataclass Implementations (for runtime use)
# =============================================================================

@dataclass
class SocialRLMessage:
    """Runtime dataclass for Social RL messages."""
    agent_id: str
    content: str
    round_number: int
    turn_number: int
    timestamp: float
    prar_cue_used: Optional[str] = None
    feedback_snapshot: Optional[Dict[str, Any]] = None
    validation_metadata: Optional[Dict[str, Any]] = None
    context_frame: Optional[Dict[str, Any]] = None
    role: str = "assistant"
    internal: bool = False

    def to_dict(self) -> SocialRLMessageDict:
        """Convert to TypedDict for serialization."""
        return {
            "agent_id": self.agent_id,
            "content": self.content,
            "round_number": self.round_number,
            "turn_number": self.turn_number,
            "timestamp": self.timestamp,
            "prar_cue_used": self.prar_cue_used,
            "feedback_snapshot": self.feedback_snapshot,
            "validation_metadata": self.validation_metadata,
        }

    def to_dict_full(self) -> Dict[str, Any]:
        """Convert to full dict including optional fields."""
        d = self.to_dict()
        if self.context_frame:
            d["context_frame"] = self.context_frame
        if self.role != "assistant":
            d["role"] = self.role
        if self.internal:
            d["internal"] = self.internal
        return d


@dataclass
class AgentFeedback:
    """Runtime dataclass for agent feedback."""
    agent_id: str
    round_number: int
    engagement: float = 0.0
    theoretical_alignment: float = 0.0
    contribution_value: float = 0.0
    direct_references: int = 0
    response_received: int = 0
    concepts_embodied: List[str] = field(default_factory=list)
    analyst_mentions: int = 0
    synthesis_inclusion: float = 0.0

    def to_dict(self) -> FeedbackVector:
        """Convert to TypedDict for serialization."""
        return {
            "agent_id": self.agent_id,
            "round_number": self.round_number,
            "engagement": self.engagement,
            "theoretical_alignment": self.theoretical_alignment,
            "contribution_value": self.contribution_value,
            "direct_references": self.direct_references,
            "response_received": self.response_received,
            "concepts_embodied": self.concepts_embodied,
            "analyst_mentions": self.analyst_mentions,
            "synthesis_inclusion": self.synthesis_inclusion,
        }


@dataclass
class RoundResult:
    """Runtime dataclass for round results."""
    round_number: int
    messages: List[SocialRLMessage] = field(default_factory=list)
    feedback: Dict[str, AgentFeedback] = field(default_factory=dict)
    policy_adaptations: List[Dict[str, Any]] = field(default_factory=list)
    synthesis: str = ""
    duration_seconds: float = 0.0
    meta: Optional[Dict[str, Any]] = None
    round_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> SocialRLRoundResultWithMeta:
        """Convert to TypedDict for serialization."""
        result: Dict[str, Any] = {
            "round_number": self.round_number,
            "messages": [m.to_dict() for m in self.messages],
            "feedback": {k: v.to_dict() for k, v in self.feedback.items()},
            "policy_adaptations": self.policy_adaptations,
            "synthesis": self.synthesis,
            "duration_seconds": self.duration_seconds,
        }
        if self.meta:
            result["meta"] = self.meta
        if self.round_config:
            result["round_config"] = self.round_config
        return result


# =============================================================================
# Utility Functions
# =============================================================================

def create_experiment_meta(
    experiment_id: str,
    prar_run_id: str,
    framework: str,
    framework_option: str,
    model: str,
    performer_temperature: float = 0.7,
    coach_temperature: float = 0.1,
    version: str = "0.2.0"
) -> ExperimentMeta:
    """Create a metadata block for experiment output."""
    return {
        "experiment_id": experiment_id,
        "prar_run_id": prar_run_id,
        "framework": framework,
        "framework_option": framework_option,
        "model": model,
        "performer_temperature": performer_temperature,
        "coach_temperature": coach_temperature,
        "social_rl_version": version,
        "timestamp": datetime.now().isoformat(),
    }


def validate_round_result(data: Dict[str, Any]) -> bool:
    """
    Validate that a dict conforms to SocialRLRoundResult schema.

    Returns True if valid, raises ValueError with details if not.
    """
    required_keys = ["round_number", "messages", "feedback", "policy_adaptations",
                     "synthesis", "duration_seconds"]

    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")

    if not isinstance(data["round_number"], int):
        raise ValueError("round_number must be an integer")

    if not isinstance(data["messages"], list):
        raise ValueError("messages must be a list")

    if not isinstance(data["feedback"], dict):
        raise ValueError("feedback must be a dict")

    if not isinstance(data["duration_seconds"], (int, float)):
        raise ValueError("duration_seconds must be a number")

    # Validate message structure
    for i, msg in enumerate(data["messages"]):
        msg_required = ["agent_id", "content", "round_number", "turn_number", "timestamp"]
        for key in msg_required:
            if key not in msg:
                raise ValueError(f"Message {i} missing required key: {key}")

    # Validate feedback structure
    for agent_id, fb in data["feedback"].items():
        if "engagement" not in fb:
            raise ValueError(f"Feedback for {agent_id} missing engagement")

    return True


def load_round_result(filepath: str) -> SocialRLRoundResultWithMeta:
    """Load and validate a round result from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    validate_round_result(data)
    return data


def save_round_result(result: RoundResult, filepath: str) -> None:
    """Save a round result to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)


# =============================================================================
# Schema Version
# =============================================================================

SCHEMA_VERSION = "0.2.0"
