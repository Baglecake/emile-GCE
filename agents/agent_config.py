"""
AgentConfig - Configuration dataclass for simulation agents.

This module defines the structure for agents that can be instantiated
from PRAR canvas definitions and executed in simulations.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class AgentConfig:
    """
    Configuration for a simulation agent.

    Agents are defined in the PRAR canvas and instantiated here for execution.
    Each agent has a role, goal, persona, and behavioral parameters.

    Attributes:
        identifier: Unique agent identifier (e.g., "Worker+Alice")
        role: The agent's role in the simulation (e.g., "Worker", "Owner")
        name: The agent's name (e.g., "Alice", "Marta")
        goal: What the agent is trying to achieve
        persona: Behavioral description and personality traits
        prompt: Compiled system prompt for LLM execution
        model: LLM model identifier (default: inherited from config)
        temperature: Generation temperature (0.0-1.0)
        max_tokens: Maximum response length
        behaviors: Conditional behavior rules (if-then patterns)
        metadata: Additional configuration data
    """
    identifier: str
    role: str
    name: str
    goal: str
    persona: str
    prompt: str
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    behaviors: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_canvas_agent(cls, canvas_agent: Dict[str, Any],
                          default_model: Optional[str] = None) -> "AgentConfig":
        """
        Create an AgentConfig from a canvas agent definition.

        Args:
            canvas_agent: Agent dict from canvas (identifier, goal, persona, prompt)
            default_model: Default model to use if not specified

        Returns:
            AgentConfig instance
        """
        identifier = canvas_agent.get("identifier", "Unknown")

        # Parse role and name from identifier (e.g., "Worker+Alice" -> "Worker", "Alice")
        if "+" in identifier:
            role, name = identifier.split("+", 1)
        else:
            role = identifier
            name = identifier

        # Parse behaviors from the behaviors field if present
        behaviors = {}
        behavior_str = canvas_agent.get("behaviors", "")
        if behavior_str and behavior_str.lower() not in ("no", "none", ""):
            # Parse "yes - If X: do Y. If Z: do W." format
            behaviors["raw"] = behavior_str

        return cls(
            identifier=identifier,
            role=role,
            name=name,
            goal=canvas_agent.get("goal", ""),
            persona=canvas_agent.get("persona", ""),
            prompt=canvas_agent.get("prompt", ""),
            model=default_model,
            behaviors=behaviors,
            metadata={
                "source": "canvas",
                "original": canvas_agent
            }
        )

    def compile_system_prompt(self, round_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Compile the full system prompt for this agent.

        Args:
            round_context: Optional round-specific context to include

        Returns:
            Complete system prompt string
        """
        parts = [self.prompt]

        if round_context:
            if round_context.get("scenario"):
                parts.append(f"\nCURRENT SCENARIO: {round_context['scenario']}")
            if round_context.get("rules"):
                parts.append(f"\nRULES: {round_context['rules']}")
            if round_context.get("tasks"):
                parts.append(f"\nTASKS: {round_context['tasks']}")

        if self.behaviors:
            behavior_text = self.behaviors.get("raw", "")
            if behavior_text:
                parts.append(f"\nBEHAVIORAL RULES: {behavior_text}")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "identifier": self.identifier,
            "role": self.role,
            "name": self.name,
            "goal": self.goal,
            "persona": self.persona,
            "prompt": self.prompt,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "behaviors": self.behaviors,
            "metadata": self.metadata
        }

    def __repr__(self) -> str:
        return f"AgentConfig({self.identifier}, role={self.role}, temp={self.temperature})"


@dataclass
class AgentResponse:
    """
    Response from an agent execution.

    Captures the output and metadata from a single agent turn.
    """
    agent_id: str
    content: str
    round_number: int
    turn_number: int
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "content": self.content,
            "round_number": self.round_number,
            "turn_number": self.turn_number,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata
        }


@dataclass
class RoundConfig:
    """
    Configuration for a simulation round.

    Rounds define the context and rules for agent interactions.
    """
    round_number: int
    scenario: str
    concept_a_manifestation: str
    concept_b_manifestation: str
    rules: str
    tasks: str
    sequence: str
    participants: List[str]
    end_condition: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_canvas_round(cls, canvas_round: Dict[str, Any]) -> "RoundConfig":
        """
        Create a RoundConfig from a canvas round definition.

        Args:
            canvas_round: Round dict from canvas

        Returns:
            RoundConfig instance
        """
        platform_config = canvas_round.get("platform_config", {})

        # Parse participants from platform config
        participants_str = platform_config.get("participants", "")
        if isinstance(participants_str, str):
            participants = [p.strip() for p in participants_str.split(",") if p.strip()]
        else:
            participants = list(participants_str) if participants_str else []

        return cls(
            round_number=canvas_round.get("round_number", 0),
            scenario=canvas_round.get("scenario", ""),
            concept_a_manifestation=canvas_round.get("concept_a_manifestation", ""),
            concept_b_manifestation=canvas_round.get("concept_b_manifestation", ""),
            rules=canvas_round.get("rules", ""),
            tasks=canvas_round.get("tasks", ""),
            sequence=canvas_round.get("sequence", ""),
            participants=participants,
            end_condition=platform_config.get("end_condition", ""),
            metadata={
                "platform_config": platform_config,
                "original": canvas_round
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "round_number": self.round_number,
            "scenario": self.scenario,
            "concept_a_manifestation": self.concept_a_manifestation,
            "concept_b_manifestation": self.concept_b_manifestation,
            "rules": self.rules,
            "tasks": self.tasks,
            "sequence": self.sequence,
            "participants": self.participants,
            "end_condition": self.end_condition,
            "metadata": self.metadata
        }
