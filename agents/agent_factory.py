"""
AgentFactory - Factory for instantiating agents from canvas definitions.

This module provides the bridge between PRAR canvas outputs and executable
simulation agents, transforming declarative definitions into runtime configurations.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from .agent_config import AgentConfig, RoundConfig


class AgentFactory:
    """
    Factory for creating AgentConfig instances from canvas definitions.

    The factory reads canvas data (either from a CanvasState object or
    loaded from state.json) and produces executable agent configurations.

    Usage:
        factory = AgentFactory.from_state_file("path/to/state.json")
        alice = factory.create("Worker+Alice")
        all_agents = factory.create_all()
    """

    def __init__(self, canvas: Dict[str, Any], default_model: Optional[str] = None):
        """
        Initialize the factory with canvas data.

        Args:
            canvas: Canvas dictionary (the "canvas" key from state.json)
            default_model: Default LLM model identifier for agents
        """
        self.canvas = canvas
        self.default_model = default_model
        self._agents_cache: Dict[str, AgentConfig] = {}
        self._rounds_cache: Dict[int, RoundConfig] = {}

    @classmethod
    def from_state_file(cls, state_path: str,
                        default_model: Optional[str] = None) -> "AgentFactory":
        """
        Create an AgentFactory from a state.json file.

        Args:
            state_path: Path to the state.json file
            default_model: Default model for agents

        Returns:
            Initialized AgentFactory

        Raises:
            FileNotFoundError: If state file doesn't exist
            KeyError: If state file lacks "canvas" key
        """
        path = Path(state_path)
        if not path.exists():
            raise FileNotFoundError(f"State file not found: {state_path}")

        with open(path, "r") as f:
            state = json.load(f)

        if "canvas" not in state:
            raise KeyError("State file missing 'canvas' key")

        return cls(state["canvas"], default_model)

    @classmethod
    def from_canvas_state(cls, canvas_state: Any,
                          default_model: Optional[str] = None) -> "AgentFactory":
        """
        Create an AgentFactory from a CanvasState object.

        Args:
            canvas_state: CanvasState instance from canvas_state.py
            default_model: Default model for agents

        Returns:
            Initialized AgentFactory
        """
        # CanvasState has a to_dict() method
        if hasattr(canvas_state, "to_dict"):
            canvas_dict = canvas_state.to_dict()
        else:
            canvas_dict = canvas_state

        return cls(canvas_dict, default_model)

    def create(self, identifier: str) -> AgentConfig:
        """
        Create an AgentConfig for a specific agent by identifier.

        Args:
            identifier: Agent identifier (e.g., "Worker+Alice")

        Returns:
            AgentConfig for the specified agent

        Raises:
            ValueError: If agent not found in canvas
        """
        if identifier in self._agents_cache:
            return self._agents_cache[identifier]

        agents = self.canvas.get("agents", [])
        for agent_data in agents:
            if agent_data.get("identifier") == identifier:
                config = AgentConfig.from_canvas_agent(
                    agent_data, self.default_model
                )
                self._agents_cache[identifier] = config
                return config

        raise ValueError(f"Agent not found: {identifier}")

    def create_all(self) -> List[AgentConfig]:
        """
        Create AgentConfig instances for all agents in the canvas.

        Returns:
            List of AgentConfig instances
        """
        agents = self.canvas.get("agents", [])
        return [
            AgentConfig.from_canvas_agent(agent, self.default_model)
            for agent in agents
        ]

    def create_round(self, round_number: int) -> RoundConfig:
        """
        Create a RoundConfig for a specific round.

        Args:
            round_number: Round number (1-indexed)

        Returns:
            RoundConfig for the specified round

        Raises:
            ValueError: If round not found in canvas
        """
        if round_number in self._rounds_cache:
            return self._rounds_cache[round_number]

        rounds = self.canvas.get("rounds", [])
        for round_data in rounds:
            if round_data.get("round_number") == round_number:
                config = RoundConfig.from_canvas_round(round_data)
                self._rounds_cache[round_number] = config
                return config

        raise ValueError(f"Round not found: {round_number}")

    def create_all_rounds(self) -> List[RoundConfig]:
        """
        Create RoundConfig instances for all rounds in the canvas.

        Returns:
            List of RoundConfig instances
        """
        rounds = self.canvas.get("rounds", [])
        return [RoundConfig.from_canvas_round(r) for r in rounds]

    def get_round_participants(self, round_number: int) -> List[AgentConfig]:
        """
        Get all agents participating in a specific round.

        Args:
            round_number: Round number (1-indexed)

        Returns:
            List of AgentConfig instances for round participants
        """
        round_config = self.create_round(round_number)
        return [self.create(pid) for pid in round_config.participants]

    def get_agent_identifiers(self) -> List[str]:
        """
        Get all agent identifiers from the canvas.

        Returns:
            List of agent identifier strings
        """
        agents = self.canvas.get("agents", [])
        return [a.get("identifier", "") for a in agents]

    def get_project_info(self) -> Dict[str, Any]:
        """
        Get project information from the canvas.

        Returns:
            Project dictionary with goal, theoretical option, concepts, etc.
        """
        return self.canvas.get("project", {})

    def get_helpers(self) -> Dict[str, str]:
        """
        Get helper function configurations from the canvas.

        Returns:
            Dictionary of helper function definitions
        """
        return self.canvas.get("helpers", {})

    def summary(self) -> str:
        """
        Generate a summary of the canvas configuration.

        Returns:
            Human-readable summary string
        """
        project = self.get_project_info()
        agents = self.get_agent_identifiers()
        rounds = self.canvas.get("rounds", [])

        lines = [
            "=== Canvas Summary ===",
            f"Goal: {project.get('goal', 'N/A')[:80]}...",
            f"Framework: {project.get('theoretical_option_label', 'N/A')}",
            f"Agents ({len(agents)}): {', '.join(agents)}",
            f"Rounds: {len(rounds)}",
            ""
        ]

        for r in rounds:
            participants = r.get("platform_config", {}).get("participants", "")
            lines.append(f"  Round {r.get('round_number')}: {participants}")

        return "\n".join(lines)


if __name__ == "__main__":
    import sys

    # Test with baseline state file
    test_paths = [
        "../prar/outputs/2025-11-23_baseline_full_qwen/state.json",
        "../prar/outputs/2025-11-23_baseline_phase1_qwen/state.json",
    ]

    for test_path in test_paths:
        path = Path(__file__).parent / test_path
        if path.exists():
            print(f"Testing with: {path}")
            factory = AgentFactory.from_state_file(str(path))

            print(factory.summary())
            print()

            # Test creating individual agent
            agents = factory.get_agent_identifiers()
            if agents:
                agent = factory.create(agents[0])
                print(f"Created agent: {agent}")
                print(f"  Role: {agent.role}")
                print(f"  Goal: {agent.goal[:60]}...")
                print()

            # Test creating all agents
            all_agents = factory.create_all()
            print(f"Created {len(all_agents)} agents")

            # Test round creation
            rounds = factory.create_all_rounds()
            print(f"Created {len(rounds)} rounds")

            for r in rounds:
                participants = factory.get_round_participants(r.round_number)
                print(f"  Round {r.round_number}: {len(participants)} participants")

            break
    else:
        print("No test state files found. Run from agents/ directory.")
        sys.exit(1)
