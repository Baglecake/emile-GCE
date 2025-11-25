"""
AgentRunner - Executes agents in simulation rounds.

This module provides the runtime for multi-agent simulations, coordinating
agent turns, managing conversation state, and collecting transcripts.
"""
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import json

# Add local_rcm to path for LLM client access
sys.path.insert(0, str(Path(__file__).parent.parent / "local_rcm"))

from agent_config import AgentConfig, AgentResponse, RoundConfig
from agent_factory import AgentFactory

# Import LLM client from local_rcm
try:
    from llm_client import LLMClient, MockClient, create_llm_client
except ImportError:
    # Fallback if running standalone
    class LLMClient:
        def send_message(self, system: str, user: str) -> str:
            return ""

    class MockClient(LLMClient):
        def __init__(self):
            self.call_count = 0

        def send_message(self, system: str, user: str) -> str:
            self.call_count += 1
            return f"[Mock response #{self.call_count}]"

    def create_llm_client(provider: str = "mock", **kwargs) -> LLMClient:
        return MockClient()


@dataclass
class Message:
    """A single message in the conversation."""
    agent_id: str
    content: str
    round_number: int
    turn_number: int
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "content": self.content,
            "round_number": self.round_number,
            "turn_number": self.turn_number,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class RoundTranscript:
    """Complete transcript of a simulation round."""
    round_number: int
    round_config: Dict[str, Any]
    messages: List[Message] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: Message):
        self.messages.append(message)

    def finalize(self):
        self.end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "round_config": self.round_config,
            "messages": [m.to_dict() for m in self.messages],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": (self.end_time - self.start_time) if self.end_time else None,
            "metadata": self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class AgentRunner:
    """
    Executes agents in simulation rounds.

    The runner coordinates turn-taking, manages conversation context,
    and collects transcripts for analysis.

    Usage:
        factory = AgentFactory.from_state_file("state.json")
        runner = AgentRunner(factory, llm_client)

        transcript = runner.execute_round(1)
        runner.save_transcript(transcript, "round1_transcript.json")
    """

    def __init__(
        self,
        factory: AgentFactory,
        llm_client: LLMClient,
        verbose: bool = True
    ):
        """
        Initialize the agent runner.

        Args:
            factory: AgentFactory with canvas configuration
            llm_client: LLM client for agent execution
            verbose: Whether to print execution details
        """
        self.factory = factory
        self.llm = llm_client
        self.verbose = verbose
        self.transcripts: Dict[int, RoundTranscript] = {}

    def execute_agent_turn(
        self,
        agent: AgentConfig,
        round_config: RoundConfig,
        conversation_history: List[Message],
        turn_number: int
    ) -> AgentResponse:
        """
        Execute a single agent turn.

        Args:
            agent: The agent configuration
            round_config: Current round configuration
            conversation_history: Messages so far in this round
            turn_number: Current turn number

        Returns:
            AgentResponse with the agent's output
        """
        # Build the system prompt with round context
        round_context = {
            "scenario": round_config.scenario,
            "rules": round_config.rules,
            "tasks": round_config.tasks
        }
        system_prompt = agent.compile_system_prompt(round_context)

        # Build conversation context for the user message
        if conversation_history:
            context_lines = ["CONVERSATION SO FAR:"]
            for msg in conversation_history[-10:]:  # Last 10 messages
                context_lines.append(f"[{msg.agent_id}]: {msg.content}")
            context = "\n".join(context_lines)
            user_message = f"{context}\n\nIt is now your turn to respond as {agent.name}."
        else:
            user_message = f"The round begins. You are {agent.name}. {round_config.scenario}\n\nRespond to start the conversation."

        # Execute via LLM
        start_time = time.time()
        try:
            content = self.llm.send_message(system_prompt, user_message)
        except Exception as e:
            content = f"[Error: {str(e)}]"

        latency_ms = (time.time() - start_time) * 1000

        return AgentResponse(
            agent_id=agent.identifier,
            content=content,
            round_number=round_config.round_number,
            turn_number=turn_number,
            latency_ms=latency_ms,
            metadata={"model": agent.model or "default"}
        )

    def execute_round(
        self,
        round_number: int,
        max_turns: Optional[int] = None,
        turn_callback: Optional[Callable[[Message], None]] = None
    ) -> RoundTranscript:
        """
        Execute a complete simulation round.

        Args:
            round_number: Which round to execute (1-indexed)
            max_turns: Override maximum turns (uses end_condition if None)
            turn_callback: Optional callback after each turn

        Returns:
            RoundTranscript with all messages
        """
        round_config = self.factory.create_round(round_number)
        participants = self.factory.get_round_participants(round_number)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"ROUND {round_number}: {round_config.scenario[:60]}...")
            print(f"Participants: {[p.identifier for p in participants]}")
            print(f"{'='*60}\n")

        transcript = RoundTranscript(
            round_number=round_number,
            round_config=round_config.to_dict()
        )

        # Parse end condition for max messages
        end_condition = round_config.end_condition.lower()
        if max_turns is None:
            if "total messages:" in end_condition:
                try:
                    max_turns = int(end_condition.split(":")[-1].strip())
                except ValueError:
                    max_turns = 15
            elif "per participant:" in end_condition:
                try:
                    per_p = int(end_condition.split(":")[-1].strip())
                    max_turns = per_p * len(participants)
                except ValueError:
                    max_turns = 15
            else:
                max_turns = 15

        conversation_history: List[Message] = []
        turn = 0

        # Simple round-robin turn order
        while turn < max_turns:
            for agent in participants:
                if turn >= max_turns:
                    break

                turn += 1
                response = self.execute_agent_turn(
                    agent, round_config, conversation_history, turn
                )

                message = Message(
                    agent_id=response.agent_id,
                    content=response.content,
                    round_number=round_number,
                    turn_number=turn,
                    metadata={"latency_ms": response.latency_ms}
                )

                conversation_history.append(message)
                transcript.add_message(message)

                if self.verbose:
                    print(f"[Turn {turn}] {agent.identifier}:")
                    print(f"  {response.content[:200]}{'...' if len(response.content) > 200 else ''}")
                    print()

                if turn_callback:
                    turn_callback(message)

        transcript.finalize()
        self.transcripts[round_number] = transcript

        if self.verbose:
            print(f"\nRound {round_number} complete: {len(transcript.messages)} messages")
            duration = transcript.end_time - transcript.start_time
            print(f"Duration: {duration:.1f}s")

        return transcript

    def execute_all_rounds(
        self,
        max_turns_per_round: Optional[int] = None
    ) -> List[RoundTranscript]:
        """
        Execute all rounds in the canvas.

        Args:
            max_turns_per_round: Override max turns for all rounds

        Returns:
            List of RoundTranscripts
        """
        rounds = self.factory.create_all_rounds()
        transcripts = []

        for round_config in rounds:
            transcript = self.execute_round(
                round_config.round_number,
                max_turns=max_turns_per_round
            )
            transcripts.append(transcript)

        return transcripts

    def save_transcript(
        self,
        transcript: RoundTranscript,
        output_path: str
    ):
        """Save a transcript to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(transcript.to_json())

        if self.verbose:
            print(f"Transcript saved: {path}")

    def save_all_transcripts(self, output_dir: str):
        """Save all collected transcripts to a directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for round_num, transcript in self.transcripts.items():
            self.save_transcript(
                transcript,
                str(output_path / f"round{round_num}_transcript.json")
            )


def run_simulation(
    state_path: str,
    output_dir: str,
    provider: str = "mock",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_turns: Optional[int] = None,
    verbose: bool = True
) -> List[RoundTranscript]:
    """
    Run a complete simulation from a state file.

    Args:
        state_path: Path to state.json
        output_dir: Directory for output files
        provider: LLM provider (mock, vllm, openai, etc.)
        model: Model identifier
        base_url: API base URL
        api_key: API key
        max_turns: Max turns per round
        verbose: Print execution details

    Returns:
        List of RoundTranscripts
    """
    # Create factory
    factory = AgentFactory.from_state_file(state_path, default_model=model)

    if verbose:
        print(factory.summary())
        print()

    # Create LLM client
    llm = create_llm_client(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key
    )

    # Create runner and execute
    runner = AgentRunner(factory, llm, verbose=verbose)
    transcripts = runner.execute_all_rounds(max_turns_per_round=max_turns)

    # Save outputs
    runner.save_all_transcripts(output_dir)

    # Save combined summary
    summary = {
        "canvas_summary": factory.summary(),
        "rounds_executed": len(transcripts),
        "total_messages": sum(len(t.messages) for t in transcripts),
        "provider": provider,
        "model": model
    }

    summary_path = Path(output_dir) / "simulation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\nSimulation complete. Outputs saved to: {output_dir}")

    return transcripts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run agent simulation")
    parser.add_argument("--state", required=True, help="Path to state.json")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--provider", default="mock", help="LLM provider")
    parser.add_argument("--model", help="Model identifier")
    parser.add_argument("--base-url", help="API base URL")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--max-turns", type=int, help="Max turns per round")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    run_simulation(
        state_path=args.state,
        output_dir=args.output,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_turns=args.max_turns,
        verbose=not args.quiet
    )
