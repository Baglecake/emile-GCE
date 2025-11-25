"""
SocialRLRunner - Main execution engine for Social RL simulations.

This integrates all components:
- ContextInjector: Dynamic manifestation generation
- SocialFeedbackExtractor: Extract learning signals
- ProcessRetriever: PRAR-based reasoning guidance
- Coach/Performer pipeline: Validated generation

The result: agents that learn through social interaction,
guided by process retrieval, without explicit reward functions.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

# Use relative imports for social_rl modules
from .context_injector import (
    ContextInjector, TheoreticalFramework, TurnContext,
    ManifestationType, create_context_injector_from_canvas
)
from .feedback_extractor import (
    SocialFeedbackExtractor, SocialFeedback, ConceptMarkers,
    create_extractor_for_framework
)
from .process_retriever import ProcessRetriever, ReasoningPolicy
from .dual_llm_client import DualLLMClient, DualLLMConfig, GenerationResult


def _get_default_output_dir(experiment_id: str = None) -> str:
    """Get default output directory based on environment."""
    import datetime
    import os
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Use experiment_id if provided, otherwise use timestamp
    if experiment_id:
        dir_name = experiment_id
    else:
        dir_name = f"social_rl_{timestamp}"

    # Always use current working directory - works for local, Colab, VS Code + Colab
    cwd = os.getcwd()
    return os.path.join(cwd, "outputs", dir_name)


@dataclass
class SocialRLConfig:
    """Configuration for Social RL execution."""
    # Manifestation mode
    manifestation_mode: str = "progressive"  # static, progressive, reactive, adaptive

    # Feedback extraction
    extract_feedback_per_turn: bool = True
    adapt_policies_per_round: bool = True

    # Process retrieval
    use_prar_cues: bool = True
    prar_intensity: str = "medium"  # low, medium, high

    # Coach/Performer settings
    use_coach_validation: bool = True
    coach_temperature: float = 0.1
    performer_temperature: float = 0.7
    max_validation_retries: int = 2

    # Output settings
    verbose: bool = True
    save_feedback_history: bool = True
    auto_save: bool = True  # Auto-save after each round
    output_dir: str = ""    # Empty = auto-detect

    # Challenge mode for A/B testing (empirical semiotics)
    challenge_mode: str = "adaptive"  # off, adaptive, always


@dataclass
class SocialRLMessage:
    """Enhanced message with Social RL metadata."""
    agent_id: str
    content: str
    round_number: int
    turn_number: int
    timestamp: float = field(default_factory=time.time)

    # Social RL additions
    turn_context: Optional[Dict[str, Any]] = None
    prar_cue_used: str = ""
    feedback_snapshot: Optional[Dict[str, float]] = None
    validation_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "content": self.content,
            "round_number": self.round_number,
            "turn_number": self.turn_number,
            "timestamp": self.timestamp,
            "prar_cue_used": self.prar_cue_used,
            "feedback_snapshot": self.feedback_snapshot,
            "validation_metadata": self.validation_metadata
        }


@dataclass
class SocialRLRoundResult:
    """Result of a Social RL round."""
    round_number: int
    messages: List[SocialRLMessage]
    feedback: Dict[str, SocialFeedback]
    policy_adaptations: List[Dict[str, Any]]
    synthesis: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "messages": [m.to_dict() for m in self.messages],
            "feedback": {k: v.to_dict() for k, v in self.feedback.items()},
            "policy_adaptations": self.policy_adaptations,
            "synthesis": self.synthesis,
            "duration_seconds": self.duration_seconds
        }


class SocialRLRunner:
    """
    Main execution engine for Social RL simulations.

    Combines:
    - Dynamic context injection (manifestations adapt per turn)
    - Social feedback extraction (interactions become learning signals)
    - Process retrieval (PRAR guides reasoning, not content)
    - Coach/Performer validation (authentic + governed responses)

    The key innovation: RL-like learning emerges from social interaction
    without explicit reward functions or weight updates.
    """

    def __init__(
        self,
        canvas: Dict[str, Any],
        llm_client: Any,
        config: SocialRLConfig = None,
        dual_llm_client: Optional[DualLLMClient] = None,
        experiment_id: Optional[str] = None
    ):
        """
        Initialize the Social RL runner.

        Args:
            canvas: Canvas configuration (from state.json["canvas"])
            llm_client: LLM client for generation (used if dual_llm_client not provided)
            config: Social RL configuration
            dual_llm_client: Optional DualLLMClient for Coach/Performer architecture
            experiment_id: Optional experiment ID for output directory naming
        """
        self.canvas = canvas
        self.llm = llm_client
        self.config = config or SocialRLConfig()
        self.dual_llm = dual_llm_client
        self.experiment_id = experiment_id

        # Extract framework info
        project = canvas.get("project", {})
        self.framework_option = project.get("theoretical_option", "A")

        # Initialize components
        self.context_injector = create_context_injector_from_canvas(
            canvas, self.config.manifestation_mode
        )
        self.feedback_extractor = create_extractor_for_framework(self.framework_option)
        self.process_retriever = ProcessRetriever(
            self.framework_option,
            challenge_mode=self.config.challenge_mode
        )

        # State
        self.round_results: Dict[int, SocialRLRoundResult] = {}
        self.accumulated_feedback: Dict[str, Dict[str, float]] = {}

        # Setup output directory
        if self.config.output_dir:
            self.output_dir = Path(self.config.output_dir)
        else:
            self.output_dir = Path(_get_default_output_dir(experiment_id))

        # Create output directory
        if self.config.auto_save:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.verbose:
            print(f"SocialRLRunner initialized")
            print(f"  Framework: {project.get('theoretical_option_label', self.framework_option)}")
            print(f"  Manifestation mode: {self.config.manifestation_mode}")
            print(f"  PRAR cues: {self.config.use_prar_cues}")
            print(f"  Dual-LLM Client: {'enabled' if self.dual_llm else 'disabled'}")
            if self.config.auto_save:
                print(f"  Output dir: {self.output_dir}")

    def execute_round(
        self,
        round_number: int,
        max_turns: Optional[int] = None,
        turn_callback: Optional[Callable[[SocialRLMessage], None]] = None
    ) -> SocialRLRoundResult:
        """
        Execute a complete Social RL round.

        Args:
            round_number: Which round to execute
            max_turns: Override max turns
            turn_callback: Optional callback after each turn

        Returns:
            SocialRLRoundResult with messages, feedback, and adaptations
        """
        start_time = time.time()

        # Get round configuration
        round_config = self._get_round_config(round_number)
        participants = self._get_participants(round_number)

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"SOCIAL RL ROUND {round_number}")
            print(f"Scenario: {round_config.get('scenario', '')[:60]}...")
            print(f"Participants: {[p.get('identifier') for p in participants]}")
            print(f"{'='*60}\n")

        # Parse max turns
        if max_turns is None:
            max_turns = self._parse_max_turns(round_config.get("end_condition", "15"))

        messages: List[SocialRLMessage] = []
        policy_adaptations = []
        turn = 0

        # Main turn loop
        while turn < max_turns:
            for agent in participants:
                if turn >= max_turns:
                    break

                turn += 1

                # Execute turn with Social RL components
                message = self._execute_social_rl_turn(
                    agent, round_config, messages, turn
                )
                messages.append(message)

                if self.config.verbose:
                    print(f"[Turn {turn}] {agent.get('identifier')}:")
                    print(f"  {message.content[:150]}{'...' if len(message.content) > 150 else ''}")
                    if message.prar_cue_used:
                        print(f"  [PRAR: {message.prar_cue_used[:50]}...]")
                    print()

                if turn_callback:
                    turn_callback(message)

                # Extract per-turn feedback if enabled
                if self.config.extract_feedback_per_turn and turn % 3 == 0:
                    self._extract_incremental_feedback(messages, participants)

        # Extract final round feedback
        participant_ids = [p.get("identifier") for p in participants]
        round_feedback = self.feedback_extractor.extract_round_feedback(
            round_number,
            [{"agent_id": m.agent_id, "content": m.content} for m in messages],
            participant_ids
        )

        # Adapt policies based on feedback if enabled
        if self.config.adapt_policies_per_round and round_number > 1:
            adaptations = self._adapt_policies_from_feedback(round_number)
            policy_adaptations.extend(adaptations)

        # Update accumulated feedback
        for agent_id, fb in round_feedback.items():
            self.accumulated_feedback[agent_id] = fb.as_reward_signal()

        duration = time.time() - start_time

        result = SocialRLRoundResult(
            round_number=round_number,
            messages=messages,
            feedback=round_feedback,
            policy_adaptations=policy_adaptations,
            duration_seconds=duration
        )

        self.round_results[round_number] = result

        # Auto-save round result
        if self.config.auto_save:
            self._save_round(result)

        if self.config.verbose:
            print(f"\nRound {round_number} complete: {len(messages)} messages in {duration:.1f}s")
            self._print_feedback_summary(round_feedback)

        return result

    def _execute_social_rl_turn(
        self,
        agent: Dict[str, Any],
        round_config: Dict[str, Any],
        history: List[SocialRLMessage],
        turn_number: int
    ) -> SocialRLMessage:
        """
        Execute a single turn with full Social RL pipeline.

        1. Generate dynamic context (ContextInjector)
        2. Retrieve reasoning policy (ProcessRetriever)
        3. Compile dynamic prompt
        4. Generate with Coach/Performer validation
        5. Attach feedback snapshot
        """
        agent_id = agent.get("identifier", "Unknown")

        # 1. Generate dynamic turn context
        turn_context = self.context_injector.generate_turn_context(
            agent_id=agent_id,
            agent_config=agent,
            round_config=round_config,
            turn_number=turn_number,
            conversation_history=[
                {"agent_id": m.agent_id, "content": m.content}
                for m in history
            ],
            accumulated_feedback=self.accumulated_feedback
        )

        # 2. Retrieve reasoning policy
        role = agent.get("role", "Worker")
        agent_feedback = self.accumulated_feedback.get(agent_id, {})
        policy = self.process_retriever.retrieve_policy(
            role=role,
            feedback=agent_feedback,
            round_number=round_config.get("round_number", 1),
            turn_number=turn_number
        )

        # Generate PRAR cue
        prar_cue = ""
        if self.config.use_prar_cues:
            prar_cue = self.process_retriever.generate_rcm_cue(
                policy, agent_feedback, self.config.prar_intensity
            )

        # 3. Compile dynamic prompt
        system_prompt = self.context_injector.compile_dynamic_prompt(agent, turn_context)
        if prar_cue:
            system_prompt += f"\n\n=== REASONING GUIDANCE ===\n{prar_cue}"

        # 4. Build user message (conversation context)
        user_message = self._build_user_message(history, agent, round_config)

        # 5. Generate response (with validation if enabled)
        if self.config.use_coach_validation:
            content, validation_meta = self._generate_with_validation(
                system_prompt, user_message, round_config.get("rules", ""),
                agent.get("behaviors", {}).get("raw", ""),
                agent_id=agent_id,
                turn_number=turn_number
            )
        else:
            content = self._generate_simple(system_prompt, user_message)
            validation_meta = None

        # Create message with Social RL metadata
        return SocialRLMessage(
            agent_id=agent_id,
            content=content,
            round_number=round_config.get("round_number", 1),
            turn_number=turn_number,
            turn_context=turn_context.to_dict(),
            prar_cue_used=prar_cue,
            feedback_snapshot=agent_feedback.copy() if agent_feedback else None,
            validation_metadata=validation_meta
        )

    def _build_user_message(
        self,
        history: List[SocialRLMessage],
        agent: Dict[str, Any],
        round_config: Dict[str, Any]
    ) -> str:
        """Build the user message with conversation context."""
        if history:
            context_lines = ["CONVERSATION SO FAR:"]
            for msg in history[-10:]:
                context_lines.append(f"[{msg.agent_id}]: {msg.content}")
            context = "\n".join(context_lines)
            return f"{context}\n\nIt is now your turn to respond as {agent.get('name', 'Unknown')}."
        else:
            return f"The round begins. {round_config.get('scenario', '')}\n\nRespond as {agent.get('name', 'Unknown')}."

    def _generate_with_validation(
        self,
        system_prompt: str,
        user_message: str,
        rules: str,
        behaviors: str,
        agent_id: str = "Unknown",
        turn_number: int = 0
    ) -> tuple:
        """Generate with Coach/Performer validation pattern."""
        metadata = {"attempts": 0, "validations": [], "filtered": False, "used_dual_llm": False}

        # Use DualLLMClient if available
        if self.dual_llm is not None:
            metadata["used_dual_llm"] = True
            rules_list = [r.strip() for r in rules.split(".") if r.strip()] if rules else []

            result: GenerationResult = self.dual_llm.generate_validated(
                system_prompt=system_prompt,
                user_message=user_message,
                agent_id=agent_id,
                rules=rules_list,
                context={"behaviors": behaviors},
                turn_number=turn_number
            )

            metadata["attempts"] = result.retries + 1
            metadata["validations"] = [
                {"valid": c.accepted, "issues": c.violations}
                for c in result.coach_critiques
            ]

            content = result.content
            if "[If " in content or "[if " in content:
                content = self._filter_prompt_leaks(content)
                metadata["filtered"] = True

            return content, metadata

        # Fallback: Original validation logic
        for attempt in range(self.config.max_validation_retries + 1):
            metadata["attempts"] = attempt + 1

            # Performer generates
            raw_output = self.llm.send_message(system_prompt, user_message)

            # Simple prompt leak filter
            if "[If " in raw_output or "[if " in raw_output:
                raw_output = self._filter_prompt_leaks(raw_output)
                metadata["filtered"] = True

            # Coach validates (simplified - full impl would use separate call)
            if rules:
                is_valid, issues = self._validate_output(raw_output, rules, behaviors)
                metadata["validations"].append({"valid": is_valid, "issues": issues})

                if is_valid:
                    return raw_output, metadata

                # Add feedback for retry
                user_message += f"\n\n[Previous attempt had issues: {issues}. Please try again following the rules.]"
            else:
                return raw_output, metadata

        return raw_output, metadata

    def _generate_simple(self, system_prompt: str, user_message: str) -> str:
        """Simple generation without validation."""
        return self.llm.send_message(system_prompt, user_message)

    def _filter_prompt_leaks(self, text: str) -> str:
        """Remove prompt leaks from output."""
        import re
        # Remove [If X: do Y] patterns
        return re.sub(r'\[If [^\]]+\]', '', text).strip()

    def _validate_output(self, output: str, rules: str, behaviors: str) -> tuple:
        """Simplified validation check."""
        issues = []
        output_lower = output.lower()

        # Check for obvious rule violations
        if "cannot" in rules.lower():
            cannot_parts = rules.lower().split("cannot:")
            if len(cannot_parts) > 1:
                prohibited = cannot_parts[1].split(",")
                for item in prohibited:
                    item = item.strip().split(".")[0]
                    if item and item in output_lower:
                        issues.append(f"Prohibited: {item}")

        return len(issues) == 0, issues

    def _extract_incremental_feedback(
        self,
        messages: List[SocialRLMessage],
        participants: List[Dict[str, Any]]
    ):
        """Extract feedback incrementally during round."""
        participant_ids = [p.get("identifier") for p in participants]
        msg_dicts = [{"agent_id": m.agent_id, "content": m.content} for m in messages]

        # Extract but don't update accumulated yet
        _ = self.feedback_extractor.extract_round_feedback(
            0,  # Temp round number
            msg_dicts,
            participant_ids
        )

    def _adapt_policies_from_feedback(self, round_number: int) -> List[Dict[str, Any]]:
        """Adapt policies based on feedback delta between rounds."""
        adaptations = []

        if round_number < 2 or (round_number - 1) not in self.round_results:
            return adaptations

        comparison = self.feedback_extractor.compare_rounds(round_number - 1, round_number)

        for agent_id, deltas in comparison.items():
            role = agent_id.split("+")[0] if "+" in agent_id else "Worker"

            self.process_retriever.adapt_policy(role, deltas, learning_rate=0.15)

            adaptations.append({
                "agent_id": agent_id,
                "role": role,
                "deltas": deltas,
                "round": round_number
            })

        return adaptations

    def _get_round_config(self, round_number: int) -> Dict[str, Any]:
        """Get round configuration from canvas."""
        for r in self.canvas.get("rounds", []):
            if r.get("round_number") == round_number:
                return r
        raise ValueError(f"Round {round_number} not found in canvas")

    def _get_participants(self, round_number: int) -> List[Dict[str, Any]]:
        """Get participant agent configs for a round."""
        round_config = self._get_round_config(round_number)
        platform_config = round_config.get("platform_config", {})

        participants_str = platform_config.get("participants", "")
        if isinstance(participants_str, str):
            participant_ids = [p.strip() for p in participants_str.split(",") if p.strip()]
        else:
            participant_ids = list(participants_str) if participants_str else []

        agents = []
        for agent in self.canvas.get("agents", []):
            if agent.get("identifier") in participant_ids:
                # Parse role from identifier
                identifier = agent.get("identifier", "")
                if "+" in identifier:
                    role, name = identifier.split("+", 1)
                else:
                    role, name = identifier, identifier
                agent["role"] = role
                agent["name"] = name
                agents.append(agent)

        return agents

    def _parse_max_turns(self, end_condition: str) -> int:
        """Parse max turns from end condition."""
        end_lower = end_condition.lower()
        if "total messages:" in end_lower:
            try:
                return int(end_lower.split(":")[-1].strip())
            except ValueError:
                pass
        return 15

    def _print_feedback_summary(self, feedback: Dict[str, SocialFeedback]):
        """Print feedback summary."""
        print("\nFeedback Summary:")
        for agent_id, fb in feedback.items():
            print(f"  {agent_id}:")
            print(f"    Engagement: {fb.engagement:.2f}")
            print(f"    Alignment: {fb.theoretical_alignment:.2f}")
            print(f"    Contribution: {fb.contribution_value:.2f}")

    def execute_all_rounds(
        self,
        max_turns_per_round: Optional[int] = None
    ) -> List[SocialRLRoundResult]:
        """Execute all rounds in the canvas."""
        rounds = self.canvas.get("rounds", [])
        results = []

        for round_config in rounds:
            result = self.execute_round(
                round_config.get("round_number", 1),
                max_turns=max_turns_per_round
            )
            results.append(result)

        return results

    def generate_report(self) -> str:
        """Generate comprehensive Social RL report."""
        report = ["=" * 60, "SOCIAL RL SIMULATION REPORT", "=" * 60, ""]

        # Framework info
        project = self.canvas.get("project", {})
        report.append(f"Framework: {project.get('theoretical_option_label', 'Unknown')}")
        report.append(f"Manifestation Mode: {self.config.manifestation_mode}")
        report.append("")

        # Per-round summaries
        for round_num, result in sorted(self.round_results.items()):
            report.append(f"--- Round {round_num} ---")
            report.append(f"Messages: {len(result.messages)}")
            report.append(f"Duration: {result.duration_seconds:.1f}s")

            if result.feedback:
                report.append("Feedback:")
                for agent_id, fb in result.feedback.items():
                    report.append(f"  {agent_id}: eng={fb.engagement:.2f}, align={fb.theoretical_alignment:.2f}")

            if result.policy_adaptations:
                report.append(f"Policy Adaptations: {len(result.policy_adaptations)}")

            report.append("")

        # Cumulative feedback
        if self.accumulated_feedback:
            report.append("--- Cumulative Feedback ---")
            for agent_id, fb in self.accumulated_feedback.items():
                report.append(f"{agent_id}: {fb}")

        # Process retrieval history
        report.append("")
        report.append(self.process_retriever.get_history_summary())

        return "\n".join(report)

    def _save_round(self, result: SocialRLRoundResult):
        """Save a single round result (called automatically if auto_save=True)."""
        try:
            # Ensure directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            output_file = self.output_dir / f"round{result.round_number}_social_rl.json"
            with open(output_file, "w") as f:
                json.dump(result.to_dict(), f, indent=2)

            # Also save policy state after each round
            policy_file = self.output_dir / "policy_state.json"
            self.process_retriever.save_policy_state(str(policy_file))

            if self.config.verbose:
                print(f"  [SAVED] {output_file}")
        except Exception as e:
            print(f"  [SAVE ERROR] Failed to save round {result.round_number}: {e}")

    def save_results(self, output_dir: str = None):
        """Save all results to files."""
        output_path = Path(output_dir) if output_dir else self.output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        # Save round results
        for round_num, result in self.round_results.items():
            with open(output_path / f"round{round_num}_social_rl.json", "w") as f:
                json.dump(result.to_dict(), f, indent=2)

        # Save policy state
        self.process_retriever.save_policy_state(str(output_path / "policy_state.json"))

        # Save report
        with open(output_path / "social_rl_report.txt", "w") as f:
            f.write(self.generate_report())

        if self.config.verbose:
            print(f"\nResults saved to: {output_dir}")


# Convenience function for quick testing
def create_social_rl_runner(
    state_path: str,
    llm_client: Any,
    mode: str = "progressive",
    dual_llm_client: Optional[DualLLMClient] = None
) -> SocialRLRunner:
    """Create a SocialRLRunner from a state file.

    Args:
        state_path: Path to state.json file
        llm_client: LLM client for generation
        mode: Manifestation mode (static, progressive, reactive, adaptive)
        dual_llm_client: Optional DualLLMClient for Coach/Performer architecture

    Returns:
        Configured SocialRLRunner
    """
    with open(state_path, "r") as f:
        state = json.load(f)

    canvas = state.get("canvas", state)

    config = SocialRLConfig(
        manifestation_mode=mode,
        verbose=True
    )

    return SocialRLRunner(canvas, llm_client, config, dual_llm_client=dual_llm_client)


if __name__ == "__main__":
    print("=== SocialRLRunner ===")
    print("Social RL: Learning through interaction, guided by process retrieval.")
    print()
    print("Usage:")
    print("  from social_rl.runner import create_social_rl_runner")
    print("  runner = create_social_rl_runner('state.json', llm_client)")
    print("  results = runner.execute_all_rounds()")
    print("  print(runner.generate_report())")
