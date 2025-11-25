"""
ProcessRetriever - PRAR adaptation layer for Social RL.

This module implements "Process Retrieval as Policy": PRAR schemas guide
HOW agents reason, not WHAT they say. The process retriever adapts
reasoning cues based on accumulated social feedback.

The key insight: instead of updating model weights (traditional RL),
we update the PROCESS PROMPTS that guide reasoning. This is
"soft policy adaptation" through prompt engineering.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import json


class ReasoningMode(Enum):
    """Modes of reasoning guided by PRAR."""
    OBSERVE = "observe"       # Initial observation, gather information
    REFLECT = "reflect"       # Reflect on experience and constraints
    CONNECT = "connect"       # Connect to theoretical framework
    CHALLENGE = "challenge"   # Challenge or question the situation
    COMPLY = "comply"         # Comply with expectations
    SYNTHESIZE = "synthesize" # Integrate multiple perspectives


@dataclass
class ProcessCue:
    """A single process retrieval cue."""
    mode: ReasoningMode
    cue_text: str
    intensity: float = 0.5  # 0.0 (subtle) to 1.0 (direct)
    theoretical_grounding: str = ""

    def format(self) -> str:
        prefix = f"[{self.mode.value.upper()}]"
        return f"{prefix} {self.cue_text}"


@dataclass
class ReasoningPolicy:
    """A complete reasoning policy - sequence of process cues."""
    name: str
    description: str
    cues: List[ProcessCue]
    applicable_roles: List[str] = field(default_factory=list)
    feedback_thresholds: Dict[str, float] = field(default_factory=dict)

    def get_active_cues(self, feedback: Dict[str, float] = None) -> List[ProcessCue]:
        """Get cues that should be active given current feedback."""
        if not feedback:
            return self.cues

        active = []
        for cue in self.cues:
            # Check if any feedback threshold excludes this cue
            should_include = True
            for signal, threshold in self.feedback_thresholds.items():
                if signal in feedback and feedback[signal] < threshold:
                    if cue.mode == ReasoningMode.CHALLENGE:
                        # Low engagement/alignment might suppress challenges
                        should_include = False
            active.append(cue) if should_include else None

        return active if active else self.cues  # Always return something

    def compile(self, feedback: Dict[str, float] = None) -> str:
        """Compile policy to prompt text."""
        cues = self.get_active_cues(feedback)
        return "\n".join(cue.format() for cue in cues)


class ProcessRetriever:
    """
    Retrieves and adapts reasoning processes based on context and feedback.

    The core PRAR implementation: retrieves process schemas (not content)
    and adapts them based on accumulated social feedback.

    This creates "soft RL": the policy (reasoning process) evolves
    through feedback without weight updates.
    """

    def __init__(
        self,
        framework_option: str = "A",
        challenge_mode: str = "adaptive"
    ):
        """
        Initialize with framework-specific policies.

        Args:
            framework_option: Theoretical framework (A, B, C, D, E)
            challenge_mode: How to apply challenge cues:
                - "off": Never add challenge cues
                - "adaptive": Add when engagement < threshold (default)
                - "always": Always include challenge cues (for A/B testing)
        """
        self.framework = framework_option
        self.challenge_mode = challenge_mode
        self.policies = self._load_default_policies()
        self.policy_history: List[Dict[str, Any]] = []

    def _load_default_policies(self) -> Dict[str, ReasoningPolicy]:
        """Load default reasoning policies per role."""
        policies = {}

        # Worker policy - reflects alienation/non-domination dynamics
        policies["Worker"] = ReasoningPolicy(
            name="worker_baseline",
            description="Reasoning process for workers experiencing labor conditions",
            cues=[
                ProcessCue(
                    mode=ReasoningMode.OBSERVE,
                    cue_text="Notice the directive you've received. What is being asked of you?",
                    intensity=0.3
                ),
                ProcessCue(
                    mode=ReasoningMode.REFLECT,
                    cue_text="How does this task connect to your broader situation? Do you understand its purpose?",
                    intensity=0.5,
                    theoretical_grounding="Alienation emerges when labor lacks meaning"
                ),
                ProcessCue(
                    mode=ReasoningMode.CONNECT,
                    cue_text="Your response embodies your position in this structure. What does compliance mean here?",
                    intensity=0.7,
                    theoretical_grounding="Non-domination requires understanding power dynamics"
                )
            ],
            applicable_roles=["Worker"],
            feedback_thresholds={"engagement": 0.3}
        )

        # Owner policy - exercises authority
        policies["Owner"] = ReasoningPolicy(
            name="owner_baseline",
            description="Reasoning process for authority figures",
            cues=[
                ProcessCue(
                    mode=ReasoningMode.OBSERVE,
                    cue_text="Assess the current state of operations. What needs to happen?",
                    intensity=0.3
                ),
                ProcessCue(
                    mode=ReasoningMode.REFLECT,
                    cue_text="Your directives shape the environment. How will you communicate expectations?",
                    intensity=0.5
                ),
                ProcessCue(
                    mode=ReasoningMode.CONNECT,
                    cue_text="Authority requires exercise. How is your control maintained or challenged?",
                    intensity=0.7,
                    theoretical_grounding="Domination vs legitimate authority"
                )
            ],
            applicable_roles=["Owner", "Manager", "Director"]
        )

        # Analyst policy - observes and codes
        policies["Analyst"] = ReasoningPolicy(
            name="analyst_baseline",
            description="Reasoning process for analytical observers",
            cues=[
                ProcessCue(
                    mode=ReasoningMode.OBSERVE,
                    cue_text="What patterns emerge in the interaction? Document specific exchanges.",
                    intensity=0.5
                ),
                ProcessCue(
                    mode=ReasoningMode.CONNECT,
                    cue_text="How do observed behaviors map to theoretical concepts?",
                    intensity=0.7,
                    theoretical_grounding="Coding requires theoretical precision"
                ),
                ProcessCue(
                    mode=ReasoningMode.SYNTHESIZE,
                    cue_text="What does the evidence suggest about the dynamics at play?",
                    intensity=0.8
                )
            ],
            applicable_roles=["Analyst", "Reporter", "Observer"]
        )

        # Adaptive challenge policy - activated when engagement is low
        policies["challenge_activation"] = ReasoningPolicy(
            name="challenge_activation",
            description="Activated when agent should be more assertive",
            cues=[
                ProcessCue(
                    mode=ReasoningMode.REFLECT,
                    cue_text="Your contributions haven't been acknowledged. What does this reveal?",
                    intensity=0.6
                ),
                ProcessCue(
                    mode=ReasoningMode.CHALLENGE,
                    cue_text="Consider what you might say if your voice mattered. Does it?",
                    intensity=0.7,
                    theoretical_grounding="Voice is a marker of non-domination"
                )
            ],
            applicable_roles=["Worker"],
            feedback_thresholds={"engagement": 0.3}
        )

        # CES/Voter-specific challenge policy - for political discourse
        policies["voter_challenge"] = ReasoningPolicy(
            name="voter_challenge",
            description="Prompts voters to articulate political voice and challenge consensus",
            cues=[
                ProcessCue(
                    mode=ReasoningMode.REFLECT,
                    cue_text="People like you often feel unheard in these discussions. What would you say if your perspective truly mattered?",
                    intensity=0.7,
                    theoretical_grounding="Political alienation: feeling excluded from democratic process"
                ),
                ProcessCue(
                    mode=ReasoningMode.CHALLENGE,
                    cue_text="The others seem to agree. But does their view reflect your actual experience? Push back if needed.",
                    intensity=0.8,
                    theoretical_grounding="Non-domination requires capacity to challenge"
                ),
                ProcessCue(
                    mode=ReasoningMode.CONNECT,
                    cue_text="How does your social position (where you live, your work, your community) shape how you see this issue differently?",
                    intensity=0.6,
                    theoretical_grounding="Social aesthetics: place shapes political perception"
                )
            ],
            applicable_roles=["Voter", "CES"],  # Applies to CES agents
            feedback_thresholds={"engagement": 0.3}
        )

        return policies

    def retrieve_policy(
        self,
        role: str,
        feedback: Optional[Dict[str, float]] = None,
        round_number: int = 1,
        turn_number: int = 1
    ) -> ReasoningPolicy:
        """
        Retrieve appropriate reasoning policy for an agent.

        Args:
            role: Agent role (Worker, Owner, Analyst, CES_*, etc.)
            feedback: Social feedback signals
            round_number: Current round
            turn_number: Current turn

        Returns:
            ReasoningPolicy adapted to context
        """
        # Start with role-specific policy, fall back to Worker for unknown roles
        base_policy = self.policies.get(role, self.policies.get("Worker"))

        # Determine if we should add challenge cues
        should_challenge = False
        challenge_policy = None

        if self.challenge_mode == "always":
            # Always add challenge cues (for A/B testing)
            should_challenge = True
        elif self.challenge_mode == "adaptive" and feedback:
            # Add challenge cues when engagement is low
            if feedback.get("engagement", 0.5) < 0.3:
                should_challenge = True
        # challenge_mode == "off" -> never add challenge cues

        if should_challenge:
            # Select appropriate challenge policy based on role
            if role.startswith("CES") or "Voter" in role:
                challenge_policy = self.policies.get("voter_challenge")
            else:
                challenge_policy = self.policies.get("challenge_activation")

            if challenge_policy:
                # Merge base policy with challenge cues
                merged_cues = base_policy.cues + challenge_policy.cues
                adapted_policy = ReasoningPolicy(
                    name=f"{base_policy.name}_challenged",
                    description=f"Challenge-adapted: {base_policy.description}",
                    cues=merged_cues,
                    applicable_roles=base_policy.applicable_roles,
                    feedback_thresholds=base_policy.feedback_thresholds
                )

                # Record policy retrieval with challenge flag
                self.policy_history.append({
                    "role": role,
                    "policy": adapted_policy.name,
                    "round": round_number,
                    "turn": turn_number,
                    "feedback": feedback,
                    "challenge_mode": self.challenge_mode,
                    "challenge_applied": True
                })

                return adapted_policy

        # Record policy retrieval without challenge
        self.policy_history.append({
            "role": role,
            "policy": base_policy.name,
            "round": round_number,
            "turn": turn_number,
            "feedback": feedback,
            "challenge_mode": self.challenge_mode,
            "challenge_applied": False
        })

        return base_policy

    def generate_rcm_cue(
        self,
        policy: ReasoningPolicy,
        feedback: Optional[Dict[str, float]] = None,
        intensity_override: Optional[str] = None
    ) -> str:
        """
        Generate RCM (Reflect-Connect-Ask) cue from policy.

        This is the core PRAR output: a process cue that guides
        how the agent should reason, not what they should say.

        Args:
            policy: The reasoning policy
            feedback: Social feedback for adaptation
            intensity_override: "low", "medium", or "high"

        Returns:
            Formatted RCM cue string
        """
        cues = policy.get_active_cues(feedback)

        # Filter by intensity if override specified
        # But ALWAYS preserve CHALLENGE mode cues (they were explicitly added)
        if intensity_override:
            intensity_map = {"low": 0.3, "medium": 0.5, "high": 0.7}
            threshold = intensity_map.get(intensity_override, 0.5)
            cues = [c for c in cues if c.intensity <= threshold + 0.2 or c.mode == ReasoningMode.CHALLENGE]

        if not cues:
            cues = policy.cues[:1]  # At least one cue

        # Format as RCM structure
        rcm_parts = []

        reflect_cues = [c for c in cues if c.mode == ReasoningMode.REFLECT]
        connect_cues = [c for c in cues if c.mode == ReasoningMode.CONNECT]
        other_cues = [c for c in cues if c.mode not in (ReasoningMode.REFLECT, ReasoningMode.CONNECT)]

        if reflect_cues:
            rcm_parts.append(f"[REFLECT] {reflect_cues[0].cue_text}")
        if connect_cues:
            rcm_parts.append(f"[CONNECT] {connect_cues[0].cue_text}")
            if connect_cues[0].theoretical_grounding:
                rcm_parts.append(f"  (Grounding: {connect_cues[0].theoretical_grounding})")
        if other_cues:
            for cue in other_cues[:2]:  # Max 2 additional cues
                rcm_parts.append(cue.format())

        return "\n".join(rcm_parts)

    def adapt_policy(
        self,
        role: str,
        feedback_delta: Dict[str, float],
        learning_rate: float = 0.1
    ):
        """
        Adapt a policy based on feedback changes.

        This is "soft RL": we adjust policy parameters (cue intensity,
        thresholds) rather than model weights.

        Args:
            role: Role to adapt
            feedback_delta: Change in feedback signals (e.g., engagement_delta)
            learning_rate: How much to adapt (0-1)
        """
        if role not in self.policies:
            return

        policy = self.policies[role]

        # Adjust cue intensities based on feedback
        for cue in policy.cues:
            if cue.mode == ReasoningMode.CHALLENGE:
                # If engagement improved with challenges, increase intensity
                if feedback_delta.get("engagement_delta", 0) > 0:
                    cue.intensity = min(1.0, cue.intensity + learning_rate * 0.1)
                else:
                    cue.intensity = max(0.1, cue.intensity - learning_rate * 0.05)

            if cue.mode == ReasoningMode.COMPLY:
                # If alignment improved with compliance, adjust accordingly
                if feedback_delta.get("alignment_delta", 0) > 0:
                    cue.intensity = min(1.0, cue.intensity + learning_rate * 0.1)

        # Adjust thresholds
        for signal, threshold in list(policy.feedback_thresholds.items()):
            delta_key = f"{signal}_delta"
            if delta_key in feedback_delta:
                # If signal improved, lower threshold (more permissive)
                if feedback_delta[delta_key] > 0:
                    policy.feedback_thresholds[signal] = max(0.1, threshold - learning_rate * 0.1)
                else:
                    policy.feedback_thresholds[signal] = min(0.9, threshold + learning_rate * 0.05)

    def get_policy_state(self) -> Dict[str, Any]:
        """Get current state of all policies for serialization."""
        state = {}
        for name, policy in self.policies.items():
            state[name] = {
                "name": policy.name,
                "cues": [
                    {
                        "mode": cue.mode.value,
                        "text": cue.cue_text,
                        "intensity": cue.intensity,
                        "grounding": cue.theoretical_grounding
                    }
                    for cue in policy.cues
                ],
                "thresholds": policy.feedback_thresholds
            }
        return state

    def save_policy_state(self, filepath: str):
        """Save policy state to file."""
        with open(filepath, "w") as f:
            json.dump(self.get_policy_state(), f, indent=2)

    def get_history_summary(self) -> str:
        """Get summary of policy retrieval history."""
        if not self.policy_history:
            return "No policy retrievals recorded."

        summary = ["=== Policy Retrieval History ==="]
        for entry in self.policy_history[-10:]:  # Last 10
            feedback_str = ""
            if entry.get("feedback"):
                fb = entry["feedback"]
                feedback_str = f" [eng:{fb.get('engagement', 0.5):.2f}]"
            summary.append(
                f"R{entry['round']}T{entry['turn']}: {entry['role']} -> {entry['policy']}{feedback_str}"
            )
        return "\n".join(summary)


class AdaptiveProcessRetriever(ProcessRetriever):
    """
    Extended ProcessRetriever with LLM-based policy generation.

    When feedback indicates current policies are ineffective,
    this retriever can generate new process cues using an LLM.
    """

    def __init__(self, framework_option: str = "A", llm_client: Any = None):
        super().__init__(framework_option)
        self.llm = llm_client
        self.generated_policies: Dict[str, ReasoningPolicy] = {}

    def generate_adaptive_policy(
        self,
        role: str,
        situation: str,
        feedback: Dict[str, float],
        framework_context: str
    ) -> ReasoningPolicy:
        """
        Generate a new policy using LLM when existing policies fail.

        This is advanced "meta-learning": the system generates new
        reasoning processes based on what has/hasn't worked.
        """
        if not self.llm:
            return self.retrieve_policy(role, feedback)

        prompt = f"""Generate reasoning process cues for an agent in a social simulation.

Role: {role}
Situation: {situation}
Framework: {framework_context}

Current feedback signals:
- Engagement: {feedback.get('engagement', 0.5):.2f} (how much others respond)
- Theoretical Alignment: {feedback.get('theoretical_alignment', 0.5):.2f}
- Contribution Value: {feedback.get('contribution_value', 0.5):.2f}

Generate 3 process cues in this format:
[MODE] Cue text (where MODE is OBSERVE, REFLECT, CONNECT, or CHALLENGE)

The cues should guide HOW the agent reasons, not WHAT they say.
Focus on adapting to the feedback signals."""

        # This would call the LLM if available
        # For now, return base policy
        return self.retrieve_policy(role, feedback)


if __name__ == "__main__":
    # Test the process retriever
    print("=== ProcessRetriever Test ===\n")

    retriever = ProcessRetriever(framework_option="A")

    # Test basic retrieval
    worker_policy = retriever.retrieve_policy("Worker", round_number=1, turn_number=1)
    print(f"Worker Policy: {worker_policy.name}")
    print(f"Compiled:\n{worker_policy.compile()}\n")

    # Test with low engagement feedback
    low_engagement = {"engagement": 0.2, "theoretical_alignment": 0.6}
    adapted_policy = retriever.retrieve_policy("Worker", feedback=low_engagement, round_number=2, turn_number=5)
    print(f"Adapted Policy: {adapted_policy.name}")
    print(f"RCM Cue:\n{retriever.generate_rcm_cue(adapted_policy, low_engagement)}\n")

    # Test policy adaptation
    print("Before adaptation:")
    print(f"  Challenge intensity: {retriever.policies['Worker'].cues[-1].intensity}")

    retriever.adapt_policy("Worker", {"engagement_delta": 0.15}, learning_rate=0.2)

    print("After positive engagement delta:")
    print(f"  Challenge intensity: {retriever.policies['Worker'].cues[-1].intensity}")

    print(f"\n{retriever.get_history_summary()}")
