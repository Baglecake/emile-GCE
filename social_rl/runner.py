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
from typing import Dict, Any, List, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from agents.identity_core import IdentityCore

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

# Phase 2a: IdentityCore integration (optional)
try:
    from agents.identity_core import IdentityCore, IdentityVector
    IDENTITY_CORE_AVAILABLE = True
except ImportError:
    IDENTITY_CORE_AVAILABLE = False
    IdentityCore = None
    IdentityVector = None


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

    # Phase 2a: IdentityCore integration
    use_identity_cores: bool = False  # Enable per-agent identity tracking
    identity_modulates_temperature: bool = True  # Let IdentityCore set temperature

    # Temporal compression (Phase 2b: will be calibrated from CES)
    temporal_config: Dict[str, Any] = field(default_factory=lambda: {
        'years_per_experiment': 4,
        'years_per_round': {'R1': 1.5, 'R2': 1.5, 'R3': 1.0}
    })

    # Phase 2b: Grit mode (static vs dynamic calibration)
    grit_mode: str = "dynamic"  # "static" = fixed at creation, "dynamic" = per-turn calibration
    grit_smoothing: float = 0.3  # Weight for new grit level (0.3 = 30% new, 70% old)
    grit_min_words: int = 30     # Minimum words even for STRONG grit
    grit_max_words: int = 300    # Maximum words even for NONE grit

    # Phase 2b: Verbosity penalty (Option B - social throttling)
    # Verbosity lowers engagement → triggers higher grit → natural throttling
    verbosity_penalty_enabled: bool = True
    verbosity_penalty_alpha: float = 0.15  # Penalty scaling factor
    verbosity_max_penalty: float = 0.25    # Maximum penalty cap

    # Phase 2b: Token cap throttling (hard enforcement)
    # If agent overshoots word limit, reduce max_tokens for next turn
    token_throttle_enabled: bool = True
    token_throttle_factor: float = 0.75    # Multiply cap by this after overshoot
    token_min_cap: int = 80                # Floor for token cap (~40 words)


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

        # Phase 2a: IdentityCore tracking (optional)
        self.identity_cores: Dict[str, 'IdentityCore'] = {}
        if self.config.use_identity_cores and IDENTITY_CORE_AVAILABLE:
            self._initialize_identity_cores()

        # Phase 2b: Dynamic grit states (per-agent grit level calibrated each turn)
        # Maps agent_id -> current GritLevel (updated by calibrate_grit_to_ces)
        self.grit_states: Dict[str, str] = {}  # agent_id -> "NONE"/"LIGHT"/"MODERATE"/"STRONG"

        # Phase 2b: Token caps (per-agent max_tokens, reduced after verbosity overshoot)
        # Maps agent_id -> current max_tokens limit
        self.token_caps: Dict[str, int] = {}  # agent_id -> max_tokens

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

        # Update accumulated feedback (use to_dict to include direct_references, response_received)
        for agent_id, fb in round_feedback.items():
            self.accumulated_feedback[agent_id] = fb.to_dict()

        # Phase 2a: Update identity cores with observed behavior
        if self.identity_cores:
            # First, gather all agent engagements for TE calculation
            agent_engagements = {
                aid: fb.as_reward_signal().get('engagement', 0.0)
                for aid, fb in round_feedback.items()
            }

            # Update TE histories BEFORE identity core update (critical ordering)
            for agent_id, ic in self.identity_cores.items():
                if agent_id in agent_engagements:
                    # Identity scalar: delta_I (how much identity has changed)
                    identity_scalar = ic.compute_delta_I()

                    # Behavior scalar: this agent's engagement this round
                    behavior_scalar = agent_engagements.get(agent_id, 0.0)

                    # Others scalar: mean engagement of OTHER agents
                    other_engagements = [v for k, v in agent_engagements.items() if k != agent_id]
                    others_scalar = sum(other_engagements) / len(other_engagements) if other_engagements else 0.0

                    # Update TE histories
                    ic.update_te_histories(identity_scalar, behavior_scalar, others_scalar)

            # Now update identity cores
            for agent_id, fb in round_feedback.items():
                self._update_identity_core(agent_id, fb.as_reward_signal(), round_number)

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

        # 4b. Get current grit state (needed for token cap initialization)
        agent_prompt = agent.get("prompt", "")
        old_grit = self.grit_states.get(agent_id)
        if old_grit is None:
            # Initialize from prompt
            if "GRIT-STRONG" in agent_prompt:
                old_grit = "STRONG"
            elif "GRIT-MODERATE" in agent_prompt:
                old_grit = "MODERATE"
            elif "GRIT-LIGHT" in agent_prompt:
                old_grit = "LIGHT"
            else:
                old_grit = "NONE"

        # 5. IDENTITY-GROUNDED EXPRESSION CAPACITY
        # Expression capacity emerges from the social field:
        # - Identity salience: how invested in this political identity
        # - Natality: capacity for new beginnings (τ-derived)
        # The hard cap is just a safety rail; capacity is constituted by the field.
        #
        # Formula: cap = base_cap × f_salience × f_natality
        # Where f_salience, f_natality ∈ [0.5, 1.0]

        TOKEN_CAPS_BY_GRIT = {
            "STRONG": 100,   # ~50 words
            "MODERATE": 150, # ~75 words
            "LIGHT": 256,    # ~128 words
            "NONE": 512      # default
        }

        # Get identity metrics for capacity computation
        identity_salience = 0.5  # Default neutral salience
        natality_t = 0.5  # Default neutral natality

        # Get salience from grit level (inverse relationship)
        # Higher grit = lower salience (more constrained identity)
        salience_map = {"STRONG": 0.15, "MODERATE": 0.30, "LIGHT": 0.45, "NONE": 0.75}
        identity_salience = salience_map.get(old_grit, 0.5)

        # Get STATEFUL natality from IdentityCore if available
        # This is the natality_t that gets updated by recognition/overshoot
        if agent_id in self.identity_cores:
            natality_t = self.identity_cores[agent_id].get_natality()

        # Compute identity-grounded capacity
        base_cap = TOKEN_CAPS_BY_GRIT.get(old_grit, 512)
        f_salience = 0.5 + 0.5 * identity_salience  # [0.5, 1.0]
        f_natality = 0.5 + 0.5 * natality_t         # [0.5, 1.0]
        soft_cap = int(base_cap * f_salience * f_natality)

        # Safety rails (hard clamp)
        max_tokens = max(self.config.token_min_cap, min(512, soft_cap))

        # Persist for logging (but recompute each turn from identity state)
        self.token_caps[agent_id] = max_tokens

        # 5. Generate response (with validation if enabled)
        if self.config.use_coach_validation:
            content, validation_meta = self._generate_with_validation(
                system_prompt, user_message, round_config.get("rules", ""),
                agent.get("behaviors", {}).get("raw", ""),
                agent_id=agent_id,
                turn_number=turn_number,
                max_tokens=max_tokens
            )
        else:
            content = self._generate_simple(system_prompt, user_message)
            validation_meta = None

        # 5b. DYNAMIC GRIT CALIBRATION: Adjust grit level based on engagement vs CES target
        # This is the "active dynamics" - grit level changes per-turn based on overshoot
        # See dev-notes.md for design rationale (smoothing, clamping, CES targets)
        current_engagement = agent_feedback.get('engagement', 0.5)

        # Grit level ordering for smoothing (NONE=0, LIGHT=1, MODERATE=2, STRONG=3)
        GRIT_LEVELS = {"NONE": 0, "LIGHT": 1, "MODERATE": 2, "STRONG": 3}
        GRIT_NAMES = {0: "NONE", 1: "LIGHT", 2: "MODERATE", 3: "STRONG"}

        # old_grit already computed in step 4b
        grit_level = old_grit  # Default: keep current level

        # Dynamic calibration if enabled
        if self.config.grit_mode == "dynamic":
            try:
                from agents.ces_generators.grit_config import calibrate_grit_to_ces, GritLevel

                # Infer identity_salience from current grit level
                salience_map = {"STRONG": 0.15, "MODERATE": 0.30, "LIGHT": 0.45, "NONE": 0.75}
                identity_metrics = {'identity_salience': salience_map.get(old_grit, 0.5)}

                # Get proposed new grit level from CES calibration
                dynamic_constraint = calibrate_grit_to_ces(agent_id, current_engagement, identity_metrics)
                proposed_grit = dynamic_constraint.level.value.upper()

                # SMOOTH the update (prevent whiplash)
                old_level = GRIT_LEVELS.get(old_grit, 0)
                proposed_level = GRIT_LEVELS.get(proposed_grit, 0)
                smoothed_level = round(
                    (1 - self.config.grit_smoothing) * old_level +
                    self.config.grit_smoothing * proposed_level
                )
                smoothed_level = max(0, min(3, smoothed_level))  # Clamp to valid range
                grit_level = GRIT_NAMES[smoothed_level]

                # Update state
                self.grit_states[agent_id] = grit_level

                if self.config.verbose and (grit_level != old_grit or grit_level != "NONE"):
                    print(f"    [GRIT-DYNAMIC] {agent_id}: {old_grit}→{grit_level} (eng={current_engagement:.2f})")
            except ImportError:
                pass  # Keep static grit if calibration not available
        else:
            # Static mode: just track current state
            self.grit_states[agent_id] = old_grit

        # 5c. GRIT ENFORCEMENT: Truncate based on grit level with clamping
        # Word limits per grit level (from dev-notes.md)
        WORD_LIMITS = {
            "STRONG": 50,    # ~2-3 sentences
            "MODERATE": 100, # ~4-6 sentences
            "LIGHT": 150,    # ~5-8 sentences
            "NONE": self.config.grit_max_words
        }

        words = content.split()
        word_limit = WORD_LIMITS.get(grit_level, self.config.grit_max_words)

        # Enforce minimum (don't over-truncate)
        word_limit = max(word_limit, self.config.grit_min_words)

        # 5d. RECOGNITION-DRIVEN NATALITY UPDATE
        # Expression capacity is constituted by the social field, not crude punishment.
        # Identity mechanics live in IdentityCore - runner just orchestrates.

        verbosity_penalty = 0.0
        engagement_base = agent_feedback.get('engagement', 0.5)

        # Compute recognition_score from feedback
        # Recognition = (direct_references + response_received) / expected
        direct_refs = agent_feedback.get('direct_references', 0)
        response_received = agent_feedback.get('response_received', 0)
        num_other_agents = max(1, len(self.canvas.get("agents", [])) - 1)
        raw_recognition = (direct_refs + response_received) / num_other_agents
        recognition_score = max(0.0, min(1.0, raw_recognition))

        # Compute overshoot ratio (if any)
        overshoot_ratio = 0.0
        if len(words) > word_limit:
            overshoot = len(words) - word_limit
            overshoot_ratio = overshoot / float(word_limit)

        # UPDATE NATALITY in IdentityCore (the identity mechanic lives there)
        # This implements: overshoot + low recognition → natality drains ("talked over")
        #                  overshoot + high recognition → natality rises ("held and amplified")
        if agent_id in self.identity_cores:
            core = self.identity_cores[agent_id]
            old_natality = core.get_natality()

            # Call IdentityCore's update_natality method
            new_natality = core.update_natality(
                recognition_score=recognition_score,
                overshoot_ratio=overshoot_ratio
            )

            if self.config.verbose and abs(new_natality - old_natality) > 0.01:
                print(f"    [NATALITY] {agent_id}: {old_natality:.2f}→{new_natality:.2f} "
                      f"(recog={recognition_score:.2f}, overshoot={overshoot_ratio:.2f})")

            # Stage 3: SurplusTrace management
            # 1. Decay existing traces (channels lose efficacy unless re-cohered)
            core.decay_traces()

            # 2. Try to create trace on high-surplus, high-recognition events
            round_number = round_config.get("round_number", 1)
            semiotic_regime = round_config.get("semiotic_regime", "UNKNOWN")
            contribution_value = agent_feedback.get('contribution_value', 0.0) if agent_feedback else 0.0
            engagement_fb = agent_feedback.get('engagement', 0.0) if agent_feedback else 0.0

            trace = core.maybe_create_trace(
                round_number=round_number,
                turn_number=turn_number,
                semiotic_regime=semiotic_regime,
                recognition_score=recognition_score,
                contribution_value=contribution_value,
                engagement=engagement_fb,
            )
            if trace and self.config.verbose:
                print(f"    [TRACE] Created for {agent_id}: weight={trace.weight:.3f}, "
                      f"regime={semiotic_regime}")

            # 3. Revalorize existing traces if current ΔI is similar
            if len(core.history) >= 2:
                prev_vec = core.history[-2][1]
                curr_vec = core.vector
                delta_I_vec = curr_vec - prev_vec
                delta_I_arr = delta_I_vec.to_array()

                num_revalorized = core.revalorize_traces(delta_I_arr, recognition_score)
                if num_revalorized > 0 and self.config.verbose:
                    print(f"    [TRACE] Revalorized {num_revalorized} trace(s) for {agent_id}")

        # Verbosity penalty (mild engagement nudge - main effect is via natality)
        if self.config.verbosity_penalty_enabled and overshoot_ratio > 0:
            # Small engagement penalty (mild nudge, not main effect)
            verbosity_penalty = min(
                overshoot_ratio * self.config.verbosity_penalty_alpha * 0.5,  # Reduced
                self.config.verbosity_max_penalty * 0.5
            )

            # Apply penalty to engagement
            penalized_engagement = max(0.0, engagement_base - verbosity_penalty)

            # Update accumulated feedback
            if agent_id not in self.accumulated_feedback:
                self.accumulated_feedback[agent_id] = {}
            self.accumulated_feedback[agent_id]['engagement'] = penalized_engagement
            self.accumulated_feedback[agent_id]['engagement_base'] = engagement_base
            self.accumulated_feedback[agent_id]['verbosity_penalty'] = verbosity_penalty
            self.accumulated_feedback[agent_id]['word_count'] = len(words)
            self.accumulated_feedback[agent_id]['word_limit'] = word_limit
            self.accumulated_feedback[agent_id]['recognition_score'] = recognition_score

            if self.config.verbose and verbosity_penalty > 0.01:
                print(f"    [VERBOSITY] {agent_id}: {len(words)} words > {word_limit} limit → "
                      f"eng {engagement_base:.2f}→{penalized_engagement:.2f} (penalty={verbosity_penalty:.2f})")

        if len(words) > word_limit + 10:  # Allow small buffer
            # Find sentence boundary near limit
            truncated = " ".join(words[:word_limit])
            for end_char in [". ", "? ", "! "]:
                last_pos = truncated.rfind(end_char)
                if last_pos > self.config.grit_min_words:
                    truncated = truncated[:last_pos + 1]
                    break
            content = truncated.strip()
            if self.config.verbose:
                print(f"    [GRIT] Truncated from {len(words)} to {len(content.split())} words (limit={word_limit})")

        # Build feedback snapshot with identity metrics
        feedback_snapshot = agent_feedback.copy() if agent_feedback else {}
        feedback_snapshot['word_count'] = len(content.split())  # Post-truncation count
        feedback_snapshot['word_limit'] = word_limit
        feedback_snapshot['grit_level'] = grit_level
        feedback_snapshot['verbosity_penalty'] = verbosity_penalty
        feedback_snapshot['max_tokens'] = max_tokens  # Identity-grounded capacity
        feedback_snapshot['identity_salience'] = identity_salience
        feedback_snapshot['natality_t'] = natality_t
        feedback_snapshot['recognition_score'] = recognition_score
        feedback_snapshot['f_salience'] = f_salience
        feedback_snapshot['f_natality'] = f_natality
        if verbosity_penalty > 0:
            feedback_snapshot['engagement_base'] = engagement_base

        # Create message with Social RL metadata
        return SocialRLMessage(
            agent_id=agent_id,
            content=content,
            round_number=round_config.get("round_number", 1),
            turn_number=turn_number,
            turn_context=turn_context.to_dict(),
            prar_cue_used=prar_cue,
            feedback_snapshot=feedback_snapshot,
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
        turn_number: int = 0,
        max_tokens: Optional[int] = None
    ) -> tuple:
        """Generate with Coach/Performer validation pattern."""
        metadata = {"attempts": 0, "validations": [], "filtered": False, "used_dual_llm": False}
        if max_tokens:
            metadata["max_tokens"] = max_tokens

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
                turn_number=turn_number,
                max_tokens=max_tokens
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

            # Save round result
            output_file = self.output_dir / f"round{result.round_number}_social_rl.json"
            round_data = result.to_dict()

            # Phase 2a: Include identity core states in round data
            if self.identity_cores:
                round_data['identity_states'] = self.get_identity_states()

            with open(output_file, "w") as f:
                json.dump(round_data, f, indent=2)

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

        # Phase 2a: Save identity core states
        if self.identity_cores:
            identity_state = {
                agent_id: core.get_state()
                for agent_id, core in self.identity_cores.items()
            }
            with open(output_path / "identity_cores.json", "w") as f:
                json.dump(identity_state, f, indent=2)

        if self.config.verbose:
            print(f"\nResults saved to: {output_dir}")

    # =========================================================================
    # Phase 2a: IdentityCore Integration
    # =========================================================================

    def _initialize_identity_cores(self):
        """Initialize IdentityCore for each agent in the canvas."""
        if not IDENTITY_CORE_AVAILABLE:
            return

        for agent in self.canvas.get("agents", []):
            agent_id = agent.get("identifier", "")
            if not agent_id:
                continue

            # Infer group_id from agent identifier
            group_id = self._infer_group_id(agent_id)

            # Extract identity metrics from canvas attributes
            attrs = agent.get("attributes", {})
            identity_salience = attrs.get("identity_salience", 0.5)
            tie_to_place = attrs.get("tie_to_place", 0.5)

            # Create initial identity vector from 7D canvas data (Phase 2.4)
            # or fall back to neutral baseline
            identity_7d = attrs.get("identity_vector_7d", {})
            if identity_7d:
                initial_vec = IdentityVector(values=identity_7d)
            else:
                initial_vec = IdentityVector(values={
                    'engagement': 0.5,  # Neutral engagement
                    'institutional_faith': 0.8,  # Default moderate faith
                    'social_friction': 0.2,  # Low initial friction
                })

            self.identity_cores[agent_id] = IdentityCore(
                agent_id=agent_id,
                group_id=group_id,
                initial_vector=initial_vec,
                identity_salience=identity_salience,
                tie_to_place=tie_to_place,
            )

            # Phase 2b: Load drift priors for tau calibration
            # Build profile from agent attributes for group modifiers
            profile = attrs.get("ces_profile", {})
            if not profile:
                # Infer basic profile from agent attributes if no CES profile
                profile = {
                    "Region": attrs.get("region", "Ontario"),
                    "cps21_urban_rural": 1 if "urban" in agent_id.lower() else (3 if "rural" in agent_id.lower() else 2),
                }
            self.identity_cores[agent_id].load_drift_priors(profile)

        if self.config.verbose and self.identity_cores:
            print(f"  IdentityCore: Initialized {len(self.identity_cores)} identity cores")

    def _infer_group_id(self, agent_id: str) -> str:
        """Infer CES strata group_id from agent identifier."""
        aid_lower = agent_id.lower()

        # Location
        if 'urban' in aid_lower:
            location = 'urban'
        elif 'rural' in aid_lower:
            location = 'rural'
        elif 'suburban' in aid_lower:
            location = 'suburban'
        else:
            location = ''

        # Tenure
        if 'renter' in aid_lower:
            tenure = 'renter'
        elif 'owner' in aid_lower or 'homeowner' in aid_lower:
            tenure = 'owner'
        else:
            tenure = ''

        # Political lean
        if 'progressive' in aid_lower or 'liberal' in aid_lower:
            lean = 'left'
        elif 'conservative' in aid_lower:
            lean = 'right'
        elif 'swing' in aid_lower or 'moderate' in aid_lower:
            lean = 'center'
        else:
            lean = ''

        # Engagement level
        if 'disengaged' in aid_lower:
            engagement = 'disengaged'
        elif 'engaged' in aid_lower or 'active' in aid_lower:
            engagement = 'engaged'
        else:
            engagement = ''

        parts = [p for p in [location, tenure, lean, engagement] if p]
        return '_'.join(parts) if parts else 'unknown'

    def _update_identity_core(
        self,
        agent_id: str,
        feedback: Dict[str, float],
        round_number: int
    ):
        """Update an agent's IdentityCore with observed behavior."""
        if agent_id not in self.identity_cores:
            return

        core = self.identity_cores[agent_id]

        # Compute sim_time from temporal config
        temporal_cfg = self.config.temporal_config
        years_per_round = temporal_cfg.get('years_per_round', {})
        sim_time = sum(
            years_per_round.get(f'R{i}', 1.0)
            for i in range(1, round_number + 1)
        )

        # Extract vector from feedback
        # Map feedback signals to identity dimensions
        engagement = feedback.get('engagement', core.vector.engagement)
        # contribution_value maps loosely to institutional_faith
        faith = 1.0 - feedback.get('critical_ratio', 0.0)  # inverse of criticism
        friction = feedback.get('direct_references', core.vector.social_friction)

        new_vec = IdentityVector(values={
            'engagement': engagement,
            'institutional_faith': faith,
            'social_friction': friction,
        })

        core.update(new_vec, sim_time)

    def _get_identity_temperature(self, agent_id: str) -> Optional[float]:
        """Get temperature from IdentityCore if available."""
        if agent_id in self.identity_cores:
            return self.identity_cores[agent_id].compute_temperature()
        return None

    def get_identity_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current state of all identity cores."""
        return {
            agent_id: core.get_state()
            for agent_id, core in self.identity_cores.items()
        }


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
