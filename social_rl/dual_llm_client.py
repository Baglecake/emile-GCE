"""
Dual-LLM Client for Social RL

Implements the Coach/Performer architecture with distinct LLM clients
for validation (Coach) and generation (Performer) roles.

Coach (low temperature):
- Validates agent outputs against rules and constraints
- Generates corrective feedback when violations detected
- Enforces PRAR protocol and theoretical grounding

Performer (higher temperature):
- Generates agent dialogue and responses
- Inhabits agent personas authentically
- Produces creative, contextually appropriate content
"""

from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time


@dataclass
class DualLLMConfig:
    """Configuration for dual-LLM architecture."""
    performer_temperature: float = 0.7
    coach_temperature: float = 0.1
    performer_max_tokens: int = 512
    coach_max_tokens: int = 256
    log_coach_critiques: bool = True
    max_validation_retries: int = 2


@dataclass
class CoachCritique:
    """Record of a coach critique/validation."""
    agent_id: str
    turn_number: int
    original_content: str
    critique: str
    violations: List[str]
    suggested_revision: Optional[str]
    timestamp: float
    accepted: bool


@dataclass
class GenerationResult:
    """Result from a generation with optional validation."""
    content: str
    agent_id: str
    mode: str  # "performer" or "coach"
    temperature: float
    validation_passed: bool = True
    retries: int = 0
    coach_critiques: List[CoachCritique] = field(default_factory=list)
    duration_seconds: float = 0.0


class LLMClientProtocol(ABC):
    """Protocol for LLM clients used with DualLLMClient."""

    @abstractmethod
    def send_message(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """Send a message and get a response."""
        pass


class DualLLMClient:
    """
    Dual-LLM client implementing Coach/Performer architecture.

    The Coach validates outputs for rule compliance and theoretical alignment.
    The Performer generates agent dialogue and responses.

    Usage:
        # With two separate clients
        dual = DualLLMClient(performer_client, coach_client)

        # With single client (different temperatures)
        dual = DualLLMClient(single_client)

        # Generate with validation
        result = dual.generate_validated(
            system_prompt="You are Alice, a factory worker...",
            user_message="Respond to Marta's directive.",
            agent_id="Worker+Alice",
            rules=["No direct confrontation", "Stay in character"],
            context={"round": 1, "turn": 3}
        )
    """

    def __init__(
        self,
        performer_client: LLMClientProtocol,
        coach_client: Optional[LLMClientProtocol] = None,
        config: Optional[DualLLMConfig] = None
    ):
        """
        Initialize dual-LLM client.

        Args:
            performer_client: LLM client for generation
            coach_client: LLM client for validation (defaults to performer_client)
            config: Configuration options
        """
        self.performer = performer_client
        self.coach = coach_client or performer_client
        self.config = config or DualLLMConfig()
        self._critique_log: List[CoachCritique] = []

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        mode: str = "performer",
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response using the specified mode.

        Args:
            system_prompt: System prompt for the LLM
            user_message: User message/context
            mode: "performer" or "coach"
            max_tokens: Override default max tokens

        Returns:
            Generated text response
        """
        if mode == "performer":
            client = self.performer
            temp = self.config.performer_temperature
            tokens = max_tokens or self.config.performer_max_tokens
        elif mode == "coach":
            client = self.coach
            temp = self.config.coach_temperature
            tokens = max_tokens or self.config.coach_max_tokens
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'performer' or 'coach'.")

        return client.send_message(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temp,
            max_tokens=tokens
        )

    def generate_validated(
        self,
        system_prompt: str,
        user_message: str,
        agent_id: str,
        rules: List[str],
        context: Optional[Dict[str, Any]] = None,
        turn_number: int = 0
    ) -> GenerationResult:
        """
        Generate a response with coach validation.

        The performer generates initial content, then the coach validates
        against rules. If violations are found, the performer regenerates
        with corrective feedback.

        Args:
            system_prompt: System prompt for performer
            user_message: User message/context
            agent_id: Agent identifier for logging
            rules: List of rules to validate against
            context: Additional context for validation
            turn_number: Current turn number

        Returns:
            GenerationResult with content, validation status, and critiques
        """
        start_time = time.time()
        critiques = []
        retries = 0

        # Initial generation
        content = self.generate(system_prompt, user_message, mode="performer")

        # Validation loop
        for attempt in range(self.config.max_validation_retries + 1):
            # Validate with coach
            is_valid, violations, suggested = self._validate_with_coach(
                content=content,
                agent_id=agent_id,
                rules=rules,
                context=context
            )

            if is_valid:
                break

            # Log critique
            critique = CoachCritique(
                agent_id=agent_id,
                turn_number=turn_number,
                original_content=content,
                critique=f"Violations: {', '.join(violations)}",
                violations=violations,
                suggested_revision=suggested,
                timestamp=time.time(),
                accepted=False
            )
            critiques.append(critique)
            self._critique_log.append(critique)

            if attempt < self.config.max_validation_retries:
                # Regenerate with corrective feedback
                retries += 1
                corrective_prompt = self._build_corrective_prompt(
                    system_prompt, violations, suggested
                )
                content = self.generate(corrective_prompt, user_message, mode="performer")

        duration = time.time() - start_time

        return GenerationResult(
            content=content,
            agent_id=agent_id,
            mode="performer",
            temperature=self.config.performer_temperature,
            validation_passed=len(critiques) == 0 or critiques[-1].accepted if critiques else True,
            retries=retries,
            coach_critiques=critiques,
            duration_seconds=duration
        )

    def _validate_with_coach(
        self,
        content: str,
        agent_id: str,
        rules: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str], Optional[str]]:
        """
        Validate content using the coach.

        Returns:
            Tuple of (is_valid, violations, suggested_revision)
        """
        validation_prompt = self._build_validation_prompt(agent_id, rules, context)
        validation_request = f"""
Validate this agent response:

Agent: {agent_id}
Content: {content}

Check for rule violations and respond in this format:
VALID: [yes/no]
VIOLATIONS: [comma-separated list or "none"]
SUGGESTION: [brief suggestion for improvement or "none"]
"""

        response = self.generate(validation_prompt, validation_request, mode="coach")

        # Parse validation response
        is_valid = "VALID: yes" in response.lower() or "valid: yes" in response.lower()
        violations = []
        suggested = None

        if "VIOLATIONS:" in response:
            viol_line = response.split("VIOLATIONS:")[1].split("\n")[0].strip()
            if viol_line.lower() != "none":
                violations = [v.strip() for v in viol_line.split(",")]

        if "SUGGESTION:" in response:
            sugg_line = response.split("SUGGESTION:")[1].split("\n")[0].strip()
            if sugg_line.lower() != "none":
                suggested = sugg_line

        return is_valid, violations, suggested

    def _build_validation_prompt(
        self,
        agent_id: str,
        rules: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the coach's validation system prompt."""
        rules_text = "\n".join(f"- {rule}" for rule in rules)
        context_text = ""
        if context:
            context_text = f"\nContext: {context}"

        return f"""You are a validation coach for a social simulation.
Your role is to check agent outputs for rule violations.

Agent being validated: {agent_id}

Rules to enforce:
{rules_text}
{context_text}

Be strict but fair. Only flag clear violations, not stylistic preferences.
Respond concisely in the specified format."""

    def _build_corrective_prompt(
        self,
        original_prompt: str,
        violations: List[str],
        suggested: Optional[str]
    ) -> str:
        """Build a corrective prompt incorporating coach feedback."""
        violation_text = ", ".join(violations)
        correction = f"""

IMPORTANT: Your previous response had these issues: {violation_text}
"""
        if suggested:
            correction += f"Suggestion: {suggested}\n"
        correction += "Please regenerate while avoiding these issues."

        return original_prompt + correction

    def get_coach_critique_for_message(
        self,
        content: str,
        agent_id: str,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate a coach critique for logging (not for regeneration).

        This is used to log internal coach observations that can be
        analyzed later, even when content is accepted.
        """
        if not self.config.log_coach_critiques:
            return None

        critique_prompt = """You are an analytical observer of a social simulation.
Provide a brief internal note about this agent's utterance.
Focus on:
- Theoretical alignment (does it embody the expected concepts?)
- Social dynamics (how does it position the agent?)
- Potential areas for development

Keep your observation to 1-2 sentences."""

        request = f"""
Agent: {agent_id}
Round: {context.get('round_number', '?')}
Turn: {context.get('turn_number', '?')}
Content: {content}

Provide your internal observation:"""

        return self.generate(critique_prompt, request, mode="coach")

    @property
    def critique_log(self) -> List[CoachCritique]:
        """Get the log of all coach critiques."""
        return self._critique_log

    def clear_critique_log(self) -> None:
        """Clear the critique log."""
        self._critique_log = []


# =============================================================================
# Factory Functions
# =============================================================================

def create_dual_llm_client(
    performer_client: LLMClientProtocol,
    coach_client: Optional[LLMClientProtocol] = None,
    performer_temp: float = 0.7,
    coach_temp: float = 0.1,
    log_critiques: bool = True
) -> DualLLMClient:
    """
    Create a DualLLMClient with the specified configuration.

    Args:
        performer_client: LLM client for generation
        coach_client: LLM client for validation (defaults to performer)
        performer_temp: Temperature for performer (default 0.7)
        coach_temp: Temperature for coach (default 0.1)
        log_critiques: Whether to log coach critiques

    Returns:
        Configured DualLLMClient
    """
    config = DualLLMConfig(
        performer_temperature=performer_temp,
        coach_temperature=coach_temp,
        log_coach_critiques=log_critiques
    )
    return DualLLMClient(performer_client, coach_client, config)


def create_dual_llm_from_single(
    client: LLMClientProtocol,
    performer_temp: float = 0.7,
    coach_temp: float = 0.1
) -> DualLLMClient:
    """
    Create a DualLLMClient using a single client with different temperatures.

    This is the simpler setup where both roles use the same model but
    with different temperature settings.
    """
    return create_dual_llm_client(
        performer_client=client,
        coach_client=client,
        performer_temp=performer_temp,
        coach_temp=coach_temp
    )


def create_true_dual_llm(
    performer_base_url: str,
    performer_model: str,
    coach_base_url: str,
    coach_model: str,
    performer_temp: float = 0.7,
    coach_temp: float = 0.1,
    api_key: str = "not-needed",
    timeout: float = 180.0
) -> DualLLMClient:
    """
    Create a TRUE dual-LLM client with two separate endpoints/models.

    This is the research-grade setup where Coach and Performer are
    genuinely different models running on separate GPUs.

    Args:
        performer_base_url: vLLM endpoint for Performer (e.g., A100 with 14B)
        performer_model: Model name for Performer
        coach_base_url: vLLM endpoint for Coach (e.g., A40 with 7B)
        coach_model: Model name for Coach
        performer_temp: Temperature for Performer (default 0.7)
        coach_temp: Temperature for Coach (default 0.1)
        api_key: API key if required
        timeout: Request timeout in seconds

    Returns:
        DualLLMClient with two separate model backends

    Example:
        dual = create_true_dual_llm(
            performer_base_url="https://a100-pod-8000.proxy.runpod.net/v1",
            performer_model="Qwen/Qwen2.5-14B-Instruct",
            coach_base_url="https://a40-pod-8000.proxy.runpod.net/v1",
            coach_model="Qwen/Qwen2.5-7B-Instruct"
        )
    """
    # Import here to avoid circular deps
    from openai import OpenAI

    class VLLMClient:
        """Lightweight vLLM client wrapper."""
        def __init__(self, base_url: str, model: str, api_key: str, timeout: float):
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                default_headers={"ngrok-skip-browser-warning": "true"}  # Fix 403 errors with ngrok tunnels
            )
            self.model = model

        def send_message(
            self,
            system_prompt: str,
            user_message: str,
            temperature: float = 0.7,
            max_tokens: int = 512
        ) -> str:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

    # Create separate clients
    performer_client = VLLMClient(performer_base_url, performer_model, api_key, timeout)
    coach_client = VLLMClient(coach_base_url, coach_model, api_key, timeout)

    config = DualLLMConfig(
        performer_temperature=performer_temp,
        coach_temperature=coach_temp,
        log_coach_critiques=True
    )

    return DualLLMClient(performer_client, coach_client, config)
