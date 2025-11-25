"""
Semiotic Coding Module for Social RL

Implements the empirical semiotics methodology from Social Aesthetics:
- Architectural parameters (temperatures, roles, constraints) = independent variables
- Semiotic patterns in discourse = dependent variables

Categories are grounded in theory:
- Justification vs Assertion → non-domination / arbitrary power
- Voice / Recognition markers → alienation, inclusion
- Relational stance → bridging vs dismissive

Usage:
    from social_rl.semiotic_coder import SemioticCoder

    coder = SemioticCoder(llm_client)
    coded = coder.code_transcript(round_data)
    summary = coder.compute_semiotic_summary(coded)
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import re


# =============================================================================
# CODEBOOK: Semiotic Categories
# =============================================================================

class JustificationType(Enum):
    """Justification vs Assertion coding.

    Connects to non-domination: justified speech explains reasons,
    while assertion exercises arbitrary discursive power.
    """
    JUSTIFICATORY = "justificatory"  # Explains why: "because", "so that", "this matters for"
    ASSERTIVE = "assertive"          # Flat claim: "we must", "it is necessary", "obviously"
    NEUTRAL = "neutral"              # Neither strongly justified nor asserted


class VoiceMarker(Enum):
    """Voice and recognition markers.

    Captures alienation (feeling unheard, excluded) vs
    recognition (being acknowledged, included in "we").
    """
    ALIENATED = "alienated"          # "my vote doesn't matter", "no one listens", "people like me never..."
    EMPOWERED = "empowered"          # "we can change this", "our voice matters", "I believe we..."
    CONDITIONAL = "conditional"      # "I would vote if...", "maybe if things were different..."
    NEUTRAL = "neutral"              # No strong voice markers


class RelationalStance(Enum):
    """Relational stance markers.

    Tracks how agents position themselves toward others:
    bridging (recognition, engagement) vs dismissive (rejection).
    """
    BRIDGING = "bridging"            # "I hear what you're saying", "that's a fair point", "building on that..."
    DISMISSIVE = "dismissive"        # "you're wrong", "that doesn't make sense", "you don't understand"
    DIRECT_ADDRESS = "direct"        # Second-person engagement: "you", "your perspective"
    IMPERSONAL = "impersonal"        # Third-person distance: "one", "people", "they"
    NEUTRAL = "neutral"


@dataclass
class SemioticCode:
    """Coding for a single utterance/turn."""
    agent_id: str
    turn_number: int
    round_number: int
    content: str

    # Category codes
    justification: JustificationType = JustificationType.NEUTRAL
    voice: VoiceMarker = VoiceMarker.NEUTRAL
    stance: RelationalStance = RelationalStance.NEUTRAL

    # Extracted markers (for transparency)
    justification_markers: List[str] = field(default_factory=list)
    voice_markers: List[str] = field(default_factory=list)
    stance_markers: List[str] = field(default_factory=list)

    # Confidence (if using LLM coder)
    confidence: float = 1.0
    coder_notes: str = ""


@dataclass
class SemioticSummary:
    """Per-agent semiotic summary across a round or experiment."""
    agent_id: str
    total_turns: int

    # Justification ratios
    justification_ratio: float = 0.0  # % justificatory
    assertion_ratio: float = 0.0      # % assertive

    # Voice distribution
    alienation_count: int = 0
    empowerment_count: int = 0
    conditional_count: int = 0

    # Stance patterns
    bridging_count: int = 0
    dismissive_count: int = 0
    direct_address_count: int = 0

    # Derived metrics
    voice_valence: float = 0.0        # (empowered - alienated) / total
    stance_valence: float = 0.0       # (bridging - dismissive) / total


# =============================================================================
# LEXICON-BASED CODING (fast, transparent)
# =============================================================================

JUSTIFICATION_MARKERS = {
    "justificatory": [
        "because", "since", "so that", "in order to", "this matters",
        "the reason is", "which is why", "given that", "considering",
        "it follows that", "therefore", "as a result", "due to",
        "the importance of", "this connects to", "reflecting on"
    ],
    "assertive": [
        "we must", "you have to", "it is necessary", "obviously",
        "clearly", "simply put", "the fact is", "there's no question",
        "undeniably", "without doubt", "absolutely", "certainly",
        "everyone knows", "it's obvious that"
    ]
}

VOICE_MARKERS = {
    "alienated": [
        "my vote doesn't matter", "no one listens", "people like me",
        "never heard", "doesn't represent", "left out", "ignored",
        "nothing changes", "what's the point", "won't make a difference",
        "they don't care", "not for people like", "excluded from",
        "doesn't speak to", "feel disconnected", "feel like outsider"
    ],
    "empowered": [
        "we can change", "our voice matters", "i believe we",
        "together we", "make a difference", "have the power",
        "our community", "we deserve", "stand together",
        "fight for", "demand", "our right to", "collective action"
    ],
    "conditional": [
        "i would vote if", "maybe if", "if things were different",
        "i might consider", "under certain conditions", "depending on",
        "if they actually", "when they start", "once i see"
    ]
}

STANCE_MARKERS = {
    "bridging": [
        "i hear what you", "that's a fair point", "building on",
        "i understand your", "you raise a good", "i appreciate",
        "thank you for sharing", "that resonates", "i can see",
        "valid concern", "makes sense", "agree with", "good point"
    ],
    "dismissive": [
        "you're wrong", "doesn't make sense", "you don't understand",
        "that's naive", "you're missing", "completely wrong",
        "doesn't work", "unrealistic", "you're overreacting",
        "that's ridiculous", "not how it works"
    ],
    "direct": [
        "you said", "your perspective", "you mentioned", "you all",
        "your point", "you believe", "your experience", "you're"
    ]
}


def lexicon_code_utterance(content: str, agent_id: str, turn: int, round_num: int) -> SemioticCode:
    """Code an utterance using lexicon matching.

    Fast, transparent, reproducible. Use as baseline or
    supplement to LLM coding.
    """
    content_lower = content.lower()
    code = SemioticCode(
        agent_id=agent_id,
        turn_number=turn,
        round_number=round_num,
        content=content
    )

    # Justification coding
    just_markers = []
    assert_markers = []
    for marker in JUSTIFICATION_MARKERS["justificatory"]:
        if marker in content_lower:
            just_markers.append(marker)
    for marker in JUSTIFICATION_MARKERS["assertive"]:
        if marker in content_lower:
            assert_markers.append(marker)

    code.justification_markers = just_markers + assert_markers
    if len(just_markers) > len(assert_markers):
        code.justification = JustificationType.JUSTIFICATORY
    elif len(assert_markers) > len(just_markers):
        code.justification = JustificationType.ASSERTIVE
    else:
        code.justification = JustificationType.NEUTRAL

    # Voice coding
    alien_markers = []
    empower_markers = []
    cond_markers = []
    for marker in VOICE_MARKERS["alienated"]:
        if marker in content_lower:
            alien_markers.append(marker)
    for marker in VOICE_MARKERS["empowered"]:
        if marker in content_lower:
            empower_markers.append(marker)
    for marker in VOICE_MARKERS["conditional"]:
        if marker in content_lower:
            cond_markers.append(marker)

    code.voice_markers = alien_markers + empower_markers + cond_markers
    if len(alien_markers) > max(len(empower_markers), len(cond_markers)):
        code.voice = VoiceMarker.ALIENATED
    elif len(empower_markers) > max(len(alien_markers), len(cond_markers)):
        code.voice = VoiceMarker.EMPOWERED
    elif len(cond_markers) > 0:
        code.voice = VoiceMarker.CONDITIONAL
    else:
        code.voice = VoiceMarker.NEUTRAL

    # Stance coding
    bridge_markers = []
    dismiss_markers = []
    direct_markers = []
    for marker in STANCE_MARKERS["bridging"]:
        if marker in content_lower:
            bridge_markers.append(marker)
    for marker in STANCE_MARKERS["dismissive"]:
        if marker in content_lower:
            dismiss_markers.append(marker)
    for marker in STANCE_MARKERS["direct"]:
        if marker in content_lower:
            direct_markers.append(marker)

    code.stance_markers = bridge_markers + dismiss_markers + direct_markers
    if len(dismiss_markers) > len(bridge_markers):
        code.stance = RelationalStance.DISMISSIVE
    elif len(bridge_markers) > len(dismiss_markers):
        code.stance = RelationalStance.BRIDGING
    elif len(direct_markers) > 0:
        code.stance = RelationalStance.DIRECT_ADDRESS
    else:
        code.stance = RelationalStance.NEUTRAL

    code.coder_notes = "lexicon-based"
    return code


# =============================================================================
# LLM-BASED CODING (richer, requires client)
# =============================================================================

LLM_CODING_PROMPT = """You are a semiotic coder for social discourse analysis.

Code this utterance on three dimensions:

1. JUSTIFICATION TYPE
   - justificatory: Explains reasons ("because", "so that", "this matters for...")
   - assertive: Flat claims without reasoning ("we must", "obviously", "it is necessary")
   - neutral: Neither strongly justified nor asserted

2. VOICE MARKERS
   - alienated: Expresses powerlessness, exclusion ("my vote doesn't matter", "no one listens")
   - empowered: Expresses agency, collective power ("we can change this", "our voice matters")
   - conditional: Contingent engagement ("I would vote if...", "maybe if things were different")
   - neutral: No strong voice markers

3. RELATIONAL STANCE
   - bridging: Acknowledges others positively ("I hear you", "that's a fair point")
   - dismissive: Rejects or diminishes others ("you're wrong", "doesn't make sense")
   - direct_address: Uses second-person engagement ("you", "your perspective")
   - impersonal: Third-person distance ("one", "people", "they")
   - neutral: No clear stance

Respond in this exact JSON format:
{
  "justification": "justificatory|assertive|neutral",
  "voice": "alienated|empowered|conditional|neutral",
  "stance": "bridging|dismissive|direct_address|impersonal|neutral",
  "justification_markers": ["extracted phrases..."],
  "voice_markers": ["extracted phrases..."],
  "stance_markers": ["extracted phrases..."],
  "confidence": 0.0-1.0,
  "notes": "brief reasoning"
}

UTTERANCE TO CODE:
Agent: {agent_id}
Content: {content}
"""


class SemioticCoder:
    """Semiotic coder using lexicon and/or LLM."""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        use_llm: bool = False,
        llm_temperature: float = 0.1
    ):
        """
        Args:
            llm_client: Optional LLM client for richer coding
            use_llm: If True, use LLM coding (falls back to lexicon if no client)
            llm_temperature: Temperature for LLM coder (low for consistency)
        """
        self.llm_client = llm_client
        self.use_llm = use_llm and llm_client is not None
        self.llm_temperature = llm_temperature

    def code_utterance(
        self,
        content: str,
        agent_id: str,
        turn: int,
        round_num: int
    ) -> SemioticCode:
        """Code a single utterance."""
        if self.use_llm:
            return self._llm_code(content, agent_id, turn, round_num)
        else:
            return lexicon_code_utterance(content, agent_id, turn, round_num)

    def _llm_code(
        self,
        content: str,
        agent_id: str,
        turn: int,
        round_num: int
    ) -> SemioticCode:
        """Code using LLM."""
        prompt = LLM_CODING_PROMPT.format(agent_id=agent_id, content=content)

        try:
            response = self.llm_client.send_message(
                system_prompt="You are a precise semiotic coder. Respond only with valid JSON.",
                user_message=prompt,
                temperature=self.llm_temperature,
                max_tokens=512
            )

            # Parse JSON response
            # Handle potential markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]

            data = json.loads(response)

            code = SemioticCode(
                agent_id=agent_id,
                turn_number=turn,
                round_number=round_num,
                content=content,
                justification=JustificationType(data.get("justification", "neutral")),
                voice=VoiceMarker(data.get("voice", "neutral")),
                stance=RelationalStance(data.get("stance", "neutral")),
                justification_markers=data.get("justification_markers", []),
                voice_markers=data.get("voice_markers", []),
                stance_markers=data.get("stance_markers", []),
                confidence=data.get("confidence", 0.8),
                coder_notes=f"llm-coded: {data.get('notes', '')}"
            )
            return code

        except Exception as e:
            # Fall back to lexicon coding
            code = lexicon_code_utterance(content, agent_id, turn, round_num)
            code.coder_notes = f"llm-fallback (error: {str(e)[:50]})"
            return code

    def code_transcript(self, round_data: Dict) -> List[SemioticCode]:
        """Code all messages in a round transcript.

        Args:
            round_data: Round data from social_rl.json files

        Returns:
            List of SemioticCode objects
        """
        codes = []
        round_num = round_data.get("round_number", 1)
        messages = round_data.get("messages", [])

        for msg in messages:
            code = self.code_utterance(
                content=msg.get("content", ""),
                agent_id=msg.get("agent_id", "unknown"),
                turn=msg.get("turn_number", 0),
                round_num=round_num
            )
            codes.append(code)

        return codes

    def compute_semiotic_summary(
        self,
        codes: List[SemioticCode],
        agent_id: Optional[str] = None
    ) -> Dict[str, SemioticSummary]:
        """Compute per-agent semiotic summaries.

        Args:
            codes: List of coded utterances
            agent_id: If provided, compute only for this agent

        Returns:
            Dict mapping agent_id to SemioticSummary
        """
        # Group by agent
        by_agent: Dict[str, List[SemioticCode]] = {}
        for code in codes:
            if agent_id and code.agent_id != agent_id:
                continue
            if code.agent_id not in by_agent:
                by_agent[code.agent_id] = []
            by_agent[code.agent_id].append(code)

        summaries = {}
        for aid, agent_codes in by_agent.items():
            total = len(agent_codes)
            if total == 0:
                continue

            # Count categories
            just_count = sum(1 for c in agent_codes if c.justification == JustificationType.JUSTIFICATORY)
            assert_count = sum(1 for c in agent_codes if c.justification == JustificationType.ASSERTIVE)

            alien_count = sum(1 for c in agent_codes if c.voice == VoiceMarker.ALIENATED)
            empower_count = sum(1 for c in agent_codes if c.voice == VoiceMarker.EMPOWERED)
            cond_count = sum(1 for c in agent_codes if c.voice == VoiceMarker.CONDITIONAL)

            bridge_count = sum(1 for c in agent_codes if c.stance == RelationalStance.BRIDGING)
            dismiss_count = sum(1 for c in agent_codes if c.stance == RelationalStance.DISMISSIVE)
            direct_count = sum(1 for c in agent_codes if c.stance == RelationalStance.DIRECT_ADDRESS)

            summary = SemioticSummary(
                agent_id=aid,
                total_turns=total,
                justification_ratio=just_count / total,
                assertion_ratio=assert_count / total,
                alienation_count=alien_count,
                empowerment_count=empower_count,
                conditional_count=cond_count,
                bridging_count=bridge_count,
                dismissive_count=dismiss_count,
                direct_address_count=direct_count,
                voice_valence=(empower_count - alien_count) / total if total > 0 else 0,
                stance_valence=(bridge_count - dismiss_count) / total if total > 0 else 0
            )
            summaries[aid] = summary

        return summaries

    def to_dataframe_rows(self, codes: List[SemioticCode]) -> List[Dict]:
        """Convert coded utterances to rows suitable for pandas DataFrame."""
        rows = []
        for code in codes:
            rows.append({
                "agent_id": code.agent_id,
                "turn_number": code.turn_number,
                "round_number": code.round_number,
                "content_preview": code.content[:100] + "..." if len(code.content) > 100 else code.content,
                "justification": code.justification.value,
                "voice": code.voice.value,
                "stance": code.stance.value,
                "justification_markers": "|".join(code.justification_markers),
                "voice_markers": "|".join(code.voice_markers),
                "stance_markers": "|".join(code.stance_markers),
                "confidence": code.confidence,
                "coder_notes": code.coder_notes
            })
        return rows


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compare_conditions(
    condition_a_codes: List[SemioticCode],
    condition_b_codes: List[SemioticCode],
    condition_a_name: str = "A",
    condition_b_name: str = "B"
) -> Dict:
    """Compare semiotic patterns between two experimental conditions.

    Returns summary statistics for A/B comparison.
    """
    coder = SemioticCoder()

    summary_a = coder.compute_semiotic_summary(condition_a_codes)
    summary_b = coder.compute_semiotic_summary(condition_b_codes)

    comparison = {
        "condition_a": condition_a_name,
        "condition_b": condition_b_name,
        "by_agent": {}
    }

    # Get all agents
    all_agents = set(summary_a.keys()) | set(summary_b.keys())

    for agent in all_agents:
        a = summary_a.get(agent)
        b = summary_b.get(agent)

        comparison["by_agent"][agent] = {
            f"{condition_a_name}_justification_ratio": a.justification_ratio if a else None,
            f"{condition_b_name}_justification_ratio": b.justification_ratio if b else None,
            "justification_delta": (b.justification_ratio - a.justification_ratio) if (a and b) else None,

            f"{condition_a_name}_voice_valence": a.voice_valence if a else None,
            f"{condition_b_name}_voice_valence": b.voice_valence if b else None,
            "voice_delta": (b.voice_valence - a.voice_valence) if (a and b) else None,

            f"{condition_a_name}_stance_valence": a.stance_valence if a else None,
            f"{condition_b_name}_stance_valence": b.stance_valence if b else None,
            "stance_delta": (b.stance_valence - a.stance_valence) if (a and b) else None,
        }

    return comparison


def load_and_code_experiment(experiment_dir: str, use_llm: bool = False, llm_client: Any = None) -> Dict:
    """Load an experiment directory and code all rounds.

    Args:
        experiment_dir: Path to experiment output directory
        use_llm: Whether to use LLM coding
        llm_client: LLM client if using LLM coding

    Returns:
        Dict with coded rounds and summaries
    """
    from pathlib import Path

    exp_path = Path(experiment_dir)
    coder = SemioticCoder(llm_client=llm_client, use_llm=use_llm)

    results = {
        "experiment_dir": str(exp_path),
        "rounds": {},
        "all_codes": [],
        "overall_summary": {}
    }

    # Find all round files
    round_files = sorted(exp_path.glob("round*_social_rl.json"))

    for rf in round_files:
        with open(rf) as f:
            round_data = json.load(f)

        round_num = round_data.get("round_number", int(rf.stem.split("_")[0].replace("round", "")))
        codes = coder.code_transcript(round_data)
        summary = coder.compute_semiotic_summary(codes)

        results["rounds"][round_num] = {
            "codes": [c.__dict__ for c in codes],
            "summary": {k: v.__dict__ for k, v in summary.items()}
        }
        results["all_codes"].extend(codes)

    # Overall summary
    results["overall_summary"] = {
        k: v.__dict__ for k, v in coder.compute_semiotic_summary(results["all_codes"]).items()
    }

    return results
