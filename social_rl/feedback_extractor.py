"""
SocialFeedbackExtractor - Extract learning signals from social interactions.

This module implements "Social Feedback as Reward": instead of external reward
signals, agents learn from social dynamics:
- Engagement: Did others respond to/reference this agent?
- Theoretical Alignment: Did behavior align with framework concepts?
- Contribution Value: Did the synthesis include this agent's contributions?

This is the "reward function" for Social RL, but it emerges from
interaction rather than being externally defined.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict
import re
import json


@dataclass
class SocialFeedback:
    """Feedback signals extracted from social interaction."""
    agent_id: str
    round_number: int

    # Core signals (0.0-1.0 scale)
    engagement: float = 0.5          # How much others engaged with this agent
    theoretical_alignment: float = 0.5  # Alignment with theoretical framework
    contribution_value: float = 0.5   # Quality/impact of contributions

    # Detailed metrics
    direct_references: int = 0       # Times name was mentioned
    response_received: int = 0       # Times others responded to this agent
    concepts_embodied: List[str] = field(default_factory=list)  # Which concepts were demonstrated
    analyst_mentions: int = 0        # Times analyst coded this agent's behavior
    synthesis_inclusion: float = 0.0  # Proportion of synthesis referencing this agent

    # Social dynamics
    initiated_exchanges: int = 0     # Times this agent started new topics
    questions_asked: int = 0         # Questions this agent posed
    questions_answered: int = 0      # Questions this agent answered
    directives_given: int = 0        # Commands/instructions issued
    compliance_shown: int = 0        # Times this agent complied

    def to_dict(self) -> Dict[str, Any]:
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
            "synthesis_inclusion": self.synthesis_inclusion
        }

    def as_reward_signal(self) -> Dict[str, float]:
        """Convert to simple reward signal dict."""
        return {
            "engagement": self.engagement,
            "theoretical_alignment": self.theoretical_alignment,
            "contribution_value": self.contribution_value
        }


@dataclass
class ConceptMarkers:
    """Markers for detecting theoretical concepts in text."""
    concept_a_name: str
    concept_a_markers: List[str]  # Words/phrases indicating concept A
    concept_b_name: str
    concept_b_markers: List[str]  # Words/phrases indicating concept B

    @classmethod
    def for_option_a(cls) -> "ConceptMarkers":
        """Markers for Class Conflict / Alienation framework."""
        return cls(
            concept_a_name="Alienation",
            concept_a_markers=[
                "disconnect", "separate", "meaningless", "mechanical", "routine",
                "why am i", "what's the point", "just following orders", "don't understand",
                "no control", "no say", "not my decision", "told to do",
                "estranged", "alienat", "isolated", "powerless"
            ],
            concept_b_name="Non-domination",
            concept_b_markers=[
                "arbitrary", "domination", "control", "power", "authority",
                "because i said so", "don't question", "must comply", "no choice",
                "forced", "coerced", "subject to", "at the mercy",
                "can't refuse", "have to", "no option"
            ]
        )

    @classmethod
    def for_option_b(cls) -> "ConceptMarkers":
        """Markers for Democratic Participation framework."""
        return cls(
            concept_a_name="Civic Virtue",
            concept_a_markers=[
                "community", "collective", "together", "public good", "civic",
                "responsibility", "duty", "contribute", "participate",
                "common interest", "shared", "we all"
            ],
            concept_b_name="Self-Interest",
            concept_b_markers=[
                "benefit", "profit", "gain", "my interest", "personal",
                "what's in it for me", "rational", "maximize", "optimize",
                "individual", "my needs", "self"
            ]
        )


class SocialFeedbackExtractor:
    """
    Extracts social feedback signals from conversation transcripts.

    The core insight: social interaction contains implicit reward signals:
    - Being referenced = social validation
    - Having points addressed = engagement
    - Embodying framework concepts = theoretical alignment
    - Inclusion in synthesis = contribution value

    This allows RL-like learning without explicit reward functions.
    """

    def __init__(
        self,
        concept_markers: ConceptMarkers,
        analyst_agent_id: Optional[str] = None
    ):
        """
        Initialize the extractor.

        Args:
            concept_markers: Markers for detecting theoretical concepts
            analyst_agent_id: Agent ID for the analyst (if any)
        """
        self.markers = concept_markers
        self.analyst_id = analyst_agent_id or "Analyst+Reporter"

        # Cumulative feedback storage
        self.round_feedback: Dict[int, Dict[str, SocialFeedback]] = {}

    def extract_round_feedback(
        self,
        round_number: int,
        messages: List[Dict[str, Any]],
        participants: List[str],
        synthesis: Optional[str] = None
    ) -> Dict[str, SocialFeedback]:
        """
        Extract feedback for all participants from a round transcript.

        Args:
            round_number: Round number
            messages: List of message dicts with agent_id and content
            participants: List of participant agent IDs
            synthesis: Optional final synthesis text

        Returns:
            Dict mapping agent_id to SocialFeedback
        """
        feedback = {
            pid: SocialFeedback(agent_id=pid, round_number=round_number)
            for pid in participants
        }

        # Build name mapping (Worker+Alice -> Alice, alice)
        name_map = self._build_name_map(participants)

        # Process each message
        for i, msg in enumerate(messages):
            speaker_id = msg.get("agent_id", "")
            content = msg.get("content", "")

            if speaker_id not in feedback:
                continue

            speaker_fb = feedback[speaker_id]

            # 1. Detect concept embodiment in this message
            concepts = self._detect_concepts(content)
            speaker_fb.concepts_embodied.extend(concepts)

            # 2. Count social behaviors
            social_counts = self._count_social_behaviors(content)
            speaker_fb.questions_asked += social_counts["questions"]
            speaker_fb.directives_given += social_counts["directives"]
            speaker_fb.compliance_shown += social_counts["compliance"]

            # 3. Check if this message initiates a new topic
            if i == 0 or self._is_topic_shift(content, messages[i-1].get("content", "")):
                speaker_fb.initiated_exchanges += 1

            # 4. Check for references to other agents
            for other_id in participants:
                if other_id == speaker_id:
                    continue
                names = name_map.get(other_id, set())
                if any(name.lower() in content.lower() for name in names):
                    feedback[other_id].direct_references += 1
                    feedback[other_id].response_received += 1

        # 5. Extract analyst feedback if analyst messages exist
        analyst_messages = [m for m in messages if self.analyst_id in m.get("agent_id", "")]
        if analyst_messages:
            self._extract_analyst_feedback(analyst_messages, feedback, name_map)

        # 6. Process synthesis inclusion
        if synthesis:
            self._extract_synthesis_inclusion(synthesis, feedback, name_map)

        # 7. Calculate aggregate scores
        for agent_id, fb in feedback.items():
            fb.engagement = self._calculate_engagement(fb, len(messages))
            fb.theoretical_alignment = self._calculate_alignment(fb)
            fb.contribution_value = self._calculate_contribution(fb, len(participants))

        # Store for cumulative access
        self.round_feedback[round_number] = feedback

        return feedback

    def _build_name_map(self, participants: List[str]) -> Dict[str, Set[str]]:
        """Build mapping from agent_id to possible name references."""
        name_map = {}
        for pid in participants:
            names = {pid}  # Full identifier
            if "+" in pid:
                role, name = pid.split("+", 1)
                names.add(name)
                names.add(name.lower())
                names.add(role)
            name_map[pid] = names
        return name_map

    def _detect_concepts(self, content: str) -> List[str]:
        """Detect which theoretical concepts are embodied in this message."""
        content_lower = content.lower()
        concepts = []

        # Check for concept A markers
        a_count = sum(1 for marker in self.markers.concept_a_markers
                      if marker.lower() in content_lower)
        if a_count >= 2:
            concepts.append(self.markers.concept_a_name)

        # Check for concept B markers
        b_count = sum(1 for marker in self.markers.concept_b_markers
                      if marker.lower() in content_lower)
        if b_count >= 2:
            concepts.append(self.markers.concept_b_name)

        return concepts

    def _count_social_behaviors(self, content: str) -> Dict[str, int]:
        """Count social behaviors in a message."""
        counts = {"questions": 0, "directives": 0, "compliance": 0}

        # Questions
        counts["questions"] = content.count("?")

        # Directives (imperative patterns)
        directive_patterns = [
            r"\b(must|should|need to|have to|will)\b",
            r"\b(do this|complete|finish|start|begin)\b",
            r"\b(i want you to|you will|you must)\b"
        ]
        for pattern in directive_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                counts["directives"] += 1

        # Compliance (agreement patterns)
        compliance_patterns = [
            r"\b(yes|okay|understood|i('ll| will)|certainly|of course)\b",
            r"\b(right away|immediately|i understand)\b"
        ]
        for pattern in compliance_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                counts["compliance"] += 1

        return counts

    def _is_topic_shift(self, current: str, previous: str) -> bool:
        """Detect if current message shifts the topic."""
        # Simple heuristic: new topics often start with certain patterns
        topic_shift_patterns = [
            r"^(now|next|let's|moving on|another thing)",
            r"^(i think|in my view|actually)",
            r"^(what about|have you considered|shouldn't we)"
        ]
        for pattern in topic_shift_patterns:
            if re.search(pattern, current, re.IGNORECASE):
                return True
        return False

    def _extract_analyst_feedback(
        self,
        analyst_messages: List[Dict[str, Any]],
        feedback: Dict[str, SocialFeedback],
        name_map: Dict[str, Set[str]]
    ):
        """Extract feedback from analyst's observations."""
        analyst_text = " ".join(m.get("content", "") for m in analyst_messages)

        for agent_id, names in name_map.items():
            if agent_id == self.analyst_id:
                continue

            # Count how many times analyst mentions this agent
            mentions = sum(
                1 for name in names
                if name.lower() in analyst_text.lower()
            )
            feedback[agent_id].analyst_mentions = mentions

            # Check if analyst associates agent with concepts
            for name in names:
                name_context = self._extract_context_around(analyst_text, name)
                if name_context:
                    concepts = self._detect_concepts(name_context)
                    feedback[agent_id].concepts_embodied.extend(concepts)

    def _extract_context_around(self, text: str, term: str, window: int = 100) -> str:
        """Extract text context around a term."""
        text_lower = text.lower()
        term_lower = term.lower()

        idx = text_lower.find(term_lower)
        if idx == -1:
            return ""

        start = max(0, idx - window)
        end = min(len(text), idx + len(term) + window)
        return text[start:end]

    def _extract_synthesis_inclusion(
        self,
        synthesis: str,
        feedback: Dict[str, SocialFeedback],
        name_map: Dict[str, Set[str]]
    ):
        """Extract how much each agent is included in synthesis."""
        synthesis_lower = synthesis.lower()
        synthesis_length = len(synthesis_lower)

        for agent_id, names in name_map.items():
            # Count character positions where agent is referenced
            char_coverage = 0
            for name in names:
                for match in re.finditer(re.escape(name.lower()), synthesis_lower):
                    # Count ~50 chars around each mention as "coverage"
                    char_coverage += min(100, synthesis_length - match.start())

            # Normalize to 0-1 scale
            feedback[agent_id].synthesis_inclusion = min(1.0, char_coverage / max(synthesis_length, 1))

    def _calculate_engagement(self, fb: SocialFeedback, total_messages: int) -> float:
        """Calculate engagement score (0-1)."""
        if total_messages == 0:
            return 0.5

        # Factors: references, responses, exchanges initiated
        reference_score = min(1.0, fb.direct_references / max(total_messages * 0.3, 1))
        response_score = min(1.0, fb.response_received / max(total_messages * 0.2, 1))
        initiative_score = min(1.0, fb.initiated_exchanges / max(total_messages * 0.1, 1))

        return (reference_score * 0.4 + response_score * 0.4 + initiative_score * 0.2)

    def _calculate_alignment(self, fb: SocialFeedback) -> float:
        """Calculate theoretical alignment score (0-1)."""
        # More concepts embodied = higher alignment
        unique_concepts = len(set(fb.concepts_embodied))
        concept_score = min(1.0, unique_concepts / 2.0)  # Max 2 concepts

        # Analyst mentions boost alignment
        analyst_score = min(1.0, fb.analyst_mentions / 3.0)

        return concept_score * 0.7 + analyst_score * 0.3

    def _calculate_contribution(self, fb: SocialFeedback, num_participants: int) -> float:
        """Calculate contribution value score (0-1)."""
        # Synthesis inclusion is main factor
        synthesis_score = fb.synthesis_inclusion

        # Questions asked show engagement
        question_score = min(1.0, fb.questions_asked / 3.0)

        # Balance between directing and complying (neither extreme is ideal)
        activity_balance = 1.0 - abs(fb.directives_given - fb.compliance_shown) / max(
            fb.directives_given + fb.compliance_shown, 1
        )

        return synthesis_score * 0.5 + question_score * 0.3 + activity_balance * 0.2

    def get_cumulative_feedback(self) -> Dict[str, Dict[str, float]]:
        """Get cumulative feedback across all rounds."""
        cumulative = defaultdict(lambda: {"engagement": [], "theoretical_alignment": [], "contribution_value": []})

        for round_num, round_fb in self.round_feedback.items():
            for agent_id, fb in round_fb.items():
                cumulative[agent_id]["engagement"].append(fb.engagement)
                cumulative[agent_id]["theoretical_alignment"].append(fb.theoretical_alignment)
                cumulative[agent_id]["contribution_value"].append(fb.contribution_value)

        # Average across rounds
        result = {}
        for agent_id, signals in cumulative.items():
            result[agent_id] = {
                key: sum(values) / len(values) if values else 0.5
                for key, values in signals.items()
            }
        return result

    def compare_rounds(self, round_a: int, round_b: int) -> Dict[str, Dict[str, float]]:
        """Compare feedback between two rounds to detect learning."""
        if round_a not in self.round_feedback or round_b not in self.round_feedback:
            return {}

        fb_a = self.round_feedback[round_a]
        fb_b = self.round_feedback[round_b]

        comparison = {}
        for agent_id in set(fb_a.keys()) & set(fb_b.keys()):
            comparison[agent_id] = {
                "engagement_delta": fb_b[agent_id].engagement - fb_a[agent_id].engagement,
                "alignment_delta": fb_b[agent_id].theoretical_alignment - fb_a[agent_id].theoretical_alignment,
                "contribution_delta": fb_b[agent_id].contribution_value - fb_a[agent_id].contribution_value
            }
        return comparison

    def generate_feedback_report(self, round_number: int) -> str:
        """Generate human-readable feedback report for a round."""
        if round_number not in self.round_feedback:
            return f"No feedback data for round {round_number}"

        report = [f"=== Social Feedback Report: Round {round_number} ===\n"]

        for agent_id, fb in self.round_feedback[round_number].items():
            report.append(f"\n{agent_id}:")
            report.append(f"  Engagement: {fb.engagement:.2f}")
            report.append(f"  Theoretical Alignment: {fb.theoretical_alignment:.2f}")
            report.append(f"  Contribution Value: {fb.contribution_value:.2f}")
            report.append(f"  Direct References: {fb.direct_references}")
            report.append(f"  Concepts Embodied: {list(set(fb.concepts_embodied))}")
            if fb.analyst_mentions > 0:
                report.append(f"  Analyst Mentions: {fb.analyst_mentions}")

        return "\n".join(report)


# Convenience function
def create_extractor_for_framework(framework_option: str, analyst_id: str = None) -> SocialFeedbackExtractor:
    """Create extractor for a specific framework option."""
    markers_map = {
        "A": ConceptMarkers.for_option_a,
        "B": ConceptMarkers.for_option_b,
    }

    markers_fn = markers_map.get(framework_option.upper(), ConceptMarkers.for_option_a)
    return SocialFeedbackExtractor(markers_fn(), analyst_id)


if __name__ == "__main__":
    # Test the feedback extractor
    print("=== SocialFeedbackExtractor Test ===\n")

    extractor = SocialFeedbackExtractor(ConceptMarkers.for_option_a())

    # Test messages
    test_messages = [
        {"agent_id": "Owner+Marta", "content": "Alice, you will work on the assembly line today. No questions."},
        {"agent_id": "Worker+Alice", "content": "I understand, Marta. I'll head there now. Though I don't really understand why we changed the process."},
        {"agent_id": "Worker+Ben", "content": "Alice, did you hear about the new quotas? They've increased again."},
        {"agent_id": "Worker+Alice", "content": "Yes, Ben. It feels meaningless - we just follow orders without any say in how things work."},
        {"agent_id": "Owner+Marta", "content": "Less talking, more working. Ben, you must complete your station first."},
        {"agent_id": "Worker+Ben", "content": "Yes, okay. I'll get to it right away."},
    ]

    feedback = extractor.extract_round_feedback(
        round_number=1,
        messages=test_messages,
        participants=["Owner+Marta", "Worker+Alice", "Worker+Ben"]
    )

    print(extractor.generate_feedback_report(1))
