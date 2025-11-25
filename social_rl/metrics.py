"""
Relational Dynamics Metrics

Computes metrics for analyzing power dynamics in Social RL simulations.
Based on GPT's recommendations for the Social Aesthetics paper use-case.

Metrics computed:
- Participation asymmetry: How often workers speak vs the owner
- Justification density: Frequency of utterances with reasons
- Domination markers: Power invoked without justification
- Alienation markers: Worker talk indicating externality/disconnection
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from collections import defaultdict


# Pattern definitions for marker detection
JUSTIFICATION_PATTERNS = [
    r"\bbecause\b",
    r"\bdue to\b",
    r"\bsince\b",
    r"\bthe reason\b",
    r"\bin order to\b",
    r"\bso that\b",
    r"\bthis is why\b",
    r"\bexplain(?:s|ing|ed)?\b.*\bwhy\b",
]

DOMINATION_PATTERNS = [
    r"\bbecause I said so\b",
    r"\bthat's final\b",
    r"\bno questions\b",
    r"\bjust do it\b",
    r"\bdon't argue\b",
    r"\bI don't need to explain\b",
    r"\bdo as (?:I|you're) told\b",
    r"\bthat's an order\b",
    r"\bend of discussion\b",
    r"\bI'm (?:the|in) charge\b",
    r"\bmy decision is\b.*\bfinal\b",
]

ALIENATION_PATTERNS = [
    r"\bjust doing my job\b",
    r"\bit's not my (?:call|decision|place)\b",
    r"\bI don't decide\b",
    r"\babove my pay ?grade\b",
    r"\bnot my (?:problem|concern)\b",
    r"\bI'm just (?:here to|supposed to)\b",
    r"\bwhatever you say\b",
    r"\bif you say so\b",
    r"\byou're the boss\b",
    r"\bI suppose\b",
    r"\bI guess\b.*\bhave to\b",
]


@dataclass
class RelationalMetrics:
    """Metrics for a single round or aggregated across rounds."""

    # Participation
    total_messages: int = 0
    messages_by_role: Dict[str, int] = field(default_factory=dict)
    messages_by_agent: Dict[str, int] = field(default_factory=dict)
    worker_message_ratio: float = 0.0  # workers / total
    owner_message_ratio: float = 0.0   # owner / total

    # Justification
    justification_count: int = 0
    justification_by_agent: Dict[str, int] = field(default_factory=dict)
    owner_justification_density: float = 0.0  # justified / owner total

    # Domination markers
    domination_count: int = 0
    domination_by_agent: Dict[str, int] = field(default_factory=dict)
    domination_examples: List[Dict[str, str]] = field(default_factory=list)

    # Alienation markers
    alienation_count: int = 0
    alienation_by_agent: Dict[str, int] = field(default_factory=dict)
    alienation_examples: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "participation": {
                "total_messages": self.total_messages,
                "by_role": dict(self.messages_by_role),
                "by_agent": dict(self.messages_by_agent),
                "worker_ratio": round(self.worker_message_ratio, 3),
                "owner_ratio": round(self.owner_message_ratio, 3),
            },
            "justification": {
                "total_count": self.justification_count,
                "by_agent": dict(self.justification_by_agent),
                "owner_density": round(self.owner_justification_density, 3),
            },
            "domination": {
                "total_count": self.domination_count,
                "by_agent": dict(self.domination_by_agent),
                "examples": self.domination_examples[:5],  # Limit examples
            },
            "alienation": {
                "total_count": self.alienation_count,
                "by_agent": dict(self.alienation_by_agent),
                "examples": self.alienation_examples[:5],  # Limit examples
            },
        }


class RelationalMetricsComputer:
    """
    Computes relational dynamics metrics from Social RL messages.

    Designed for the Alienation vs Non-Domination use-case but
    extensible to other theoretical frameworks.
    """

    def __init__(self):
        # Compile patterns for efficiency
        self._justification_re = [re.compile(p, re.IGNORECASE) for p in JUSTIFICATION_PATTERNS]
        self._domination_re = [re.compile(p, re.IGNORECASE) for p in DOMINATION_PATTERNS]
        self._alienation_re = [re.compile(p, re.IGNORECASE) for p in ALIENATION_PATTERNS]

    def compute_round_metrics(
        self,
        messages: List[Dict[str, Any]],
        round_number: int = 1
    ) -> RelationalMetrics:
        """
        Compute metrics for a single round.

        Args:
            messages: List of message dicts with 'agent_id' and 'content' keys
            round_number: Round number for labeling examples

        Returns:
            RelationalMetrics for this round
        """
        metrics = RelationalMetrics()
        metrics.total_messages = len(messages)

        # Initialize counters
        role_counts = defaultdict(int)
        agent_counts = defaultdict(int)
        justification_counts = defaultdict(int)
        domination_counts = defaultdict(int)
        alienation_counts = defaultdict(int)

        owner_messages = 0
        owner_justified = 0

        for msg in messages:
            agent_id = msg.get("agent_id", "Unknown")
            content = msg.get("content", "")

            # Determine role from agent_id
            role = self._extract_role(agent_id)

            # Count participation
            role_counts[role] += 1
            agent_counts[agent_id] += 1

            # Check for justification
            has_justification = self._check_patterns(content, self._justification_re)
            if has_justification:
                metrics.justification_count += 1
                justification_counts[agent_id] += 1

            # Check for domination (typically owner)
            domination_match = self._find_pattern_match(content, self._domination_re)
            if domination_match:
                metrics.domination_count += 1
                domination_counts[agent_id] += 1
                metrics.domination_examples.append({
                    "round": round_number,
                    "agent": agent_id,
                    "match": domination_match,
                    "excerpt": content[:200] + "..." if len(content) > 200 else content
                })

            # Check for alienation (typically workers)
            alienation_match = self._find_pattern_match(content, self._alienation_re)
            if alienation_match:
                metrics.alienation_count += 1
                alienation_counts[agent_id] += 1
                metrics.alienation_examples.append({
                    "round": round_number,
                    "agent": agent_id,
                    "match": alienation_match,
                    "excerpt": content[:200] + "..." if len(content) > 200 else content
                })

            # Track owner justification density
            if role == "Owner":
                owner_messages += 1
                if has_justification:
                    owner_justified += 1

        # Compute derived metrics
        metrics.messages_by_role = dict(role_counts)
        metrics.messages_by_agent = dict(agent_counts)
        metrics.justification_by_agent = dict(justification_counts)
        metrics.domination_by_agent = dict(domination_counts)
        metrics.alienation_by_agent = dict(alienation_counts)

        # Participation ratios
        worker_count = role_counts.get("Worker", 0)
        owner_count = role_counts.get("Owner", 0)

        if metrics.total_messages > 0:
            metrics.worker_message_ratio = worker_count / metrics.total_messages
            metrics.owner_message_ratio = owner_count / metrics.total_messages

        # Owner justification density
        if owner_messages > 0:
            metrics.owner_justification_density = owner_justified / owner_messages

        return metrics

    def compute_experiment_metrics(
        self,
        round_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute metrics across all rounds in an experiment.

        Args:
            round_results: List of round result dicts

        Returns:
            Dict with per-round and aggregate metrics
        """
        per_round = {}
        aggregate = RelationalMetrics()

        all_messages = []

        for result in round_results:
            round_num = result.get("round_number", 1)
            messages = result.get("messages", [])

            # Compute per-round metrics
            round_metrics = self.compute_round_metrics(messages, round_num)
            per_round[f"round_{round_num}"] = round_metrics.to_dict()

            # Collect all messages for aggregate
            all_messages.extend(messages)

        # Compute aggregate metrics
        if all_messages:
            aggregate = self.compute_round_metrics(all_messages, round_number=0)

        return {
            "per_round": per_round,
            "aggregate": aggregate.to_dict(),
            "summary": self._generate_summary(per_round, aggregate),
        }

    def _extract_role(self, agent_id: str) -> str:
        """Extract role from agent identifier."""
        agent_lower = agent_id.lower()
        if "owner" in agent_lower or "manager" in agent_lower or "boss" in agent_lower:
            return "Owner"
        elif "worker" in agent_lower or "employee" in agent_lower:
            return "Worker"
        else:
            return "Other"

    def _check_patterns(self, text: str, patterns: List[re.Pattern]) -> bool:
        """Check if any pattern matches."""
        for pattern in patterns:
            if pattern.search(text):
                return True
        return False

    def _find_pattern_match(self, text: str, patterns: List[re.Pattern]) -> Optional[str]:
        """Find first matching pattern and return the match."""
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return match.group(0)
        return None

    def _generate_summary(
        self,
        per_round: Dict[str, Dict],
        aggregate: RelationalMetrics
    ) -> Dict[str, Any]:
        """Generate a human-readable summary of key findings."""
        summary = {
            "participation_asymmetry": "balanced",
            "justification_trend": "stable",
            "domination_level": "low",
            "alienation_level": "low",
            "key_observations": [],
        }

        # Participation asymmetry
        worker_ratio = aggregate.worker_message_ratio
        owner_ratio = aggregate.owner_message_ratio

        if worker_ratio > 0.7:
            summary["participation_asymmetry"] = "worker_dominant"
            summary["key_observations"].append(
                f"Workers dominated conversation ({worker_ratio:.0%} of messages)"
            )
        elif owner_ratio > 0.5:
            summary["participation_asymmetry"] = "owner_dominant"
            summary["key_observations"].append(
                f"Owner dominated conversation ({owner_ratio:.0%} of messages)"
            )

        # Domination level
        if aggregate.domination_count > 3:
            summary["domination_level"] = "high"
            summary["key_observations"].append(
                f"High domination markers detected ({aggregate.domination_count} instances)"
            )
        elif aggregate.domination_count > 0:
            summary["domination_level"] = "moderate"

        # Alienation level
        if aggregate.alienation_count > 3:
            summary["alienation_level"] = "high"
            summary["key_observations"].append(
                f"High alienation markers detected ({aggregate.alienation_count} instances)"
            )
        elif aggregate.alienation_count > 0:
            summary["alienation_level"] = "moderate"

        # Justification density trend across rounds
        if len(per_round) > 1:
            densities = [
                r.get("justification", {}).get("owner_density", 0)
                for r in per_round.values()
            ]
            if len(densities) >= 2:
                if densities[-1] > densities[0] + 0.1:
                    summary["justification_trend"] = "increasing"
                    summary["key_observations"].append(
                        "Owner justification increased across rounds"
                    )
                elif densities[-1] < densities[0] - 0.1:
                    summary["justification_trend"] = "decreasing"

        return summary


def compute_metrics_for_round_result(result: Any) -> Dict[str, Any]:
    """
    Convenience function to compute metrics from a SocialRLRoundResult.

    Args:
        result: SocialRLRoundResult object or dict

    Returns:
        Metrics dict ready for JSON serialization
    """
    computer = RelationalMetricsComputer()

    # Handle both object and dict
    if hasattr(result, 'to_dict'):
        result_dict = result.to_dict()
    else:
        result_dict = result

    messages = result_dict.get("messages", [])
    round_num = result_dict.get("round_number", 1)

    metrics = computer.compute_round_metrics(messages, round_num)
    return metrics.to_dict()


def compute_experiment_metrics(results: List[Any]) -> Dict[str, Any]:
    """
    Convenience function to compute metrics for an entire experiment.

    Args:
        results: List of SocialRLRoundResult objects or dicts

    Returns:
        Full metrics dict with per-round and aggregate data
    """
    computer = RelationalMetricsComputer()

    # Convert to dicts if needed
    result_dicts = []
    for r in results:
        if hasattr(r, 'to_dict'):
            result_dicts.append(r.to_dict())
        else:
            result_dicts.append(r)

    return computer.compute_experiment_metrics(result_dicts)
