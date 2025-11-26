"""
WorldState - Gives agents a world to live in.

Instead of static scenarios, agents experience:
1. Events that affect them differently based on identity vectors
2. Topics that rotate with identity-specific stakes
3. Frustration that accumulates when repeatedly ignored
4. Position hardening when repeatedly unheard

The core insight: a normal person would get fed up with the same question.
Identity must manifest in differential response to shared events.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import random
from enum import Enum


# =============================================================================
# EVENTS - Things that happen in the world
# =============================================================================

@dataclass
class WorldEvent:
    """An event that happens in the world, affecting agents differentially."""
    id: str
    headline: str
    description: str

    # Which identity dimensions this event is salient to (and direction)
    # e.g., {"ideology": -0.3} means left-leaning agents care more
    salience_weights: Dict[str, float] = field(default_factory=dict)

    # Emotional valence by identity position
    # e.g., {"rural": "threatening", "urban": "opportunity"}
    valence_map: Dict[str, str] = field(default_factory=dict)

    # Stakes - what's at risk for different groups
    stakes_map: Dict[str, str] = field(default_factory=dict)


# Event library - things that can happen
EVENT_LIBRARY = [
    WorldEvent(
        id="housing_crisis",
        headline="Housing prices surge 15% in major cities",
        description="A new report shows housing costs outpacing wages for third straight year.",
        salience_weights={
            "sociogeographic": -0.4,  # Urban dwellers care more
            "tie_to_place": 0.3,      # Those rooted care more
            "engagement": 0.2,        # Engaged people follow news
        },
        valence_map={
            "renter": "threatening",
            "owner": "mixed",
            "rural": "distant",
            "urban": "urgent",
        },
        stakes_map={
            "renter": "Your ability to stay in your community",
            "owner": "Your property value vs. your children's future",
            "rural": "Urbanites moving to your area, changing it",
            "urban": "Whether you can afford to stay",
        }
    ),
    WorldEvent(
        id="factory_closure",
        headline="Major employer announces plant closure, 500 jobs affected",
        description="The region's largest manufacturer cites 'changing market conditions'.",
        salience_weights={
            "sociogeographic": 0.5,   # Rural/small town cares more
            "tie_to_place": 0.4,      # Rooted people affected
            "engagement": 0.1,
            "institutional_faith": -0.2,  # Tests faith in system
        },
        valence_map={
            "rural": "devastating",
            "suburban": "concerning",
            "urban": "distant",
            "working_class": "threatening",
        },
        stakes_map={
            "rural": "Your town's survival",
            "suburban": "Supply chain effects on your life",
            "urban": "Abstract economic news",
            "working_class": "Your livelihood, your identity",
        }
    ),
    WorldEvent(
        id="climate_policy",
        headline="Government announces new carbon pricing policy",
        description="The policy aims to reduce emissions by 40% but will increase energy costs.",
        salience_weights={
            "ideology": -0.5,         # Left supports, right opposes
            "sociogeographic": 0.3,   # Rural pays more (heating, driving)
            "engagement": 0.3,        # Political junkies care
        },
        valence_map={
            "left": "necessary progress",
            "right": "government overreach",
            "rural": "burden on us",
            "urban": "about time",
        },
        stakes_map={
            "left": "The planet's future",
            "right": "Freedom from government control",
            "rural": "Your heating bill, your way of life",
            "urban": "Air quality, climate action",
        }
    ),
    WorldEvent(
        id="immigration_debate",
        headline="Immigration levels reach record high amid labor shortage",
        description="Employers say they need workers; some residents worry about community change.",
        salience_weights={
            "ideology": 0.4,          # Right more concerned
            "tie_to_place": 0.3,      # Rooted worry about change
            "social_friction": 0.4,   # Friction-sensitive care
        },
        valence_map={
            "left": "economic necessity, humanitarian duty",
            "right": "too fast, threatens cohesion",
            "rural": "changing our community",
            "urban": "enriching diversity",
        },
        stakes_map={
            "left": "Workers' rights, humanitarian values",
            "right": "Community identity, social cohesion",
            "rural": "The character of your town",
            "urban": "Labor market dynamics",
        }
    ),
    WorldEvent(
        id="tech_layoffs",
        headline="Tech sector announces 10,000 layoffs despite record profits",
        description="Companies cite 'efficiency gains from AI' while posting billion-dollar earnings.",
        salience_weights={
            "ideology": -0.3,         # Left sees exploitation
            "institutional_faith": -0.3,  # Tests faith in corporations
            "engagement": 0.2,
        },
        valence_map={
            "left": "corporate greed exposed",
            "right": "market correction",
            "working_class": "could be us next",
            "professional": "job security anxiety",
        },
        stakes_map={
            "left": "Worker dignity vs. shareholder value",
            "right": "Free market efficiency",
            "working_class": "Whether any job is safe",
            "professional": "Your career trajectory",
        }
    ),
    WorldEvent(
        id="healthcare_wait",
        headline="Emergency room wait times hit 12-hour average",
        description="Health system officials blame staffing shortages and funding gaps.",
        salience_weights={
            "institutional_faith": 0.4,   # Tests faith in public systems
            "ideology": 0.2,              # Right: system failure; Left: underfunding
            "engagement": 0.2,
        },
        valence_map={
            "left": "consequence of austerity",
            "right": "government incompetence",
            "rural": "we never had good access anyway",
            "urban": "system overwhelmed",
        },
        stakes_map={
            "left": "Public healthcare's survival",
            "right": "Proof private would be better",
            "rural": "The hospital you depend on",
            "urban": "Access when you need it",
        }
    ),
    WorldEvent(
        id="election_called",
        headline="Snap election called for next month",
        description="Polls show a tight race with no party holding a clear lead.",
        salience_weights={
            "engagement": 0.5,            # Engaged people activated
            "partisanship": 0.4,          # Partisans care intensely
            "ideology": 0.1,
        },
        valence_map={
            "engaged": "finally, our chance",
            "disengaged": "here we go again",
            "partisan": "critical moment",
            "independent": "exhausting",
        },
        stakes_map={
            "engaged": "Your voice in democracy",
            "disengaged": "Does it even matter?",
            "partisan": "Your team's future",
            "independent": "Choosing the least bad option",
        }
    ),
    WorldEvent(
        id="local_business_closes",
        headline="Beloved local business closes after 40 years",
        description="Owner cites rising costs and competition from online retailers.",
        salience_weights={
            "tie_to_place": 0.5,          # Place-attached mourn it
            "sociogeographic": 0.3,       # Small town/rural care more
            "social_friction": 0.2,
        },
        valence_map={
            "rooted": "losing part of ourselves",
            "mobile": "that's how markets work",
            "rural": "another piece of our town gone",
            "urban": "sad but replaceable",
        },
        stakes_map={
            "rooted": "Your community's identity",
            "mobile": "Minor inconvenience",
            "rural": "The fabric of your town",
            "urban": "A nostalgic loss",
        }
    ),
]


# =============================================================================
# TOPICS - What agents discuss each round
# =============================================================================

@dataclass
class DiscussionTopic:
    """A topic for discussion with identity-specific stakes."""
    id: str
    question: str
    framing: str

    # How this topic connects to identity dimensions
    identity_stakes: Dict[str, str] = field(default_factory=dict)

    # Positions that different identity profiles might take
    position_seeds: Dict[str, str] = field(default_factory=dict)


TOPIC_ROTATION = [
    DiscussionTopic(
        id="belonging",
        question="What does it mean to belong somewhere?",
        framing="Consider your connection to place, community, and identity.",
        identity_stakes={
            "high_tie_to_place": "Everything - your roots define you",
            "low_tie_to_place": "Freedom to move matters more than staying",
            "rural": "Community is survival",
            "urban": "Belonging is chosen, not inherited",
        },
        position_seeds={
            "rural_rooted": "You belong where your family is buried",
            "urban_mobile": "You belong wherever you build your life",
            "conservative": "Belonging requires earning your place over time",
            "progressive": "Belonging should be open to all who show up",
        }
    ),
    DiscussionTopic(
        id="trust_institutions",
        question="Can we trust our institutions to solve problems?",
        framing="Think about government, media, corporations, courts.",
        identity_stakes={
            "high_institutional_faith": "They're imperfect but necessary",
            "low_institutional_faith": "They've failed us repeatedly",
            "engaged": "We must reform them",
            "disengaged": "Why bother?",
        },
        position_seeds={
            "institutionalist": "Work within the system to change it",
            "skeptic": "The system protects itself, not us",
            "radical": "Burn it down and rebuild",
            "pragmatist": "Use what works, bypass what doesn't",
        }
    ),
    DiscussionTopic(
        id="change_pace",
        question="Is change happening too fast, too slow, or about right?",
        framing="Technology, society, culture, economy - the pace of transformation.",
        identity_stakes={
            "conservative": "We're losing what made us who we are",
            "progressive": "Not fast enough - people are suffering",
            "older": "I don't recognize the world anymore",
            "younger": "Why are we still stuck in the past?",
        },
        position_seeds={
            "traditionalist": "Some things should never change",
            "accelerationist": "Faster - the old world is dying anyway",
            "cautious": "Change is fine but let people adjust",
            "frustrated": "It's not the pace, it's who benefits",
        }
    ),
    DiscussionTopic(
        id="fairness",
        question="Is society fundamentally fair or unfair?",
        framing="Consider who gets ahead and why.",
        identity_stakes={
            "high_engagement": "Unfairness demands action",
            "low_engagement": "Unfair but nothing I can do",
            "high_friction": "The game is rigged",
            "low_friction": "It's imperfect but workable",
        },
        position_seeds={
            "meritocrat": "Hard work gets rewarded, mostly",
            "structuralist": "The system advantages some from birth",
            "fatalist": "Life isn't fair, never was",
            "reformist": "We can make it fairer",
        }
    ),
    DiscussionTopic(
        id="us_vs_them",
        question="Who are 'we' and who are 'they'?",
        framing="Division seems everywhere. Where do you draw lines?",
        identity_stakes={
            "high_partisanship": "The other side threatens everything",
            "low_partisanship": "I don't see sides that way",
            "high_friction": "They don't understand us",
            "low_friction": "We have more in common than not",
        },
        position_seeds={
            "polarized": "You're either with us or against us",
            "bridge_builder": "Division is manufactured to control us",
            "tribalist": "My people first, always",
            "cosmopolitan": "Artificial boundaries harm everyone",
        }
    ),
    DiscussionTopic(
        id="voice_heard",
        question="Do people like you have a voice that matters?",
        framing="Think about whether your perspective is represented.",
        identity_stakes={
            "rural": "We're ignored by cities and media",
            "urban": "We're vilified as out of touch",
            "disengaged": "Nobody speaks for people like me",
            "engaged": "I fight to be heard",
        },
        position_seeds={
            "silenced": "They pretend to listen but never act",
            "empowered": "When we organize, we win",
            "cynical": "Voices don't matter, only money",
            "hopeful": "Things are slowly getting better",
        }
    ),
]


# =============================================================================
# FRUSTRATION & FATIGUE
# =============================================================================

@dataclass
class AgentState:
    """Tracks an agent's accumulated world-state."""
    agent_id: str

    # Frustration from being ignored (0-1)
    frustration: float = 0.0

    # Fatigue from repetitive engagement (0-1)
    fatigue: float = 0.0

    # Position hardening from unresolved disagreement (0-1)
    entrenchment: float = 0.0

    # Events this agent has strong reactions to
    event_reactions: List[Dict[str, Any]] = field(default_factory=list)

    # Topics where agent took strong positions
    position_history: List[Dict[str, Any]] = field(default_factory=list)

    # Recognition history (were they acknowledged?)
    recognition_history: List[float] = field(default_factory=list)


# =============================================================================
# WORLD STATE ENGINE
# =============================================================================

class WorldStateEngine:
    """
    Manages the evolving world that agents inhabit.

    Each round:
    1. Potentially injects a world event
    2. Selects a discussion topic
    3. Computes per-agent impact based on identity vectors
    4. Updates frustration/fatigue based on recognition patterns
    5. Generates identity-grounded context for each agent
    """

    def __init__(
        self,
        event_probability: float = 0.4,
        topic_rotation_mode: str = "sequential",  # or "random"
        seed: Optional[int] = None
    ):
        self.event_probability = event_probability
        self.topic_rotation_mode = topic_rotation_mode
        self.rng = random.Random(seed)

        # Agent states
        self.agent_states: Dict[str, AgentState] = {}

        # Current world state
        self.current_event: Optional[WorldEvent] = None
        self.current_topic: Optional[DiscussionTopic] = None
        self.round_number: int = 0

        # History
        self.event_history: List[WorldEvent] = []
        self.topic_history: List[DiscussionTopic] = []

    def initialize_agent(self, agent_id: str) -> None:
        """Register an agent in the world."""
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = AgentState(agent_id=agent_id)

    def advance_round(self) -> Tuple[Optional[WorldEvent], DiscussionTopic]:
        """
        Advance to next round, selecting event and topic.

        Returns:
            Tuple of (event or None, topic)
        """
        self.round_number += 1

        # Maybe inject an event
        if self.rng.random() < self.event_probability:
            # Pick event not recently used
            available = [e for e in EVENT_LIBRARY if e not in self.event_history[-3:]]
            if available:
                self.current_event = self.rng.choice(available)
                self.event_history.append(self.current_event)
            else:
                self.current_event = None
        else:
            self.current_event = None

        # Select topic
        if self.topic_rotation_mode == "sequential":
            idx = (self.round_number - 1) % len(TOPIC_ROTATION)
            self.current_topic = TOPIC_ROTATION[idx]
        else:
            self.current_topic = self.rng.choice(TOPIC_ROTATION)

        self.topic_history.append(self.current_topic)

        return self.current_event, self.current_topic

    def compute_agent_impact(
        self,
        agent_id: str,
        identity_vector: Dict[str, float],
        group_id: str
    ) -> Dict[str, Any]:
        """
        Compute how current world state affects this agent.

        Args:
            agent_id: Agent identifier
            identity_vector: 7D identity vector
            group_id: Agent's sociogeographic group

        Returns:
            Dict with event_reaction, topic_stakes, and context additions
        """
        self.initialize_agent(agent_id)
        state = self.agent_states[agent_id]

        result = {
            "event_context": None,
            "topic_context": None,
            "frustration_note": None,
            "position_seed": None,
        }

        # === EVENT IMPACT ===
        if self.current_event:
            # Compute salience based on identity vector
            salience = 0.0
            for dim, weight in self.current_event.salience_weights.items():
                dim_value = identity_vector.get(dim, 0.5)
                # Weight direction matters: positive weight = high value cares more
                # negative weight = low value cares more
                if weight > 0:
                    salience += weight * dim_value
                else:
                    salience += abs(weight) * (1 - dim_value)

            # Find valence based on group
            valence = "neutral"
            for key, val in self.current_event.valence_map.items():
                if key in group_id.lower():
                    valence = val
                    break

            # Find stakes
            stakes = "This affects the broader community."
            for key, val in self.current_event.stakes_map.items():
                if key in group_id.lower():
                    stakes = val
                    break

            # Record reaction
            state.event_reactions.append({
                "event_id": self.current_event.id,
                "salience": salience,
                "valence": valence,
                "round": self.round_number
            })

            # Generate context if salient enough
            if salience > 0.3:
                result["event_context"] = (
                    f"[RECENT NEWS: {self.current_event.headline}]\n"
                    f"For you, this feels {valence}. At stake: {stakes}"
                )

        # === TOPIC CONTEXT ===
        if self.current_topic:
            # Find relevant stakes
            topic_stakes = "This matters to everyone differently."
            position_seed = None

            # Match based on identity characteristics
            engagement = identity_vector.get("engagement", 0.5)
            ideology = identity_vector.get("ideology", 0.5)
            tie_to_place = identity_vector.get("tie_to_place", 0.5)
            partisanship = identity_vector.get("partisanship", 0.5)
            institutional_faith = identity_vector.get("institutional_faith", 0.5)
            social_friction = identity_vector.get("social_friction", 0.5)

            # Find matching stakes
            for key, val in self.current_topic.identity_stakes.items():
                if "high_engagement" in key and engagement > 0.6:
                    topic_stakes = val
                elif "low_engagement" in key and engagement < 0.4:
                    topic_stakes = val
                elif "high_tie_to_place" in key and tie_to_place > 0.6:
                    topic_stakes = val
                elif "low_tie_to_place" in key and tie_to_place < 0.4:
                    topic_stakes = val
                elif "high_institutional_faith" in key and institutional_faith > 0.6:
                    topic_stakes = val
                elif "low_institutional_faith" in key and institutional_faith < 0.4:
                    topic_stakes = val
                elif "high_friction" in key and social_friction > 0.6:
                    topic_stakes = val
                elif "low_friction" in key and social_friction < 0.4:
                    topic_stakes = val
                elif "rural" in key and "rural" in group_id.lower():
                    topic_stakes = val
                elif "urban" in key and "urban" in group_id.lower():
                    topic_stakes = val
                elif "conservative" in key and ideology > 0.6:
                    topic_stakes = val
                elif "progressive" in key and ideology < 0.4:
                    topic_stakes = val

            # Find position seed based on profile
            for key, val in self.current_topic.position_seeds.items():
                if "rural" in key and "rural" in group_id.lower():
                    position_seed = val
                elif "urban" in key and "urban" in group_id.lower():
                    position_seed = val
                elif "conservative" in key and ideology > 0.6:
                    position_seed = val
                elif "progressive" in key and ideology < 0.4:
                    position_seed = val
                elif "institutionalist" in key and institutional_faith > 0.6:
                    position_seed = val
                elif "skeptic" in key and institutional_faith < 0.4:
                    position_seed = val

            result["topic_context"] = (
                f"[DISCUSSION: {self.current_topic.question}]\n"
                f"{self.current_topic.framing}\n"
                f"For someone like you: {topic_stakes}"
            )
            result["position_seed"] = position_seed

        # === FRUSTRATION CONTEXT ===
        if state.frustration > 0.3:
            if state.frustration > 0.7:
                result["frustration_note"] = (
                    "[You've been consistently ignored. Why keep trying? "
                    "Maybe silence is louder than words that nobody hears.]"
                )
            else:
                result["frustration_note"] = (
                    "[Your contributions haven't been acknowledged much. "
                    "Do you need to be louder, or is this conversation not for you?]"
                )

        return result

    def update_agent_recognition(
        self,
        agent_id: str,
        was_recognized: bool,
        recognition_score: float = 0.0
    ) -> None:
        """
        Update agent's recognition state after a round.

        Args:
            agent_id: Agent identifier
            was_recognized: Whether agent was acknowledged by others
            recognition_score: 0-1 recognition level
        """
        self.initialize_agent(agent_id)
        state = self.agent_states[agent_id]

        # Update recognition history
        state.recognition_history.append(recognition_score)

        # Update frustration (EMA)
        alpha = 0.3
        if was_recognized:
            # Recognition reduces frustration
            state.frustration = max(0, state.frustration * (1 - alpha) + 0.0 * alpha)
        else:
            # Lack of recognition increases frustration
            state.frustration = min(1.0, state.frustration * (1 - alpha) + 0.8 * alpha)

        # Update fatigue (always increases slightly, reduced by engagement)
        state.fatigue = min(1.0, state.fatigue + 0.05)
        if recognition_score > 0.5:
            state.fatigue = max(0, state.fatigue - 0.1)

    def update_agent_position(
        self,
        agent_id: str,
        topic_id: str,
        position_taken: str,
        was_challenged: bool
    ) -> None:
        """
        Track position history for entrenchment calculation.

        Args:
            agent_id: Agent identifier
            topic_id: Topic this position was on
            position_taken: The position they expressed
            was_challenged: Whether others disagreed
        """
        self.initialize_agent(agent_id)
        state = self.agent_states[agent_id]

        state.position_history.append({
            "topic_id": topic_id,
            "position": position_taken,
            "challenged": was_challenged,
            "round": self.round_number
        })

        # Entrenchment increases when challenged but not moved
        if was_challenged:
            state.entrenchment = min(1.0, state.entrenchment + 0.1)

    def get_world_context_injection(
        self,
        agent_id: str,
        identity_vector: Dict[str, float],
        group_id: str
    ) -> str:
        """
        Get full world context to inject into agent's prompt.

        This is the main interface - call this when generating agent prompts.
        """
        impact = self.compute_agent_impact(agent_id, identity_vector, group_id)

        parts = []

        if impact["event_context"]:
            parts.append(impact["event_context"])

        if impact["topic_context"]:
            parts.append(impact["topic_context"])

        if impact["frustration_note"]:
            parts.append(impact["frustration_note"])

        if impact["position_seed"]:
            parts.append(f"\n[A natural starting point for you: {impact['position_seed']}]")

        if parts:
            return "\n\n=== WORLD CONTEXT ===\n" + "\n\n".join(parts)
        else:
            return ""

    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current world state for logging."""
        return {
            "round": self.round_number,
            "current_event": self.current_event.id if self.current_event else None,
            "current_topic": self.current_topic.id if self.current_topic else None,
            "agent_states": {
                agent_id: {
                    "frustration": state.frustration,
                    "fatigue": state.fatigue,
                    "entrenchment": state.entrenchment,
                    "recent_recognitions": state.recognition_history[-3:] if state.recognition_history else []
                }
                for agent_id, state in self.agent_states.items()
            }
        }


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def create_world_engine(
    seed: int = 42,
    event_probability: float = 0.4,
    topic_mode: str = "sequential"
) -> WorldStateEngine:
    """Create a WorldStateEngine with default settings."""
    return WorldStateEngine(
        event_probability=event_probability,
        topic_rotation_mode=topic_mode,
        seed=seed
    )


if __name__ == "__main__":
    # Test the world state engine
    print("=== WorldState Test ===\n")

    engine = create_world_engine(seed=42)

    # Test identity vectors
    test_agents = {
        "Urban_Progressive": {
            "vector": {"engagement": 0.8, "ideology": 0.25, "sociogeographic": 0.3,
                      "tie_to_place": 0.4, "partisanship": 0.9, "institutional_faith": 0.6,
                      "social_friction": 0.4},
            "group": "urban_left"
        },
        "Rural_Conservative": {
            "vector": {"engagement": 0.6, "ideology": 0.75, "sociogeographic": 0.8,
                      "tie_to_place": 0.9, "partisanship": 0.8, "institutional_faith": 0.3,
                      "social_friction": 0.6},
            "group": "rural_right"
        },
        "Disengaged_Renter": {
            "vector": {"engagement": 0.1, "ideology": 0.5, "sociogeographic": 0.4,
                      "tie_to_place": 0.2, "partisanship": 0.2, "institutional_faith": 0.3,
                      "social_friction": 0.5},
            "group": "renter_disengaged"
        }
    }

    # Simulate 5 rounds
    for round_num in range(1, 6):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}")
        print('='*60)

        event, topic = engine.advance_round()

        if event:
            print(f"\n📰 EVENT: {event.headline}")
        print(f"\n💬 TOPIC: {topic.question}")

        for agent_id, agent_data in test_agents.items():
            context = engine.get_world_context_injection(
                agent_id,
                agent_data["vector"],
                agent_data["group"]
            )
            print(f"\n--- {agent_id} ---")
            print(context[:500] if len(context) > 500 else context)

            # Simulate recognition (Urban gets recognized, Disengaged doesn't)
            if "Urban" in agent_id:
                engine.update_agent_recognition(agent_id, True, 0.7)
            elif "Rural" in agent_id:
                engine.update_agent_recognition(agent_id, True, 0.4)
            else:
                engine.update_agent_recognition(agent_id, False, 0.1)

    print("\n\n=== Final Agent States ===")
    summary = engine.get_state_summary()
    for agent_id, state in summary["agent_states"].items():
        print(f"\n{agent_id}:")
        print(f"  Frustration: {state['frustration']:.2f}")
        print(f"  Fatigue: {state['fatigue']:.2f}")
