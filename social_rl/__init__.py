"""
Social RL - Reinforcement Learning through Social Interaction

A novel approach to agent learning that uses social dynamics as the reward signal:
- Process Retrieval as Policy: PRAR guides HOW agents reason
- Social Feedback as Reward: Interaction signals drive adaptation
- Dynamic Context Injection: Manifestations evolve per turn
- Theoretical Grounding: Framework constraints prevent drift

This creates emergent learning without explicit reward functions or weight updates.

Components:
- ContextInjector: Dynamic round/turn manifestation generation
- SocialFeedbackExtractor: Extract learning signals from interaction
- ProcessRetriever: PRAR-based reasoning policy retrieval
- SocialRLRunner: Main execution engine

Usage:
    from social_rl import SocialRLRunner, SocialRLConfig

    # From state file
    runner = create_social_rl_runner('state.json', llm_client, mode='progressive')

    # Execute
    results = runner.execute_all_rounds()

    # Report
    print(runner.generate_report())
"""

from .context_injector import (
    ContextInjector,
    TurnContext,
    TheoreticalFramework,
    ManifestationType,
    create_context_injector_from_canvas
)

from .feedback_extractor import (
    SocialFeedbackExtractor,
    SocialFeedback,
    ConceptMarkers,
    create_extractor_for_framework
)

from .process_retriever import (
    ProcessRetriever,
    ReasoningPolicy,
    ProcessCue,
    ReasoningMode,
    AdaptiveProcessRetriever
)

from .runner import (
    SocialRLRunner,
    SocialRLConfig,
    SocialRLMessage,
    SocialRLRoundResult,
    create_social_rl_runner
)

__all__ = [
    # Context Injection
    "ContextInjector",
    "TurnContext",
    "TheoreticalFramework",
    "ManifestationType",
    "create_context_injector_from_canvas",

    # Feedback Extraction
    "SocialFeedbackExtractor",
    "SocialFeedback",
    "ConceptMarkers",
    "create_extractor_for_framework",

    # Process Retrieval
    "ProcessRetriever",
    "ReasoningPolicy",
    "ProcessCue",
    "ReasoningMode",
    "AdaptiveProcessRetriever",

    # Runner
    "SocialRLRunner",
    "SocialRLConfig",
    "SocialRLMessage",
    "SocialRLRoundResult",
    "create_social_rl_runner",

    # Dual-LLM Client
    "DualLLMClient",
    "DualLLMConfig",
    "create_dual_llm_client",
    "create_dual_llm_from_single",

    # Schema
    "SCHEMA_VERSION",
]

# Import dual-LLM client
from .dual_llm_client import (
    DualLLMClient,
    DualLLMConfig,
    create_dual_llm_client,
    create_dual_llm_from_single,
)

# Import schema version
from .schema import SCHEMA_VERSION

__version__ = "0.2.0"
