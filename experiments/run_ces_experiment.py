#!/usr/bin/env python3
"""
CES-Grounded Social RL Experiment

Demonstrates Social Aesthetics through CES-derived agents in sociogeographic context.
NO alienation/non-domination - this is real social dynamics from empirical data.

Key innovation: Agents are dynamically generated from CES 2021 profiles,
not static personas. Their constraints come from actual survey distributions.

Usage:
    python experiments/run_ces_experiment.py --provider mock --max-turns 12
    python experiments/run_ces_experiment.py --provider ollama --model qwen2.5:7b
"""

import sys
import json
import argparse
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import CES generators
from agents.ces_generators import (
    CESAgentConfig,
    ces_row_to_agent,
    CESVariableMapper,
)

# Import semiotic analysis tools for ADAPTIVE mode tracking
from social_rl.semiotic_coder import (
    SemioticCoder, JustificationType, VoiceMarker, RelationalStance
)
from social_rl.context_injector import ManifestationType
from experiments.social_aesthetics_regimes import identify_regime, RegimeType


def compute_round_semiotic_metrics(messages: List[Any]) -> Dict[str, float]:
    """
    Compute aggregate semiotic metrics for a round from agent messages.

    Returns dict with:
        - engagement: Mean of message lengths relative to max (proxy for substantive engagement)
        - voice_valence: (empowered - alienated) / total voice markers
        - stance_valence: (bridging - dismissive) / total stance markers
        - justificatory_pct: Fraction of justificatory speech acts
    """
    if not messages:
        return {
            "engagement": 0.0,
            "voice_valence": 0.0,
            "stance_valence": 0.0,
            "justificatory_pct": 0.0
        }

    # Helper to extract content and agent_id from message (handles both dict and object)
    def get_msg_content(m):
        if hasattr(m, 'content'):
            return m.content
        elif isinstance(m, dict):
            return m.get("content", "")
        return ""

    def get_msg_agent_id(m):
        if hasattr(m, 'agent_id'):
            return m.agent_id
        elif isinstance(m, dict):
            return m.get("agent_id", "unknown")
        return "unknown"

    def get_turn_number(m, idx):
        if hasattr(m, 'turn_number'):
            return m.turn_number
        elif isinstance(m, dict):
            return m.get("turn_number", idx)
        return idx

    # Proxy for engagement: normalized message length diversity
    # Messages that are longer and more varied indicate higher engagement
    msg_lengths = [len(get_msg_content(m)) for m in messages]
    max_len = max(msg_lengths) if msg_lengths else 1
    engagement = sum(l / max_len for l in msg_lengths) / len(msg_lengths) if msg_lengths else 0.0

    # Parse semiotic markers from messages using lexicon coder
    coder = SemioticCoder()  # Uses lexicon-based coding by default
    voice_empowered = 0
    voice_alienated = 0
    stance_bridging = 0
    stance_dismissive = 0
    justificatory_count = 0
    total_speech_acts = 0

    for idx, msg in enumerate(messages):
        content = get_msg_content(msg)
        if not content:
            continue

        # Code the message using code_utterance()
        coded = coder.code_utterance(
            content=content,
            agent_id=get_msg_agent_id(msg),
            turn=get_turn_number(msg, idx),
            round_num=1  # Round number not critical for aggregate metrics
        )

        # Voice markers (coded.voice is a VoiceMarker enum)
        if coded.voice == VoiceMarker.EMPOWERED:
            voice_empowered += 1
        elif coded.voice == VoiceMarker.ALIENATED:
            voice_alienated += 1

        # Stance markers (coded.stance is a RelationalStance enum)
        if coded.stance == RelationalStance.BRIDGING:
            stance_bridging += 1
        elif coded.stance == RelationalStance.DISMISSIVE:
            stance_dismissive += 1

        # Justification types (coded.justification is a JustificationType enum)
        if coded.justification == JustificationType.JUSTIFICATORY:
            justificatory_count += 1
        total_speech_acts += 1

    # Compute valences
    total_voice = voice_empowered + voice_alienated
    voice_valence = (voice_empowered - voice_alienated) / total_voice if total_voice > 0 else 0.0

    total_stance = stance_bridging + stance_dismissive
    stance_valence = (stance_bridging - stance_dismissive) / total_stance if total_stance > 0 else 0.5  # Default to neutral

    justificatory_pct = justificatory_count / total_speech_acts if total_speech_acts > 0 else 0.0

    return {
        "engagement": min(engagement, 1.0),  # Clamp to [0,1]
        "voice_valence": max(-1.0, min(1.0, voice_valence)),  # Clamp to [-1,1]
        "stance_valence": max(0.0, min(1.0, (stance_valence + 1) / 2)),  # Map [-1,1] to [0,1]
        "justificatory_pct": min(justificatory_pct, 1.0)
    }

# Import LLM client
import importlib.util
llm_spec = importlib.util.spec_from_file_location(
    "llm_client",
    str(PROJECT_ROOT / "local_rcm" / "llm_client.py")
)
llm_module = importlib.util.module_from_spec(llm_spec)
llm_spec.loader.exec_module(llm_module)

OllamaClient = llm_module.OllamaClient
OpenAIClient = llm_module.OpenAIClient
MockClient = llm_module.MockClient


class DualLLMCompatibleClient:
    """Wrapper for DualLLMClient compatibility."""

    def __init__(self, client, default_temperature: float = 0.7):
        self._client = client
        self._default_temperature = default_temperature
        self._calls = []

    def send_message(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = None,
        max_tokens: int = 512
    ) -> str:
        self._calls.append({
            "temperature": temperature or self._default_temperature,
            "max_tokens": max_tokens,
            "prompt_length": len(system_prompt) + len(user_message)
        })
        return self._client.send_message(system_prompt, user_message)

    @property
    def call_log(self):
        return self._calls


# =============================================================================
# Simulated CES Data (representative profiles)
# =============================================================================

def create_ces_profiles() -> List[Dict[str, Any]]:
    """
    Create simulated CES profiles representing key voter clusters.

    These are based on real CES 2021 distributions, not fantasy personas.
    Each represents a meaningfully different sociogeographic position.
    """
    return [
        # Urban progressive - GTA, young, university-educated, NDP-leaning
        {
            "cps21_ResponseId": "CES_Urban_Progressive",
            "cps21_province": 35,  # Ontario
            "cps21_yob": 1995,     # ~26 in 2021
            "cps21_genderid": 2,   # Woman
            "cps21_education": 9,  # Bachelor's
            "cps21_income_cat": 4, # Lower-middle
            "cps21_urban_rural": 1,  # Urban
            "cps21_pid_party": 3,  # NDP
            "cps21_lr_scale": 2.5, # Left
            "cps21_turnout": 1,    # Certain to vote
            "cps21_bornin_canada": 1,
            "cps21_riding_id": "ON_TOR_01",
        },
        # Suburban swing voter - 905 region, middle-aged, undecided
        {
            "cps21_ResponseId": "CES_Suburban_Swing",
            "cps21_province": 35,  # Ontario
            "cps21_yob": 1975,     # ~46 in 2021
            "cps21_genderid": 1,   # Man
            "cps21_education": 7,  # College
            "cps21_income_cat": 7, # Upper-middle
            "cps21_urban_rural": 2,  # Suburban
            "cps21_pid_party": 8,  # No party ID
            "cps21_lr_scale": 5.5, # Centre-right
            "cps21_turnout": 2,    # Likely
            "cps21_bornin_canada": 1,
            "cps21_riding_id": "ON_905_02",
        },
        # Rural conservative - Northern Ontario, older, CPC-leaning
        {
            "cps21_ResponseId": "CES_Rural_Conservative",
            "cps21_province": 35,  # Ontario
            "cps21_yob": 1960,     # ~61 in 2021
            "cps21_genderid": 1,   # Man
            "cps21_education": 5,  # High school
            "cps21_income_cat": 5, # Middle
            "cps21_urban_rural": 3,  # Rural
            "cps21_pid_party": 2,  # Conservative
            "cps21_lr_scale": 7.5, # Right
            "cps21_turnout": 1,    # Certain to vote
            "cps21_bornin_canada": 1,
            "cps21_riding_id": "ON_NORTH_03",
        },
        # Disengaged young renter - urban, low income, doesn't usually vote
        {
            "cps21_ResponseId": "CES_Disengaged_Renter",
            "cps21_province": 35,  # Ontario
            "cps21_yob": 1998,     # ~23 in 2021
            "cps21_genderid": 3,   # Non-binary
            "cps21_education": 8,  # Some university
            "cps21_income_cat": 2, # Low
            "cps21_urban_rural": 1,  # Urban
            "cps21_pid_party": 8,  # No party ID
            "cps21_lr_scale": 4.0, # Centre-left
            "cps21_turnout": 3,    # Unlikely to vote
            "cps21_bornin_canada": 1,
            "cps21_riding_id": "ON_TOR_04",
        },
    ]


# =============================================================================
# CES Canvas Builder
# =============================================================================

def build_ces_canvas(
    agents: List[CESAgentConfig],
    scenario: str = "federal_election",
    rounds: int = 2
) -> Dict[str, Any]:
    """
    Build a Social RL canvas from CES agent configurations.

    This creates a framework-agnostic canvas suitable for
    sociogeographic political simulation.
    """
    # Convert agents to canvas format
    canvas_agents = []
    for agent in agents:
        canvas_agent = agent.to_canvas_agent()
        # Add simulation-specific fields
        canvas_agent["prompt"] = _build_agent_prompt(agent)
        canvas_agents.append(canvas_agent)

    # Build participant list
    participants = ", ".join([f"CES_{a.source_id}" for a in agents])

    canvas = {
        "project": {
            "goal": "Understand political discourse dynamics among CES-grounded voter types",
            "theoretical_option": "SA",  # Social Aesthetics
            "theoretical_option_label": "Social Aesthetics: Sociogeographic Political Dynamics",
            "setting": "Online political discussion among Ontario voters during federal election campaign",
            "data_source": "CES 2021 (simulated representative profiles)",
        },
        "agents": canvas_agents,
        "rounds": [],
    }

    # Build rounds
    if scenario == "federal_election":
        canvas["rounds"] = [
            {
                "round_number": 1,
                "scenario": "Housing affordability debate: Voters discuss which party's housing policy would actually help them.",
                "tasks": "Each participant shares their perspective on housing affordability based on their actual situation and location. Discuss policy proposals.",
                "rules": "Speak from your own CES-grounded perspective. Reference your actual location (urban/suburban/rural). Engage with others' views.",
                "platform_config": {
                    "participants": participants,
                    "who_sends": "All",
                    "order": "Round-robin",
                    "end_condition": "Total messages: 12"
                }
            },
            {
                "round_number": 2,
                "scenario": "Turnout and participation: Should voting be mandatory? Does politics actually matter for people like you?",
                "tasks": "Discuss political engagement from your actual turnout likelihood. Address whether parties reach out to voters like you.",
                "rules": "Be authentic to your CES profile's engagement level. Reference real geographic and demographic factors.",
                "platform_config": {
                    "participants": participants,
                    "who_sends": "All",
                    "order": "Round-robin",
                    "end_condition": "Total messages: 12"
                }
            }
        ]

    if rounds > 2:
        canvas["rounds"].append({
            "round_number": 3,
            "scenario": "Cross-cutting exposure: Can voters with different positions find common ground?",
            "tasks": "Identify shared concerns across urban/rural, left/right divides. Discuss what would make politics work better.",
            "rules": "Actively engage with perspectives different from your own. Seek understanding, not just disagreement.",
            "platform_config": {
                "participants": participants,
                "who_sends": "All",
                "order": "Round-robin",
                "end_condition": "Total messages: 12"
            }
        })

    return canvas


def _build_agent_prompt(agent: CESAgentConfig) -> str:
    """Build agent prompt from CES configuration."""
    return f"""ROLE: You are a Canadian voter with the following CES-derived profile.

PROFILE:
- Location: {agent.province_name} ({agent.urban_rural})
- Demographics: {agent.age_group}, {agent.gender}, {agent.education_label}
- Income: Quintile {agent.income_quintile}
- Political leaning: {agent.party_name} ({agent.ideology_lr}/10 on left-right scale)
- Turnout likelihood: {agent.turnout_likelihood:.0%}

PERSONA: {agent.persona_description}

CONSTRAINTS:
{chr(10).join('- ' + c for c in agent.constraints) if agent.constraints else '- Speak authentically from your socioeconomic position'}

GOALS: {agent._generate_goal()}

Respond naturally as this person would in a political discussion. Your views should reflect your actual CES profile - don't suddenly become someone you're not."""


# =============================================================================
# Experiment Runner
# =============================================================================

def run_ces_experiment(
    model: str = "qwen2.5:7b",
    provider: str = "ollama",
    rounds: int = 2,
    max_turns: int = 12,
    use_dual_llm: bool = True,
    verbose: bool = True,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    experiment_id: Optional[str] = None,
    # TRUE dual-LLM parameters
    performer_url: Optional[str] = None,
    performer_model: Optional[str] = None,
    coach_url: Optional[str] = None,
    coach_model: Optional[str] = None,
    # Challenge mode for A/B testing
    challenge_mode: str = "adaptive",
    # Context injection mode (static/progressive/reactive/adaptive)
    context_mode: str = "adaptive",
    # 2x2x2 sweep condition and seed tracking
    condition: Optional[str] = None,  # e.g., "A", "B", ..., "H"
    seed: Optional[int] = None,  # Seed number for replication
) -> Dict[str, Any]:
    """
    Run a CES-grounded Social RL experiment.

    For TRUE dual-LLM (two separate models on different GPUs):
        --performer-url https://a100-pod/v1 --performer-model Qwen/Qwen2.5-14B-Instruct
        --coach-url https://a40-pod/v1 --coach-model Qwen/Qwen2.5-7B-Instruct
    """
    from social_rl.runner import SocialRLRunner, SocialRLConfig
    from social_rl.dual_llm_client import DualLLMClient, DualLLMConfig, create_true_dual_llm

    # Check for TRUE dual-LLM mode
    true_dual_mode = performer_url and coach_url and performer_model and coach_model

    print("=" * 60)
    print("CES-GROUNDED SOCIAL RL EXPERIMENT")
    print("=" * 60)
    if true_dual_mode:
        print(f"Mode: TRUE DUAL-LLM (two separate models)")
        print(f"Performer: {performer_model} @ {performer_url}")
        print(f"Coach: {coach_model} @ {coach_url}")
    else:
        print(f"Model: {model}")
        print(f"Provider: {provider}")
    print(f"Rounds: {rounds}")
    print(f"Theory: Social Aesthetics (NOT alienation)")
    print("=" * 60)

    # Generate experiment ID
    import os
    if not experiment_id:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        experiment_id = f"ces_experiment_{timestamp}"

    # Create LLM client(s)
    effective_api_key = api_key or os.environ.get("RUNPOD_API_KEY") or os.environ.get("OPENAI_API_KEY")

    # TRUE dual-LLM mode: two separate endpoints
    if true_dual_mode:
        print(f"\nCreating TRUE dual-LLM with separate endpoints...")
        dual_llm = create_true_dual_llm(
            performer_base_url=performer_url,
            performer_model=performer_model,
            coach_base_url=coach_url,
            coach_model=coach_model,
            api_key=effective_api_key or "not-needed"
        )
        # Create a wrapped client for the runner (uses performer for non-dual calls)
        base_client = OpenAIClient(
            api_key=effective_api_key or "not-needed",
            model=performer_model,
            base_url=performer_url,
            timeout=180.0
        )
        wrapped_client = DualLLMCompatibleClient(base_client)
        print("TRUE dual-LLM configured (separate models on separate GPUs)\n")
    else:
        # Standard single-client mode
        if provider == "mock":
            base_client = MockClient()
            print("\n[Using MockClient for testing]\n")
        elif provider == "ollama":
            print(f"\nConnecting to Ollama ({model})...")
            base_client = OllamaClient(model=model)
        elif provider == "vllm":
            if not base_url:
                base_url = os.environ.get("OPENAI_BASE_URL")
            if not base_url:
                raise ValueError("--base-url required for vLLM provider")
            print(f"\nConnecting to vLLM: {base_url}")
            base_client = OpenAIClient(
                api_key=effective_api_key or "not-needed",
                model=model,
                base_url=base_url,
                timeout=180.0
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        wrapped_client = DualLLMCompatibleClient(base_client)

        # Create Dual-LLM client (pseudo - same model, different temps)
        dual_llm = None
        if use_dual_llm:
            dual_config = DualLLMConfig(
                performer_temperature=0.7,
                coach_temperature=0.1,
                log_coach_critiques=True,
                max_validation_retries=1
            )
            dual_llm = DualLLMClient(
                performer_client=wrapped_client,
                coach_client=wrapped_client,
                config=dual_config
            )
            print("Pseudo dual-LLM configured (same model, different temps)\n")

    # Generate CES agents
    print("Generating CES-grounded agents...")
    mapper = CESVariableMapper()
    ces_profiles = create_ces_profiles()
    agents = [ces_row_to_agent(p, mapper) for p in ces_profiles]

    for agent in agents:
        print(f"  - {agent.source_id}: {agent.province_name} {agent.urban_rural}, {agent.party_name}")

    # Build canvas
    print("\nBuilding CES canvas...")
    canvas = build_ces_canvas(agents, scenario="federal_election", rounds=rounds)

    # Configure runner
    config = SocialRLConfig(
        manifestation_mode=context_mode,  # Use context_mode for 2x2x2 sweep
        use_prar_cues=True,
        prar_intensity="low",  # Less prescriptive for naturalistic discourse
        use_coach_validation=use_dual_llm,
        verbose=verbose,
        auto_save=True,
        challenge_mode=challenge_mode  # For A/B testing: "off", "adaptive", "always"
    )
    print(f"Context mode: {context_mode}")
    print(f"Challenge mode: {challenge_mode}")

    # Create runner
    runner = SocialRLRunner(
        canvas=canvas,
        llm_client=wrapped_client,
        config=config,
        dual_llm_client=dual_llm,
        experiment_id=experiment_id
    )

    # Execute rounds with semiotic tracking
    results = []
    semiotic_state_log = []  # Track semiotic state per round

    for round_num in range(1, min(rounds + 1, len(canvas.get("rounds", [])) + 1)):
        print(f"\n{'='*60}")
        print(f"EXECUTING ROUND {round_num}")
        print(f"{'='*60}\n")

        try:
            result = runner.execute_round(round_num, max_turns=max_turns)
            results.append(result)

            # === SEMIOTIC STATE TRACKING (émile-inspired) ===
            # Get messages from this round
            round_messages = result.messages if hasattr(result, 'messages') else []
            if not round_messages and hasattr(result, 'to_dict'):
                round_messages = result.to_dict().get('messages', [])

            # Compute round metrics
            round_metrics = compute_round_semiotic_metrics(round_messages)

            # Update context_injector's semiotic tracker (if ADAPTIVE mode)
            ema_metrics = {}
            collapse_type = "none"
            divergence_injected = False
            if hasattr(runner, 'context_injector') and runner.context_injector:
                ci = runner.context_injector
                if hasattr(ci, 'update_semiotic_state'):
                    ci.update_semiotic_state(round_metrics)
                if hasattr(ci, 'get_ema_metrics'):
                    ema_metrics = ci.get_ema_metrics() or {}
                if hasattr(ci, 'was_divergence_injected'):
                    divergence_injected = ci.was_divergence_injected()
                # Get collapse type from should_inject_divergence if tracker exists
                if hasattr(ci, 'semiotic_tracker') and ci.semiotic_tracker:
                    _, detected_collapse = ci.semiotic_tracker.should_inject_divergence()
                    collapse_type = detected_collapse or "none"

            # Classify regime for this round
            regime = identify_regime(
                round_metrics["engagement"],
                round_metrics["voice_valence"],
                round_metrics["stance_valence"],
                round_metrics["justificatory_pct"]
            )
            regime_name = regime.name if regime else "UNKNOWN"

            # Build semiotic state entry
            state_entry = {
                "round_number": round_num,
                "regime": regime_name,
                "raw_metrics": round_metrics,
                "ema_metrics": ema_metrics,
                "collapse_type": collapse_type,
                "divergence_injected": divergence_injected
            }
            semiotic_state_log.append(state_entry)

            # Print semiotic state summary
            print(f"\n  [SEMIOTIC STATE] Round {round_num}:")
            print(f"    Regime: {regime_name}")
            print(f"    Engagement: {round_metrics['engagement']:.2f}")
            print(f"    Voice: {round_metrics['voice_valence']:.2f}")
            print(f"    Stance: {round_metrics['stance_valence']:.2f}")
            print(f"    Justification: {round_metrics['justificatory_pct']:.1%}")
            if divergence_injected:
                print(f"    → DIVERGENCE INJECTED ({collapse_type})")

        except Exception as e:
            print(f"Error in round {round_num}: {e}")
            import traceback
            traceback.print_exc()
            break

    # Generate report
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    report = runner.generate_report()
    print(report)

    # Build meta with 2x2x2 architecture parameters
    meta = {
        "experiment_id": experiment_id,
        "experiment_type": "CES-grounded",
        "framework": "Social Aesthetics",
        "data_source": "CES 2021 (simulated)",
        "agents": [a.source_id for a in agents],
        "model": model,
        "provider": provider,
        "rounds_executed": len(results),
        "timestamp": datetime.datetime.now().isoformat(),
        # 2x2x2 architecture sweep parameters
        "condition": condition,  # A-H condition label
        "seed": seed,  # Replication seed number
        "context_mode": context_mode,
        "challenge_mode": challenge_mode,
        "dual_llm": use_dual_llm,
        "true_dual_llm": true_dual_mode,
        # Regime trajectory summary
        "regime_trajectory": [s["regime"] for s in semiotic_state_log],
        "final_regime": semiotic_state_log[-1]["regime"] if semiotic_state_log else None,
        "divergence_events": sum(1 for s in semiotic_state_log if s["divergence_injected"])
    }

    # Save meta
    if results and runner.output_dir:
        meta_path = runner.output_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [SAVED] {meta_path}")

        # Save CES profiles
        profiles_path = runner.output_dir / "ces_profiles.json"
        with open(profiles_path, "w") as f:
            json.dump(ces_profiles, f, indent=2)
        print(f"  [SAVED] {profiles_path}")

        # Save semiotic state log (émile-style tracking)
        semiotic_log_path = runner.output_dir / "semiotic_state_log.json"
        with open(semiotic_log_path, "w") as f:
            json.dump({
                "experiment_id": experiment_id,
                "context_mode": context_mode,
                "challenge_mode": challenge_mode,
                "rounds": semiotic_state_log,
                # Include divergence log from context_injector if available
                "divergence_log": (
                    runner.context_injector.get_divergence_log()
                    if hasattr(runner, 'context_injector') and runner.context_injector
                    and hasattr(runner.context_injector, 'get_divergence_log')
                    else []
                )
            }, f, indent=2)
        print(f"  [SAVED] {semiotic_log_path}")

    if results:
        print(f"\nResults saved to: {runner.output_dir}")

    return {
        "meta": meta,
        "results": [r.to_dict() for r in results],
        "report": report,
        "output_dir": str(runner.output_dir) if results else None,
        "llm_calls": wrapped_client.call_log
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run a CES-grounded Social RL experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with mock client
  python experiments/run_ces_experiment.py --provider mock

  # Run with local Ollama
  python experiments/run_ces_experiment.py --model qwen2.5:7b --rounds 2

  # Run with RunPod vLLM
  python experiments/run_ces_experiment.py \\
    --provider vllm \\
    --base-url https://YOUR_ENDPOINT/v1 \\
    --model Qwen/Qwen2.5-7B-Instruct
"""
    )

    parser.add_argument("--model", "-m", default="qwen2.5:7b", help="Model name")
    parser.add_argument("--provider", "-p", default="ollama", choices=["ollama", "vllm", "mock"])
    parser.add_argument("--rounds", "-r", type=int, default=2, help="Number of rounds")
    parser.add_argument("--max-turns", "-t", type=int, default=12, help="Max turns per round")
    parser.add_argument("--no-dual-llm", action="store_true", help="Disable Dual-LLM")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce verbosity")
    parser.add_argument("--base-url", help="Base URL for vLLM endpoint (single model mode)")
    parser.add_argument("--api-key", help="API key for vLLM")
    parser.add_argument("--experiment-id", help="Custom experiment ID")

    # TRUE dual-LLM arguments
    parser.add_argument("--performer-url", help="Base URL for Performer endpoint (TRUE dual-LLM)")
    parser.add_argument("--performer-model", help="Model name for Performer (e.g., Qwen/Qwen2.5-14B-Instruct)")
    parser.add_argument("--coach-url", help="Base URL for Coach endpoint (TRUE dual-LLM)")
    parser.add_argument("--coach-model", help="Model name for Coach (e.g., Qwen/Qwen2.5-7B-Instruct)")

    # Challenge mode for A/B testing (empirical semiotics)
    parser.add_argument(
        "--challenge-mode",
        choices=["off", "adaptive", "always"],
        default="adaptive",
        help="Challenge cue mode: 'off' (baseline), 'adaptive' (when engagement<0.3), 'always' (A/B test)"
    )

    # Context injection mode for 2x2x2 architecture sweep
    parser.add_argument(
        "--context-mode",
        choices=["static", "progressive", "reactive", "adaptive"],
        default="adaptive",
        help="Context injection mode: 'static', 'progressive', 'reactive', or 'adaptive' (with émile-style existential pressure)"
    )

    # 2x2x2 sweep condition and seed tracking
    parser.add_argument(
        "--condition",
        choices=["A", "B", "C", "D", "E", "F", "G", "H"],
        help="2x2x2 sweep condition label (A-H). See ARCHITECTURE_SWEEP_2x2x2 for definitions."
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed number for replication (e.g., 1, 2, 3)"
    )

    args = parser.parse_args()

    try:
        results = run_ces_experiment(
            model=args.model,
            provider=args.provider,
            rounds=args.rounds,
            max_turns=args.max_turns,
            use_dual_llm=not args.no_dual_llm,
            verbose=not args.quiet,
            base_url=args.base_url,
            api_key=args.api_key,
            experiment_id=args.experiment_id,
            # TRUE dual-LLM parameters
            performer_url=args.performer_url,
            performer_model=args.performer_model,
            coach_url=args.coach_url,
            coach_model=args.coach_model,
            # Challenge mode for A/B testing
            challenge_mode=args.challenge_mode,
            # Context injection mode for 2x2x2 architecture sweep
            context_mode=args.context_mode,
            # 2x2x2 sweep condition and seed tracking
            condition=args.condition,
            seed=args.seed,
        )

        print(f"\nExperiment completed!")
        print(f"Experiment ID: {results['meta']['experiment_id']}")
        print(f"Total LLM calls: {len(results['llm_calls'])}")

    except KeyboardInterrupt:
        print("\nExperiment interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\nExperiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
