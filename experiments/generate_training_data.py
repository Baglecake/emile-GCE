#!/usr/bin/env python3
"""
Generate JSONL training data for LoRA fine-tuning via Unsloth.

Runs batch experiments with émile-gce and extracts grounded responses.
Output format matches Unsloth expectations:
    {"text": "Instruction: <prompt>\n\nResponse: <grounded response>"}

Usage:
    python experiments/generate_training_data.py --runs 25 --rounds 4
    python experiments/generate_training_data.py --extract-only outputs/
    python experiments/generate_training_data.py --sanity-check 10
"""

import argparse
import json
import sys
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# RunPod endpoints from test_models.py
LLAMA_CONFIG = {
    "name": "llama",
    "base_url": "https://coaapc0tyag7h3-8000.proxy.runpod.net/v1",
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "api_key": "sk-1234",
}

MISTRAL_CONFIG = {
    "name": "mistral",
    "base_url": "https://9qrgc461yk73t4-8080.proxy.runpod.net/v1",
    "model": "mistralai/Mistral-Nemo-Instruct-2407",
    "api_key": "sk-1234",
}

# Training topics - varied political discussions for diverse training data
TRAINING_TOPICS = [
    # Housing (key friction point)
    "What do you think about housing density in your area?",
    "Should the government build more affordable housing?",
    "Is foreign investment driving up housing prices?",
    "Do you support rent control?",

    # Economy / Cost of living
    "How has inflation affected your household?",
    "Should minimum wage be raised to $20/hour?",
    "Is the carbon tax hurting or helping your community?",
    "What should the government prioritize: debt reduction or spending?",

    # Healthcare
    "How are healthcare wait times where you live?",
    "Should dental care be covered by the government?",
    "Is healthcare a provincial or federal responsibility?",
    "Would you support private healthcare options?",

    # Immigration
    "Is Canada accepting too many or too few immigrants?",
    "How has immigration changed your community?",
    "Should temporary foreign workers have a path to citizenship?",

    # Climate / Environment
    "Do you support the carbon tax?",
    "Should Canada do more on climate change?",
    "How would transitioning off oil affect your region?",

    # Democracy / Voting
    "Should voting be mandatory in Canada?",
    "Do you feel represented by your current MP?",
    "Should Canada switch to proportional representation?",
    "Why don't more young people vote?",

    # Regional friction
    "Do Western provinces get a fair deal in Confederation?",
    "Is Quebec treated differently than other provinces?",
    "Should Northern Ontario separate from Southern Ontario?",
    "Are rural communities forgotten by politicians?",
]


def get_agent_instruction_context(profile: Dict[str, Any]) -> str:
    """Build instruction context from CES profile."""
    agent_id = profile.get("cps21_ResponseId", "Unknown")

    # Urban/rural
    ur = profile.get("cps21_urban_rural", 2)
    location = {1: "urban", 2: "suburban", 3: "rural"}.get(ur, "suburban")

    # Riding for place specificity
    riding = profile.get("cps21_riding_id", "Unknown")

    # Income/education for class grounding
    income = profile.get("cps21_income_cat", 5)
    edu = profile.get("cps21_education", 5)

    # Party for political orientation
    party_map = {1: "Liberal", 2: "Conservative", 3: "NDP", 4: "Bloc", 5: "Green", 8: "None"}
    party = party_map.get(profile.get("cps21_pid_party", 8), "None")

    # Turnout for engagement level
    turnout = profile.get("cps21_turnout", 2)
    engagement = {1: "engaged voter", 2: "likely voter", 3: "unlikely voter"}.get(turnout, "voter")

    return f"[{agent_id}] {location} {engagement}, {party}-leaning, riding {riding}"


import re

# Patterns to REJECT entirely (AI refusals, pure simulacra)
REJECT_PATTERNS = [
    r"^I cannot",
    r"^As an AI",
    r"^I'm not able to",
    r"^I appreciate your perspective",
    r"^I understand your point, but",
    r"^I want to acknowledge",
]

# Patterns to STRIP from start (bio dumps, intros) - content after may be good
STRIP_PATTERNS = [
    r"^As a \d+[- ]?year[- ]?old[^.]*[.,]\s*",
    r"^I'm a \d+[- ]?year[- ]?old[^.]*[.,]\s*",
    r"^As a fellow \d+[- ]?year[- ]?old[^.]*[.,]\s*",
    r"^As a university-educated[^.]*[.,]\s*",
    r"^I'm a university-educated[^.]*[.,]\s*",
    r"^As CES_[^.]*[.,]\s*",
    r"^Listen up,?\s*I'm CES_[^.]*[.,]\s*",
    r"^I'm CES_[^.]*[.,]\s*",
    r"^Hello everyone[^.]*[.,]\s*",
    r"^Hi everyone[^.]*[.,]\s*",
    r"^My name is[^.]*[.,]\s*",
    r"^I'd like to directly address[^.]*[.,]\s*",
]

COMPILED_REJECTS = [re.compile(p, re.IGNORECASE) for p in REJECT_PATTERNS]
COMPILED_STRIPS = [re.compile(p, re.IGNORECASE) for p in STRIP_PATTERNS]


def should_reject(content: str) -> bool:
    """Check if content should be completely rejected."""
    for pattern in COMPILED_REJECTS:
        if pattern.search(content):
            return True
    return False


def clean_content(content: str) -> tuple[str, bool]:
    """
    Strip bio-dump intros while keeping the substance.
    Returns (cleaned_content, was_modified).
    """
    cleaned = content
    was_modified = False
    for pattern in COMPILED_STRIPS:
        new_cleaned = pattern.sub("", cleaned, count=1)
        if new_cleaned != cleaned:
            was_modified = True
            cleaned = new_cleaned
    # Capitalize first letter if we stripped something
    if was_modified and cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
    return cleaned.strip(), was_modified


def extract_training_pairs(output_dir: Path, filter_simulacra: bool = True) -> List[Dict[str, str]]:
    """
    Extract instruction/response pairs from experiment output.

    Args:
        output_dir: Path to experiment output directory
        filter_simulacra: If True, reject/clean messages matching bad patterns

    Returns list of dicts with 'instruction', 'response', 'agent_context'.
    """
    pairs = []
    rejected_count = 0
    cleaned_count = 0

    # Load CES profiles
    profiles_path = output_dir / "ces_profiles.json"
    if not profiles_path.exists():
        return pairs

    with open(profiles_path) as f:
        profiles = json.load(f)

    profile_map = {p["cps21_ResponseId"]: p for p in profiles}

    # Find round result files (handles both round_N.json and roundN_social_rl.json formats)
    for round_file in sorted(list(output_dir.glob("round_*.json")) + list(output_dir.glob("round*_social_rl.json"))):
        with open(round_file) as f:
            round_data = json.load(f)

        # Get scenario as the instruction base
        scenario = round_data.get("scenario", "Political discussion")

        # Extract messages
        messages = round_data.get("messages", [])
        for msg in messages:
            agent_id = msg.get("agent_id", "")
            content = msg.get("content", "")

            if not content or len(content) < 20:
                continue

            # Filter: reject pure simulacra
            if filter_simulacra and should_reject(content):
                rejected_count += 1
                continue

            # Clean: strip bio-dump intros
            if filter_simulacra:
                content, was_cleaned = clean_content(content)
                if was_cleaned:
                    cleaned_count += 1
                # Skip if cleaning left too little
                if len(content) < 20:
                    rejected_count += 1
                    continue

            # Get profile for this agent
            profile = profile_map.get(agent_id, {})
            agent_context = get_agent_instruction_context(profile)

            # Build instruction - combine scenario with a question format
            # For training, we want: "You are [context]. [Question]"
            instruction = f"You are {agent_context}. {scenario}"

            pairs.append({
                "instruction": instruction,
                "response": content,
                "agent_id": agent_id,
                "agent_context": agent_context,
            })

    if rejected_count > 0 or cleaned_count > 0:
        print(f"  [{output_dir.name}] Rejected: {rejected_count}, Cleaned: {cleaned_count}")

    return pairs


def pairs_to_jsonl(pairs: List[Dict], output_path: Path, agent_filter: Optional[str] = None):
    """
    Convert pairs to Unsloth JSONL format.

    Format: {"text": "Instruction: ...\n\nResponse: ..."}
    """
    with open(output_path, "w") as f:
        for pair in pairs:
            if agent_filter and agent_filter not in pair.get("agent_id", ""):
                continue

            text = f"Instruction: {pair['instruction']}\n\nResponse: {pair['response']}"
            f.write(json.dumps({"text": text}) + "\n")


def run_batch_experiments(
    num_runs: int,
    rounds_per_run: int,
    model_config: Dict,
    output_base: Path,
    verbose: bool = True,
) -> List[Path]:
    """
    Run batch experiments and return output directories.
    """
    from experiments.run_ces_experiment import run_ces_experiment

    output_dirs = []
    topics = TRAINING_TOPICS.copy()
    random.shuffle(topics)

    for i in range(num_runs):
        # Cycle through topics
        topic_idx = i % len(topics)

        experiment_id = f"train_gen_{model_config['name']}_{i:03d}_{datetime.now().strftime('%H%M%S')}"

        if verbose:
            print(f"\n[{i+1}/{num_runs}] Running experiment: {experiment_id}")
            print(f"  Topic: {topics[topic_idx][:50]}...")

        try:
            result = run_ces_experiment(
                model=model_config["model"],
                provider="vllm",
                base_url=model_config["base_url"],
                api_key=model_config["api_key"],
                rounds=rounds_per_run,
                max_turns=12,
                use_dual_llm=False,  # Simpler for training data
                verbose=False,
                experiment_id=experiment_id,
                context_mode="progressive",  # Consistent mode
            )

            if result.get("output_dir"):
                output_dirs.append(Path(result["output_dir"]))
                if verbose:
                    print(f"  Saved to: {result['output_dir']}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    return output_dirs


def sanity_check(pairs: List[Dict], num_samples: int = 10):
    """
    Print sample pairs for manual review (Gemini's "Friction Test").
    """
    print("\n" + "="*60)
    print("SANITY CHECK: Review for 'friction' (not polite simulacra)")
    print("="*60)

    samples = random.sample(pairs, min(num_samples, len(pairs)))

    for i, pair in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Agent: {pair.get('agent_id', 'Unknown')}")
        print(f"Context: {pair.get('agent_context', 'Unknown')}")
        print(f"Response: {pair['response'][:300]}...")

        # Friction indicators
        has_place = any(w in pair['response'].lower() for w in
                       ['here', 'my area', 'our town', 'downtown', 'rural', 'north'])
        has_opinion = any(w in pair['response'].lower() for w in
                         ["don't", "won't", "hate", "love", "ridiculous", "stupid", "great"])
        is_short = len(pair['response'].split()) < 100

        friction_score = sum([has_place, has_opinion, is_short])
        print(f"Friction indicators: place={has_place}, opinion={has_opinion}, concise={is_short}")
        print(f"Friction score: {friction_score}/3 {'(GOOD)' if friction_score >= 2 else '(needs work)'}")


def main():
    parser = argparse.ArgumentParser(description="Generate training data for LoRA fine-tuning")

    parser.add_argument("--runs", type=int, default=25, help="Number of experiments to run")
    parser.add_argument("--rounds", type=int, default=4, help="Rounds per experiment")
    parser.add_argument("--model", choices=["llama", "mistral", "both"], default="both",
                       help="Which model(s) to use for generation")
    parser.add_argument("--extract-only", type=str, help="Extract from existing outputs (path)")
    parser.add_argument("--sanity-check", type=int, help="Run sanity check on N samples")
    parser.add_argument("--output", type=str, default="training_data", help="Output directory name")

    args = parser.parse_args()

    output_base = Path("outputs") / args.output
    output_base.mkdir(parents=True, exist_ok=True)

    all_pairs = []

    # Extract-only mode
    if args.extract_only:
        print(f"Extracting from: {args.extract_only}")
        extract_path = Path(args.extract_only)

        if extract_path.is_dir():
            # Check if it's a single experiment or a parent directory
            if (extract_path / "ces_profiles.json").exists():
                all_pairs.extend(extract_training_pairs(extract_path))
            else:
                # Scan subdirectories
                for subdir in extract_path.iterdir():
                    if subdir.is_dir() and (subdir / "ces_profiles.json").exists():
                        all_pairs.extend(extract_training_pairs(subdir))

        print(f"Extracted {len(all_pairs)} training pairs")

    # Run new experiments
    else:
        configs = []
        if args.model in ["llama", "both"]:
            configs.append(LLAMA_CONFIG)
        if args.model in ["mistral", "both"]:
            configs.append(MISTRAL_CONFIG)

        for config in configs:
            print(f"\n{'='*60}")
            print(f"Generating training data with {config['name'].upper()}")
            print(f"{'='*60}")

            output_dirs = run_batch_experiments(
                num_runs=args.runs,
                rounds_per_run=args.rounds,
                model_config=config,
                output_base=output_base,
            )

            # Extract pairs from all outputs
            for odir in output_dirs:
                all_pairs.extend(extract_training_pairs(odir))

    print(f"\nTotal training pairs: {len(all_pairs)}")

    # Sanity check
    if args.sanity_check and all_pairs:
        sanity_check(all_pairs, args.sanity_check)

    # Save JSONL files
    if all_pairs:
        # All agents combined
        all_jsonl = output_base / "training_all.jsonl"
        pairs_to_jsonl(all_pairs, all_jsonl)
        print(f"\nSaved: {all_jsonl} ({len(all_pairs)} examples)")

        # Per-agent splits
        agent_ids = set(p.get("agent_id", "") for p in all_pairs)
        for agent_id in agent_ids:
            if not agent_id:
                continue
            agent_pairs = [p for p in all_pairs if p.get("agent_id") == agent_id]
            agent_jsonl = output_base / f"training_{agent_id.lower()}.jsonl"
            pairs_to_jsonl(agent_pairs, agent_jsonl)
            print(f"Saved: {agent_jsonl} ({len(agent_pairs)} examples)")

        # Archetype splits for Gemini's recommendation
        # Urban/Progressive → Llama training
        # Rural/Conservative → Mistral training
        urban_pairs = [p for p in all_pairs if "Urban" in p.get("agent_id", "")]
        rural_pairs = [p for p in all_pairs if "Rural" in p.get("agent_id", "")]

        if urban_pairs:
            pairs_to_jsonl(urban_pairs, output_base / "training_urban_progressive.jsonl")
            print(f"Saved: training_urban_progressive.jsonl ({len(urban_pairs)} examples)")

        if rural_pairs:
            pairs_to_jsonl(rural_pairs, output_base / "training_rural_conservative.jsonl")
            print(f"Saved: training_rural_conservative.jsonl ({len(rural_pairs)} examples)")


if __name__ == "__main__":
    main()
