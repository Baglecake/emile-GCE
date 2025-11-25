#!/usr/bin/env python3
"""
Extract Empirical Identity Vectors from Simulation Logs

Based on Gemini's vector extraction methodology (notes/vector_ideas_and_issues).
Computes identity vectors from observed behavior rather than static CES profiles.

Usage:
    python3 extract_identity_vectors.py outputs/G_seed2_fixed
    python3 extract_identity_vectors.py outputs/ces_experiment_2025-11-24_163237
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np


def load_round_logs(run_dir: Path) -> List[Dict[str, Any]]:
    """Load all round logs from experiment directory."""
    rounds = []
    for i in range(1, 4):  # Rounds 1-3
        round_file = run_dir / f"round{i}_social_rl.json"
        if round_file.exists():
            with open(round_file) as f:
                rounds.append(json.load(f))
    return rounds


def extract_vectors_for_agent(agent_id: str, rounds: List[Dict]) -> Dict[str, Any]:
    """
    Extract empirical identity vectors for a single agent.

    Vectors:
        - engagement: Average engagement across rounds (network centrality)
        - institutional_faith: 1.0 - (critical_concepts / total_concepts)
        - social_friction: Direct references + voice markers
        - message_length: Average message length (prose style)
    """
    engagement_scores = []
    concepts_embodied = []
    direct_refs = []
    message_lengths = []

    for round_data in rounds:
        # Extract feedback metrics
        feedback = round_data.get('feedback', {}).get(agent_id, {})
        engagement_scores.append(feedback.get('engagement', 0.0))

        # Extract concepts from feedback
        concepts = feedback.get('concepts_embodied', [])
        concepts_embodied.extend(concepts)

        # Extract message data
        messages = [m for m in round_data.get('messages', []) if m['agent_id'] == agent_id]
        for msg in messages:
            message_lengths.append(len(msg.get('content', '')))
            # Count if this message was referenced (would need full analysis)

        # Get direct references from feedback
        direct_refs.append(feedback.get('direct_references', 0))

    # Calculate vectors
    vec_engagement = np.mean(engagement_scores) if engagement_scores else 0.0

    # Institutional Faith: 1.0 - (critical_concepts / total)
    critical_concepts = {'Alienation', 'Domination', 'Exploitation',
                        'Oppression', 'Coercion', 'Powerlessness'}
    crit_count = sum(1 for c in concepts_embodied if c in critical_concepts)
    total_concepts = len(concepts_embodied) if concepts_embodied else 1
    vec_faith = 1.0 - (crit_count / total_concepts)

    # Social Friction: average direct references
    vec_friction = np.mean(direct_refs) if direct_refs else 0.0

    # Message length (prose style - not in original vectors but useful)
    avg_msg_length = np.mean(message_lengths) if message_lengths else 0.0

    return {
        "agent_id": agent_id,
        "identity_vectors": {
            "engagement": round(vec_engagement, 3),
            "institutional_faith": round(vec_faith, 3),
            "social_friction": round(vec_friction, 3),
        },
        "behavioral_metrics": {
            "avg_message_length": round(avg_msg_length, 1),
            "total_messages": len(message_lengths),
            "concepts_embodied": concepts_embodied,
            "engagement_by_round": [round(e, 3) for e in engagement_scores],
        }
    }


def extract_all_vectors(run_dir: Path) -> Dict[str, Any]:
    """Extract vectors for all agents in an experiment."""
    rounds = load_round_logs(run_dir)

    if not rounds:
        raise ValueError(f"No round logs found in {run_dir}")

    # Get all agent IDs from first round feedback
    agent_ids = list(rounds[0].get('feedback', {}).keys())

    vectors = {}
    for agent_id in agent_ids:
        vectors[agent_id] = extract_vectors_for_agent(agent_id, rounds)

    # Add metadata
    meta_file = run_dir / "meta.json"
    meta = {}
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)

    return {
        "experiment_id": meta.get("experiment_id", run_dir.name),
        "condition": meta.get("condition", "unknown"),
        "seed": meta.get("seed", "unknown"),
        "agents": vectors
    }


def compare_experiments(exp1_dir: Path, exp2_dir: Path, output_file: Path = None):
    """
    Compare vectors between two experiments (e.g., G2 vs G6).

    Shows Prior (CES) vs Posterior (observed) comparison.
    """
    print("=" * 80)
    print("IDENTITY VECTOR COMPARISON")
    print("=" * 80)

    exp1 = extract_all_vectors(exp1_dir)
    exp2 = extract_all_vectors(exp2_dir)

    print(f"\nExperiment 1: {exp1['experiment_id']}")
    print(f"  Condition: {exp1['condition']}, Seed: {exp1['seed']}")

    print(f"\nExperiment 2: {exp2['experiment_id']}")
    print(f"  Condition: {exp2['condition']}, Seed: {exp2['seed']}")

    # Focus on Disengaged Renter comparison
    print("\n" + "=" * 80)
    print("DISENGAGED RENTER: VECTOR GAP ANALYSIS")
    print("=" * 80)

    for agent_id in exp1['agents'].keys():
        if 'Disengaged' in agent_id or 'Renter' in agent_id:
            v1 = exp1['agents'][agent_id]
            v2 = exp2['agents'][agent_id]

            print(f"\nAgent: {agent_id}")
            print(f"\n{'Metric':<25} {'Exp1 (no grit)':<20} {'Exp2 (with grit)':<20} {'Delta'}")
            print("-" * 80)

            # Engagement
            e1 = v1['identity_vectors']['engagement']
            e2 = v2['identity_vectors']['engagement']
            print(f"{'Engagement':<25} {e1:<20.3f} {e2:<20.3f} {e2-e1:+.3f}")

            # Faith
            f1 = v1['identity_vectors']['institutional_faith']
            f2 = v2['identity_vectors']['institutional_faith']
            print(f"{'Institutional Faith':<25} {f1:<20.3f} {f2:<20.3f} {f2-f1:+.3f}")

            # Message length (behavioral)
            l1 = v1['behavioral_metrics']['avg_message_length']
            l2 = v2['behavioral_metrics']['avg_message_length']
            print(f"{'Avg Message Length':<25} {l1:<20.1f} {l2:<20.1f} {l2-l1:+.1f}")

            print(f"\nEngagement by round:")
            print(f"  Exp1: {v1['behavioral_metrics']['engagement_by_round']}")
            print(f"  Exp2: {v2['behavioral_metrics']['engagement_by_round']}")

    # Print all agents summary
    print("\n" + "=" * 80)
    print("ALL AGENTS SUMMARY")
    print("=" * 80)

    for agent_id in exp1['agents'].keys():
        v1 = exp1['agents'][agent_id]
        v2 = exp2['agents'][agent_id]

        short_id = agent_id.replace('CES_CES_', '').replace('CES_', '')
        e1 = v1['identity_vectors']['engagement']
        e2 = v2['identity_vectors']['engagement']
        l1 = v1['behavioral_metrics']['avg_message_length']
        l2 = v2['behavioral_metrics']['avg_message_length']

        print(f"\n{short_id}:")
        print(f"  Engagement:    {e1:.3f} → {e2:.3f} ({e2-e1:+.3f})")
        print(f"  Message Length: {l1:.0f} → {l2:.0f} ({l2-l1:+.0f} chars)")

    # Save full results
    if output_file:
        results = {
            "experiment_1": exp1,
            "experiment_2": exp2
        }
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[SAVED] {output_file}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    run_dir = Path(sys.argv[1])

    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist")
        sys.exit(1)

    # Single experiment mode
    if len(sys.argv) == 2:
        vectors = extract_all_vectors(run_dir)
        print(json.dumps(vectors, indent=2))

    # Comparison mode
    elif len(sys.argv) >= 3:
        run_dir2 = Path(sys.argv[2])
        output_file = Path(sys.argv[3]) if len(sys.argv) > 3 else None
        compare_experiments(run_dir, run_dir2, output_file)


if __name__ == "__main__":
    main()
