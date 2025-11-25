#!/usr/bin/env python3
"""
Extract Empirical Identity Vectors from Simulation Logs

Based on Gemini's vector extraction methodology (notes/vector_ideas_and_issues).
Computes identity vectors from observed behavior rather than static CES profiles.

Supports two modes:
  - Per-experiment: Aggregate vectors across all rounds (original behavior)
  - Per-round: Extract vectors for each round, enabling ΔI, coherence, natality

Usage:
    # Per-experiment mode (default)
    python3 extract_identity_vectors.py outputs/G_seed2_fixed

    # Per-round mode (Phase 2a)
    python3 extract_identity_vectors.py outputs/G_seed2_fixed --per-round

    # Comparison mode
    python3 extract_identity_vectors.py outputs/G2 outputs/G6 [output.json]
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


# Critical concepts for institutional_faith computation
CRITICAL_CONCEPTS = {'Alienation', 'Domination', 'Exploitation',
                     'Oppression', 'Coercion', 'Powerlessness'}


def load_round_logs(run_dir: Path) -> List[Dict[str, Any]]:
    """Load all round logs from experiment directory."""
    rounds = []
    for i in range(1, 4):  # Rounds 1-3
        round_file = run_dir / f"round{i}_social_rl.json"
        if round_file.exists():
            with open(round_file) as f:
                rounds.append(json.load(f))
    return rounds


# =============================================================================
# Per-Round Vector Extraction (Phase 2a)
# =============================================================================

def extract_single_round_vector(agent_id: str, round_data: Dict) -> Dict[str, float]:
    """
    Extract identity vector for a single agent in a single round.

    Returns vector with: engagement, institutional_faith, social_friction
    """
    feedback = round_data.get('feedback', {}).get(agent_id, {})

    # Engagement (direct from feedback)
    engagement = feedback.get('engagement', 0.0)

    # Institutional Faith from concepts
    concepts = feedback.get('concepts_embodied', [])
    crit_count = sum(1 for c in concepts if c in CRITICAL_CONCEPTS)
    total = len(concepts) if concepts else 1
    faith = 1.0 - (crit_count / total)

    # Social Friction from direct references
    friction = feedback.get('direct_references', 0)

    return {
        'engagement': round(engagement, 4),
        'institutional_faith': round(faith, 4),
        'social_friction': round(float(friction), 4),
    }


def vector_to_array(vec: Dict[str, float]) -> np.ndarray:
    """Convert identity vector dict to numpy array for math operations."""
    return np.array([vec['engagement'], vec['institutional_faith'], vec['social_friction']])


def compute_delta_I(v0: Dict[str, float], vt: Dict[str, float]) -> float:
    """Compute magnitude of identity change: |I_t - I_0|"""
    arr0 = vector_to_array(v0)
    arrt = vector_to_array(vt)
    return float(np.linalg.norm(arrt - arr0))


def compute_coherence_cos(v0: Dict[str, float], vt: Dict[str, float]) -> float:
    """
    Compute directional coherence: cos(I_t, I_0)

    High (→1): Identity direction preserved
    Low (→0): Identity has rotated significantly
    """
    arr0 = vector_to_array(v0)
    arrt = vector_to_array(vt)

    norm0 = np.linalg.norm(arr0)
    normt = np.linalg.norm(arrt)

    if norm0 < 1e-8 or normt < 1e-8:
        return 1.0  # No movement = coherent

    return float(np.dot(arr0, arrt) / (norm0 * normt))


def extract_per_round_vectors(run_dir: Path) -> Dict[str, Any]:
    """
    Extract identity vectors per-round for all agents.

    Returns structure supporting:
      - Per-round identity vectors (I_t for t in [R1, R2, R3])
      - ΔI trajectories (cumulative change from R1)
      - Coherence trajectories (cos similarity to initial)
      - Natality z-score preparation (μ, σ of ΔI)

    Output format designed for IdentityCore integration (Phase 2a).
    """
    rounds = load_round_logs(run_dir)
    if not rounds:
        raise ValueError(f"No round logs found in {run_dir}")

    agent_ids = list(rounds[0].get('feedback', {}).keys())

    # Load metadata
    meta_file = run_dir / "meta.json"
    meta = json.load(open(meta_file)) if meta_file.exists() else {}

    # Default temporal compression (configurable in Phase 2b)
    temporal_config = {
        'years_per_experiment': 4,
        'years_per_round': {'R1': 1.5, 'R2': 1.5, 'R3': 1.0}
    }

    agents_data = {}
    for agent_id in agent_ids:
        # Extract per-round vectors
        round_vectors = []
        for i, round_data in enumerate(rounds):
            vec = extract_single_round_vector(agent_id, round_data)
            round_vectors.append({
                'round': i + 1,
                'sim_time': sum(temporal_config['years_per_round'][f'R{j+1}']
                               for j in range(i)),  # cumulative time
                'vector': vec
            })

        # Compute trajectories
        v0 = round_vectors[0]['vector']
        trajectory = []
        delta_history = []

        for rv in round_vectors:
            vt = rv['vector']
            delta_I = compute_delta_I(v0, vt)
            coherence = compute_coherence_cos(v0, vt)

            trajectory.append({
                'round': rv['round'],
                'sim_time': rv['sim_time'],
                'vector': vt,
                'delta_I': round(delta_I, 4),
                'coherence_cos': round(coherence, 4),
            })

            if rv['round'] > 1:
                # ΔI from previous round (for natality z-score)
                v_prev = round_vectors[rv['round'] - 2]['vector']
                delta_from_prev = compute_delta_I(v_prev, vt)
                delta_history.append(delta_from_prev)

        # Compute natality preparation stats (μ, σ of round-to-round ΔI)
        if delta_history:
            delta_mu = float(np.mean(delta_history))
            delta_sigma = float(np.std(delta_history)) if len(delta_history) > 1 else 0.1
        else:
            delta_mu, delta_sigma = 0.0, 0.1

        agents_data[agent_id] = {
            'agent_id': agent_id,
            'group_id': infer_group_id(agent_id),  # CES strata mapping
            'initial_vector': v0,
            'final_vector': round_vectors[-1]['vector'],
            'trajectory': trajectory,
            # Phase 2b hooks (placeholders)
            'empirical_delta_mu': None,  # To be filled from multi-wave CES
            'empirical_delta_sigma': None,
            # Natality prep from this sim
            'observed_delta_mu': round(delta_mu, 4),
            'observed_delta_sigma': round(delta_sigma, 4),
        }

    return {
        'experiment_id': meta.get('experiment_id', run_dir.name),
        'condition': meta.get('condition', 'unknown'),
        'seed': meta.get('seed', 'unknown'),
        'mode': 'per_round',
        'temporal_config': temporal_config,
        'agents': agents_data,
    }


def infer_group_id(agent_id: str) -> str:
    """
    Infer CES strata group_id from agent_id.

    Maps agent names to sociogeographic strata for Phase 2b CES calibration.
    Example: 'CES_Urban_Renter_Progressive' -> 'urban_renter_left'
    """
    aid_lower = agent_id.lower()

    # Location
    if 'urban' in aid_lower:
        location = 'urban'
    elif 'rural' in aid_lower:
        location = 'rural'
    elif 'suburban' in aid_lower:
        location = 'suburban'
    else:
        location = 'unknown'

    # Tenure
    if 'renter' in aid_lower:
        tenure = 'renter'
    elif 'owner' in aid_lower or 'homeowner' in aid_lower:
        tenure = 'owner'
    else:
        tenure = 'unknown'

    # Political lean
    if 'progressive' in aid_lower or 'liberal' in aid_lower or 'left' in aid_lower:
        lean = 'left'
    elif 'conservative' in aid_lower or 'right' in aid_lower:
        lean = 'right'
    elif 'moderate' in aid_lower or 'centrist' in aid_lower:
        lean = 'center'
    else:
        lean = 'unknown'

    # Engagement level
    if 'disengaged' in aid_lower:
        engagement = 'disengaged'
    elif 'engaged' in aid_lower or 'active' in aid_lower:
        engagement = 'engaged'
    else:
        engagement = 'unknown'

    return f"{location}_{tenure}_{lean}_{engagement}".replace('_unknown', '')


def print_per_round_summary(data: Dict[str, Any]) -> None:
    """Print human-readable summary of per-round extraction."""
    print("=" * 80)
    print(f"PER-ROUND IDENTITY VECTOR EXTRACTION")
    print(f"Experiment: {data['experiment_id']}")
    print(f"Condition: {data['condition']}, Seed: {data['seed']}")
    print("=" * 80)

    for agent_id, agent_data in data['agents'].items():
        short_id = agent_id.replace('CES_CES_', '').replace('CES_', '')
        print(f"\n{short_id}")
        print(f"  Group ID: {agent_data['group_id']}")
        print(f"  Initial: {agent_data['initial_vector']}")
        print(f"  Final:   {agent_data['final_vector']}")
        print(f"  Observed ΔI stats: μ={agent_data['observed_delta_mu']:.4f}, σ={agent_data['observed_delta_sigma']:.4f}")
        print("  Trajectory:")
        for t in agent_data['trajectory']:
            print(f"    R{t['round']} (t={t['sim_time']:.1f}y): "
                  f"ΔI={t['delta_I']:.3f}, cos={t['coherence_cos']:.3f}")


# =============================================================================
# Per-Experiment Vector Extraction (Original)
# =============================================================================

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
    crit_count = sum(1 for c in concepts_embodied if c in CRITICAL_CONCEPTS)
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

    # Check for --per-round flag
    per_round_mode = '--per-round' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    if not args:
        print(__doc__)
        sys.exit(1)

    run_dir = Path(args[0])

    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist")
        sys.exit(1)

    # Per-round mode (Phase 2a)
    if per_round_mode:
        data = extract_per_round_vectors(run_dir)
        if '--json' in sys.argv:
            print(json.dumps(data, indent=2))
        else:
            print_per_round_summary(data)
        # Optionally save to file
        if len(args) > 1:
            output_file = Path(args[1])
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\n[SAVED] {output_file}")

    # Single experiment mode (original)
    elif len(args) == 1:
        vectors = extract_all_vectors(run_dir)
        print(json.dumps(vectors, indent=2))

    # Comparison mode
    elif len(args) >= 2:
        run_dir2 = Path(args[1])
        output_file = Path(args[2]) if len(args) > 2 else None
        compare_experiments(run_dir, run_dir2, output_file)


if __name__ == "__main__":
    main()
