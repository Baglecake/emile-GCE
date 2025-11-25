#!/usr/bin/env python3
"""
Analyze Disengaged Renter engagement across all 2×2×2 sweep conditions.
Identifies which architectural components drive hyper-enfranchisement.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# 2×2×2 Architecture Sweep Conditions
CONDITIONS = {
    'A': {'challenge': 'off', 'context': 'progressive', 'dual': True},
    'B': {'challenge': 'off', 'context': 'progressive', 'dual': False},
    'C': {'challenge': 'off', 'context': 'adaptive', 'dual': True},
    'D': {'challenge': 'off', 'context': 'adaptive', 'dual': False},
    'E': {'challenge': 'always', 'context': 'progressive', 'dual': True},
    'F': {'challenge': 'always', 'context': 'progressive', 'dual': False},
    'G': {'challenge': 'always', 'context': 'adaptive', 'dual': True},
    'H': {'challenge': 'always', 'context': 'adaptive', 'dual': False},
}


def extract_disengaged_renter(exp_dir: Path) -> Dict:
    """Extract Disengaged Renter metrics from experiment directory."""
    import subprocess
    import json

    result = subprocess.run(
        ['python3', 'extract_identity_vectors.py', str(exp_dir)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return None

    data = json.loads(result.stdout)
    agent_data = data['agents'].get('CES_CES_Disengaged_Renter', {})

    return {
        'condition': data.get('condition', 'unknown'),
        'seed': data.get('seed', 'unknown'),
        'engagement': agent_data.get('identity_vectors', {}).get('engagement', 0.0),
        'trajectory': agent_data.get('behavioral_metrics', {}).get('engagement_by_round', []),
        'msg_length': agent_data.get('behavioral_metrics', {}).get('avg_message_length', 0.0),
    }


def main():
    print("=" * 80)
    print("DISENGAGED RENTER: FULL SWEEP ANALYSIS")
    print("=" * 80)
    print("\nExpected CES baseline: ~0.17")
    print("\n" + "=" * 80)

    results = {}

    # Extract all conditions, seeds 2-3
    for cond in CONDITIONS.keys():
        results[cond] = {}
        for seed in [2, 3]:
            exp_dir = Path(f'outputs/{cond}_seed{seed}_fixed')
            if exp_dir.exists():
                data = extract_disengaged_renter(exp_dir)
                if data:
                    results[cond][seed] = data

    # Print by condition
    print("\n## BY CONDITION (Seeds 2-3)\n")
    print(f"{'Cond':<6} {'Seed':<6} {'Engagement':<12} {'Trajectory':<25} {'Config'}")
    print("-" * 90)

    for cond in sorted(CONDITIONS.keys()):
        config = CONDITIONS[cond]
        config_str = f"ch={config['challenge']:<7} ctx={config['context']:<11} dual={config['dual']}"

        for seed in [2, 3]:
            if seed in results.get(cond, {}):
                r = results[cond][seed]
                traj_str = str(r['trajectory'])
                print(f"{cond:<6} {seed:<6} {r['engagement']:<12.3f} {traj_str:<25} {config_str}")

    # Calculate averages per condition
    print("\n" + "=" * 80)
    print("\n## AVERAGES BY CONDITION (across seeds 2-3)\n")
    print(f"{'Cond':<6} {'Avg Eng':<10} {'Deviation':<12} {'Config'}")
    print("-" * 80)

    averages = []
    for cond in sorted(CONDITIONS.keys()):
        if cond in results and results[cond]:
            engagements = [results[cond][seed]['engagement'] for seed in results[cond].keys()]
            avg_eng = sum(engagements) / len(engagements)
            deviation = avg_eng - 0.17  # vs CES expected

            config = CONDITIONS[cond]
            config_str = f"ch={config['challenge']:<7} ctx={config['context']:<11} dual={config['dual']}"

            print(f"{cond:<6} {avg_eng:<10.3f} {deviation:+.3f} ({deviation/0.17*100:+.0f}%)  {config_str}")
            averages.append((cond, avg_eng, config))

    # Rank by closeness to CES expected (0.17)
    print("\n" + "=" * 80)
    print("\n## RANKED BY CLOSENESS TO CES EXPECTED (0.17)\n")
    print(f"{'Rank':<6} {'Cond':<6} {'Avg Eng':<10} {'Distance':<12} {'Config'}")
    print("-" * 80)

    sorted_by_distance = sorted(averages, key=lambda x: abs(x[1] - 0.17))
    for rank, (cond, avg_eng, config) in enumerate(sorted_by_distance, 1):
        distance = abs(avg_eng - 0.17)
        config_str = f"ch={config['challenge']:<7} ctx={config['context']:<11} dual={config['dual']}"
        print(f"{rank:<6} {cond:<6} {avg_eng:<10.3f} {distance:<12.3f} {config_str}")

    # Factorial Analysis
    print("\n" + "=" * 80)
    print("\n## FACTORIAL ANALYSIS: Which components help?\n")

    # Get averages
    avg_dict = {cond: avg_eng for cond, avg_eng, _ in averages}

    if all(c in avg_dict for c in ['A', 'B', 'E', 'F']):
        # Effect of dual-LLM (progressive context)
        dual_effect_prog = (avg_dict['A'] + avg_dict['E']) / 2 - (avg_dict['B'] + avg_dict['F']) / 2
        print(f"Dual-LLM effect (progressive): {dual_effect_prog:+.3f}")
        print(f"  Dual (A+E)/2 = {(avg_dict['A'] + avg_dict['E']) / 2:.3f}")
        print(f"  Single (B+F)/2 = {(avg_dict['B'] + avg_dict['F']) / 2:.3f}")

    if all(c in avg_dict for c in ['C', 'D', 'G', 'H']):
        # Effect of dual-LLM (adaptive context)
        dual_effect_adap = (avg_dict['C'] + avg_dict['G']) / 2 - (avg_dict['D'] + avg_dict['H']) / 2
        print(f"\nDual-LLM effect (adaptive): {dual_effect_adap:+.3f}")
        print(f"  Dual (C+G)/2 = {(avg_dict['C'] + avg_dict['G']) / 2:.3f}")
        print(f"  Single (D+H)/2 = {(avg_dict['D'] + avg_dict['H']) / 2:.3f}")

    if all(c in avg_dict for c in ['A', 'C', 'E', 'G']):
        # Effect of adaptive context (dual-LLM)
        context_effect_dual = (avg_dict['C'] + avg_dict['G']) / 2 - (avg_dict['A'] + avg_dict['E']) / 2
        print(f"\nAdaptive context effect (dual): {context_effect_dual:+.3f}")
        print(f"  Adaptive (C+G)/2 = {(avg_dict['C'] + avg_dict['G']) / 2:.3f}")
        print(f"  Progressive (A+E)/2 = {(avg_dict['A'] + avg_dict['E']) / 2:.3f}")

    if all(c in avg_dict for c in ['A', 'C', 'E', 'G']):
        # Effect of challenge mode (dual-LLM)
        challenge_effect_dual = (avg_dict['E'] + avg_dict['G']) / 2 - (avg_dict['A'] + avg_dict['C']) / 2
        print(f"\nChallenge mode effect (dual): {challenge_effect_dual:+.3f}")
        print(f"  Challenge=always (E+G)/2 = {(avg_dict['E'] + avg_dict['G']) / 2:.3f}")
        print(f"  Challenge=off (A+C)/2 = {(avg_dict['A'] + avg_dict['C']) / 2:.3f}")

    print("\n" + "=" * 80)
    print("\n## KEY FINDINGS\n")

    best_cond, best_eng, best_config = sorted_by_distance[0]
    worst_cond, worst_eng, worst_config = sorted_by_distance[-1]

    print(f"✅ BEST: Condition {best_cond} (avg={best_eng:.3f}, distance={abs(best_eng-0.17):.3f})")
    print(f"   Config: {best_config}")

    print(f"\n❌ WORST: Condition {worst_cond} (avg={worst_eng:.3f}, distance={abs(worst_eng-0.17):.3f})")
    print(f"   Config: {worst_config}")

    if 'G' in avg_dict:
        g_rank = [i for i, (c, _, _) in enumerate(sorted_by_distance, 1) if c == 'G'][0]
        print(f"\n📊 Condition G (ENGAGED_HARMONY): Rank #{g_rank}/8")
        print(f"   Average engagement: {avg_dict['G']:.3f}")
        print(f"   Deviation from CES: {(avg_dict['G'] - 0.17):+.3f} ({(avg_dict['G'] - 0.17)/0.17*100:+.0f}%)")


if __name__ == '__main__':
    main()
