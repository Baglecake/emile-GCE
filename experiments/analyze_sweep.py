#!/usr/bin/env python3
"""
2x2x2 Architecture Sweep Analysis Script

Scans outputs/ directories for CES experiments and prints a compact
regime trajectory table for easy H5 hypothesis checking.

Usage:
    python experiments/analyze_sweep.py [--outputs-dir outputs/]

Output format:
    Cond  Seed  R1                R2                      R3                      div_events
    A     1     ACTIVE_CONTEST    PATERNALISTIC_HARMONY   PATERNALISTIC_HARMONY   0
    G     1     ACTIVE_CONTEST    STIMULATED_DIALOGUE     PRODUCTIVE_DISSONANCE   1
    ...
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict


# Regime name abbreviations for compact display
REGIME_ABBREV = {
    "ACTIVE_CONTESTATION": "ACTIVE",
    "PATERNALISTIC_HARMONY": "PAT_HARM",
    "STIMULATED_DIALOGUE": "STIM_DIA",
    "PROCEDURALIST_RETREAT": "PROC_RET",
    "PRODUCTIVE_DISSONANCE": "PROD_DIS",
    "UNKNOWN": "UNKNOWN",
    None: "-",
}


def abbrev_regime(regime: Optional[str]) -> str:
    """Abbreviate regime name for compact display."""
    return REGIME_ABBREV.get(regime, regime[:8] if regime else "-")


def load_experiment(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """Load meta.json and semiotic_state_log.json from an experiment directory."""
    meta_path = exp_dir / "meta.json"
    semiotic_path = exp_dir / "semiotic_state_log.json"

    if not meta_path.exists():
        return None

    try:
        with open(meta_path) as f:
            meta = json.load(f)

        semiotic_log = None
        if semiotic_path.exists():
            with open(semiotic_path) as f:
                semiotic_log = json.load(f)

        return {
            "dir": exp_dir.name,
            "meta": meta,
            "semiotic_log": semiotic_log,
        }
    except Exception as e:
        print(f"Warning: Failed to load {exp_dir}: {e}")
        return None


def infer_condition(meta: Dict[str, Any]) -> Optional[str]:
    """
    Infer condition label from meta.json if not explicitly set.

    Based on the 2x2x2 design:
        A: challenge=off,  context=progressive, dual=True
        B: challenge=off,  context=progressive, dual=False
        C: challenge=off,  context=adaptive,    dual=True
        D: challenge=off,  context=adaptive,    dual=False
        E: challenge=always, context=progressive, dual=True
        F: challenge=always, context=progressive, dual=False
        G: challenge=always, context=adaptive,    dual=True
        H: challenge=always, context=adaptive,    dual=False
    """
    if meta.get("condition"):
        return meta["condition"]

    challenge = meta.get("challenge_mode", "adaptive")
    context = meta.get("context_mode", "adaptive")
    dual = meta.get("dual_llm", True)

    # Map to condition
    if challenge == "off" and context == "progressive" and dual:
        return "A"
    elif challenge == "off" and context == "progressive" and not dual:
        return "B"
    elif challenge == "off" and context == "adaptive" and dual:
        return "C"
    elif challenge == "off" and context == "adaptive" and not dual:
        return "D"
    elif challenge == "always" and context == "progressive" and dual:
        return "E"
    elif challenge == "always" and context == "progressive" and not dual:
        return "F"
    elif challenge == "always" and context == "adaptive" and dual:
        return "G"
    elif challenge == "always" and context == "adaptive" and not dual:
        return "H"

    return None


def extract_regime_trajectory(exp_data: Dict[str, Any]) -> List[Optional[str]]:
    """Extract regime trajectory (R1, R2, R3) from experiment data."""
    meta = exp_data["meta"]

    # Try to get from meta.json directly
    trajectory = meta.get("regime_trajectory", [])

    # Pad to 3 rounds
    while len(trajectory) < 3:
        trajectory.append(None)

    return trajectory[:3]


def print_trajectory_table(experiments: List[Dict[str, Any]]) -> None:
    """Print the compact regime trajectory table."""
    # Sort by condition, then seed
    def sort_key(exp):
        cond = exp.get("condition") or "Z"
        seed = exp.get("seed") or 0
        return (cond, seed)

    experiments = sorted(experiments, key=sort_key)

    # Header
    print("\n" + "=" * 90)
    print("2x2x2 ARCHITECTURE SWEEP - REGIME TRAJECTORIES")
    print("=" * 90)
    print(f"{'Cond':<5} {'Seed':<5} {'R1':<10} {'R2':<10} {'R3':<10} {'Div':<4} {'Dir':<30}")
    print("-" * 90)

    # Group by condition for summary
    condition_results = defaultdict(list)

    for exp in experiments:
        cond = exp.get("condition") or "?"
        seed = exp.get("seed") or "-"
        trajectory = exp.get("trajectory", [None, None, None])
        div_events = exp.get("divergence_events", 0)
        dir_name = exp.get("dir", "")[:28]

        r1 = abbrev_regime(trajectory[0] if len(trajectory) > 0 else None)
        r2 = abbrev_regime(trajectory[1] if len(trajectory) > 1 else None)
        r3 = abbrev_regime(trajectory[2] if len(trajectory) > 2 else None)

        print(f"{cond:<5} {str(seed):<5} {r1:<10} {r2:<10} {r3:<10} {div_events:<4} {dir_name:<30}")

        condition_results[cond].append({
            "seed": seed,
            "r3": trajectory[2] if len(trajectory) > 2 else None,
            "div": div_events,
        })

    # Summary by condition
    print("\n" + "=" * 90)
    print("CONDITION SUMMARY (R3 regime counts)")
    print("=" * 90)

    for cond in sorted(condition_results.keys()):
        results = condition_results[cond]
        r3_counts = defaultdict(int)
        total_div = 0

        for r in results:
            r3_regime = r["r3"] or "UNKNOWN"
            r3_counts[r3_regime] += 1
            total_div += r["div"]

        regime_str = ", ".join(f"{abbrev_regime(k)}:{v}" for k, v in sorted(r3_counts.items()))
        print(f"  {cond}: n={len(results)} | R3: {regime_str} | div_events: {total_div}")

    # H5 check
    print("\n" + "=" * 90)
    print("H5 HYPOTHESIS CHECK")
    print("=" * 90)

    g_results = condition_results.get("G", [])
    if g_results:
        prod_dis_count = sum(1 for r in g_results if r["r3"] == "PRODUCTIVE_DISSONANCE")
        print(f"  Condition G (ADAPTIVE + challenge ON + dual-LLM):")
        print(f"    {prod_dis_count}/{len(g_results)} seeds achieved PRODUCTIVE_DISSONANCE in R3")

        if prod_dis_count > len(g_results) / 2:
            print("    -> H5 SUPPORTED (majority hit target regime)")
        elif prod_dis_count > 0:
            print("    -> H5 PARTIAL SUPPORT (some seeds hit target)")
        else:
            print("    -> H5 NOT SUPPORTED (no seeds hit target regime)")
    else:
        print("  No Condition G runs found yet. Run with --condition G --seed 1")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze 2x2x2 architecture sweep results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Condition Definitions:
    A: challenge=off,    context=progressive, dual=True
    B: challenge=off,    context=progressive, dual=False
    C: challenge=off,    context=adaptive,    dual=True
    D: challenge=off,    context=adaptive,    dual=False
    E: challenge=always, context=progressive, dual=True
    F: challenge=always, context=progressive, dual=False
    G: challenge=always, context=adaptive,    dual=True   (TARGET for H5)
    H: challenge=always, context=adaptive,    dual=False
"""
    )

    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory containing experiment outputs (default: outputs/)"
    )
    parser.add_argument(
        "--filter-condition",
        choices=["A", "B", "C", "D", "E", "F", "G", "H"],
        help="Only show results for a specific condition"
    )

    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        print(f"Error: Outputs directory not found: {outputs_dir}")
        return

    # Scan all subdirectories
    experiments = []

    for exp_dir in sorted(outputs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        exp_data = load_experiment(exp_dir)
        if not exp_data:
            continue

        meta = exp_data["meta"]

        # Infer condition if not set
        condition = infer_condition(meta)
        seed = meta.get("seed")

        # Extract trajectory
        trajectory = extract_regime_trajectory(exp_data)

        # Get divergence events
        div_events = meta.get("divergence_events", 0)

        exp_entry = {
            "dir": exp_dir.name,
            "condition": condition,
            "seed": seed,
            "trajectory": trajectory,
            "divergence_events": div_events,
            "meta": meta,
        }

        # Filter if requested
        if args.filter_condition:
            if condition != args.filter_condition:
                continue

        experiments.append(exp_entry)

    if not experiments:
        print("No experiments found in outputs directory.")
        print("\nTo run a condition, use:")
        print("  python experiments/run_ces_experiment.py --condition G --seed 1 ...")
        return

    print_trajectory_table(experiments)


if __name__ == "__main__":
    main()
