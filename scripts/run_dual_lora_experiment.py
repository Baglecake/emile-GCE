#!/usr/bin/env python3
"""
Run émile-GCE experiment with dual LoRA-tuned models on RunPod.

Usage:
    python scripts/run_dual_lora_experiment.py \
        --performer-url http://195.26.233.30:8000/v1 \
        --coach-url http://69.30.85.15:8000/v1
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from social_rl.dual_llm_client import create_true_dual_llm


def test_connection(dual_client):
    """Quick test to verify both endpoints are working."""
    print("Testing Performer (Llama + LoRA)...")
    try:
        response = dual_client.generate(
            system_prompt="You are a Canadian voter from Ontario.",
            user_message="In one sentence, what matters most to you in an election?",
            mode="performer"
        )
        print(f"  Performer: {response[:100]}...")
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    print("\nTesting Coach (Mistral + LoRA)...")
    try:
        response = dual_client.generate(
            system_prompt="You are a validation coach.",
            user_message="Is this response appropriate: 'I care about healthcare.'",
            mode="coach"
        )
        print(f"  Coach: {response[:100]}...")
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    return True


def run_mini_experiment(dual_client):
    """Run a minimal 2-agent, 2-round experiment to test the setup."""
    print("\n" + "="*60)
    print("MINI EXPERIMENT: 2 agents, 2 rounds")
    print("="*60)

    # Simple agent configs
    agents = [
        {
            "id": "Urban_Progressive",
            "system": "You are [CES_Urban_Progressive], an urban engaged voter from Toronto, NDP-leaning. You care about housing affordability and transit. Be direct and opinionated.",
        },
        {
            "id": "Rural_Conservative",
            "system": "You are [CES_Rural_Conservative], a rural voter from Northern Ontario, Conservative-leaning. You care about jobs and cost of living. Be practical and no-nonsense.",
        }
    ]

    topic = "Should the government prioritize building more housing in cities or improving infrastructure in rural areas?"

    conversation = []

    for round_num in range(1, 3):
        print(f"\n--- Round {round_num} ---")
        for agent in agents:
            # Build context from conversation history
            history = "\n".join([f"{m['agent']}: {m['content']}" for m in conversation[-4:]])

            user_msg = f"""Topic: {topic}

Previous discussion:
{history if history else "(Opening the discussion)"}

Share your perspective. Reference what others said if relevant. Keep it under 80 words."""

            # Generate with validation
            result = dual_client.generate_validated(
                system_prompt=agent["system"],
                user_message=user_msg,
                agent_id=agent["id"],
                rules=[
                    "Stay in character as a Canadian voter",
                    "Reference previous speakers when building on their points",
                    "Express genuine opinions, not neutral summaries"
                ],
                turn_number=len(conversation) + 1
            )

            conversation.append({
                "agent": agent["id"],
                "content": result.content,
                "validated": result.validation_passed
            })

            status = "✓" if result.validation_passed else f"✗ (retries: {result.retries})"
            print(f"\n[{agent['id']}] {status}")
            print(f"  {result.content[:200]}{'...' if len(result.content) > 200 else ''}")

    print("\n" + "="*60)
    print("Experiment complete!")
    print(f"Total messages: {len(conversation)}")
    print(f"Validated: {sum(1 for m in conversation if m['validated'])}/{len(conversation)}")


def main():
    parser = argparse.ArgumentParser(description="Run émile-GCE with dual LoRA models")
    parser.add_argument("--performer-url", required=True, help="vLLM endpoint for Performer (14B pod)")
    parser.add_argument("--coach-url", required=True, help="vLLM endpoint for Coach (7B pod)")
    parser.add_argument("--performer-model", default="emile-llama", help="LoRA adapter name for Performer")
    parser.add_argument("--coach-model", default="emile-mistral", help="LoRA adapter name for Coach")
    parser.add_argument("--api-key", default="sk-1234", help="API key for vLLM endpoints")
    parser.add_argument("--test-only", action="store_true", help="Only test connection, don't run experiment")
    args = parser.parse_args()

    print("Connecting to RunPod endpoints...")
    print(f"  Performer: {args.performer_url} (model: {args.performer_model})")
    print(f"  Coach: {args.coach_url} (model: {args.coach_model})")

    dual = create_true_dual_llm(
        performer_base_url=args.performer_url,
        performer_model=args.performer_model,
        coach_base_url=args.coach_url,
        coach_model=args.coach_model,
        performer_temp=0.8,  # Slightly higher for more variety
        coach_temp=0.1,      # Low for consistent validation
        api_key=args.api_key,
    )

    if not test_connection(dual):
        print("\nConnection test failed. Check your endpoints.")
        sys.exit(1)

    if args.test_only:
        print("\nConnection test passed!")
        sys.exit(0)

    run_mini_experiment(dual)


if __name__ == "__main__":
    main()
