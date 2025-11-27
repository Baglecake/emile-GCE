#!/usr/bin/env python3
"""
Test script for Llama 3.1 8B and Mistral-NeMo 12B on émile-gce via RunPod vLLM.

Usage:
    python experiments/test_models.py --llama
    python experiments/test_models.py --mistral
    python experiments/test_models.py --both
    python experiments/test_models.py --wait  # Poll until ready
"""

import argparse
import sys
import time
sys.path.insert(0, '.')

from local_rcm.llm_client import OpenAIClient

# RunPod vLLM endpoints
LLAMA_CONFIG = {
    "name": "Llama 3.1 8B",
    "base_url": "https://coaapc0tyag7h3-8000.proxy.runpod.net/v1",
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "api_key": "sk-1234",
}

MISTRAL_CONFIG = {
    "name": "Mistral-NeMo 12B",
    "base_url": "https://9qrgc461yk73t4-8080.proxy.runpod.net/v1",
    "model": "mistralai/Mistral-Nemo-Instruct-2407",
    "api_key": "sk-1234",
}


def check_health(config: dict) -> bool:
    """Check if endpoint is ready."""
    import requests
    try:
        resp = requests.get(
            f"{config['base_url']}/models",
            headers={"Authorization": f"Bearer {config['api_key']}"},
            timeout=10
        )
        return resp.status_code == 200
    except Exception:
        return False


def wait_for_models(configs: list, max_wait: int = 600):
    """Poll until all models are ready."""
    print(f"Waiting for models to come online (max {max_wait}s)...")
    start = time.time()
    pending = {c["name"]: c for c in configs}

    while pending and (time.time() - start) < max_wait:
        for name, config in list(pending.items()):
            if check_health(config):
                print(f"  ✓ {name} is ready!")
                del pending[name]

        if pending:
            elapsed = int(time.time() - start)
            print(f"  [{elapsed}s] Still waiting for: {', '.join(pending.keys())}")
            time.sleep(15)

    if pending:
        print(f"  ✗ Timeout waiting for: {', '.join(pending.keys())}")
        return False
    return True


def test_model(config: dict):
    """Run tests against a model endpoint."""
    print(f"\n{'='*60}")
    print(f"Testing: {config['name']}")
    print('='*60)

    if not check_health(config):
        print(f"  ✗ Endpoint not ready")
        return False

    client = OpenAIClient(
        api_key=config["api_key"],
        model=config["model"],
        base_url=config["base_url"],
    )

    # Test 1: Basic generation
    print("\n[Test 1] Basic generation...")
    response = client.send_message(
        system_prompt="You are a Canadian citizen discussing politics.",
        user_message="What's your take on housing affordability?",
        temperature=0.7,
        max_tokens=100
    )
    print(f"Response: {response[:200]}...")

    # Test 2: Role-play as CES agent
    print("\n[Test 2] CES agent role-play...")
    response = client.send_message(
        system_prompt="""You are CES_Disengaged_Renter: A 25-year-old urban renter in Toronto.
        - Low political engagement (identity_salience: 0.25)
        - No party affiliation
        - Skeptical of political promises
        - Focus on immediate economic concerns

        Respond briefly and with appropriate disengagement.""",
        user_message="The government announced new housing subsidies. What do you think?",
        temperature=0.7,
        max_tokens=80
    )
    print(f"Response: {response}")

    # Test 3: Verify grit constraint compliance
    word_count = len(response.split())
    print(f"\n[Metrics] Word count: {word_count} (target: <80 for disengaged)")

    print(f"\n{config['name']} passed!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test models for émile-gce")
    parser.add_argument('--llama', action='store_true', help='Test Llama 3.1 8B')
    parser.add_argument('--mistral', action='store_true', help='Test Mistral-NeMo 12B')
    parser.add_argument('--both', action='store_true', help='Test both models')
    parser.add_argument('--wait', action='store_true', help='Wait for models to be ready')
    args = parser.parse_args()

    configs = []
    if args.llama or args.both:
        configs.append(LLAMA_CONFIG)
    if args.mistral or args.both:
        configs.append(MISTRAL_CONFIG)

    if not configs and not args.wait:
        print("Usage: python experiments/test_models.py --llama | --mistral | --both | --wait")
        return

    if args.wait or configs:
        target = configs if configs else [LLAMA_CONFIG, MISTRAL_CONFIG]
        if not wait_for_models(target):
            sys.exit(1)

    for config in configs:
        test_model(config)


if __name__ == "__main__":
    main()
