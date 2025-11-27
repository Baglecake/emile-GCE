#!/usr/bin/env python3
"""
Clean training data for LoRA fine-tuning.

Removes:
- "Reflection trap" rows (From this discussion, I learned...)
- Bio-dump intros (Hello, I'm CES_...)
- Near-duplicate slogans

Usage:
    python experiments/clean_training_data.py team_llama.jsonl team_llama_CLEAN.jsonl
    python experiments/clean_training_data.py team_mistral.jsonl team_mistral_CLEAN.jsonl
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

# Patterns to REJECT entire rows (reflection trap)
REJECT_PATTERNS = [
    r"^From this discussion",  # Catch all reflection starts
    r"^From my standpoint",    # Wishy-washy framing
    r"^Initially,?\s*I was (quite )?(set|convinced)",
    r"^In conclusion",
    r"^Overall,?\s*(I|this discussion)",
    r"^To summarize",
    r"^In summary",
    r"^This conversation (has )?taught me",
    r"^Reflecting on (this|our) (discussion|conversation)",
    r"^Absolutely,?\s*and I think",  # Agreeable filler
]

# Patterns to STRIP from start of response (bio-dumps, greetings)
STRIP_PATTERNS = [
    r"^Hello,?\s*(everyone|I'm|my name)",
    r"^Hi,?\s*(everyone|I'm|my name)",
    r"^Bonjour,?\s*(everyone|I'm|CES_)",
    r"^I'm CES_[A-Za-z0-9_]+[^.]*[.,]\s*",
    r"^As CES_[A-Za-z0-9_]+[^.]*[.,]\s*",
    r"^\[CES_[A-Za-z0-9_]+\]:?\s*",
    r"^Hello,\s*I'm CES_[A-Za-z0-9_]+[^.]*[.,]\s*",
    r"^As a \d+[- ]?year[- ]?old[^.]*[.,]\s*",
    r"^I'm a \d+[- ]?year[- ]?old[^.]*[.,]\s*",
    r"^Well,?\s*I'm a \d+[- ]?year[- ]?old[^.]*[.,]\s*",
    r"^I appreciate (everyone's|the|your)[^.]*[.,]\s*",
    r"^I'd like to (acknowledge|thank|appreciate)[^.]*[.,]\s*",
    r"^Thank you (for|,)[^.]*[.,]\s*",
]

COMPILED_REJECTS = [re.compile(p, re.IGNORECASE) for p in REJECT_PATTERNS]
COMPILED_STRIPS = [re.compile(p, re.IGNORECASE) for p in STRIP_PATTERNS]


def should_reject(response: str) -> bool:
    """Check if response should be completely rejected."""
    for pattern in COMPILED_REJECTS:
        if pattern.search(response):
            return True
    return False


def clean_response(response: str) -> str:
    """Strip bio-dump intros while keeping substance."""
    cleaned = response
    for pattern in COMPILED_STRIPS:
        cleaned = pattern.sub("", cleaned, count=1)

    # Capitalize first letter if we stripped something
    if cleaned != response and cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()

    return cleaned.strip()


def get_response_fingerprint(response: str, n_words: int = 15) -> str:
    """Get fingerprint for deduplication (first N words, normalized)."""
    words = response.lower().split()[:n_words]
    return " ".join(words)


def clean_jsonl(input_path: Path, output_path: Path) -> dict:
    """Clean JSONL file and return stats."""
    stats = {
        "total": 0,
        "rejected": 0,
        "cleaned": 0,
        "deduplicated": 0,
        "kept": 0,
    }

    seen_fingerprints = Counter()
    cleaned_rows = []

    with open(input_path) as f:
        for line in f:
            stats["total"] += 1
            row = json.loads(line)
            text = row.get("text", "")

            # Extract response from "Instruction: ...\n\nResponse: ..."
            if "Response: " in text:
                parts = text.split("Response: ", 1)
                instruction = parts[0]
                response = parts[1]
            else:
                continue

            # Check for rejection
            if should_reject(response):
                stats["rejected"] += 1
                continue

            # Clean response
            cleaned = clean_response(response)
            if cleaned != response:
                stats["cleaned"] += 1

            # Skip if too short after cleaning
            if len(cleaned) < 30:
                stats["rejected"] += 1
                continue

            # Deduplication check
            fingerprint = get_response_fingerprint(cleaned)
            seen_fingerprints[fingerprint] += 1

            if seen_fingerprints[fingerprint] > 2:  # Allow up to 2 similar
                stats["deduplicated"] += 1
                continue

            # Keep this row
            stats["kept"] += 1
            cleaned_rows.append({
                "text": f"{instruction}Response: {cleaned}"
            })

    # Write cleaned output
    with open(output_path, "w") as f:
        for row in cleaned_rows:
            f.write(json.dumps(row) + "\n")

    return stats


def main():
    if len(sys.argv) < 3:
        print("Usage: python clean_training_data.py <input.jsonl> <output.jsonl>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    print(f"Cleaning: {input_path}")
    stats = clean_jsonl(input_path, output_path)

    print(f"\n=== Cleaning Results ===")
    print(f"Total rows:      {stats['total']}")
    print(f"Rejected:        {stats['rejected']} (reflection trap, too short)")
    print(f"Cleaned:         {stats['cleaned']} (bio-dumps stripped)")
    print(f"Deduplicated:    {stats['deduplicated']} (>2 similar)")
    print(f"Kept:            {stats['kept']}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
