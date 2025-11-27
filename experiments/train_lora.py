#!/usr/bin/env python3
"""
LoRA fine-tuning with Unsloth for émile-gce agents.

Based on Gemini's recommendations:
- Llama 3.1 8B → Urban Progressive (verbose, articulate)
- Mistral-NeMo 12B → Rural Conservative (direct, no-nonsense)

Run on RunPod with GPU. Upload training_*.jsonl files first.

Usage:
    python experiments/train_lora.py --model llama --data training_urban_progressive.jsonl
    python experiments/train_lora.py --model mistral --data training_rural_conservative.jsonl
"""

import argparse
import os
from pathlib import Path


def train_with_unsloth(
    model_name: str,
    data_path: str,
    output_dir: str,
    max_steps: int = 500,
    learning_rate: float = 2e-4,
    batch_size: int = 2,
    gradient_accumulation: int = 4,
):
    """
    Fine-tune model with Unsloth LoRA.

    Requires: pip install unsloth
    """
    try:
        from unsloth import FastLanguageModel
        from datasets import load_dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError:
        print("ERROR: Unsloth not installed. Run:")
        print("  pip install unsloth")
        print("  pip install --no-deps xformers trl peft accelerate bitsandbytes")
        return

    print(f"Loading model: {model_name}")
    print(f"Training data: {data_path}")

    # Load model with 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
    )

    # Load dataset
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"Loaded {len(dataset)} training examples")

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        warmup_steps=20,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=output_dir,
        save_steps=100,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    print("\nStarting training...")
    trainer.train()

    # Save LoRA adapter
    lora_path = Path(output_dir) / "lora_adapter"
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"\nLoRA adapter saved to: {lora_path}")

    # Optional: Save merged model for inference
    print("\nMerging LoRA weights for inference...")
    model = FastLanguageModel.for_inference(model)

    merged_path = Path(output_dir) / "merged_model"
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to: {merged_path}")

    return str(lora_path)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning with Unsloth")

    parser.add_argument("--model", choices=["llama", "mistral"], required=True,
                       help="Base model: llama (8B) or mistral (12B)")
    parser.add_argument("--data", required=True, help="Training JSONL file")
    parser.add_argument("--output", default="lora_output", help="Output directory")
    parser.add_argument("--steps", type=int, default=500, help="Max training steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")

    args = parser.parse_args()

    # Model mapping
    model_map = {
        "llama": "unsloth/Meta-Llama-3.1-8B",
        "mistral": "unsloth/Mistral-Nemo-Base-2407",
    }

    model_name = model_map[args.model]
    output_dir = f"{args.output}_{args.model}"

    train_with_unsloth(
        model_name=model_name,
        data_path=args.data,
        output_dir=output_dir,
        max_steps=args.steps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
