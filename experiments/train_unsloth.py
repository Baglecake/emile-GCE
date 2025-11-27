#!/usr/bin/env python3
"""
Unsloth LoRA training script for émile-gce.
Copy this to RunPod and run: python train_unsloth.py

Usage:
    python train_unsloth.py --model llama --data team_llama_CLEAN.jsonl
    python train_unsloth.py --model mistral --data team_mistral_CLEAN.jsonl
"""

import os
os.environ["TRANSFORMERS_NO_MAMBA"] = "1"  # Avoid mamba_ssm import errors

import argparse
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

MODELS = {
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral": "unsloth/Mistral-Nemo-Base-2407",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llama", "mistral"], default="llama")
    parser.add_argument("--data", default="data/team_llama_CLEAN.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    model_name = MODELS[args.model]
    output_dir = args.output or f"outputs/{args.model}_lora"

    print(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # Fix: ensure pad token is set
    tokenizer.pad_token = tokenizer.eos_token

    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
    )

    print(f"Loading dataset: {args.data}")
    dataset = load_dataset("json", data_files=args.data, split="train")
    print(f"Dataset size: {len(dataset)} examples")

    print(f"Training for {args.epochs} epochs...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            logging_steps=10,
            save_steps=100,
            warmup_steps=10,
            learning_rate=2e-4,
            bf16=True,
            report_to="none",
        ),
    )

    trainer.train()

    print(f"Saving adapter to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Done!")

if __name__ == "__main__":
    main()
