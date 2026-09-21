"""Optional LoRA/QLoRA fine-tuning for fraud-analyst explanations, not decisions."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                                  DataCollatorForLanguageModeling, Trainer, TrainingArguments)
    except ImportError as error:
        raise SystemExit("Install optional dependencies first: pip install -r requirements-llm.txt") from error

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("llm/data/fraud_analyst_sft.jsonl"))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", type=Path, default=Path("artifacts/fraud_analyst_lora"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}. Run `python -m src.llm_dataset` first.")
    if args.load_in_4bit and not torch.cuda.is_available():
        raise SystemExit("QLoRA requires a CUDA GPU. Use WSL2/Ubuntu with an NVIDIA GPU, or omit --load-in-4bit.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    quantization = None
    if args.load_in_4bit:
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                          bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=quantization, device_map="auto" if args.load_in_4bit else None
    )
    adapter = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_rank, lora_alpha=args.lora_alpha,
                         lora_dropout=.05, target_modules="all-linear")
    model = get_peft_model(model, adapter)
    model.print_trainable_parameters()
    dataset = load_dataset("json", data_files=str(args.dataset), split="train")

    def tokenize(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return tokenizer(text, truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    training = TrainingArguments(
        output_dir=str(args.output), num_train_epochs=args.epochs, learning_rate=2e-4,
        per_device_train_batch_size=1, gradient_accumulation_steps=8, logging_steps=10,
        save_strategy="epoch", report_to="none", fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=training, train_dataset=tokenized,
                      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved local LoRA adapter to {args.output}")


if __name__ == "__main__":
    main()
