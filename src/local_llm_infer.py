"""Run a locally fine-tuned LoRA adapter on redacted fraud evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .llm_dataset import SYSTEM_PROMPT
def generate_case_brief(
    evidence: dict,
    adapter: str = "artifacts/fraud_analyst_lora",
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens: int = 200,
) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = Path(adapter)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
    )

    model = PeftModel.from_pretrained(base, adapter_path).eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Create a concise analyst case brief from this "
                f"redacted event: {json.dumps(evidence)}"
            ),
        },
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    input_ids = encoded["input_ids"].to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(
        generated[0][input_ids.shape[-1]:],
        skip_special_tokens=True,
    )

    return answer.strip()


def generate_case_brief(
    evidence: dict,
    adapter: Path = Path("artifacts/fraud_analyst_lora"),
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens: int = 180,
) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
    )

    model = PeftModel.from_pretrained(base, adapter).eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Create a concise analyst case brief from this redacted event: "
                + json.dumps(evidence)
            ),
        },
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    input_ids = encoded["input_ids"].to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        generated[0][input_ids.shape[-1]:],
        skip_special_tokens=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("artifacts/fraud_analyst_lora"),
    )

    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )

    parser.add_argument(
        "--case-json",
        required=True,
        help="Redacted structured transaction evidence as JSON.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=180,
    )

    args = parser.parse_args()

    if not args.adapter.exists():
        raise SystemExit(
            f"Adapter not found: {args.adapter}. "
            "Run `python -m src.train_local_llm` first."
        )

    try:
        evidence = json.loads(args.case_json)
    except json.JSONDecodeError as error:
        raise SystemExit("--case-json must be valid JSON") from error

    print(
        generate_case_brief(
            evidence,
            adapter=args.adapter,
            base_model=args.base_model,
            max_new_tokens=args.max_new_tokens,
        )
    )


if __name__ == "__main__":
    main()