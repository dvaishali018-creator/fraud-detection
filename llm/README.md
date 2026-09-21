# Custom Local Fraud-Analyst LLM

## What this component is

This project does **not** train a foundation LLM from zero. That would require an enormous, licensed text corpus and multi-GPU compute. Instead, it creates your own lightweight **LoRA adapter** on top of an existing open-weight 1.5B-parameter instruction model. The adapter learns to turn redacted transaction evidence into a concise analyst case brief.

It must **not** approve, block, or determine fraud probability. The calibrated tabular model in `src/train.py` remains responsible for the fraud score and the cost-aware decision engine remains responsible for action routing.

## Data format and privacy

`python -m src.llm_dataset` creates training JSONL without transaction ID, customer ID, names, addresses, full card numbers, email, phone, IP address, or device IDs. Before using real data, have a supervisor/organisation approve the de-identification process and retain only analyst-reviewed examples.

An example trains this kind of output:

```json
{"assessment":"suspicious","evidence":["rapid transaction burst"],"recommended_review_step":"verify with analyst and supporting account evidence"}
```

## Build steps

```powershell
# First run the normal detector and obtain reviewed, labelled cases.
python -m src.llm_dataset --input data\raw\transactions.csv --output llm\data\fraud_analyst_sft.jsonl

# In a separate GPU-capable environment only:
python -m pip install -r requirements-llm.txt
python -m src.train_local_llm --load-in-4bit --epochs 2
```

The resulting adapter is saved in `artifacts/fraud_analyst_lora`. Preserve the base-model name, LoRA parameters, prompt template, dataset version, evaluation results, and safety limitations in your dissertation.

## Validate after training

Do not connect the adapter to a dashboard before checking its answers on held-out redacted cases. For a model trained with `Qwen/Qwen2.5-0.5B-Instruct`, run:

```powershell
python -m src.local_llm_infer --base-model Qwen/Qwen2.5-0.5B-Instruct --case-json '{"amount":1200,"merchant_category":"electronics","country":"NG","hour":2,"device_trusted":0,"account_age_days":3,"failed_attempts_24h":4,"prior_txn_count_1h":4,"prior_txn_count_24h":8,"amount_to_prior_avg":5.1,"new_merchant_for_customer":1,"new_country_for_customer":1}'
```

Check that it returns valid, evidence-grounded JSON and does not invent information. Evaluate at least 200 held-out cases with a reviewer before describing the component as useful. Keep the LLM advisory-only: the tabular ML model and the cost engine remain the only decision components.

## Recommended dataset sizes

These are practical research targets, not guaranteed thresholds.

| Goal | Transaction dataset | LLM case-brief dataset |
|---|---:|---:|
| Functional demonstration | 10,000+ events; synthetic permitted | 500+ synthetic examples, clearly labelled as demo-only |
| Final-year evaluation | 50,000+ events and 1,000+ confirmed fraud labels | 5,000+ reviewed/de-identified cases |
| Strong publication study | 100,000–500,000+ time-ordered events and 2,000–5,000+ fraud labels across multiple months | 10,000–20,000+ high-quality reviewed cases plus a held-out expert-reviewed test set |

The 12,000 generated rows in this repository are only for software demonstration. Do not report their metrics as evidence of real-world fraud performance.

## Operating system and hardware

| Task | Recommended environment |
|---|---|
| Dashboard, API and tabular ML | Windows 11 64-bit, Python 3.12–3.14, 16 GB RAM recommended; no GPU required. |
| Local LLM inference | Windows 11 or Ubuntu 22.04/24.04, 16 GB RAM; GPU improves usability. |
| LoRA/QLoRA training | Ubuntu 22.04/24.04 or WSL2 on Windows 11, NVIDIA GPU with 12 GB VRAM minimum and 16 GB preferred, CUDA driver, 32 GB system RAM recommended. |
| No NVIDIA GPU | Use a hosted GPU notebook/lab or omit LLM fine-tuning; the main fraud detector still runs locally. |

WSL2 provides a Linux environment with GPU-accelerated ML support on Windows. LoRA keeps base model weights frozen and only trains small adapter weights, reducing memory and compute needs. [Microsoft WSL GPU guidance](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute), [Hugging Face LoRA guidance](https://huggingface.co/docs/peft/main/conceptual_guides/lora)
