# ETL Pipeline LLM Fine-Tuning

A production-oriented starter project for fine-tuning a large language model to analyze ETL pipeline failures. Given an incident message, the intended model output identifies the failure category, explains the likely root cause, and suggests a remediation step.

## Goals

- Prepare supervised fine-tuning (SFT) data from ETL failure examples.
- Fine-tune efficiently with PEFT methods such as LoRA and QLoRA.
- Evaluate predictions against a held-out set of ETL incidents.
- Provide a clean foundation for local or GPU-backed inference.

## Dataset

The included JSON datasets are under `data/raw/`:

| File | Purpose | Records |
| --- | --- | ---: |
| `etl_failures.json` | SFT training examples | 11 |
| `etl_eval.json` | Held-out evaluation examples | 6 |

Training records use the following structure:

```json
{
  "instruction": "Analyze the ETL pipeline failure and identify the failure category, root cause, and recommended action.",
  "input": "BigQuery load failed because column customer_id was not found in the target table.",
  "output": {
    "category": "SCHEMA_ERROR",
    "root_cause": "The source and target schemas are inconsistent.",
    "recommendation": "Compare schemas and update the mapping before rerunning the load."
  }
}
```

Examples cover schema, connection, authentication, timeout, transformation, data-quality, duplicate-event, and unknown failures.

## Repository layout

```text
.
├── adapters/             # Saved LoRA/QLoRA adapters (local artifacts)
├── configs/              # Model, LoRA, and training configuration modules
├── data/raw/             # Training and evaluation JSON datasets
├── experiments/          # Experiment outputs and notes
├── logs/                 # Training and application logs
├── notebooks/            # Exploratory notebooks
├── outputs/              # Checkpoints and generated artifacts
├── src/
│   ├── dataset/          # Loading, preprocessing, and chat formatting
│   ├── evaluation/       # Evaluation helpers
│   ├── inference/        # Prediction interface
│   ├── models/           # Model and tokenizer loading
│   ├── training/         # Trainer and callbacks
│   └── utils/            # Environment, logging, and seed utilities
├── train.py              # Training entry point (scaffold)
└── infer.py              # Inference entry point (scaffold)
```

## Setup

This project requires Python 3.12.

```bash
git clone https://github.com/Deepanshu-Paul/ETL-Pipeline-LLM-FineTuning.git
cd ETL-Pipeline-LLM-FineTuning
python -m venv .venv
```

Activate the virtual environment, then install dependencies:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

`requirements.txt` includes PyTorch with CUDA 12.4 packages. Ensure your NVIDIA driver and hardware are compatible, or install a suitable PyTorch build for your environment before installing the remaining dependencies.

## Status

The project structure and example datasets are in place. The modules in `src/`, configuration files, and the `train.py` / `infer.py` entry points are currently empty scaffolds and must be implemented before model training or inference can run.

## Planned workflow

1. Load and validate `data/raw/etl_failures.json`.
2. Convert each record into the selected model's chat template.
3. Load a base causal language model and tokenizer.
4. Apply LoRA or QLoRA configuration and run SFT.
5. Save the adapter to `adapters/` and checkpoints to `outputs/`.
6. Evaluate predictions with `data/raw/etl_eval.json`.
7. Serve or run local inference through `infer.py`.

## License

This project is licensed under the [MIT License](LICENSE).
