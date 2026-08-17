"""
Experiment 09

Base vs LoRA Token Probability Comparison

Purpose:
- Compare the probability of the correct first assistant token
  before and after LoRA fine-tuning.
- Connect LoRA weight updates to logits and probabilities.
"""

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel


# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

ADAPTER_PATH = "outputs/qwen-etl-lora"

CORRECT_TOKEN = "{'"
CORRECT_TOKEN_ID = 13608


# ----------------------------------------------------------
# Prompt
# ----------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are Qwen, created by Alibaba Cloud. "
    "You are a helpful assistant."
)

INSTRUCTION = (
    "Analyze the ETL pipeline failure and identify "
    "the failure category, root cause, and "
    "recommended action."
)

FAILURE = (
    "BigQuery load failed because column "
    "customer_id was not found in the target table."
)


def create_messages():

    return [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE,
        },
        {
            "role": "user",
            "content": (
                f"{INSTRUCTION}\n\n"
                f"ETL Failure:\n"
                f"{FAILURE}"
            ),
        },
    ]


# ----------------------------------------------------------
# Probability Inspection
# ----------------------------------------------------------

def inspect_probability(
    model,
    tokenizer,
    model_name,
):

    messages = create_messages()

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    # ------------------------------------------------------
    # Last input position
    # ------------------------------------------------------

    logits = outputs.logits[:, -1, :]

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    correct_probability = probabilities[
        0,
        CORRECT_TOKEN_ID,
    ]

    token_loss = -torch.log(
        correct_probability
    )

    # ------------------------------------------------------
    # Top Predictions
    # ------------------------------------------------------

    top_probabilities, top_ids = torch.topk(
        probabilities[0],
        k=10,
    )

    print("=" * 60)
    print(model_name)
    print("=" * 60)

    print(
        f"Correct Token       : "
        f"{CORRECT_TOKEN}"
    )

    print(
        f"Correct Token ID    : "
        f"{CORRECT_TOKEN_ID}"
    )

    print(
        f"Correct Probability : "
        f"{correct_probability.item():.10f}"
    )

    print(
        f"Token Loss          : "
        f"{token_loss.item():.6f}"
    )

    print()

    print("Top 10 Predictions")
    print()

    for probability, token_id in zip(
        top_probabilities,
        top_ids,
    ):

        token = tokenizer.decode(
            [token_id.item()]
        )

        print(
            f"{token_id.item():6d} | "
            f"{probability.item():.10f} | "
            f"{repr(token)}"
        )

    print()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    print("=" * 60)
    print("Loading Tokenizer")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print()

    # ------------------------------------------------------
    # Base Model
    # ------------------------------------------------------

    print("=" * 60)
    print("Loading Base Model")
    print("=" * 60)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    base_model.eval()

    print("✓ Base model loaded.")
    print()

    # ------------------------------------------------------
    # Base Probability
    # ------------------------------------------------------

    inspect_probability(
        base_model,
        tokenizer,
        "BASE MODEL",
    )

    # ------------------------------------------------------
    # LoRA Model
    # ------------------------------------------------------

    print("=" * 60)
    print("Loading LoRA Adapter")
    print("=" * 60)

    lora_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    lora_model.eval()

    print("✓ LoRA adapter loaded.")
    print()

    # ------------------------------------------------------
    # LoRA Probability
    # ------------------------------------------------------

    inspect_probability(
        lora_model,
        tokenizer,
        "FINE-TUNED MODEL + LoRA",
    )


if __name__ == "__main__":
    main()