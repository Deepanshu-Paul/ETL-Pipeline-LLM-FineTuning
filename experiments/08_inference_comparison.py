"""
Experiment 08

Base Model vs Fine-Tuned Model

Purpose:
- Load the original Qwen model.
- Load the trained LoRA adapter.
- Run the same ETL failures through both.
- Compare their outputs.

This verifies whether LoRA fine-tuning actually
changed model behavior.
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


# ----------------------------------------------------------
# Test Cases
# ----------------------------------------------------------

TEST_CASES = [
    {
        "name": "Schema Error",
        "input": (
            "BigQuery load failed because column "
            "customer_id was not found in the target table."
        ),
    },
    {
        "name": "Authentication Error",
        "input": (
            "BigQuery authentication failed because "
            "the configured service account credentials "
            "were rejected."
        ),
    },
]


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


def create_messages(failure):

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
                f"{failure}"
            ),
        },
    ]


# ----------------------------------------------------------
# Generate
# ----------------------------------------------------------

def generate_response(
    model,
    tokenizer,
    failure,
):

    messages = create_messages(failure)

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Move the complete input dictionary to the model device.
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[-1]

    generated_tokens = outputs[
        0,
        input_length:,
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response


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

    print(
        f"Tokenizer Class : "
        f"{tokenizer.__class__.__name__}"
    )

    print()

    # ------------------------------------------------------
    # Load Base Model
    # ------------------------------------------------------

    print("=" * 60)
    print("Loading Base Model")
    print("=" * 60)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    print("✓ Base model loaded.")
    print()

    # ------------------------------------------------------
    # Load LoRA Adapter
    # ------------------------------------------------------

    print("=" * 60)
    print("Loading LoRA Adapter")
    print("=" * 60)

    fine_tuned_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    fine_tuned_model.eval()

    print("✓ LoRA adapter loaded.")
    print()

    # ------------------------------------------------------
    # Compare
    # ------------------------------------------------------

    for test_case in TEST_CASES:

        name = test_case["name"]
        failure = test_case["input"]

        print("=" * 60)
        print(f"Test Case: {name}")
        print("=" * 60)

        print()
        print("ETL Failure:")
        print(failure)
        print()

        # --------------------------------------------------
        # Base Model
        # --------------------------------------------------

        print("-" * 60)
        print("BASE MODEL")
        print("-" * 60)

        base_response = generate_response(
            base_model,
            tokenizer,
            failure,
        )

        print(base_response)
        print()

        # --------------------------------------------------
        # Fine-Tuned Model
        # --------------------------------------------------

        print("-" * 60)
        print("FINE-TUNED MODEL + LoRA")
        print("-" * 60)

        fine_tuned_response = generate_response(
            fine_tuned_model,
            tokenizer,
            failure,
        )

        print(fine_tuned_response)
        print()


if __name__ == "__main__":
    main()