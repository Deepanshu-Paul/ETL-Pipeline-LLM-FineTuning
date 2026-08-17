"""
Experiment 06

Loss Inspection

Purpose:
- Load the base Qwen model.
- Inspect logits for one target position.
- Convert logits to probabilities.
- Calculate the probability of the correct token.
- Understand cross-entropy loss mathematically.

This experiment is for understanding the mechanics.
It does not perform training.
"""

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    print("=" * 60)
    print("Loading Model and Tokenizer")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
    )

    model.eval()

    print("✓ Model loaded.")
    print()

    # ------------------------------------------------------
    # Example
    # ------------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": (
                "Analyze this ETL failure.\n\n"
                "ETL Failure:\n"
                "BigQuery load failed because column "
                "customer_id was not found in the target table."
            ),
        }
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    encoding = tokenizer(
        formatted_text,
        add_special_tokens=False,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(
        model.device
    )

    print("=" * 60)
    print("Input")
    print("=" * 60)

    print(f"Token Count : {input_ids.shape[1]}")
    print()

    # ------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids
        )

    logits = outputs.logits

    print("=" * 60)
    print("Logits")
    print("=" * 60)

    print(f"Logits Shape : {tuple(logits.shape)}")
    print()

    # ------------------------------------------------------
    # Inspect final prediction
    # ------------------------------------------------------

    final_logits = logits[0, -1]

    print(
        f"Vocabulary Size : "
        f"{final_logits.shape[0]}"
    )

    # ------------------------------------------------------
    # Convert logits to probabilities
    # ------------------------------------------------------

    probabilities = torch.softmax(
        final_logits,
        dim=-1,
    )

    print("=" * 60)
    print("Probability Distribution")
    print("=" * 60)

    print(
        f"Probability Sum : "
        f"{probabilities.sum().item():.6f}"
    )

    print()

    # ------------------------------------------------------
    # Top predictions
    # ------------------------------------------------------

    top_probabilities, top_ids = torch.topk(
        probabilities,
        k=10,
    )

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
            f"{probability.item():.6f} | "
            f"{repr(token)}"
        )

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
            f"{probability.item():.6f} | "
            f"{repr(token)}"
        )

    print()

    # ------------------------------------------------------
    # Probability of the correct first assistant token
    # ------------------------------------------------------

    correct_token = "{'"
    correct_token_id = 13608

    correct_probability = probabilities[
        correct_token_id
    ]

    token_loss = -torch.log(
        correct_probability
    )

    print("=" * 60)
    print("Correct Token Analysis")
    print("=" * 60)

    print(f"Correct Token       : {correct_token}")
    print(f"Correct Token ID    : {correct_token_id}")
    print(
        f"Correct Probability : "
        f"{correct_probability.item():.10f}"
    )
    print(
        f"Token Loss          : "
        f"{token_loss.item():.6f}"
    )

    print()


if __name__ == "__main__":
    main()