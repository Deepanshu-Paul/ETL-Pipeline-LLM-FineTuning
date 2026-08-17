"""
Experiment 05

Dataset Preparation

Purpose:
- Load the ETL failure dataset.
- Inspect its structure.
- Verify the number of examples.
- Inspect individual training examples.

This experiment will later evolve to include:
- Dataset validation
- Train/validation split
- Formatting for SFT
- Tokenization
- Sequence-length analysis
"""

from datasets import load_dataset
from datasets import load_dataset
from transformers import AutoTokenizer


# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

DATASET_PATH = "data/raw/etl_failures.json"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ----------------------------------------------------------
# Dataset Loading
# ----------------------------------------------------------

def load_etl_dataset():

    print("=" * 60)
    print("Loading ETL Dataset")
    print("=" * 60)

    dataset = load_dataset(
        "json",
        data_files=DATASET_PATH,
        split="train",
    )

    print("✓ Dataset loaded.")
    print()

    return dataset


# ----------------------------------------------------------
# Dataset Inspection
# ----------------------------------------------------------

def inspect_dataset(dataset):

    print("=" * 60)
    print("Dataset Information")
    print("=" * 60)

    print(f"Number of examples : {len(dataset)}")
    print(f"Columns            : {dataset.column_names}")
    print()

    print("=" * 60)
    print("First Example")
    print("=" * 60)

    example = dataset[0]

    print(f"Instruction : {example['instruction']}")
    print(f"Input       : {example['input']}")
    print(f"Output      : {example['output']}")

    print()

def inspect_categories(dataset):

    print("=" * 60)
    print("Dataset Categories")
    print("=" * 60)

    for index, example in enumerate(dataset):

        category = example["output"]["category"]

        print(
            f"{index + 1:2d}. "
            f"{category}"
        )

    print()


def validate_dataset(dataset):

    print("=" * 60)
    print("Dataset Validation")
    print("=" * 60)

    required_input_fields = {
        "instruction",
        "input",
        "output",
    }

    required_output_fields = {
        "category",
        "root_cause",
        "recommendation",
    }

    errors = []

    for index, example in enumerate(dataset):

        # Check top-level fields
        missing_fields = (
            required_input_fields
            - set(example.keys())
        )

        if missing_fields:
            errors.append(
                f"Example {index + 1}: "
                f"missing fields {missing_fields}"
            )
            continue

        # Check output fields
        output = example["output"]

        if not isinstance(output, dict):
            errors.append(
                f"Example {index + 1}: "
                "output is not a dictionary"
            )
            continue

        missing_output_fields = (
            required_output_fields
            - set(output.keys())
        )

        if missing_output_fields:
            errors.append(
                f"Example {index + 1}: "
                f"missing output fields "
                f"{missing_output_fields}"
            )

    if errors:

        print("✗ Dataset validation failed.")
        print()

        for error in errors:
            print(error)

    else:

        print("✓ Dataset validation passed.")

    print()


def split_dataset(dataset):

    print("=" * 60)
    print("Train / Validation Split")
    print("=" * 60)

    # Fixed indices make the experiment reproducible.
    validation_indices = [1, 4]

    train_indices = [
        index
        for index in range(len(dataset))
        if index not in validation_indices
    ]

    train_dataset = dataset.select(train_indices)
    validation_dataset = dataset.select(validation_indices)

    print(f"Training examples   : {len(train_dataset)}")
    print(f"Validation examples : {len(validation_dataset)}")

    print()
    print("Validation Examples")
    print()

    for index, example in enumerate(validation_dataset):

        category = example["output"]["category"]

        print(
            f"{index + 1}. "
            f"{category} | "
            f"{example['input']}"
        )

    print()

    return train_dataset, validation_dataset


def inspect_sft_formatting(dataset, tokenizer):

    print("=" * 60)
    print("SFT Formatting")
    print("=" * 60)

    example = dataset[0]

    messages = [
        {
            "role": "user",
            "content": (
                f"{example['instruction']}\n\n"
                f"ETL Failure:\n"
                f"{example['input']}"
            ),
        },
        {
            "role": "assistant",
            "content": str(example["output"]),
        },
    ]

    print("Messages")
    print(messages)
    print()

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    print("=" * 60)
    print("Qwen Chat Template Output")
    print("=" * 60)

    print(formatted_text)
    print()


def inspect_formatted_tokens(tokenizer, dataset):

    print("=" * 60)
    print("SFT Tokenization")
    print("=" * 60)

    example = dataset[0]

    messages = [
        {
            "role": "user",
            "content": (
                f"{example['instruction']}\n\n"
                f"ETL Failure:\n"
                f"{example['input']}"
            ),
        },
        {
            "role": "assistant",
            "content": str(example["output"]),
        },
    ]

    # ----------------------------------------------------------
    # Step 1: Apply Qwen chat template
    # ----------------------------------------------------------

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    print("Formatted Text")
    print(formatted_text)
    print()

    # ----------------------------------------------------------
    # Step 2: Tokenize the formatted conversation
    # ----------------------------------------------------------

    encoding = tokenizer(
        formatted_text,
        add_special_tokens=False,
    )

    token_ids = encoding["input_ids"]

    print("=" * 60)
    print("Tokenization Result")
    print("=" * 60)

    print(f"Token Count : {len(token_ids)}")
    print()

    # ----------------------------------------------------------
    # Step 3: Inspect token IDs
    # ----------------------------------------------------------

    first_30_ids = token_ids[:30]

    print("First 30 Token IDs")
    print(first_30_ids)
    print()

    # ----------------------------------------------------------
    # Step 4: Convert IDs back to tokens
    # ----------------------------------------------------------

    first_30_tokens = tokenizer.convert_ids_to_tokens(
        first_30_ids
    )

    print("First 30 Tokens")
    print(first_30_tokens)
    print()

    print("=" * 60)
    print("Full Token Sequence")
    print("=" * 60)

    all_tokens = tokenizer.convert_ids_to_tokens(
        token_ids
    )

    for index, token in enumerate(all_tokens):

        print(
            f"{index:3d}  {token}"
        )

    print()

def inspect_sft_labels(tokenizer, dataset):

    print("=" * 60)
    print("SFT Labels and Loss Mask")
    print("=" * 60)

    example = dataset[0]

    messages = [
        {
            "role": "user",
            "content": (
                f"{example['instruction']}\n\n"
                f"ETL Failure:\n"
                f"{example['input']}"
            ),
        },
        {
            "role": "assistant",
            "content": str(example["output"]),
        },
    ]

    # ----------------------------------------------------------
    # Build the complete conversation
    # ----------------------------------------------------------

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    encoding = tokenizer(
        formatted_text,
        add_special_tokens=False,
    )

    token_ids = encoding["input_ids"]

    # ----------------------------------------------------------
    # Find the assistant response boundary
    # ----------------------------------------------------------

    assistant_marker = "<|im_start|>assistant\n"

    marker_encoding = tokenizer(
        assistant_marker,
        add_special_tokens=False,
    )

    marker_ids = marker_encoding["input_ids"]

    # Find the last occurrence of the assistant marker.
    assistant_start = None

    for index in range(
        len(token_ids) - len(marker_ids) + 1
    ):

        if token_ids[
            index:index + len(marker_ids)
        ] == marker_ids:

            assistant_start = (
                index + len(marker_ids)
            )

    if assistant_start is None:
        raise ValueError(
            "Could not locate assistant response boundary."
        )

    # ----------------------------------------------------------
    # Construct labels
    # ----------------------------------------------------------

    labels = [-100] * len(token_ids)

    for index in range(
        assistant_start,
        len(token_ids),
    ):
        labels[index] = token_ids[index]

    # ----------------------------------------------------------
    # Display results
    # ----------------------------------------------------------

    print(f"Total Tokens       : {len(token_ids)}")
    print(f"Assistant Start    : {assistant_start}")
    print(
        f"Target Token Count : "
        f"{len(token_ids) - assistant_start}"
    )
    print()

    print("Position | Token ID | Label")
    print("-" * 60)

    tokens = tokenizer.convert_ids_to_tokens(
        token_ids
    )

    for index in range(len(token_ids)):

        if (
            index < assistant_start - 3
            or index > assistant_start + 5
        ):
            continue

        print(
            f"{index:8d} | "
            f"{token_ids[index]:8d} | "
            f"{labels[index]:8d} | "
            f"{tokens[index]}"
        )

    print()

    ignored = sum(
        1
        for label in labels
        if label == -100
    )

    supervised = len(labels) - ignored

    print(f"Ignored Tokens      : {ignored}")
    print(f"Supervised Tokens   : {supervised}")
    print()

def load_tokenizer():

    print("=" * 60)
    print("Loading Tokenizer")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("✓ Tokenizer loaded.")
    print()

    return tokenizer

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    dataset = load_etl_dataset()

    inspect_dataset(dataset)

    inspect_categories(dataset)

    validate_dataset(dataset)

    train_dataset, validation_dataset = split_dataset(
        dataset
    )

    tokenizer = load_tokenizer()

    inspect_sft_formatting(
        train_dataset,
        tokenizer,
    )

    inspect_formatted_tokens(
        tokenizer,
        train_dataset,
    )

    inspect_sft_labels(
        tokenizer,
        train_dataset,
    )


if __name__ == "__main__":
    main()