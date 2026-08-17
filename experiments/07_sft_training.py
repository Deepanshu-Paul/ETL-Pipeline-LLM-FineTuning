"""
Experiment 07

Actual SFT Training with LoRA

Purpose:
- Load the ETL dataset.
- Format examples using the Qwen chat template.
- Load the Qwen base model.
- Apply LoRA.
- Configure TRL SFTTrainer.
- Perform the first real fine-tuning run.

This is the first experiment in which model parameters
are actually trained on the ETL dataset.
"""

import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import LoraConfig

from trl import (
    SFTTrainer,
    SFTConfig,
)


# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DATASET_PATH = "data/raw/etl_failures.json"

OUTPUT_DIR = "outputs/qwen-etl-lora"


# ----------------------------------------------------------
# Dataset
# ----------------------------------------------------------

def load_etl_dataset():

    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    dataset = load_dataset(
        "json",
        data_files=DATASET_PATH,
        split="train",
    )

    print(f"Dataset Size : {len(dataset)}")
    print("✓ Dataset loaded.")
    print()

    return dataset


def split_dataset(dataset):

    validation_indices = [1, 4]

    train_indices = [
        index
        for index in range(len(dataset))
        if index not in validation_indices
    ]

    train_dataset = dataset.select(
        train_indices
    )

    validation_dataset = dataset.select(
        validation_indices
    )

    print("=" * 60)
    print("Dataset Split")
    print("=" * 60)

    print(
        f"Training examples   : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation examples : "
        f"{len(validation_dataset)}"
    )

    print()

    return train_dataset, validation_dataset


def prepare_sft_dataset(dataset):

    print("=" * 60)
    print("Preparing SFT Dataset")
    print("=" * 60)

    def format_example(example):

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are Qwen, created by Alibaba Cloud. "
                    "You are a helpful assistant."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{example['instruction']}\n\n"
                    f"ETL Failure:\n"
                    f"{example['input']}"
                ),
            },
        ]

        completion = [
            {
                "role": "assistant",
                "content": str(example["output"]),
            }
        ]

        return {
            "prompt": prompt,
            "completion": completion,
        }

    dataset = dataset.map(
        format_example,
    )

    print(
        f"Dataset Size : {len(dataset)}"
    )

    print()
    print("Example Prompt")
    print("-" * 60)
    print(dataset[0]["prompt"])

    print()
    print("Example Completion")
    print("-" * 60)
    print(dataset[0]["completion"])

    print()

    return dataset
# ----------------------------------------------------------
# Formatting
# ----------------------------------------------------------

def format_example(example):

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

    return messages


def formatting_func(example):

    messages = format_example(example)

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


# ----------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------

def load_tokenizer():

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

    print(
        f"Vocabulary Size : "
        f"{tokenizer.vocab_size}"
    )

    print()

    return tokenizer


# ----------------------------------------------------------
# Model
# ----------------------------------------------------------

def load_model():

    print("=" * 60)
    print("Loading Base Model")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    print("✓ Model loaded.")
    print()

    return model


# ----------------------------------------------------------
# LoRA
# ----------------------------------------------------------

def create_lora_config():

    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    global tokenizer

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------

    dataset = load_etl_dataset()

    train_dataset, validation_dataset = (
        split_dataset(dataset)
    )

    # ------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------

    tokenizer = load_tokenizer()

    # ------------------------------------------------------
    # Prepare SFT Dataset
    # ------------------------------------------------------

    train_dataset = prepare_sft_dataset(
        train_dataset,
    )

    validation_dataset = prepare_sft_dataset(
        validation_dataset,
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = load_model()

    # ------------------------------------------------------
    # LoRA
    # ------------------------------------------------------

    lora_config = create_lora_config()

    # ------------------------------------------------------
    # Training Configuration
    # ------------------------------------------------------

    training_config = SFTConfig(
        output_dir=OUTPUT_DIR,

        # Batch
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,

        # Training
        num_train_epochs=3,
        learning_rate=2e-4,

        # Precision
        bf16=True,

        # Memory
        gradient_checkpointing=True,

        # Sequence length
        max_length=256,

        # Loss
        completion_only_loss=True,

        # Evaluation
        eval_strategy="epoch",

        # Logging
        logging_strategy="steps",
        logging_steps=1,

        # Checkpoints
        save_strategy="epoch",
        save_total_limit=2,

        # Reproducibility
        seed=42,

        # Dataset processing
        packing=False,
        dataset_num_proc=None,

        # No external tracker
        report_to="none",
    )

    # ------------------------------------------------------
    # Trainer
    # ------------------------------------------------------

    trainer = SFTTrainer(
        model=model,
        args=training_config,

        train_dataset=train_dataset,
        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        peft_config=lora_config,

    )

    # ------------------------------------------------------
    # Parameter Statistics
    # ------------------------------------------------------

    print("=" * 60)
    print("Training Parameters")
    print("=" * 60)

    trainer.model.print_trainable_parameters()

    print()

    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------

    print("=" * 60)
    print("Starting SFT Training")
    print("=" * 60)

    trainer.train()

    trainer.save_model(OUTPUT_DIR)

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    print("=" * 60)
    print("Saving LoRA Adapter")
    print("=" * 60)

    trainer.save_model(OUTPUT_DIR)

    print("✓ Training complete.")
    print(
        f"Adapter saved to : "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()