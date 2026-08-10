"""
Experiment 02

LLM Inspector

Purpose:
--------
Load an LLM from Hugging Face and inspect its architecture,
configuration, tokenizer, and hardware usage.

This experiment will gradually evolve throughout the project.
Initially it only loads the model and prints basic information.
Later we will add:

- Parameter inspection
- GPU memory usage
- Architecture visualization
- Tokenizer analysis
- LoRA inspection
- Quantization comparison
"""

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# ----------------------------------------------------------
# Loading Functions
# ----------------------------------------------------------

def load_tokenizer(model_name: str):
    """
    Download and load the tokenizer.

    Returns
    -------
    AutoTokenizer
    """

    print("=" * 60)
    print("Loading Tokenizer...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("✓ Tokenizer loaded.\n")

    return tokenizer


def load_model(model_name: str):
    """
    Download and load the language model.

    Returns
    -------
    AutoModelForCausalLM
    """

    print("=" * 60)
    print("Loading Model...")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )

    print("✓ Model loaded.\n")

    return model


# ----------------------------------------------------------
# Inspection Functions
# ----------------------------------------------------------

def inspect_tokenizer(tokenizer):
    """
    Display basic tokenizer information.
    """

    print("=" * 60)
    print("Tokenizer")
    print("=" * 60)

    print(f"Tokenizer Class : {tokenizer.__class__.__name__}")
    print(f"Vocabulary Size : {tokenizer.vocab_size}")
    print()

def inspect_parameters(model):
    """
    Display model parameter statistics.

    Parameters
    ----------
    model:
        Loaded Hugging Face causal language model.
    """

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_parameters = total_parameters - trainable_parameters

    trainable_percentage = (
        trainable_parameters / total_parameters * 100
        if total_parameters > 0
        else 0
    )

    print("=" * 60)
    print("Parameter Statistics")
    print("=" * 60)

    print(f"Total Parameters     : {total_parameters:,}")
    print(f"Trainable Parameters : {trainable_parameters:,}")
    print(f"Frozen Parameters    : {frozen_parameters:,}")
    print(f"Trainable Percentage : {trainable_percentage:.4f}%")
    print()

def inspect_config(model):
    """
    Display the model configuration.
    """

    print("=" * 60)
    print("Model Configuration")
    print("=" * 60)

    print(model.config)
    print()

def inspect_memory(model):
    """
    Display model memory and device information.
    """

    print("=" * 60)
    print("Memory & Device")
    print("=" * 60)

    print(f"Model Device : {model.device}")
    print(f"Model Dtype  : {model.dtype}")

    if model.device.type == "cuda":
        device_index = model.device.index or 0

        allocated = torch.cuda.memory_allocated(device_index)
        reserved = torch.cuda.memory_reserved(device_index)

        print(
            f"GPU Allocated : {allocated / (1024 ** 2):.2f} MB"
        )
        print(
            f"GPU Reserved  : {reserved / (1024 ** 2):.2f} MB"
        )

        total_memory = torch.cuda.get_device_properties(
            device_index
        ).total_memory

        print(
            f"GPU Total     : {total_memory / (1024 ** 3):.2f} GB"
        )

    else:
        print("GPU Allocated : N/A")
        print("GPU Reserved  : N/A")
        print("GPU Total     : N/A")

    print()

def inspect_tokenization(tokenizer):
    """
    Demonstrate how an ETL-related sentence is converted
    into tokens and token IDs.
    """

    text = (
        "The ETL pipeline failed because the source schema "
        "changed unexpectedly."
    )

    print("=" * 60)
    print("Tokenizer Demonstration")
    print("=" * 60)

    print(f"Input Text : {text}")

    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text, add_special_tokens=True)

    print(f"Tokens     : {tokens}")
    print(f"Token IDs  : {token_ids}")
    print(f"Token Count: {len(token_ids)}")

    print()

def inspect_vocabulary(tokenizer, model):
    """
    Compare tokenizer vocabulary information with
    the model's embedding vocabulary.
    """

    print("=" * 60)
    print("Vocabulary Inspection")
    print("=" * 60)

    tokenizer_vocab = tokenizer.get_vocab()
    model_vocab = model.config.vocab_size

    print(f"Tokenizer vocab_size       : {tokenizer.vocab_size}")
    print(f"Tokenizer get_vocab() size : {len(tokenizer_vocab)}")
    print(f"Model vocab_size           : {model_vocab}")
    print(f"Difference                 : {model_vocab - len(tokenizer_vocab)}")

    print()
    print("Special Tokens")
    print(f"PAD token : {tokenizer.pad_token} ({tokenizer.pad_token_id})")
    print(f"BOS token : {tokenizer.bos_token} ({tokenizer.bos_token_id})")
    print(f"EOS token : {tokenizer.eos_token} ({tokenizer.eos_token_id})")

    print()

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    tokenizer = load_tokenizer(MODEL_NAME)

    model = load_model(MODEL_NAME)

    inspect_tokenizer(tokenizer)

    inspect_config(model)

    inspect_parameters(model)

    inspect_memory(model)

    inspect_tokenization(tokenizer)

    inspect_vocabulary(tokenizer, model)


if __name__ == "__main__":
    main()