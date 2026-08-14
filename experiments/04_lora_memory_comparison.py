"""
Experiment 04

LoRA Memory Comparison

Purpose:
- Load the base model.
- Establish a baseline for parameter count and GPU memory.
- Later compare full fine-tuning with LoRA.

This experiment will gradually evolve to measure:
- Total parameters
- Trainable parameters
- Frozen parameters
- GPU memory
- LoRA trainable parameters
- LoRA memory usage
"""

import torch

from transformers import AutoModelForCausalLM


# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    print("=" * 60)
    print("Loading Base Model")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
    )

    print("✓ Model loaded.")
    print()

    print("=" * 60)
    print("Base Model")
    print("=" * 60)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Total Parameters     : {total_parameters:,}")
    print(f"Trainable Parameters : {trainable_parameters:,}")
    print(
        f"Frozen Parameters    : "
        f"{total_parameters - trainable_parameters:,}"
    )

    print()

    if torch.cuda.is_available():

        device_index = model.device.index or 0

        allocated = torch.cuda.memory_allocated(device_index)
        reserved = torch.cuda.memory_reserved(device_index)

        print("=" * 60)
        print("GPU Memory")
        print("=" * 60)

        print(
            f"GPU Allocated : "
            f"{allocated / (1024 ** 2):.2f} MB"
        )

        print(
            f"GPU Reserved  : "
            f"{reserved / (1024 ** 2):.2f} MB"
        )

        print()


if __name__ == "__main__":
    main()