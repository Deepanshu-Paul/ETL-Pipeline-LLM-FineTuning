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
from peft import LoraConfig, get_peft_model


# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def inspect_lora_layer(model):
    """
    Inspect the LoRA matrices attached to one attention projection.
    """

    print("=" * 60)
    print("LoRA Layer Inspection")
    print("=" * 60)

    # First Transformer layer -> self attention -> q_proj
    q_proj = model.base_model.model.model.layers[0].self_attn.q_proj

    print(f"Layer Type : {q_proj.__class__.__name__}")
    print()

    print("Original Weight")
    print(f"Shape : {tuple(q_proj.base_layer.weight.shape)}")
    print()

    print("LoRA A")
    print(
        f"Shape : "
        f"{tuple(q_proj.lora_A['default'].weight.shape)}"
    )
    print()

    print("LoRA B")
    print(
        f"Shape : "
        f"{tuple(q_proj.lora_B['default'].weight.shape)}"
    )

    print()

def run_training_probe(model):
    """
    Run one tiny forward + backward pass to observe
    training-time memory usage.

    This is NOT the actual fine-tuning loop.
    It is only a memory/gradient experiment.
    """

    print("=" * 60)
    print("Training Memory Probe")
    print("=" * 60)

    device = model.device

    # A tiny dummy sequence of token IDs.
    input_ids = torch.tensor(
        [[785, 468, 13470, 15301, 4641, 13]],
        dtype=torch.long,
        device=device,
    )

    print(f"Input Shape : {tuple(input_ids.shape)}")

    # ----------------------------------------------------------
    # Forward Pass
    # ----------------------------------------------------------

    outputs = model(
        input_ids=input_ids,
        labels=input_ids,
    )

    loss = outputs.loss

    print(f"Loss        : {loss.item():.6f}")

    # ----------------------------------------------------------
    # Backward Pass
    # ----------------------------------------------------------

    loss.backward()

    print("✓ Backward pass completed.")
    print()

    # ----------------------------------------------------------
    # Memory After Backward
    # ----------------------------------------------------------

    if torch.cuda.is_available():

        device_index = model.device.index or 0

        allocated = torch.cuda.memory_allocated(device_index)
        reserved = torch.cuda.memory_reserved(device_index)

        print("=" * 60)
        print("GPU Memory After Backward")
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

def inspect_gradients(model):
    """
    Verify which parameters received gradients
    after the backward pass.
    """

    print("=" * 60)
    print("Gradient Inspection")
    print("=" * 60)

    base_gradient_count = 0
    lora_gradient_count = 0

    for name, parameter in model.named_parameters():

        if parameter.grad is not None:

            if "lora_" in name:
                lora_gradient_count += 1

                print(
                    f"LoRA gradient : "
                    f"{name} | "
                    f"shape={tuple(parameter.grad.shape)}"
                )

            else:
                base_gradient_count += 1

    print()
    print(f"Base parameters with gradients : {base_gradient_count}")
    print(f"LoRA parameters with gradients  : {lora_gradient_count}")
    print()


def run_optimizer_probe(model):
    """
    Perform one optimizer step and verify that
    a LoRA parameter actually changes.
    """

    print("=" * 60)
    print("Optimizer Update Probe")
    print("=" * 60)

    device = model.device

    input_ids = torch.tensor(
        [[785, 468, 13470, 15301, 4641, 13]],
        dtype=torch.long,
        device=device,
    )

    # ----------------------------------------------------------
    # Optimizer
    # ----------------------------------------------------------

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=1e-3,
    )

    # Pick one LoRA parameter to observe.
    lora_parameter = model.base_model.model.model.layers[
        0
    ].self_attn.q_proj.lora_A["default"].weight

    value_before = lora_parameter.detach().clone()

    # ----------------------------------------------------------
    # Forward Pass
    # ----------------------------------------------------------

    outputs = model(
        input_ids=input_ids,
        labels=input_ids,
    )

    loss = outputs.loss

    print(f"Loss Before Update : {loss.item():.6f}")

    # ----------------------------------------------------------
    # Backward Pass
    # ----------------------------------------------------------

    optimizer.zero_grad()

    loss.backward()

    # ----------------------------------------------------------
    # Optimizer Step
    # ----------------------------------------------------------

    optimizer.step()

    value_after = lora_parameter.detach().clone()

    difference = torch.abs(
        value_after - value_before
    ).max().item()

    print(
        f"Maximum Weight Change : "
        f"{difference:.10f}"
    )

    print()

    if difference > 0:
        print("✓ LoRA parameter was updated.")
    else:
        print("✗ LoRA parameter did not change.")

    print()

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

    # ----------------------------------------------------------
    # Apply LoRA
    # ----------------------------------------------------------

    print("=" * 60)
    print("Applying LoRA")
    print("=" * 60)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    print("✓ LoRA applied.")
    print()

    # ----------------------------------------------------------
    # LoRA Parameter Statistics
    # ----------------------------------------------------------

    print("=" * 60)
    print("LoRA Parameter Statistics")
    print("=" * 60)

    model.print_trainable_parameters()

    inspect_lora_layer(model)

    run_training_probe(model)

    inspect_gradients(model)

    run_optimizer_probe(model)

    # ----------------------------------------------------------
    # GPU Memory After LoRA
    # ----------------------------------------------------------

    if torch.cuda.is_available():

        device_index = model.device.index or 0

        allocated = torch.cuda.memory_allocated(device_index)
        reserved = torch.cuda.memory_reserved(device_index)

        print("=" * 60)
        print("GPU Memory After LoRA")
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