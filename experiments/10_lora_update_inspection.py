"""
Experiment 10

LoRA Weight Update Inspection

Purpose:
- Load the trained LoRA adapter.
- Inspect LoRA A and B matrices.
- Compute the effective low-rank update:
      Delta W = B @ A
- Compare the LoRA update magnitude with
  the original base weight magnitude.
"""

import torch

from transformers import AutoModelForCausalLM

from peft import PeftModel


# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

ADAPTER_PATH = "outputs/qwen-etl-lora"


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

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

    print("=" * 60)
    print("Loading LoRA Adapter")
    print("=" * 60)

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    print("✓ LoRA adapter loaded.")
    print()

    # ------------------------------------------------------
    # Find first LoRA layer
    # ------------------------------------------------------

    for name, module in model.named_modules():

        if hasattr(module, "lora_A"):

            print("=" * 60)
            print("LoRA Layer Inspection")
            print("=" * 60)

            print(
                f"Layer Name : {name}"
            )

            print(
                f"Layer Type : "
                f"{module.__class__.__name__}"
            )

            # --------------------------------------------------
            # Select the active adapter
            # --------------------------------------------------

            adapter_name = "default"

            lora_A = (
                module.lora_A[adapter_name]
                .weight
            )

            lora_B = (
                module.lora_B[adapter_name]
                .weight
            )

            # --------------------------------------------------
            # Original Base Weight
            # --------------------------------------------------

            base_weight = module.base_layer.weight

            # --------------------------------------------------
            # LoRA Update
            # --------------------------------------------------

            delta_W = (
                lora_B.float()
                @ lora_A.float()
            )

            # Account for LoRA scaling.
            scaling = (
                module.scaling[adapter_name]
            )

            delta_W = (
                delta_W * scaling
            )

            # --------------------------------------------------
            # Statistics
            # --------------------------------------------------

            print()

            print("Original Weight")
            print(
                f"Shape : "
                f"{tuple(base_weight.shape)}"
            )

            print()

            print("LoRA A")
            print(
                f"Shape : "
                f"{tuple(lora_A.shape)}"
            )

            print()

            print("LoRA B")
            print(
                f"Shape : "
                f"{tuple(lora_B.shape)}"
            )

            print()

            print("Effective LoRA Update")
            print(
                f"Delta W Shape : "
                f"{tuple(delta_W.shape)}"
            )

            print()

            # --------------------------------------------------
            # Norms
            # --------------------------------------------------

            base_norm = (
                base_weight.float()
                .norm()
            )

            a_norm = (
                lora_A.float()
                .norm()
            )

            b_norm = (
                lora_B.float()
                .norm()
            )

            delta_norm = (
                delta_W.norm()
            )

            relative_update = (
                delta_norm / base_norm
            )

            print("=" * 60)
            print("LoRA Update Magnitude")
            print("=" * 60)

            print(
                f"||W||          : "
                f"{base_norm.item():.6f}"
            )

            print(
                f"||A||          : "
                f"{a_norm.item():.6f}"
            )

            print(
                f"||B||          : "
                f"{b_norm.item():.6f}"
            )

            print(
                f"||Delta W||    : "
                f"{delta_norm.item():.6f}"
            )

            print(
                f"||Delta W|| / ||W|| : "
                f"{relative_update.item():.8f}"
            )

            print()

            # --------------------------------------------------
            # Parameter Count
            # --------------------------------------------------

            print("=" * 60)
            print("Parameter Count")
            print("=" * 60)

            print(
                f"LoRA A Parameters : "
                f"{lora_A.numel():,}"
            )

            print(
                f"LoRA B Parameters : "
                f"{lora_B.numel():,}"
            )

            print(
                f"Total LoRA Parameters : "
                f"{lora_A.numel() + lora_B.numel():,}"
            )

            print()

            # --------------------------------------------------
            # Only inspect first LoRA layer
            # --------------------------------------------------

            break


if __name__ == "__main__":
    main()