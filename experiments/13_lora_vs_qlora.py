import gc
import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def print_memory(title):

    print("=" * 60)
    print(title)
    print("=" * 60)

    print(
        f"GPU Allocated : "
        f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )

    print(
        f"GPU Reserved  : "
        f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB"
    )

    print()


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


def cleanup():

    gc.collect()
    torch.cuda.empty_cache()


def main():

    print("=" * 60)
    print("LoRA vs QLoRA Memory Comparison")
    print("=" * 60)

    print(
        f"GPU : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print()

    # ======================================================
    # LoRA — BF16 Base
    # ======================================================

    cleanup()

    print("=" * 60)
    print("Loading BF16 Base Model + LoRA")
    print("=" * 60)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = create_lora_config()

    lora_model = get_peft_model(
        base_model,
        lora_config,
    )

    print("✓ BF16 LoRA model loaded.")

    print_memory(
        "BF16 LoRA GPU Memory"
    )

    lora_model.print_trainable_parameters()

    # Record memory

    lora_memory = (
        torch.cuda.memory_allocated()
    )

    # Remove model

    del lora_model
    del base_model

    cleanup()

    # ======================================================
    # QLoRA — 4-bit NF4 Base
    # ======================================================

    print("=" * 60)
    print("Loading 4-Bit NF4 Base Model + LoRA")
    print("=" * 60)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    qlora_base_model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
        )
    )

    qlora_model = get_peft_model(
        qlora_base_model,
        create_lora_config(),
    )

    print("✓ 4-bit QLoRA model loaded.")

    print_memory(
        "4-Bit QLoRA GPU Memory"
    )

    qlora_model.print_trainable_parameters()

    qlora_memory = (
        torch.cuda.memory_allocated()
    )

    # ======================================================
    # Comparison
    # ======================================================

    print()
    print("=" * 60)
    print("MEMORY COMPARISON")
    print("=" * 60)

    print(
        f"BF16 LoRA : "
        f"{lora_memory / 1024**2:.2f} MB"
    )

    print(
        f"QLoRA     : "
        f"{qlora_memory / 1024**2:.2f} MB"
    )

    reduction = (
        1 -
        qlora_memory / lora_memory
    ) * 100

    print(
        f"Memory Reduction : "
        f"{reduction:.2f}%"
    )

    print()


if __name__ == "__main__":
    main()