import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def print_gpu_memory(title):

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


def main():

    print("=" * 60)
    print("4-BIT NF4 MODEL LOADING")
    print("=" * 60)

    print(
        f"PyTorch : {torch.__version__}"
    )

    print(
        f"CUDA    : {torch.version.cuda}"
    )

    print(
        f"GPU     : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print()

    torch.cuda.empty_cache()

    print_gpu_memory(
        "GPU Memory Before Loading"
    )

    # ------------------------------------------------------
    # 4-bit Quantization Configuration
    # ------------------------------------------------------

    quantization_config = BitsAndBytesConfig(

        load_in_4bit=True,

        bnb_4bit_quant_type="nf4",

        bnb_4bit_compute_dtype=torch.bfloat16,

        bnb_4bit_use_double_quant=True,
    )

    print("=" * 60)
    print("Quantization Configuration")
    print("=" * 60)

    print(
        "4-bit loading       : True"
    )

    print(
        "Quantization type   : NF4"
    )

    print(
        "Compute dtype       : BF16"
    )

    print(
        "Double quantization : True"
    )

    print()

    # ------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # ------------------------------------------------------
    # Load Quantized Model
    # ------------------------------------------------------

    print("=" * 60)
    print("Loading 4-Bit Model")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(

        MODEL_NAME,

        quantization_config=quantization_config,

        device_map="auto",
    )

    print("✓ 4-bit model loaded.")
    print()

    # ------------------------------------------------------
    # Model Information
    # ------------------------------------------------------

    print("=" * 60)
    print("Model Information")
    print("=" * 60)

    print(
        f"Model Class : "
        f"{model.__class__.__name__}"
    )

    print(
        f"Embedding dtype : "
        f"{model.get_input_embeddings().weight.dtype}"
    )

    print()

    # ------------------------------------------------------
    # GPU Memory
    # ------------------------------------------------------

    print_gpu_memory(
        "GPU Memory After 4-Bit Loading"
    )

    # ------------------------------------------------------
    # Quantization Check
    # ------------------------------------------------------

    print("=" * 60)
    print("Quantization Check")
    print("=" * 60)

    print(
        f"is_loaded_in_4bit : "
        f"{getattr(model, 'is_loaded_in_4bit', False)}"
    )

    print(
        f"is_loaded_in_8bit : "
        f"{getattr(model, 'is_loaded_in_8bit', False)}"
    )

    print()

    # ------------------------------------------------------
    # Simple Inference Check
    # ------------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": (
                "What is a schema error "
                "in an ETL pipeline?"
            ),
        }
    ]

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

    print("=" * 60)
    print("Inference Check")
    print("=" * 60)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    print(response)
    print()


if __name__ == "__main__":
    main()