"""
Experiment 02

Load a small Hugging Face model and inspect its configuration.

Purpose:
- Verify Transformers installation
- Understand tokenizer vs model
- Inspect model configuration
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"


def main():

    print("=" * 60)
    print("Loading Tokenizer...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Tokenizer Class : {tokenizer.__class__.__name__}")
    print(f"Vocabulary Size : {tokenizer.vocab_size}")
    print()

    print("=" * 60)
    print("Loading Model...")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
    )

    print()

    print("=" * 60)
    print("Model Information")
    print("=" * 60)

    print(f"Model Class : {model.__class__.__name__}")
    print(f"Model Name  : {MODEL_NAME}")

    print()
    print("Config")
    print(model.config)


if __name__ == "__main__":
    main()