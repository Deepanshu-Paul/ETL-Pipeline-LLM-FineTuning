import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

ADAPTER_PATH = "outputs/qwen-etl-lora"

EVAL_PATH = "data/raw/etl_eval.json"


def load_eval_dataset():

    with open(
        EVAL_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def build_prompt(failure):

    return [
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
                "Analyze the ETL pipeline failure and "
                "identify the failure category, root cause, "
                "and recommended action.\n\n"
                f"ETL Failure:\n{failure}"
            ),
        },
    ]


def extract_category(response):

    response_upper = response.upper()

    categories = [
        "SCHEMA_ERROR",
        "AUTH_ERROR",
        "TIMEOUT_ERROR",
        "QUOTA_ERROR",
        "CONNECTION_ERROR",
        "DATA_ERROR",
    ]

    for category in categories:

        if category in response_upper:
            return category

    return None


def generate_response(
    model,
    tokenizer,
    messages,
):

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

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
        )

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )


def evaluate_model(
    model,
    tokenizer,
    eval_dataset,
    model_name,
):

    print("=" * 60)
    print(model_name)
    print("=" * 60)

    correct = 0

    total = len(eval_dataset)

    for index, example in enumerate(
        eval_dataset,
        start=1,
    ):

        messages = build_prompt(
            example["input"]
        )

        response = generate_response(
            model,
            tokenizer,
            messages,
        )

        predicted_category = extract_category(
            response
        )

        expected_category = (
            example["expected_category"]
        )

        is_correct = (
            predicted_category
            == expected_category
        )

        if is_correct:
            correct += 1

        print()
        print(
            f"Test Case {index}"
        )

        print(
            f"Expected  : "
            f"{expected_category}"
        )

        print(
            f"Predicted : "
            f"{predicted_category}"
        )

        print(
            f"Correct   : "
            f"{is_correct}"
        )

        print(
            f"Response  : "
            f"{response[:300]}"
        )

    accuracy = (
        correct / total
    ) * 100

    print()
    print("=" * 60)
    print("Evaluation Result")
    print("=" * 60)

    print(
        f"Correct : "
        f"{correct}/{total}"
    )

    print(
        f"Category Accuracy : "
        f"{accuracy:.2f}%"
    )

    print()


def main():

    eval_dataset = load_eval_dataset()

    print("=" * 60)
    print("Loading Tokenizer")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # ------------------------------------------------------
    # Base Model
    # ------------------------------------------------------

    print("=" * 60)
    print("Loading Base Model")
    print("=" * 60)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    evaluate_model(
        base_model,
        tokenizer,
        eval_dataset,
        "BASE MODEL",
    )

    del base_model

    torch.cuda.empty_cache()

    # ------------------------------------------------------
    # QLoRA Model
    # ------------------------------------------------------

    print("=" * 60)
    print("Loading QLoRA Adapter")
    print("=" * 60)

    qlora_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    qlora_model = PeftModel.from_pretrained(
        qlora_base,
        ADAPTER_PATH,
    )

    qlora_model.eval()

    print("✓ Adapter loaded.")

    evaluate_model(
        qlora_model,
        tokenizer,
        eval_dataset,
        "QLoRA MODEL",
    )


if __name__ == "__main__":
    main()