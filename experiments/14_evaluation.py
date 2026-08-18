import json
import torch
from pydantic import BaseModel,Field
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

ADAPTER_PATH = "outputs/qwen-etl-lora"

EVAL_PATH = "data/raw/etl_eval.json"

class ETLAnalysis(BaseModel):

    category: str = Field(
        min_length=3
    )

    root_cause: str = Field(
        min_length=20
    )

    recommendation: str = Field(
        min_length=20
    )

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

    text = response.upper()

    # ------------------------------------------------------
    # Exact canonical labels
    # ------------------------------------------------------

    exact_categories = [
        "SCHEMA_ERROR",
        "AUTH_ERROR",
        "TIMEOUT_ERROR",
        "QUOTA_ERROR",
        "CONNECTION_ERROR",
        "DATA_ERROR",
    ]

    for category in exact_categories:

        if category in text:
            return category

    # ------------------------------------------------------
    # Semantic normalization
    # ------------------------------------------------------

    if any(
        phrase in text
        for phrase in [
            "SCHEMA",
            "SCHEMA ERROR",
            "SCHEMA MISMATCH",
            "DESTINATION SCHEMA",
            "TARGET SCHEMA",
            "FIELD DOES NOT EXIST",
            "COLUMN DOES NOT EXIST",
        ]
    ):
        return "SCHEMA_ERROR"

    if any(
        phrase in text
        for phrase in [
            "AUTHENTICATION",
            "AUTHENTICATION ERROR",
            "PERMISSION ERROR",
            "PERMISSION DENIED",
            "ACCESS DENIED",
            "IAM",
            "CREDENTIAL",
        ]
    ):
        return "AUTH_ERROR"

    if any(
        phrase in text
        for phrase in [
            "TIMEOUT",
            "TIME OUT",
            "EXECUTION TIME EXCEEDED",
            "EXECUTION TIME",
            "TIME LIMIT",
        ]
    ):
        return "TIMEOUT_ERROR"

    if any(
        phrase in text
        for phrase in [
            "QUOTA",
            "QUOTA EXCEEDED",
            "RATE LIMIT",
        ]
    ):
        return "QUOTA_ERROR"

    if any(
        phrase in text
        for phrase in [
            "CONNECTION RESET",
            "DATABASE CONNECTION",
            "CONNECTION FAILURE",
            "CONNECTION ERROR",
            "NETWORK CONNECTION",
        ]
    ):
        return "CONNECTION_ERROR"

    if any(
        phrase in text
        for phrase in [
            "DATA VALIDATION",
            "DATA TRANSFORMATION",
            "DATA ERROR",
            "INVALID DATA",
            "INVALID VALUE",
            "TYPE CONVERSION",
        ]
    ):
        return "DATA_ERROR"

    return None

def validate_analysis(data):

    try:
        analysis = ETLAnalysis.model_validate(data)

        return analysis, True

    except Exception as error:

        print(
            f"Pydantic validation failed: {error}"
        )

        return None, False

def extract_analysis(response):

    category = extract_category(response)
#-------------------------------------------------
#----------------TEMP----------------------------
    print()
    print("DEBUG CATEGORY")
    print(f"Extracted category : {category}")
    print()
#-------------------------------------------------
#----------------TEMP----------------------------

    response_lower = response.lower()

    root_cause = ""
    recommendation = ""

    # ------------------------------------------------------
    # Root Cause
    # ------------------------------------------------------

    root_cause_markers = [
        "root cause:",
        "**root cause:**",
        "root cause",
    ]

    for marker in root_cause_markers:

        if marker in response_lower:

            start = (
                response_lower.find(marker)
                + len(marker)
            )

            remaining = response[start:]

            recommendation_position = (
                remaining.lower().find(
                    "recommended action"
                )
            )

            if recommendation_position != -1:

                root_cause = (
                    remaining[
                        :recommendation_position
                    ]
                    .strip()
                )

            else:

                root_cause = remaining.strip()

            break

    # ------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------

    recommendation_markers = [
        "recommended action:",
        "**recommended action:**",
        "recommended action",
    ]

    for marker in recommendation_markers:

        if marker in response_lower:

            start = (
                response_lower.find(marker)
                + len(marker)
            )

            recommendation = (
                response[start:]
                .strip()
            )

            break

    data = {
        "category": category,
        "root_cause": root_cause,
        "recommendation": recommendation,
    }

    return data

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

        analysis_data = extract_analysis(
            response
        )

        analysis, is_valid = validate_analysis(
            analysis_data
        )

        if analysis is not None:
            predicted_category = analysis.category
        else:
            predicted_category = None

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
            f"Valid Structure : "
            f"{is_valid}"
        )

        if analysis is not None:

            print(
                f"Category        : "
                f"{analysis.category}"
            )

            print(
                f"Root Cause      : "
                f"{analysis.root_cause[:150]}"
            )

            print(
                f"Recommendation  : "
                f"{analysis.recommendation[:150]}"
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