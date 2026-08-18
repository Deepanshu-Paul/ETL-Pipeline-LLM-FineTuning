import json
import re

import torch

from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel


# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

ADAPTER_PATH = "outputs/qwen-etl-lora"

EVAL_PATH = "data/raw/etl_eval.json"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ----------------------------------------------------------
# Output Schema
# ----------------------------------------------------------

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


# ----------------------------------------------------------
# Dataset
# ----------------------------------------------------------

def load_eval_dataset():

    with open(
        EVAL_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ----------------------------------------------------------
# Prompt
# ----------------------------------------------------------

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


# ----------------------------------------------------------
# Category Extraction
# ----------------------------------------------------------

def extract_category(response):

    text = response.upper()

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


# ----------------------------------------------------------
# Structured Extraction
# ----------------------------------------------------------

def extract_analysis(response):

    category = extract_category(
        response
    )

    response_lower = response.lower()

    root_cause = ""
    recommendation = ""

    # ------------------------------------------------------
    # Root Cause
    # ------------------------------------------------------

    root_markers = [
        "root cause:",
        "**root cause:**",
        "root cause",
    ]

    for marker in root_markers:

        position = response_lower.find(
            marker
        )

        if position != -1:

            start = (
                position + len(marker)
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

                root_cause = (
                    remaining.strip()
                )

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

        position = response_lower.find(
            marker
        )

        if position != -1:

            start = (
                position + len(marker)
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


# ----------------------------------------------------------
# Pydantic Validation
# ----------------------------------------------------------

def validate_analysis(data):

    try:

        analysis = (
            ETLAnalysis.model_validate(
                data
            )
        )

        return analysis

    except Exception:

        return None


# ----------------------------------------------------------
# Model Generation
# ----------------------------------------------------------

def generate_response(
    model,
    tokenizer,
    failure,
):

    messages = build_prompt(
        failure
    )

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
            max_new_tokens=160,
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

    return response


# ----------------------------------------------------------
# Semantic Similarity
# ----------------------------------------------------------

def semantic_similarity(
    embedding_model,
    prediction,
    reference,
):

    embeddings = embedding_model.encode(
        [
            prediction,
            reference,
        ],
        convert_to_tensor=True,
    )

    return cos_sim(
        embeddings[0],
        embeddings[1],
    ).item()


# ----------------------------------------------------------
# Evaluate Model
# ----------------------------------------------------------

def evaluate_model(
    model,
    tokenizer,
    eval_dataset,
    embedding_model,
    model_name,
):

    print("=" * 60)
    print(model_name)
    print("=" * 60)

    category_correct = 0

    valid_structures = 0

    root_cause_scores = []

    recommendation_scores = []

    total = len(
        eval_dataset
    )

    for index, example in enumerate(
        eval_dataset,
        start=1,
    ):

        response = generate_response(
            model,
            tokenizer,
            example["input"],
        )

        extracted = extract_analysis(
            response
        )

        analysis = validate_analysis(
            extracted
        )

        category = extracted[
            "category"
        ]

        category_is_correct = (
            category
            == example[
                "expected_category"
            ]
        )

        if category_is_correct:
            category_correct += 1

        if analysis is not None:

            valid_structures += 1

            root_score = (
                semantic_similarity(
                    embedding_model,
                    analysis.root_cause,
                    example[
                        "expected_root_cause"
                    ],
                )
            )

            recommendation_score = (
                semantic_similarity(
                    embedding_model,
                    analysis.recommendation,
                    example[
                        "expected_recommendation"
                    ],
                )
            )

            root_cause_scores.append(
                root_score
            )

            recommendation_scores.append(
                recommendation_score
            )

        else:

            root_score = None
            recommendation_score = None

        # --------------------------------------------------
        # Case Output
        # --------------------------------------------------

        print()
        print(
            f"Test Case {index}"
        )

        print(
            f"Expected Category : "
            f"{example['expected_category']}"
        )

        print(
            f"Predicted Category : "
            f"{category}"
        )

        print(
            f"Category Correct : "
            f"{category_is_correct}"
        )

        print(
            f"Valid Structure : "
            f"{analysis is not None}"
        )

        if root_score is not None:

            print(
                f"Root Cause Similarity : "
                f"{root_score:.4f}"
            )

            print(
                f"Recommendation Similarity : "
                f"{recommendation_score:.4f}"
            )

        print()
        print(
            f"Response : "
            f"{response[:500]}"
        )

    # ------------------------------------------------------
    # Aggregate Metrics
    # ------------------------------------------------------

    category_accuracy = (
        category_correct / total
    ) * 100

    structure_validity = (
        valid_structures / total
    ) * 100

    if root_cause_scores:

        average_root_cause = (
            sum(root_cause_scores)
            / len(root_cause_scores)
        )

    else:

        average_root_cause = 0.0

    if recommendation_scores:

        average_recommendation = (
            sum(recommendation_scores)
            / len(recommendation_scores)
        )

    else:

        average_recommendation = 0.0

    print()
    print("=" * 60)
    print(
        f"{model_name} SUMMARY"
    )
    print("=" * 60)

    print(
        f"Category Accuracy : "
        f"{category_accuracy:.2f}%"
    )

    print(
        f"Structure Validity : "
        f"{structure_validity:.2f}%"
    )

    print(
        f"Average Root Cause Similarity : "
        f"{average_root_cause:.4f}"
    )

    print(
        f"Average Recommendation Similarity : "
        f"{average_recommendation:.4f}"
    )

    print()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    eval_dataset = (
        load_eval_dataset()
    )

    print("=" * 60)
    print("Loading Tokenizer")
    print("=" * 60)

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    print()

    # ------------------------------------------------------
    # Embedding Model
    # ------------------------------------------------------

    print("=" * 60)
    print("Loading Evaluation Model")
    print("=" * 60)

    embedding_model = (
        SentenceTransformer(
            EMBEDDING_MODEL
        )
    )

    print(
        "✓ Embedding model loaded."
    )

    print()

    # ------------------------------------------------------
    # Base Model
    # ------------------------------------------------------

    print("=" * 60)
    print("Loading Base Model")
    print("=" * 60)

    base_model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    )

    evaluate_model(
        base_model,
        tokenizer,
        eval_dataset,
        embedding_model,
        "BASE MODEL",
    )

    del base_model

    torch.cuda.empty_cache()

    # ------------------------------------------------------
    # QLoRA Model
    # ------------------------------------------------------

    print("=" * 60)
    print("Loading QLoRA Model")
    print("=" * 60)

    qlora_base = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    )

    qlora_model = (
        PeftModel.from_pretrained(
            qlora_base,
            ADAPTER_PATH,
        )
    )

    qlora_model.eval()

    print(
        "✓ QLoRA adapter loaded."
    )

    print()

    evaluate_model(
        qlora_model,
        tokenizer,
        eval_dataset,
        embedding_model,
        "QLoRA MODEL",
    )


if __name__ == "__main__":
    main()