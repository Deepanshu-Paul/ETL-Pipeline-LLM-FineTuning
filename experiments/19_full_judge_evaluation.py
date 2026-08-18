import json
import torch

from pydantic import BaseModel, Field

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

JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

EVAL_PATH = "data/raw/etl_eval.json"


# ----------------------------------------------------------
# Judge Output Schema
# ----------------------------------------------------------

class JudgeResult(BaseModel):

    root_cause_correct: bool

    root_cause_has_unsupported_claim: bool

    recommendation_appropriate: bool

    recommendation_complete: bool

    explanation: str = Field(
        min_length=20
    )


# ----------------------------------------------------------
# Load Evaluation Dataset
# ----------------------------------------------------------

def load_eval_dataset():

    with open(
        EVAL_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ----------------------------------------------------------
# Build ETL Prompt
# ----------------------------------------------------------

def build_etl_prompt(failure):

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
# Generate Model Response
# ----------------------------------------------------------

def generate_response(
    model,
    tokenizer,
    failure,
):

    messages = build_etl_prompt(
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

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )


# ----------------------------------------------------------
# Build Judge Prompt
# ----------------------------------------------------------

def build_judge_prompt(
    failure,
    reference_root_cause,
    predicted_root_cause,
    reference_recommendation,
    predicted_recommendation,
):

    return f"""
You are evaluating an ETL failure analysis produced by another AI model.

Evaluate the predicted root cause and recommendation against
the reference answer.

ETL FAILURE:
{failure}

REFERENCE ROOT CAUSE:
{reference_root_cause}

PREDICTED ROOT CAUSE:
{predicted_root_cause}

REFERENCE RECOMMENDATION:
{reference_recommendation}

PREDICTED RECOMMENDATION:
{predicted_recommendation}

Evaluate the prediction using the following criteria.

ROOT CAUSE:

Determine whether the predicted root cause correctly identifies
the main underlying cause supported by the ETL failure.

Set root_cause_correct to true if the prediction correctly
identifies the underlying cause.

Set root_cause_has_unsupported_claim to true if the prediction
adds a specific causal claim that is NOT supported by the
ETL failure or reference answer.

For example, if the failure only says that a database connection
was reset, claiming that a firewall caused the reset would be an
unsupported claim unless the evidence supports it.

RECOMMENDATION:

Determine whether the predicted recommendation is technically
appropriate for the identified failure.

Set recommendation_appropriate to true if the recommendation
would reasonably help address the failure.

Set recommendation_complete to true if the recommendation
covers the important remediation steps from the reference.

Different wording is completely acceptable.

Do NOT require the prediction to use the same words as the
reference answer.

Judge technical meaning and correctness.

Return ONLY this JSON object:

{{
  "root_cause_correct": true,
  "root_cause_has_unsupported_claim": false,
  "recommendation_appropriate": true,
  "recommendation_complete": true,
  "explanation": "Briefly explain the assessment."
}}
""".strip()


# ----------------------------------------------------------
# Generate Judge Response
# ----------------------------------------------------------

def generate_judge_response(
    model,
    tokenizer,
    prompt,
):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict evaluator of "
                "ETL pipeline failure analyses."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
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

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
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


# ----------------------------------------------------------
# Parse Judge Result
# ----------------------------------------------------------

def parse_judge_result(response):

    try:

        response = response.strip()

        if response.startswith("```"):

            response = response.replace(
                "```json",
                "",
                1,
            )

            response = response.replace(
                "```",
                "",
                1,
            )

            response = response.strip()

        data = json.loads(
            response
        )

        return JudgeResult.model_validate(
            data
        )

    except Exception as error:

        print(
            "Judge parsing failed:"
        )

        print(error)

        print()
        print(
            "Raw judge response:"
        )

        print(response)

        return None


# ----------------------------------------------------------
# Deterministic Scoring
# ----------------------------------------------------------

def calculate_scores(
    judge_result,
):

    if not judge_result.root_cause_correct:

        root_cause_score = 0

    elif (
        judge_result
        .root_cause_has_unsupported_claim
    ):

        root_cause_score = 1

    else:

        root_cause_score = 2

    if not judge_result.recommendation_appropriate:

        recommendation_score = 0

    elif not judge_result.recommendation_complete:

        recommendation_score = 1

    else:

        recommendation_score = 2

    return (
        root_cause_score,
        recommendation_score,
    )


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

    return {
        "category": category,
        "root_cause": root_cause,
        "recommendation": recommendation,
    }


# ----------------------------------------------------------
# Evaluate One Model
# ----------------------------------------------------------

def evaluate_model(
    model,
    tokenizer,
    judge_model,
    judge_tokenizer,
    dataset,
    model_name,
):

    print("=" * 60)
    print(model_name)
    print("=" * 60)

    category_correct = 0

    valid_structure = 0

    root_scores = []

    recommendation_scores = []

    total = len(dataset)

    for index, example in enumerate(
        dataset,
        start=1,
    ):

        # --------------------------------------------------
        # Generate model response
        # --------------------------------------------------

        response = generate_response(
            model,
            tokenizer,
            example["input"],
        )

        extracted = extract_analysis(
            response
        )

        category = extracted[
            "category"
        ]

        # --------------------------------------------------
        # Category
        # --------------------------------------------------

        category_is_correct = (
            category
            == example[
                "expected_category"
            ]
        )

        if category_is_correct:

            category_correct += 1

        # --------------------------------------------------
        # Structural validation
        # --------------------------------------------------

        try:

            if (
                len(
                    extracted[
                        "category"
                    ]
                    or ""
                ) >= 3
                and len(
                    extracted[
                        "root_cause"
                    ]
                ) >= 20
                and len(
                    extracted[
                        "recommendation"
                    ]
                ) >= 20
            ):

                valid_structure += 1

                structure_valid = True

            else:

                structure_valid = False

        except Exception:

            structure_valid = False

        # --------------------------------------------------
        # Judge
        # --------------------------------------------------

        judge_prompt = build_judge_prompt(
            example["input"],
            example[
                "expected_root_cause"
            ],
            extracted[
                "root_cause"
            ],
            example[
                "expected_recommendation"
            ],
            extracted[
                "recommendation"
            ],
        )

        judge_response = (
            generate_judge_response(
                judge_model,
                judge_tokenizer,
                judge_prompt,
            )
        )

        judge_result = parse_judge_result(
            judge_response
        )

        if judge_result:

            (
                root_score,
                recommendation_score,
            ) = calculate_scores(
                judge_result
            )

            root_scores.append(
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
            f"{structure_valid}"
        )

        print(
            f"Root Cause Score : "
            f"{root_score}"
        )

        print(
            f"Recommendation Score : "
            f"{recommendation_score}"
        )

        print(
            f"Response : "
            f"{response[:300]}"
        )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    category_accuracy = (
        category_correct / total
    ) * 100

    structure_accuracy = (
        valid_structure / total
    ) * 100

    average_root_score = (
        sum(root_scores)
        / len(root_scores)
        if root_scores
        else 0
    )

    average_recommendation_score = (
        sum(recommendation_scores)
        / len(recommendation_scores)
        if recommendation_scores
        else 0
    )

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
        f"{structure_accuracy:.2f}%"
    )

    print(
        f"Average Root Cause Score : "
        f"{average_root_score:.2f} / 2"
    )

    print(
        f"Average Recommendation Score : "
        f"{average_recommendation_score:.2f} / 2"
    )

    return {
        "category_accuracy":
            category_accuracy,

        "structure_validity":
            structure_accuracy,

        "root_cause_score":
            average_root_score,

        "recommendation_score":
            average_recommendation_score,
    }


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    dataset = (
        load_eval_dataset()
    )

    print("=" * 60)
    print("Loading ETL Tokenizer")
    print("=" * 60)

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    print()

    # ------------------------------------------------------
    # Load Judge
    # ------------------------------------------------------

    print("=" * 60)
    print("Loading Judge Model")
    print("=" * 60)

    judge_tokenizer = (
        AutoTokenizer.from_pretrained(
            JUDGE_MODEL
        )
    )

    judge_model = (
        AutoModelForCausalLM.from_pretrained(
            JUDGE_MODEL,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    )

    judge_model.eval()

    print(
        "✓ Judge model loaded."
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

    base_results = evaluate_model(
        base_model,
        tokenizer,
        judge_model,
        judge_tokenizer,
        dataset,
        "BASE MODEL",
    )

    del base_model

    torch.cuda.empty_cache()

    # ------------------------------------------------------
    # QLoRA Model
    # ------------------------------------------------------

    print()
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

    qlora_results = evaluate_model(
        qlora_model,
        tokenizer,
        judge_model,
        judge_tokenizer,
        dataset,
        "QLoRA MODEL",
    )

    # ------------------------------------------------------
    # Final Comparison
    # ------------------------------------------------------

    print()
    print("=" * 60)
    print("BASE vs QLoRA")
    print("=" * 60)

    print()

    print(
        f"{'Metric':<35}"
        f"{'Base':>12}"
        f"{'QLoRA':>12}"
    )

    print("-" * 60)

    print(
        f"{'Category Accuracy':<35}"
        f"{base_results['category_accuracy']:>11.2f}%"
        f"{qlora_results['category_accuracy']:>11.2f}%"
    )

    print(
        f"{'Structure Validity':<35}"
        f"{base_results['structure_validity']:>11.2f}%"
        f"{qlora_results['structure_validity']:>11.2f}%"
    )

    print(
        f"{'Root Cause Score':<35}"
        f"{base_results['root_cause_score']:>12.2f}"
        f"{qlora_results['root_cause_score']:>12.2f}"
    )

    print(
        f"{'Recommendation Score':<35}"
        f"{base_results['recommendation_score']:>12.2f}"
        f"{qlora_results['recommendation_score']:>12.2f}"
    )


if __name__ == "__main__":
    main()