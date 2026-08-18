import json
import torch

from pydantic import BaseModel, Field

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

JUDGE_MODEL = (
    "Qwen/Qwen2.5-1.5B-Instruct"
)


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
# Prompt
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

        # Remove Markdown code fences
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
        print("Raw judge response:")
        print(response)

        return None

def calculate_scores(judge_result):

    if not judge_result.root_cause_correct:
        root_cause_score = 0

    elif judge_result.root_cause_has_unsupported_claim:
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
# Main
# ----------------------------------------------------------

def main():

    print("=" * 60)
    print("LLM-AS-A-JUDGE")
    print("=" * 60)

    print(
        f"Judge Model : "
        f"{JUDGE_MODEL}"
    )

    print()

    # ------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------

    tokenizer = (
        AutoTokenizer.from_pretrained(
            JUDGE_MODEL
        )
    )

    # ------------------------------------------------------
    # Load model
    # ------------------------------------------------------

    print(
        "Loading judge model..."
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            JUDGE_MODEL,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    )

    model.eval()

    print(
        "✓ Judge model loaded."
    )

    print()

    # ------------------------------------------------------
    # Test example
    # ------------------------------------------------------

    failure = (
        "The ETL pipeline failed because the "
        "source database connection was reset "
        "while extracting records."
    )

    reference_root_cause = (
        "The connection to the source database "
        "was unexpectedly reset during data extraction."
    )

    predicted_root_cause = (
        "The source database connection was lost "
        "during extraction because of a network issue."
    )

    reference_recommendation = (
        "Check network and database connectivity, "
        "connection timeouts, and retry configuration "
        "before rerunning the extraction."
    )

    predicted_recommendation = (
        "Check network connectivity and configure "
        "retries before rerunning the extraction."
    )

    prompt = build_judge_prompt(
        failure,
        reference_root_cause,
        predicted_root_cause,
        reference_recommendation,
        predicted_recommendation,
    )

    print(
        "=" * 60
    )

    print(
        "Evaluating Example"
    )

    print(
        "=" * 60
    )

    response = generate_judge_response(
        model,
        tokenizer,
        prompt,
    )

    print()
    print(
        "Raw Judge Response:"
    )
    print(
        response
    )

    print()

    result = parse_judge_result(
        response
    )

    if result:

        root_cause_score, recommendation_score = (
            calculate_scores(result)
        )

        print(
            f"Root Cause Correct : "
            f"{result.root_cause_correct}"
        )

        print(
            f"Unsupported Root Cause Claim : "
            f"{result.root_cause_has_unsupported_claim}"
        )

        print(
            f"Recommendation Appropriate : "
            f"{result.recommendation_appropriate}"
        )

        print(
            f"Recommendation Complete : "
            f"{result.recommendation_complete}"
        )

        print()

        print(
            f"Root Cause Score : "
            f"{root_cause_score}"
        )

        print(
            f"Recommendation Score : "
            f"{recommendation_score}"
        )

    print(
        f"Explanation : "
        f"{result.explanation}"
    )



if __name__ == "__main__":
    main()