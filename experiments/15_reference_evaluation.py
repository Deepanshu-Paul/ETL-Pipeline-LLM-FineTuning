import json
import re

from pydantic import BaseModel, Field


# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

EVAL_PATH = "data/raw/etl_eval.json"


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
# Text Normalization
# ----------------------------------------------------------

def normalize_text(text):

    text = text.lower()

    # Remove markdown
    text = re.sub(
        r"[*#`]",
        " ",
        text,
    )

    # Remove punctuation
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ----------------------------------------------------------
# Simple Word Overlap
# ----------------------------------------------------------

def word_overlap(
    reference,
    prediction,
):

    reference_words = set(
        normalize_text(
            reference
        ).split()
    )

    prediction_words = set(
        normalize_text(
            prediction
        ).split()
    )

    if not reference_words:
        return 0.0

    intersection = (
        reference_words
        & prediction_words
    )

    return (
        len(intersection)
        / len(reference_words)
    )


# ----------------------------------------------------------
# Evaluate One Prediction
# ----------------------------------------------------------

def evaluate_prediction(
    prediction,
    reference,
):

    root_cause_score = word_overlap(
        reference["expected_root_cause"],
        prediction.root_cause,
    )

    recommendation_score = word_overlap(
        reference["expected_recommendation"],
        prediction.recommendation,
    )

    return {
        "category_correct": (
            prediction.category
            == reference["expected_category"]
        ),
        "root_cause_score": (
            root_cause_score
        ),
        "recommendation_score": (
            recommendation_score
        ),
    }


# ----------------------------------------------------------
# Demonstration
# ----------------------------------------------------------

def main():

    dataset = load_eval_dataset()

    print("=" * 60)
    print("REFERENCE-BASED EVALUATION")
    print("=" * 60)

    print(
        f"Evaluation examples : "
        f"{len(dataset)}"
    )

    print()

    # ------------------------------------------------------
    # Show reference structure
    # ------------------------------------------------------

    example = dataset[0]

    print("=" * 60)
    print("Example Reference")
    print("=" * 60)

    print(
        f"Category : "
        f"{example['expected_category']}"
    )

    print(
        f"Root Cause : "
        f"{example['expected_root_cause']}"
    )

    print(
        f"Recommendation : "
        f"{example['expected_recommendation']}"
    )

    print()

    # ------------------------------------------------------
    # Demonstrate scoring
    # ------------------------------------------------------

    prediction = ETLAnalysis(
        category=example[
            "expected_category"
        ],
        root_cause=example[
            "expected_root_cause"
        ],
        recommendation=example[
            "expected_recommendation"
        ],
    )

    result = evaluate_prediction(
        prediction,
        example,
    )

    print("=" * 60)
    print("Perfect Prediction Test")
    print("=" * 60)

    print(
        f"Category Correct : "
        f"{result['category_correct']}"
    )

    print(
        f"Root Cause Score : "
        f"{result['root_cause_score']:.3f}"
    )

    print(
        f"Recommendation Score : "
        f"{result['recommendation_score']:.3f}"
    )

    print()


if __name__ == "__main__":
    main()