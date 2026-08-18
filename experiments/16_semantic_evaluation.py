import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():

    print("=" * 60)
    print("SEMANTIC SIMILARITY TEST")
    print("=" * 60)

    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("✓ Embedding model loaded.")
    print()

    # ------------------------------------------------------
    # Example 1 — Similar meaning, different wording
    # ------------------------------------------------------

    reference_1 = (
        "The source database connection was "
        "unexpectedly reset during data extraction."
    )

    prediction_1 = (
        "The database lost connectivity while "
        "records were being extracted."
    )

    # ------------------------------------------------------
    # Example 2 — Different meaning
    # ------------------------------------------------------

    reference_2 = (
        "The source database connection was "
        "unexpectedly reset during data extraction."
    )

    prediction_2 = (
        "The service account lacks permission "
        "to write to the BigQuery dataset."
    )

    # ------------------------------------------------------
    # Create embeddings
    # ------------------------------------------------------

    embeddings = model.encode(
        [
            reference_1,
            prediction_1,
            prediction_2,
        ],
        convert_to_tensor=True,
    )

    similarity_1 = cos_sim(
        embeddings[0],
        embeddings[1],
    ).item()

    similarity_2 = cos_sim(
        embeddings[0],
        embeddings[2],
    ).item()

    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print()
    print("Example 1")
    print("-" * 60)

    print(
        f"Reference : {reference_1}"
    )

    print(
        f"Prediction: {prediction_1}"
    )

    print(
        f"Semantic Similarity : "
        f"{similarity_1:.4f}"
    )

    print()
    print("Example 2")
    print("-" * 60)

    print(
        f"Reference : {reference_2}"
    )

    print(
        f"Prediction: {prediction_2}"
    )

    print(
        f"Semantic Similarity : "
        f"{similarity_2:.4f}"
    )

    print()

    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    print(
        "Example 1 should have a higher similarity "
        "because both sentences express the same idea."
    )

    print(
        "Example 2 should have a lower similarity "
        "because it describes an authentication problem."
    )


if __name__ == "__main__":
    main()