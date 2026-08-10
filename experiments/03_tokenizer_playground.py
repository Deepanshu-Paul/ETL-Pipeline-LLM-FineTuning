"""
Experiment 03

Tokenizer Playground

Purpose:
- Understand how text is converted into tokens.
- Inspect token IDs.
- Understand encoding and decoding.
- Experiment with different types of text.
"""
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# ----------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------

def inspect_domain_terms(tokenizer):
    """
    Inspect how the tokenizer handles ETL and data-engineering terms.
    """

    terms = [
        "ETL",
        "DBT",
        "BigQuery",
        "Pub/Sub",
        "schema",
        "CDC",
        "Airflow",
        "Dataflow",
    ]

    print("=" * 60)
    print("Domain Term Tokenization")
    print("=" * 60)

    for term in terms:

        tokens = tokenizer.tokenize(term)
        token_ids = tokenizer.encode(
            term,
            add_special_tokens=False,
        )

        print(f"\nTerm      : {term}")
        print(f"Tokens    : {tokens}")
        print(f"Token IDs : {token_ids}")
        print(f"Count     : {len(token_ids)}")

    print()


def inspect_sentence_efficiency(tokenizer):
    """
    Measure how many tokens are required to represent
    realistic ETL-related sentences.
    """

    sentences = [
        "The ETL pipeline failed because the source schema changed.",
        "The Airflow Dataflow job failed while loading data into BigQuery.",
        "The DBT transformation failed because of a schema mismatch.",
        "The CDC pipeline received duplicate Pub/Sub messages.",
    ]

    print("=" * 60)
    print("Sentence Tokenization Efficiency")
    print("=" * 60)

    for sentence in sentences:

        token_ids = tokenizer.encode(
            sentence,
            add_special_tokens=False,
        )

        word_count = len(sentence.split())
        token_count = len(token_ids)
        ratio = token_count / word_count

        print(f"\nSentence    : {sentence}")
        print(f"Word Count  : {word_count}")
        print(f"Token Count : {token_count}")
        print(f"Token/Word  : {ratio:.2f}")

    print()


def inspect_id_mapping(tokenizer):
    """
    Show the mapping between token IDs and token strings.
    """

    text = "The ETL pipeline failed."

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    print("=" * 60)
    print("Token ID Mapping")
    print("=" * 60)

    print(f"Text : {text}")
    print()

    for token_id, token in zip(token_ids, tokens):
        print(f"{token_id:>6}  →  {token}")

    print()


def inspect_embedding(tokenizer, model):
    """
    Inspect the embedding vector associated with one token.
    """

    text = "pipeline"

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    token_id = token_ids[0]

    embedding = model.get_input_embeddings()
    vector = embedding.weight[token_id]

    print("=" * 60)
    print("Embedding Inspection")
    print("=" * 60)

    print(f"Text       : {text}")
    print(f"Token ID   : {token_id}")
    print(f"Vector Shape: {tuple(vector.shape)}")
    print(f"Vector Dtype: {vector.dtype}")

    print()
    print("First 10 embedding values:")
    print(vector[:10].detach().cpu())
    print()


# ----------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------

def load_tokenizer():
    """Load and report the tokenizer for the configured model."""
    print("=" * 60)
    print("Loading Tokenizer")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Tokenizer Class : {tokenizer.__class__.__name__}")
    print(f"Vocabulary Size : {tokenizer.vocab_size}")
    print()
    return tokenizer


def load_model():
    """Load the causal language model for embedding inspection."""
    print("=" * 60)
    print("Loading Model")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
    )

    print("✓ Model loaded.")
    print()
    return model


def inspect_basic_tokenization(tokenizer, text):
    """Print the tokenization and decoding results for a sample prompt."""
    print("=" * 60)
    print("Tokenization")
    print("=" * 60)

    print(f"Input Text : {text}")

    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=True,
    )

    print(f"Tokens     : {tokens}")
    print(f"Token IDs  : {token_ids}")
    print(f"Token Count: {len(token_ids)}")
    print()

    decoded_text = tokenizer.decode(token_ids)

    print("=" * 60)
    print("Decoding")
    print("=" * 60)

    print(f"Decoded Text: {decoded_text}")
    print()


def inspect_special_tokens(tokenizer):
    """Print BOS/EOS/PAD token information."""
    print("=" * 60)
    print("Special Tokens")
    print("=" * 60)

    print(f"BOS Token    : {tokenizer.bos_token}")
    print(f"BOS Token ID : {tokenizer.bos_token_id}")

    print(f"EOS Token    : {tokenizer.eos_token}")
    print(f"EOS Token ID : {tokenizer.eos_token_id}")

    print(f"PAD Token    : {tokenizer.pad_token}")
    print(f"PAD Token ID : {tokenizer.pad_token_id}")
    print()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    tokenizer = load_tokenizer()
    model = load_model()

    text = "The ETL pipeline failed because the source schema changed."
    inspect_basic_tokenization(tokenizer, text)
    inspect_special_tokens(tokenizer)
    inspect_domain_terms(tokenizer)
    inspect_sentence_efficiency(tokenizer)
    inspect_id_mapping(tokenizer)
    inspect_embedding(tokenizer, model)


if __name__ == "__main__":
    main()