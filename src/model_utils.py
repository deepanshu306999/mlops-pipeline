"""
Task 3 — Select & Load a Model from Hugging Face.

Helper functions to load the tokenizer and model with the correct number of
output labels derived from id2label.json. Imported by both the Kaggle training
notebook and src/inference.py so the loading logic lives in one place.
"""
import json
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Compact (~268 MB) pre-trained model — fits comfortably in Kaggle free GPU.
DEFAULT_BASE_MODEL = "distilbert-base-uncased"


def load_label_maps(mapping_path="id2label.json"):
    """Read id2label.json and return (id2label, label2id) with int keys."""
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"{mapping_path} not found. Run `python src/data_prep.py` first."
        )
    with open(mapping_path) as f:
        raw = json.load(f)
    id2label = {int(k): v for k, v in raw.items()}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def load_tokenizer(model_name=DEFAULT_BASE_MODEL):
    """Task 3.1 — load the tokenizer for the chosen model."""
    return AutoTokenizer.from_pretrained(model_name)


def load_model(model_name=DEFAULT_BASE_MODEL, mapping_path="id2label.json"):
    """Task 3.2 — load the model with the correct number of output labels."""
    id2label, label2id = load_label_maps(mapping_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )
    return model


if __name__ == "__main__":
    id2label, label2id = load_label_maps()
    print(f"Labels ({len(id2label)}): {id2label}")
    tok = load_tokenizer()
    print(f"Loaded tokenizer: {tok.__class__.__name__}")
    print("Loading base model (this downloads weights the first time)...")
    mdl = load_model()
    print(f"Loaded model: {mdl.__class__.__name__} "
          f"with {mdl.config.num_labels} output labels.")
