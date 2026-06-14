"""
Tasks 6 & 7 — Inference entry point.

Loads the fine-tuned model from the Hugging Face Hub and classifies an input
string. Designed to run inside the Docker image and the GitHub Actions
inference workflow.

Configuration comes from environment variables (never hardcoded):
    HF_MODEL_NAME : HF repo of the fine-tuned model (has a default)
    INPUT_TEXT    : text to classify
    HF_TOKEN      : optional; only needed if the model repo is private
"""
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Sensible default so the image runs even with no args; override at run time.
DEFAULT_MODEL = os.environ.get(
    "HF_MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english"
)


def predict(text, model_name=DEFAULT_MODEL, token=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=token)
    model.eval()

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs))
    label = model.config.id2label.get(pred_id, str(pred_id))
    return label, float(probs[pred_id])


def main():
    model_name = os.environ.get("HF_MODEL_NAME", DEFAULT_MODEL)
    text = os.environ.get("INPUT_TEXT")
    if not text:
        text = sys.argv[1] if len(sys.argv) > 1 else "I really enjoyed this!"
    token = os.environ.get("HF_TOKEN")  # optional

    label, confidence = predict(text, model_name=model_name, token=token)

    print("=" * 50)
    print(f"Model : {model_name}")
    print(f"Input : {text}")
    print(f"Label : {label}")
    print(f"Score : {confidence:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
