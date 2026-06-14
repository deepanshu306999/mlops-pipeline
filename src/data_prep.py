"""
Task 2 — Data Preparation & Normalisation.

Reusable script that downloads the SST-2 sentiment dataset, inspects it,
cleans the text, encodes labels, and saves an id2label.json mapping.

Run:
    python src/data_prep.py
"""
import json
import os
import re
import argparse

from datasets import load_dataset


# SST-2 is a binary sentiment task: 0 = negative, 1 = positive.
ID2LABEL = {0: "negative", 1: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def clean_text(text):
    """Normalise a single text sample.

    Steps (justified in the report):
      - lowercase           -> reduces vocabulary sparsity
      - collapse whitespace -> removes tokenisation noise
      - strip control chars -> removes corrupt/non-printable characters
    We deliberately KEEP punctuation, because DistilBERT's WordPiece
    tokenizer uses it and sentiment cues (e.g. "!") carry signal.
    """
    if text is None:
        return ""
    text = text.lower()
    text = "".join(ch for ch in text if ch.isprintable())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def inspect(dataset, split_name):
    """Print size, structure, class distribution and quality issues."""
    split = dataset[split_name]
    print(f"\n[{split_name}] size: {len(split)}")
    print(f"[{split_name}] columns: {split.column_names}")

    labels = split["label"]
    counts = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    print(f"[{split_name}] class distribution: "
          + ", ".join(f"{ID2LABEL.get(k, k)}={v}" for k, v in sorted(counts.items())))

    empties = sum(1 for s in split["sentence"] if s is None or not s.strip())
    print(f"[{split_name}] empty/missing sentences: {empties}")
    return counts


def prepare(sample_size, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # SST-2 from the GLUE benchmark. Test labels are hidden (-1), so we build
    # our own train/validation split from the labelled training data.
    print("Downloading SST-2 (glue/sst2)...")
    raw = load_dataset("glue", "sst2")

    print("\n=== RAW DATA INSPECTION ===")
    inspect(raw, "train")
    inspect(raw, "validation")

    # --- Clean ---
    def _clean(example):
        example["sentence"] = clean_text(example["sentence"])
        return example

    cleaned = raw.map(_clean)

    # Remove duplicates and empty rows from the training split.
    seen = set()
    keep_idx = []
    for i, s in enumerate(cleaned["train"]["sentence"]):
        if not s:
            continue          # drop missing/empty
        if s in seen:
            continue          # drop exact duplicates
        seen.add(s)
        keep_idx.append(i)
    train_clean = cleaned["train"].select(keep_idx)
    print(f"\nAfter cleaning: kept {len(train_clean)} of {len(cleaned['train'])} "
          f"train rows ({len(cleaned['train']) - len(train_clean)} dropped).")

    # --- Optional subsample so Kaggle GPU training stays fast ---
    if sample_size and sample_size < len(train_clean):
        train_clean = train_clean.shuffle(seed=42).select(range(sample_size))
        print(f"Subsampled training set to {sample_size} rows for fast GPU training.")

    # Use GLUE validation (labelled) as our evaluation set.
    val_clean = cleaned["validation"]

    # --- Encode labels: SST-2 already provides integer labels 0/1. ---
    # Save the mapping file (this is the only data artifact we commit).
    map_path = os.path.join(os.path.dirname(out_dir), "id2label.json")
    with open(map_path, "w") as f:
        json.dump({str(k): v for k, v in ID2LABEL.items()}, f, indent=2)
    print(f"\nSaved label mapping -> {map_path}")

    # --- Save prepared data locally (NOT committed; see .gitignore) ---
    train_clean.to_json(os.path.join(out_dir, "train.jsonl"))
    val_clean.to_json(os.path.join(out_dir, "validation.jsonl"))
    print(f"Saved prepared data -> {out_dir}/ (ignored by git)")

    print("\nDone. Commit only id2label.json, never the data files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=20000,
                        help="Max training rows to keep (keeps Kaggle GPU fast).")
    parser.add_argument("--out-dir", type=str, default="data",
                        help="Where to write prepared data (git-ignored).")
    args = parser.parse_args()
    prepare(args.sample_size, args.out_dir)
