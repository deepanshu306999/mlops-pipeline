# End-to-End MLOps Pipeline — Sentiment Classification

Fine-tune **DistilBERT** on the **SST-2** sentiment dataset, containerise inference with
**Docker**, run experiments on **Kaggle**, automate with **GitHub Actions**, and track
everything on **Weights & Biases**.

> Task: binary sentiment classification (`negative` / `positive`).
> Model: [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) (~67M params, ~268 MB) fine-tuned.

---

## Project structure

```
mlops-pipeline/
├── README.md
├── LICENSE                     # MIT
├── .gitignore
├── requirements.txt
├── id2label.json              # label mapping (committed; data files are NOT)
├── Dockerfile
├── src/
│   ├── data_prep.py           # Task 2 — clean + prepare SST-2
│   ├── model_utils.py         # Task 3 — load tokenizer + model
│   └── inference.py           # Tasks 6/7 — run a prediction
├── notebooks/
│   ├── kaggle_training_v1.ipynb  # Task 4 — experiment v1 (lr 3e-5, 2 epochs)
│   └── kaggle_training_v2.ipynb  # Task 4/5 — experiment v2 (lr 5e-5, 3 epochs) + push to HF
└── .github/workflows/
    ├── ci.yml                 # Task 7.1 — lint on push to develop
    └── inference.yml          # Task 7.2 — manual inference run
```

## Setup — install and run each script

```bash
# 1. Clone and enter the repo
git clone https://github.com/deepanshu306999/mlops-pipeline.git
cd mlops-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare the data (downloads SST-2, cleans it, writes id2label.json)
python src/data_prep.py

# 4. (Training happens on Kaggle — see notebooks/kaggle_training_v1.ipynb & _v2.ipynb)

# 5. Run inference locally against the trained HF model
HF_MODEL_NAME=deepanshu30699/distilbert-sst2-mlops \
INPUT_TEXT="This movie was absolutely fantastic!" \
python src/inference.py
```

## Live links (confirm each opens publicly before submission)

| Resource | Link |
|---|---|
| GitHub repository (public) | https://github.com/deepanshu306999/mlops-pipeline |
| Kaggle notebook — Version 1 | https://www.kaggle.com/code/deepanshu30699/mlops-sst2-v1 |
| Kaggle notebook — Version 2 | https://www.kaggle.com/code/deepanshu30699/mlops-sst2-v2 |
| Hugging Face model (public) | https://huggingface.co/deepanshu30699/distilbert-sst2-mlops |
| Docker image (public) | https://hub.docker.com/r/deepanshu30699/mlops-a3-inference |
| W&B project dashboard (public) | https://wandb.ai/deepanshu30699/mlops-assignment3 |

> Kaggle URLs above use the slug `mlops-sst2-v1/v2` — Kaggle auto-generates the slug
> from your notebook title, so after saving each notebook **Public**, copy the real URL
> from the browser and paste it here and in the report.

## Notes

- All training runs on **Kaggle free GPU (T4)**. GitHub Actions is used **only** for CI
  (linting) and inference — never for training.
- Tokens (`HF_TOKEN`, `WANDB_API_KEY`) are loaded from **Kaggle Secrets** and **GitHub
  Secrets** — they are never committed to the repo.
