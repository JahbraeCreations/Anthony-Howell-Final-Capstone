# Same-Day E-Commerce Capstone — Product Reviews Track



## Problem

We analyze Amazon India product listings with aggregated review text to study sentiment, rating prediction, and review summarization. The dataset has **no per-review timestamps**, so we do not report “sentiment over time”; exploratory work uses rating, discount, and category instead.

## Dataset

- **Source:** Kaggle-style Amazon product / reviews export (CSV).
- **Raw file:** `data/raw/amazon.csv` (copy your full `amazon.csv` here).
- **Processed:** `data/processed/reviews_clean.parquet` (created by `scripts/train_and_export.py` or the preprocessing notebook).

## Repository layout

```text
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── model_1.pkl          # Sentiment: Logistic Regression + TF-IDF + numerics + category OHE
│   ├── model_2.pkl          # Sentiment: Decision Tree
│   ├── model_3.pkl          # Sentiment: Random Forest (dense head; smaller TF-IDF branch)
│   ├── model_4.pkl          # Rating regression: Ridge on the same feature union
│   ├── deep_rating.keras    # GRU rating regressor when TensorFlow/Keras is available
│   ├── deep_tokenizer.json  # Keras tokenizer sidecar (Tokenizer.to_json())
│   ├── deep_meta.json       # Shared deep metadata (e.g., max_len)
│   ├── deep_rating.pt       # GRU fallback artifact when TensorFlow is unavailable
│   ├── deep_tokenizer.joblib
│   ├── innovation_model.pkl # Summarization config (extractive + optional HF model name)
│   └── innovation_config.json
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_features_classical_models.ipynb
│   └── 03_deep_rating_model.ipynb
├── app/
│   ├── config.py
│   ├── text_clean.py
│   ├── summarize.py
│   ├── sklearn_extra.py
│   └── gradio_app.py
└── scripts/
    └── train_and_export.py
```

## Setup

```bash
cd "/path/to/FINAL CAPSTONE"
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional (for abstractive summarization in the UI):

```bash
pip install transformers torch --extra-index-url https://download.pytorch.org/whl/cpu
```

Optional (to enable TensorFlow/Keras deep training on Python 3.11-3.12):

```bash
pip install tensorflow
```

## Train models and write processed data

```bash
python scripts/train_and_export.py
```

This downloads NLTK VADER data as needed, writes `reviews_clean.parquet`, trains exactly four sklearn pipelines (`model_1.pkl` to `model_4.pkl`), trains the GRU regressor, and writes `innovation_model.pkl`.

Deep artifacts by environment:

- TensorFlow available (typically Python 3.11-3.12): `deep_rating.keras` + `deep_tokenizer.json` (+ `deep_meta.json`)
- TensorFlow unavailable (e.g., Python 3.14): `deep_rating.pt` + `deep_tokenizer.joblib`

## Notebooks

```bash
python -m jupyter notebook
```

Open `notebooks/01_*` through `03_*`. The notebooks mirror the script stages; notebook 03 now loads deep artifacts with Keras-first / PyTorch-fallback behavior via `app/config.py`.

## Run the Gradio app

```bash
python app/gradio_app.py
```

Paste two or three short reviews in the text box, separated by `|`, pick summarization mode, and inspect summaries plus sklearn demo predictions.

## Known limitations

- Rows are **product-level** bundles of review text, not individual reviews.
- No review timestamps in the CSV.
- Abstractive summarization needs `transformers` + `torch`; otherwise the app falls back to extractive text.
- Sklearn demo predictions in the app use **neutral filler values** for numeric and category fields (only your pasted text differs); for rigorous metrics, use the training notebook outputs.
- Sentiment labels are **highly imbalanced** toward positives (`rating >= 3`), so the negative class is rare in the holdout split.
- **Deep model format:** the trainer prefers Keras (`deep_rating.keras` + `deep_tokenizer.json`) when TensorFlow is installed, and automatically falls back to PyTorch (`deep_rating.pt` + `deep_tokenizer.joblib`) when it is not.

### macOS: NLTK SSL errors

If `nltk.download` fails with `CERTIFICATE_VERIFY_FAILED`, install certificates for the Python.org build (or use a conda env), then rerun downloads. Until VADER downloads successfully, training sets `vader_compound` to `0.0` (the pipeline still runs).
