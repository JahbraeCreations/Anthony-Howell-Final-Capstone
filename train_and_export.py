"""
Train classical sklearn pipelines, deep GRU rating model, and export artifacts.
Run from repo root: python scripts/train_and_export.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app import config  # noqa: E402
from app.sklearn_extra import sparse_to_dense  # noqa: E402
from app.text_clean import clean_review_text  # noqa: E402


def _parse_inr_price(s: pd.Series) -> pd.Series:
    def one(x):
        if pd.isna(x):
            return np.nan
        t = str(x).replace("₹", "").replace(",", "").strip()
        try:
            return float(t)
        except ValueError:
            return np.nan

    return s.map(one)


def _parse_discount_pct(s: pd.Series) -> pd.Series:
    def one(x):
        if pd.isna(x):
            return np.nan
        t = str(x).replace("%", "").strip()
        try:
            return float(t)
        except ValueError:
            return np.nan

    return s.map(one)


def _parse_rating_count(s: pd.Series) -> pd.Series:
    def one(x):
        if pd.isna(x):
            return np.nan
        t = str(x).replace(",", "").strip()
        try:
            return float(t)
        except ValueError:
            return np.nan

    return s.map(one)


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_CSV)
    df["review_content"] = df["review_content"].astype(str)
    df = df[df["review_content"].str.strip().ne("") & df["review_content"].ne("nan")]

    df["discounted_price_num"] = _parse_inr_price(df["discounted_price"])
    df["actual_price_num"] = _parse_inr_price(df["actual_price"])
    df["discount_pct"] = _parse_discount_pct(df["discount_percentage"])
    mask_disc = df["discount_pct"].isna() & df["discounted_price_num"].notna() & df["actual_price_num"].notna()
    df.loc[mask_disc, "discount_pct"] = (
        (df.loc[mask_disc, "actual_price_num"] - df.loc[mask_disc, "discounted_price_num"])
        / df.loc[mask_disc, "actual_price_num"].replace(0, np.nan)
        * 100.0
    ).clip(lower=0, upper=100)
    df["discount_pct"] = df["discount_pct"].fillna(df["discount_pct"].median())

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].clip(1, 5)

    df["cleaned_text"] = df["review_content"].map(lambda t: clean_review_text(t, remove_stopwords=False))
    df = df[df["cleaned_text"].str.len() > 0]

    df["review_word_count"] = df["cleaned_text"].str.split().str.len().clip(upper=5000)
    df["rating_count_num"] = _parse_rating_count(df["rating_count"])
    df["rating_count_num"] = df["rating_count_num"].fillna(df["rating_count_num"].median())

    both = df["actual_price_num"].notna() & df["discounted_price_num"].notna() & (df["actual_price_num"] > 0)
    df["price_drop_pct"] = np.where(
        both,
        (df["actual_price_num"] - df["discounted_price_num"]) / df["actual_price_num"] * 100.0,
        np.nan,
    )
    df["price_drop_pct"] = df["price_drop_pct"].fillna(df["price_drop_pct"].median())

    try:
        import ssl

        import certifi
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer

        ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
        nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()
        df["vader_compound"] = df["cleaned_text"].map(lambda t: sia.polarity_scores(t)["compound"])
    except Exception:
        df["vader_compound"] = 0.0

    df["category_primary"] = df["category"].astype(str).str.split("|").str[0].fillna("unknown")
    df["sentiment_label"] = (df["rating"] >= 3).astype(int)

    return df.reset_index(drop=True)


def build_column_transformer(*, tfidf_max_features: int = 6000) -> ColumnTransformer:
    numeric = ["review_word_count", "vader_compound", "discount_pct", "price_drop_pct", "rating_count_num"]
    return ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(1, 2)), "cleaned_text"),
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, max_categories=20),
                ["category_primary"],
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def train_classical(df: pd.DataFrame) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X = df[["cleaned_text", "review_word_count", "vader_compound", "discount_pct", "price_drop_pct", "rating_count_num", "category_primary"]]
    y_cls = df["sentiment_label"]
    y_reg = df["rating"]

    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
    )

    pre = build_column_transformer()

    m1 = Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=200, class_weight="balanced", random_state=42))])
    m1.fit(X_train, yc_train)
    pred = m1.predict(X_test)
    print("Logistic Regression", accuracy_score(yc_test, pred), f1_score(yc_test, pred))
    joblib.dump(m1, config.MODEL_1)

    pre2 = build_column_transformer()
    m2 = Pipeline([("prep", pre2), ("clf", DecisionTreeClassifier(max_depth=12, random_state=42))])
    m2.fit(X_train, yc_train)
    pred = m2.predict(X_test)
    print("Decision Tree", accuracy_score(yc_test, pred), f1_score(yc_test, pred))
    joblib.dump(m2, config.MODEL_2)

    # RandomForest does not accept sparse X; densify a smaller TF-IDF block for this head only.
    pre3 = build_column_transformer(tfidf_max_features=1200)
    m3 = Pipeline(
        [
            ("prep", pre3),
            ("dense", FunctionTransformer(sparse_to_dense, validate=False)),
            ("clf", RandomForestClassifier(n_estimators=80, max_depth=20, random_state=42, n_jobs=-1)),
        ]
    )
    m3.fit(X_train, yc_train)
    pred = m3.predict(X_test)
    print("Random Forest", accuracy_score(yc_test, pred), f1_score(yc_test, pred))
    joblib.dump(m3, config.MODEL_3)

    pre4 = build_column_transformer()
    m4 = Pipeline([("prep", pre4), ("reg", Ridge(alpha=2.0, random_state=42))])
    m4.fit(X_train, yr_train)
    pred_r = m4.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(yr_test, pred_r)))
    print("Ridge RMSE", rmse, "R2", r2_score(yr_test, pred_r))
    joblib.dump(m4, config.MODEL_4)

    print("Confusion matrix (LogReg):\n", confusion_matrix(yc_test, m1.predict(X_test)))
    print(classification_report(yc_test, m1.predict(X_test), digits=3))


def _build_vocab(texts: list[str], max_words: int = 12000) -> dict[str, int]:
    from collections import Counter

    counts = Counter()
    for t in texts:
        counts.update(t.split())
    vocab: dict[str, int] = {"<pad>": 0, "<unk>": 1}
    for w, _ in counts.most_common(max(1, max_words - len(vocab))):
        if w not in vocab:
            vocab[w] = len(vocab)
        if len(vocab) >= max_words:
            break
    return vocab


def _encode(texts: list[str], vocab: dict[str, int], max_len: int) -> np.ndarray:
    rows = []
    unk = vocab.get("<unk>", 1)
    for t in texts:
        ids = [vocab.get(w, unk) for w in t.split()][:max_len]
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        rows.append(ids)
    return np.asarray(rows, dtype=np.int64)


def _train_deep_torch(df: pd.DataFrame) -> None:
    """PyTorch GRU fallback for environments without TensorFlow."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    texts = df["cleaned_text"].tolist()
    y = df["rating"].values.astype(np.float32)
    max_len = 120
    vocab = _build_vocab(texts, max_words=12000)
    X = _encode(texts, vocab, max_len)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.long)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    vocab_size = len(vocab)
    emb_dim, hid_dim = 64, 64

    class RatingGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.drop = nn.Dropout(0.2)
            self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
            self.fc = nn.Linear(hid_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            e = self.drop(self.emb(x))
            _, h = self.gru(e)
            return self.fc(h[-1]).squeeze(-1)

    model = RatingGRU()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

    for epoch in range(1, 13):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(xb)
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = float(loss_fn(val_pred, y_val_t))
        print(f"epoch {epoch:02d} train_mse~{total/len(X_train_t):.4f} val_mse={val_loss:.4f}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "max_len": max_len,
            "emb_dim": emb_dim,
            "hid_dim": hid_dim,
            "vocab_size": vocab_size,
        },
        config.DEEP_RATING_TORCH,
    )
    joblib.dump({"vocab": vocab, "max_len": max_len}, config.DEEP_TOKENIZER_TORCH)
    print("Saved deep model to", config.DEEP_RATING_TORCH)


def _train_deep_keras(df: pd.DataFrame) -> None:
    """TensorFlow/Keras GRU trainer for Python versions with TF wheels."""
    import tensorflow as tf

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    texts = df["cleaned_text"].tolist()
    y = df["rating"].values.astype(np.float32)
    max_len = 120
    max_words = 12000

    tok = tf.keras.preprocessing.text.Tokenizer(num_words=max_words, oov_token="<unk>")
    tok.fit_on_texts(texts)
    seqs = tok.texts_to_sequences(texts)
    X = tf.keras.utils.pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    vocab_size = min(max_words, len(tok.word_index) + 1)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, mask_zero=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GRU(64),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=12,
        batch_size=32,
        verbose=2,
    )

    model.save(config.DEEP_RATING_KERAS)
    config.DEEP_TOKENIZER_JSON.write_text(tok.to_json(), encoding="utf-8")
    config.DEEP_META_JSON.write_text(
        json.dumps({"max_len": max_len, "max_words": max_words, "backend": "keras"}, indent=2),
        encoding="utf-8",
    )
    print("Saved deep model to", config.DEEP_RATING_KERAS)


def train_deep(df: pd.DataFrame) -> None:
    """Train GRU rating regressor, preferring TensorFlow/Keras with PyTorch fallback."""
    try:
        import tensorflow  # noqa: F401

        tf_available = True
    except Exception as exc:
        tf_available = False
        print("TensorFlow unavailable; using PyTorch fallback:", exc)

    if tf_available:
        try:
            _train_deep_keras(df)
            return
        except Exception as exc:
            print("TensorFlow path failed; using PyTorch fallback:", exc)

    _train_deep_torch(df)


def save_processed(df: pd.DataFrame) -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.REVIEWS_CLEAN, index=False)
    print("Wrote", config.REVIEWS_CLEAN)


def save_innovation_artifact() -> None:
    """Lightweight innovation metadata; summarization logic lives in the app."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "summarization_config",
        "extractive": {"method": "tfidf_sentence_scoring", "max_sentences": 3},
        "abstractive": {"model_name": "sshleifer/distilbart-cnn-12-6", "max_length": 120, "min_length": 20},
    }
    joblib.dump(payload, config.INNOVATION_MODEL)
    with open(config.MODELS_DIR / "innovation_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("Wrote", config.INNOVATION_MODEL)


def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_prepare()
    print("Prepared shape:", df.shape)
    save_processed(df)
    train_classical(df)
    train_deep(df)
    save_innovation_artifact()


if __name__ == "__main__":
    main()
