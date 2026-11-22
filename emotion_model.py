"""Training and prediction code for the emotion classifier."""
import re
import string
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).parent / "train.txt"


def clean_text(text: str) -> str:
    """Lowercase, then remove punctuation, digits and non-ASCII characters (emojis)."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = "".join(ch for ch in text if ch.isascii() and not ch.isdigit())
    return re.sub(r"\s+", " ", text).strip()


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";", header=None, names=["text", "emotion"])
    df["text"] = df["text"].map(clean_text)
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("vec", CountVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )


def train_model() -> Pipeline:
    df = load_data()
    model = build_pipeline()
    model.fit(df["text"], df["emotion"])
    return model


def predict_emotion(model: Pipeline, text: str):
    """Return {emotion: probability}, or None if no known words are in the text."""
    cleaned = clean_text(text)
    if model.named_steps["vec"].transform([cleaned]).nnz == 0:
        return None
    probs = model.predict_proba([cleaned])[0]
    return dict(zip(model.classes_, probs))
