from collections import Counter
from pathlib import Path
import hashlib
import html
import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")


def normalize_text(text):
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", normalize_text(text))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_data(source=None):
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    if source is None:
        candidates = [ROOT / "imdb.zip", ROOT.parent / "lab04_nlp" / "imdb.zip"]
        source = next((path for path in candidates if path.exists()), None)
    if source is None or not Path(source).exists():
        raise FileNotFoundError("Đặt imdb.zip tại thư mục lab hoặc truyền --data-source PATH.")
    source = Path(source).resolve()
    raw = pd.read_csv(source)
    if not {"review", "sentiment"}.issubset(raw.columns):
        raise ValueError("IMDb cần cột review và sentiment.")
    raw.insert(0, "source_id", range(len(raw)))
    text = raw.review.fillna("").map(normalize_text)
    labels = raw.sentiment.astype("string").str.strip().str.lower()
    valid = text.ne("") & labels.notna() & labels.ne("").fillna(False)
    if not labels[valid].isin(["negative", "positive"]).all():
        raise ValueError("Nhãn sentiment không hợp lệ.")
    cleaned = raw.loc[valid].copy()
    cleaned["text"], cleaned["label"] = text[valid], labels[valid]
    conflicts = cleaned.groupby("text").label.transform("nunique")
    conflict_count = cleaned.loc[conflicts.gt(1), "text"].nunique()
    cleaned = cleaned.loc[conflicts.eq(1)].drop_duplicates("text", keep="first")
    cleaned = cleaned[["source_id", "review", "text", "label"]].reset_index(drop=True)
    train, remaining = train_test_split(cleaned, test_size=0.4, stratify=cleaned.label, random_state=42)
    validation, test = train_test_split(remaining, test_size=0.5, stratify=remaining.label, random_state=42)
    frames = {name: frame.reset_index(drop=True) for name, frame in zip(SPLITS, [train, validation, test])}
    for name, frame in frames.items():
        other_path = ROOT.parent / "lab04_nlp" / "data" / f"{name}.csv"
        if other_path.exists():
            old = pd.read_csv(other_path)
            try:
                pd.testing.assert_frame_equal(frame, old, check_dtype=False)
            except AssertionError as error:
                raise ValueError(f"Split {name} khác Lab 04; cần kiểm tra trước khi so sánh baseline.")
        frame.to_csv(data_dir / f"{name}.csv", index=False)
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            assert set(frames[left].source_id).isdisjoint(frames[right].source_id)
            assert set(frames[left].text).isdisjoint(frames[right].text)
    report = {
        "source": str(source), "source_sha256": sha256(source), "raw_samples": len(raw),
        "classes": int(labels.nunique()), "raw_class_counts": labels.value_counts().to_dict(),
        "missing_text": int(raw.review.isna().sum()), "missing_labels": int(labels.isna().sum()),
        "empty_text": int(text.eq("").sum()), "exact_duplicate_reviews": int(raw.review.duplicated().sum()),
        "normalized_duplicate_reviews": int(text.duplicated().sum()), "conflicting_texts": int(conflict_count),
        "clean_samples": len(cleaned), "removed_rows": len(raw) - len(cleaned),
        "split_seed": 42, "split_ratios": [0.6, 0.2, 0.2],
        "tokenizer": "lowercase, HTML removal, [a-z]+(?:'[a-z]+)?; giữ stopwords và phủ định",
        "split_sha256": {name: sha256(data_dir / f"{name}.csv") for name in SPLITS},
    }
    save_json(data_dir / "inspection.json", report)
    return frames, report


def build_vocabulary(texts, max_size=20000, min_frequency=2):
    counts = Counter(token for text in texts for token in tokenize(text))
    words = sorted(counts, key=lambda word: (-counts[word], word))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word in words:
        if len(vocab) >= max_size:
            break
        if counts[word] >= min_frequency:
            vocab[word] = len(vocab)
    return vocab


def encode(text, vocabulary, max_length):
    tokens = tokenize(text)
    if not tokens:
        raise ValueError("Review không chứa token hợp lệ.")
    return [vocabulary.get(token, 1) for token in tokens[:max_length]]


def describe_sequences(frames, vocabulary):
    rows, lengths_rows, retention_rows = [], [], []
    for name, frame in frames.items():
        documents = frame.text.map(tokenize)
        lengths = documents.map(len).to_numpy()
        counts = frame.label.value_counts()
        rows.append({"split": name, "samples": len(frame), "negative": int(counts["negative"]),
                     "positive": int(counts["positive"]), "min_length": int(lengths.min()),
                     "median_length": float(np.median(lengths)), "mean_length": float(lengths.mean()),
                     "p90_length": float(np.quantile(lengths, .9)), "p95_length": float(np.quantile(lengths, .95)),
                     "max_length": int(lengths.max()),
                     "oov_token_percent": 100 * sum(token not in vocabulary for doc in documents for token in doc) / lengths.sum()})
        lengths_rows.extend({"split": name, "source_id": int(source_id), "length": int(length)}
                            for source_id, length in zip(frame.source_id, lengths))
        for maximum in [30, 50, 100, 200]:
            kept = np.minimum(lengths, maximum)
            retention_rows.append({"split": name, "max_length": maximum,
                                   "tokens_retained_percent": 100 * kept.sum() / lengths.sum(),
                                   "mean_document_retained_percent": 100 * np.mean(kept / lengths),
                                   "truncated_documents_percent": 100 * np.mean(lengths > maximum)})
    return pd.DataFrame(rows), pd.DataFrame(lengths_rows), pd.DataFrame(retention_rows)
