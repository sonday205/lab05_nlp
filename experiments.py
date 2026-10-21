import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
import platform
import random
from time import perf_counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, precision_recall_fscore_support, log_loss)
from threadpoolctl import threadpool_limits
from torch import nn
from torch.utils.data import DataLoader

from models import SequenceClassifier, ReviewDataset, collate_reviews
from preprocessing import (ROOT, SPLITS, prepare_data, build_vocabulary, describe_sequences,
                           save_json, sha256, tokenize)
from rnn_cell import check_framework_parity

OUT = ROOT / "artifacts"
DEFAULT_CONFIG = {"seed": 42, "epochs": 8, "batch_size": 128, "embedding_dim": 100,
                  "hidden_size": 64, "dropout": .2, "learning_rate": .001,
                  "weight_decay": 0., "gradient_clip": 1., "vocabulary_limit": 20000,
                  "min_frequency": 2, "threshold": .5, "num_layers": 1, "bidirectional": False}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def metrics(labels, probabilities):
    predictions = np.asarray(probabilities) >= .5
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    return {"accuracy": float(accuracy_score(labels, predictions)), "precision": float(precision),
            "recall": float(recall), "f1": float(f1),
            "macro_f1": float(precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)[2])}


def make_loader(frame, vocabulary, config, training=False):
    dataset = ReviewDataset(frame, vocabulary, config["max_length"])
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=training,
                      collate_fn=collate_reviews, num_workers=0, pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(config["seed"]))


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probabilities, total_loss = [], [], 0.
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    for tokens, lengths, labels in loader:
        tokens, lengths, labels = tokens.to(device), lengths.to(device), labels.to(device)
        logits = model(tokens, lengths)
        total_loss += criterion(logits, labels).item()
        all_labels.extend(labels.cpu().tolist())
        all_probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    return {"loss": total_loss / len(all_labels), **metrics(all_labels, all_probabilities)}, np.array(all_probabilities)


def train_recurrent(frames, vocabulary, config, device, signature, force=False):
    name = f'{config["architecture"]}_len{config["max_length"]}'
    directory = OUT / name
    directory.mkdir(exist_ok=True)
    metrics_path = directory / "metrics.json"
    if metrics_path.exists() and not force:
        row = json.loads(metrics_path.read_text())
        if row.get("signature") == signature:
            print(f"Reuse completed {name}", flush=True)
            return row
    set_seed(config["seed"])
    train_loader = make_loader(frames["train"], vocabulary, config, training=True)
    validation_loader = make_loader(frames["validation"], vocabulary, config)
    model = SequenceClassifier(len(vocabulary), config["architecture"], config["embedding_dim"],
                               config["hidden_size"], config["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    best_f1, best_epoch, history = -1., 0, []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    synchronize(device)
    start = perf_counter()
    fit_seconds = validation_seconds = 0.
    peak_rss = psutil.Process().memory_info().rss
    print(f"Training {name}: {len(frames['train'])} train, {config['epochs']} epochs", flush=True)
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss, total_grad_norm, batches = 0., 0., 0
        synchronize(device)
        epoch_start = perf_counter()
        for tokens, lengths, labels in train_loader:
            tokens, lengths, labels = tokens.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tokens, lengths), labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {name}, epoch {epoch}")
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient: {name}, epoch {epoch}")
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total_grad_norm += float(grad_norm)
            batches += 1
        synchronize(device)
        epoch_fit = perf_counter() - epoch_start
        validation_start = perf_counter()
        validation, _ = evaluate(model, validation_loader, device)
        synchronize(device)
        epoch_validation = perf_counter() - validation_start
        fit_seconds += epoch_fit
        validation_seconds += epoch_validation
        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        item = {"epoch": epoch, "train_loss": total_loss / len(frames["train"]),
                **{f"validation_{key}": value for key, value in validation.items()},
                "fit_seconds": epoch_fit, "validation_seconds": epoch_validation,
                "mean_gradient_norm_before_clip": total_grad_norm / batches}
        history.append(item)
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        if validation["f1"] > best_f1:
            best_f1, best_epoch = validation["f1"], epoch
            torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "config": config, "vocabulary": vocabulary, "best_epoch": epoch,
                        "signature": signature, "validation_f1": best_f1}, directory / "best.pt")
        print(f"  {name} epoch {epoch}/{config['epochs']} train_loss={item['train_loss']:.4f} "
              f"val_loss={validation['loss']:.4f} val_f1={validation['f1']:.4f} "
              f"time={epoch_fit + epoch_validation:.1f}s", flush=True)
    synchronize(device)
    training_seconds = perf_counter() - start
    peak_gpu = torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else 0.
    checkpoint = torch.load(directory / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    validation, _ = evaluate(model, validation_loader, device)
    test_loader = make_loader(frames["test"], vocabulary, config)
    test, probabilities = evaluate(model, test_loader, device)
    prediction = probabilities >= .5
    pd.DataFrame({"source_id": frames["test"].source_id, "label": frames["test"].label,
                  "positive_probability": probabilities, "predicted": np.where(prediction, "positive", "negative")}
                 ).to_csv(directory / "test_predictions.csv", index=False)
    save_json(directory / "classification_report.json", classification_report(
        frames["test"].label.eq("positive"), prediction, target_names=["negative", "positive"], output_dict=True, zero_division=0))
    row = {"model": name, "architecture": config["architecture"], "representation": "Embedding",
           "max_length": config["max_length"], "parameters": sum(p.numel() for p in model.parameters()),
           "vocabulary_size": len(vocabulary), "epochs": config["epochs"], "best_epoch": best_epoch,
           "training_seconds": training_seconds, "fit_seconds": fit_seconds, "validation_seconds": validation_seconds,
           "validation_loss": validation["loss"], "validation_f1": validation["f1"],
           "peak_gpu_allocated_mb": peak_gpu, "peak_process_rss_sampled_mb": peak_rss / 1024 ** 2,
           "device": str(device), "split": "test", "signature": signature, **test}
    save_json(metrics_path, row)
    del train_loader, validation_loader, test_loader, model, optimizer, checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def train_baseline(frames, signature, force=False):
    directory = OUT / "TFIDF_LR"
    directory.mkdir(exist_ok=True)
    metrics_path = directory / "metrics.json"
    if metrics_path.exists() and not force:
        row = json.loads(metrics_path.read_text())
        if row.get("signature") == signature:
            return row
    print("Training TF-IDF + LR baseline (Lab 04 configuration, full reviews)...", flush=True)
    start = perf_counter()
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, dtype=np.float32)
    features = vectorizer.fit_transform(frames["train"].text)
    classifier = LogisticRegression(C=2., solver="liblinear", max_iter=2000, random_state=42)
    with threadpool_limits(limits=4):
        classifier.fit(features, frames["train"].label.eq("positive"))
    training_seconds = perf_counter() - start
    if classifier.n_iter_.max() >= classifier.max_iter:
        raise RuntimeError("Logistic Regression chưa hội tụ.")
    validation_probabilities = classifier.predict_proba(vectorizer.transform(frames["validation"].text))[:, 1]
    probabilities = classifier.predict_proba(vectorizer.transform(frames["test"].text))[:, 1]
    joblib.dump((vectorizer, classifier), directory / "model.joblib")
    pd.DataFrame({"source_id": frames["test"].source_id, "label": frames["test"].label,
                  "positive_probability": probabilities,
                  "predicted": np.where(probabilities >= .5, "positive", "negative")}
                 ).to_csv(directory / "test_predictions.csv", index=False)
    row = {"model": "TFIDF_LR", "architecture": "LR", "representation": "TF-IDF unigram+bigram",
           "max_length": None, "parameters": int(classifier.coef_.size + classifier.intercept_.size),
           "vocabulary_size": len(vectorizer.vocabulary_), "epochs": None, "best_epoch": None,
           "training_seconds": training_seconds, "fit_seconds": training_seconds,
           "validation_f1": metrics(frames["validation"].label.eq("positive"), validation_probabilities)["f1"],
           "validation_loss": log_loss(frames["validation"].label.eq("positive"), validation_probabilities),
           "device": "cpu", "split": "test", "signature": signature,
           "loss": log_loss(frames["test"].label.eq("positive"), probabilities),
           **metrics(frames["test"].label.eq("positive"), probabilities)}
    save_json(metrics_path, row)
    return row


@torch.inference_mode()
def probabilities_for_texts(model, vocabulary, config, texts, device):
    from preprocessing import encode
    items = [(torch.tensor(encode(text, vocabulary, config["max_length"])), torch.tensor(0.)) for text in texts]
    tokens, lengths, _ = collate_reviews(items)
    return torch.sigmoid(model(tokens.to(device), lengths.to(device))).cpu().numpy()


def run_probes(results, device):
    rows, order_rows = [], []
    models = ["TFIDF_LR", "RNN_len200", "LSTM_len200", "GRU_len200"]
    cases = []
    neutral = ("the camera moves through the room and shows the people sitting near the table ".split())
    for total_length in [50, 100, 200]:
        for keyword in ["excellent", "terrible"]:
            for position in ["beginning", "middle", "end"]:
                words = (neutral * 20)[:total_length]
                index = {"beginning": 5, "middle": total_length // 2, "end": total_length - 5}[position]
                words[index] = keyword
                cases.append({"length": total_length, "position": position, "keyword_index": index,
                              "keyword": keyword, "expected_sentiment": "positive" if keyword == "excellent" else "negative",
                              "text": " ".join(words)})
    cases.extend({"length": len(tokenize(text)), "position": "natural_example", "keyword_index": -1,
                  "keyword": keyword, "expected_sentiment": label, "text": text}
                 for text, keyword, label in [
                     ("The movie was, despite many unnecessary scenes, ultimately excellent.", "excellent", "positive"),
                     ("The movie was, despite many unnecessary scenes, ultimately terrible.", "terrible", "negative")])
    order_texts = ["dog bites man", "man bites dog", "this movie is good", "this movie is not good",
                   "the movie was good but the ending was bad", "the movie was bad but the ending was good"]
    rng = random.Random(42)
    test_sample = pd.read_csv(OUT / "order_test_sample.csv")
    original = test_sample.text.tolist()
    shuffled = []
    for text in original:
        words = tokenize(text)[:100]
        rng.shuffle(words)
        shuffled.append(" ".join(words))
    original = [" ".join(tokenize(text)[:100]) for text in original]
    for name in models:
        if name == "TFIDF_LR":
            vectorizer, classifier = joblib.load(OUT / name / "model.joblib")
            predict_many = lambda texts: classifier.predict_proba(vectorizer.transform(texts))[:, 1]
            model, config, vocabulary = None, None, None
        else:
            checkpoint = torch.load(OUT / name / "best.pt", map_location=device, weights_only=True)
            config, vocabulary = checkpoint["config"], checkpoint["vocabulary"]
            model = SequenceClassifier(len(vocabulary), config["architecture"], config["embedding_dim"],
                                       config["hidden_size"], config["dropout"]).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            predict_many = lambda texts: probabilities_for_texts(model, vocabulary, config, texts, device)
        probabilities = predict_many([case["text"] for case in cases])
        for case, probability in zip(cases, probabilities):
            rows.append({"model": name, **case, "positive_probability": float(probability),
                         "predicted": "positive" if probability >= .5 else "negative"})
        order_probabilities = predict_many(order_texts)
        for text, probability in zip(order_texts, order_probabilities):
            order_rows.append({"model": name, "text": text, "positive_probability": float(probability)})
        original_probabilities, shuffled_probabilities = [], []
        for index in range(0, len(original), 64):
            original_probabilities.extend(predict_many(original[index:index + 64]))
            shuffled_probabilities.extend(predict_many(shuffled[index:index + 64]))
        pd.DataFrame({"source_id": test_sample.source_id, "label": test_sample.label,
                      "original_probability": original_probabilities, "shuffled_probability": shuffled_probabilities,
                      "original_first_100_tokens": original, "shuffled_same_tokens": shuffled}).to_csv(
                          OUT / f"order_shuffle_{name}.csv", index=False)
        if model is not None:
            del model, checkpoint
    pd.DataFrame(rows).to_csv(OUT / "long_range_predictions.csv", index=False)
    pd.DataFrame(order_rows).to_csv(OUT / "order_pair_predictions.csv", index=False)
    shuffle_rows = []
    for name in models:
        frame = pd.read_csv(OUT / f"order_shuffle_{name}.csv")
        for order in ["original", "shuffled"]:
            shuffle_rows.append({"model": name, "order": order, "samples": len(frame),
                                 **metrics(frame.label.eq("positive"), frame[f"{order}_probability"])})
    pd.DataFrame(shuffle_rows).to_csv(OUT / "order_shuffle_metrics.csv", index=False)


def save_figures(results, frames, lengths):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in SPLITS:
        values = lengths.loc[lengths.split.eq(name), "length"]
        axes[0].hist(values, bins=80, histtype="step", label=name, density=True)
    axes[0].set(xlabel="Tokens / review", ylabel="Density", title="Sequence lengths (all reviews)")
    axes[0].legend()
    distribution = pd.DataFrame({name: frame.label.value_counts() for name, frame in frames.items()})
    distribution.T.plot.bar(ax=axes[1], rot=0, color=["#2563eb", "#f59e0b"])
    axes[1].set(ylabel="Reviews", title="Class distribution")
    figure.tight_layout()
    figure.savefig(ROOT / "sequence_lengths.png", dpi=160)
    plt.close(figure)
    neural = results.loc[results.architecture.isin(["RNN", "LSTM", "GRU"])]
    figure, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    for axis, row in zip(axes.flat, neural.itertuples()):
        history = pd.read_csv(OUT / row.model / "history.csv")
        axis.plot(history.epoch, history.train_loss, label="Train")
        axis.plot(history.epoch, history.validation_loss, label="Validation")
        axis.axvline(row.best_epoch, ls="--", color="gray", alpha=.7)
        axis.set(title=row.model, xlabel="Epoch", ylabel="BCE loss")
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(ROOT / "learning_curves.png", dpi=160)
    plt.close(figure)
    comparison_names = ["TFIDF_LR", "RNN_len100", "LSTM_len100", "GRU_len100"]
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, name in zip(axes, comparison_names):
        prediction = pd.read_csv(OUT / name / "test_predictions.csv")
        matrix = confusion_matrix(prediction.label, prediction.predicted, labels=["negative", "positive"])
        pd.DataFrame(matrix, index=["negative", "positive"], columns=["negative", "positive"]).to_csv(OUT / name / "confusion_matrix.csv")
        ConfusionMatrixDisplay(matrix, display_labels=["negative", "positive"]).plot(ax=axis, colorbar=False, cmap="Blues")
        axis.set_title(name)
    figure.tight_layout()
    figure.savefig(ROOT / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for architecture in ["RNN", "LSTM", "GRU"]:
        frame = neural.loc[neural.architecture.eq(architecture)].sort_values("max_length")
        axes[0].plot(frame.max_length, frame.f1, "o-", label=architecture)
        axes[1].plot(frame.max_length, frame.training_seconds, "o-", label=architecture)
        axes[2].plot(frame.max_length, frame.peak_gpu_allocated_mb, "o-", label=architecture)
    for axis, ylabel in zip(axes, ["Test F1 (positive)", "Training + validation (s)", "Peak GPU allocated (MiB)"]):
        axis.set(xlabel="Max length", ylabel=ylabel, xticks=[50, 100, 200])
        axis.legend()
    figure.tight_layout()
    figure.savefig(ROOT / "sequence_length_comparison.png", dpi=160)
    plt.close(figure)


def save_examples(frames, selection):
    name = selection["application_model"]
    predictions = pd.read_csv(OUT / name / "test_predictions.csv")
    merged = frames["test"].merge(predictions.drop(columns="label"), on="source_id", validate="one_to_one")
    merged["correct"] = merged.label.eq(merged.predicted)
    selected = []
    for correct in [True, False]:
        candidates = merged.loc[merged.correct.eq(correct)]
        selected.append(candidates.head(3))
    examples = pd.concat(selected).copy()
    examples["input_tokens"] = examples.text.map(lambda text: len(tokenize(text)))
    examples["max_length"] = selection["application_max_length"]
    examples["retained_text"] = examples.text.map(lambda text: " ".join(tokenize(text)[:selection["application_max_length"]]))
    examples["analysis_student"] = ""
    examples.to_csv(OUT / "error_examples.csv", index=False)
    lines = ["# Error analysis", "", f"Model: `{name}` (chọn bằng validation F1).", "",
             "Dữ liệu đầy đủ và phần text được giữ: `artifacts/error_examples.csv`.", "",
             "| source_id | True | Predicted | P(positive) | Tokens | Đúng | Phân tích của sinh viên |",
             "|---|---|---|---:|---:|---|---|"]
    for row in examples.itertuples():
        lines.append(f"| {row.source_id} | {row.label} | {row.predicted} | {row.positive_probability:.4f} | {row.input_tokens} | {row.correct} | |")
    lines.extend(["", "Sinh viên tự điền nguyên nhân lỗi dựa trên review gốc, retained_text và dự đoán.",
                  "Đối chiếu negation, contrast, từ hiếm/OOV, truncation, sarcasm và context; không mặc định lỗi do sequence modeling.", ""])
    (ROOT / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    from predict import SentimentPredictor
    predictor = SentimentPredictor(device="cpu")
    application_texts = {
        "short": "I really enjoyed this movie. The acting was excellent.",
        "long": " ".join(["The film follows the characters as they travel through different places and talk about their lives."] * 18) + " Despite the slow pace, I loved the thoughtful ending and the excellent performances.",
        "negation": "I did not enjoy this movie. It was not good and I would not recommend it.",
        "contrast": "I thought the movie would be terrible, but it was surprisingly good.",
    }
    app = [{"case": kind, "text": text, **predictor.predict(text), "context_analysis_student": ""}
           for kind, text in application_texts.items()]
    pd.DataFrame(app).to_csv(OUT / "application_predictions.csv", index=False)
    drops = []
    original = app[-1]["positive_probability"]
    words = tokenize(application_texts["contrast"])
    for index in range(len(words)):
        changed = " ".join(words[:index] + words[index + 1:])
        probability = predictor.predict(changed)["positive_probability"]
        drops.append({"removed_index": index, "removed_token": words[index], "original_positive_probability": original,
                      "ablated_positive_probability": probability, "delta_positive_probability": probability - original})
    pd.DataFrame(drops).to_csv(OUT / "application_token_ablation.csv", index=False)


def run_all(force=False, epochs=8, data_source=None):
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    config = {**DEFAULT_CONFIG, "epochs": epochs}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames, inspection = prepare_data(data_source)
    vocabulary = build_vocabulary(frames["train"].text, config["vocabulary_limit"], config["min_frequency"])
    save_json(OUT / "vocabulary.json", vocabulary)
    summary, lengths, retention = describe_sequences(frames, vocabulary)
    summary.to_csv(ROOT / "data" / "split_summary.csv", index=False)
    lengths.to_csv(ROOT / "data" / "sequence_lengths.csv", index=False)
    retention.to_csv(ROOT / "data" / "token_retention.csv", index=False)
    source_hashes = {name: sha256(ROOT / name) for name in ["experiments.py", "models.py", "preprocessing.py", "rnn_cell.py", "predict.py"]}
    signature = hashlib.sha256(json.dumps({"config": config, "splits": inspection["split_sha256"],
                                           "source": source_hashes}, sort_keys=True).encode()).hexdigest()
    save_json(OUT / "rnn_cell_check.json", check_framework_parity())
    manifest = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                "architectures": ["RNN", "LSTM", "GRU"], "max_lengths": [50, 100, 200],
                "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                "pandas": pd.__version__, "sklearn": sklearn.__version__, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(), "source_sha256": source_hashes,
                "signature": signature, "split_sha256": inspection["split_sha256"],
                "selection": "highest validation binary F1; earliest epoch on ties; test used only after checkpoint selection",
                "timing": "wall-clock training includes validation and checkpoint IO, excludes data preparation/test; fit_seconds is forward/backward/optimizer only",
                "memory": "GPU torch.cuda.max_memory_allocated; CPU process RSS sampled after each epoch, not a continuous peak",
                "seed_note": "single seed 42; deterministic cuDNN; results may vary across hardware/PyTorch versions"}
    save_json(OUT / "run_manifest.json", manifest)
    rows = [train_baseline(frames, signature, force)]
    for architecture in ["RNN", "LSTM", "GRU"]:
        for maximum in [50, 100, 200]:
            run_config = {**config, "architecture": architecture, "max_length": maximum}
            rows.append(train_recurrent(frames, vocabulary, run_config, device, signature, force))
            pd.DataFrame(rows).to_csv(ROOT / "results.csv", index=False)
    results = pd.DataFrame(rows)
    gated = results.loc[results.architecture.isin(["LSTM", "GRU"])].sort_values("validation_f1", ascending=False, kind="stable")
    selected = gated.iloc[0]
    selection = {"application_model": selected.model,
                 "application_checkpoint": f"artifacts/{selected.model}/best.pt",
                 "application_max_length": int(selected.max_length),
                 "criterion": "validation F1 among LSTM/GRU; test scores not used",
                 "validation_f1": float(selected.validation_f1)}
    save_json(OUT / "selection.json", selection)
    retention_train = retention.loc[retention.split.eq("train"), ["max_length", "tokens_retained_percent"]]
    results = results.merge(retention_train, on="max_length", how="left", validate="many_to_one")
    results.to_csv(ROOT / "results.csv", index=False)
    results.loc[results.model.isin(["TFIDF_LR", "RNN_len100", "LSTM_len100", "GRU_len100"])].to_csv(OUT / "architecture_comparison.csv", index=False)
    results.loc[results.architecture.eq("LSTM")].to_csv(OUT / "sequence_length_comparison.csv", index=False)
    frames["test"].sample(n=512, random_state=42)[["source_id", "text", "label"]].to_csv(OUT / "order_test_sample.csv", index=False)
    run_probes(results, device)
    save_figures(results, frames, lengths)
    save_examples(frames, selection)
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["application_selection"] = selection
    save_json(OUT / "run_manifest.json", manifest)
    print(results[["model", "best_epoch", "validation_f1", "accuracy", "f1", "training_seconds"]].to_string(index=False), flush=True)
    print(f"Application model: {selected.model}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Train lại toàn bộ model, ghi đè kết quả.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--data-source", type=str)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs phải >= 1")
    run_all(force=args.force, epochs=args.epochs, data_source=args.data_source)
