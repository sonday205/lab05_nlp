import json
from pathlib import Path
import subprocess
import sys

import joblib
import nbformat
import numpy as np
import pandas as pd
import torch

from experiments import metrics, set_seed
from models import SequenceClassifier, ReviewDataset, collate_reviews
from predict import SentimentPredictor
from preprocessing import ROOT, SPLITS, sha256, save_json
from torch.utils.data import DataLoader


def verify():
    torch.set_num_threads(4)
    out = ROOT / "artifacts"
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert "completed_at_utc" in manifest
    for name, expected_hash in manifest["source_sha256"].items():
        assert sha256(ROOT / name) == expected_hash, name
    frames = {name: pd.read_csv(ROOT / "data" / f"{name}.csv") for name in SPLITS}
    for name, expected_hash in manifest["split_sha256"].items():
        assert sha256(ROOT / "data" / f"{name}.csv") == expected_hash, name
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            assert set(frames[left].source_id).isdisjoint(frames[right].source_id)
            assert set(frames[left].text).isdisjoint(frames[right].text)
    results = pd.read_csv(ROOT / "results.csv")
    assert len(results) == 10 and results.model.nunique() == 10
    neural = results.loc[results.architecture.isin(["RNN", "LSTM", "GRU"])]
    assert set(zip(neural.architecture, neural.max_length)) == {
        (architecture, maximum) for architecture in ["RNN", "LSTM", "GRU"] for maximum in [50, 100, 200]}
    verification = []
    device = torch.device(manifest["device"])
    for row in results.itertuples():
        directory = out / row.model
        predictions = pd.read_csv(directory / "test_predictions.csv")
        assert predictions.source_id.tolist() == frames["test"].source_id.tolist()
        assert predictions.label.tolist() == frames["test"].label.tolist()
        probabilities = predictions.positive_probability.to_numpy()
        assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
        assert np.array_equal(predictions.predicted.eq("positive"), probabilities >= .5)
        calculated = metrics(frames["test"].label.eq("positive"), probabilities)
        for key, value in calculated.items():
            assert abs(value - getattr(row, key)) < 1e-10, (row.model, key)
        sample = frames["test"].iloc[:128]
        cpu_error = 0.
        cpu_changed_classes = 0
        if row.architecture == "LR":
            vectorizer, classifier = joblib.load(directory / "model.joblib")
            reloaded_probabilities = classifier.predict_proba(vectorizer.transform(sample.text))[:, 1]
        else:
            history = pd.read_csv(directory / "history.csv")
            assert history.epoch.tolist() == list(range(1, 9))
            assert row.epochs == 8
            best = history.loc[history.validation_f1.idxmax()]
            assert int(best.epoch) == row.best_epoch
            assert abs(best.validation_f1 - row.validation_f1) < 1e-10
            assert abs(best.validation_loss - row.validation_loss) < 1e-6
            checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
            config, vocabulary = checkpoint["config"], checkpoint["vocabulary"]
            assert checkpoint["signature"] == manifest["signature"]
            assert checkpoint["best_epoch"] == row.best_epoch
            set_seed(config["seed"])
            model = SequenceClassifier(len(vocabulary), config["architecture"], config["embedding_dim"],
                                       config["hidden_size"], config["dropout"]).eval()
            model.load_state_dict(checkpoint["state_dict"])
            assert model.embedding.weight[0].eq(0).all()
            assert sum(parameter.numel() for parameter in model.parameters()) == row.parameters
            loader = DataLoader(ReviewDataset(sample, vocabulary, config["max_length"]), batch_size=128, collate_fn=collate_reviews)
            tokens, lengths, _ = next(iter(loader))
            with torch.inference_mode():
                cpu_probabilities = torch.sigmoid(model(tokens, lengths)).numpy()
                model = model.to(device)
                reloaded_probabilities = torch.sigmoid(model(tokens.to(device), lengths.to(device))).cpu().numpy()
            cpu_error = float(np.max(np.abs(cpu_probabilities - probabilities[:len(sample)])))
            cpu_changed_classes = int(np.sum((cpu_probabilities >= .5) != (probabilities[:len(sample)] >= .5)))
        error = float(np.max(np.abs(reloaded_probabilities - probabilities[:len(sample)])))
        assert error < 1e-5, (row.model, error)
        verification.append({"model": row.model, "test_samples": len(predictions), "metrics_match": True,
                             "checkpoint_reload_samples": len(sample), "reload_device": str(device),
                             "reload_max_probability_error": error,
                             "cpu_vs_saved_gpu_max_probability_error": cpu_error,
                             "cpu_vs_saved_gpu_changed_classes": cpu_changed_classes})
    selection = json.loads((out / "selection.json").read_text())
    gated = results.loc[results.architecture.isin(["LSTM", "GRU"])]
    assert selection["application_model"] == gated.sort_values("validation_f1", ascending=False, kind="stable").iloc[0].model
    predictor = SentimentPredictor(device="cpu")
    application = pd.read_csv(out / "application_predictions.csv")
    assert set(application["case"]) == {"short", "long", "negation", "contrast"}
    for row in application.itertuples():
        actual = predictor.predict(row.text)
        assert abs(actual["positive_probability"] - row.positive_probability) < 1e-6
        assert actual["sentiment"] == row.sentiment
    for invalid in ["", "   ", "<br /> 1234 !!!"]:
        try:
            predictor.predict(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Empty review must be rejected")
    examples = pd.read_csv(out / "error_examples.csv")
    assert len(examples) == 6 and examples.correct.sum() == 3
    assert examples.analysis_student.fillna("").str.strip().ne("").all()
    assert application.context_analysis_student.fillna("").str.strip().ne("").all()
    probes = pd.read_csv(out / "long_range_predictions.csv")
    assert len(probes) == 80
    assert {"beginning", "middle", "end", "natural_example"} == set(probes.position)
    assert len(pd.read_csv(out / "order_shuffle_metrics.csv")) == 8
    for image in ["sequence_lengths.png", "learning_curves.png", "confusion_matrix.png", "sequence_length_comparison.png"]:
        assert (ROOT / image).stat().st_size > 10000
    notebook = nbformat.read(ROOT / "sequence_classification.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) == 14
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs)
    completed = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_implementation"],
                               capture_output=True, text=True, check=True)
    report = {"status": "passed", "unit_tests": 6, "models": verification,
              "source_hashes_match": True, "split_hashes_match": True, "no_split_overlap": True,
              "selection_uses_validation": True, "application_reload_parity": True,
              "invalid_input_rejected": True, "six_error_examples": True,
              "long_range_probes": len(probes), "unit_test_output": completed.stderr,
              "analysis_examples_complete": True, "application_comments_complete": True,
              "notebook": {"file": "sequence_classification.ipynb", "executed_code_cells": len(code_cells), "errors": 0}}
    save_json(out / "verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    verify()
