from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
cells = []


def markdown(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


def discussion_section(title):
    text = (ROOT / "discussion.md").read_text(encoding="utf-8")
    heading = f"## {title}\n"
    if heading not in text:
        raise ValueError(f"Thiếu phần nhận xét: {title}")
    return text.split(heading, 1)[1].split("\n## ", 1)[0].strip()


markdown("# LAB 05 — RNN, LSTM và GRU\nNguyễn Huy Sơn — 23001922\n\nKết quả thực nghiệm ngày 15/10/2026 trên toàn bộ IMDb đã làm sạch.\n\nBài tính viết tay: [calculations.pdf](calculations.pdf). Dự đoán theo cơ chế: [prediction.md](prediction.md).")
code('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
from experiments import run_all
from preprocessing import tokenize, encode
from rnn_cell import check_framework_parity
from predict import SentimentPredictor

ROOT = Path.cwd()
if not (ROOT / "experiments.py").exists():
    raise RuntimeError("Chạy notebook trong thư mục lab5_nlp.")
OUT = ROOT / "artifacts"
if not (OUT / "run_manifest.json").exists() or not (ROOT / "results.csv").exists():
    run_all()
manifest = json.loads((OUT / "run_manifest.json").read_text())
if "completed_at_utc" not in manifest:
    run_all()
    manifest = json.loads((OUT / "run_manifest.json").read_text())
results = pd.read_csv(ROOT / "results.csv")
display(pd.Series({key: manifest[key] for key in ["device", "gpu", "python", "torch", "started_at_utc", "completed_at_utc"]}))
display(pd.Series(manifest["config"]))''')
markdown("## 1. Dataset và độ dài sequence")
code('''inspection = json.loads((ROOT / "data/inspection.json").read_text())
display(pd.Series({key: value for key, value in inspection.items() if key not in ["split_sha256", "tokenizer"]}))
display(pd.read_csv(ROOT / "data/split_summary.csv"))
display(Image(filename=str(ROOT / "sequence_lengths.png")))''')
markdown(discussion_section("Dataset và lựa chọn max_length"))
markdown("## 2. Tokenizer, vocabulary, padding và truncation\n`PAD=0`, `UNK=1`. Vocabulary chỉ học trên train; giữ stopwords và phủ định. Cắt từ bên phải; padding ở bên phải. Model lấy timestep cuối có token thật.")
code('''import torch
from models import collate_reviews

vocabulary = json.loads((OUT / "vocabulary.json").read_text())
texts = ["I really like this movie", "I do not like it"]
sequences = [encode(text, vocabulary, max_length=100) for text in texts]
padded, lengths, _ = collate_reviews([(torch.tensor(ids), torch.tensor(0.)) for ids in sequences])
display(pd.DataFrame({"text": texts, "tokens": [tokenize(text) for text in texts], "ids": sequences}))
print("Padded:", padded.tolist())
print("Lengths:", lengths.tolist(), "Vocabulary:", len(vocabulary))
display(pd.read_csv(ROOT / "data/token_retention.csv"))''')
markdown(discussion_section("Padding và truncation"))
markdown("## 3. Simple RNN cell\nSource: `rnn_cell.py`. Kiểm tra bằng input ngẫu nhiên riêng, so sánh cả forward và gradient với `torch.nn.RNNCell`.")
code('''import inspect
from rnn_cell import SimpleRNNCell

print(inspect.getsource(SimpleRNNCell))
display(pd.Series(check_framework_parity()))''')
markdown(discussion_section("RNN forward và information flow"))
markdown("### LSTM gates và GRU\n\n" + discussion_section("LSTM gates và GRU"))
markdown((ROOT / "prediction.md").read_text(encoding="utf-8").replace("# Prediction", "### Prediction", 1))
markdown("## 4. Baseline và so sánh architecture\nTF-IDF + LR dùng cấu hình đã chọn từ Lab 04, đọc review đầy đủ. Các recurrent model so sánh tại `max_length=100`, cùng cấu hình huấn luyện. `training_seconds` của recurrent gồm train, validation và lưu checkpoint; không gồm chuẩn bị dữ liệu hoặc test.")
code('''columns = ["model", "parameters", "vocabulary_size", "epochs", "best_epoch", "validation_f1",
           "training_seconds", "accuracy", "precision", "recall", "f1", "macro_f1"]
display(pd.read_csv(OUT / "architecture_comparison.csv")[columns].round(4))''')
markdown(discussion_section("Baseline và so sánh architecture"))
markdown("## 5. Thay đổi max_length\n9 cấu hình: 3 architecture × 3 độ dài. Chọn checkpoint bằng F1 validation; threshold 0.5. Mỗi cấu hình chạy đủ 8 epoch. `% tokens retained` là tổng token giữ lại / tổng token gốc trên train.")
code('''display(results[["model", "max_length", "tokens_retained_percent", "validation_loss", "validation_f1",
                 "training_seconds", "peak_gpu_allocated_mb", "accuracy", "f1"]].round(4))
display(Image(filename=str(ROOT / "sequence_length_comparison.png")))''')
markdown(discussion_section("Sequence length và computational cost"))
markdown("## 6. Learning curves")
code('''display(Image(filename=str(ROOT / "learning_curves.png")))
histories = pd.concat([pd.read_csv(OUT / name / "history.csv").assign(model=name)
                       for name in results.loc[results.architecture.isin(["RNN", "LSTM", "GRU"]), "model"]], ignore_index=True)
display(histories[["model", "epoch", "train_loss", "validation_loss", "validation_f1"]].round(4))''')
markdown(discussion_section("Learning curves và overfitting"))
markdown("## 7. Confusion matrix\nTrục dọc: nhãn thật. Trục ngang: nhãn dự đoán. Positive là lớp dương khi tính precision/recall/F1.")
code('''display(Image(filename=str(ROOT / "confusion_matrix.png")))''')
markdown(discussion_section("Confusion matrix và model ứng dụng"))
markdown("## 8. Kiểm tra thứ tự từ\n512 review test cố định. So sánh 100 token đầu với phiên bản xáo trộn đúng các token đó, giữ nguyên model. Các ví dụ `dog bites man` chỉ ghi xác suất sentiment, không dùng để đánh giá hiểu ngữ nghĩa. TF-IDF baseline có bigram nên cũng có thể đổi dự đoán khi đảo từ.")
code('''display(pd.read_csv(OUT / "order_pair_predictions.csv").pivot(index="text", columns="model", values="positive_probability").round(4))
display(pd.read_csv(OUT / "order_shuffle_metrics.csv").round(4))''')
markdown(discussion_section("Thứ tự từ và bằng chứng thực nghiệm"))
markdown("## 9. Vị trí từ khóa và long-range dependency\nGiữ cùng độ dài và mẫu nền lặp; thay một token nền bằng từ khóa tại mỗi vị trí. Đổi vị trí `excellent`/`terrible` ở đầu, giữa, cuối; dùng model đã train ở `max_length=200`. Đây là probe nhỏ trên câu tổng hợp, không chứng minh vanishing gradient.")
code('''probes = pd.read_csv(OUT / "long_range_predictions.csv")
display(probes[["model", "length", "position", "keyword", "expected_sentiment", "predicted", "positive_probability"]].round(4))
display(probes.loc[probes.position.eq("natural_example"), ["model", "text", "predicted", "positive_probability"]])''')
markdown(discussion_section("Long-range dependency"))
markdown("## 10. Phân tích mẫu đúng và sai\n3 đúng và 3 sai từ model ứng dụng. Review gốc, phần token được giữ và nhận xét nằm trong CSV; bản tóm tắt ở [error_analysis.md](error_analysis.md).")
code('''examples = pd.read_csv(OUT / "error_examples.csv").fillna("")
display(examples[["source_id", "label", "predicted", "positive_probability", "input_tokens", "correct", "analysis_student"]])
for row in examples.itertuples():
    print(f"\\nsource_id={row.source_id} | {row.label} -> {row.predicted}")
    print("Retained:", row.retained_text)
    print("Full review:", row.review)''')
markdown("## 11. Ứng dụng sentiment classifier\nModel LSTM/GRU được chọn bằng F1 validation. Xác suất là đầu ra sigmoid, chưa hiệu chỉnh calibration. Có review ngắn, dài, phủ định và tương phản.")
code('''selection = json.loads((OUT / "selection.json").read_text())
display(pd.Series(selection))
display(pd.read_csv(OUT / "application_predictions.csv").fillna(""))
predictor = SentimentPredictor(device="cpu")
display(predictor.predict("I really enjoyed this movie."))
display(pd.read_csv(OUT / "application_token_ablation.csv").sort_values("delta_positive_probability"))''')
markdown(discussion_section("Phân tích ví dụ ứng dụng"))
markdown("## 12. Đối chiếu Lab 04\nChỉ đọc artifact cũ, đối chiếu source_id và tính lại metrics từ predictions. Thời gian của Lab 04 là số đo ở lần chạy cũ, không so trực tiếp với thời gian của lần chạy này.")
code('''from experiments import metrics

previous = ROOT.parent / "lab04_nlp"
if (previous / "artifacts/test_predictions.csv").exists():
    old_predictions = pd.read_csv(previous / "artifacts/test_predictions.csv")
    test = pd.read_csv(ROOT / "data/test.csv")
    assert old_predictions.source_id.tolist() == test.source_id.tolist()
    assert old_predictions.label.tolist() == test.label.tolist()
    old_results = pd.read_csv(previous / "results.csv")
    references = []
    for name in ["tfidf_bi_50k_LR", "avg_word2vec_LR"]:
        values = metrics(test.label.eq("positive"), old_predictions[name].eq("positive").astype(float))
        saved = old_results.loc[old_results.model.eq(name)].iloc[0]
        for key in values:
            assert abs(values[key] - saved[key]) < 1e-10
        references.append({"model": name, "source": "Lab 04, verified saved test predictions",
                           **values, "historical_training_seconds": saved.training_seconds})
    reference = pd.DataFrame(references)
    reference.to_csv(OUT / "lab04_reference_verified.csv", index=False)
    display(reference.round(4))
elif (OUT / "lab04_reference_verified.csv").exists():
    print("Kết quả Lab 04 đã được đối chiếu và lưu từ lần chạy gốc:")
    display(pd.read_csv(OUT / "lab04_reference_verified.csv").round(4))
else:
    print("Không có artifact Lab 04 để đối chiếu tại đường dẫn này.")''')
markdown(discussion_section("So sánh với Lab 04 và giới hạn"))
markdown("## 13. Kiểm chứng")
code('''import subprocess
import sys

completed = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_implementation"],
                           capture_output=True, text=True, check=True)
print(completed.stderr)
assert len(results) == 10
assert results.loc[results.architecture.ne("LR"), "epochs"].eq(8).all()
print("10 model configurations; 9 recurrent models × 8 epochs; complete artifacts.")''')
markdown((ROOT / "reflection.md").read_text(encoding="utf-8").replace("# Reflection", "## 14. Reflection", 1))
markdown("## 15. Ôn tập\n\n" + discussion_section("Ôn tập ngắn"))
notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"}})
nbf.write(notebook, ROOT / "sequence_classification.ipynb")
print(f"Created notebook: {len(cells)} cells")
