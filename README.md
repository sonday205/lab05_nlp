# LAB 05 — RNN, LSTM và GRU

Nguyễn Huy Sơn — 23001922.

Phân loại sentiment IMDb bằng TF-IDF + Logistic Regression và RNN/LSTM/GRU. Kết quả huấn luyện được lưu từ ngày 15/10/2026.

## File chính

| File | Nội dung |
|---|---|
| [calculations.pdf](calculations.pdf) | Bài tính viết tay, 2 trang |
| [prediction.md](prediction.md) | Dự đoán theo cơ chế để đối chiếu thực nghiệm |
| [sequence_classification.ipynb](sequence_classification.ipynb) | Notebook có output và giải thích ngắn |
| [rnn_cell.py](rnn_cell.py) | RNN cell tự viết, đối chiếu forward/gradient với PyTorch |
| [results.csv](results.csv) | Metrics và chi phí của 10 cấu hình |
| [discussion.md](discussion.md) | Nhận xét dataset, length, learning curves, probes và ứng dụng |
| [error_analysis.md](error_analysis.md) | Phân tích 3 mẫu đúng, 3 mẫu sai |
| [reflection.md](reflection.md) | Tổng kết và hướng cải thiện |
| [CHECKLIST.md](CHECKLIST.md) | Đối chiếu yêu cầu và kết quả kiểm tra |

Các hình: `sequence_lengths.png`, `learning_curves.png`, `confusion_matrix.png`, `sequence_length_comparison.png`. Checkpoint, history, predictions và kiểm chứng nằm trong `artifacts/`.

## Cấu hình

- IMDb: 50.000 mẫu gốc; sau loại trùng còn 49.578 mẫu. Split stratified seed 42: 29.746 train / 9.916 validation / 9.916 test, khớp Lab 04.
- Vocabulary 20.000, PAD=0/UNK=1, chỉ học trên train; giữ stopwords và phủ định. Right truncation, dynamic right-padding; dùng hidden tại token thật cuối.
- Recurrent: embedding 100, hidden 64, một lớp/một chiều, dropout 0,2; Adam lr=0,001; batch 128; gradient clipping 1; 8 epoch.
- Thử 3 architecture × max_length 50/100/200. Chọn checkpoint và model ứng dụng bằng F1 validation, threshold 0,5; không chọn bằng test.
- Baseline: TF-IDF unigram+bigram 50.000 features, min_df=2, LR C=2/liblinear; đọc toàn bộ review.

## Kết quả chính

| Model | Length | Test accuracy | Test F1 |
|---|---:|---:|---:|
| TF-IDF + LR | Toàn bộ | 0,9081 | 0,9096 |
| RNN | 100 | 0,7304 | 0,7544 |
| LSTM | 100 | 0,8199 | 0,8318 |
| GRU | 100 | 0,8200 | 0,8288 |
| GRU chọn bằng validation | 200 | 0,8562 | 0,8606 |

Model ứng dụng: `GRU_len200`, epoch 7, F1 validation 0,8612. Đầy đủ precision, recall, tham số, thời gian và memory trong `results.csv`.

## Chạy

Môi trường lần huấn luyện: Python 3.13, PyTorch 2.6.0+cu124, RTX 4050 Laptop. Thư viện trong `requirements.txt`.

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest -v test_implementation
python3 execute_notebook.py
python3 predict.py "I really enjoyed this movie."
```

Repo kèm checkpoint GRU-200 đã chọn và các bảng/biểu đồ cần cho notebook, nên có thể xem lại kết quả và chạy predict ngay sau khi cài thư viện. Model nạp được trên CPU; output sigmoid có thể chênh nhẹ so với GPU.

Muốn train lại toàn bộ, đặt IMDb CSV gồm hai cột `review`, `sentiment` trong `imdb.zip` tại đây hoặc `../lab04_nlp/`, rồi chạy:

```bash
PYTHONHASHSEED=42 python3 -u experiments.py --force
python3 execute_notebook.py
python3 verify_results.py
```

Có thể truyền `--data-source /path/to/imdb.zip`. Script tái sử dụng kết quả nếu source/cấu hình/split khớp; `--force` train lại. Kiểm chứng đầy đủ bằng `verify_results.py` cần các split và checkpoint của mọi cấu hình được tạo ở lần huấn luyện. Dữ liệu IMDb đầy đủ và các checkpoint còn lại không nằm trong repo.

Huấn luyện lại tái tạo các bảng mẫu và `error_analysis.md`; sao lưu phần nhận xét hiện có trước khi chạy.

`build_notebook.py` tạo lại notebook từ các mục nhận xét trong `discussion.md`, sau đó chạy `execute_notebook.py` để có output.

## Cách đọc số đo

Precision/recall/F1 dùng positive làm lớp dương; có thêm macro F1. Thời gian recurrent gồm train, validation và lưu checkpoint, không gồm chuẩn bị dữ liệu/test; `fit_seconds` chỉ đo train. GPU memory là peak allocated của PyTorch; CPU RSS chỉ lấy mẫu cuối epoch.

Chỉ có một seed; thời gian chưa đo lặp. Baseline và recurrent có lượng context khác nhau. Probe nhỏ không chứng minh vanishing gradient; sigmoid chưa được calibration. Reload cùng GPU khớp predictions đã lưu; kiểm tra 128 mẫu đầu/model trên CPU có độ lệch xác suất tối đa khoảng 0,001653, không đổi nhãn trong mẫu đã kiểm tra.
