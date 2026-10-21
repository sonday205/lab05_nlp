# Đối chiếu W5

Phạm vi: nội dung kỹ thuật, file bài nộp và khả năng chạy lại kết quả. Số đo huấn luyện giữ từ ngày 15/10/2026.

| Yêu cầu | Kết quả | Bằng chứng |
|---|---|---|
| Phần tính toán và giải thích viết tay | Có | `calculations.pdf` 2 trang |
| 5 prediction có reason và confidence | Có nội dung; chưa xác minh thời điểm | `prediction.md` là bản tổng hợp để đối chiếu thực nghiệm đã chạy; không có bản gốc trước huấn luyện để kiểm tra |
| Kiểm tra số mẫu, lớp, thiếu dữ liệu, trùng lặp, min/median/mean/max length | Đạt | `data/inspection.json`, `data/split_summary.csv`, notebook mục 1 |
| Histogram sequence length và lý do chọn max_length | Đạt | `sequence_lengths.png`, `discussion.md` |
| Tokenizer, vocabulary, PAD/UNK, padding/truncation | Đạt | Vocabulary chỉ học trên train; notebook mục 2 |
| Tự viết SimpleRNNCell, đối chiếu framework | Đạt | `rnn_cell.py`, test forward/gradient, notebook mục 3 |
| TF-IDF + LR baseline | Đạt | `results.csv`, notebook mục 4 |
| RNN/LSTM/GRU với điều kiện huấn luyện tương đồng | Đạt | 3 architecture × 3 length, mỗi cấu hình 8 epoch; `artifacts/run_manifest.json` |
| Length 50/100/200, % token giữ, thời gian, loss, F1, memory | Đạt | `data/token_retention.csv`, `results.csv`, `sequence_length_comparison.png` |
| Learning curves train/validation và nhận xét overfitting | Đạt | `learning_curves.png`, notebook mục 6 |
| Probe đầu/giữa/cuối, phân biệt quan sát và lý thuyết | Đạt | 80 prediction; notebook mục 9 ghi rõ giới hạn probe |
| Accuracy/precision/recall/F1, tham số, epoch, length, vocabulary | Đạt | `results.csv`, báo cáo từng model trong `artifacts/` |
| Confusion matrix; phân tích ít nhất 3 đúng và 3 sai | Đạt | `confusion_matrix.png`, `error_analysis.md`, `artifacts/error_examples.csv` |
| So sánh Lab 04, giới hạn recurrent, hướng attention | Đạt | Notebook mục 12, `discussion.md`, `reflection.md` |
| Ứng dụng review ngắn/dài/negation/contrast và giải thích context | Đạt | `predict.py`, checkpoint GRU-200, notebook mục 11 |
| Individual learning check | Có phần ôn tập | Notebook mục 15; đánh giá vấn đáp trực tiếp do giảng viên thực hiện |
| Các mục deliverables mục 29 | Có, phần tính toán dùng PDF | Giữ `calculations.pdf` theo định dạng được chọn; kèm source huấn luyện và kết quả hỗ trợ |

## Kiểm chứng

- 6 unit test: forward, gradient, nhiều timestep, padding, tokenizer/vocabulary và collation.
- Notebook: 14 cell code đã chạy, không có error output.
- Metrics được tính lại từ predictions của cả 10 cấu hình trên 9.916 mẫu test.
- Checkpoint reload trên GPU được đối chiếu với predictions gốc; CPU/GPU được báo cáo riêng.
- Bản public được kiểm tra trong thư mục độc lập, không có dữ liệu IMDb đầy đủ hoặc thư mục Lab 04; notebook dùng bảng tham chiếu đã lưu và checkpoint GRU-200.

## Điểm đã chỉnh khi rà soát

- Giữ phần tính toán bằng `calculations.pdf`; README dẫn trực tiếp tới PDF.
- Thêm fallback đọc bảng Lab 04 đã kiểm chứng khi không có thư mục lab trước.
- Kèm checkpoint ứng dụng và các bảng/biểu đồ cần thiết; hướng dẫn kiểm chứng đầy đủ sau khi train lại trong README.
