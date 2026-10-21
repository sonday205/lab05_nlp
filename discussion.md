# Nhận xét thực nghiệm

Số liệu lấy từ lần chạy đã lưu ngày 15/10/2026. Chỉ có một seed, không suy rộng thành kết luận cho mọi dataset.

## Dataset và lựa chọn max_length

Dữ liệu gốc có 50.000 review, hai lớp cân bằng, không thiếu text/label. Loại 422 bản trùng sau chuẩn hóa còn 49.578 mẫu. Train có median 173, mean 230,10 và p95 588 token; cần xem phân bố độ dài và tỉ lệ token giữ lại khi chọn max_length.

Với train, length 50/100/200 giữ lần lượt 21,63%/41,74%/68,81% tổng token. Chọn 200 trong phạm vi khảo sát để giữ nhiều context hơn, nhưng vẫn cắt khoảng 40,80% review. Mốc 30 chỉ giữ 13,03% tổng token; phần kết luận hoặc chuyển ý ở cuối dễ bị mất.

## Padding và truncation

Padding thêm PAD để ghép batch, không bổ sung nội dung ngôn ngữ. Truncation bỏ token vượt giới hạn nên có thể mất thông tin. Model lấy hidden tại token thật cuối, tránh dùng hidden sau các PAD. Vocabulary chỉ học trên train để tránh rò rỉ dữ liệu.

## RNN forward và information flow

`x_t` là embedding hiện tại, `h_prev` là trạng thái từ các token trước. Hai phép nhân ma trận đưa chúng về cùng không gian hidden; cộng bias rồi qua `tanh` để tạo `h_t`. Các trọng số dùng chung ở mọi timestep. Vì `h_t` phụ thuộc `h_prev`, hai câu cùng từ nhưng khác thứ tự có thể tạo trạng thái cuối khác nhau.

Khi đọc `like`, thông tin `not` từ bước trước có thể làm thay đổi cách biểu diễn cảm xúc. Average Word2Vec lấy trung bình nên không giữ quan hệ thứ tự trực tiếp. Với câu dài, trạng thái cuối có thể mất chi tiết ở xa; gradient truyền qua nhiều bước có thể rất nhỏ hoặc rất lớn. Gradient clipping hạn chế gradient quá lớn, không giải quyết triệt để vanishing gradient.

## LSTM gates và GRU

Forget gate điều chỉnh lượng cell state cũ được giữ; input gate điều chỉnh thông tin mới; output gate điều chỉnh phần cell state đưa ra hidden state. Forget gate gần 1 giữ nhiều thông tin cũ, gần 0 loại phần lớn thông tin đó. Đường cập nhật cell state giúp LSTM truyền thông tin qua nhiều bước, nhưng không đảm bảo nhớ mọi dependency.

GRU dùng update/reset gate và không có cell state riêng. Tại cùng hidden size, GRU có ít tham số hơn LSTM: 2.031.937 so với 2.042.561, tính cả embedding; RNN có 2.010.689 tham số.

## Baseline và so sánh architecture

Tại length 100, F1 của RNN/LSTM/GRU là 0,7544/0,8318/0,8288; accuracy tương ứng 0,7304/0,8199/0,8200. GRU có accuracy nhỉnh hơn LSTM rất nhỏ nhưng F1 thấp hơn, nên không thể xếp hạng chỉ từ accuracy. Các model gated tốt hơn RNN trong lần chạy này, phù hợp với kỳ vọng về khả năng giữ context, nhưng chưa chứng minh riêng cơ chế nào tạo ra mức tăng.

TF-IDF + LR đạt F1 0,9096, cao hơn cả các recurrent model đã thử. Baseline đọc toàn bộ review còn recurrent bị truncate, nên đây không phải phép so sánh cùng lượng context. Kết quả không ủng hộ nhận định model sequence luôn tốt hơn model tĩnh.

## Sequence length và computational cost

Với LSTM, length 50 → 100 → 200 làm test F1 tăng 0,7842 → 0,8318 → 0,8638; GPU memory tăng khoảng 153,61 → 207,85 → 317,06 MiB. RNN và GRU cũng tăng F1 trong ba mốc đã thử. Chưa thể kết luận tăng length luôn cải thiện performance ngoài phạm vi này.

Thời gian LSTM đo được là 14,89/18,35/14,44 giây, không tăng đơn điệu; GRU cũng dao động nhẹ. Do chỉ đo một lần, không thể dùng các số này để kết luận length lớn chạy nhanh hơn. RNN 50 → 200 tăng từ 14,76 lên 25,37 giây. Trade-off context/bộ nhớ thể hiện rõ hơn trade-off thời gian trong run này; cần đo lặp để so sánh tốc độ chắc chắn hơn.

## Learning curves và overfitting

Nhiều model có train loss tiếp tục giảm trong khi validation loss tăng. Ví dụ GRU-200: validation loss thấp nhất khoảng 0,3627 ở epoch 5, tăng lên 0,5103 ở epoch 8; train loss giảm từ 0,2411 xuống 0,1217. Đây là dấu hiệu overfitting theo loss.

Checkpoint ứng dụng ở epoch 7 vì F1 validation cao nhất, không phải vì loss thấp nhất. Loss đo cả độ tin cậy xác suất, còn F1 dùng nhãn sau threshold 0,5 nên hai tiêu chí có thể chọn epoch khác nhau. LSTM/GRU length 50 và 100 cũng có khoảng cách train/validation tăng về cuối; không nên chỉ nhìn epoch cuối hoặc train loss.

## Thứ tự từ và bằng chứng thực nghiệm

Trên 512 review với cùng 100 token đầu, xáo trộn làm F1 LSTM-200 giảm từ 0,8275 xuống 0,8185; RNN-200 và GRU-200 lại tăng nhẹ. TF-IDF bigram giảm từ 0,8497 xuống 0,8476. Kết quả cho thấy prediction phụ thuộc cách sắp xếp token, nhưng chưa chứng minh mọi recurrent model tận dụng thứ tự tốt hơn baseline.

Khả năng cập nhật hidden theo thứ tự là thuộc tính kiến trúc; lợi ích classification phải được kiểm tra bằng dữ liệu. Các câu `dog bites man` chỉ là probe sentiment, không phải bài kiểm tra model hiểu vai nghĩa của câu.

## Long-range dependency

Với câu tổng hợp 200 token, LSTM-200 cho P(positive) khi từ `terrible` ở đầu/giữa/cuối lần lượt khoảng 0,7646/0,7536/0,2702. Cùng từ khóa và độ dài nhưng đổi vị trí đã đổi prediction; ở ví dụ này, từ gần cuối có tác động rõ hơn. GRU-200 lại dự đoán negative cho cả ba vị trí của `excellent`, cho thấy probe lặp từ có thể tạo hành vi khác review tự nhiên.

Đây là quan sát trên tập nhỏ, ngoài phân bố review thông thường; từ nền bị thay tại mỗi vị trí cũng có thể khác nhau, nên chưa cô lập hoàn toàn ảnh hưởng vị trí. Nó không trực tiếp đo gradient và không chứng minh đầy đủ vanishing gradient; giải thích lý thuyết cần phân biệt với kết quả probe.

## Confusion matrix và model ứng dụng

Tại length 100, false positive/false negative của RNN là 1.803/870, LSTM là 1.225/561, GRU là 1.129/656. Các model đều nhầm negative thành positive nhiều hơn chiều ngược lại; cần xem cả precision và recall để nhận ra xu hướng này.

GRU-200 được chọn bằng F1 validation 0,8612; test accuracy 0,8562, precision 0,8377, recall 0,8848 và F1 0,8606. LSTM-200 có test F1 0,8638 cao hơn một chút, nhưng không đổi lựa chọn model dựa trên test.

## Phân tích ví dụ ứng dụng

- **Ngắn:** `enjoyed`, `excellent` là tín hiệu tích cực; model dự đoán positive, P(positive)=0,9586.
- **Dài:** review có 301 token, chỉ giữ 200. Cụm `loved ... excellent performances` ở cuối bị cắt, nên không thể giải thích P(positive)=0,9958 bằng cụm này. Prediction dựa trên phần mô tả lặp được giữ; ví dụ cho thấy cần kiểm tra input thực tế và không xem xác suất cao là bảo đảm đúng.
- **Negation:** `did not enjoy`, `not good`, `not recommend` cùng thể hiện đánh giá tiêu cực; prediction negative, P(positive)=0,1049. Một ví dụ đúng chưa chứng minh model xử lý mọi phủ định.
- **Contrast:** câu kỳ vọng phim tệ nhưng sau đó đánh giá `surprisingly good` có ý cuối tích cực; model lại dự đoán negative, P(positive)=0,2631. Bỏ `terrible` làm P(positive) tăng lên 0,5931, còn bỏ `good` làm giảm xuống 0,1712. Kết quả phù hợp với việc tín hiệu tiêu cực lấn át chuyển ý, nhưng token ablation không xác định duy nhất nguyên nhân bên trong model.

## So sánh với Lab 04 và giới hạn

Average Word2Vec + LR đạt F1 0,8675 trên predictions Lab 04 đã đối chiếu lại, không được train lại trong run này. RNN/LSTM/GRU giữ trạng thái phụ thuộc context thay vì lấy trung bình toàn câu nên có khả năng phân biệt thứ tự, phủ định và chuyển ý tốt hơn về biểu diễn. Khả năng đó không bảo đảm metrics cao hơn: cấu hình recurrent hiện tại chưa vượt TF-IDF + LR.

Giới hạn còn lại gồm truncation, từ hiếm/OOV, context bị nén vào hidden state, huấn luyện tuần tự, overfitting và một seed duy nhất. Attention cho phép kết hợp trực tiếp thông tin từ nhiều vị trí, giảm việc chỉ dựa vào trạng thái cuối; vẫn cần kiểm tra dữ liệu, chi phí và chất lượng thực tế.
