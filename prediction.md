# Prediction

Tóm tắt dự đoán theo cơ chế để đối chiếu với thực nghiệm đã chạy.

| Câu | Prediction | Reason | Confidence |
|---|---|---|---|
| 1 | RNN xử lý thứ tự từ trực tiếp hơn TF-IDF + LR. | Hidden state cập nhật theo từng token; TF-IDF unigram không giữ thứ tự, bigram chỉ giữ quan hệ cục bộ. | Cao |
| 2 | LSTM có khả năng giữ dependency dài tốt hơn RNN thường. | Cell state và các gate điều chỉnh việc giữ, thêm và xuất thông tin; kết quả còn phụ thuộc dữ liệu và huấn luyện. | Cao về cơ chế; trung bình về mức tăng metrics |
| 3 | Tăng length từ 50 lên 200 thường làm RNN tốn thời gian và bộ nhớ hơn. | Có nhiều timestep cần xử lý và truyền gradient; dependency xa cũng khó học hơn. | Cao về chi phí; trung bình về độ khó tối ưu |
| 4 | Cắt review còn 30 token có thể làm giảm chất lượng classification. | Nhận xét cuối, phủ định hoặc đoạn chuyển ý có thể nằm sau phần được giữ. | Cao |
| 5 | LSTM không đảm bảo accuracy cao hơn RNN trên mọi dataset. | Dữ liệu, độ dài, nhiễu và hyperparameter ảnh hưởng kết quả; model đơn giản vẫn có thể phù hợp hơn. | Cao |
