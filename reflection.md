# Reflection

RNN/LSTM/GRU bổ sung cách biểu diễn văn bản theo thứ tự và context thay vì chỉ đếm từ hoặc lấy trung bình embedding. Trong run này, LSTM/GRU tốt hơn RNN ở cả ba độ dài, nhưng TF-IDF + LR vẫn có F1 cao nhất. Kiến trúc phức tạp hơn không tự bảo đảm kết quả tốt hơn; baseline đọc toàn bộ review còn recurrent bị giới hạn context.

Tăng max_length từ 50 lên 200 giúp giữ nhiều token và tăng F1 trong các cấu hình đã thử, đồng thời tăng GPU memory. Learning curves cho thấy nhiều model bắt đầu overfit. Vì vậy cần chọn checkpoint bằng validation và xem cả loss, precision, recall, F1 thay vì chỉ nhìn accuracy hoặc epoch cuối.

Các ví dụ lỗi cho thấy review dài, cảm xúc pha trộn và câu chuyển ý vẫn khó. Không thể quy mọi lỗi cho vanishing gradient: cần đối chiếu text được giữ, từ hiếm và cách diễn đạt. Bước tiếp theo phù hợp là đo nhiều seed, thử regularization/early stopping và tìm hiểu attention để truy cập thông tin ở nhiều vị trí thay vì chỉ dùng hidden cuối.
