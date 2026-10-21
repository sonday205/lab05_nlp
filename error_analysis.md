# Error analysis

Model: `GRU_len200`, chọn bằng F1 validation. Dưới đây là 3 mẫu đúng và 3 mẫu sai; xác suất là P(positive).

| source_id | Nhãn thật | Dự đoán | P(positive) | Token gốc/giữ | Phân tích |
|---|---|---|---:|---:|---|
| 14141 | negative | negative | 0.0026 | 194/194 | Các cụm "crap", "too bad to be watched", "not good", "music is bad" nhất quán với nhãn negative. Review 194 token được giữ đủ; dự đoán đúng, không có dấu hiệu lỗi do truncation. |
| 4254 | negative | negative | 0.0023 | 214/200 | Dù mở đầu có "a little fun", đoạn giữ lại chứa "this bad", "drag forever" và "the worst film" nên prediction negative phù hợp. Có cắt 14/214 token, nhưng phần được giữ đã có nhiều bằng chứng tiêu cực. |
| 25127 | positive | positive | 0.7859 | 128/128 | Các cụm "remarkably natural", "remarkably realistic", "very witty" khen bộ phim; "my only complaint" chỉ nêu một hạn chế. Model dự đoán positive đúng trên đủ 128 token, dù review có cảm xúc pha trộn. |
| 9212 | negative | positive | 0.5229 | 538/200 | Review negative dài 538 token, model chỉ thấy 200 và dự đoán positive sát ngưỡng (0,5229). Truncation làm mất 338 token và cắt giữa câu, nhưng phần giữ lại vẫn có "painful experience", "repellent", "disgusting"; do đó chưa thể quy lỗi chỉ cho truncation hay dependency dài. |
| 33874 | negative | positive | 0.7587 | 193/193 | Không bị truncate (193 token). Review có "lost interest" nhưng cũng khen diễn xuất bằng "extremely convincingly", "a great performance" và kết thúc với "positive aspects"; sentiment pha trộn có thể khiến model nghiêng positive so với nhãn negative. Chưa có bằng chứng để kết luận lỗi do dependency dài hoặc nhãn sai. |
| 3848 | negative | positive | 0.7503 | 163/163 | Không bị truncate (163 token). Các lời khen như "masterpiece" và "considerably better" nói về phim được đem ra so sánh, trong khi phim đang đánh giá bị chê "embarrassing", "none of intensity or realism". Model có thể gán tín hiệu khen nhầm đối tượng hoặc xử lý chưa tốt phủ định/so sánh; đây là giả thuyết về context, chưa phải nguyên nhân đã xác nhận. |

Review gốc và phần token thực sự được giữ nằm trong `artifacts/error_examples.csv`. Các nguyên nhân lỗi là giả thuyết từ văn bản và prediction; chưa có thí nghiệm kiểm soát để tách riêng ảnh hưởng của vocabulary, context và truncation.
