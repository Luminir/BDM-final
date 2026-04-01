# "Hanoi Air Quality Forecast”
Thầy giáo: Chấp nhận có điều kiện : Đây là một proposal khá mạnh: câu hỏi rõ, dữ liệu hợp lý, baseline tốt, và có logic đánh giá theo thời gian tương đối chuẩn.
## Major concern:
• Proposal hơi tham khi muốn chạy cả Prophet, SARIMA, LSTM/GRU và cả PySpark. Điều này có thể vượt quá phạm vi project.
• Một số claim định lượng trong phần expected insights xuất hiện quá sớm khi chưa có phân tích.
• Cần đảm bảo không dùng feature engineered từ tương lai gây leakage.
## Minor concern:
• Có thể cắt bớt phần contextual narrative để proposal gọn hơn.
• Nên làm rõ horizon chính: 24h hay 168h.
Top fixes to require immediately
• Chọn 2–3 baseline/model đủ mạnh thay vì cố làm hết.
• Đóng khung lại expected outcome theo hướng giả thuyết cần kiểm chứng, không phải kết luận dự báo sẵn.																	
