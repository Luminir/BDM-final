
# Hanoi PM2.5 Forecasting Project

## Tổng quan

Dự án này dự báo PM2.5 theo giờ tại Hà Nội để hỗ trợ sinh viên lên kế hoạch học tập, di chuyển và hoạt động hằng ngày. Repo gồm:

- Bộ dữ liệu đã xử lý cho mô hình hóa  
- Script tạo dự báo baseline  
- Ứng dụng Streamlit bằng tiếng Việt để người dùng không biết lập trình vẫn có thể xem dự báo  

## Chạy nhanh ứng dụng web

Thực hiện trong thư mục `BDM-final`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run web/app.py
````

Sau khi chạy lệnh cuối, trình duyệt sẽ mở ứng dụng dự báo PM2.5.

## Tính năng của ứng dụng

* Đọc trực tiếp `predictions.csv`, `model_bundle.joblib` và bộ dữ liệu lịch sử
* Tách 2 chế độ:

  * `Historical`: xem lại dự báo trên khoảng dữ liệu địa phương
  * `Upcoming planner`: lập kế hoạch 30 ngày tới với nguồn dự báo theo thứ tự ưu tiên
* Khung giờ chọn theo từng giờ (`00:00` → `23:00`), không còn bỏ qua thay đổi theo giờ
* Gán nhãn `Forecast source` và `Confidence` cho mỗi dự báo
* Cho phép so sánh thêm với baseline lag-1 khi dữ liệu lịch sử có sẵn
* Cho phép tải file CSV của khung dữ liệu đang xem

## Chạy script tạo dự báo

Nếu cần tạo lại file dự báo:

```powershell
python main/hanoi_pm25_forecast.py
```

Script sẽ đọc `hanoi_aqi_ml_ready_fixed.csv` và ghi:

* `predictions.csv`: dự báo holdout để app dùng lại
* `model_bundle.joblib`: mô hình local và metadata để app replay / dự báo tiếp

## Thông tin dữ liệu

* 14,451 bản ghi theo giờ
* Khoảng thời gian: `2024-02-14 09:00:00` đến `2026-01-26 07:00:00`
* Biến mục tiêu: `pm25`
* Dữ liệu không có giá trị thiếu theo README hiện tại

## Luồng dự báo trong app

* Ngày nằm trong bộ dữ liệu local: dùng dự báo local đã lưu hoặc replay từ `model_bundle.joblib`
* Từ hôm nay đến 7 ngày tới: ưu tiên Open-Meteo Air Quality API
* Ngày 8–16: dùng Open-Meteo weather + mô hình local
* Ngày 17–30: dùng fallback theo `day_of_week + hour`

## File chính

* `hanoi_aqi_ml_ready_fixed.csv`: bộ dữ liệu chính
* `predictions.csv`: kết quả dự báo để app sử dụng
* `model_bundle.joblib`: artifact mô hình local để app nạp lại
* `main/forecasting_core.py`: logic forecasting dùng chung
* `main/hanoi_pm25_forecast.py`: script huấn luyện và tạo artifact
* `web/app.py`: ứng dụng Streamlit

## Ghi chú

* Ứng dụng là công cụ hỗ trợ học tập và lập kế hoạch hằng ngày, không phải tư vấn y khoa.
* Mức PM2.5 càng thấp thì chất lượng không khí càng tốt.

