# Hanoi PM2.5 Forecasting Project

## Tong quan

Du an nay du bao PM2.5 theo gio tai Ha Noi de ho tro sinh vien len ke hoach hoc tap, di chuyen va hoat dong hang ngay. Repo gom:

- bo du lieu da xu ly cho mo hinh hoa
- script tao du bao baseline
- ung dung Streamlit bang tieng Viet de nguoi dung khong biet lap trinh van co the xem du bao

## Chay nhanh ung dung web

Thuc hien trong thu muc `BDM-final`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run web/app.py
```

Sau khi chay lenh cuoi, trinh duyet se mo ung dung du bao PM2.5.

## Tinh nang cua ung dung

- Doc truc tiep file `predictions.csv`, khong can upload du lieu
- Hien thi du bao PM2.5 theo ngay va theo gio bang tieng Viet
- Su dung `linear_pred` lam du bao chinh
- Cho phep so sanh them voi baseline `persistence_pred`
- Dua ra goi y de sinh vien chon khung gio de di hoc, di chuyen, van dong ngoai troi
- Cho phep tai file CSV cua khung du lieu dang xem

## Chay script tao du bao

Neu can tao lai file du bao:

```powershell
python main/hanoi_pm25_forecast.py
```

Script se doc `hanoi_aqi_ml_ready_fixed.csv` va ghi ket qua vao `predictions.csv`.

## Thong tin du lieu

- 14,451 ban ghi theo gio
- Khoang thoi gian: `2024-02-14 09:00:00` den `2026-01-26 07:00:00`
- Bien muc tieu: `pm25`
- Du lieu khong co gia tri thieu theo README hien tai

## File chinh

- `hanoi_aqi_ml_ready_fixed.csv`: bo du lieu chinh
- `predictions.csv`: ket qua du bao de app su dung
- `main/hanoi_pm25_forecast.py`: script huan luyen va tao du bao baseline
- `web/app.py`: ung dung Streamlit

## Ghi chu

- Ung dung la cong cu ho tro hoc tap va lap ke hoach hang ngay, khong phai tu van y khoa.
- Muc PM2.5 cang thap thi chat luong khong khi cang tot.

