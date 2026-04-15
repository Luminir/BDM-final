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

- Doc truc tiep `predictions.csv`, `model_bundle.joblib` va bo du lieu lich su
- Tach 2 che do:
  - `Historical`: xem lai du bao tren khoang du lieu dia phuong
  - `Upcoming planner`: lap ke hoach 30 ngay toi voi nguon du bao theo thu tu uu tien
- Khung gio chon theo tung gio (`00:00` -> `23:00`), khong con bo qua thay doi theo gio
- Gan nhan `Forecast source` va `Confidence` cho moi du bao
- Cho phep so sanh them voi baseline lag-1 khi du lieu lich su co san
- Cho phep tai file CSV cua khung du lieu dang xem

## Chay script tao du bao

Neu can tao lai file du bao:

```powershell
python main/hanoi_pm25_forecast.py
```

Script se doc `hanoi_aqi_ml_ready_fixed.csv` va ghi:

- `predictions.csv`: du bao holdout de app dung lai
- `model_bundle.joblib`: mo hinh local va metadata de app replay / du bao tiep

## Thong tin du lieu

- 14,451 ban ghi theo gio
- Khoang thoi gian: `2024-02-14 09:00:00` den `2026-01-26 07:00:00`
- Bien muc tieu: `pm25`
- Du lieu khong co gia tri thieu theo README hien tai

## Luong du bao trong app

- Ngay nam trong bo du lieu local: dung du bao local da luu hoac replay tu `model_bundle.joblib`
- Tu hom nay den 7 ngay toi: uu tien Open-Meteo Air Quality API
- Ngay 8-16: dung Open-Meteo weather + mo hinh local
- Ngay 17-30: dung fallback theo `day_of_week + hour`

## File chinh

- `hanoi_aqi_ml_ready_fixed.csv`: bo du lieu chinh
- `predictions.csv`: ket qua du bao de app su dung
- `model_bundle.joblib`: artifact mo hinh local de app nap lai
- `main/forecasting_core.py`: logic forecasting dung chung
- `main/hanoi_pm25_forecast.py`: script huan luyen va tao artifact
- `web/app.py`: ung dung Streamlit

## Ghi chu

- Ung dung la cong cu ho tro hoc tap va lap ke hoach hang ngay, khong phai tu van y khoa.
- Muc PM2.5 cang thap thi chat luong khong khi cang tot.

