# Hanoi PM2.5 Air Quality Dataset with Weather Features

## Overview

14,451 timestamped PM2.5 observations in Hanoi, Vietnam (Feb 2024 to Jan 2026), enriched with temporal, lag, rolling, and weather-based features for forecasting tasks.

## Quick Stats

- Rows: 14,451
- Columns: 34 total (`datetime` + 1 target + 32 predictors)
- Time range (UTC): `2024-02-14 09:00:00` to `2026-01-26 07:00:00`
- Missing values: 0%
- Mean PM2.5: 33.28 ug/m3
- Median PM2.5: 28.41 ug/m3

## Column Groups

### Target variable

- `pm25`: PM2.5 concentration (ug/m3)

### Time and metadata

- `datetime`: hourly timestamp (UTC)
- `source`: PM2.5 data source label

### Temporal features (8)

- `year`, `month`, `day`, `hour`
- `day_of_week` (0=Monday, 6=Sunday)
- `is_weekend` (0/1)
- `season` (`dry` or `wet`)
- `is_dry_season` (0/1)

### Lag and rolling features (7)

- `pm25_lag1`
- `pm25_lag24`
- `pm25_lag168`
- `pm25_rolling_3h`
- `pm25_rolling_24h`
- `pm25_rolling_7d`
- `pm25_rolling_24h_std`

### Weather and derived features (16)

- `temperature`, `humidity`, `dew_point`
- `precipitation`, `rain`, `is_raining`
- `pressure_msl`, `surface_pressure`, `pressure_diff`
- `wind_speed`, `wind_direction`, `wind_gusts`, `wind_u`, `wind_v`
- `cloud_cover`
- `temp_humidity`

## Sample Usage

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

df = pd.read_csv(
    "hanoi_aqi_ml_ready_fixed.csv",
    parse_dates=["datetime"]
).sort_values("datetime")

split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

features = ["pm25_lag1", "pm25_rolling_24h", "temperature", "humidity"]
X_train = train[features]
y_train = train["pm25"]
X_test = test[features]
y_test = test["pm25"]

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("R2:", round(r2_score(y_test, pred), 3))
print("RMSE:", round(root_mean_squared_error(y_test, pred), 3))
```

Expected baseline with the split above: `R2 ~ 0.80`, `RMSE ~ 6.92 ug/m3`.

## Notes

- The dataset has no missing values and no duplicate timestamps.
- Timestamps are not perfectly continuous (there are occasional multi-hour gaps), so use time-aware validation.

## Files

- `hanoi_aqi_ml_ready_fixed.csv`: main modeling dataset
- `data_dictionary.csv`: column definitions, units, and examples

## License

CC0 (Public Domain)

## Acknowledgments

- OpenAQ + WAQI for PM2.5 data
- Open-Meteo for weather data


## FOCUS
`python -m venv .venv  `
`.\.venv\Scripts\Activate.ps1`
```bash
pip install -r requirements.txt
python main/hanoi_pm25_forecast.py
```

