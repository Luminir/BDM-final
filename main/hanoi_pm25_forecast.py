from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_CSV = "hanoi_aqi_ml_ready_fixed.csv"
DEFAULT_ZIP = "hanoi-air-quality-pm2-5-weather-data-2024-2026.zip"
FEATURE_COLUMNS = ["pm25_lag1", "pm25_rolling_24h", "temperature", "humidity"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate baseline PM2.5 forecasting models on Hanoi dataset."
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_CSV,
        help=f"Path to CSV dataset (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--zip",
        default=DEFAULT_ZIP,
        help=f"Path to zip archive used when CSV is missing (default: {DEFAULT_ZIP})",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test ratio for time-based split (default: 0.2)",
    )
    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Path to write prediction results CSV (default: predictions.csv)",
    )
    return parser.parse_args()


def ensure_dataset(csv_path: Path, zip_path: Path) -> Path:
    if csv_path.exists():
        return csv_path

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset not found. Missing both '{csv_path}' and '{zip_path}'."
        )

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        if csv_path.name in members:
            zf.extract(csv_path.name, csv_path.parent)
            return csv_path

        candidates = [m for m in members if m.lower().endswith(".csv")]
        if not candidates:
            raise FileNotFoundError("No CSV file found inside the zip archive.")

        first_csv = candidates[0]
        zf.extract(first_csv, csv_path.parent)

        extracted = csv_path.parent / first_csv
        extracted.rename(csv_path)
        return csv_path


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    required_cols = {"datetime", "pm25", *FEATURE_COLUMNS}
    missing = required_cols.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KeyError(f"Dataset is missing required columns: {missing_text}")

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def time_split(df: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (0 < test_ratio < 1):
        raise ValueError("--test-ratio must be between 0 and 1.")

    split_idx = int(len(df) * (1 - test_ratio))
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("Invalid split index. Adjust --test-ratio.")

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def calc_metrics(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    y_true_np = np.asarray(y_true, dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true_np, y_pred_np)))
    mae = float(mean_absolute_error(y_true_np, y_pred_np))
    r2 = float(r2_score(y_true_np, y_pred_np))

    non_zero = y_true_np != 0
    if non_zero.any():
        mape = float(np.mean(np.abs((y_true_np[non_zero] - y_pred_np[non_zero]) / y_true_np[non_zero])) * 100)
    else:
        mape = float("nan")

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def format_metrics(name: str, metrics: dict[str, float]) -> str:
    return (
        f"{name:<14} | "
        f"RMSE: {metrics['RMSE']:.3f} | "
        f"MAE: {metrics['MAE']:.3f} | "
        f"R2: {metrics['R2']:.4f} | "
        f"MAPE: {metrics['MAPE']:.2f}%"
    )


def safe_text(value: Path | str) -> str:
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    try:
        text.encode(encoding)
        return text
    except UnicodeEncodeError:
        return text.encode("ascii", "backslashreplace").decode("ascii")


def main() -> int:
    args = parse_args()
    csv_path = Path(args.data)
    zip_path = Path(args.zip)
    output_path = Path(args.output)

    try:
        dataset_path = ensure_dataset(csv_path, zip_path)
        df = load_data(dataset_path)
        train, test = time_split(df, args.test_ratio)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("Dataset loaded")
    print(f"- Rows: {len(df):,}")
    print(f"- Columns: {len(df.columns)}")
    print(f"- Time range: {df['datetime'].min()} -> {df['datetime'].max()}")
    print(f"- Train rows: {len(train):,} | Test rows: {len(test):,}")
    print()

    y_train = train["pm25"]
    y_test = test["pm25"]

    # Baseline model: use lag-1 as naive persistence estimate.
    persistence_pred = test["pm25_lag1"]
    persistence_metrics = calc_metrics(y_test, persistence_pred)

    # Linear regression baseline.
    model = LinearRegression()
    model.fit(train[FEATURE_COLUMNS], y_train)
    linear_pred = model.predict(test[FEATURE_COLUMNS])
    linear_metrics = calc_metrics(y_test, linear_pred)

    print("Model performance")
    print(format_metrics("Persistence", persistence_metrics))
    print(format_metrics("LinearReg", linear_metrics))

    out_df = pd.DataFrame(
        {
            "datetime": test["datetime"],
            "actual_pm25": y_test,
            "persistence_pred": persistence_pred,
            "linear_pred": linear_pred,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print()
    print(f"Saved predictions to: {safe_text(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
