from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_CSV = "hanoi_aqi_ml_ready_fixed.csv"
DEFAULT_ZIP = "hanoi-air-quality-pm2-5-weather-data-2024-2026.zip"
DEFAULT_MODEL_OUTPUT = "model_bundle.joblib"
NON_FEATURE_COLUMNS = {"datetime", "pm25", "source", "season"}
PREDICTION_OUTPUT_COLUMNS = [
    "datetime",
    "actual_pm25",
    "forecast_pm25",
    "baseline_pred",
    "source_label",
]


def ensure_dataset(csv_path: Path, zip_path: Path) -> Path:
    if csv_path.exists():
        return csv_path

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset not found. Missing both '{csv_path}' and '{zip_path}'."
        )

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        members = zip_file.namelist()
        if csv_path.name in members:
            zip_file.extract(csv_path.name, csv_path.parent)
            return csv_path

        csv_candidates = [member for member in members if member.lower().endswith(".csv")]
        if not csv_candidates:
            raise FileNotFoundError("No CSV file found inside the zip archive.")

        first_csv = csv_candidates[0]
        zip_file.extract(first_csv, csv_path.parent)
        extracted_path = csv_path.parent / first_csv
        extracted_path.rename(csv_path)
        return csv_path


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    required_columns = {"datetime", "pm25"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise KeyError(f"Dataset is missing required columns: {missing_text}")

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_columns = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns found for model training.")
    return feature_columns


def time_split(df: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (0 < test_ratio < 1):
        raise ValueError("--test-ratio must be between 0 and 1.")

    split_index = int(len(df) * (1 - test_ratio))
    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Invalid split index. Adjust --test-ratio.")

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def calc_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    y_true_np = np.asarray(y_true, dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true_np, y_pred_np)))
    mae = float(mean_absolute_error(y_true_np, y_pred_np))
    r2 = float(r2_score(y_true_np, y_pred_np))

    non_zero = y_true_np != 0
    if non_zero.any():
        mape = float(
            np.mean(np.abs((y_true_np[non_zero] - y_pred_np[non_zero]) / y_true_np[non_zero]))
            * 100
        )
    else:
        mape = float("nan")

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def format_metrics(name: str, metrics: dict[str, float]) -> str:
    return (
        f"{name:<18} | "
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


def build_dow_hour_profile(df: pd.DataFrame) -> dict[str, object]:
    profile_df = df[["datetime", "pm25"]].copy()
    profile_df["day_of_week"] = profile_df["datetime"].dt.dayofweek
    profile_df["hour"] = profile_df["datetime"].dt.hour

    return {
        "dow_hour_profile": profile_df.groupby(["day_of_week", "hour"])["pm25"].mean(),
        "global_mean": float(profile_df["pm25"].mean()),
    }


def apply_dow_hour_profile(
    target_datetimes: pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
    profile_bundle: dict[str, object],
) -> pd.DataFrame:
    forecast_df = pd.DataFrame({"datetime": pd.to_datetime(target_datetimes)})
    forecast_df["day_of_week"] = forecast_df["datetime"].dt.dayofweek
    forecast_df["hour"] = forecast_df["datetime"].dt.hour
    profile = profile_bundle["dow_hour_profile"]
    global_mean = float(profile_bundle["global_mean"])

    forecast_df["forecast_pm25"] = [
        float(profile.get((day_of_week, hour), global_mean))
        for day_of_week, hour in zip(forecast_df["day_of_week"], forecast_df["hour"])
    ]
    return forecast_df[["datetime", "forecast_pm25"]]


def build_predictions_frame(
    test_df: pd.DataFrame,
    forecast_values: np.ndarray,
    baseline_values: np.ndarray,
    source_label: str,
) -> pd.DataFrame:
    output_df = pd.DataFrame(
        {
            "datetime": test_df["datetime"],
            "actual_pm25": test_df["pm25"],
            "forecast_pm25": np.clip(np.asarray(forecast_values, dtype=float), 0, None),
            "baseline_pred": np.asarray(baseline_values, dtype=float),
            "source_label": source_label,
        }
    )
    return output_df[PREDICTION_OUTPUT_COLUMNS]


def train_local_model(
    df: pd.DataFrame,
    test_ratio: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    feature_columns = get_feature_columns(df)
    usable_df = df.dropna(subset=["pm25", *feature_columns]).copy()
    train_df, test_df = time_split(usable_df, test_ratio)

    model = LinearRegression()
    model.fit(train_df[feature_columns], train_df["pm25"])

    baseline_pred = test_df["pm25_lag1"].to_numpy(dtype=float)
    local_pred = np.clip(model.predict(test_df[feature_columns]), 0, None)

    baseline_metrics = calc_metrics(test_df["pm25"], baseline_pred)
    local_model_metrics = calc_metrics(test_df["pm25"], local_pred)

    fallback_eval_profile = build_dow_hour_profile(train_df)
    fallback_eval_pred = apply_dow_hour_profile(
        test_df["datetime"],
        fallback_eval_profile,
    )["forecast_pm25"].to_numpy(dtype=float)
    fallback_metrics = calc_metrics(test_df["pm25"], fallback_eval_pred)

    predictions_df = build_predictions_frame(
        test_df=test_df,
        forecast_values=local_pred,
        baseline_values=baseline_pred,
        source_label="Saved local model forecast",
    )

    model_bundle = {
        "artifact_version": 2,
        "feature_columns": feature_columns,
        "model": model,
        "metrics": {
            "baseline": baseline_metrics,
            "local_model": local_model_metrics,
            "fallback": fallback_metrics,
        },
        "local_history_start": usable_df["datetime"].min(),
        "local_history_end": usable_df["datetime"].max(),
        "test_window_start": test_df["datetime"].min(),
        "test_window_end": test_df["datetime"].max(),
        "planner_profile": build_dow_hour_profile(usable_df),
    }
    return model_bundle, predictions_df


def predict_local_model(
    feature_df: pd.DataFrame,
    model_bundle: dict[str, object],
) -> np.ndarray:
    feature_columns = model_bundle["feature_columns"]
    missing_columns = [column for column in feature_columns if column not in feature_df.columns]
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise KeyError(f"Feature frame is missing model columns: {missing_text}")

    prediction_values = model_bundle["model"].predict(feature_df[feature_columns])
    return np.clip(np.asarray(prediction_values, dtype=float), 0, None)


def build_historical_forecast_frame(
    history_df: pd.DataFrame,
    saved_predictions_df: pd.DataFrame,
    model_bundle: dict[str, object],
) -> pd.DataFrame:
    replay_pred = predict_local_model(history_df, model_bundle)
    history_forecasts = pd.DataFrame(
        {
            "datetime": history_df["datetime"],
            "forecast_pm25": replay_pred,
            "actual_pm25": history_df["pm25"],
            "baseline_pred": history_df["pm25_lag1"],
            "source_label": "Local model replay from dataset features",
            "confidence_label": "High confidence",
        }
    )

    if saved_predictions_df.empty:
        return history_forecasts

    merged_df = history_forecasts.merge(
        saved_predictions_df[
            ["datetime", "forecast_pm25", "baseline_pred", "source_label"]
        ].rename(
            columns={
                "forecast_pm25": "saved_forecast_pm25",
                "baseline_pred": "saved_baseline_pred",
                "source_label": "saved_source_label",
            }
        ),
        on="datetime",
        how="left",
    )

    saved_mask = merged_df["saved_forecast_pm25"].notna()
    merged_df.loc[saved_mask, "forecast_pm25"] = merged_df.loc[saved_mask, "saved_forecast_pm25"]
    merged_df.loc[saved_mask, "baseline_pred"] = merged_df.loc[saved_mask, "saved_baseline_pred"]
    merged_df.loc[saved_mask, "source_label"] = merged_df.loc[saved_mask, "saved_source_label"]
    merged_df.loc[saved_mask, "confidence_label"] = "High confidence"

    return merged_df[
        [
            "datetime",
            "forecast_pm25",
            "source_label",
            "confidence_label",
            "actual_pm25",
            "baseline_pred",
        ]
    ]


def get_safety_recommendations(pm25: float, construction_risk: str) -> dict[str, str]:
    """Generates safety advice for different categories based on PM2.5 and construction risk."""
    is_high_risk = construction_risk == "High"
    is_mod_risk = construction_risk == "Moderate"

    # Base advice based on PM2.5
    if pm25 <= 12:
        base_cat = "excellent"
    elif pm25 <= 35.4:
        base_cat = "good"
    elif pm25 <= 55.4:
        base_cat = "moderate"
    elif pm25 <= 150.4:
        base_cat = "unhealthy"
    else:
        base_cat = "hazardous"

    recommendations = {}

    # Category: Exercise
    if base_cat == "excellent":
        msg = "Perfect conditions for high-intensity outdoor training."
    elif base_cat == "good":
        msg = "Good for exercise, but sensitive groups should monitor symptoms."
    elif base_cat == "moderate":
        msg = "Consider switching to indoor exercise or reducing intensity."
    else:
        msg = "Avoid outdoor exercise. Use indoor facilities with air filtration."
    
    if is_high_risk:
        msg += " Warning: High construction activity nearby may cause localized dust spikes."
    recommendations["Exercise"] = msg

    # Category: Commuting
    if base_cat in ["excellent", "good"]:
        msg = "Open-air commuting (cycling, walking) is safe."
    elif base_cat == "moderate":
        msg = "Standard face mask recommended for long commutes."
    else:
        msg = "Wear an N95 mask and keep vehicle windows closed."
    
    if is_mod_risk or is_high_risk:
        msg += " Pro tip: Active construction may lead to traffic congestion and road dust."
    recommendations["Commuting"] = msg

    # Category: Hanging Out
    if base_cat in ["excellent", "good"]:
        msg = "Great weather for outdoor cafes or park visits."
    elif base_cat == "moderate":
        msg = "Outdoor hangouts are fine, but prefer venues away from main roads."
    else:
        msg = "Opt for indoor malls or air-conditioned cafes."
    
    if is_high_risk:
        msg += " Avoid areas near large project sites for social gatherings."
    recommendations["Hanging Out"] = msg

    return recommendations


def standardize_forecast_frame(
    forecast_df: pd.DataFrame,
    *,
    source_label: str,
    confidence_label: str,
) -> pd.DataFrame:
    standardized = forecast_df.copy()
    standardized["source_label"] = source_label
    standardized["confidence_label"] = confidence_label
    if "actual_pm25" not in standardized.columns:
        standardized["actual_pm25"] = np.nan
    if "baseline_pred" not in standardized.columns:
        standardized["baseline_pred"] = np.nan
    return standardized[
        [
            "datetime",
            "forecast_pm25",
            "source_label",
            "confidence_label",
            "actual_pm25",
            "baseline_pred",
        ]
    ]


def engineer_weather_features(
    weather_df: pd.DataFrame,
    previous_pressure_msl: float | None = None,
) -> pd.DataFrame:
    feature_df = weather_df.copy()
    feature_df["datetime"] = pd.to_datetime(feature_df["datetime"])
    feature_df = feature_df.sort_values("datetime").reset_index(drop=True)

    dt = feature_df["datetime"]
    feature_df["year"] = dt.dt.year
    feature_df["month"] = dt.dt.month
    feature_df["day"] = dt.dt.day
    feature_df["hour"] = dt.dt.hour
    feature_df["day_of_week"] = dt.dt.dayofweek
    feature_df["is_weekend"] = (feature_df["day_of_week"] >= 5).astype(int)
    feature_df["is_dry_season"] = feature_df["month"].isin([11, 12, 1, 2, 3, 4]).astype(int)

    radians = np.radians(feature_df["wind_direction"].astype(float))
    feature_df["wind_u"] = feature_df["wind_speed"].astype(float) * np.cos(radians)
    feature_df["wind_v"] = feature_df["wind_speed"].astype(float) * np.sin(radians)
    feature_df["temp_humidity"] = (
        feature_df["temperature"].astype(float) * feature_df["humidity"].astype(float) / 100.0
    )
    feature_df["pressure_diff"] = feature_df["pressure_msl"].astype(float).diff()
    if not feature_df.empty:
        if previous_pressure_msl is None:
            feature_df.loc[feature_df.index[0], "pressure_diff"] = 0.0
        else:
            feature_df.loc[feature_df.index[0], "pressure_diff"] = (
                float(feature_df.loc[feature_df.index[0], "pressure_msl"]) - previous_pressure_msl
            )
    feature_df["is_raining"] = (
        (feature_df["rain"].astype(float) > 0) | (feature_df["precipitation"].astype(float) > 0)
    ).astype(int)
    return feature_df


def recursive_weather_forecast(
    weather_df: pd.DataFrame,
    history_df: pd.DataFrame,
    model_bundle: dict[str, object],
    fixed_future_pm25: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if weather_df.empty:
        return pd.DataFrame(columns=["datetime", "forecast_pm25", "source_label", "confidence_label"])

    previous_pressure_msl = float(history_df.sort_values("datetime").iloc[-1]["pressure_msl"])
    prepared_weather = engineer_weather_features(weather_df, previous_pressure_msl=previous_pressure_msl)
    fixed_lookup: dict[pd.Timestamp, float] = {}
    if fixed_future_pm25 is not None and not fixed_future_pm25.empty:
        fixed_lookup = {
            pd.Timestamp(row["datetime"]): float(row["forecast_pm25"])
            for _, row in fixed_future_pm25.iterrows()
        }

    pm25_history = history_df.sort_values("datetime")["pm25"].astype(float).tolist()
    rolling_window = list(pm25_history)
    forecast_rows: list[dict[str, object]] = []

    for row in prepared_weather.itertuples(index=False):
        current_timestamp = pd.Timestamp(row.datetime)
        known_pm25 = fixed_lookup.get(current_timestamp)

        if known_pm25 is None:
            feature_payload = {
                "temperature": float(row.temperature),
                "humidity": float(row.humidity),
                "dew_point": float(row.dew_point),
                "precipitation": float(row.precipitation),
                "rain": float(row.rain),
                "pressure_msl": float(row.pressure_msl),
                "surface_pressure": float(row.surface_pressure),
                "cloud_cover": float(row.cloud_cover),
                "wind_speed": float(row.wind_speed),
                "wind_direction": float(row.wind_direction),
                "wind_gusts": float(row.wind_gusts),
                "year": int(row.year),
                "month": int(row.month),
                "day": int(row.day),
                "hour": int(row.hour),
                "day_of_week": int(row.day_of_week),
                "is_weekend": int(row.is_weekend),
                "is_dry_season": int(row.is_dry_season),
                "pm25_lag1": float(rolling_window[-1]),
                "pm25_lag24": float(rolling_window[-24]),
                "pm25_lag168": float(rolling_window[-168]),
                "pm25_rolling_3h": float(np.mean(rolling_window[-3:])),
                "pm25_rolling_24h": float(np.mean(rolling_window[-24:])),
                "pm25_rolling_7d": float(np.mean(rolling_window[-168:])),
                "pm25_rolling_24h_std": float(np.std(rolling_window[-24:], ddof=0)),
                "wind_u": float(row.wind_u),
                "wind_v": float(row.wind_v),
                "temp_humidity": float(row.temp_humidity),
                "pressure_diff": float(row.pressure_diff),
                "is_raining": int(row.is_raining),
            }
            feature_frame = pd.DataFrame([feature_payload])
            known_pm25 = float(predict_local_model(feature_frame, model_bundle)[0])
            source_label = "Weather-driven local model forecast"
            confidence_label = "Medium confidence"
        else:
            source_label = "Open-Meteo air-quality forecast"
            confidence_label = "High confidence"

        rolling_window.append(float(known_pm25))
        forecast_rows.append(
            {
                "datetime": current_timestamp,
                "forecast_pm25": float(known_pm25),
                "source_label": source_label,
                "confidence_label": confidence_label,
                "actual_pm25": np.nan,
                "baseline_pred": np.nan,
            }
        )

    return pd.DataFrame(forecast_rows)
