from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from joblib import load


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from main.forecasting_core import (  # noqa: E402
    apply_dow_hour_profile,
    build_historical_forecast_frame,
    get_safety_recommendations,
    load_dataset,
    recursive_weather_forecast,
    standardize_forecast_frame,
)
from main.location_mining import process_projects, get_district_risk, ALL_DISTRICTS, URBAN_DISTRICTS


APP_TIMEZONE = "Asia/Bangkok"
PREDICTIONS_PATH = ROOT_DIR / "predictions.csv"
HISTORY_PATH = ROOT_DIR / "hanoi_aqi_ml_ready_fixed.csv"
MODEL_BUNDLE_PATH = ROOT_DIR / "model_bundle.joblib"
PROJECTS_PATH = ROOT_DIR / "area_projects_details.csv"

HANOI_LATITUDE = 21.0285
HANOI_LONGITUDE = 105.8542
PLANNER_HORIZON_DAYS = 30
LIVE_AIR_QUALITY_DAYS = 7
LIVE_WEATHER_DAYS = 16

PREDICTION_COLUMNS = {
    "datetime",
    "actual_pm25",
    "forecast_pm25",
    "baseline_pred",
    "source_label",
}
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_FIELD_MAP = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "dew_point_2m": "dew_point",
    "precipitation": "precipitation",
    "rain": "rain",
    "pressure_msl": "pressure_msl",
    "surface_pressure": "surface_pressure",
    "cloud_cover": "cloud_cover",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
    "wind_gusts_10m": "wind_gusts",
}

AQI_COLORS = {
    "Fresh": ("#d1fae5", "#065f46"),
    "Moderate": ("#fef9c3", "#854d0e"),
    "Sensitive groups should be careful": ("#ffedd5", "#9a3412"),
    "Unhealthy": ("#fee2e2", "#991b1b"),
    "Very unhealthy": ("#fce7f3", "#9d174d"),
}

RISK_COLORS = {
    "Low": ("#d1fae5", "#065f46"),
    "Medium": ("#fef9c3", "#854d0e"),
    "High": ("#fee2e2", "#991b1b"),
}


def configure_page() -> None:
    st.set_page_config(page_title="Hanoi Air Planner", page_icon="🌬️", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(210,244,234,0.88), transparent 28%),
                radial-gradient(circle at top left, rgba(226,238,255,0.92), transparent 30%),
                linear-gradient(180deg, #f7fbff 0%, #eef5f8 56%, #ecf7f1 100%);
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        section[data-testid="stSidebar"] {
            background: rgba(239,247,248,0.97);
            border-right: 1px solid #e2e8f0;
        }
        section[data-testid="stSidebar"] .stMarkdown h2 {
            color: #64748b;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.08em;
            margin-bottom: 4px;
            text-transform: uppercase;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #f1f5f9;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(22,56,71,0.05);
            padding: 16px 20px;
        }
        [data-testid="stMetricLabel"] {
            color: #64748b !important;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        [data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-size: 26px;
            font-weight: 700;
        }
        .hai-card {
            background: #ffffff;
            border: 1px solid #f1f5f9;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(22,56,71,0.06);
            margin-bottom: 16px;
            padding: 24px;
        }
        .hai-card-title {
            color: #64748b;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.07em;
            margin-bottom: 4px;
            text-transform: uppercase;
        }
        .hai-card-heading {
            color: #0f172a;
            font-size: 20px;
            font-weight: 700;
        }
        .hai-hero-value {
            color: #005c55;
            font-size: 56px;
            font-weight: 800;
            line-height: 1;
        }
        .hai-hero-unit {
            color: #64748b;
            font-size: 18px;
            font-weight: 500;
        }
        .hai-badge {
            border-radius: 999px;
            display: inline-block;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.06em;
            padding: 4px 12px;
            text-transform: uppercase;
        }
        .hai-highlight-row {
            display: grid;
            gap: 16px;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            margin-bottom: 20px;
        }
        .hai-highlight-item {
            background: #ffffff;
            border: 1px solid #f1f5f9;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(22,56,71,0.04);
            min-width: 0;
            padding: 14px 18px;
        }
        .hai-highlight-label {
            color: #94a3b8;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }
        .hai-highlight-value {
            color: #005c55;
            font-size: 22px;
            font-weight: 800;
            margin: 2px 0;
        }
        .hai-highlight-sub {
            color: #64748b;
            font-family: "Segoe UI Mono", Consolas, monospace;
            font-size: 12px;
            font-weight: 500;
        }
        .hai-advice-item {
            align-items: flex-start;
            border-bottom: 1px solid #f8fafc;
            display: flex;
            gap: 14px;
            padding: 14px 0;
        }
        .hai-advice-icon {
            align-items: center;
            border-radius: 999px;
            display: flex;
            flex-shrink: 0;
            font-size: 20px;
            height: 42px;
            justify-content: center;
            width: 42px;
        }
        .hai-advice-title {
            color: #0f172a;
            font-size: 14px;
            font-weight: 700;
        }
        .hai-advice-body {
            color: #475569;
            font-size: 13px;
            line-height: 1.55;
            margin-top: 2px;
        }
        .hai-page-title {
            color: #005c55;
            font-size: 32px;
            font-weight: 800;
            margin-bottom: 2px;
        }
        .hai-page-subtitle {
            color: #475569;
            font-size: 16px;
            margin-bottom: 28px;
        }
        .hai-page-subtitle b {
            color: #0f172a;
        }
        [data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
        }
        thead tr th {
            background: #f8fafc !important;
            color: #64748b !important;
            font-size: 11px !important;
            font-weight: 700 !important;
            letter-spacing: 0.05em !important;
            text-transform: uppercase !important;
        }
        button[data-baseweb="tab"] {
            font-size: 12px !important;
            font-weight: 700 !important;
            letter-spacing: 0.04em !important;
        }
        .stDownloadButton > button {
            background: #005c55 !important;
            border: none !important;
            border-radius: 8px !important;
            color: #ffffff !important;
            font-size: 12px !important;
            font-weight: 700 !important;
            letter-spacing: 0.05em !important;
            padding: 10px 20px !important;
            text-transform: uppercase !important;
        }
        .stDownloadButton > button:hover {
            background: #00403b !important;
        }
        hr {
            border-color: #e2e8f0 !important;
            margin: 28px 0 !important;
        }
        [data-testid="stAlert"] {
            border-radius: 8px !important;
        }
        .js-plotly-plot {
            border-radius: 8px;
        }
        @media (max-width: 760px) {
            .hai-highlight-row {
                grid-template-columns: 1fr;
            }
            .hai-hero-value {
                font-size: 42px;
            }
        }
        .block-container {
            padding-top: 2.75rem;
            max-width: 1180px;
        }
        .hai-top-nav {
            align-items: center;
            background: rgba(255,255,255,0.86);
            border: 1px solid rgba(120,246,237,0.45);
            border-radius: 24px;
            box-shadow: 0 20px 40px -18px rgba(46,186,178,0.25);
            display: flex;
            justify-content: space-between;
            margin-bottom: 24px;
            padding: 14px 22px;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        .hai-brand {
            color: #006a65;
            font-size: 24px;
            font-weight: 900;
            letter-spacing: -0.02em;
        }
        .hai-nav-links {
            display: flex;
            gap: 22px;
            color: #64748b;
            font-size: 14px;
            font-weight: 700;
        }
        .hai-nav-links span:first-child {
            color: #006a65;
            border-bottom: 2px solid #2ebab2;
        }
        .hai-card {
            background: rgba(255,255,255,0.88);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.9);
            border-radius: 18px;
            box-shadow: 0 20px 40px -18px rgba(46,186,178,0.22);
        }
        .hai-page-title {
            color: #071e27;
            font-size: 34px;
            font-weight: 900;
            letter-spacing: -0.01em;
            line-height: 1.18;
            padding-top: 4px;
        }
        .hai-page-subtitle {
            color: #3c4948;
            font-size: 18px;
        }
        .hai-hero-card {
            background:
                radial-gradient(circle at top right, rgba(253,192,3,0.22), transparent 34%),
                linear-gradient(135deg, rgba(255,255,255,0.96), rgba(230,246,255,0.88));
            overflow: hidden;
            position: relative;
        }
        .hai-hero-value {
            color: #071e27;
            font-size: 72px;
            font-weight: 900;
            line-height: 0.95;
        }
        .hai-hero-unit {
            color: #3c4948;
            font-size: 22px;
            font-weight: 800;
        }
        .hai-badge {
            border-radius: 999px;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.55);
        }
        .hai-confidence-pill {
            align-items: center;
            background: rgba(46,186,178,0.18);
            border-radius: 999px;
            color: #004541;
            display: inline-flex;
            font-size: 12px;
            font-weight: 800;
            gap: 6px;
            padding: 7px 12px;
        }
        .hai-metric-card {
            align-items: center;
            border-radius: 18px;
            display: flex;
            gap: 14px;
            min-height: 96px;
            padding: 18px;
        }
        .hai-metric-icon {
            align-items: center;
            border-radius: 16px;
            color: #ffffff;
            display: flex;
            font-size: 24px;
            height: 48px;
            justify-content: center;
            width: 48px;
        }
        .hai-action-grid {
            display: grid;
            gap: 18px;
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }
        .hai-recommend-banner {
            background:
                radial-gradient(circle at top right, rgba(255,133,95,0.2), transparent 30%),
                linear-gradient(135deg, rgba(255,255,255,0.96), rgba(224,255,245,0.9));
            border: 1px solid rgba(120,246,237,0.5);
            border-radius: 18px;
            box-shadow: 0 20px 40px -22px rgba(0,80,76,0.35);
            margin: 18px 0;
            padding: 22px;
        }
        .hai-recommend-banner h2 {
            color: #071e27;
            font-size: 24px;
            font-weight: 900;
            margin: 0 0 8px;
        }
        .hai-recommend-banner p {
            color: #3c4948;
            font-size: 15px;
            line-height: 1.55;
            margin: 0;
        }
        .hai-recommend-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 16px;
        }
        .hai-recommend-chip {
            background: #ffffff;
            border: 1px solid #cfe6f2;
            border-radius: 999px;
            color: #006a65;
            font-size: 13px;
            font-weight: 800;
            padding: 8px 12px;
        }
        .hai-live-summary {
            align-items: center;
            background:
                radial-gradient(circle at top right, rgba(255,133,95,0.16), transparent 30%),
                linear-gradient(135deg, rgba(255,255,255,0.94), rgba(230,246,255,0.9));
            border: 1px solid rgba(120,246,237,0.45);
            border-radius: 18px;
            box-shadow: 0 16px 34px -24px rgba(0,80,76,0.32);
            display: flex;
            gap: 18px;
            justify-content: space-between;
            margin: 0 0 20px;
            padding: 16px 18px;
        }
        .hai-live-summary strong {
            color: #071e27;
            display: block;
            font-size: 18px;
            font-weight: 900;
            line-height: 1.25;
        }
        .hai-live-summary span {
            color: #3c4948;
            font-size: 14px;
            line-height: 1.45;
        }
        .hai-live-pill {
            background: #fdc003;
            border-radius: 999px;
            color: #6c5000;
            flex: 0 0 auto;
            font-size: 13px;
            font-weight: 900;
            padding: 8px 12px;
        }
        .hai-action-card {
            background: #ffffff;
            border: 1px solid rgba(207,230,242,0.8);
            border-radius: 18px;
            box-shadow: 0 18px 34px -26px rgba(0,80,76,0.35);
            padding: 20px;
        }
        .hai-action-card h3 {
            color: #071e27;
            font-size: 18px;
            font-weight: 800;
            margin: 10px 0 6px;
        }
        .hai-action-card p {
            color: #3c4948;
            font-size: 14px;
            line-height: 1.55;
            margin: 0;
        }
        .hai-section-tabs {
            align-items: center;
            border-bottom: 1px solid #cfe6f2;
            display: flex;
            gap: 28px;
            margin: 8px 0 18px;
            padding-bottom: 12px;
        }
        .hai-section-tabs span {
            color: #6c7a78;
            font-size: 20px;
            font-weight: 800;
        }
        .hai-section-tabs span:first-child {
            color: #006a65;
            border-bottom: 4px solid #006a65;
            padding-bottom: 12px;
        }
        .hai-progress-row {
            margin: 15px 0;
        }
        .hai-progress-head {
            display: flex;
            font-size: 14px;
            font-weight: 800;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        .hai-progress-track {
            background: #cfe6f2;
            border-radius: 999px;
            height: 12px;
            overflow: hidden;
        }
        .hai-progress-fill {
            border-radius: 999px;
            height: 100%;
        }
        .hai-scenic-card {
            background:
                linear-gradient(180deg, rgba(0,69,65,0.05), rgba(0,69,65,0.75)),
                radial-gradient(circle at 25% 18%, rgba(255,223,158,0.95), transparent 15%),
                linear-gradient(135deg, #78f6ed 0%, #dff4ff 42%, #59dad1 100%);
            border-radius: 18px;
            min-height: 260px;
            overflow: hidden;
            padding: 24px;
            position: relative;
        }
        .hai-scenic-card:before {
            background:
                radial-gradient(ellipse at 20% 100%, #006a65 0%, #006a65 16%, transparent 17%),
                radial-gradient(ellipse at 52% 100%, #2ebab2 0%, #2ebab2 19%, transparent 20%),
                radial-gradient(ellipse at 85% 100%, #00504c 0%, #00504c 14%, transparent 15%);
            bottom: 0;
            content: "";
            height: 120px;
            left: 0;
            position: absolute;
            right: 0;
        }
        .hai-scenic-text {
            bottom: 22px;
            color: #ffffff;
            left: 24px;
            position: absolute;
            right: 24px;
            z-index: 1;
        }
        .hai-scenic-text strong {
            display: block;
            font-size: 24px;
            font-weight: 900;
        }
        section[data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top, rgba(120,246,237,0.28), transparent 35%),
                rgba(248,252,255,0.97);
        }
        .stDownloadButton > button {
            border-radius: 999px !important;
        }
        @media (max-width: 900px) {
            .hai-nav-links {
                display: none;
            }
            .hai-action-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def current_local_date() -> pd.Timestamp:
    return pd.Timestamp.now(tz=APP_TIMEZONE).normalize().tz_localize(None)


def current_local_datetime() -> pd.Timestamp:
    return pd.Timestamp.now(tz=APP_TIMEZONE).tz_localize(None)


def safe_mtime(path: Path) -> float:
    return path.stat().st_mtime


def format_date_label(value: pd.Timestamp | object) -> str:
    return pd.Timestamp(value).strftime("%A, %d %B %Y")


def build_target_timestamp(selected_date: pd.Timestamp, selected_hour: str) -> pd.Timestamp:
    return pd.Timestamp(selected_date).normalize() + pd.Timedelta(hours=int(selected_hour[:2]))


@st.cache_data(show_spinner=False)
def load_predictions(csv_path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    df = pd.read_csv(csv_path, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    missing = PREDICTION_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"predictions.csv is missing required columns: {', '.join(sorted(missing))}")
    return df


@st.cache_data(show_spinner=False)
def load_history(csv_path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return load_dataset(Path(csv_path))


@st.cache_resource(show_spinner=False)
def load_model_bundle(bundle_path: str, modified_time: float) -> dict[str, object]:
    del modified_time
    return load(bundle_path)


def get_air_quality_band(value: float) -> dict[str, str]:
    if value <= 12:
        return {"label": "Fresh", "advice": "Great time for walking, light exercise, or a longer commute."}
    if value <= 35.4:
        return {"label": "Moderate", "advice": "Most people will be fine, but sensitive users may want a mask."}
    if value <= 55.4:
        return {"label": "Sensitive groups should be careful", "advice": "Reduce outdoor time if you are sensitive to pollution."}
    if value <= 150.4:
        return {"label": "Unhealthy", "advice": "Keep outdoor time short and wear a mask when commuting."}
    return {"label": "Very unhealthy", "advice": "Try to stay indoors unless necessary."}


def request_json(base_url: str, params: dict[str, object]) -> dict[str, object]:
    request = Request(
        url=f"{base_url}?{urlencode(params)}",
        headers={"User-Agent": "HanoiAirPlanner/2.0"},
    )
    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} from Open-Meteo") from exc
    except URLError as exc:
        raise RuntimeError(f"Open-Meteo request failed: {exc.reason}") from exc


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_open_meteo_air_quality(today_marker: str, forecast_days: int) -> pd.DataFrame:
    del today_marker
    payload = request_json(
        AIR_QUALITY_URL,
        {
            "latitude": HANOI_LATITUDE,
            "longitude": HANOI_LONGITUDE,
            "timezone": APP_TIMEZONE,
            "hourly": "pm2_5",
            "forecast_days": forecast_days,
        },
    )
    hourly = payload.get("hourly", {})
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(hourly.get("time", [])),
            "forecast_pm25": pd.to_numeric(hourly.get("pm2_5", []), errors="coerce"),
        }
    ).dropna(subset=["forecast_pm25"])
    if df.empty:
        raise RuntimeError("Open-Meteo air-quality response did not include hourly pm2_5 data.")
    return standardize_forecast_frame(
        df[["datetime", "forecast_pm25"]],
        source_label="Open-Meteo air-quality forecast",
        confidence_label="High confidence",
    )


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_open_meteo_weather(today_marker: str, forecast_days: int) -> pd.DataFrame:
    del today_marker
    payload = request_json(
        WEATHER_URL,
        {
            "latitude": HANOI_LATITUDE,
            "longitude": HANOI_LONGITUDE,
            "timezone": APP_TIMEZONE,
            "hourly": ",".join(WEATHER_FIELD_MAP.keys()),
            "forecast_days": forecast_days,
        },
    )
    hourly = payload.get("hourly", {})
    if "time" not in hourly:
        raise RuntimeError("Open-Meteo weather response did not include hourly time data.")
    weather_df = pd.DataFrame({"datetime": pd.to_datetime(hourly["time"])})
    for api_name, local_name in WEATHER_FIELD_MAP.items():
        if api_name not in hourly:
            raise RuntimeError(f"Open-Meteo weather response is missing '{api_name}'.")
        weather_df[local_name] = (
            pd.to_numeric(pd.Series(hourly[api_name]), errors="coerce").fillna(0.0).to_numpy()
        )
    return weather_df.fillna(0.0)


def overlay_forecasts(base_df: pd.DataFrame, overlay_df: pd.DataFrame | None) -> pd.DataFrame:
    if overlay_df is None or overlay_df.empty:
        return base_df
    merged = base_df.merge(
        overlay_df.rename(
            columns={
                "forecast_pm25": "overlay_forecast_pm25",
                "source_label": "overlay_source_label",
                "confidence_label": "overlay_confidence_label",
                "actual_pm25": "overlay_actual_pm25",
                "baseline_pred": "overlay_baseline_pred",
            }
        ),
        on="datetime",
        how="left",
    )
    mask = merged["overlay_forecast_pm25"].notna()
    for base_col, overlay_col in {
        "forecast_pm25": "overlay_forecast_pm25",
        "source_label": "overlay_source_label",
        "confidence_label": "overlay_confidence_label",
        "actual_pm25": "overlay_actual_pm25",
        "baseline_pred": "overlay_baseline_pred",
    }.items():
        merged.loc[mask, base_col] = merged.loc[mask, overlay_col]
    return merged[["datetime", "forecast_pm25", "source_label", "confidence_label", "actual_pm25", "baseline_pred"]]


def build_planner_forecasts(
    history_df: pd.DataFrame,
    model_bundle: dict[str, object],
    today: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str]]:
    planner_hours = pd.date_range(today, periods=PLANNER_HORIZON_DAYS * 24, freq="h")
    fallback_df = standardize_forecast_frame(
        apply_dow_hour_profile(planner_hours, model_bundle["planner_profile"]),
        source_label="Offline day-of-week/hour fallback",
        confidence_label="Low confidence",
    )

    warnings: list[str] = []
    today_key = today.strftime("%Y-%m-%d")
    air_df = None
    weather_df = None

    try:
        air_df = fetch_open_meteo_air_quality(today_key, LIVE_AIR_QUALITY_DAYS)
    except Exception as exc:
        warnings.append(f"Live AQI API unavailable, so the near-term planner may use model fallback. Detail: {exc}")

    try:
        weather_df = fetch_open_meteo_weather(today_key, LIVE_WEATHER_DAYS)
    except Exception as exc:
        warnings.append(f"Live weather API unavailable, so mid-range planner rows may use offline fallback. Detail: {exc}")

    overlay_df = None
    pure_model_df = None
    if weather_df is not None and not weather_df.empty:
        # Main forecast (hybrid: API first 7 days, model after)
        overlay_df = recursive_weather_forecast(
            weather_df=weather_df,
            history_df=history_df,
            model_bundle=model_bundle,
            fixed_future_pm25=None if air_df is None else air_df[["datetime", "forecast_pm25"]],
        )
        # History-based model prediction (Pure model for comparison)
        pure_model_df = recursive_weather_forecast(
            weather_df=weather_df,
            history_df=history_df,
            model_bundle=model_bundle,
            fixed_future_pm25=None,
        ).rename(columns={"forecast_pm25": "history_model_pm25"})

    final_df = overlay_forecasts(fallback_df, overlay_df)
    if pure_model_df is not None:
        final_df = final_df.merge(pure_model_df[["datetime", "history_model_pm25"]], on="datetime", how="left")
    else:
        final_df["history_model_pm25"] = np.nan

    return final_df.sort_values("datetime").reset_index(drop=True), warnings


def slice_day(df: pd.DataFrame, selected_date: pd.Timestamp) -> pd.DataFrame:
    mask = df["datetime"].dt.date == pd.Timestamp(selected_date).date()
    return df.loc[mask].sort_values("datetime").reset_index(drop=True)


def lookup_target(day_df: pd.DataFrame, target_timestamp: pd.Timestamp) -> pd.Series:
    exact = day_df.loc[day_df["datetime"] == target_timestamp]
    if not exact.empty:
        return exact.iloc[0]
    return day_df.loc[(day_df["datetime"] - target_timestamp).abs().idxmin()]


def build_table(day_df: pd.DataFrame) -> pd.DataFrame:
    table_df = day_df[["datetime", "forecast_pm25", "confidence_label", "source_label"]].copy()
    table_df["Air quality"] = table_df["forecast_pm25"].map(lambda value: get_air_quality_band(float(value))["label"])
    table_df["Suggestion"] = table_df["forecast_pm25"].map(lambda value: get_air_quality_band(float(value))["advice"])
    table_df["Time"] = table_df["datetime"].dt.strftime("%H:%M")
    return table_df[["Time", "forecast_pm25", "Air quality", "confidence_label", "source_label", "Suggestion"]].rename(
        columns={
            "forecast_pm25": "Predicted PM2.5",
            "confidence_label": "Confidence",
            "source_label": "Forecast source",
        }
    )


def build_chart(
    day_df: pd.DataFrame,
    target_timestamp: pd.Timestamp,
    show_baseline: bool,
    show_history_model: bool = False,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=day_df["datetime"],
            y=day_df["forecast_pm25"],
            fill="tozeroy",
            fillcolor="rgba(0,92,85,0.07)",
            mode="lines+markers",
            name="Main forecast",
            line={"color": "#005c55", "width": 3},
            marker={"size": 6, "color": "#005c55"},
        )
    )
    if show_history_model and "history_model_pm25" in day_df.columns and day_df["history_model_pm25"].notna().any():
        figure.add_trace(
            go.Scatter(
                x=day_df["datetime"],
                y=day_df["history_model_pm25"],
                mode="lines",
                name="History-based model",
                line={"color": "#6366f1", "dash": "dot", "width": 2.5},
            )
        )
    if day_df["actual_pm25"].notna().any():
        figure.add_trace(
            go.Scatter(
                x=day_df["datetime"],
                y=day_df["actual_pm25"],
                mode="lines+markers",
                name="Actual PM2.5",
                line={"color": "#163847", "width": 3},
                marker={"size": 5},
            )
        )
    if show_baseline and day_df["baseline_pred"].notna().any():
        figure.add_trace(
            go.Scatter(
                x=day_df["datetime"],
                y=day_df["baseline_pred"],
                mode="lines",
                name="Baseline",
                line={"color": "#f97316", "dash": "dash", "width": 2},
            )
        )
    target_row = lookup_target(day_df, target_timestamp)
    figure.add_vline(x=target_timestamp, line_dash="dot", line_color="#2563eb", line_width=2)
    figure.add_annotation(
        x=target_timestamp,
        y=float(target_row["forecast_pm25"]),
        text=f"<b>{target_timestamp:%H:%M}</b> - {float(target_row['forecast_pm25']):.1f} ug/m3",
        showarrow=True,
        arrowhead=2,
        ay=-50,
        bgcolor="#ffffff",
        bordercolor="#e2e8f0",
        borderpad=6,
        borderwidth=1,
        font={"size": 12, "color": "#0f172a"},
    )
    figure.update_layout(
        height=420,
        margin={"l": 12, "r": 12, "t": 16, "b": 12},
        xaxis_title="Hour",
        yaxis_title="PM2.5 (ug/m3)",
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        paper_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": -0.18, "x": 0, "font": {"size": 12}},
        xaxis={"gridcolor": "#f1f5f9", "linecolor": "#e2e8f0"},
        yaxis={"gridcolor": "#f1f5f9", "linecolor": "#e2e8f0"},
        font={"family": "Inter, sans-serif"},
    )
    return figure


def render_sidebar(
    historical_df: pd.DataFrame,
    model_bundle: dict[str, object],
    today: pd.Timestamp,
) -> tuple[str, pd.Timestamp, str, bool]:
    with st.sidebar:
        st.markdown("## Planner Controls 🌬️")
        st.caption("Vibrant Breeze System")
        mode = st.radio("Mode", ["Historical 📜", "Upcoming planner 📅"], index=1)
        mode_key = "Historical" if mode.startswith("Historical") else "Upcoming planner"

        if mode_key == "Historical":
            date_options = sorted(historical_df["datetime"].dt.date.unique())
            selected_date = st.selectbox(
                "Date",
                date_options,
                index=date_options.index(pd.Timestamp(model_bundle["local_history_end"]).date()),
                format_func=format_date_label,
            )
            day_df = slice_day(historical_df, pd.Timestamp(selected_date))
            hour_options = day_df["datetime"].dt.strftime("%H:%M").tolist()
            default_hour = "08:00" if "08:00" in hour_options else hour_options[0]
            selected_hour = st.selectbox("Hour", hour_options, index=hour_options.index(default_hour))
            show_baseline = st.toggle("Show comparison lines", value=True)
            show_history_model = False
            st.info(
                f"Local dataset: {pd.Timestamp(model_bundle['local_history_start']):%d %b %Y} to "
                f"{pd.Timestamp(model_bundle['local_history_end']):%d %b %Y}. "
                f"Saved holdout: {pd.Timestamp(model_bundle['test_window_start']):%d %b %Y} to "
                f"{pd.Timestamp(model_bundle['test_window_end']):%d %b %Y}."
            )
        else:
            date_options = [today.date() + pd.Timedelta(days=offset) for offset in range(PLANNER_HORIZON_DAYS)]
            selected_date = st.selectbox("Date", date_options, index=0, format_func=format_date_label)
            current_hour = current_local_datetime().hour
            selected_hour = st.selectbox("Hour", [f"{hour:02d}:00" for hour in range(24)], index=current_hour)
            show_baseline = False
            show_history_model = st.toggle("Show history-based model", value=True)
            st.info(
                f"Direct AQI: {today:%d %b %Y} to "
                f"{(today + pd.Timedelta(days=LIVE_AIR_QUALITY_DAYS - 1)):%d %b %Y}. "
                f"Weather model: {(today + pd.Timedelta(days=LIVE_AIR_QUALITY_DAYS)):%d %b %Y} to "
                f"{(today + pd.Timedelta(days=LIVE_WEATHER_DAYS - 1)):%d %b %Y}. "
                f"Offline fallback through day {PLANNER_HORIZON_DAYS}."
            )
            st.caption(
                f"Local file ends {pd.Timestamp(model_bundle['local_history_end']):%d %b %Y}; "
                "dates in the gap before today are intentionally hidden."
            )

    return mode_key, pd.Timestamp(selected_date), selected_hour, show_baseline, show_history_model


def render_district_selector() -> str:
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Location Settings 📍")
    default_district = "Thanh Xuân" if "Thanh Xuân" in ALL_DISTRICTS else ALL_DISTRICTS[0]
    district = st.sidebar.selectbox(
        "Trip Destination (District)",
        ALL_DISTRICTS,
        index=ALL_DISTRICTS.index(default_district),
    )
    return district


def render_mining_dashboard(density_df: pd.DataFrame) -> None:
    st.markdown("---")
    st.header("🔍 Data Mining Insights: Construction Patterns")
    st.write("We mined unstructured project names to discover where development is most active in Hanoi (2025-2026).")
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("Project Density by District")
        # Simple bar chart
        chart_df = density_df.sort_values("project_count", ascending=False).head(10)
        st.bar_chart(chart_df.set_index("district")["project_count"])
        
    with col2:
        st.subheader("Urban vs Rural Distribution")
        urban_count = density_df[density_df["district"].isin(URBAN_DISTRICTS)]["project_count"].sum()
        rural_count = density_df[~density_df["district"].isin(URBAN_DISTRICTS)]["project_count"].sum()
        
        st.write(f"Total Active Projects Mined: **{density_df['project_count'].sum()}**")
        st.write(f"🏢 Urban Districts: **{urban_count}** projects")
        st.write(f"🌳 Rural/Suburban: **{rural_count}** projects")
        
        st.info("💡 Insight: High project density often correlates with higher PM10/PM2.5 'dust spikes' that models may miss without location context.")

def _badge(label: str, color_map: dict[str, tuple[str, str]]) -> str:
    bg, fg = color_map.get(label, ("#e2e8f0", "#475569"))
    return f'<span class="hai-badge" style="background:{bg};color:{fg}">{label}</span>'


def _card_open(title: str, heading: str = "") -> str:
    heading_html = f'<div class="hai-card-heading">{heading}</div>' if heading else ""
    return f'<div class="hai-card"><div class="hai-card-title">{title}</div>{heading_html}'


def _card_close() -> str:
    return "</div>"


def render_top_nav() -> None:
    st.markdown(
        """
        <div class="hai-top-nav">
            <div class="hai-brand">Hanoi Air 🌬️</div>
            <div class="hai-nav-links">
                <span>Dashboard</span>
                <span>Map</span>
                <span>Insights</span>
                <span>Community</span>
            </div>
            <div style="color:#64748b;font-size:20px;">🔔 👤</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_forecast_snapshot(
    target_row: pd.Series,
    selected_timestamp: pd.Timestamp,
    selected_district: str,
    band: dict[str, str],
    district_risk: dict[str, object],
) -> None:
    pm25 = float(target_row["forecast_pm25"])
    risk_level = str(district_risk["level"])
    risk_bg, risk_fg = RISK_COLORS.get(risk_level, ("#e2e8f0", "#475569"))
    aqi_bg, aqi_fg = AQI_COLORS.get(band["label"], ("#e2e8f0", "#475569"))

    st.markdown(
        f"""
        <div class="hai-card hai-hero-card">
        <div style="display:flex;justify-content:space-between;gap:18px;align-items:flex-start;flex-wrap:wrap;">
            <div>
                <span class="hai-badge" style="background:{aqi_bg};color:{aqi_fg};">{band["label"]} 😐</span>
                <div style="margin-top:18px; display:flex; align-items:baseline; gap:12px;">
                    <span class="hai-hero-value">{pm25:.1f}</span>
                    <span class="hai-hero-unit">PM2.5</span>
                </div>
                <div style="margin-top:10px;color:#3c4948;font-weight:700;">📍 {selected_district} · {selected_timestamp:%H:%M}</div>
            </div>
            <div style="text-align:right;">
                <div class="hai-confidence-pill">✅ {target_row.get("confidence_label", "Forecast ready")} ✨</div>
                <div style="margin-top:10px;color:#6c7a78;font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.08em;">
                    Source: {target_row.get("source_label", "-")}
                </div>
            </div>
        </div>
        <div style="margin-top:24px; padding-top:18px; border-top:1px solid rgba(207,230,242,0.9);
                    display:flex; gap:24px; flex-wrap:wrap;">
            <div>
                <div style="font-size:10px;font-weight:700;letter-spacing:.06em;
                            text-transform:uppercase;color:#94a3b8;">Construction Risk</div>
                <div style="margin-top:4px;">
                    <span class="hai-badge" style="background:{risk_bg};color:{risk_fg}">
                        {risk_level} Risk
                    </span>
                </div>
            </div>
            <div>
                <div style="font-size:10px;font-weight:700;letter-spacing:.06em;
                            text-transform:uppercase;color:#94a3b8;">Daily Advice</div>
                <div style="margin-top:6px;font-size:13px;font-weight:600;color:#0f172a;">
                    {band["advice"]}
                </div>
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_advice_card(safety_recs: dict[str, str]) -> None:
    meta = {
        "Exercise": ("fitness_center", "Exercise 🏃", "Morning plan 👟", "#e0fff5"),
        "Commuting": ("masks", "Commuting 🚲", "Route choice 😷", "#e6f6ff"),
        "Hanging Out": ("self_improvement", "Social ☕", "Outdoor timing 🧘", "#fff4d6"),
    }
    cards = []
    for key, advice in safety_recs.items():
        _, tab_label, title, bg = meta.get(key, ("eco", key, key, "#ffffff"))
        cards.append(
            f"""
            <div class="hai-action-card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                    <span style="background:{bg};color:#006a65;border-radius:999px;padding:9px 12px;font-weight:900;">
                        {tab_label}
                    </span>
                    <span style="font-size:11px;font-weight:900;color:#006a65;background:rgba(0,106,101,0.1);border-radius:999px;padding:5px 8px;">
                        RECOMMENDED
                    </span>
                </div>
                <h3>{title}</h3>
                <p>{advice}</p>
            </div>
            """
        )
    st.markdown(
        f"""
        <div class="hai-section-tabs">
            <span>Exercise 🏃</span>
            <span>Commuting 🚲</span>
            <span>Social ☕</span>
        </div>
        <div class="hai-action-grid">
            {''.join(cards)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_banner(
    safety_recs: dict[str, str],
    band: dict[str, str],
    selected_district: str,
    selected_timestamp: pd.Timestamp,
) -> None:
    primary_recommendation = safety_recs.get("Exercise") or next(iter(safety_recs.values()))
    st.markdown(
        f"""
        <div class="hai-recommend-banner">
            <div style="display:flex;justify-content:space-between;gap:18px;align-items:flex-start;flex-wrap:wrap;">
                <div>
                    <div class="hai-card-title">Recommended For Your Selected Hour ✅</div>
                    <h2>{selected_timestamp:%H:%M} in {selected_district}</h2>
                    <p>{primary_recommendation}</p>
                </div>
                <span class="hai-badge" style="background:#fdc003;color:#6c5000;">{band["label"]}</span>
            </div>
            <div class="hai-recommend-chip-row">
                <span class="hai-recommend-chip">🏃 Exercise: {safety_recs.get("Exercise", "Check conditions")}</span>
                <span class="hai-recommend-chip">🚲 Commute: {safety_recs.get("Commuting", "Check conditions")}</span>
                <span class="hai-recommend-chip">☕ Social: {safety_recs.get("Hanging Out", "Check conditions")}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_highlight_row(day_df: pd.DataFrame) -> None:
    ranked = day_df.sort_values("forecast_pm25").reset_index(drop=True)
    cleanest = ranked.iloc[0]
    worst = ranked.iloc[-1]
    avg = day_df["forecast_pm25"].mean()
    clean_band = get_air_quality_band(float(cleanest["forecast_pm25"]))
    worst_band = get_air_quality_band(float(worst["forecast_pm25"]))
    clean_bg, clean_fg = AQI_COLORS.get(clean_band["label"], ("#e2e8f0", "#475569"))
    worst_bg, worst_fg = AQI_COLORS.get(worst_band["label"], ("#e2e8f0", "#475569"))

    st.markdown(
        f"""
        <div class="hai-highlight-row">
            <div class="hai-metric-card" style="background:#e6f6ff;border:1px solid #ffffff;">
                <div class="hai-metric-icon" style="background:#2ebab2;">🌿</div>
                <div>
                    <div class="hai-highlight-label">Cleanest Hour 🌿</div>
                    <div class="hai-highlight-value">{cleanest["datetime"].strftime("%H:%M")}</div>
                    <div class="hai-highlight-sub">{float(cleanest["forecast_pm25"]):.1f} ug/m3 · {clean_band["label"]}</div>
                </div>
            </div>
            <div class="hai-metric-card" style="background:#ffdad6;">
                <div class="hai-metric-icon" style="background:#ba1a1a;">⚠️</div>
                <div>
                    <div class="hai-highlight-label" style="color:#93000a;">Most Polluted ⚠️</div>
                    <div class="hai-highlight-value" style="color:#991b1b">{worst["datetime"].strftime("%H:%M")}</div>
                    <div class="hai-highlight-sub">{float(worst["forecast_pm25"]):.1f} ug/m3 · {worst_band["label"]}</div>
                </div>
            </div>
            <div class="hai-metric-card" style="background:#cfe6f2;">
                <div class="hai-metric-icon" style="background:#3c4948;">📊</div>
                <div>
                    <div class="hai-highlight-label">Day Average 📊</div>
                    <div class="hai-highlight-value">{avg:.1f} ug/m3</div>
                    <div class="hai-highlight-sub">std dev {day_df["forecast_pm25"].std():.1f}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_forecast_engines(model_bundle: dict[str, object]) -> None:
    metric_rows = [
        {"Model": "Local linear model", "Use case": "Historical dataset rows", "RMSE": round(model_bundle["metrics"]["local_model"]["RMSE"], 3), "MAE": round(model_bundle["metrics"]["local_model"]["MAE"], 3), "R2": f"{model_bundle['metrics']['local_model']['R2']:.4f}", "Status": "Optimal"},
        {"Model": "Lag-1 baseline", "Use case": "Reference benchmark", "RMSE": round(model_bundle["metrics"]["baseline"]["RMSE"], 3), "MAE": round(model_bundle["metrics"]["baseline"]["MAE"], 3), "R2": f"{model_bundle['metrics']['baseline']['R2']:.4f}", "Status": "Standby"},
        {"Model": "Day-of-week/hour fallback", "Use case": "Planner days 17-30 and offline", "RMSE": round(model_bundle["metrics"]["fallback"]["RMSE"], 3), "MAE": round(model_bundle["metrics"]["fallback"]["MAE"], 3), "R2": f"{model_bundle['metrics']['fallback']['R2']:.4f}", "Status": "Standby"},
        {"Model": "Open-Meteo live AQI", "Use case": "Planner near term", "RMSE": "-", "MAE": "-", "R2": "-", "Status": "Active"},
    ]
    st.markdown(_card_open("Forecast Engines", "Active Models and Performance"), unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
    st.markdown(_card_close(), unsafe_allow_html=True)


def render_mining_dashboard(density_df: pd.DataFrame, selected_district: str) -> None:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(_card_open("Construction Impact 🏗️", "Construction Site Distribution"), unsafe_allow_html=True)
    st.write("Unstructured project names mined to reveal where development is most active across Hanoi (2025-2026).")

    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.markdown("**Project Density by District 📍** (Top 10)")
        chart_df = density_df.sort_values("project_count", ascending=False).head(10)
        fig = go.Figure(
            go.Bar(
                x=chart_df["district"],
                y=chart_df["project_count"],
                marker_color=["#ff855f" if d == selected_district else "#2ebab2" for d in chart_df["district"]],
            )
        )
        fig.update_layout(
            height=260,
            margin={"l": 0, "r": 0, "t": 8, "b": 0},
            plot_bgcolor="#ffffff",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis={"gridcolor": "#f1f5f9"},
            yaxis={"gridcolor": "#f1f5f9", "title": "Projects"},
            font={"family": "Inter, sans-serif", "size": 12},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        urban_count = density_df[density_df["district"].isin(URBAN_DISTRICTS)]["project_count"].sum()
        rural_count = density_df[~density_df["district"].isin(URBAN_DISTRICTS)]["project_count"].sum()
        total = density_df["project_count"].sum()
        urban_pct = 0 if total == 0 else round((urban_count / total) * 100)
        rural_pct = 0 if total == 0 else round((rural_count / total) * 100)
        selected_count = int(
            density_df.loc[density_df["district"] == selected_district, "project_count"].sum()
        )
        selected_pct = 0 if total == 0 else min(100, round((selected_count / max(total, 1)) * 100))
        st.markdown(
            f"""
            <div>
                <div class="hai-progress-row">
                    <div class="hai-progress-head"><span>Urban Central 🏢</span><span>{urban_pct}%</span></div>
                    <div class="hai-progress-track"><div class="hai-progress-fill" style="width:{urban_pct}%;background:#ff855f;"></div></div>
                </div>
                <div class="hai-progress-row">
                    <div class="hai-progress-head"><span>Suburban 🏡</span><span>{rural_pct}%</span></div>
                    <div class="hai-progress-track"><div class="hai-progress-fill" style="width:{rural_pct}%;background:#2ebab2;"></div></div>
                </div>
                <div class="hai-progress-row">
                    <div class="hai-progress-head"><span>{selected_district} 📌</span><span>{selected_count} projects</span></div>
                    <div class="hai-progress-track"><div class="hai-progress-fill" style="width:{max(selected_pct, 6)}%;background:#fdc003;"></div></div>
                </div>
                <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:18px;">
                    <div style="background:#e6f6ff;border-radius:14px;padding:12px;">
                        <div style="font-size:11px;font-weight:800;color:#6c7a78;">Total</div>
                        <div style="font-size:22px;font-weight:900;color:#071e27;">{total}</div>
                    </div>
                    <div style="background:#f0fdf4;border-radius:14px;padding:12px;">
                        <div style="font-size:11px;font-weight:800;color:#6c7a78;">Urban</div>
                        <div style="font-size:22px;font-weight:900;color:#006a65;">{urban_count}</div>
                    </div>
                    <div style="background:#fff4d6;border-radius:14px;padding:12px;">
                        <div style="font-size:11px;font-weight:800;color:#6c7a78;">Suburban</div>
                        <div style="font-size:22px;font-weight:900;color:#785900;">{rural_count}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(
            "High project density often correlates with PM10/PM2.5 dust spikes "
            "that models may miss without location context."
        )
    st.markdown(_card_close(), unsafe_allow_html=True)


def render_scenic_card() -> None:
    st.markdown(
        """
        <div class="hai-scenic-card">
            <div class="hai-scenic-text">
                <strong>High Visibility Expected 📸</strong>
                <span>Tomorrow looks better for outdoor photos and West Lake walks when PM2.5 is lower.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_download_section(day_df: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(_card_open("Download Your Day Plan", f"Hourly schedule for {selected_date:%d %b %Y}"), unsafe_allow_html=True)
    plan_df = build_table(day_df)
    st.dataframe(plan_df, use_container_width=True, hide_index=True)
    st.download_button(
        label="Download as CSV",
        data=BytesIO(plan_df.to_csv(index=False).encode("utf-8-sig")),
        file_name="hanoi_pm25_day_plan.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown(_card_close(), unsafe_allow_html=True)


def main() -> None:
    configure_page()

    required_paths = [PREDICTIONS_PATH, HISTORY_PATH, MODEL_BUNDLE_PATH]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        st.error(
            "Missing required artifact(s): "
            + ", ".join(missing)
            + ". Run `python main/hanoi_pm25_forecast.py` to regenerate them."
        )
        st.stop()

    predictions_df = load_predictions(str(PREDICTIONS_PATH), safe_mtime(PREDICTIONS_PATH))
    history_df = load_history(str(HISTORY_PATH), safe_mtime(HISTORY_PATH))
    model_bundle = load_model_bundle(str(MODEL_BUNDLE_PATH), safe_mtime(MODEL_BUNDLE_PATH))
    historical_df = build_historical_forecast_frame(history_df, predictions_df, model_bundle)

    today = current_local_date()
    planner_df, warnings = build_planner_forecasts(history_df, model_bundle, today)

    mode, selected_date, selected_hour, show_baseline, show_history_model = render_sidebar(historical_df, model_bundle, today)
    selected_district = render_district_selector()
    render_top_nav()
    st.markdown(
        f"""
        <div class="hai-page-title">Hanoi Air Planner 🌬️</div>
        <div class="hai-page-subtitle">
            Stay vibrant and safe with source-aware PM2.5 forecasts for <b>{selected_district}</b> · {format_date_label(selected_date)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_timestamp = build_target_timestamp(selected_date, selected_hour)
    day_df = slice_day(historical_df if mode == "Historical" else planner_df, selected_date)
    if day_df.empty:
        st.error(f"No forecast rows are available for {format_date_label(selected_date)}.")
        st.stop()

    target_row = lookup_target(day_df, selected_timestamp)
    band = get_air_quality_band(float(target_row["forecast_pm25"]))
    
    # Mining integration
    density_df = process_projects(PROJECTS_PATH)
    district_risk = get_district_risk(selected_district, density_df)
    safety_recs = get_safety_recommendations(float(target_row["forecast_pm25"]), district_risk["level"])

    if mode != "Historical":
        for warning in warnings:
            st.warning(warning)

    top_col, side_col = st.columns([2, 1], gap="large")
    with top_col:
        render_forecast_snapshot(target_row, selected_timestamp, selected_district, band, district_risk)
        if district_risk["projects"]:
            with st.expander(f"Active projects in {selected_district} ({len(district_risk['projects'])})"):
                for project in district_risk["projects"]:
                    st.write(f"- {project}")

    with side_col:
        render_highlight_row(day_df)

    render_recommendation_banner(safety_recs, band, selected_district, selected_timestamp)
    render_advice_card(safety_recs)

    st.markdown(_card_open("Daily Forecast Timeline 📈", "PM2.5 Projection (24h)"), unsafe_allow_html=True)
    st.plotly_chart(build_chart(day_df, selected_timestamp, show_baseline, show_history_model), use_container_width=True)
    st.markdown(_card_close(), unsafe_allow_html=True)

    render_forecast_engines(model_bundle)
    mine_col, scenic_col = st.columns([1, 1], gap="large")
    with mine_col:
        render_mining_dashboard(density_df, selected_district)
    with scenic_col:
        st.markdown("<hr>", unsafe_allow_html=True)
        render_scenic_card()
    render_download_section(day_df, selected_date)

    st.caption(
        f"Planner source windows: direct AQI through {(today + pd.Timedelta(days=LIVE_AIR_QUALITY_DAYS - 1)):%d %b %Y}, "
        f"weather-driven local model through {(today + pd.Timedelta(days=LIVE_WEATHER_DAYS - 1)):%d %b %Y}, "
        f"and low-confidence fallback through {(today + pd.Timedelta(days=PLANNER_HORIZON_DAYS - 1)):%d %b %Y}."
    )
    return

    top_col, side_col = st.columns([1, 1.4])
    with top_col:
        st.subheader("Forecast snapshot")
        st.metric("Predicted PM2.5", f"{float(target_row['forecast_pm25']):.1f} ug/m3")
        st.write(f"Time: `{selected_timestamp:%H:%M}`")
        st.write(f"District: `{selected_district}`")
        st.write(f"Construction Risk: `{district_risk['level']}`")
        st.write(f"AQI Band: `{band['label']}`")
    
    with side_col:
        st.subheader("Recommended category advice")
        tabs = st.tabs(["🏃 Exercise", "🚲 Commuting", "☕ Hanging Out"])
        with tabs[0]:
            st.write(safety_recs["Exercise"])
        with tabs[1]:
            st.write(safety_recs["Commuting"])
        with tabs[2]:
            st.write(safety_recs["Hanging Out"])
        
        if district_risk["projects"]:
            with st.expander(f"View projects in {selected_district}"):
                for p in district_risk["projects"]:
                    st.write(f"- {p}")

    ranked = day_df.sort_values("forecast_pm25").reset_index(drop=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Cleanest hour", ranked.iloc[0]["datetime"].strftime("%H:%M"), f"{ranked.iloc[0]['forecast_pm25']:.1f} ug/m3")
    c2.metric("Most polluted hour", ranked.iloc[-1]["datetime"].strftime("%H:%M"), f"{ranked.iloc[-1]['forecast_pm25']:.1f} ug/m3")
    c3.metric("Day average", f"{day_df['forecast_pm25'].mean():.1f} ug/m3")

    st.subheader("Daily forecast timeline")
    st.plotly_chart(build_chart(day_df, selected_timestamp, show_baseline, show_history_model), use_container_width=True)

    st.subheader("Forecast engines")
    metric_rows = [
        {"Model": "Local linear model", "Use case": "Historical dataset rows", "RMSE": round(model_bundle["metrics"]["local_model"]["RMSE"], 3), "MAE": round(model_bundle["metrics"]["local_model"]["MAE"], 3), "Accuracy note": f"R2 = {model_bundle['metrics']['local_model']['R2']:.4f}"},
        {"Model": "Lag-1 baseline", "Use case": "Reference benchmark", "RMSE": round(model_bundle["metrics"]["baseline"]["RMSE"], 3), "MAE": round(model_bundle["metrics"]["baseline"]["MAE"], 3), "Accuracy note": f"R2 = {model_bundle['metrics']['baseline']['R2']:.4f}"},
        {"Model": "Day-of-week/hour fallback", "Use case": "Planner days 17-30 and offline fallback", "RMSE": round(model_bundle["metrics"]["fallback"]["RMSE"], 3), "MAE": round(model_bundle["metrics"]["fallback"]["MAE"], 3), "Accuracy note": f"R2 = {model_bundle['metrics']['fallback']['R2']:.4f}"},
        {"Model": "Open-Meteo live AQI", "Use case": "Planner near term", "RMSE": np.nan, "MAE": np.nan, "Accuracy note": "Direct external forecast source"},
    ]
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

    st.subheader("Download your day plan")
    plan_df = build_table(day_df)
    st.dataframe(plan_df, use_container_width=True, hide_index=True)
    st.download_button(
        label="Download this day as CSV",
        data=BytesIO(plan_df.to_csv(index=False).encode("utf-8-sig")),
        file_name="hanoi_pm25_day_plan.csv",
        mime="text/csv",
        use_container_width=True,
    )

    render_mining_dashboard(density_df)

    st.caption(
        f"Planner source windows: direct AQI through {(today + pd.Timedelta(days=LIVE_AIR_QUALITY_DAYS - 1)):%d %b %Y}, weather-driven local model through {(today + pd.Timedelta(days=LIVE_WEATHER_DAYS - 1)):%d %b %Y}, and low-confidence fallback through {(today + pd.Timedelta(days=PLANNER_HORIZON_DAYS - 1)):%d %b %Y}."
    )


def main() -> None:
    configure_page()

    required_paths = [PREDICTIONS_PATH, HISTORY_PATH, MODEL_BUNDLE_PATH]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        st.error(
            "Missing required artifact(s): "
            + ", ".join(missing)
            + ". Run `python main/hanoi_pm25_forecast.py` to regenerate them."
        )
        st.stop()

    predictions_df = load_predictions(str(PREDICTIONS_PATH), safe_mtime(PREDICTIONS_PATH))
    history_df = load_history(str(HISTORY_PATH), safe_mtime(HISTORY_PATH))
    model_bundle = load_model_bundle(str(MODEL_BUNDLE_PATH), safe_mtime(MODEL_BUNDLE_PATH))
    historical_df = build_historical_forecast_frame(history_df, predictions_df, model_bundle)

    today = current_local_date()
    planner_df, warnings = build_planner_forecasts(history_df, model_bundle, today)

    mode, selected_date, selected_hour, show_baseline, show_history_model = render_sidebar(
        historical_df,
        model_bundle,
        today,
    )
    selected_district = render_district_selector()
    selected_timestamp = build_target_timestamp(selected_date, selected_hour)
    now = current_local_datetime()
    current_hour_timestamp = now.normalize() + pd.Timedelta(hours=now.hour)
    is_real_time_view = mode != "Historical" and selected_timestamp == current_hour_timestamp
    snapshot_heading = "Right now in real life" if is_real_time_view else "Your selected hour"
    advice_heading = "Recommended for right now" if is_real_time_view else "Pick the plan that matches your selected time"
    display_time = now.strftime("%H:%M") if is_real_time_view else selected_timestamp.strftime("%H:%M")
    day_df = slice_day(historical_df if mode == "Historical" else planner_df, selected_date)
    if day_df.empty:
        st.error(f"Sorry, no forecast rows are available for {format_date_label(selected_date)}.")
        st.stop()

    target_row = lookup_target(day_df, selected_timestamp)
    band = get_air_quality_band(float(target_row["forecast_pm25"]))
    density_df = process_projects(PROJECTS_PATH)
    district_risk = get_district_risk(selected_district, density_df)
    safety_recs = get_safety_recommendations(float(target_row["forecast_pm25"]), district_risk["level"])

    st.markdown(
        f"""
        <div class="hai-page-title">Hanoi Air Planner 🌬️</div>
        <div class="hai-page-subtitle">
            Friendly PM2.5 planning for <b>{selected_district}</b> · {format_date_label(selected_date)} · {display_time}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if mode != "Historical":
        for warning in warnings:
            st.warning(warning)

    top_col, side_col = st.columns([1, 1.4])
    with top_col:
        st.markdown(_card_open("Forecast Snapshot ✨", snapshot_heading), unsafe_allow_html=True)
        st.metric("Predicted PM2.5 🌫️", f"{float(target_row['forecast_pm25']):.1f} ug/m3")
        st.write(f"🕒 Time now: `{display_time}`" if is_real_time_view else f"🕒 Time: `{selected_timestamp:%H:%M}`")
        st.write(f"📍 District: `{selected_district}`")
        st.write(f"🏗️ Construction risk: `{district_risk['level']}`")
        st.write(f"🌈 Air quality: `{band['label']}`")
        st.info(f"Quick tip: {band['advice']}")
        st.markdown(_card_close(), unsafe_allow_html=True)

    with side_col:
        st.markdown(
            _card_open("Recommended Category Advice 💡", advice_heading),
            unsafe_allow_html=True,
        )
        tabs = st.tabs(["🏃 Exercise", "🚲 Commuting", "☕ Hanging out"])
        with tabs[0]:
            st.write(safety_recs["Exercise"])
        with tabs[1]:
            st.write(safety_recs["Commuting"])
        with tabs[2]:
            st.write(safety_recs["Hanging Out"])

        if district_risk["projects"]:
            with st.expander(f"🏗️ View projects in {selected_district}"):
                for project in district_risk["projects"]:
                    st.write(f"- {project}")
        st.markdown(_card_close(), unsafe_allow_html=True)

    ranked = day_df.sort_values("forecast_pm25").reset_index(drop=True)
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "🌿 Cleanest hour",
        ranked.iloc[0]["datetime"].strftime("%H:%M"),
        f"{ranked.iloc[0]['forecast_pm25']:.1f} ug/m3",
    )
    c2.metric(
        "⚠️ Most polluted hour",
        ranked.iloc[-1]["datetime"].strftime("%H:%M"),
        f"{ranked.iloc[-1]['forecast_pm25']:.1f} ug/m3",
    )
    c3.metric("📊 Day average", f"{day_df['forecast_pm25'].mean():.1f} ug/m3")

    st.markdown(_card_open("Daily Forecast Timeline 📈", "How PM2.5 changes through the day"), unsafe_allow_html=True)
    st.plotly_chart(
        build_chart(day_df, selected_timestamp, show_baseline, show_history_model),
        use_container_width=True,
    )
    st.markdown(_card_close(), unsafe_allow_html=True)

    st.markdown(_card_open("Forecast Engines 🧠", "What powers this forecast"), unsafe_allow_html=True)
    metric_rows = [
        {"Model": "Local linear model", "Use case": "Historical dataset rows", "RMSE": round(model_bundle["metrics"]["local_model"]["RMSE"], 3), "MAE": round(model_bundle["metrics"]["local_model"]["MAE"], 3), "Accuracy note": f"R2 = {model_bundle['metrics']['local_model']['R2']:.4f}"},
        {"Model": "Lag-1 baseline", "Use case": "Reference benchmark", "RMSE": round(model_bundle["metrics"]["baseline"]["RMSE"], 3), "MAE": round(model_bundle["metrics"]["baseline"]["MAE"], 3), "Accuracy note": f"R2 = {model_bundle['metrics']['baseline']['R2']:.4f}"},
        {"Model": "Day-of-week/hour fallback", "Use case": "Planner days 17-30 and offline fallback", "RMSE": round(model_bundle["metrics"]["fallback"]["RMSE"], 3), "MAE": round(model_bundle["metrics"]["fallback"]["MAE"], 3), "Accuracy note": f"R2 = {model_bundle['metrics']['fallback']['R2']:.4f}"},
        {"Model": "Open-Meteo live AQI", "Use case": "Planner near term", "RMSE": np.nan, "MAE": np.nan, "Accuracy note": "Direct external forecast source"},
    ]
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
    st.markdown(_card_close(), unsafe_allow_html=True)

    st.markdown(_card_open("Download Your Day Plan 📥", "Save the hourly table as CSV"), unsafe_allow_html=True)
    plan_df = build_table(day_df)
    st.dataframe(plan_df, use_container_width=True, hide_index=True)
    st.download_button(
        label="Download this day as CSV",
        data=BytesIO(plan_df.to_csv(index=False).encode("utf-8-sig")),
        file_name="hanoi_pm25_day_plan.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown(_card_close(), unsafe_allow_html=True)

    render_mining_dashboard(density_df, selected_district)

    st.caption(
        f"Planner source windows: direct AQI through {(today + pd.Timedelta(days=LIVE_AIR_QUALITY_DAYS - 1)):%d %b %Y}, "
        f"weather-driven local model through {(today + pd.Timedelta(days=LIVE_WEATHER_DAYS - 1)):%d %b %Y}, "
        f"and low-confidence fallback through {(today + pd.Timedelta(days=PLANNER_HORIZON_DAYS - 1)):%d %b %Y}."
    )


def main() -> None:
    configure_page()

    required_paths = [PREDICTIONS_PATH, HISTORY_PATH, MODEL_BUNDLE_PATH]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        st.error(
            "Missing required artifact(s): "
            + ", ".join(missing)
            + ". Run `python main/hanoi_pm25_forecast.py` to regenerate them."
        )
        st.stop()

    predictions_df = load_predictions(str(PREDICTIONS_PATH), safe_mtime(PREDICTIONS_PATH))
    history_df = load_history(str(HISTORY_PATH), safe_mtime(HISTORY_PATH))
    model_bundle = load_model_bundle(str(MODEL_BUNDLE_PATH), safe_mtime(MODEL_BUNDLE_PATH))
    historical_df = build_historical_forecast_frame(history_df, predictions_df, model_bundle)

    today = current_local_date()
    planner_df, warnings = build_planner_forecasts(history_df, model_bundle, today)
    mode, selected_date, selected_hour, show_baseline, show_history_model = render_sidebar(
        historical_df,
        model_bundle,
        today,
    )
    selected_district = render_district_selector()
    selected_timestamp = build_target_timestamp(selected_date, selected_hour)
    now = current_local_datetime()
    current_hour_timestamp = now.normalize() + pd.Timedelta(hours=now.hour)
    is_real_time_view = mode != "Historical" and selected_timestamp == current_hour_timestamp
    display_time = now.strftime("%H:%M") if is_real_time_view else selected_timestamp.strftime("%H:%M")

    day_df = slice_day(historical_df if mode == "Historical" else planner_df, selected_date)
    if day_df.empty:
        st.error(f"Sorry, no forecast rows are available for {format_date_label(selected_date)}.")
        st.stop()

    target_row = lookup_target(day_df, selected_timestamp)
    band = get_air_quality_band(float(target_row["forecast_pm25"]))
    density_df = process_projects(PROJECTS_PATH)
    district_risk = get_district_risk(selected_district, density_df)

    st.markdown(
        f"""
        <div class="hai-page-title">Hanoi Air Planner 🌬️</div>
        <div class="hai-page-subtitle">
            Friendly PM2.5 planning for <b>{selected_district}</b> · {format_date_label(selected_date)} · {display_time}
        </div>
        """,
        unsafe_allow_html=True,
    )

    real_time_label = "right now" if is_real_time_view else "selected time"
    st.markdown(
        f"""
        <div class="hai-live-summary">
            <div>
                <strong>PM2.5 for {real_time_label}: {float(target_row['forecast_pm25']):.1f} ug/m3 · {band['label']}</strong>
                <span>{display_time} · {selected_district} · Construction risk: {district_risk['level']} · {band['advice']}</span>
            </div>
            <div class="hai-live-pill">Live guide 💡</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if mode != "Historical":
        for warning in warnings:
            st.warning(warning)

    if district_risk["projects"]:
        with st.expander(f"🏗️ View projects in {selected_district}"):
            for project in district_risk["projects"]:
                st.write(f"- {project}")

    ranked = day_df.sort_values("forecast_pm25").reset_index(drop=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("🌿 Cleanest hour", ranked.iloc[0]["datetime"].strftime("%H:%M"), f"{ranked.iloc[0]['forecast_pm25']:.1f} ug/m3")
    c2.metric("⚠️ Most polluted hour", ranked.iloc[-1]["datetime"].strftime("%H:%M"), f"{ranked.iloc[-1]['forecast_pm25']:.1f} ug/m3")
    c3.metric("📊 Day average", f"{day_df['forecast_pm25'].mean():.1f} ug/m3")

    st.markdown(_card_open("Daily Forecast Timeline 📈", "How PM2.5 changes through the day"), unsafe_allow_html=True)
    st.plotly_chart(build_chart(day_df, selected_timestamp, show_baseline, show_history_model), use_container_width=True)
    st.markdown(_card_close(), unsafe_allow_html=True)

    st.markdown(_card_open("Forecast Engines 🧠", "What powers this forecast"), unsafe_allow_html=True)
    metric_rows = [
        {"Model": "Local linear model", "Use case": "Historical dataset rows", "RMSE": round(model_bundle["metrics"]["local_model"]["RMSE"], 3), "MAE": round(model_bundle["metrics"]["local_model"]["MAE"], 3), "Accuracy note": f"R2 = {model_bundle['metrics']['local_model']['R2']:.4f}"},
        {"Model": "Lag-1 baseline", "Use case": "Reference benchmark", "RMSE": round(model_bundle["metrics"]["baseline"]["RMSE"], 3), "MAE": round(model_bundle["metrics"]["baseline"]["MAE"], 3), "Accuracy note": f"R2 = {model_bundle['metrics']['baseline']['R2']:.4f}"},
        {"Model": "Day-of-week/hour fallback", "Use case": "Planner days 17-30 and offline fallback", "RMSE": round(model_bundle["metrics"]["fallback"]["RMSE"], 3), "MAE": round(model_bundle["metrics"]["fallback"]["MAE"], 3), "Accuracy note": f"R2 = {model_bundle['metrics']['fallback']['R2']:.4f}"},
        {"Model": "Open-Meteo live AQI", "Use case": "Planner near term", "RMSE": np.nan, "MAE": np.nan, "Accuracy note": "Direct external forecast source"},
    ]
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
    st.markdown(_card_close(), unsafe_allow_html=True)

    st.markdown(_card_open("Download Your Day Plan 📥", "Save the hourly table as CSV"), unsafe_allow_html=True)
    plan_df = build_table(day_df)
    st.dataframe(plan_df, use_container_width=True, hide_index=True)
    st.download_button(
        label="Download this day as CSV",
        data=BytesIO(plan_df.to_csv(index=False).encode("utf-8-sig")),
        file_name="hanoi_pm25_day_plan.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown(_card_close(), unsafe_allow_html=True)

    render_mining_dashboard(density_df, selected_district)

    st.caption(
        f"Planner source windows: direct AQI through {(today + pd.Timedelta(days=LIVE_AIR_QUALITY_DAYS - 1)):%d %b %Y}, "
        f"weather-driven local model through {(today + pd.Timedelta(days=LIVE_WEATHER_DAYS - 1)):%d %b %Y}, "
        f"and low-confidence fallback through {(today + pd.Timedelta(days=PLANNER_HORIZON_DAYS - 1)):%d %b %Y}."
    )


if __name__ == "__main__":
    main()
