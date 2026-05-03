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

from main.forecasting_core import (
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


def configure_page() -> None:
    st.set_page_config(page_title="Hanoi Air Planner", page_icon="Air", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(210,244,234,0.88), transparent 28%),
                radial-gradient(circle at top left, rgba(226,238,255,0.92), transparent 30%),
                linear-gradient(180deg, #f7fbff 0%, #eef5f8 56%, #ecf7f1 100%);
        }
        section[data-testid="stSidebar"] {
            background: rgba(239,247,248,0.95);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def current_local_date() -> pd.Timestamp:
    return pd.Timestamp.now(tz=APP_TIMEZONE).normalize().tz_localize(None)


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


def build_chart(day_df: pd.DataFrame, target_timestamp: pd.Timestamp, show_baseline: bool, show_history_model: bool = False) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=day_df["datetime"], y=day_df["forecast_pm25"], mode="lines+markers", name="Main forecast", line={"color": "#0f766e", "width": 4}))
    if show_history_model and "history_model_pm25" in day_df.columns and day_df["history_model_pm25"].notna().any():
        figure.add_trace(go.Scatter(x=day_df["datetime"], y=day_df["history_model_pm25"], mode="lines", name="History-based model", line={"color": "#6366f1", "dash": "dot", "width": 3}))
    if day_df["actual_pm25"].notna().any():
        figure.add_trace(go.Scatter(x=day_df["datetime"], y=day_df["actual_pm25"], mode="lines+markers", name="Actual PM2.5", line={"color": "#163847", "width": 3}))
    if show_baseline and day_df["baseline_pred"].notna().any():
        figure.add_trace(go.Scatter(x=day_df["datetime"], y=day_df["baseline_pred"], mode="lines", name="Baseline", line={"color": "#f97316", "dash": "dash", "width": 2.5}))
    target_row = lookup_target(day_df, target_timestamp)
    figure.add_vline(x=target_timestamp, line_dash="dot", line_color="#2563eb", line_width=2)
    figure.add_annotation(x=target_timestamp, y=float(target_row["forecast_pm25"]), text="Selected hour", showarrow=True, arrowhead=2, ay=-45)
    figure.update_layout(height=440, margin={"l": 10, "r": 10, "t": 10, "b": 10}, xaxis_title="Hour", yaxis_title="PM2.5 (ug/m3)", hovermode="x unified")
    return figure


def render_sidebar(
    historical_df: pd.DataFrame,
    model_bundle: dict[str, object],
    today: pd.Timestamp,
) -> tuple[str, pd.Timestamp, str, bool]:
    st.sidebar.markdown("## Planning controls")
    mode = st.sidebar.radio("Mode", ["Historical", "Upcoming planner"], index=1)

    if mode == "Historical":
        date_options = sorted(historical_df["datetime"].dt.date.unique())
        selected_date = st.sidebar.selectbox("Date", date_options, index=date_options.index(pd.Timestamp(model_bundle["local_history_end"]).date()), format_func=format_date_label)
        day_df = slice_day(historical_df, pd.Timestamp(selected_date))
        hour_options = day_df["datetime"].dt.strftime("%H:%M").tolist()
        default_hour = "08:00" if "08:00" in hour_options else hour_options[0]
        selected_hour = st.sidebar.selectbox("Hour", hour_options, index=hour_options.index(default_hour))
        show_baseline = st.sidebar.toggle("Show comparison lines", value=True)
        show_history_model = False
        st.sidebar.info(
            f"Local dataset: {pd.Timestamp(model_bundle['local_history_start']):%d %b %Y} to {pd.Timestamp(model_bundle['local_history_end']):%d %b %Y}. "
            f"Saved holdout forecast: {pd.Timestamp(model_bundle['test_window_start']):%d %b %Y} to {pd.Timestamp(model_bundle['test_window_end']):%d %b %Y}."
        )
    else:
        date_options = [today.date() + pd.Timedelta(days=offset) for offset in range(PLANNER_HORIZON_DAYS)]
        selected_date = st.sidebar.selectbox("Date", date_options, index=0, format_func=format_date_label)
        selected_hour = st.sidebar.selectbox("Hour", [f"{hour:02d}:00" for hour in range(24)], index=8)
        show_baseline = False
        show_history_model = st.sidebar.toggle("Show history-based model", value=True)
        st.sidebar.info(
            f"Direct AQI: {today:%d %b %Y} to {(today + pd.Timedelta(days=LIVE_AIR_QUALITY_DAYS - 1)):%d %b %Y}. "
            f"Weather-driven local model: {(today + pd.Timedelta(days=LIVE_AIR_QUALITY_DAYS)):%d %b %Y} to {(today + pd.Timedelta(days=LIVE_WEATHER_DAYS - 1)):%d %b %Y}. "
            f"Offline fallback: {(today + pd.Timedelta(days=LIVE_WEATHER_DAYS)):%d %b %Y} to {(today + pd.Timedelta(days=PLANNER_HORIZON_DAYS - 1)):%d %b %Y}."
        )
        st.sidebar.caption(
            f"The local file ends on {pd.Timestamp(model_bundle['local_history_end']):%d %b %Y}, so dates in the gap before today are intentionally hidden."
        )

    return mode, pd.Timestamp(selected_date), selected_hour, show_baseline, show_history_model


def render_district_selector() -> str:
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Location settings")
    district = st.sidebar.selectbox("Trip Destination (District)", ALL_DISTRICTS, index=ALL_DISTRICTS.index("Thanh Xuân"))
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

    st.title("Hanoi Air Planner")
    st.write(
        f"Local history runs through {pd.Timestamp(model_bundle['local_history_end']):%d %b %Y}. "
        f"The upcoming planner covers {today:%d %b %Y} to {(today + pd.Timedelta(days=PLANNER_HORIZON_DAYS - 1)):%d %b %Y} with source-aware hourly forecasts."
    )

    mode, selected_date, selected_hour, show_baseline, show_history_model = render_sidebar(historical_df, model_bundle, today)
    selected_district = render_district_selector()
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


if __name__ == "__main__":
    main()
