from __future__ import annotations

from datetime import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = ROOT_DIR / "predictions.csv"
HISTORY_PATH = ROOT_DIR / "hanoi_aqi_ml_ready_fixed.csv"
PREDICTION_COLUMNS = {
    "datetime",
    "actual_pm25",
    "persistence_pred",
    "linear_pred",
}
HISTORY_COLUMNS = {"datetime", "pm25"}


def configure_page() -> None:
    st.set_page_config(
        page_title="Hanoi Air Planner",
        page_icon="🌿",
        layout="wide",
    )


@st.cache_data(show_spinner=False)
def load_predictions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    missing = PREDICTION_COLUMNS.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"predictions.csv is missing required columns: {missing_text}")

    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    return df


@st.cache_data(show_spinner=False)
def load_history(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    missing = HISTORY_COLUMNS.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"hanoi_aqi_ml_ready_fixed.csv is missing required columns: {missing_text}")

    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date
    return df


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    y_true_np = np.asarray(y_true, dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true_np, y_pred_np))),
        "MAE": float(mean_absolute_error(y_true_np, y_pred_np)),
        "R2": float(r2_score(y_true_np, y_pred_np)),
    }


def get_air_quality_band(value: float) -> dict[str, str]:
    if value <= 12:
        return {
            "label": "Fresh",
            "icon": "🟢",
            "color": "#1f9d7a",
            "advice": "Great time for walking, light exercise, or a longer campus commute.",
        }
    if value <= 35.4:
        return {
            "label": "Moderate",
            "icon": "🟡",
            "color": "#d6a400",
            "advice": "Most people will be fine, but sensitive users may want a mask for long outdoor trips.",
        }
    if value <= 55.4:
        return {
            "label": "Sensitive groups should be careful",
            "icon": "🟠",
            "color": "#dd6b20",
            "advice": "If you are sensitive to air pollution, reduce outdoor time and use a good mask.",
        }
    if value <= 150.4:
        return {
            "label": "Unhealthy",
            "icon": "🔴",
            "color": "#d64545",
            "advice": "Keep outdoor time short, choose indoor study spots, and wear a mask when commuting.",
        }
    return {
        "label": "Very unhealthy",
        "icon": "🟣",
        "color": "#7b2cbf",
        "advice": "Try to stay indoors unless necessary and avoid outdoor exercise.",
    }


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    dt = enriched["datetime"]
    enriched["month"] = dt.dt.month
    enriched["hour"] = dt.dt.hour
    enriched["day_of_week"] = dt.dt.dayofweek
    enriched["day_of_year"] = dt.dt.dayofyear
    enriched["is_weekend"] = (enriched["day_of_week"] >= 5).astype(int)
    enriched["is_dry_season"] = enriched["month"].isin([11, 12, 1, 2, 3, 4]).astype(int)
    enriched["hour_sin"] = np.sin(2 * np.pi * enriched["hour"] / 24)
    enriched["hour_cos"] = np.cos(2 * np.pi * enriched["hour"] / 24)
    enriched["dow_sin"] = np.sin(2 * np.pi * enriched["day_of_week"] / 7)
    enriched["dow_cos"] = np.cos(2 * np.pi * enriched["day_of_week"] / 7)
    enriched["doy_sin"] = np.sin(2 * np.pi * enriched["day_of_year"] / 366)
    enriched["doy_cos"] = np.cos(2 * np.pi * enriched["day_of_year"] / 366)
    return enriched


def build_profiles(history_df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "hour_profile": history_df.groupby("hour")["pm25"].mean(),
        "month_profile": history_df.groupby("month")["pm25"].mean(),
        "dow_profile": history_df.groupby("day_of_week")["pm25"].mean(),
        "month_hour_profile": history_df.groupby(["month", "hour"])["pm25"].mean(),
        "weekend_hour_profile": history_df.groupby(["is_weekend", "hour"])["pm25"].mean(),
    }


def apply_profiles(df: pd.DataFrame, profiles: dict[str, pd.Series], global_mean: float) -> pd.DataFrame:
    enriched = df.copy()
    enriched["hour_profile"] = enriched["hour"].map(profiles["hour_profile"]).fillna(global_mean)
    enriched["month_profile"] = enriched["month"].map(profiles["month_profile"]).fillna(global_mean)
    enriched["dow_profile"] = enriched["day_of_week"].map(profiles["dow_profile"]).fillna(global_mean)
    enriched["month_hour_profile"] = [
        profiles["month_hour_profile"].get((month, hour), global_mean)
        for month, hour in zip(enriched["month"], enriched["hour"])
    ]
    enriched["weekend_hour_profile"] = [
        profiles["weekend_hour_profile"].get((weekend, hour), global_mean)
        for weekend, hour in zip(enriched["is_weekend"], enriched["hour"])
    ]
    return enriched


@st.cache_resource(show_spinner=False)
def train_calendar_model(history_csv_path: Path) -> dict[str, object]:
    history_df = load_history(history_csv_path)
    history_features = add_time_features(history_df)
    profiles = build_profiles(history_features)
    global_mean = float(history_features["pm25"].mean())
    modeled_df = apply_profiles(history_features, profiles, global_mean)

    split_idx = int(len(modeled_df) * 0.8)
    test_df = modeled_df.iloc[split_idx:].copy()
    test_pred = np.clip(
        (
            0.45 * test_df["month_hour_profile"]
            + 0.25 * test_df["weekend_hour_profile"]
            + 0.20 * test_df["hour_profile"]
            + 0.10 * test_df["month_profile"]
        ),
        0,
        None,
    )
    metrics = calculate_metrics(test_df["pm25"], test_pred)

    return {
        "profiles": profiles,
        "global_mean": global_mean,
        "metrics": metrics,
        "history_df": history_df,
    }


def build_future_features(target_datetimes: pd.Series, model_bundle: dict[str, object]) -> pd.DataFrame:
    future_df = pd.DataFrame({"datetime": pd.to_datetime(target_datetimes)})
    future_df = add_time_features(future_df)
    future_df = apply_profiles(
        future_df,
        model_bundle["profiles"],
        float(model_bundle["global_mean"]),
    )
    return future_df


def estimate_future_pm25(target_datetimes: pd.Series, model_bundle: dict[str, object]) -> pd.DataFrame:
    future_features = build_future_features(target_datetimes, model_bundle)
    future_features["forecast_pm25"] = np.clip(
        (
            0.45 * future_features["month_hour_profile"]
            + 0.25 * future_features["weekend_hour_profile"]
            + 0.20 * future_features["hour_profile"]
            + 0.10 * future_features["month_profile"]
        ),
        0,
        None,
    )
    return future_features


def build_selected_day_forecast(
    selected_date: pd.Timestamp,
    predictions_df: pd.DataFrame,
    model_bundle: dict[str, object],
) -> pd.DataFrame:
    day_start = pd.Timestamp(selected_date).normalize()
    hourly_range = pd.date_range(day_start, periods=24, freq="h")
    future_day_df = estimate_future_pm25(pd.Series(hourly_range), model_bundle)
    future_day_df["source_label"] = "Calendar-based future estimate"

    known_day = predictions_df.loc[predictions_df["date"] == day_start.date()].copy()
    if known_day.empty:
        future_day_df["actual_pm25"] = np.nan
        future_day_df["persistence_pred"] = np.nan
        future_day_df["linear_pred"] = np.nan
        return future_day_df

    merged_df = future_day_df.merge(
        known_day[["datetime", "actual_pm25", "persistence_pred", "linear_pred"]],
        on="datetime",
        how="left",
    )
    exact_mask = merged_df["linear_pred"].notna()
    merged_df.loc[exact_mask, "forecast_pm25"] = merged_df.loc[exact_mask, "linear_pred"]
    merged_df.loc[exact_mask, "source_label"] = "Exact model forecast from predictions.csv"
    return merged_df


def format_datetime_label(value: pd.Timestamp) -> str:
    return value.strftime("%A, %d %B %Y")


def build_target_timestamp(selected_date: pd.Timestamp, selected_hour: int) -> pd.Timestamp:
    return pd.Timestamp(selected_date).normalize() + pd.Timedelta(hours=selected_hour)


def lookup_target_forecast(day_df: pd.DataFrame, target_timestamp: pd.Timestamp) -> pd.Series:
    match = day_df.loc[day_df["datetime"] == target_timestamp]
    if match.empty:
        nearest_index = (day_df["datetime"] - target_timestamp).abs().idxmin()
        return day_df.loc[nearest_index]
    return match.iloc[0]


def build_planning_table(day_df: pd.DataFrame) -> pd.DataFrame:
    plan_df = day_df[["datetime", "forecast_pm25", "source_label"]].copy()
    plan_df["Air quality"] = plan_df["forecast_pm25"].map(lambda value: get_air_quality_band(float(value))["label"])
    plan_df["Suggestion"] = plan_df["forecast_pm25"].map(
        lambda value: get_air_quality_band(float(value))["advice"]
    )
    plan_df["Time"] = plan_df["datetime"].dt.strftime("%H:%M")
    return plan_df[["Time", "forecast_pm25", "Air quality", "Suggestion", "source_label"]].rename(
        columns={
            "forecast_pm25": "Predicted PM2.5",
            "source_label": "Forecast source",
        }
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(210, 244, 234, 0.9), transparent 28%),
                radial-gradient(circle at top left, rgba(230, 241, 255, 0.95), transparent 30%),
                linear-gradient(180deg, #f7fbff 0%, #eef5f8 55%, #edf7f2 100%);
        }
        .hero-card, .glass-card {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(255, 255, 255, 0.65);
            backdrop-filter: blur(14px);
            border-radius: 24px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 14px 30px rgba(25, 65, 90, 0.08);
        }
        .hero-title {
            font-size: 2.35rem;
            line-height: 1.1;
            font-weight: 800;
            color: #153b4d;
            margin-bottom: 0.4rem;
        }
        .hero-subtitle {
            color: #4f6f7c;
            font-size: 1.02rem;
            margin-bottom: 0;
        }
        .pill-row {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }
        .pill {
            background: #f1f7f8;
            border: 1px solid #d7e7ea;
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            font-size: 0.9rem;
            color: #214b5d;
        }
        .insight-card {
            border-radius: 22px;
            padding: 1rem 1.1rem;
            color: white;
            min-height: 135px;
            box-shadow: 0 12px 24px rgba(21, 59, 77, 0.12);
        }
        .insight-soft { background: linear-gradient(135deg, #0f766e, #34d399); }
        .insight-warm { background: linear-gradient(135deg, #9a3412, #fb923c); }
        .insight-slate { background: linear-gradient(135deg, #1d4ed8, #38bdf8); }
        .insight-label {
            font-size: 0.88rem;
            opacity: 0.9;
            margin-bottom: 0.3rem;
        }
        .insight-value {
            font-size: 1.8rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.2rem;
        }
        .insight-copy {
            font-size: 0.94rem;
            opacity: 0.96;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
        }
        section[data-testid="stSidebar"] {
            background: rgba(239, 247, 248, 0.95);
            border-right: 1px solid rgba(44, 100, 120, 0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(predictions_df: pd.DataFrame, model_bundle: dict[str, object]) -> None:
    metrics = model_bundle["metrics"]
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">Hanoi Air Planner</div>
            <p class="hero-subtitle">
                Pick any date and time to estimate Hanoi's PM2.5 level, compare safer hours,
                and plan classes, commuting, workouts, or outdoor breaks with confidence.
            </p>
            <div class="pill-row">
                <div class="pill">🌤 Built for non-technical users</div>
                <div class="pill">📅 Future date friendly</div>
                <div class="pill">📈 Historical range: {predictions_df['datetime'].min():%b %d, %Y} to {predictions_df['datetime'].max():%b %d, %Y}</div>
                <div class="pill">🤖 Future model RMSE: {metrics['RMSE']:.2f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(predictions_df: pd.DataFrame, history_df: pd.DataFrame) -> tuple[pd.Timestamp, time, bool]:
    st.sidebar.markdown("## Plan your day")
    st.sidebar.write("Choose a date and hour. The app will use the exact saved forecast when available, then switch to a future estimate outside that range.")

    min_date = history_df["datetime"].min().date()
    max_date = predictions_df["datetime"].max().date()
    default_date = max_date
    far_future = pd.Timestamp.today().date().replace(year=pd.Timestamp.today().year + 3)

    selected_date = st.sidebar.date_input(
        "Date",
        value=default_date,
        min_value=min_date,
        max_value=far_future,
    )
    selected_time = st.sidebar.time_input("Time", value=time(8, 0), step=1800)
    show_baseline = st.sidebar.toggle("Show comparison lines", value=True)

    st.sidebar.markdown("### What you are seeing")
    st.sidebar.info(
        "Inside the saved forecast window, the app shows the exact linear-model forecast from `predictions.csv`. "
        "For other dates, it uses a calendar-pattern model trained on the historical Hanoi dataset."
    )
    st.sidebar.caption("Forecasts are hourly. If you pick 08:30, the planner uses the 08:00-09:00 hour block.")

    return pd.Timestamp(selected_date), selected_time, show_baseline


def render_prediction_spotlight(target_row: pd.Series, target_timestamp: pd.Timestamp) -> None:
    band = get_air_quality_band(float(target_row["forecast_pm25"]))
    badge_html = (
        f"<span style='display:inline-block;padding:0.35rem 0.7rem;border-radius:999px;"
        f"background:{band['color']};color:white;font-weight:700;'>{band['icon']} {band['label']}</span>"
    )
    st.markdown("## Your selected moment")

    summary_col, tip_col = st.columns([1.4, 1])
    with summary_col:
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-size:0.95rem;color:#4f6f7c;margin-bottom:0.5rem;">Selected time</div>
                <div style="font-size:1.9rem;font-weight:800;color:#163847;">{target_timestamp:%A, %d %B %Y at %H:%M}</div>
                <div style="margin:0.8rem 0 0.75rem 0;font-size:3rem;font-weight:900;color:#153b4d;">
                    {target_row['forecast_pm25']:.1f} <span style="font-size:1.1rem;font-weight:600;">ug/m3</span>
                </div>
                <div style="margin-bottom:0.8rem;">{badge_html}</div>
                <div style="color:#526f7a;font-size:0.98rem;">
                    Forecast source: <strong>{target_row['source_label']}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tip_col:
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-size:1rem;font-weight:700;color:#153b4d;margin-bottom:0.6rem;">Smart planning tip</div>
                <div style="color:#45606b;line-height:1.65;">{band['advice']}</div>
                <div style="margin-top:1rem;color:#45606b;line-height:1.6;">
                    Use this moment as your anchor: if this hour looks heavy, try shifting commutes,
                    outdoor exercise, or café breaks closer to the cleaner hours shown below.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_insight_cards(day_df: pd.DataFrame) -> None:
    ranked = day_df.sort_values("forecast_pm25").reset_index(drop=True)
    best_row = ranked.iloc[0]
    worst_row = ranked.iloc[-1]
    avg_value = float(day_df["forecast_pm25"].mean())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="insight-card insight-soft">
                <div class="insight-label">🌤 Best hour to step outside</div>
                <div class="insight-value">{best_row['datetime']:%H:%M}</div>
                <div class="insight-copy">{best_row['forecast_pm25']:.1f} ug/m3 predicted for the cleanest hour of the day.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="insight-card insight-warm">
                <div class="insight-label">🚦 Most polluted hour</div>
                <div class="insight-value">{worst_row['datetime']:%H:%M}</div>
                <div class="insight-copy">{worst_row['forecast_pm25']:.1f} ug/m3 predicted. Consider a shorter outdoor window then.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="insight-card insight-slate">
                <div class="insight-label">📘 Day average</div>
                <div class="insight-value">{avg_value:.1f}</div>
                <div class="insight-copy">Average PM2.5 predicted across all 24 hours for this selected date.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_daily_chart(day_df: pd.DataFrame, show_baseline: bool, selected_timestamp: pd.Timestamp) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=day_df["datetime"],
            y=day_df["forecast_pm25"],
            mode="lines+markers",
            name="Main forecast",
            line={"color": "#0f766e", "width": 4},
            marker={"size": 8},
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
                marker={"size": 7},
            )
        )

    if show_baseline and day_df["persistence_pred"].notna().any():
        figure.add_trace(
            go.Scatter(
                x=day_df["datetime"],
                y=day_df["persistence_pred"],
                mode="lines",
                name="Persistence baseline",
                line={"color": "#f97316", "dash": "dash", "width": 2.5},
            )
        )

    target_value = float(lookup_target_forecast(day_df, selected_timestamp)["forecast_pm25"])
    figure.add_vline(
        x=selected_timestamp,
        line_dash="dot",
        line_color="#2563eb",
        line_width=2,
    )
    figure.add_annotation(
        x=selected_timestamp,
        y=target_value,
        text="Selected hour",
        showarrow=True,
        arrowhead=2,
        ay=-45,
        bgcolor="rgba(37,99,235,0.12)",
        bordercolor="#2563eb",
    )

    figure.update_layout(
        height=470,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis_title="Hour",
        yaxis_title="PM2.5 (ug/m3)",
        hovermode="x unified",
        legend_title="Series",
        plot_bgcolor="rgba(255,255,255,0.45)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def render_timeline(day_df: pd.DataFrame, show_baseline: bool, selected_timestamp: pd.Timestamp) -> None:
    st.markdown("## Daily forecast timeline")
    chart_col, side_col = st.columns([1.65, 1])

    with chart_col:
        figure = build_daily_chart(day_df, show_baseline, selected_timestamp)
        st.plotly_chart(figure, use_container_width=True)
        st.caption(
            "The green line is the app's main forecast. When the selected date is inside the saved test window, "
            "you can also compare it against actual PM2.5 and the persistence baseline."
        )

    with side_col:
        ranked = day_df.sort_values("forecast_pm25").reset_index(drop=True)
        band = get_air_quality_band(float(day_df["forecast_pm25"].mean()))
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-size:1rem;font-weight:800;color:#163847;margin-bottom:0.7rem;">Day summary</div>
                <div style="margin-bottom:0.55rem;color:#4b6570;"><strong>Cleaner window:</strong> {ranked.iloc[0]['datetime']:%H:%M}</div>
                <div style="margin-bottom:0.55rem;color:#4b6570;"><strong>Pollution peak:</strong> {ranked.iloc[-1]['datetime']:%H:%M}</div>
                <div style="margin-bottom:0.55rem;color:#4b6570;"><strong>Overall feel:</strong> {band['icon']} {band['label']}</div>
                <div style="margin-top:0.8rem;color:#4b6570;line-height:1.6;">{band['advice']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_model_comparison(predictions_df: pd.DataFrame, model_bundle: dict[str, object]) -> None:
    st.markdown("## Forecast engines behind the app")
    linear_metrics = calculate_metrics(predictions_df["actual_pm25"], predictions_df["linear_pred"])
    persistence_metrics = calculate_metrics(predictions_df["actual_pm25"], predictions_df["persistence_pred"])
    future_metrics = model_bundle["metrics"]

    metrics_df = pd.DataFrame(
        [
            {
                "Model": "Exact linear forecast from predictions.csv",
                "Use case": "Dates inside the saved forecast window",
                "RMSE": round(linear_metrics["RMSE"], 3),
                "MAE": round(linear_metrics["MAE"], 3),
                "Accuracy note": f"R2 = {linear_metrics['R2']:.4f}",
            },
            {
                "Model": "Persistence baseline",
                "Use case": "Simple benchmark",
                "RMSE": round(persistence_metrics["RMSE"], 3),
                "MAE": round(persistence_metrics["MAE"], 3),
                "Accuracy note": f"R2 = {persistence_metrics['R2']:.4f}",
            },
            {
                "Model": "Calendar-based future estimator",
                "Use case": "Any future day and hour",
                "RMSE": round(future_metrics["RMSE"], 3),
                "MAE": round(future_metrics["MAE"], 3),
                "Accuracy note": "Pattern-based planner for future dates",
            },
        ]
    )

    st.write(
        "This app uses two layers. First, it reuses the saved linear-model forecast from `predictions.csv` whenever your chosen timestamp is already available. "
        "For any other date, it switches to a calendar-pattern model trained on Hanoi's historical PM2.5 rhythm by month, weekday, and hour."
    )
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)


def render_downloads(day_df: pd.DataFrame) -> None:
    st.markdown("## Download your day plan")
    planning_df = build_planning_table(day_df)
    st.dataframe(planning_df, use_container_width=True, hide_index=True)
    csv_bytes = planning_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Download this day as CSV",
        data=BytesIO(csv_bytes),
        file_name="hanoi_pm25_day_plan.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_footer() -> None:
    st.caption(
        "Planning note: future dates outside the saved forecast range are estimated from historical seasonal patterns, "
        "so they are best used for everyday planning rather than medical decisions."
    )


def main() -> None:
    configure_page()
    inject_styles()

    if not PREDICTIONS_PATH.exists():
        st.error(f"Missing file: {PREDICTIONS_PATH}")
        st.stop()
    if not HISTORY_PATH.exists():
        st.error(f"Missing file: {HISTORY_PATH}")
        st.stop()

    try:
        predictions_df = load_predictions(PREDICTIONS_PATH)
        model_bundle = train_calendar_model(HISTORY_PATH)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    render_hero(predictions_df, model_bundle)
    selected_date, selected_time, show_baseline = render_sidebar(predictions_df, model_bundle["history_df"])
    selected_timestamp = build_target_timestamp(selected_date, selected_time.hour)

    day_df = build_selected_day_forecast(selected_date, predictions_df, model_bundle)
    target_row = lookup_target_forecast(day_df, selected_timestamp)

    render_prediction_spotlight(target_row, selected_timestamp)
    render_insight_cards(day_df)
    render_timeline(day_df, show_baseline, selected_timestamp)
    render_model_comparison(predictions_df, model_bundle)
    render_downloads(day_df)
    render_footer()


if __name__ == "__main__":
    main()
