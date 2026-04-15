# BDM Final Project Proposal (Updated to Match Current Implementation)

## Team Information

- Team Number: 17
- Members: Tran Nam Son (2301140089), Nguyen Duc Manh (2301140061), Do Hoang Khoi (2301140054)

## Project Title

Forecasting Hourly PM2.5 in Hanoi for Student Daily Planning Using a Local-First Hybrid Forecasting Pipeline

## Track

Time Series Mining and Forecasting (with Data Preparation and Feature Engineering)

## Problem Statement

Students in Hanoi need practical hourly air-quality guidance for commuting, classes, exercise, and outdoor activities. Traditional one-shot model demos are not enough for planning because users need:

- Reliable forecasts for known historical windows
- Fresh near-term estimates for upcoming days
- Clear confidence and source transparency

This project builds a local-first forecasting app that combines offline model artifacts with live weather and air-quality feeds to support real daily decisions.

## Research Questions

1. Can a feature-rich linear model outperform a simple lag-1 baseline for hourly PM2.5 forecasting on Hanoi data?
2. How should we design a robust planner when local historical data ends, but users still need a 30-day forward view?
3. Does a source-aware forecast stack (historical replay, live AQI, weather-driven model, fallback profile) provide clearer and more trustworthy planning signals?

## Why This Topic Matters (Insight)

Hanoi students face frequent PM2.5 spikes that directly affect day-to-day behavior. The goal is not only "model accuracy" but also "decision usability":

- Hour-by-hour choices are practical for class schedules and commuting.
- Source labels and confidence labels reduce blind trust in uncertain long-range forecasts.
- A graceful fallback design is essential when APIs fail or data windows are limited.

## Dataset Plan

- Main dataset: https://www.kaggle.com/datasets/diabolicfox/hanoi-air-quality-pm2-5-weather-data-2024-2026
- Current local file: `hanoi_aqi_ml_ready_fixed.csv`
- Size and coverage: 14,451 hourly rows, from 2024-02-14 09:00:00 to 2026-01-26 07:00:00
- Target: `pm25`
- Key feature group 1: weather variables (temperature, humidity, dew point, wind, pressure, cloud cover, rain)
- Key feature group 2: temporal variables (year, month, day, hour, day_of_week, season flags)
- Key feature group 3: lag and rolling PM2.5 variables (`pm25_lag1`, `pm25_lag24`, `pm25_lag168`, rolling stats)

## Privacy and Ethics

- Uses public environmental data only
- No personal identifiers
- App includes non-medical disclaimer and confidence labels to avoid overclaiming certainty

## Implemented Approach

### 1. Shared Forecasting Core

Implemented in `main/forecasting_core.py`:

- Dataset loading and validation
- Feature-column discovery for model training
- Time-based split and metric calculation (RMSE, MAE, R2, MAPE)
- Local model training and artifact packaging
- Historical replay generation
- Recursive weather-driven forecasting
- Day-of-week and hour fallback profile for extended horizon

### 2. Local Training and Artifacts

Implemented in `main/hanoi_pm25_forecast.py`:

- Trains `LinearRegression` on numeric engineered features (excluding `datetime`, `pm25`, `source`, `season`)
- Keeps time-based split (`--test-ratio`)
- Exports `predictions.csv` with columns `datetime`, `actual_pm25`, `forecast_pm25`, `baseline_pred`, `source_label`
- Exports `model_bundle.joblib` with fitted model, feature list, metrics, local history/test window metadata, and planner fallback profile

### 3. Hybrid Forecasting Web App

Implemented in `web/app.py`:

- Two user modes: `Historical` (local artifact replay) and `Upcoming planner` (30-day hourly planning)
- Hourly-only selection (`00:00` to `23:00`) to avoid misleading minute behavior
- Forecast source window 1 (days 0-7): Open-Meteo air-quality API (`pm2_5`)
- Forecast source window 2 (days 8-16): Open-Meteo weather API + recursive local model forecast
- Forecast source window 3 (days 17-30): offline day_of_week + hour fallback profile
- Adds per-row labels: `source_label` and `confidence_label`
- Graceful degradation: API failures trigger warnings and fallback use

## Current Baseline Results (From Regenerated Artifacts)

Using the current implementation on the same dataset split:

- Persistence baseline: RMSE 7.156, MAE 5.121, R2 0.7903, MAPE 26.85%
- Local linear model: RMSE 6.701, MAE 4.850, R2 0.8161, MAPE 26.65%
- Extended fallback profile (day_of_week + hour): RMSE 18.309, MAE 15.361, R2 -0.3729, MAPE 110.87%

Interpretation:

- Local linear model improves over persistence on holdout.
- Fallback profile is intentionally lower-confidence and used only for outer planner horizon or outage scenarios.

## Evaluation Plan

### Quantitative Evaluation

- Model metrics on time-based holdout: RMSE, MAE, R2, MAPE
- Comparison target: persistence baseline vs local linear model

### Product-Level Validation

- Historical mode check 1: different hours on the same date must produce different values
- Historical mode check 2: baseline and actual overlays are shown only when available
- Planner mode check 1: source labels reflect window logic (0-7, 8-16, 17-30)
- Planner mode check 2: Tuesday vs Wednesday in fallback window should not be identical by default
- Reliability check: API outage handling keeps planner functional and explicit about confidence and source

## Deliverables

- Codebase with shared forecasting core and web app
- `predictions.csv` and `model_bundle.joblib` artifacts
- Streamlit app for interactive planning and CSV export
- Documentation (`README.md`, updated `project_proposal.md`)

## Risks and Mitigation

- Risk: live API downtime or rate issues
- Mitigation: cached calls, warning messages, offline fallback profile
- Risk: long-horizon uncertainty
- Mitigation: explicit confidence labels and source transparency
- Risk: dataset window ends before current date
- Mitigation: mode-gated date selection and planner-only future windows

## Scope Statement

The project prioritizes practical planning support over long-range scientific certainty. The system is designed as a transparent, local-first decision aid rather than a medical or regulatory forecasting tool.
