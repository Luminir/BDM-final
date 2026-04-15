from __future__ import annotations

import argparse
import sys
from pathlib import Path

from joblib import dump

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from main.forecasting_core import (
    DEFAULT_CSV,
    DEFAULT_MODEL_OUTPUT,
    DEFAULT_ZIP,
    ensure_dataset,
    format_metrics,
    load_dataset,
    safe_text,
    train_local_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the local Hanoi PM2.5 forecasting model and export app artifacts."
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
    parser.add_argument(
        "--model-output",
        default=DEFAULT_MODEL_OUTPUT,
        help=f"Path to write the serialized model bundle (default: {DEFAULT_MODEL_OUTPUT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.data)
    zip_path = Path(args.zip)
    output_path = Path(args.output)
    model_output_path = Path(args.model_output)

    try:
        dataset_path = ensure_dataset(csv_path, zip_path)
        dataset_df = load_dataset(dataset_path)
        model_bundle, predictions_df = train_local_model(dataset_df, args.test_ratio)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("Dataset loaded")
    print(f"- Rows: {len(dataset_df):,}")
    print(f"- Columns: {len(dataset_df.columns)}")
    print(
        f"- Local history window: {model_bundle['local_history_start']} -> {model_bundle['local_history_end']}"
    )
    print(
        f"- Saved forecast window: {model_bundle['test_window_start']} -> {model_bundle['test_window_end']}"
    )
    print(f"- Model feature count: {len(model_bundle['feature_columns'])}")
    print()

    print("Model performance")
    print(format_metrics("Persistence", model_bundle["metrics"]["baseline"]))
    print(format_metrics("Local linear model", model_bundle["metrics"]["local_model"]))
    print(format_metrics("DoW/hour fallback", model_bundle["metrics"]["fallback"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model_bundle, model_output_path)

    print()
    print(f"Saved predictions to: {safe_text(output_path)}")
    print(f"Saved model bundle to: {safe_text(model_output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
