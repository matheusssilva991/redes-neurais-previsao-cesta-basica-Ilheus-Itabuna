"""
Shared training workflow helpers for notebooks and scripts.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from typing import Iterator

from .data_utils import (
    create_time_sequences,
    generate_forecast,
    load_unified_data,
    prepare_training_data,
    save_forecasts,
    save_model,
    train_model,
)

try:
    from models import get_model
except ImportError:  # pragma: no cover - package import fallback
    from src.models import get_model


@contextmanager
def suppress_output(enabled: bool = True) -> Iterator[None]:
    """Temporarily suppress stdout and stderr."""
    if not enabled:
        yield
        return

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        yield


def format_forecast_values(results: list[float]) -> list[float]:
    """Convert normalized forecast values back to Reais for display."""
    return [round(value * 1000, 2) for value in results]


def train_and_forecast(
    *,
    region: str,
    series_name: str,
    forecast_type: str,
    model_name: str,
    look_back: int,
    forecast_horizon: int,
    epochs: int,
    batch_size: int,
    save_onnx: bool,
    silence_training: bool,
    subdir: str | None = None,
) -> dict[str, object]:
    """Train one series, save the model/forecast, and return a compact summary."""
    with suppress_output(silence_training):
        df = load_unified_data(region, series_name)
        df = create_time_sequences(df, look_back, forecast_horizon)
        X_train, y_train, X_val = prepare_training_data(
            df,
            look_back,
            forecast_horizon,
        )

        model = get_model(model_name, look_back, forecast_horizon)
        history = train_model(
            model,
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
        )

        save_model(
            model,
            model_name,
            region,
            series_name,
            forecast_horizon,
            subdir=subdir,
            save_onnx=save_onnx,
        )

        results = generate_forecast(model, X_val, batch_size=batch_size)
        save_forecasts(
            results,
            model_name,
            series_name,
            region,
            forecast_horizon,
            forecast_type=forecast_type,
        )

    return {
        "region": region,
        "series": series_name,
        "samples": len(X_train),
        "final_loss": round(float(history.history["loss"][-1]), 6),
        "forecasts": format_forecast_values(results),
    }
