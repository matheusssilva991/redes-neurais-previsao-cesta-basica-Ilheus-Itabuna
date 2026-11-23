"""
Shared training workflow helpers for notebooks and scripts.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO

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
def suppress_output(enabled: bool = True) -> Generator[None, None, None]:
    """Redireciona temporariamente a saída padrão e a saída de erro do Python.

    Args:
        enabled: Se verdadeiro, descarta o texto escrito no bloco com `with`.

    Yields:
        None: O bloco é executado com o redirecionamento solicitado.
    """
    if not enabled:
        yield
        return

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        yield


def format_forecast_values(results: list[float]) -> list[float]:
    """Converte as previsões normalizadas em valores para exibição.

    Args:
        results: Previsões na escala do modelo.

    Returns:
        Valores em reais, multiplicados por 1000 e arredondados a duas casas.
    """
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
    """Treina, salva o modelo e gera a previsão de uma série mensal.

    Args:
        region: Cidade da série, como `ilheus` ou `itabuna`.
        series_name: Identificador da série, como `cesta_basica` ou `arroz`.
        forecast_type: Destino do JSON, `cesta` ou `produtos`.
        model_name: Arquitetura `RNN`, `LSTM` ou `CNN`.
        look_back: Número de meses usados em cada entrada.
        forecast_horizon: Número de meses futuros a prever.
        epochs: Número de épocas de treinamento.
        batch_size: Tamanho do lote de treino e de inferência.
        save_onnx: Se verdadeiro, tenta exportar também um modelo ONNX.
        silence_training: Se verdadeiro, redireciona stdout e stderr do Python
            durante o fluxo.
        subdir: Subdiretório opcional para o modelo dentro da pasta da cidade.

    Returns:
        Dicionário com `region`, `series`, `samples` (janelas de treino),
        `final_loss` (perda da última época) e `forecasts` (previsões em
        reais, arredondadas a duas casas). O JSON guarda os valores na
        escala do modelo.
    """
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
