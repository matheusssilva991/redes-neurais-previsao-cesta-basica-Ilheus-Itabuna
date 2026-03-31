"""
Utilities package for data processing and chart generation.

Este pacote fornece ferramentas para:
- Processamento de dados (data_utils)
- Geração de gráficos (chart_utils)
- Logging estruturado (logger)

Uso:
    from utils import load_data, train_model, setup_plot_style
    from utils.logger import setup_logger
"""

from .data_utils import (
    load_data,
    create_time_sequences,
    prepare_training_data,
    train_model,
    save_model,
    generate_forecast,
    save_forecasts,
)

from .chart_utils import (
    PlotConfig,
    extract_dates_from_dataframe,
    derive_month_labels_from_history,
    load_forecast_from_json,
    load_forecasts_for_regions,
    load_product_forecasts,
    setup_plot_style,
    format_yticks_with_comma,
    add_forecast_annotation,
    save_figure,
    load_product_historical_data,
    plot_product_chart,
    plot_forecast_only_chart,
)

from .logger import setup_logger, default_logger

__all__ = [
    # Data utilities
    "load_data",
    "create_time_sequences",
    "prepare_training_data",
    "train_model",
    "save_model",
    "generate_forecast",
    "save_forecasts",
    # Chart utilities
    "PlotConfig",
    "extract_dates_from_dataframe",
    "derive_month_labels_from_history",
    "load_forecast_from_json",
    "load_forecasts_for_regions",
    "load_product_forecasts",
    "setup_plot_style",
    "format_yticks_with_comma",
    "add_forecast_annotation",
    "save_figure",
    "load_product_historical_data",
    "plot_product_chart",
    "plot_forecast_only_chart",
    # Logger
    "setup_logger",
    "default_logger",
]
