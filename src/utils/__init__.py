"""
Utilities package for data processing and chart generation.

Este pacote fornece ferramentas para:
- Processamento de dados (data_utils)
- Geração de gráficos (chart_utils)
- Logging estruturado (logger)

Uso:
    from utils.data_utils import load_data, train_model
    from utils.chart_utils import setup_plot_style
    from utils.logger import setup_logger
"""

from importlib import import_module

_EXPORTS = {
    # Data utilities
    "load_data": ".data_utils",
    "create_time_sequences": ".data_utils",
    "prepare_training_data": ".data_utils",
    "train_model": ".data_utils",
    "save_model": ".data_utils",
    "generate_forecast": ".data_utils",
    "save_forecasts": ".data_utils",
    "load_unified_data": ".data_utils",
    # Training workflow utilities
    "format_forecast_values": ".training_workflow",
    "suppress_output": ".training_workflow",
    "train_and_forecast": ".training_workflow",
    # Chart utilities
    "PlotConfig": ".chart_utils",
    "extract_dates_from_dataframe": ".chart_utils",
    "derive_month_labels_from_history": ".chart_utils",
    "load_forecast_from_json": ".chart_utils",
    "load_forecasts_for_regions": ".chart_utils",
    "load_product_forecasts": ".chart_utils",
    "setup_plot_style": ".chart_utils",
    "format_yticks_with_comma": ".chart_utils",
    "add_forecast_annotation": ".chart_utils",
    "save_figure": ".chart_utils",
    "load_product_historical_data": ".chart_utils",
    "plot_product_chart": ".chart_utils",
    "plot_forecast_only_chart": ".chart_utils",
    # Monthly data utilities
    "PRODUCT_CESTA_BASICA": ".monthly_data",
    "add_or_update_prices": ".monthly_data",
    "load_price_series": ".monthly_data",
    "load_master_table": ".monthly_data",
    "month_end": ".monthly_data",
    "normalize_city": ".monthly_data",
    "normalize_month": ".monthly_data",
    "normalize_product": ".monthly_data",
    "save_master_table": ".monthly_data",
    "validate_complete_month": ".monthly_data",
    # Logger
    "setup_logger": ".logger",
    "default_logger": ".logger",
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    *_EXPORTS,
]
