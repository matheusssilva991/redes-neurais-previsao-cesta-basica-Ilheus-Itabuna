"""
Configurações centralizadas do projeto.

Este pacote organiza as configurações em módulos temáticos:
- base.py: caminhos, regiões e produtos
- charts.py: configurações de gráficos (cores, marcadores, labels)
- models.py: configurações de modelos (arquitetura de redes neurais)
- training.py: hiperparâmetros de treinamento

Uso:
    from config import CHART_PRODUCTS_MARKERS, RNN_UNITS, DATA_DIR
    from config.charts import CHART_CESTA_COLORS_REAL
    from config.models import LSTM_UNITS_1
"""

# ============================================================================
# BASE (Caminhos, Regiões, Produtos)
# ============================================================================
from config.base import (
    PROJECT_ROOT,
    SRC_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    MASTER_DATA_FILE,
    MODELS_DIR,
    FORECASTS_DIR,
    FORECASTS_PRODUTOS_DIR,
    FIGURES_DIR,
    REGIOES,
    PRODUTOS,
)

# ============================================================================
# CHARTS (Gráficos)
# ============================================================================
from config.charts import (
    CHART_FIGSIZE,
    CHART_DPI,
    CHART_GRID_ALPHA,
    CHART_MONTH_LABELS_PT,
    CHART_PRODUCTS_MARKERS,
    CHART_PRODUCTS_COLORS_FORECAST,
    CHART_PRODUCTS_COLORS_REAL,
    CHART_PRODUCTS_MARKER_SIZES,
    CHART_LINE_SIZE,
    CHART_CESTA_MARKERS,
    CHART_CESTA_COLORS_REAL,
    CHART_CESTA_COLORS_FORECAST,
    CHART_CESTA_LABELS_REAL,
    CHART_CESTA_LABELS_FORECAST,
    CHART_PRODUTOS_QUANTIDADES,
    PRODUTOS_QUANTIDADES,
)

# ============================================================================
# MODELS (Modelos de Redes Neurais)
# ============================================================================
from config.models import (
    RNN_UNITS,
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    CNN_FILTERS,
    CNN_KERNEL_SIZE,
    CNN_POOL_SIZE,
    CNN_DENSE_UNITS,
)

# ============================================================================
# TRAINING (Hiperparâmetros de Treinamento)
# ============================================================================
from config.training import (
    DEFAULT_LOOK_BACK,
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NORMALIZATION_FACTOR,
)

# Exportação clara
__all__ = [
    # Base
    "PROJECT_ROOT",
    "SRC_DIR",
    "DATA_DIR",
    "OUTPUT_DIR",
    "MASTER_DATA_FILE",
    "MODELS_DIR",
    "FORECASTS_DIR",
    "FORECASTS_PRODUTOS_DIR",
    "FIGURES_DIR",
    "REGIOES",
    "PRODUTOS",
    # Charts
    "CHART_FIGSIZE",
    "CHART_DPI",
    "CHART_GRID_ALPHA",
    "CHART_MONTH_LABELS_PT",
    "CHART_PRODUCTS_MARKERS",
    "CHART_PRODUCTS_COLORS_FORECAST",
    "CHART_PRODUCTS_COLORS_REAL",
    "CHART_PRODUCTS_MARKER_SIZES",
    "CHART_LINE_SIZE",
    "CHART_CESTA_MARKERS",
    "CHART_CESTA_COLORS_REAL",
    "CHART_CESTA_COLORS_FORECAST",
    "CHART_CESTA_LABELS_REAL",
    "CHART_CESTA_LABELS_FORECAST",
    "CHART_PRODUTOS_QUANTIDADES",
    "PRODUTOS_QUANTIDADES",
    # Models
    "RNN_UNITS",
    "LSTM_UNITS_1",
    "LSTM_UNITS_2",
    "CNN_FILTERS",
    "CNN_KERNEL_SIZE",
    "CNN_POOL_SIZE",
    "CNN_DENSE_UNITS",
    # Training
    "DEFAULT_LOOK_BACK",
    "DEFAULT_FORECAST_HORIZON",
    "DEFAULT_EPOCHS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_NORMALIZATION_FACTOR",
]
