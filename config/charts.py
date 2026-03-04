"""
Configurações de gráficos: cores, marcadores, labels e estilos visuais.
"""

# Configurações gerais de gráficos
CHART_FIGSIZE = (15, 5)
CHART_DPI = 300
CHART_GRID_ALPHA = 0.5

# Labels padronizados (português)
CHART_MONTH_LABELS_PT = [
    "Jan",
    "Fev",
    "Mar",
    "Abr",
    "Mai",
    "Jun",
    "Jul",
    "Ago",
    "Set",
    "Out",
    "Nov",
    "Dez",
]

# ============================================================================
# CONFIGURAÇÕES DE PRODUTOS
# ============================================================================

CHART_PRODUCTS_MARKERS = ["s", ".", "v", "p", "p", "X", "*", "D", "^", "8", "P", ">"]

CHART_PRODUCTS_COLORS_FORECAST = [
    "#0004c7",
    "#c9261a",
    "#c1c718",
    "#57210a",
    "#6fad11",
    "#039eff",
    "#ffb100",
    "#00fa08",
    "#ff0044",
    "#210109",
    "#780c6d",
    "#b200b5",
]

CHART_PRODUCTS_COLORS_REAL = [
    "#7578ff",
    "#e6837c",
    "#c0c28c",
    "#634b41",
    "#aac77f",
    "#aadbfa",
    "#f7d381",
    "#a2fca4",
    "#f79cb4",
    "#736e6f",
    "#854e7f",
    "#c877c9",
]

CHART_PRODUCTS_MARKER_SIZES = [8, 17, 10, 10, 10, 10, 15, 8, 10, 10, 10, 10]
CHART_LINE_SIZE = 3

# ============================================================================
# CONFIGURAÇÕES DE CESTA BÁSICA
# ============================================================================

CHART_CESTA_MARKERS = ["s", "v"]
CHART_CESTA_COLORS_REAL = ["#7578ff", "#f5a87f"]
CHART_CESTA_COLORS_FORECAST = ["#0004c7", "#f25705"]
CHART_CESTA_LABELS_REAL = ["Ilhéus Valores Reais", "Itabuna Valores Reais"]
CHART_CESTA_LABELS_FORECAST = ["Ilhéus Previsão", "Itabuna Previsão"]

# ============================================================================
# QUANTIDADES DE PRODUTOS
# ============================================================================

# Quantidades específicas usadas nos gráficos históricos (preserva escala visual atual)
CHART_PRODUTOS_QUANTIDADES = {
    "acucar": 3.0,
    "arroz": 3.6,
    "banana": 7.5,
    "cafe": 0.3,
    "carne": 4.5,
    "farinha": 3.0,
    "feijao": 4.5,
    "leite": 6.0,
    "manteiga": 0.75,
    "oleo": 1.0,
    "pao": 6.0,
    "tomate": 12.0,
}

# Quantidades dos produtos na cesta básica
PRODUTOS_QUANTIDADES = {
    "acucar": 3.0,
    "arroz": 3.0,
    "banana": 90.0,
    "cafe": 0.3,
    "carne": 6.0,
    "farinha": 1.5,
    "feijao": 4.5,
    "leite": 7.5,
    "manteiga": 0.75,
    "oleo": 0.9,
    "pao": 6.0,
    "tomate": 9.0,
}
