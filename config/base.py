"""
Configurações base do projeto: caminhos, regiões e produtos.
"""

from pathlib import Path

# Diretórios base
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Diretórios de dados
DATA_PRODUTOS_DIR = DATA_DIR / "datasets_produtos"
DATA_ILHEUS_DIR = DATA_PRODUTOS_DIR / "ilheus"
DATA_ITABUNA_DIR = DATA_PRODUTOS_DIR / "itabuna"

# Diretórios de saída
MODELS_DIR = OUTPUT_DIR / "models"
FORECASTS_DIR = OUTPUT_DIR / "previsoes_cesta"
FORECASTS_PRODUTOS_DIR = OUTPUT_DIR / "previsoes_produtos"
FIGURES_DIR = OUTPUT_DIR / "figure"

# Regiões e produtos
REGIOES = ["ilheus", "itabuna"]
PRODUTOS = [
    "acucar",
    "arroz",
    "banana",
    "cafe",
    "carne",
    "farinha",
    "feijao",
    "leite",
    "manteiga",
    "oleo",
    "pao",
    "tomate",
]
