"""
Utility functions for chart generation.

Este módulo fornece funções para:
- Carregar previsões de arquivos JSON
- Configurar e formatar gráficos matplotlib
- Plotar dados históricos e previsões
- Salvar figuras em alta resolução
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import json

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from config import (
    FORECASTS_DIR,
    FORECASTS_PRODUTOS_DIR,
    CHART_FIGSIZE,
    CHART_DPI,
    CHART_GRID_ALPHA,
    CHART_MONTH_LABELS_PT,
    PRODUTOS_QUANTIDADES,
    DEFAULT_NORMALIZATION_FACTOR,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class PlotConfig:
    """
    Configuração para plotagem de gráficos de produtos.

    Attributes:
        produtos: Lista de nomes de produtos
        valores_reais: Dict com valores históricos por produto
        previsoes: Dict com previsões por produto
        markers: Lista de estilos de marcadores matplotlib
        colors_real: Lista de cores para valores históricos
        colors_prev: Lista de cores para previsões
        markers_size: Lista de tamanhos de marcadores
        line_size: Espessura das linhas
        x_labels: Labels do eixo X
        limite_y: Limite superior do eixo Y
    """

    produtos: List[str]
    valores_reais: Dict[str, List[float]]
    previsoes: Dict[str, List[float]]
    markers: List[str]
    colors_real: List[str]
    colors_prev: List[str]
    markers_size: List[int]
    line_size: float
    x_labels: List[str]
    limite_y: int


def month_to_number(value: Any) -> Optional[int]:
    """
    Converte representação de mês para número (1-12).

    Aceita valores numéricos e textos em português (abreviado/completo).
    """
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        month = int(value)
        return month if 1 <= month <= 12 else None

    text = str(value).strip().lower()
    mapping = {
        "jan": 1,
        "janeiro": 1,
        "fev": 2,
        "fevereiro": 2,
        "mar": 3,
        "marco": 3,
        "março": 3,
        "abr": 4,
        "abril": 4,
        "mai": 5,
        "maio": 5,
        "jun": 6,
        "junho": 6,
        "jul": 7,
        "julho": 7,
        "ago": 8,
        "agosto": 8,
        "set": 9,
        "setembro": 9,
        "out": 10,
        "outubro": 10,
        "nov": 11,
        "novembro": 11,
        "dez": 12,
        "dezembro": 12,
    }
    return mapping.get(text[:3], mapping.get(text))


def extract_dates_from_dataframe(df: pd.DataFrame) -> Optional[List[pd.Timestamp]]:
    """
    Extrai datas mensais de um DataFrame com colunas `ano` e `mes`.
    """
    if not {"ano", "mes"}.issubset(df.columns):
        return None

    years = df["ano"].tolist()
    months = [month_to_number(month) for month in df["mes"].tolist()]
    if any(month is None for month in months):
        return None

    try:
        return [
            pd.Timestamp(year=int(year), month=int(month), day=1)
            for year, month in zip(years, months)
        ]
    except Exception:
        return None


def derive_month_labels_from_history(
    df: pd.DataFrame,
    hist_months: int,
    forecast_horizon: int,
    month_labels: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], int]:
    """
    Deriva labels de meses históricos e previstos a partir de um DataFrame.

    Returns:
        Tuple[List[str], List[str], int]: (meses_anteriores, meses_previstos, ano_previsao)
    """
    if month_labels is None:
        month_labels = CHART_MONTH_LABELS_PT

    dates = extract_dates_from_dataframe(df)
    if dates and len(dates) >= hist_months:
        history_dates = dates[-hist_months:]
    else:
        # Usar o mês anterior (último mês completo) em vez de hoje
        last_complete_month = pd.Timestamp.today().replace(day=1) - pd.DateOffset(
            months=1
        )
        history_dates = list(
            pd.date_range(end=last_complete_month, periods=hist_months, freq="MS")
        )

    previous_months = [month_labels[date.month - 1] for date in history_dates]
    last_history_date = history_dates[-1]
    forecast_dates = [
        last_history_date + pd.DateOffset(months=step)
        for step in range(1, forecast_horizon + 1)
    ]
    forecast_months = [month_labels[date.month - 1] for date in forecast_dates]
    forecast_year = forecast_dates[-1].year

    return previous_months, forecast_months, forecast_year


def load_forecast_from_json(
    filepath: Union[str, Path],
    denormalize: bool = True,
    normalization_factor: int = DEFAULT_NORMALIZATION_FACTOR,
) -> List[float]:
    """
    Carrega previsões de um arquivo JSON.

    Suporta AMBOS os formatos:
    - Novo formato (array nativo): {"produto": [0.551, 0.563, 0.572]}
    - Formato legado (string): {"produto": "[0.551, 0.563, 0.572]"}

    Args:
        filepath (str | Path): Caminho para o arquivo JSON
        denormalize (bool, optional): Se True, multiplica valores por 1000.
                                     Padrão: True.
        normalization_factor (int, optional): Fator de desnormalização.
                                             Padrão: 1000.

    Returns:
        List[float]: Lista de valores previstos (desnormalizados se denormalize=True)

    Raises:
        FileNotFoundError: Se arquivo não existir
        ValueError: Se JSON inválido ou formato incorreto

    Examples:
        >>> values = load_forecast_from_json("previsao_RNN_cesta_basica_ilheus.json")
        >>> values
        [551.0, 563.0, 572.0]  # Desnormalizados (× 1000)

        >>> # Manter normalizados
        >>> values = load_forecast_from_json("previsao.json", denormalize=False)
        >>> values
        [0.551, 0.563, 0.572]

    Notes:
        - Compatível com formato antigo (string) e novo (array)
        - Desnormalização padrão é × 1000 (valores em reais)
        - Procura pela primeira chave disponível no JSON
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error(f"Arquivo não encontrado: {filepath}")
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    try:
        logger.debug(f"Carregando previsões de: {filepath}")
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        logger.error(f"Erro ao decodificar JSON de {filepath}: {e}")
        raise ValueError(f"JSON inválido em {filepath}: {e}") from e
    except Exception as e:
        logger.error(f"Erro ao ler arquivo {filepath}: {e}")
        raise

    if not data:
        raise ValueError(f"JSON vazio em {filepath}")

    key = list(data.keys())[0]
    values_data = data[key]

    # Suportar ambos os formatos: array nativo ou string
    if isinstance(values_data, list):
        # Novo formato: array nativo
        values = [float(v) for v in values_data]
        logger.debug(f"Formato JSON: array nativo ({len(values)} valores)")
    elif isinstance(values_data, str):
        # Formato legado: string
        values_str = values_data.replace("[", "").replace("]", "")
        values = [float(v.strip()) for v in values_str.split(",")]
        logger.debug(f"Formato JSON: string legado ({len(values)} valores)")
    else:
        raise ValueError(
            f"Formato inesperado em {filepath}: esperado lista ou string, "
            f"recebido {type(values_data)}"
        )

    if denormalize:
        values = [v * normalization_factor for v in values]

    logger.info(f"Previsões carregadas: {len(values)} valores de {filepath.name}")
    return values


def load_forecasts_for_regions(
    model: str, object_name: str, regions: List[str], forecast_horizon: int = 3
) -> Dict[str, List[float]]:
    """
    Carrega previsões de múltiplas regiões.

    Args:
        model (str): Nome do modelo (RNN, LSTM, CNN)
        object_name (str): Nome do objeto (ex: 'cesta_basica')
        regions (List[str]): Lista de nomes de regiões (ex: ['ilheus', 'itabuna'])
        forecast_horizon (int, optional): Horizonte de previsão (3 ou 12).
                                         Padrão: 3.

    Returns:
        Dict[str, List[float]]: Dicionário com regiões como chaves e
                               listas de previsões como valores.
                               Valores estão desnormalizados (em reais).

    Raises:
        FileNotFoundError: Se arquivo de alguma região não existir

    Examples:
        >>> forecasts = load_forecasts_for_regions(
        ...     'RNN', 'cesta_basica', ['ilheus', 'itabuna'], forecast_horizon=3
        ... )
        >>> forecasts
        {'ilheus': [551.0, 563.0, 572.0], 'itabuna': [588.0, 610.0, 612.0]}

    Notes:
        - Automaticamente detecta formato JSON (legado ou novo)
        - Valores retornados são desnormalizados (× 1000)
    """
    results = {}
    logger.info(
        f"Carregando previsões: modelo={model}, objeto={object_name}, "
        f"regiões={regions}, horizon={forecast_horizon}"
    )

    for region in regions:
        if forecast_horizon == 12:
            filename = f"previsao_{model}_12_meses_{object_name}_{region}.json"
        else:
            filename = f"previsao_{model}_{object_name}_{region}.json"

        filepath = FORECASTS_DIR / filename

        try:
            results[region] = load_forecast_from_json(filepath)
        except FileNotFoundError:
            logger.error(f"Arquivo não encontrado para região {region}: {filepath}")
            raise

    logger.info(f"Previsões carregadas para {len(results)} regiões")
    return results


def load_product_forecasts(
    model: str,
    regions: List[str],
    products: List[str],
    quantities: Optional[Dict[str, float]] = None,
    forecast_horizon: int = 12,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Carrega previsões de múltiplos produtos e regiões.

    Args:
        model (str): Nome do modelo (RNN, LSTM, CNN)
        regions (List[str]): Lista de regiões (ex: ['ilheus', 'itabuna'])
        products (List[str]): Lista de produtos (ex: ['arroz', 'feijao'])
        quantities (Dict[str, float], optional): Quantidades dos produtos na cesta.
                                                Se None, usa PRODUTOS_QUANTIDADES.
                                                Padrão: None.
        forecast_horizon (int, optional): Horizonte de previsão. Padrão: 12.

    Returns:
        Dict[str, Dict[str, List[float]]]: Dicionário aninhado:
            {região: {produto: [valores]}}
            Valores estão normalizados por quantidade e multiplicados por 1000.

    Raises:
        FileNotFoundError: Se arquivo de algum produto não existir
        ValueError: Se quantities não contiver produto necessário

    Examples:
        >>> forecasts = load_product_forecasts(
        ...     'RNN', ['ilheus'], ['arroz', 'feijao'], forecast_horizon=3
        ... )
        >>> forecasts
        {'ilheus': {'arroz': [2.5, 2.6, 2.7], 'feijao': [5.1, 5.2, 5.3]}}

    Notes:
        - Valores são divididos pela quantidade do produto
        - Depois multiplicados por 1000 para converter em reais
        - Usa PRODUTOS_QUANTIDADES por padrão
    """
    if quantities is None:
        quantities = PRODUTOS_QUANTIDADES
        logger.debug("Usando quantidades padrão de PRODUTOS_QUANTIDADES")

    results: Dict[str, Dict[str, List[float]]] = {region: {} for region in regions}

    logger.info(
        f"Carregando previsões de produtos: modelo={model}, "
        f"regiões={regions}, produtos={len(products)}, horizon={forecast_horizon}"
    )

    for region in regions:
        for product in products:
            if product not in quantities:
                raise ValueError(
                    f"Produto '{product}' não encontrado em quantities. "
                    f"Disponíveis: {list(quantities.keys())}"
                )

            if forecast_horizon == 12:
                filename = f"previsao_{model}_12_meses_{product}_{region}.json"
            else:
                filename = f"previsao_{model}_{product}_{region}.json"

            filepath = FORECASTS_PRODUTOS_DIR / region / filename

            try:
                # Carregar sem desnormalizar (já fizemos na função base)
                with open(filepath, "r", encoding="utf-8") as file:
                    data = json.load(file)

                # Suportar ambos formatos
                values_data = data[product]
                if isinstance(values_data, list):
                    values = [float(v) for v in values_data]
                elif isinstance(values_data, str):
                    values_str = values_data.replace("[", "").replace("]", "")
                    values = [float(v.strip()) for v in values_str.split(",")]
                else:
                    raise ValueError(f"Formato inesperado em {filepath}")

                # Normalizar por quantidade e converter para reais
                values = [v / quantities[product] * 1000 for v in values]
                results[region][product] = values

            except FileNotFoundError:
                logger.error(f"Arquivo não encontrado: {filepath}")
                raise
            except Exception as e:
                logger.error(f"Erro ao processar {filepath}: {e}")
                raise

    logger.info(
        f"Previsões carregadas: {len(regions)} regiões × {len(products)} produtos"
    )
    return results


def setup_plot_style(figsize: tuple = CHART_FIGSIZE) -> tuple[Figure, Axes]:
    """
    Cria e configura uma figura matplotlib.

    Args:
        figsize (tuple, optional): Tamanho da figura (largura, altura) em polegadas.
                                  Padrão: (15, 5).

    Returns:
        tuple[Figure, Axes]: Tupla contendo (figura, eixos)

    Examples:
        >>> fig, ax = setup_plot_style()
        >>> ax.plot([1, 2, 3], [1, 2, 3])
        >>> plt.show()

        >>> # Tamanho customizado
        >>> fig, ax = setup_plot_style(figsize=(12, 6))

    Notes:
        - Use ax para plotar e configurar o gráfico
        - Use fig.savefig() para salvar
    """
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def format_yticks_with_comma(
    ax: Axes,
    yticks_range: Union[range, List[float]],
    yticks_labels: Optional[List[str]] = None,
) -> None:
    """
    Formata ticks do eixo Y com vírgula como separador decimal.

    Args:
        ax (Axes): Eixo matplotlib a formatar
        yticks_range (range | List[float]): Posições dos ticks no eixo Y
        yticks_labels (List[str], optional): Labels customizados.
                                            Se None, gera automaticamente com vírgulas.
                                            Padrão: None.

    Examples:
        >>> fig, ax = setup_plot_style()
        >>> format_yticks_with_comma(ax, range(0, 100, 10))
        >>> # Ticks: 0,00  10,00  20,00  ...

        >>> # Labels customizados
        >>> format_yticks_with_comma(ax, [0, 50, 100], ['zero', 'meio', 'cem'])

    Notes:
        - Formato padrão: ".2f" (2 casas decimais)
        - Substitui ponto por vírgula (padrão BR)
        - Tamanho da fonte: 15
    """
    if yticks_labels is None:
        yticks_labels = [format(y, ".2f").replace(".", ",") for y in yticks_range]

    ax.set_yticks(list(yticks_range))
    ax.set_yticklabels(yticks_labels, size=15)


def add_forecast_annotation(
    ax: Axes, x: float, y: float, value: float, offset: float = 5, va: str = "bottom"
) -> None:
    """
    Adiciona anotação de valor a um ponto de previsão.

    Args:
        ax (Axes): Eixo matplotlib
        x (float): Coordenada X do ponto
        y (float): Coordenada Y do ponto
        value (float): Valor a exibir na anotação
        offset (float, optional): Deslocamento vertical em pontos. Padrão: 5.
        va (str, optional): Alinhamento vertical ('top' ou 'bottom'). Padrão: 'bottom'.

    Examples:
        >>> fig, ax = setup_plot_style()
        >>> ax.plot([1, 2, 3], [10, 20, 30], 'o')
        >>> add_forecast_annotation(ax, 2, 20, 20.5)
        >>> # Anotação "20,50" acima do ponto (2, 20)

        >>> # Anotação abaixo
        >>> add_forecast_annotation(ax, 3, 30, 30.2, va='top')

    Notes:
        - Formato: 2 casas decimais com vírgula
        - Tamanho da fonte: 13
        - Alinhamento horizontal: centralizado
    """
    y_position = y + offset if va == "bottom" else y - offset

    ax.annotate(
        format(value, ".2f").replace(".", ","),
        (x, y_position),
        size=13,
        va=va,
        ha="center",
    )


def save_figure(filename: Union[str, Path], dpi: int = CHART_DPI) -> None:
    """
    Salva a figura matplotlib atual.

    Args:
        filename (str | Path): Caminho do arquivo de saída.
                              Suporta: .png, .jpg, .pdf, .svg
        dpi (int, optional): Resolução em dots per inch. Padrão: 300.

    Raises:
        OSError: Se não for possível salvar o arquivo

    Examples:
        >>> fig, ax = setup_plot_style()
        >>> ax.plot([1, 2, 3], [1, 2, 3])
        >>> save_figure("meu_grafico.png")
        ✅ Gráfico salvo em: meu_grafico.png

        >>> # Alta resolução
        >>> save_figure("grafico_hd.png", dpi=600)

    Notes:
        - bbox_inches='tight' remove espaços em branco
        - DPI 300 é adequado para publicações
        - DPI 150 é suficiente para web
    """
    filename = Path(filename)

    try:
        # Criar diretório se não existir
        filename.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(fname=filename, dpi=dpi, bbox_inches="tight")
        logger.info(f"Gráfico salvo em: {filename}")
        print(f"✅ Gráfico salvo em: {filename}")

    except Exception as e:
        logger.error(f"Erro ao salvar gráfico: {e}")
        raise


def load_product_historical_data(
    data_folder: Union[str, Path],
    sub_pasta: str,
    region: str,
    products: List[str],
    quantities: Optional[Dict[str, float]] = None,
    n_months: int = 9,
) -> Dict[str, List[float]]:
    """
    Carrega dados históricos de produtos de arquivos Excel.

    Args:
        data_folder (str | Path): Caminho da pasta base de dados
        sub_pasta (str): Nome do subdiretório (ex: 'datasets_produtos/ilheus')
        region (str): Nome da região ('ilheus' ou 'itabuna')
        products (List[str]): Lista de produtos a carregar
        quantities (Dict[str, float], optional): Quantidades dos produtos.
                                                Se None, usa PRODUTOS_QUANTIDADES.
        n_months (int, optional): Número de últimos meses a carregar. Padrão: 9.

    Returns:
        Dict[str, List[float]]: Dicionário com produtos como chaves e
                               listas de valores normalizados por quantidade.

    Raises:
        FileNotFoundError: Se arquivo de algum produto não existir
        ValueError: Se produto não estiver em quantities

    Examples:
        >>> historico = load_product_historical_data(
        ...     "../data", "datasets_produtos/ilheus", "ilheus",
        ...     ["arroz", "feijao"], n_months=6
        ... )
        >>> historico
        {'arroz': [2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
         'feijao': [5.1, 5.2, 5.3, 5.4, 5.5, 5.6]}

    Notes:
        - Valores são divididos pela quantidade do produto
        - Lê arquivos Excel no formato: {produto}_{regiao}.xlsx
        - Espera coluna 'preco' no Excel
    """
    if quantities is None:
        quantities = PRODUTOS_QUANTIDADES
        logger.debug("Usando quantidades padrão de PRODUTOS_QUANTIDADES")

    path = Path(data_folder)
    data_path = path / sub_pasta

    if not data_path.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {data_path}")

    logger.info(
        f"Carregando histórico: região={region}, produtos={len(products)}, "
        f"meses={n_months}"
    )

    results = {}
    for product in products:
        if product not in quantities:
            raise ValueError(
                f"Produto '{product}' não encontrado em quantities. "
                f"Disponíveis: {list(quantities.keys())}"
            )

        file_path = data_path / f"{product}_{region}.xlsx"

        if not file_path.exists():
            logger.error(f"Arquivo não encontrado: {file_path}")
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        try:
            df = pd.read_excel(file_path)

            if "preco" not in df.columns:
                raise ValueError(f"Coluna 'preco' não encontrada em {file_path}")

            values = df["preco"].tail(n_months).to_list()
            # Normalize by quantity
            values = [v / quantities[product] for v in values]
            results[product] = values

            logger.debug(f"Produto '{product}' carregado: {len(values)} valores")

        except Exception as e:
            logger.error(f"Erro ao processar {file_path}: {e}")
            raise

    logger.info(f"Histórico carregado: {len(results)} produtos")
    return results


def plot_product_chart(
    ax: Axes,
    produtos: List[str],
    valores_reais: Dict[str, List[float]],
    previsoes: Dict[str, List[float]],
    markers: List[str],
    colors_real: List[str],
    colors_prev: List[str],
    markers_size: List[int],
    line_size: float,
    x_label: List[str],
    limite_y: int,
) -> None:
    """
    Plota gráfico de produtos com valores históricos e previsões.

    Cria visualização que conecta dados históricos (9 meses) com previsões
    (3 meses futuros), usando cores e marcadores diferentes.

    Args:
        ax (Axes): Eixo matplotlib onde plotar
        produtos (List[str]): Lista de nomes de produtos
        valores_reais (Dict[str, List[float]]): Valores históricos por produto
        previsoes (Dict[str, List[float]]): Previsões por produto
        markers (List[str]): Estilos de marcadores ('o', 's', '^', etc.)
        colors_real (List[str]): Cores para valores históricos
        colors_prev (List[str]): Cores para previsões
        markers_size (List[int]): Tamanhos dos marcadores
        line_size (float): Espessura das linhas
        x_label (List[str]): Labels do eixo X (12 meses)
        limite_y (int): Limite superior do eixo Y

    Examples:
        >>> config = PlotConfig(
        ...     produtos=["arroz", "feijao"],
        ...     valores_reais={"arroz": [2.5, 2.6, ..., 2.9],  # 9 valores
        ...                    "feijao": [5.1, 5.2, ..., 5.9]},
        ...     previsoes={"arroz": [3.0, 3.1, 3.2],  # 3 valores
        ...                "feijao": [6.0, 6.1, 6.2]},
        ...     markers=["o", "s"],
        ...     colors_real=["blue", "green"],
        ...     colors_prev=["red", "orange"],
        ...     markers_size=[8, 8],
        ...     line_size=2.0,
        ...     x_labels=["Jan", "Fev", ..., "Dez"],
        ...     limite_y=100
        ... )
        >>> fig, ax = setup_plot_style()
        >>> plot_product_chart(ax, **vars(config))  # Desempacotar dataclass
        >>> plt.show()

    Notes:
        - Histórico: meses 0-8 (linha sólida)
        - Conexão: meses 8-9 (linha pontilhada)
        - Previsão: meses 9-11 (linha pontilhada)
        - Grade habilitada no eixo Y
        - Aplica transformação "c" → "ç" nos rótulos
    """
    for i, produto in enumerate(produtos):
        # Connection line (liga último valor real ao primeiro previsto)
        ax.plot(
            [8, 9],
            [valores_reais[produto][-1], previsoes[produto][0]],
            ":",
            color=colors_prev[i],
            lw=line_size,
            markersize=markers_size[i],
        )

        # Historical values (meses 0-8)
        ax.plot(
            [x for x in range(9)],
            valores_reais[produto],
            marker=markers[i],
            color=colors_real[i],
            lw=line_size,
            markersize=markers_size[i],
        )

        # Forecasts (meses 9-11)
        ax.plot(
            [9, 10, 11],
            previsoes[produto],
            ":",
            marker=markers[i],
            label=produto.title().replace("c", "ç"),
            color=colors_prev[i],
            lw=line_size,
            markersize=markers_size[i],
        )

    # Formatting
    ax.set_xticks([x for x in range(0, 12)])
    ax.set_xticklabels(x_label, size=15)
    ax.set_ylabel("Custo (R$)", size=20)
    ax.set_yticks([y for y in range(0, limite_y, 10)])
    ax.set_yticklabels(
        [format(y, ".2f").replace(".", ",") for y in range(0, limite_y, 10)], size=15
    )
    ax.grid(which="major", axis="y", alpha=CHART_GRID_ALPHA)


def plot_forecast_only_chart(
    ax: Axes,
    produtos: List[str],
    previsoes: Dict[str, List[float]],
    markers: List[str],
    colors: List[str],
    markers_size: List[int],
    line_size: float,
    x_labels: List[str],
    yticks_range: Union[range, List[float]],
    yticks_labels: List[str],
) -> None:
    """
    Plota gráfico de produtos com apenas valores previstos (sem histórico).

    Visualização simplificada mostrando apenas as previsões de 12 meses,
    sem dados históricos. Útil para relatórios focados em futuro.

    Args:
        ax (Axes): Eixo matplotlib onde plotar
        produtos (List[str]): Lista de nomes de produtos
        previsoes (Dict[str, List[float]]): Previsões por produto (12 valores)
        markers (List[str]): Estilos de marcadores ('o', 's', '^', etc.)
        colors (List[str]): Cores para cada produto
        markers_size (List[int]): Tamanhos dos marcadores
        line_size (float): Espessura das linhas
        x_labels (List[str]): Labels do eixo X (12 meses)
        yticks_range (range | List[float]): Posições dos ticks Y
        yticks_labels (List[str]): Labels dos ticks Y (com vírgulas)

    Examples:
        >>> fig, ax = setup_plot_style()
        >>> previsoes = {
        ...     "arroz": [2.5, 2.6, ..., 3.6],  # 12 valores
        ...     "feijao": [5.0, 5.1, ..., 6.0]
        ... }
        >>> plot_forecast_only_chart(
        ...     ax, ["arroz", "feijao"], previsoes,
        ...     ["o", "s"], ["blue", "green"], [8, 8], 2.0,
        ...     ["Jan", "Fev", ..., "Dez"],
        ...     range(0, 10), ["0,00", "1,00", ..., "9,00"]
        ... )
        >>> plt.legend()
        >>> plt.show()

    Notes:
        - Todas as linhas são tracejadas (ls="--")
        - Grade habilitada no eixo Y
        - Aplica transformação "c" → "ç" nos rótulos
        - Ideal para previsões de 12 meses
    """
    for produto, mark, color, marker_size in zip(
        produtos, markers, colors, markers_size
    ):
        ax.plot(
            range(12),
            previsoes[produto],
            label=produto.title().replace("c", "ç"),
            color=color,
            marker=mark,
            ls="--",
            markersize=marker_size,
            lw=line_size,
        )

    # Formatting
    ax.set_xticks(range(12))
    ax.set_xticklabels(x_labels, size=13)
    ax.set_yticks(list(yticks_range))
    ax.set_yticklabels(yticks_labels, size=13)
    ax.set_ylabel("Preço dos produtos da cesta básica (R$)", size=18)
    ax.grid(which="major", axis="y", alpha=CHART_GRID_ALPHA)
