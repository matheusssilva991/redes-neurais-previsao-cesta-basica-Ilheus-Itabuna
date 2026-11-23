"""
Utility functions for chart generation.

Este módulo fornece funções para:
- Carregar previsões de arquivos JSON
- Configurar e formatar gráficos matplotlib
- Plotar dados históricos e previsões
- Salvar figuras em alta resolução
"""

from typing import Dict, List, Optional, Union, Tuple
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
from .logger import setup_logger
from .monthly_data import load_price_series

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


def extract_dates_from_dataframe(df: pd.DataFrame) -> Optional[List[pd.Timestamp]]:
    """Extrai as datas da série e as desloca para o primeiro dia do mês.

    Args:
        df: DataFrame com a coluna `data` ou, na ausência dela, a coluna
            histórica `ano`.

    Returns:
        Datas na ordem das linhas, ou `None` se faltar a coluna ou se alguma
        data for inválida.
    """
    date_column = (
        "data" if "data" in df.columns else "ano" if "ano" in df.columns else None
    )
    if date_column is None:
        return None

    parsed = pd.to_datetime(df[date_column], errors="coerce")
    if not parsed.notna().all():
        return None

    return [date.replace(day=1) for date in parsed.tolist()]


def derive_month_labels_from_history(
    df: pd.DataFrame,
    hist_months: int,
    forecast_horizon: int,
    month_labels: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], int]:
    """Gera os rótulos dos meses históricos e previstos para um gráfico.

    Se a série não tiver datas válidas suficientes, considera o mês atual
    como o último mês histórico.

    Args:
        df: Série histórica em ordem cronológica.
        hist_months: Quantidade de meses históricos, maior que zero.
        forecast_horizon: Quantidade de meses previstos, maior que zero.
        month_labels: Rótulos de janeiro a dezembro. O padrão usa os nomes
            configurados em português.

    Returns:
        Tupla com os rótulos históricos, os rótulos futuros e o ano do último
        mês previsto.
    """
    if month_labels is None:
        month_labels = CHART_MONTH_LABELS_PT

    dates = extract_dates_from_dataframe(df)
    if dates and len(dates) >= hist_months:
        history_dates = dates[-hist_months:]
    else:
        last_complete_month = pd.Timestamp.today().replace(day=1)
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


def _decode_forecast_values(values_data: List[float | str] | str) -> List[float]:
    """Converte um array nativo ou uma lista legada em texto para floats."""
    if isinstance(values_data, list):
        return [float(value) for value in values_data]

    try:
        # A conversão direta para float preserva o sinal de inteiros como -0.
        decoded = json.loads(values_data, parse_int=float)
    except (json.JSONDecodeError, RecursionError):
        decoded = None

    if (
        isinstance(decoded, list)
        and decoded
        and all(isinstance(value, float) for value in decoded)
    ):
        return decoded

    # Arquivos antigos também podem conter números aceitos por float fora do JSON.
    values_str = values_data.replace("[", "").replace("]", "")
    return [float(value.strip()) for value in values_str.split(",")]


def load_forecast_from_json(
    filepath: Union[str, Path],
    denormalize: bool = True,
    normalization_factor: int = DEFAULT_NORMALIZATION_FACTOR,
) -> List[float]:
    """Carrega as previsões da primeira chave de um JSON.

    Aceita arrays JSON e o formato legado de lista gravada como texto.

    Args:
        filepath: Caminho do arquivo JSON.
        denormalize: Se verdadeiro, converte os valores para reais.
        normalization_factor: Multiplicador usado na conversão para reais.

    Returns:
        Lista de previsões na escala solicitada.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o JSON for inválido ou vazio, o valor não for uma
            lista ou texto, ou algum valor não puder ser convertido em float.
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

    if not isinstance(values_data, (list, str)):
        raise ValueError(
            f"Formato inesperado em {filepath}: esperado lista ou string, "
            f"recebido {type(values_data)}"
        )

    values = _decode_forecast_values(values_data)
    value_format = "array nativo" if isinstance(values_data, list) else "string legado"
    logger.debug(f"Formato JSON: {value_format} ({len(values)} valores)")

    if denormalize:
        values = [v * normalization_factor for v in values]

    logger.info(f"Previsões carregadas: {len(values)} valores de {filepath.name}")
    return values


def load_forecasts_for_regions(
    model: str, object_name: str, regions: List[str], forecast_horizon: int = 3
) -> Dict[str, List[float]]:
    """Carrega as previsões em reais para uma série em várias cidades.

    Args:
        model: Identificador da arquitetura, como `RNN`.
        object_name: Identificador da série no nome do arquivo.
        regions: Cidades cujos arquivos devem ser lidos.
        forecast_horizon: Horizonte em meses. Apenas 12 seleciona um nome
            com a marca `12_meses`.

    Returns:
        Dicionário com a lista de previsões em reais de cada cidade.

    Raises:
        FileNotFoundError: Se faltar o arquivo de uma das cidades.
        ValueError: Se algum arquivo tiver previsões inválidas.
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
    """Carrega previsões de produtos em reais por unidade de referência.

    O carregador aceita arrays JSON e listas gravadas como texto. Para
    cada produto, divide a previsão pela quantidade e multiplica por 1000.

    Args:
        model: Identificador da arquitetura, como `RNN`.
        regions: Cidades a carregar.
        products: Identificadores dos produtos a carregar.
        quantities: Quantidades positivas por produto na unidade adotada
            para o gráfico. Por padrão, usa `PRODUTOS_QUANTIDADES`.
        forecast_horizon: Horizonte em meses. Apenas 12 seleciona um nome
            com a marca `12_meses`.

    Returns:
        Dicionário `{cidade: {produto: [valores]}}`, na ordem das previsões.

    Raises:
        FileNotFoundError: Se faltar um arquivo de previsão.
        KeyError: Se a chave do produto estiver ausente no JSON.
        ValueError: Se faltar uma quantidade ou houver previsões inválidas.
        ZeroDivisionError: Se uma quantidade for zero.
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
                with open(filepath, "r", encoding="utf-8") as file:
                    data = json.load(file)

                values_data = data[product]
                if not isinstance(values_data, (list, str)):
                    raise ValueError(f"Formato inesperado em {filepath}")

                values = _decode_forecast_values(values_data)
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
    """Cria uma figura Matplotlib com um único eixo.

    Args:
        figsize: Largura e altura da figura, em polegadas.

    Returns:
        Tupla com a figura e o eixo criados.
    """
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def format_yticks_with_comma(
    ax: Axes,
    yticks_range: Union[range, List[float]],
    yticks_labels: Optional[List[str]] = None,
) -> None:
    """Configura as marcações e os rótulos do eixo Y.

    Args:
        ax: Eixo Matplotlib a modificar.
        yticks_range: Posições das marcações no eixo Y.
        yticks_labels: Rótulos na mesma ordem das posições. Quando omitidos,
            os valores são formatados com duas casas decimais e vírgula.
    """
    if yticks_labels is None:
        yticks_labels = [format(y, ".2f").replace(".", ",") for y in yticks_range]

    ax.set_yticks(list(yticks_range))
    ax.set_yticklabels(yticks_labels, size=15)


def add_forecast_annotation(
    ax: Axes,
    x: float,
    y: float,
    value: float,
    offset: float = 8,
    va: str = "bottom",
    x_offset: float = 0,
    color: str = "black",
    boxed: bool = False,
) -> None:
    """Anota um ponto do gráfico com um valor de duas casas decimais.

    Args:
        ax: Eixo Matplotlib a modificar.
        x: Coordenada horizontal do ponto, em unidades de dados.
        y: Coordenada vertical do ponto, em unidades de dados.
        value: Valor mostrado na anotação, com vírgula decimal.
        offset: Distância vertical em pontos.
        va: Alinhamento vertical. Use `bottom` para deslocar o texto para
            cima e `top` para baixo.
        x_offset: Deslocamento horizontal em pontos.
        color: Cor do texto.
        boxed: Se verdadeiro, adiciona um fundo branco semitransparente.
    """
    y_offset = offset if va == "bottom" else -offset
    bbox = None
    if boxed:
        bbox = {
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
        }

    ax.annotate(
        format(value, ".2f").replace(".", ","),
        xy=(x, y),
        xytext=(x_offset, y_offset),
        textcoords="offset points",
        size=13,
        va=va,
        ha="center",
        color=color,
        bbox=bbox,
    )


def save_figure(filename: Union[str, Path], dpi: int = CHART_DPI) -> None:
    """Grava a figura ativa do Matplotlib com margens ajustadas ao conteúdo.

    Cria a pasta de destino e imprime o caminho gravado.

    Args:
        filename: Caminho de saída, com uma extensão aceita pelo Matplotlib.
        dpi: Resolução da figura em pontos por polegada.

    Raises:
        OSError: Se a pasta ou o arquivo não puder ser gravado.
        ValueError: Se o formato da figura não for aceito.
    """
    filename = Path(filename)

    try:
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
    """Carrega os preços históricos de produtos da planilha mensal.

    Args:
        data_folder: Caminho legado, aceito por compatibilidade e ignorado.
        sub_pasta: Subpasta legada, aceita por compatibilidade e ignorada.
        region: Cidade da série, como `ilheus` ou `itabuna`.
        products: Identificadores dos produtos a carregar.
        quantities: Quantidades positivas por produto na unidade adotada
            para o gráfico. Por padrão, usa `PRODUTOS_QUANTIDADES`.
        n_months: Número de observações mais recentes a retornar.

    Returns:
        Dicionário com os valores históricos de cada produto em reais,
        divididos pela quantidade correspondente e em ordem cronológica.

    Raises:
        FileNotFoundError: Se a planilha unificada não existir.
        ValueError: Se faltar uma quantidade, a série estiver vazia ou a
            cidade, o produto ou os dados da planilha forem inválidos.
        ZeroDivisionError: Se uma quantidade for zero.
    """
    if quantities is None:
        quantities = PRODUTOS_QUANTIDADES
        logger.debug("Usando quantidades padrão de PRODUTOS_QUANTIDADES")

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

        try:
            df = load_price_series(region, product)
            values = df["preco"].tail(n_months).to_list()
            values = [v / quantities[product] for v in values]
            results[product] = values

            logger.debug(f"Produto '{product}' carregado: {len(values)} valores")

        except Exception as e:
            logger.error(f"Erro ao processar {region}/{product}: {e}")
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
    """Desenha nove meses de histórico e três de previsão para cada produto.

    Uma linha pontilhada liga o último mês observado ao primeiro previsto.

    Args:
        ax: Eixo Matplotlib a modificar.
        produtos: Identificadores dos produtos, na ordem das listas de estilo.
        valores_reais: Nove valores históricos em reais por produto.
        previsoes: Três valores previstos em reais por produto.
        markers: Marcadores Matplotlib na ordem de `produtos`.
        colors_real: Cores do histórico na ordem de `produtos`.
        colors_prev: Cores da previsão na ordem de `produtos`.
        markers_size: Tamanhos dos marcadores na ordem de `produtos`.
        line_size: Espessura das linhas.
        x_label: Doze rótulos dos meses, começando pelo histórico.
        limite_y: Limite superior exclusivo das marcações do eixo Y.
    """
    for i, produto in enumerate(produtos):
        ax.plot(
            [8, 9],
            [valores_reais[produto][-1], previsoes[produto][0]],
            ":",
            color=colors_prev[i],
            lw=line_size,
            markersize=markers_size[i],
        )

        ax.plot(
            [x for x in range(9)],
            valores_reais[produto],
            marker=markers[i],
            color=colors_real[i],
            lw=line_size,
            markersize=markers_size[i],
        )

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
    """Desenha os doze meses de previsão de cada produto.

    Args:
        ax: Eixo Matplotlib a modificar.
        produtos: Identificadores dos produtos a desenhar.
        previsoes: Doze valores previstos em reais por produto.
        markers: Marcadores Matplotlib na ordem de `produtos`.
        colors: Cores na ordem de `produtos`.
        markers_size: Tamanhos dos marcadores na ordem de `produtos`.
        line_size: Espessura das linhas.
        x_labels: Doze rótulos dos meses previstos.
        yticks_range: Posições das marcações do eixo Y.
        yticks_labels: Rótulos formatados na ordem das posições do eixo Y.
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

    ax.set_xticks(range(12))
    ax.set_xticklabels(x_labels, size=13)
    ax.set_yticks(list(yticks_range))
    ax.set_yticklabels(yticks_labels, size=13)
    ax.set_ylabel("Preço dos produtos da cesta básica (R$)", size=18)
    ax.grid(which="major", axis="y", alpha=CHART_GRID_ALPHA)
