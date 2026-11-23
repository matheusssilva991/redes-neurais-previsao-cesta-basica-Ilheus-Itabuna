"""
Helpers for the unified monthly price table.

The project stores all monthly values in one table:
``data/precos_mensais.xlsx``.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from openpyxl.utils.datetime import from_excel

from config import MASTER_DATA_FILE, PRODUTOS, REGIOES

PRODUCT_CESTA_BASICA = "cesta_basica"
MASTER_COLUMNS = ["data", "cidade", "produto", "preco"]


def month_end(year: int, month: int) -> date:
    """Retorna o último dia do mês, considerando anos bissextos.

    Args:
        year: Ano civil.
        month: Número do mês, de 1 a 12.

    Returns:
        Data do último dia do mês.

    Raises:
        ValueError: Se o mês estiver fora de 1 a 12 ou o ano for inválido.
    """
    return date(year, month, calendar.monthrange(year, month)[1])


def month_start(year: int, month: int) -> date:
    """Retorna o primeiro dia do mês para o ano e mês informados.

    Args:
        year: Ano civil.
        month: Número do mês, de 1 a 12.

    Returns:
        Data do primeiro dia do mês.

    Raises:
        ValueError: Se o ano ou o mês for inválido.
    """
    return date(year, month, 1)


def month_date_for_product(year: int, month: int, product: str) -> date:
    """Define a data usada para registrar uma série mensal.

    Args:
        year: Ano de referência.
        month: Número do mês, de 1 a 12.
        product: Identificador do produto já normalizado.

    Returns:
        Primeiro dia do mês para `cesta_basica` ou último dia do mês para
        os demais produtos.

    Raises:
        ValueError: Se o ano ou o mês for inválido.
    """
    if product == PRODUCT_CESTA_BASICA:
        return month_start(year, month)

    return month_end(year, month)


def normalize_city(value: object) -> str:
    """Converte um nome de cidade no identificador usado na planilha.

    Args:
        value: Nome ou identificador, como `Ilhéus` ou `itabuna`.

    Returns:
        Identificador `ilheus` ou `itabuna`.

    Raises:
        ValueError: Se a cidade for desconhecida.
    """
    return _normalize_city(value)


def normalize_product(value: object) -> str:
    """Converte o nome de um produto no identificador usado na planilha.

    Args:
        value: Nome ou identificador, como `café` ou `cesta básica`.

    Returns:
        Identificador de um produto da cesta ou `cesta_basica` para o total.

    Raises:
        ValueError: Se o produto for desconhecido.
    """
    return _normalize_product(value)


def parse_month_date(value: object, product: str) -> date:
    """Interpreta uma data da planilha seguindo a convenção da série.

    O dia original é descartado. Isso permite aceitar rótulos antigos
    como `31-02-2026`, desde que o mês e o ano sejam válidos.

    Args:
        value: Data, texto com data ou número serial do Excel que represente
            uma data.
        product: Produto ou total da cesta, no nome original ou normalizado.

    Returns:
        Primeiro dia do mês para o total da cesta ou último dia para produtos.

    Raises:
        ValueError: Se a data estiver vazia, não puder ser interpretada ou
            o produto for inválido.
    """
    product_key = _normalize_product(product)

    if pd.isna(value):
        raise ValueError("Data vazia encontrada")

    if isinstance(value, int | float):
        parsed = from_excel(value)
        return month_date_for_product(parsed.year, parsed.month, product_key)

    if isinstance(value, pd.Timestamp):
        return month_date_for_product(value.year, value.month, product_key)

    if isinstance(value, datetime):
        return month_date_for_product(value.year, value.month, product_key)

    if isinstance(value, date):
        return month_date_for_product(value.year, value.month, product_key)

    text = str(value).strip()

    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if pd.notna(parsed):
        return month_date_for_product(parsed.year, parsed.month, product_key)

    match = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", text)
    if match:
        month = int(match.group(2))
        year = int(match.group(3))
        return month_date_for_product(year, month, product_key)

    match = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        return month_date_for_product(year, month, product_key)

    raise ValueError(f"Nao foi possivel interpretar a data: {value!r}")


def normalize_month(
    value: str | date | datetime | pd.Timestamp,
    product: str = PRODUCT_CESTA_BASICA,
) -> date:
    """Converte o mês informado em uma data compatível com a série.

    Args:
        value: Rótulo `YYYY-MM`, data ou texto com data completa.
        product: Produto da série. O padrão representa o total da cesta.

    Returns:
        Primeiro dia do mês para o total da cesta ou último dia para produtos.

    Raises:
        ValueError: Se o mês, a data ou o produto for inválido.
    """
    product_key = _normalize_product(product)

    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}", value.strip()):
        year, month = value.split("-")
        return month_date_for_product(int(year), int(month), product_key)

    return parse_month_date(value, product_key)


def normalize_price(value: object) -> float:
    """Converte um preço informado na planilha ou manualmente.

    Args:
        value: Número ou texto numérico, como `12,34` ou `1.234,56`.

    Returns:
        Preço em reais como float, arredondado a duas casas decimais.

    Raises:
        ValueError: Se o preço estiver vazio ou se o texto não for numérico.
    """
    if value is None or pd.isna(value):
        raise ValueError("Preco vazio encontrado")

    if isinstance(value, str):
        value = value.strip()
        if "," in value and "." in value:
            value = value.replace(".", "").replace(",", ".")
        elif "," in value:
            value = value.replace(",", ".")

    return round(float(value), 2)


def load_master_table(path: Path = MASTER_DATA_FILE) -> pd.DataFrame:
    """Lê e normaliza a planilha unificada de preços.

    Args:
        path: Caminho do Excel com as colunas `data`, `cidade`,
            `produto` e `preco`.

    Returns:
        Tabela mensal ordenada e sem duplicatas da chave data/cidade/produto.

    Raises:
        FileNotFoundError: Se a planilha não existir.
        ValueError: Se faltarem colunas ou houver valores inválidos.
    """
    if not path.exists():
        raise FileNotFoundError(f"Tabela unica nao encontrada: {path}")

    return clean_master_dataframe(pd.read_excel(path))


def save_master_table(df: pd.DataFrame, path: Path = MASTER_DATA_FILE) -> Path:
    """Normaliza a tabela mensal e grava a planilha Excel.

    Cria a pasta de destino quando necessário e sobrescreve a planilha.

    Args:
        df: Tabela com as colunas `data`, `cidade`, `produto` e `preco`.
        path: Caminho da planilha de destino.

    Returns:
        O caminho recebido em `path`.

    Raises:
        ValueError: Se a tabela tiver colunas ausentes ou valores inválidos.
        OSError: Se a pasta ou a planilha não puder ser gravada.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_excel_with_date_format(clean_master_dataframe(df), path)
    return path


def add_or_update_prices(
    df: pd.DataFrame,
    month: str | date | datetime | pd.Timestamp,
    prices: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """Insere ou substitui preços de um mês na tabela unificada.

    Registros com a mesma chave data/cidade/produto são substituídos,
    permitindo reaplicar uma atualização após corrigir os valores.

    Args:
        df: Tabela mensal atual com as colunas de `MASTER_COLUMNS`.
        month: Mês de referência, em formato `YYYY-MM` ou como data.
        prices: Mapeamento `{cidade: {produto: preço}}` com os valores em reais.

    Returns:
        Novo DataFrame normalizado e ordenado, com as atualizações aplicadas.

    Raises:
        ValueError: Se nenhum preço for fornecido ou se os dados forem inválidos.
    """
    new_rows: list[dict[str, object]] = []

    for city, product_prices in prices.items():
        city_key = _normalize_city(city)
        for product, price in product_prices.items():
            product_key = _normalize_product(product)
            month_date = normalize_month(month, product_key)
            new_rows.append(
                {
                    "data": month_date,
                    "cidade": city_key,
                    "produto": product_key,
                    "preco": normalize_price(price),
                }
            )

    if not new_rows:
        raise ValueError("Nenhum preco foi informado")

    incoming = pd.DataFrame(new_rows, columns=MASTER_COLUMNS)
    current = clean_master_dataframe(df)
    key_columns = ["data", "cidade", "produto"]
    incoming_keys = pd.MultiIndex.from_frame(incoming[key_columns])
    current_keys = pd.MultiIndex.from_frame(current[key_columns])
    current = current.loc[~current_keys.isin(incoming_keys)]

    return clean_master_dataframe(pd.concat([current, incoming], ignore_index=True))


def validate_complete_month(
    prices: dict[str, dict[str, object]],
    *,
    required_cities: Iterable[str] = REGIOES,
    required_products: Iterable[str] = (PRODUCT_CESTA_BASICA, *PRODUTOS),
) -> None:
    """Verifica a presença dos preços obrigatórios da atualização mensal.

    Args:
        prices: Mapeamento `{cidade: {produto: preço}}` com chaves normalizadas.
        required_cities: Identificadores das cidades que precisam estar presentes.
        required_products: Produtos esperados em cada cidade, incluindo o total
            da cesta por padrão.

    Raises:
        ValueError: Se um preço obrigatório estiver ausente ou for `None`.
    """
    missing: list[str] = []

    for city in required_cities:
        product_prices = prices.get(city, {})
        for product in required_products:
            if product not in product_prices or product_prices[product] is None:
                missing.append(f"{city}/{product}")

    if missing:
        formatted = "\n".join(f"- {item}" for item in missing)
        raise ValueError(f"Preencha os precos ausentes antes de rodar:\n{formatted}")


def clean_master_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza e ordena uma tabela mensal de preços.

    Args:
        df: DataFrame com as colunas `data`, `cidade`, `produto` e `preco`.

    Returns:
        Cópia com as quatro colunas normalizadas, ordenada por data,
        cidade e produto. Para chaves repetidas, mantém a última linha
        após a ordenação.

    Raises:
        ValueError: Se faltarem colunas ou houver valores inválidos.
    """
    missing_columns = set(MASTER_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Colunas ausentes na tabela unica: {sorted(missing_columns)}")

    clean_df = df.loc[:, MASTER_COLUMNS].copy()
    clean_df["cidade"] = clean_df["cidade"].apply(_normalize_city)
    clean_df["produto"] = clean_df["produto"].apply(_normalize_product)
    clean_df["data"] = clean_df.apply(
        lambda row: parse_month_date(row["data"], row["produto"]),
        axis=1,
    )
    clean_df["preco"] = clean_df["preco"].apply(normalize_price)
    clean_df = clean_df.sort_values(["data", "cidade", "produto"])
    clean_df = clean_df.drop_duplicates(["data", "cidade", "produto"], keep="last")
    clean_df = clean_df.reset_index(drop=True)
    return clean_df


def load_price_series(
    city: str,
    product: str,
    path: Path = MASTER_DATA_FILE,
) -> pd.DataFrame:
    """Carrega uma série de preços mensais da planilha unificada.

    Args:
        city: Nome ou identificador da cidade.
        product: Nome ou identificador do produto ou do total da cesta.
        path: Caminho da planilha unificada.

    Returns:
        DataFrame em ordem cronológica, com as colunas `data` e `preco`
        e valores em reais.

    Raises:
        FileNotFoundError: Se a planilha não existir.
        ValueError: Se a cidade ou o produto for inválido, a tabela não
            puder ser normalizada ou a série estiver vazia.
    """
    city_key = normalize_city(city)
    product_key = normalize_product(product)
    df = load_master_table(path)
    filtered = df.loc[
        (df["cidade"] == city_key) & (df["produto"] == product_key), ["data", "preco"]
    ]
    if filtered.empty:
        raise ValueError(f"Serie sem dados: cidade={city_key}, produto={product_key}")

    return filtered.sort_values("data").reset_index(drop=True)


def _save_excel_with_date_format(df: pd.DataFrame, path: Path) -> None:
    """Grava uma cópia da tabela no Excel com datas no formato yyyy-mm-dd.

    Args:
        df: Tabela a exportar, com `data` como primeira coluna na tabela mensal.
        path: Caminho da planilha a criar ou sobrescrever.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    output = df.copy()

    for column in ("data", "ano"):
        if column in output.columns:
            output[column] = pd.to_datetime(output[column])

    with pd.ExcelWriter(
        path,
        engine="openpyxl",
        date_format="yyyy-mm-dd",
        datetime_format="yyyy-mm-dd",
    ) as writer:
        output.to_excel(writer, index=False)
        worksheet = writer.sheets["Sheet1"]
        for column in ("A",):
            for cell in worksheet[column][1:]:
                cell.number_format = "yyyy-mm-dd"


def _normalize_city(value: object) -> str:
    """Valida o nome da cidade em `value` e retorna seu identificador."""
    city = str(value).strip().lower()
    aliases = {
        "ilheus": "ilheus",
        "ilhéus": "ilheus",
        "itabuna": "itabuna",
    }
    if city not in aliases:
        raise ValueError(f"Cidade invalida: {value!r}")
    return aliases[city]


def _normalize_product(value: object) -> str:
    """Valida o nome do produto em `value` e retorna seu identificador."""
    product = str(value).strip().lower()
    aliases = {
        "cesta": PRODUCT_CESTA_BASICA,
        "cesta_basica": PRODUCT_CESTA_BASICA,
        "cesta basica": PRODUCT_CESTA_BASICA,
        "cesta básica": PRODUCT_CESTA_BASICA,
        "óleo": "oleo",
        "oleo": "oleo",
        "pão": "pao",
        "pao": "pao",
        "açucar": "acucar",
        "açúcar": "acucar",
        "acucar": "acucar",
        "café": "cafe",
        "cafe": "cafe",
        "feijão": "feijao",
        "feijao": "feijao",
    }
    product = aliases.get(product, product)

    allowed_products = {PRODUCT_CESTA_BASICA, *PRODUTOS}
    if product not in allowed_products:
        raise ValueError(f"Produto invalido: {value!r}")
    return product
