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
    """Return the real last day for a month."""
    return date(year, month, calendar.monthrange(year, month)[1])


def month_start(year: int, month: int) -> date:
    """Return the first day for a month."""
    return date(year, month, 1)


def month_date_for_product(year: int, month: int, product: str) -> date:
    """Return the project date convention for a product type."""
    if product == PRODUCT_CESTA_BASICA:
        return month_start(year, month)

    return month_end(year, month)


def normalize_city(value: object) -> str:
    """Normalize city names used by the monthly table."""
    return _normalize_city(value)


def normalize_product(value: object) -> str:
    """Normalize product names used by the monthly table."""
    return _normalize_product(value)


def parse_month_date(value: object, product: str) -> date:
    """
    Parse dates from the project spreadsheets and normalize by product type.

    The basket data uses the first day of the month. Product data uses the real
    last day of the month. Invalid legacy labels such as ``31-02-2026`` are
    accepted because only the month and year are needed.
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
    """Normalize a user-provided month/date by product type."""
    product_key = _normalize_product(product)

    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}", value.strip()):
        year, month = value.split("-")
        return month_date_for_product(int(year), int(month), product_key)

    return parse_month_date(value, product_key)


def normalize_price(value: object) -> float:
    """Convert manual/Excel price values to float."""
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
    """Load the unified monthly table."""
    if not path.exists():
        raise FileNotFoundError(f"Tabela unica nao encontrada: {path}")

    return clean_master_dataframe(pd.read_excel(path))


def save_master_table(df: pd.DataFrame, path: Path = MASTER_DATA_FILE) -> Path:
    """Save the unified monthly table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_excel_with_date_format(clean_master_dataframe(df), path)
    return path


def add_or_update_prices(
    df: pd.DataFrame,
    month: str | date | datetime | pd.Timestamp,
    prices: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """
    Insert or replace prices in the unified table.

    Existing rows with the same date/city/product are replaced, which makes the
    monthly update script safe to rerun after correcting a value.
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
    incoming_keys = incoming[key_columns].apply(tuple, axis=1)
    current_keys = current[key_columns].apply(tuple, axis=1)
    current = current.loc[~current_keys.isin(set(incoming_keys))]

    return clean_master_dataframe(pd.concat([current, incoming], ignore_index=True))


def validate_complete_month(
    prices: dict[str, dict[str, object]],
    *,
    required_cities: Iterable[str] = REGIOES,
    required_products: Iterable[str] = (PRODUCT_CESTA_BASICA, *PRODUTOS),
) -> None:
    """Ensure the monthly manual payload has every expected city/product."""
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
    """Normalize columns, dates, labels, prices, sorting, and duplicates."""
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
    """Load one city/product price series from the unified table."""
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
