#!/usr/bin/env python3
"""
Manual monthly price entry.

How to use:
1. Change MES_REFERENCIA to the month you are adding, in YYYY-MM format.
2. Fill every price in PRECOS_MENSAIS.
3. Run: uv run atualizar-precos

The script updates ``data/precos_mensais.xlsx``.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import MASTER_DATA_FILE  # noqa: E402
from utils.monthly_data import (  # noqa: E402
    add_or_update_prices,
    load_master_table,
    normalize_month,
    save_master_table,
    validate_complete_month,
)


# Change this value every month. Use YYYY-MM; the script will store the real
# last day of that month, for example 2026-02 -> 2026-02-28.
MES_REFERENCIA = "2026-05"


# Fill the prices below. Use numbers such as 12.34, or strings such as "12,34".
# Keep every product key so the monthly update is complete and auditable.
PRECOS_MENSAIS = {
    "ilheus": {
        "cesta_basica": None,
        "acucar": None,
        "arroz": None,
        "banana": None,
        "cafe": None,
        "carne": None,
        "farinha": None,
        "feijao": None,
        "leite": None,
        "manteiga": None,
        "oleo": None,
        "pao": None,
        "tomate": None,
    },
    "itabuna": {
        "cesta_basica": None,
        "acucar": None,
        "arroz": None,
        "banana": None,
        "cafe": None,
        "carne": None,
        "farinha": None,
        "feijao": None,
        "leite": None,
        "manteiga": None,
        "oleo": None,
        "pao": None,
        "tomate": None,
    },
}


def main() -> None:
    try:
        _main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Erro: {exc}") from None


def _main() -> None:
    validate_complete_month(PRECOS_MENSAIS)

    df = load_master_table(MASTER_DATA_FILE)
    df = add_or_update_prices(df, MES_REFERENCIA, PRECOS_MENSAIS)
    save_master_table(df, MASTER_DATA_FILE)

    month_date = normalize_month(MES_REFERENCIA)
    rows_for_month = df.loc[
        df["data"].apply(
            lambda value: value.year == month_date.year
            and value.month == month_date.month
        )
    ]

    print(f"Mes atualizado: {month_date:%Y-%m}")
    print(f"Registros do mes: {len(rows_for_month)}")
    print(f"Tabela unica salva em: {MASTER_DATA_FILE}")


if __name__ == "__main__":
    main()
