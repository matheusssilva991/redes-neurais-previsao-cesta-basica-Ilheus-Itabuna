#!/usr/bin/env python3
"""
Insercao manual de um novo mes em ``data/precos_mensais.xlsx``.

Quando usar:
- Use este script se ainda nao existir um boletim em ``.docx`` pronto para
  importacao automatica, ou se voce quiser corrigir um valor manualmente.
- O script nao baixa arquivos e nao le tabelas externas. Ele usa somente o
  mes em ``MES_REFERENCIA`` e os valores preenchidos em ``PRECOS_MENSAIS``.

Fluxo:
1. Ajuste ``MES_REFERENCIA`` no formato ``YYYY-MM``.
2. Preencha todos os precos de Ilheus e Itabuna em ``PRECOS_MENSAIS``.
3. Rode ``uv run atualizar-precos``.

Comportamento:
- O arquivo ``data/precos_mensais.xlsx`` e carregado.
- Cada combinacao de data/cidade/produto do mes informado e inserida ou
  substituida.
- Se o mes ja existir, os registros antigos daquele mes sao trocados pelos
  novos, evitando duplicatas.
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


# Mes de referencia da atualizacao manual.
# Use sempre YYYY-MM. A funcao de persistencia converte esse rotulo para a
# convencao do projeto:
# - cesta_basica => primeiro dia do mes
# - demais produtos => ultimo dia real do mes
MES_REFERENCIA = "2026-05"


# Preencha todos os valores do mes.
# Aceita float/int (12.34) ou string com virgula ("12,34").
# Nao remova chaves: a validacao exige o mes completo para manter a tabela
# auditavel e consistente entre cidades e produtos.
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
    # Garante que o payload manual esteja completo antes de tocar no Excel.
    validate_complete_month(PRECOS_MENSAIS)

    # Carrega a tabela historica, aplica o mes informado e salva a versao limpa.
    df = load_master_table(MASTER_DATA_FILE)
    df = add_or_update_prices(df, MES_REFERENCIA, PRECOS_MENSAIS)
    save_master_table(df, MASTER_DATA_FILE)

    month_date = normalize_month(MES_REFERENCIA)
    # Resume apenas o recorte do mes atualizado para facilitar a conferencia.
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
