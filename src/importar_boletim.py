#!/usr/bin/env python3
"""
Importa um novo mes a partir de um boletim salvo em ``previsoes_boletim``.

Como o fluxo funciona hoje:
- O comando procura um arquivo de boletim no caminho informado.
- Se voce passar uma pasta, ele escolhe o boletim mais recente encontrado ali.
- A importacao automatica suporta somente ``.docx``.
- O parser le o XML interno do DOCX e extrai:
  - a linha da cesta basica para o mes desejado;
  - a tabela de produtos de Ilheus;
  - a tabela de produtos de Itabuna.
- Sem ``--aplicar`` o comando mostra apenas uma previa.
- Com ``--aplicar`` os valores extraidos sao inseridos em
  ``data/precos_mensais.xlsx``.

Importante: este projeto nao grava diretamente em banco SQL. O "banco" usado
no fluxo atual e a planilha unificada ``data/precos_mensais.xlsx``, que depois
alimenta treino, previsoes e graficos.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import MASTER_DATA_FILE, PRODUTOS  # noqa: E402
from utils.boletim_data import BOLETIM_DIR, extract_prices_from_boletim  # noqa: E402
from utils.monthly_data import (  # noqa: E402
    PRODUCT_CESTA_BASICA,
    add_or_update_prices,
    load_master_table,
    save_master_table,
)


def main() -> None:
    try:
        args = _parse_args()
        # Extrai o mes inteiro a partir do boletim indicado ou do boletim mais
        # recente encontrado na pasta informada.
        bulletin = extract_prices_from_boletim(args.entrada, month=args.mes)
        _print_preview(bulletin)

        if not args.aplicar:
            print("\nPrevia concluida. Nenhum arquivo foi alterado.")
            print("Para atualizar os dados, rode novamente com --aplicar.")
            return

        # A gravacao reaproveita a mesma rotina da insercao manual, entao
        # repeticoes do mesmo mes substituem os valores anteriores.
        df = load_master_table(MASTER_DATA_FILE)
        df = add_or_update_prices(df, bulletin.month_label, bulletin.prices)
        save_master_table(df, MASTER_DATA_FILE)

        print("\nAtualizacao aplicada.")
        print(f"Tabela unica salva em: {MASTER_DATA_FILE}")
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Erro: {exc}") from None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Importa precos mensais a partir de um boletim DOCX e, opcionalmente, "
            "atualiza data/precos_mensais.xlsx."
        )
    )
    parser.add_argument(
        "entrada",
        nargs="?",
        type=Path,
        default=BOLETIM_DIR,
        help=(
            "Arquivo ou pasta do boletim. Se omitido, procura o boletim mais "
            "recente em previsoes_boletim. Pastas podem conter subpastas por mes."
        ),
    )
    parser.add_argument(
        "--mes",
        help=(
            "Mes de referencia no formato YYYY-MM. Use quando o caminho do "
            "arquivo/pasta nao deixar claro qual mes deve ser importado."
        ),
    )
    parser.add_argument(
        "--aplicar",
        action="store_true",
        help="Aplica a importacao e atualiza data/precos_mensais.xlsx.",
    )
    return parser.parse_args()


def _print_preview(bulletin) -> None:
    products = [PRODUCT_CESTA_BASICA, *PRODUTOS]
    print(f"Arquivo lido: {bulletin.source_file}")
    print(f"Mes identificado: {bulletin.month_label}")
    print("Modo: previa (use --aplicar para persistir no Excel)")
    print("")
    print(f"{'cidade':<8} {'produto':<14} {'preco':>10}")
    print("-" * 35)

    for city in ("ilheus", "itabuna"):
        for product in products:
            price = bulletin.prices[city][product]
            print(f"{city:<8} {product:<14} {price:>10.2f}")


if __name__ == "__main__":
    main()
