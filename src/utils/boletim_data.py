"""
Extract monthly prices from ACCB bulletin files.

The current supported source is DOCX. Older RTF/DOC files are detected and get a
clear error message so they can be converted before importing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

from config import PROJECT_ROOT, PRODUTOS
from .monthly_data import (
    PRODUCT_CESTA_BASICA,
    normalize_price,
    normalize_product,
    validate_complete_month,
)

BOLETIM_DIR = PROJECT_ROOT / "previsoes_boletim"
SUPPORTED_EXTENSIONS = (".docx",)
KNOWN_EXTENSIONS = (".docx", ".rtf", ".doc")

MONTH_NAMES_PT = {
    1: "janeiro",
    2: "fevereiro",
    3: "marco",
    4: "abril",
    5: "maio",
    6: "junho",
    7: "julho",
    8: "agosto",
    9: "setembro",
    10: "outubro",
    11: "novembro",
    12: "dezembro",
}

W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


@dataclass(frozen=True)
class BulletinPrices:
    """Prices extracted from one bulletin file."""

    source_file: Path
    year: int
    month: int
    prices: dict[str, dict[str, float]]

    @property
    def month_label(self) -> str:
        return f"{self.year}-{self.month:02d}"


@dataclass(frozen=True)
class DocumentBlock:
    kind: str
    text: str = ""
    rows: tuple[tuple[str, ...], ...] = ()


def find_boletim_file(path: Path = BOLETIM_DIR) -> Path:
    """Find the latest bulletin-like file in a file or directory path."""
    path = path.expanduser()
    if path.is_file():
        return path

    if not path.exists():
        raise FileNotFoundError(f"Caminho nao encontrado: {path}")

    candidates = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and not candidate.name.startswith(".~lock")
        and _is_known_bulletin_file(candidate)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum boletim .docx, .rtf ou .doc encontrado em {path}"
        )

    return max(candidates, key=_candidate_sort_key)


def extract_prices_from_boletim(
    path: Path = BOLETIM_DIR,
    *,
    month: str | None = None,
) -> BulletinPrices:
    """Extract basket and product monthly spending values from a bulletin."""
    source_file = find_boletim_file(path)
    suffix = _compound_suffix(source_file)
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Formato ainda nao suportado para importacao direta: {source_file.name}. "
            "Converta o boletim para .docx e rode o comando novamente."
        )

    year, month_number = (
        _parse_month_override(month) if month else _infer_year_month(source_file)
    )
    blocks = _read_docx_blocks(source_file)
    prices = _extract_prices_from_blocks(blocks, month_number)
    validate_complete_month(prices)

    return BulletinPrices(
        source_file=source_file,
        year=year,
        month=month_number,
        prices=prices,
    )


def _read_docx_blocks(path: Path) -> list[DocumentBlock]:
    try:
        with ZipFile(path) as docx:
            document_xml = docx.read("word/document.xml")
    except (BadZipFile, KeyError) as exc:
        raise ValueError(f"Arquivo DOCX invalido: {path}") from exc

    root = ET.fromstring(document_xml)
    body = root.find("w:body", W_NS)
    if body is None:
        raise ValueError(f"DOCX sem corpo de documento: {path}")

    blocks: list[DocumentBlock] = []
    for child in body:
        tag = _local_name(child.tag)
        if tag == "p":
            text = _paragraph_text(child)
            if text:
                blocks.append(DocumentBlock(kind="paragraph", text=text))
        elif tag == "tbl":
            rows = tuple(tuple(cell for cell in row) for row in _table_rows(child))
            if rows:
                blocks.append(DocumentBlock(kind="table", rows=rows))

    return blocks


def _extract_prices_from_blocks(
    blocks: list[DocumentBlock],
    month_number: int,
) -> dict[str, dict[str, float]]:
    prices: dict[str, dict[str, float]] = {"ilheus": {}, "itabuna": {}}
    pending_city: str | None = None

    for block in blocks:
        if block.kind == "paragraph":
            normalized = _normalize_text(block.text)
            if "preco medio" in normalized and "gasto mensal" in normalized:
                if "ilheus" in normalized:
                    pending_city = "ilheus"
                elif "itabuna" in normalized:
                    pending_city = "itabuna"
            continue

        if _looks_like_basket_table(block.rows):
            _extract_basket_prices(block.rows, month_number, prices)
            continue

        if pending_city and _looks_like_product_spending_table(block.rows):
            prices[pending_city].update(
                _extract_product_prices(block.rows, month_number)
            )
            pending_city = None

    return prices


def _extract_basket_prices(
    rows: tuple[tuple[str, ...], ...],
    month_number: int,
    prices: dict[str, dict[str, float]],
) -> None:
    target_month = MONTH_NAMES_PT[month_number]

    for row in rows:
        if not row:
            continue

        row_month = _normalize_text(row[0])
        if row_month != target_month:
            continue

        values = [cell for cell in row if cell.strip()]
        if len(values) >= 5:
            ilheus_value = values[1]
            itabuna_value = values[3]
        elif len(values) >= 3:
            ilheus_value = values[1]
            itabuna_value = values[2]
        else:
            raise ValueError("Tabela da cesta basica sem valores de Ilheus e Itabuna")

        prices["ilheus"][PRODUCT_CESTA_BASICA] = normalize_price(ilheus_value)
        prices["itabuna"][PRODUCT_CESTA_BASICA] = normalize_price(itabuna_value)
        return

    raise ValueError(
        f"Nao encontrei a linha de {target_month.title()} na tabela da cesta"
    )


def _extract_product_prices(
    rows: tuple[tuple[str, ...], ...],
    month_number: int,
) -> dict[str, float]:
    header = rows[0]
    spending_column = _find_spending_column(header, month_number)
    product_prices: dict[str, float] = {}

    for row in rows[1:]:
        if len(row) <= spending_column:
            continue

        product_name = row[0].strip()
        if not product_name or _normalize_text(product_name) == "total":
            continue

        product = normalize_product(product_name)
        product_prices[product] = normalize_price(row[spending_column])

    expected = set(PRODUTOS)
    missing = expected - set(product_prices)
    if missing:
        raise ValueError(f"Tabela de produtos sem valores para: {sorted(missing)}")

    return product_prices


def _find_spending_column(header: tuple[str, ...], month_number: int) -> int:
    month_name = MONTH_NAMES_PT[month_number]
    normalized_header = [_normalize_text(value) for value in header]

    for index, value in enumerate(normalized_header):
        if "gasto" in value and month_name in value:
            return index

    for index, value in enumerate(normalized_header):
        if "gasto" in value:
            return index

    raise ValueError("Tabela de produtos sem coluna de gasto mensal")


def _looks_like_basket_table(rows: tuple[tuple[str, ...], ...]) -> bool:
    if len(rows) < 2:
        return False

    first_cells = " ".join(cell for row in rows[:2] for cell in row)
    normalized = _normalize_text(first_cells)
    return "mes" in normalized and "ilheus" in normalized and "itabuna" in normalized


def _looks_like_product_spending_table(rows: tuple[tuple[str, ...], ...]) -> bool:
    if not rows:
        return False

    header = [_normalize_text(cell) for cell in rows[0]]
    return (
        bool(header)
        and header[0] == "produto"
        and any("gasto" in cell for cell in header)
    )


def _paragraph_text(element: ET.Element) -> str:
    return " ".join(
        "".join(
            text_node.text or "" for text_node in paragraph.findall(".//w:t", W_NS)
        ).strip()
        for paragraph in [element]
    ).strip()


def _table_rows(element: ET.Element) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in element.findall("w:tr", W_NS):
        cells: list[str] = []
        for cell in row.findall("w:tc", W_NS):
            paragraphs = [
                _paragraph_text(paragraph)
                for paragraph in cell.findall("w:p", W_NS)
                if _paragraph_text(paragraph)
            ]
            cells.append(" ".join(paragraphs).strip())
        rows.append(cells)

    return rows


def _infer_year_month(path: Path) -> tuple[int, int]:
    for part in reversed(path.parts):
        match = re.fullmatch(r"(20\d{2})(0[1-9]|1[0-2])", part)
        if match:
            return int(match.group(1)), int(match.group(2))

    match = re.search(r"(20\d{2})[-_ ]?(0[1-9]|1[0-2])", path.stem)
    if match:
        return int(match.group(1)), int(match.group(2))

    raise ValueError(
        "Nao consegui identificar o mes do boletim. Use --mes YYYY-MM para informar."
    )


def _parse_month_override(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(20\d{2})-(0[1-9]|1[0-2])", value.strip())
    if not match:
        raise ValueError("Use --mes no formato YYYY-MM, por exemplo 2026-05")

    return int(match.group(1)), int(match.group(2))


def _candidate_sort_key(path: Path) -> tuple[int, int, int, float, str]:
    try:
        year, month = _infer_year_month(path)
    except ValueError:
        year, month = 0, 0

    supported = int(_compound_suffix(path) in SUPPORTED_EXTENSIONS)
    return year, month, supported, path.stat().st_mtime, str(path)


def _is_known_bulletin_file(path: Path) -> bool:
    return _compound_suffix(path) in KNOWN_EXTENSIONS


def _compound_suffix(path: Path) -> str:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-2:] == [".rtf", ".doc"]:
        return ".doc"

    return suffixes[-1] if suffixes else ""


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.strip().lower())
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_text)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
