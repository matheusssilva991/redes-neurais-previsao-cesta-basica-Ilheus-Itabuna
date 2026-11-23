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
        """Retorna o mês de referência do boletim no formato `YYYY-MM`."""
        return f"{self.year}-{self.month:02d}"


@dataclass(frozen=True)
class DocumentBlock:
    kind: str
    text: str = ""
    rows: tuple[tuple[str, ...], ...] = ()


def find_boletim_file(path: Path = BOLETIM_DIR) -> Path:
    """Localiza um boletim a partir de um arquivo ou diretório.

    Na busca recursiva, prioriza o mês de referência mais recente.
    Para o mesmo mês, dá preferência a DOCX e depois à última modificação.

    Args:
        path: Arquivo já selecionado ou diretório onde procurar boletins.

    Returns:
        Caminho informado quando é um arquivo ou o melhor candidato da busca.

    Raises:
        FileNotFoundError: Se o caminho não existir ou a pasta não tiver
            arquivos com extensão reconhecida.
    """
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
    """Extrai os gastos mensais da cesta e dos produtos de um boletim DOCX.

    Args:
        path: Caminho do DOCX ou pasta onde procurar o boletim.
        month: Mês de referência em `YYYY-MM`. Quando omitido, é inferido
            a partir do caminho do boletim.

    Returns:
        Arquivo de origem, ano, mês e preços em reais por cidade e produto.

    Raises:
        FileNotFoundError: Se nenhum boletim for encontrado.
        ValueError: Se o formato não for suportado, o mês não for válido
            ou o documento não contiver todos os preços exigidos.
        ET.ParseError: Se o XML interno do DOCX estiver malformado.
    """
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
    """Lê parágrafos e tabelas do corpo de um DOCX na ordem do documento.

    Args:
        path: Caminho do arquivo DOCX.

    Returns:
        Lista de blocos das tabelas e dos parágrafos com conteúdo.

    Raises:
        ValueError: Se o ZIP for inválido ou faltar o XML ou o corpo do documento.
        ET.ParseError: Se o XML estiver malformado.
    """
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
    """Associa as tabelas de produtos às cidades e lê os valores do mês.

    O título com a cidade identifica a próxima tabela de gastos de produtos.

    Args:
        blocks: Parágrafos e tabelas na ordem em que aparecem no boletim.
        month_number: Mês de referência, de 1 a 12.

    Returns:
        Preços extraídos, agrupados por cidade e produto. A verificação
        da cobertura das cidades é feita pelo chamador.

    Raises:
        ValueError: Se uma tabela identificada tiver preços inválidos ou
            faltar a linha, a coluna ou os produtos esperados.
    """
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
    """Adiciona ao mapeamento os totais da cesta para as duas cidades.

    Args:
        rows: Linhas da tabela da cesta com o nome do mês na primeira célula.
        month_number: Mês de referência, de 1 a 12.
        prices: Mapeamento de preços a atualizar, com as duas cidades presentes.

    Raises:
        ValueError: Se a linha do mês faltar ou se seus valores forem inválidos.
    """
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
    """Lê os gastos mensais dos produtos de uma cidade.

    Args:
        rows: Linhas da tabela, começando pelo cabeçalho e com o produto
            na primeira coluna.
        month_number: Mês de referência, de 1 a 12.

    Returns:
        Gastos em reais por identificador de produto.

    Raises:
        ValueError: Se a coluna de gasto ou algum produto esperado estiver
            ausente, ou se houver produtos ou preços inválidos.
    """
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
    """Identifica a coluna de gasto mensal na tabela de produtos.

    Args:
        header: Células da linha de cabeçalho.
        month_number: Mês de referência, de 1 a 12.

    Returns:
        Índice da primeira coluna que menciona gasto e o mês. Se nenhuma
        coluna mencionar ambos, retorna a primeira que menciona gasto.

    Raises:
        ValueError: Se nenhuma coluna de gasto estiver presente.
    """
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
    """Verifica se as primeiras duas linhas mencionam mês e ambas as cidades."""
    if len(rows) < 2:
        return False

    first_cells = " ".join(cell for row in rows[:2] for cell in row)
    normalized = _normalize_text(first_cells)
    return "mes" in normalized and "ilheus" in normalized and "itabuna" in normalized


def _looks_like_product_spending_table(rows: tuple[tuple[str, ...], ...]) -> bool:
    """Verifica se o cabeçalho de `rows` começa com produto e menciona gasto."""
    if not rows:
        return False

    header = [_normalize_text(cell) for cell in rows[0]]
    return (
        bool(header)
        and header[0] == "produto"
        and any("gasto" in cell for cell in header)
    )


def _paragraph_text(element: ET.Element) -> str:
    """Concatena os fragmentos de texto do parágrafo XML em `element`."""
    return " ".join(
        "".join(
            text_node.text or "" for text_node in paragraph.findall(".//w:t", W_NS)
        ).strip()
        for paragraph in [element]
    ).strip()


def _table_rows(element: ET.Element) -> list[list[str]]:
    """Extrai as células da tabela XML em `element`, na ordem das linhas."""
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
    """Infere o ano e o mês de referência de um boletim.

    Args:
        path: Caminho que pode conter um componente `YYYYMM` ou um nome
            de arquivo com o ano e o mês numéricos.

    Returns:
        Tupla `(ano, mês)`, com ano entre 2000 e 2099.

    Raises:
        ValueError: Se o caminho não indicar um mês de referência.
    """
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
    """Valida o mês informado explicitamente na importação.

    Args:
        value: Rótulo `YYYY-MM`, com ano entre 2000 e 2099.

    Returns:
        Tupla `(ano, mês)`.

    Raises:
        ValueError: Se o rótulo não tiver o formato ou o intervalo esperado.
    """
    match = re.fullmatch(r"(20\d{2})-(0[1-9]|1[0-2])", value.strip())
    if not match:
        raise ValueError("Use --mes no formato YYYY-MM, por exemplo 2026-05")

    return int(match.group(1)), int(match.group(2))


def _candidate_sort_key(path: Path) -> tuple[int, int, int, float, str]:
    """Gera a chave de ordenação de um candidato a boletim.

    Args:
        path: Caminho existente do candidato a boletim.

    Returns:
        Chave com ano, mês, suporte do formato, última modificação e caminho.
        Usa ano e mês zero quando a referência não pode ser inferida.
    """
    try:
        year, month = _infer_year_month(path)
    except ValueError:
        year, month = 0, 0

    supported = int(_compound_suffix(path) in SUPPORTED_EXTENSIONS)
    return year, month, supported, path.stat().st_mtime, str(path)


def _is_known_bulletin_file(path: Path) -> bool:
    """Verifica se a extensão de `path` é reconhecida na busca de boletins."""
    return _compound_suffix(path) in KNOWN_EXTENSIONS


def _compound_suffix(path: Path) -> str:
    """Obtém a extensão de `path` em minúsculas e trata `.rtf.doc` como DOC."""
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-2:] == [".rtf", ".doc"]:
        return ".doc"

    return suffixes[-1] if suffixes else ""


def _normalize_text(value: str) -> str:
    """Prepara `value` para comparação sem distinção de caixa ou acentos."""
    normalized = unicodedata.normalize("NFKD", value.strip().lower())
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_text)


def _local_name(tag: str) -> str:
    """Obtém o nome local de `tag` removendo o namespace XML."""
    return tag.rsplit("}", 1)[-1]
