from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from config import PRODUTOS
from src.utils.boletim_data import extract_prices_from_boletim, find_boletim_file
from src.utils.monthly_data import PRODUCT_CESTA_BASICA


class BoletimDataTest(unittest.TestCase):
    def test_extract_prices_from_docx_folder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bulletin_dir = Path(tmp_dir) / "202605"
            bulletin_dir.mkdir()
            bulletin_path = bulletin_dir / "boletim-maio-2026.docx"
            _write_minimal_docx(bulletin_path)

            bulletin = extract_prices_from_boletim(bulletin_dir)

        self.assertEqual(bulletin.month_label, "2026-05")
        self.assertEqual(bulletin.source_file.name, "boletim-maio-2026.docx")
        self.assertEqual(bulletin.prices["ilheus"][PRODUCT_CESTA_BASICA], 667.35)
        self.assertEqual(bulletin.prices["itabuna"][PRODUCT_CESTA_BASICA], 665.90)
        self.assertEqual(bulletin.prices["ilheus"]["feijao"], 45.00)
        self.assertEqual(bulletin.prices["itabuna"]["cafe"], 18.40)

    def test_find_boletim_file_reports_unsupported_rtf(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bulletin_path = Path(tmp_dir) / "202604" / "boletimAbril2026.rtf"
            bulletin_path.parent.mkdir()
            bulletin_path.write_text("{\\rtf1}", encoding="utf-8")

            self.assertEqual(find_boletim_file(Path(tmp_dir)), bulletin_path)
            with self.assertRaisesRegex(ValueError, "Formato ainda nao suportado"):
                extract_prices_from_boletim(Path(tmp_dir))


def _write_minimal_docx(path: Path) -> None:
    product_rows_ilheus = _product_rows(
        {
            "Carne": "207.14",
            "Leite": "56.70",
            "Feijão": "45.00",
            "Arroz": "17.60",
            "Farinha": "24.12",
            "Tomate": "114.84",
            "Pão": "75.12",
            "Café": "19.36",
            "Banana": "48.75",
            "Açúcar": "11.31",
            "Óleo": "8.61",
            "Manteiga": "38.80",
        }
    )
    product_rows_itabuna = _product_rows(
        {
            "Carne": "203.22",
            "Leite": "57.48",
            "Feijão": "39.96",
            "Arroz": "16.02",
            "Farinha": "19.50",
            "Tomate": "117.12",
            "Pão": "87.30",
            "Café": "18.40",
            "Banana": "51.15",
            "Açúcar": "11.61",
            "Óleo": "8.44",
            "Manteiga": "35.70",
        }
    )

    body = [
        _paragraph("Tabela 1 - Custo da Cesta Básica"),
        _table(
            [
                ["Mês", "Ilhéus", "", "Itabuna", ""],
                [
                    "",
                    "Gasto Mensal (R$)",
                    "Variação Mensal (%)",
                    "Gasto Mensal (R$)",
                    "",
                ],
                ["Maio", "667.35", "-1.95", "665.90", "-0.85"],
            ]
        ),
        _paragraph("Tabela 2 - Preço Médio, Gasto Mensal e tempo - Ilhéus"),
        _table(product_rows_ilheus),
        _paragraph("Tabela 4 - Preço Médio, Gasto Mensal e tempo - Itabuna"),
        _table(product_rows_itabuna),
    ]
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>" + "".join(body) + "</w:body></w:document>"
    )

    with ZipFile(path, "w") as docx:
        docx.writestr("word/document.xml", document.encode("utf-8"))


def _product_rows(values: dict[str, str]) -> list[list[str]]:
    rows = [
        ["Produto", "Preço Médio Abril", "Preço Médio Maio", "Qtde.", "Gasto Maio (R$)"]
    ]
    for product in PRODUTOS:
        display = {
            "acucar": "Açúcar",
            "arroz": "Arroz",
            "banana": "Banana",
            "cafe": "Café",
            "carne": "Carne",
            "farinha": "Farinha",
            "feijao": "Feijão",
            "leite": "Leite",
            "manteiga": "Manteiga",
            "oleo": "Óleo",
            "pao": "Pão",
            "tomate": "Tomate",
        }[product]
        rows.append([display, "1.00", "1.00", "1.00", values[display]])
    rows.append(["Total", "", "", "", "0.00"])
    return rows


def _paragraph(text: str) -> str:
    return f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"


def _table(rows: list[list[str]]) -> str:
    xml_rows = []
    for row in rows:
        cells = "".join(f"<w:tc>{_paragraph(value)}</w:tc>" for value in row)
        xml_rows.append(f"<w:tr>{cells}</w:tr>")
    return f"<w:tbl>{''.join(xml_rows)}</w:tbl>"


if __name__ == "__main__":
    unittest.main()
