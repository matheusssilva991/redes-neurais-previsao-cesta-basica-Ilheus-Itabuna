from datetime import date
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.utils.monthly_data import (
    PRODUCT_CESTA_BASICA,
    add_or_update_prices,
    load_price_series,
    normalize_month,
    parse_month_date,
    save_master_table,
)


class MonthlyDataTest(unittest.TestCase):
    def test_product_dates_use_real_month_end(self):
        self.assertEqual(parse_month_date("31-02-2026", "arroz"), date(2026, 2, 28))
        self.assertEqual(parse_month_date("31-04-2026", "arroz"), date(2026, 4, 30))
        self.assertEqual(parse_month_date("31-02-2024", "arroz"), date(2024, 2, 29))

    def test_basket_dates_use_month_start(self):
        self.assertEqual(
            parse_month_date("31-02-2026", PRODUCT_CESTA_BASICA),
            date(2026, 2, 1),
        )
        self.assertEqual(normalize_month("2026-05"), date(2026, 5, 1))

    def test_add_or_update_prices_replaces_existing_rows(self):
        df = pd.DataFrame(
            [
                {
                    "data": date(2026, 5, 1),
                    "cidade": "ilheus",
                    "produto": PRODUCT_CESTA_BASICA,
                    "preco": 100.0,
                }
            ]
        )

        updated = add_or_update_prices(
            df,
            "2026-05",
            {"ilheus": {PRODUCT_CESTA_BASICA: "123,45"}},
        )

        self.assertEqual(len(updated), 1)
        self.assertEqual(updated.iloc[0]["data"], date(2026, 5, 1))
        self.assertEqual(updated.iloc[0]["preco"], 123.45)

    def test_add_or_update_prices_with_aliases_and_composite_keys_is_idempotent(self):
        columns = ["data", "cidade", "produto", "preco"]
        df = pd.DataFrame(
            [
                (date(2026, 4, 30), "ilheus", "arroz", 2.0),
                (date(2026, 5, 31), "ilheus", "arroz", 1.0),
                (date(2026, 5, 31), "itabuna", "arroz", 8.0),
            ],
            columns=columns,
        )
        prices = {
            "Ilhéus": {"arroz": "200,50"},
            "ilheus": {"arroz": "205,50", "cesta": "600,15"},
            "itabuna": {"óleo": "10,00"},
        }
        expected = pd.DataFrame(
            [
                (date(2026, 4, 30), "ilheus", "arroz", 2.0),
                (date(2026, 5, 1), "ilheus", PRODUCT_CESTA_BASICA, 600.15),
                (date(2026, 5, 31), "ilheus", "arroz", 205.5),
                (date(2026, 5, 31), "itabuna", "arroz", 8.0),
                (date(2026, 5, 31), "itabuna", "oleo", 10.0),
            ],
            columns=columns,
        )

        updated = add_or_update_prices(df, "2026-05", prices)
        pd.testing.assert_frame_equal(updated, expected)
        pd.testing.assert_frame_equal(
            add_or_update_prices(updated, "2026-05", prices), expected
        )

    def test_load_price_series_reads_from_unified_table(self):
        df = pd.DataFrame(
            [
                {
                    "data": date(2026, 5, 1),
                    "cidade": "ilheus",
                    "produto": PRODUCT_CESTA_BASICA,
                    "preco": 100.0,
                },
                {
                    "data": date(2026, 5, 31),
                    "cidade": "ilheus",
                    "produto": "arroz",
                    "preco": 1.23,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            table_path = Path(tmp_dir) / "precos_mensais.xlsx"
            save_master_table(df, table_path)
            basket = load_price_series("ilheus", PRODUCT_CESTA_BASICA, table_path)
            product = load_price_series("ilheus", "arroz", table_path)

        self.assertEqual(basket.iloc[0]["data"], date(2026, 5, 1))
        self.assertEqual(product.iloc[0]["data"], date(2026, 5, 31))


if __name__ == "__main__":
    unittest.main()
