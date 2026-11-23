import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.utils import chart_utils


class ForecastLoadingTest(unittest.TestCase):
    def test_load_forecast_supports_native_and_legacy_values(self):
        cases = [
            ([1, "2"], [1.0, 2.0]),
            ([], []),
            ("[1e-1, -0]", [0.1, -0.0]),
            ("1,2", [1.0, 2.0]),
            ("[+1,.5]", [1.0, 0.5]),
        ]
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "previsao.json"
            for value, expected in cases:
                with self.subTest(value=value):
                    path.write_text(json.dumps({"arroz": value, "outra": [999]}))
                    decoded = chart_utils.load_forecast_from_json(
                        path, denormalize=False
                    )
                    self.assertEqual(decoded, expected)
                    self.assertTrue(all(type(item) is float for item in decoded))
                    self.assertEqual(
                        [math.copysign(1, item) for item in decoded],
                        [math.copysign(1, item) for item in expected],
                    )
                    self.assertEqual(
                        chart_utils.load_forecast_from_json(
                            path, normalization_factor=10
                        ),
                        [item * 10 for item in expected],
                    )

    def test_load_forecast_rejects_invalid_legacy_values(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "previsao.json"
            for value in ("[]", "[true]", '["1"]', "[1,erro]", {"valor": 1}):
                with self.subTest(value=value):
                    path.write_text(json.dumps({"arroz": value}))
                    with self.assertRaises(ValueError):
                        chart_utils.load_forecast_from_json(path)

    def test_load_product_uses_requested_key_and_converts_to_reais_per_unit(self):
        with TemporaryDirectory() as tmp_dir:
            forecasts_dir = Path(tmp_dir)
            region_dir = forecasts_dir / "ilheus"
            region_dir.mkdir()
            path = region_dir / "previsao_RNN_arroz_ilheus.json"
            path.write_text(json.dumps({"outra": [999], "arroz": "[+.01,.02]"}))
            with patch.object(chart_utils, "FORECASTS_PRODUTOS_DIR", forecasts_dir):
                result = chart_utils.load_product_forecasts(
                    "RNN", ["ilheus"], ["arroz"], {"arroz": 4.0}, forecast_horizon=3
                )

        self.assertEqual(result, {"ilheus": {"arroz": [2.5, 5.0]}})
