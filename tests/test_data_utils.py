import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.utils.data_utils import (
    create_time_sequences,
    generate_forecast,
    prepare_training_data,
)


class DataUtilsTest(unittest.TestCase):
    def test_generate_forecast_uses_direct_inference_for_one_window(self):
        model = Mock(return_value=np.array([[1.0, 2.0]]))
        janela = np.zeros((1, 3, 1))

        self.assertEqual(generate_forecast(model, janela), [1.0, 2.0])
        model.assert_called_once_with(janela, training=False)
        model.predict.assert_not_called()

    def test_generate_forecast_keeps_batch_prediction_for_multiple_windows(self):
        model = Mock()
        model.predict.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        janelas = np.zeros((2, 3, 1))

        self.assertEqual(generate_forecast(model, janelas, batch_size=2), [1.0, 2.0])
        model.predict.assert_called_once_with(janelas, batch_size=2, verbose=0)

    def test_sequences_keep_legacy_window_logic(self):
        df = pd.DataFrame({"preco": list(range(1, 17))})

        sequences = create_time_sequences(df, look_back=3, forecast_horizon=2)
        X_train, y_train, X_val = prepare_training_data(
            sequences,
            look_back=3,
            forecast_horizon=2,
        )

        self.assertEqual(sequences.shape, (12, 5))
        np.testing.assert_array_equal(sequences.iloc[0].to_numpy(), [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(
            sequences.iloc[-1].to_numpy(), [12, 13, 14, 15, 16]
        )
        self.assertEqual(X_train.shape, (12, 3, 1))
        self.assertEqual(y_train.shape, (12, 2))
        np.testing.assert_array_equal(X_train[0, :, 0], [1, 2, 3])
        np.testing.assert_array_equal(y_train[0], [4, 5])
        np.testing.assert_array_equal(X_val[0, :, 0], [14, 15, 16])

    def test_prepare_training_data_accepts_raw_series_with_dates(self):
        df = pd.DataFrame(
            {
                "data": pd.date_range("2026-01-01", periods=16, freq="MS"),
                "preco": list(range(1, 17)),
            }
        ).iloc[::-1]

        X_train, y_train, X_val = prepare_training_data(
            df,
            look_back=3,
            forecast_horizon=2,
        )

        self.assertEqual(X_train.shape, (12, 3, 1))
        self.assertEqual(y_train.shape, (12, 2))
        tabular_arrays = prepare_training_data(create_time_sequences(df, 3, 2), 3)
        for raw, tabular in zip((X_train, y_train, X_val), tabular_arrays, strict=True):
            np.testing.assert_array_equal(raw, tabular.astype(np.float32), strict=True)

        for array in (X_train, y_train):
            self.assertTrue(array.flags.c_contiguous)
            self.assertTrue(array.flags.writeable)
        X_train[0, 1, 0] = -1
        y_train[0, 1] = -1
        self.assertEqual(X_train[1, 0, 0], 2)
        self.assertEqual(y_train[1, 0], 5)

    def test_time_sequences_drop_missing_prices_and_preserve_nullable_dtype(self):
        df = pd.DataFrame({"preco": pd.array([1, 2, None, 4, 5, 6], dtype="Int64")})
        df.columns = pd.Index(df.columns, dtype="string", name="campo")
        expected = pd.DataFrame(
            {
                "preco": pd.array([4], dtype="Int64"),
                "preco t(h + 1)": pd.array([5], dtype="Int64"),
                "preco t(h + 2)": pd.array([6], dtype="Int64"),
            }
        )
        expected.columns = pd.Index(expected.columns, dtype="string", name="campo")
        pd.testing.assert_frame_equal(create_time_sequences(df, 2, 1), expected)
        with self.assertRaisesRegex(ValueError, "valores invalidos"):
            prepare_training_data(df, 2, 1)

    def test_time_sequences_keep_schema_for_short_or_empty_series(self):
        expected = pd.DataFrame(
            {
                "preco": pd.Series(dtype="int64"),
                "preco t(h + 1)": pd.Series(dtype="float64"),
                "preco t(h + 2)": pd.Series(dtype="float64"),
            }
        )
        for values in ([], [1, 2]):
            with self.subTest(values=values):
                df = pd.DataFrame({"preco": pd.Series(values, dtype="int64")})
                sequences = create_time_sequences(df, 2, 1)
                pd.testing.assert_frame_equal(sequences, expected)
                arrays = prepare_training_data(sequences, 2, 1)
                self.assertEqual(
                    [array.shape for array in arrays], [(0, 2, 1), (0, 1), (0, 2, 1)]
                )
                with self.assertRaisesRegex(ValueError, "Serie insuficiente"):
                    prepare_training_data(df, 2, 1)


if __name__ == "__main__":
    unittest.main()
