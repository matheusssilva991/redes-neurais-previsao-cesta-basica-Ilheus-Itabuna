import unittest

import numpy as np
import pandas as pd

from src.utils.data_utils import create_time_sequences, prepare_training_data


class DataUtilsTest(unittest.TestCase):
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
        )

        X_train, y_train, X_val = prepare_training_data(
            df,
            look_back=3,
            forecast_horizon=2,
        )

        self.assertEqual(X_train.shape, (12, 3, 1))
        self.assertEqual(y_train.shape, (12, 2))
        np.testing.assert_array_equal(X_val[0, :, 0], [14, 15, 16])


if __name__ == "__main__":
    unittest.main()
