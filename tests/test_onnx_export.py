from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.models import get_model
from src.utils.data_utils import save_model

import keras
import numpy as np
import onnxruntime


class OnnxExportTest(unittest.TestCase):
    def test_lstm_export_has_independent_calls_and_preserves_training_state(self):
        with TemporaryDirectory() as directory, keras.device("cpu"):
            model = get_model("LSTM", 12, 3)
            stateful_layer = model.get_layer("lstm_layer_2")
            inputs = np.full((1, 12, 1), 0.4, dtype=np.float32)
            model.predict(inputs, batch_size=1, verbose=0)
            previous_state = [
                keras.ops.convert_to_numpy(s).copy() for s in stateful_layer.states
            ]

            with patch("src.utils.data_utils.MODELS_DIR", Path(directory)):
                saved = save_model(
                    model, "LSTM", "ilheus", "cesta_basica", 3, save_onnx=True
                )

            for previous, current in zip(
                previous_state, stateful_layer.states, strict=True
            ):
                np.testing.assert_array_equal(
                    previous, keras.ops.convert_to_numpy(current)
                )

            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            session = onnxruntime.InferenceSession(
                str(saved.with_suffix(".onnx")),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            input_name = session.get_inputs()[0].name
            for sample in (inputs, inputs + np.float32(0.1)):
                stateful_layer.reset_state()
                expected = model.predict(sample, batch_size=1, verbose=0)
                actual = session.run(None, {input_name: sample})[0]
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)
