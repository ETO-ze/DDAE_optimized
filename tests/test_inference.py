import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import scipy.io as sio
import yaml

from ddae_seismic.config import RunConfig
from ddae_seismic.data_io import prepare_time_traces, to_training_pairs
from ddae_seismic.scripts import infer


class ScalingModel:
    def __init__(self):
        self.batch_sizes = []

    def predict(self, x, verbose=0):
        self.batch_sizes.append(len(x))
        return 0.25 * x


class InferenceTests(unittest.TestCase):
    def test_prediction_changes_every_sample_in_both_modes(self):
        for mode in ("trace", "flatten"):
            for norm in ("none", "global_max", "per_trace_std"):
                for shape in ((3, 2), (13, 3), (3, 13)):
                    with self.subTest(mode=mode, norm=norm, shape=shape):
                        cfg = RunConfig("unused.mat", "x", win=8, stride=8,
                                        patch_mode=mode, norm=norm, batch_size=2)
                        x = (np.arange(np.prod(shape)).reshape(shape) + 1).astype(np.float32)
                        model = ScalingModel()
                        inp, den, residual = infer.denoise_array(model, x, cfg)
                        self.assertEqual(den.shape, x.shape)
                        np.testing.assert_allclose(inp, x, rtol=1e-6)
                        np.testing.assert_allclose(den, 0.25 * x, rtol=1e-6)
                        np.testing.assert_allclose(residual, 0.75 * x, rtol=1e-6)
                        self.assertTrue(model.batch_sizes)
                        self.assertLessEqual(max(model.batch_sizes), 2)

    def test_training_and_inference_share_per_trace_scaling(self):
        # Unequal trace scales ensure a wrong normalization axis cannot cancel out.
        x = np.array([[1, 2, 4, 8, 16], [10, 12, 16, 24, 40]], dtype=np.float32)
        canonical = x.T
        expected = canonical / (np.std(canonical, axis=0, keepdims=True) + 1e-8)
        normalized, _ = prepare_time_traces(x, mode="per_trace_std")
        np.testing.assert_allclose(normalized, expected, rtol=1e-6)
        for mode in ("trace", "flatten"):
            cfg = RunConfig("unused", "x", win=4, stride=4, patch_mode=mode,
                            norm="per_trace_std", batch_size=2)
            training_windows, _ = to_training_pairs(normalized, None, 4, 4, mode)
            seen = []
            model = ScalingModel()

            def predict(batch, verbose=0):
                seen.append(batch.copy())
                return batch

            model.predict = predict
            infer.denoise_array(model, x, cfg)
            np.testing.assert_allclose(np.concatenate(seen), training_windows, rtol=1e-6)

    def test_model_output_shape_must_match_windows(self):
        model = ScalingModel()
        model.predict = lambda x, verbose=0: x[:, :-1]
        with self.assertRaisesRegex(ValueError, "Model output shape"):
            infer.denoise_array(model, np.ones((10, 2)), RunConfig("unused", "x", win=4, stride=4))

    def test_empty_and_non_matrix_input_rejected(self):
        for x in (np.empty((0, 2)), np.ones(8)):
            with self.assertRaises(ValueError):
                infer.denoise_array(ScalingModel(), x, RunConfig("unused", "x"))


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
class KerasIntegrationTests(unittest.TestCase):
    def test_saved_keras_model_cli_and_mat_output(self):
        import tensorflow as tf
        from ddae_seismic.scripts._common import make_run_dir

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Tiny deterministic model: verifies save/load/predict, not denoising quality.
            inputs = tf.keras.Input(shape=(8,))
            model = tf.keras.Model(inputs, tf.keras.layers.Rescaling(0.25)(inputs))
            with patch.object(Path, "symlink_to", side_effect=OSError("no symlink privileges")):
                run = make_run_dir(str(root / "runs"), "field")
            model.save(run / "best.keras")
            x = np.arange(1, 40, dtype=np.float32).reshape(3, 13)
            sio.savemat(root / "input.mat", {"x": x})
            for mode in ("trace", "flatten"):
                cfg = {"input_mat": str(root / "input.mat"), "input_key": "x",
                       "win": 8, "stride": 8, "patch_mode": mode,
                       "norm": "per_trace_std", "batch_size": 2}
                (root / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
                out = root / f"{mode}.mat"
                argv = ["infer", "--config", str(root / "config.yaml"),
                        "--model", str(root / "runs/field/latest/best.keras"), "--out", str(out)]
                with patch("sys.argv", argv):
                    infer.main()
                result = sio.loadmat(out)
                np.testing.assert_allclose(result["denoised"], 0.25 * x, rtol=1e-6)
                np.testing.assert_allclose(result["residual"], 0.75 * x, rtol=1e-6)
                self.assertEqual(result["denoised"].shape, x.shape)


if __name__ == "__main__":
    unittest.main()
