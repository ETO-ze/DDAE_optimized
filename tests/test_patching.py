import unittest
import numpy as np

from ddae_seismic.data_io import to_training_pairs, reconstruct_from_windows, reconstruct_flattened
from ddae_seismic.utils.patching import extract_windows_1d, overlap_add_1d, window_starts


class WindowTests(unittest.TestCase):
    def test_roundtrip_boundaries_short_signals_and_tail(self):
        for nt in (1, 2, 3, 7, 8, 9, 13, 16, 19):
            for win in (1, 2, 8):
                for stride in sorted({1, max(1, win // 2), win}):
                    with self.subTest(nt=nt, win=win, stride=stride):
                        x = (np.arange(nt * 3).reshape(nt, 3) + 1).astype(np.float32)
                        windows = extract_windows_1d(x, win, stride)
                        y = reconstruct_from_windows(windows, nt, 3, win, stride)
                        np.testing.assert_allclose(y, x, rtol=1e-6, atol=1e-6)

    def test_trace_major_order_and_tail(self):
        x = np.arange(22, dtype=np.float32).reshape(11, 2)
        self.assertEqual(window_starts(11, 4, 4), [0, 4, 7])
        windows = extract_windows_1d(x, 4, 4)
        np.testing.assert_array_equal(windows[0], x[:4, 0])
        np.testing.assert_array_equal(windows[2], x[-4:, 0])
        np.testing.assert_array_equal(windows[3], x[:4, 1])

    def test_short_trace_uses_edge_padding(self):
        x = np.array([[2], [5]], dtype=np.float32)
        np.testing.assert_array_equal(extract_windows_1d(x, 4, 4), [[2, 5, 5, 5]])

    def test_constant_predictions_cover_every_sample(self):
        for n, win, stride in ((17, 8, 8), (17, 8, 3), (3, 8, 8)):
            windows = np.full((len(window_starts(n, win, stride)), win), 7.0)
            np.testing.assert_allclose(overlap_add_1d(windows, n, win, stride), 7.0)

    def test_invalid_window_parameters(self):
        for args in ((0, 4, 2), (4, 0, 2), (4, 4, 0), (10, 4, 5), (-1, 4, 2)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                window_starts(*args)
        for shape in ((0, 2), (2, 0)):
            with self.assertRaises(ValueError):
                extract_windows_1d(np.empty(shape), 4, 2)

    def test_window_count_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            reconstruct_from_windows(np.ones((3, 4)), 8, 2, 4, 4)
        with self.assertRaises(ValueError):
            overlap_add_1d(np.ones((3, 4)), 8, 4, 4)

    def test_flatten_roundtrip_and_model_predictions(self):
        for shape in ((1, 1), (3, 2), (8, 2), (11, 3)):
            for win in (1, 4, 8):
                with self.subTest(shape=shape, win=win):
                    x = (np.arange(np.prod(shape)).reshape(shape) + 1).astype(np.float32)
                    X, Y = to_training_pairs(x, 2 * x, win, 1, "flatten")
                    np.testing.assert_array_equal(Y, 2 * X)
                    result = reconstruct_flattened(0.25 * X, *shape, win)
                    np.testing.assert_array_equal(result, 0.25 * x)

    def test_flatten_order_preserves_legacy_complete_chunks(self):
        x = np.arange(18).reshape(9, 2)
        X, _ = to_training_pairs(x, None, 4, 1, "flatten")
        np.testing.assert_array_equal(X[:4].ravel(), x.T.ravel()[:16])
        np.testing.assert_array_equal(X[-1], [15, 17, 17, 17])

    def test_flatten_rejects_invalid_parameters_and_count(self):
        with self.assertRaises(ValueError):
            to_training_pairs(np.ones((4, 2)), None, 0, 1, "flatten")
        with self.assertRaises(ValueError):
            reconstruct_flattened(np.ones((1, 4)), 4, 2, 4)


if __name__ == "__main__":
    unittest.main()
