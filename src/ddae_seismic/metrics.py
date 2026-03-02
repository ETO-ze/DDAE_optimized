from __future__ import annotations

import numpy as np

def snr_db(clean: np.ndarray, test: np.ndarray, eps: float = 1e-12) -> float:
    """SNR = 10 log10( ||clean||^2 / ||clean-test||^2 )."""
    clean = np.asarray(clean, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    num = np.sum(clean**2)
    den = np.sum((clean - test)**2) + eps
    return 10.0 * np.log10(num / den)

def snr_improvement_db(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray) -> float:
    return snr_db(clean, denoised) - snr_db(clean, noisy)
