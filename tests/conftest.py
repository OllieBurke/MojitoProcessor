"""Shared pytest fixtures for MojitoProcessor unit tests."""

import numpy as np
import pytest

from MojitoProcessor.SigProcessing import SignalProcessor

# ─────────────────────────────────────────────────────────────────────────────
# SignalProcessor fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(name="sp_data")
def _sp_data() -> dict:
    """Three-channel arrays (N=1000, fs=4 Hz) for SignalProcessor tests."""
    np.random.seed(42)
    n = 1000
    return {
        "X": np.random.randn(n) * 1e-12,
        "Y": np.random.randn(n) * 1e-12,
        "Z": np.random.randn(n) * 1e-12,
    }


@pytest.fixture(name="simple_sp")
def _simple_sp(sp_data: dict) -> SignalProcessor:
    """Fresh SignalProcessor at fs=4 Hz with 1000 samples."""
    return SignalProcessor(sp_data, fs=4.0)


# ─────────────────────────────────────────────────────────────────────────────
# process_pipeline data dict fixture
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(name="pipeline_mojito_data")
def _pipeline_mojito_data() -> dict:
    """Data dict with 1000 samples at 4 Hz for process_pipeline tests."""
    np.random.seed(99)
    n, fs = 1000, 4.0
    t0_ref = 9.77298893e7  # realistic TCB start time in seconds
    return {
        "tdis": {
            "X": np.random.randn(n) * 1e-12,
            "Y": np.random.randn(n) * 1e-12,
            "Z": np.random.randn(n) * 1e-12,
        },
        "fs": fs,
        "t_tdi": t0_ref + np.arange(n) / fs,
    }
