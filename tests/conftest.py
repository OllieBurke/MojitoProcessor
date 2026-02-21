"""Shared pytest fixtures for MojitoProcessor unit tests."""

import h5py
import numpy as np
import pytest

from MojitoProcessor.mojito_loader import MojitoData, OrbitData, TimeSampling
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
# MojitoData fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_mojito_data(n: int, fs: float, seed: int = 0) -> MojitoData:
    """Helper: build a fully-populated MojitoData object."""
    np.random.seed(seed)
    dt = 1.0 / fs
    t0 = 1e9
    n_freq = 20
    n_seg = 2

    data = MojitoData()
    data.filepath = "/fake/path/test.h5"
    data.metadata = {"pipeline_name": "test", "laser_frequency": 2.8e14}
    data.sampling = TimeSampling(t0=t0, dt=dt, size=n)
    data.t0 = t0
    data.dt = dt
    data.fs = fs
    data.duration = n * dt
    data.n_samples = n
    data.t_tdi = data.sampling.t()

    for ch in ["X", "Y", "Z", "A", "E", "T"]:
        data.tdis[ch] = np.random.randn(n) * 1e-12

    data.noise_freqs = np.logspace(-4, 0, n_freq)
    for ch in ["X", "Y", "Z"]:
        data.noise_psds[ch] = np.abs(np.random.randn(n_freq))
    data.noise_cov_xyz = np.random.randn(n_seg, n_freq, 3, 3)
    data.noise_cov_aet = np.random.randn(n_seg, n_freq, 3, 3)
    return data


@pytest.fixture(name="sample_mojito_data")
def _sample_mojito_data() -> MojitoData:
    """MojitoData with 400 samples at 4 Hz (used by loader/truncation tests)."""
    return _make_mojito_data(n=400, fs=4.0, seed=0)


@pytest.fixture(name="pipeline_mojito_data")
def _pipeline_mojito_data() -> MojitoData:
    """MojitoData with 1000 samples at 4 Hz (used by process_pipeline tests)."""
    return _make_mojito_data(n=1000, fs=4.0, seed=99)


# ─────────────────────────────────────────────────────────────────────────────
# OrbitData fixture
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(name="sample_orbit_data")
def _sample_orbit_data() -> OrbitData:
    """Fully-populated OrbitData object with synthetic arrays."""
    np.random.seed(1)
    n_ltt = 300
    n_orbit = 50
    ltt_dt = 0.25
    orbit_dt = 3600.0
    t0 = 0.0

    data = OrbitData()
    data.filepath = "/fake/path/orbits.h5"
    data.metadata = {"pipeline_name": "test"}

    data.ltt_sampling = TimeSampling(t0=t0, dt=ltt_dt, size=n_ltt)
    data.ltt_t0 = t0
    data.ltt_dt = ltt_dt
    data.ltt_fs = 1.0 / ltt_dt
    data.ltt_duration = n_ltt * ltt_dt
    data.ltt_n_samples = n_ltt
    data.t_ltt = data.ltt_sampling.t()

    data.orbit_sampling = TimeSampling(t0=t0, dt=orbit_dt, size=n_orbit)
    data.orbit_t0 = t0
    data.orbit_dt = orbit_dt
    data.orbit_fs = 1.0 / orbit_dt
    data.orbit_duration = n_orbit * orbit_dt
    data.orbit_n_samples = n_orbit
    data.t_orbit = data.orbit_sampling.t()

    for link in ["12", "13", "21", "23", "31", "32"]:
        data.ltts[link] = np.random.randn(n_ltt) * 8.3
        data.ltt_derivatives[link] = np.random.randn(n_ltt) * 1e-4
    for sc in ["1", "2", "3"]:
        data.sc_positions[sc] = np.random.randn(n_orbit, 3) * 1e11
        data.sc_velocities[sc] = np.random.randn(n_orbit, 3) * 1e3
    return data


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 file fixtures (written to a temporary directory)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(name="mojito_h5_file")
def _mojito_h5_file(tmp_path):
    """Minimal HDF5 file in mojito L1 format, including AET covariance."""
    np.random.seed(7)
    n, n_freq, n_seg = 200, 15, 2
    filepath = tmp_path / "test_mojito.h5"
    with h5py.File(filepath, "w") as f:
        f.attrs["pipeline_name"] = "test_pipeline"
        f.attrs["laser_frequency"] = 2.8e14
        f.attrs["lolipops_version"] = "0.1.0"

        samp = f.require_group("tdis/sampling")
        samp.attrs["t0"] = 0.0
        samp.attrs["dt"] = 0.25
        samp.attrs["size"] = n

        tdi_grp = f["tdis"]
        for ch in ["X2", "Y2", "Z2", "A2", "E2", "T2"]:
            tdi_grp.create_dataset(ch, data=np.random.randn(n) * 1e-12)

        noise_grp = f.require_group("noise_estimates")
        freq_grp = noise_grp.require_group("log_frequency_sampling")
        freq_grp.attrs["fmin"] = 1e-4
        freq_grp.attrs["fmax"] = 2.0
        freq_grp.attrs["size"] = n_freq

        noise_grp.create_dataset("XYZ", data=np.random.randn(n_seg, n_freq, 3, 3))
        aet = np.random.randn(n_seg, n_freq, 3, 3) + 1j * np.random.randn(
            n_seg, n_freq, 3, 3
        )
        noise_grp.create_dataset("AET", data=aet)
    return filepath


@pytest.fixture(name="mojito_h5_file_no_aet")
def _mojito_h5_file_no_aet(tmp_path):
    """Mojito HDF5 file without AET covariance (only XYZ channels)."""
    np.random.seed(8)
    n, n_freq, n_seg = 100, 10, 1
    filepath = tmp_path / "test_mojito_no_aet.h5"
    with h5py.File(filepath, "w") as f:
        f.attrs["pipeline_name"] = "no_aet_pipeline"
        f.attrs["laser_frequency"] = 2.8e14
        f.attrs["lolipops_version"] = "0.1.0"

        samp = f.require_group("tdis/sampling")
        samp.attrs["t0"] = 0.0
        samp.attrs["dt"] = 0.25
        samp.attrs["size"] = n

        tdi_grp = f["tdis"]
        for ch in ["X2", "Y2", "Z2"]:
            tdi_grp.create_dataset(ch, data=np.random.randn(n) * 1e-12)

        noise_grp = f.require_group("noise_estimates")
        freq_grp = noise_grp.require_group("log_frequency_sampling")
        freq_grp.attrs["fmin"] = 1e-4
        freq_grp.attrs["fmax"] = 2.0
        freq_grp.attrs["size"] = n_freq
        noise_grp.create_dataset("XYZ", data=np.random.randn(n_seg, n_freq, 3, 3))
    return filepath


@pytest.fixture(name="orbit_h5_file")
def _orbit_h5_file(tmp_path):
    """Minimal HDF5 file in orbit format with LTTs and spacecraft data."""
    np.random.seed(3)
    n_ltt, n_orbit = 150, 20
    filepath = tmp_path / "test_orbits.h5"
    with h5py.File(filepath, "w") as f:
        f.attrs["pipeline_name"] = "orbit_pipeline"
        f.attrs["laser_frequency"] = 2.8e14
        f.attrs["lolipops_version"] = "0.1.0"

        ltt_samp = f.require_group("ltts/sampling")
        ltt_samp.attrs["t0"] = 0.0
        ltt_samp.attrs["dt"] = 0.25
        ltt_samp.attrs["size"] = n_ltt

        ltt_grp = f["ltts"]
        for link in ["12", "13", "21", "23", "31", "32"]:
            ltt_grp.create_dataset(f"ltt_{link}", data=np.random.randn(n_ltt) * 8.3)
            ltt_grp.create_dataset(
                f"ltt_derivative_{link}", data=np.random.randn(n_ltt) * 1e-4
            )

        orbit_samp = f.require_group("orbits/sampling")
        orbit_samp.attrs["t0"] = 0.0
        orbit_samp.attrs["dt"] = 3600.0
        orbit_samp.attrs["size"] = n_orbit

        orbit_grp = f["orbits"]
        for sc in ["1", "2", "3"]:
            orbit_grp.create_dataset(
                f"sc_position_{sc}", data=np.random.randn(n_orbit, 3) * 1e11
            )
            orbit_grp.create_dataset(
                f"sc_velocity_{sc}", data=np.random.randn(n_orbit, 3) * 1e3
            )
    return filepath
