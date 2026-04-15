"""Unit tests for MojitoProcessor.io.write and MojitoProcessor.io.read.load_processed."""

import json

import h5py
import numpy as np
import pytest

from MojitoProcessor.io.read import load_processed
from MojitoProcessor.io.write import write
from MojitoProcessor.process.sigprocess import SignalProcessor

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(name="two_segments")
def _two_segments() -> dict:
    """Two SignalProcessor segments with known data."""
    rng = np.random.default_rng(0)
    n, fs = 500, 4.0
    t0 = 9.77298893e7
    segments = {}
    for i in range(2):
        data = {ch: rng.standard_normal(n) for ch in ["X", "Y", "Z"]}
        segments[f"segment{i}"] = SignalProcessor(data, fs=fs, t0=t0 + i * n / fs)
    return segments


@pytest.fixture(name="raw_data")
def _raw_data() -> dict:
    """Minimal raw auxiliary data dict mirroring load_file output."""
    rng = np.random.default_rng(1)
    n_ltt, n_orbit, n_freq = 200, 50, 100
    return {
        "noise_estimates": {
            # shape (1, n_freq, 3, 3) mirrors mojito noise covariance cubes
            "xyz": rng.standard_normal((1, n_freq, 3, 3)),
            "aet": rng.standard_normal((1, n_freq, 3, 3)),
        },
        "ltts": {
            "12": rng.standard_normal(n_ltt),
            "13": rng.standard_normal(n_ltt),
        },
        "ltt_derivatives": {
            "12": rng.standard_normal(n_ltt),
            "13": rng.standard_normal(n_ltt),
        },
        "ltt_times": np.arange(n_ltt, dtype=float),
        "orbits": rng.standard_normal((n_orbit, 3, 3)),
        "velocities": rng.standard_normal((n_orbit, 3, 3)),
        "orbit_times": np.arange(n_orbit, dtype=float),
        "metadata": {
            "laser_frequency": 2.816e14,
            "pipeline_names": ["emri", "noise"],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tests: file creation and basic structure
# ─────────────────────────────────────────────────────────────────────────────


class TestWriteFileCreation:
    def test_creates_file(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        assert out.exists()

    def test_overwrites_existing_file(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        write(out, two_segments)  # second call must not raise
        assert out.exists()

    def test_processed_group_exists(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        with h5py.File(out, "r") as f:
            assert "processed" in f

    def test_pipeline_params_group_exists(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        with h5py.File(out, "r") as f:
            assert "pipeline_params" in f

    def test_no_raw_group_without_raw_data(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        with h5py.File(out, "r") as f:
            assert "raw" not in f

    def test_raw_group_written_when_provided(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            assert "raw" in f


# ─────────────────────────────────────────────────────────────────────────────
# Tests: processed segments
# ─────────────────────────────────────────────────────────────────────────────


class TestWriteProcessedSegments:
    def test_all_segment_groups_written(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        with h5py.File(out, "r") as f:
            for key in two_segments:
                assert key in f["processed"]

    def test_channel_datasets_present(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        with h5py.File(out, "r") as f:
            for ch in ["X", "Y", "Z"]:
                assert ch in f["processed/segment0"]

    def test_time_dataset_present(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        with h5py.File(out, "r") as f:
            assert "t" in f["processed/segment0"]

    def test_channel_data_correct(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        sp = two_segments["segment0"]
        with h5py.File(out, "r") as f:
            np.testing.assert_array_equal(f["processed/segment0/X"][:], sp._data["X"])

    def test_time_data_correct(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        sp = two_segments["segment0"]
        with h5py.File(out, "r") as f:
            np.testing.assert_allclose(f["processed/segment0/t"][:], sp.t)

    def test_segment_scalar_attrs(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        sp = two_segments["segment0"]
        with h5py.File(out, "r") as f:
            grp = f["processed/segment0"]
            assert grp.attrs["fs"] == pytest.approx(sp.fs)
            assert grp.attrs["N"] == sp.N
            assert grp.attrs["dt"] == pytest.approx(sp.dt)
            assert grp.attrs["T"] == pytest.approx(sp.T)
            assert grp.attrs["t0"] == pytest.approx(sp.t0)

    def test_channels_attr_is_json_list(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        with h5py.File(out, "r") as f:
            channels = json.loads(f["processed/segment0"].attrs["channels"])
        assert set(channels) == {"X", "Y", "Z"}

    def test_t0_none_stored_as_nan(self, tmp_path):
        sp = SignalProcessor({"X": np.zeros(100)}, fs=1.0, t0=None)
        out = tmp_path / "out.h5"
        write(out, {"segment0": sp})
        with h5py.File(out, "r") as f:
            assert np.isnan(f["processed/segment0"].attrs["t0"])

    def test_second_segment_data_correct(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        sp = two_segments["segment1"]
        with h5py.File(out, "r") as f:
            np.testing.assert_array_equal(f["processed/segment1/Y"][:], sp._data["Y"])


# ─────────────────────────────────────────────────────────────────────────────
# Tests: pipeline parameters
# ─────────────────────────────────────────────────────────────────────────────


class TestWritePipelineParams:
    def test_all_param_subgroups_present(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        with h5py.File(out, "r") as f:
            for name in ["filter", "downsample", "trim", "truncate", "window"]:
                assert name in f["pipeline_params"]

    def test_filter_attrs_written(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(
            out,
            two_segments,
            filter_kwargs={"highpass_cutoff": 5e-6, "order": 2},
        )
        with h5py.File(out, "r") as f:
            assert f["pipeline_params/filter"].attrs[
                "highpass_cutoff"
            ] == pytest.approx(5e-6)
            assert f["pipeline_params/filter"].attrs["order"] == 2

    def test_downsample_attrs_written(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments, downsample_kwargs={"target_fs": 0.2})
        with h5py.File(out, "r") as f:
            assert f["pipeline_params/downsample"].attrs["target_fs"] == pytest.approx(
                0.2
            )

    def test_none_value_stored_as_string(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments, filter_kwargs={"lowpass_cutoff": None})
        with h5py.File(out, "r") as f:
            assert f["pipeline_params/filter"].attrs["lowpass_cutoff"] == "None"

    def test_window_attrs_written(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments, window_kwargs={"window": "tukey", "alpha": 0.05})
        with h5py.File(out, "r") as f:
            assert f["pipeline_params/window"].attrs["window"] == "tukey"
            assert f["pipeline_params/window"].attrs["alpha"] == pytest.approx(0.05)

    def test_empty_kwargs_writes_empty_group(self, tmp_path, two_segments):
        """Passing no kwargs at all must still create the param groups."""
        out = tmp_path / "out.h5"
        write(out, two_segments)
        with h5py.File(out, "r") as f:
            grp = f["pipeline_params/filter"]
            assert len(grp.attrs) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: raw auxiliary data
# ─────────────────────────────────────────────────────────────────────────────


class TestWriteRawData:
    def test_noise_estimates_groups_present(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            assert "noise_estimates" in f["raw"]
            assert "xyz" in f["raw/noise_estimates"]
            assert "aet" in f["raw/noise_estimates"]

    def test_noise_xyz_data_correct(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            np.testing.assert_array_equal(
                f["raw/noise_estimates/xyz"][:],
                raw_data["noise_estimates"]["xyz"],
            )

    def test_ltts_present(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            assert "ltts" in f["raw"]
            assert "12" in f["raw/ltts"]
            assert "13" in f["raw/ltts"]

    def test_ltt_times_present(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            assert "times" in f["raw/ltts"]
            np.testing.assert_array_equal(f["raw/ltts/times"][:], raw_data["ltt_times"])

    def test_ltt_derivatives_present(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            assert "derivatives" in f["raw/ltts"]
            assert "12" in f["raw/ltts/derivatives"]

    def test_orbit_positions_correct(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            assert "orbits" in f["raw"]
            assert "positions" in f["raw/orbits"]
            np.testing.assert_array_equal(
                f["raw/orbits/positions"][:], raw_data["orbits"]
            )

    def test_orbit_velocities_present(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            assert "velocities" in f["raw/orbits"]

    def test_orbit_times_present(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            assert "times" in f["raw/orbits"]

    def test_metadata_laser_frequency(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            assert f["raw/metadata"].attrs["laser_frequency"] == pytest.approx(2.816e14)

    def test_metadata_pipeline_names(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            names = json.loads(f["raw/metadata"].attrs["pipeline_names"])
        assert "emri" in names
        assert "noise" in names


# ─────────────────────────────────────────────────────────────────────────────
# Tests: load_processed (round-trip with write)
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadProcessed:
    """Round-trip tests: write then load_processed must recover the original data."""

    def test_returns_dict(self, tmp_path, two_segments):
        """load_processed must return a dict."""
        out = tmp_path / "out.h5"
        write(out, two_segments)
        result = load_processed(out)
        assert isinstance(result, dict)

    def test_segment_keys_match(self, tmp_path, two_segments):
        """Returned keys must match the segments that were written."""
        out = tmp_path / "out.h5"
        write(out, two_segments)
        result = load_processed(out)
        assert set(result.keys()) == set(two_segments.keys())

    def test_values_are_signal_processors(self, tmp_path, two_segments):
        """Every value must be a SignalProcessor instance."""
        out = tmp_path / "out.h5"
        write(out, two_segments)
        result = load_processed(out)
        for sp in result.values():
            assert isinstance(sp, SignalProcessor)

    def test_fs_round_trips(self, tmp_path, two_segments):
        """Sampling frequency must survive the write/read cycle."""
        out = tmp_path / "out.h5"
        write(out, two_segments)
        result = load_processed(out)
        for key, sp in two_segments.items():
            assert result[key].fs == pytest.approx(sp.fs)

    def test_t0_round_trips(self, tmp_path, two_segments):
        """t0 must survive the write/read cycle."""
        out = tmp_path / "out.h5"
        write(out, two_segments)
        result = load_processed(out)
        for key, sp in two_segments.items():
            assert result[key].t0 == pytest.approx(sp.t0)

    def test_t0_none_round_trips(self, tmp_path):
        """t0=None must be restored as None (stored as NaN)."""
        sp = SignalProcessor({"X": np.zeros(100)}, fs=1.0, t0=None)
        out = tmp_path / "out.h5"
        write(out, {"segment0": sp})
        result = load_processed(out)
        assert result["segment0"].t0 is None

    def test_channel_data_round_trips(self, tmp_path, two_segments):
        """Channel arrays must be numerically identical after write/read."""
        out = tmp_path / "out.h5"
        write(out, two_segments)
        result = load_processed(out)
        for key, sp in two_segments.items():
            for ch in sp.channels:
                np.testing.assert_array_equal(result[key]._data[ch], sp._data[ch])

    def test_channels_list_round_trips(self, tmp_path, two_segments):
        """Channel names must be restored in the correct order."""
        out = tmp_path / "out.h5"
        write(out, two_segments)
        result = load_processed(out)
        for key, sp in two_segments.items():
            assert set(result[key].channels) == set(sp.channels)

    def test_N_round_trips(self, tmp_path, two_segments):
        """Sample count N must be consistent after round-trip."""
        out = tmp_path / "out.h5"
        write(out, two_segments)
        result = load_processed(out)
        for key, sp in two_segments.items():
            assert result[key].N == sp.N

    def test_raises_for_non_mojitoprocessor_file(self, tmp_path):
        """load_processed must raise ValueError when '/processed' group is absent."""
        out = tmp_path / "other.h5"
        with h5py.File(out, "w") as f:
            f.create_dataset("data", data=np.zeros(10))
        with pytest.raises(ValueError, match="processed"):
            load_processed(out)
