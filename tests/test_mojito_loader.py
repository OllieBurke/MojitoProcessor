"""Unit tests for MojitoProcessor.mojito_loader."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from MojitoProcessor.mojito_loader import (
    LogFrequencySampling,
    MojitoData,
    OrbitData,
    TimeSampling,
    investigate_mojito_file_structure,
    load_mojito_l1,
    load_orbits,
    truncate_mojito_data,
    truncate_orbit_data,
)

# =============================================================================
# TimeSampling
# =============================================================================


class TestTimeSampling:
    def test_stores_t0(self):
        s = TimeSampling(t0=10.0, dt=0.25, size=100)
        assert s.t0 == 10.0

    def test_stores_dt(self):
        s = TimeSampling(t0=0.0, dt=0.25, size=100)
        assert s.dt == 0.25

    def test_stores_size(self):
        s = TimeSampling(t0=0.0, dt=0.25, size=100)
        assert s.size == 100

    def test_derives_fs(self):
        s = TimeSampling(t0=0.0, dt=0.25, size=100)
        assert s.fs == pytest.approx(4.0)

    def test_derives_duration(self):
        s = TimeSampling(t0=0.0, dt=0.25, size=100)
        assert s.duration == pytest.approx(25.0)

    def test_t0_is_float(self):
        s = TimeSampling(t0=5, dt=1, size=10)
        assert isinstance(s.t0, float)

    def test_dt_is_float(self):
        s = TimeSampling(t0=0, dt=1, size=10)
        assert isinstance(s.dt, float)

    def test_size_is_int(self):
        s = TimeSampling(t0=0.0, dt=1.0, size=10.9)
        assert isinstance(s.size, int)
        assert s.size == 10

    def test_t_array_length(self):
        s = TimeSampling(t0=0.0, dt=0.5, size=50)
        assert len(s.t()) == 50

    def test_t_array_starts_at_t0(self):
        t0 = 1000.0
        s = TimeSampling(t0=t0, dt=0.25, size=100)
        assert s.t()[0] == pytest.approx(t0)

    def test_t_array_step_equals_dt(self):
        dt = 0.5
        s = TimeSampling(t0=0.0, dt=dt, size=20)
        diffs = np.diff(s.t())
        assert_array_almost_equal(diffs, dt)

    def test_t_array_last_value(self):
        s = TimeSampling(t0=0.0, dt=1.0, size=5)
        assert s.t()[-1] == pytest.approx(4.0)

    def test_large_t0(self):
        """TCB timestamps are around 1e9 seconds."""
        s = TimeSampling(t0=1e9, dt=0.25, size=10)
        assert s.t0 == pytest.approx(1e9)
        assert s.t()[0] == pytest.approx(1e9)

    def test_repr_contains_dt(self):
        s = TimeSampling(t0=0.0, dt=0.25, size=100)
        assert "dt=0.25" in repr(s)

    def test_repr_contains_size(self):
        s = TimeSampling(t0=0.0, dt=0.25, size=100)
        assert "100" in repr(s)


# =============================================================================
# LogFrequencySampling
# =============================================================================


class TestLogFrequencySampling:
    def test_stores_fmin(self):
        lfs = LogFrequencySampling(fmin=1e-4, fmax=1.0, size=50)
        assert lfs.fmin == pytest.approx(1e-4)

    def test_stores_fmax(self):
        lfs = LogFrequencySampling(fmin=1e-4, fmax=1.0, size=50)
        assert lfs.fmax == pytest.approx(1.0)

    def test_stores_size(self):
        lfs = LogFrequencySampling(fmin=1e-4, fmax=1.0, size=50)
        assert lfs.size == 50

    def test_fmin_is_float(self):
        lfs = LogFrequencySampling(fmin=1, fmax=100, size=10)
        assert isinstance(lfs.fmin, float)

    def test_size_is_int(self):
        lfs = LogFrequencySampling(fmin=1.0, fmax=100.0, size=10.7)
        assert isinstance(lfs.size, int)
        assert lfs.size == 10

    def test_f_array_length(self):
        lfs = LogFrequencySampling(fmin=1e-4, fmax=1.0, size=50)
        assert len(lfs.f()) == 50

    def test_f_array_first_element(self):
        lfs = LogFrequencySampling(fmin=1e-4, fmax=1.0, size=50)
        assert lfs.f()[0] == pytest.approx(1e-4)

    def test_f_array_last_element(self):
        lfs = LogFrequencySampling(fmin=1e-4, fmax=1.0, size=50)
        assert lfs.f()[-1] == pytest.approx(1.0)

    def test_f_array_is_log_spaced(self):
        """Adjacent ratios should be equal (log-uniform spacing)."""
        lfs = LogFrequencySampling(fmin=1e-3, fmax=1.0, size=10)
        f = lfs.f()
        ratios = f[1:] / f[:-1]
        assert_array_almost_equal(ratios, ratios[0])

    def test_f_array_monotonically_increasing(self):
        lfs = LogFrequencySampling(fmin=1e-5, fmax=0.1, size=20)
        assert np.all(np.diff(lfs.f()) > 0)

    def test_repr_contains_size(self):
        lfs = LogFrequencySampling(fmin=1e-4, fmax=1.0, size=50)
        assert "50" in repr(lfs)


# =============================================================================
# MojitoData
# =============================================================================


class TestMojitoData:
    def test_tdis_initialises_empty(self):
        assert MojitoData().tdis == {}

    def test_sampling_initialises_none(self):
        assert MojitoData().sampling is None

    def test_noise_freqs_initialises_none(self):
        assert MojitoData().noise_freqs is None

    def test_noise_cov_xyz_initialises_none(self):
        assert MojitoData().noise_cov_xyz is None

    def test_noise_cov_aet_initialises_none(self):
        assert MojitoData().noise_cov_aet is None

    def test_noise_psds_initialises_empty(self):
        assert MojitoData().noise_psds == {}

    def test_metadata_initialises_empty(self):
        assert MojitoData().metadata == {}

    def test_repr_contains_class_name(self, sample_mojito_data):
        assert "MojitoData" in repr(sample_mojito_data)

    def test_repr_contains_tdi_channels(self, sample_mojito_data):
        assert "TDI channels" in repr(sample_mojito_data)

    def test_repr_contains_filepath(self, sample_mojito_data):
        assert sample_mojito_data.filepath in repr(sample_mojito_data)


# =============================================================================
# OrbitData
# =============================================================================


class TestOrbitData:
    def test_ltts_initialises_empty(self):
        assert OrbitData().ltts == {}

    def test_ltt_derivatives_initialises_empty(self):
        assert OrbitData().ltt_derivatives == {}

    def test_sc_positions_initialises_empty(self):
        assert OrbitData().sc_positions == {}

    def test_sc_velocities_initialises_empty(self):
        assert OrbitData().sc_velocities == {}

    def test_ltt_sampling_initialises_none(self):
        assert OrbitData().ltt_sampling is None

    def test_orbit_sampling_initialises_none(self):
        assert OrbitData().orbit_sampling is None

    def test_repr_contains_class_name(self, sample_orbit_data):
        assert "OrbitData" in repr(sample_orbit_data)

    def test_repr_contains_ltt_links(self, sample_orbit_data):
        assert "LTT links" in repr(sample_orbit_data)


# =============================================================================
# load_mojito_l1
# =============================================================================


class TestLoadMojitoL1:
    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_mojito_l1("/nonexistent/path/file.h5")

    def test_string_path_accepted(self, mojito_h5_file):
        data = load_mojito_l1(str(mojito_h5_file))
        assert isinstance(data, MojitoData)

    def test_returns_mojito_data(self, mojito_h5_file):
        assert isinstance(load_mojito_l1(mojito_h5_file), MojitoData)

    def test_filepath_stored(self, mojito_h5_file):
        data = load_mojito_l1(mojito_h5_file)
        assert data.filepath == mojito_h5_file

    def test_xyz_channels_loaded(self, mojito_h5_file):
        data = load_mojito_l1(mojito_h5_file)
        for ch in ["X", "Y", "Z"]:
            assert ch in data.tdis

    def test_aet_channels_loaded_when_present(self, mojito_h5_file):
        data = load_mojito_l1(mojito_h5_file)
        for ch in ["A", "E", "T"]:
            assert ch in data.tdis

    def test_aet_channels_absent_when_not_in_file(self, mojito_h5_file_no_aet):
        data = load_mojito_l1(mojito_h5_file_no_aet)
        for ch in ["A", "E", "T"]:
            assert ch not in data.tdis

    def test_tdi_channel_lengths_match_n_samples(self, mojito_h5_file):
        data = load_mojito_l1(mojito_h5_file)
        for arr in data.tdis.values():
            assert len(arr) == data.n_samples

    def test_n_samples_correct(self, mojito_h5_file):
        assert load_mojito_l1(mojito_h5_file).n_samples == 200

    def test_dt_correct(self, mojito_h5_file):
        assert load_mojito_l1(mojito_h5_file).dt == pytest.approx(0.25)

    def test_fs_correct(self, mojito_h5_file):
        assert load_mojito_l1(mojito_h5_file).fs == pytest.approx(4.0)

    def test_t0_correct(self, mojito_h5_file):
        assert load_mojito_l1(mojito_h5_file).t0 == pytest.approx(0.0)

    def test_duration_correct(self, mojito_h5_file):
        data = load_mojito_l1(mojito_h5_file)
        assert data.duration == pytest.approx(200 * 0.25)

    def test_t_tdi_length(self, mojito_h5_file):
        data = load_mojito_l1(mojito_h5_file)
        assert len(data.t_tdi) == data.n_samples

    def test_t_tdi_starts_at_t0(self, mojito_h5_file):
        data = load_mojito_l1(mojito_h5_file)
        assert data.t_tdi[0] == pytest.approx(data.t0)

    def test_noise_freqs_loaded(self, mojito_h5_file):
        data = load_mojito_l1(mojito_h5_file)
        assert data.noise_freqs is not None
        assert len(data.noise_freqs) == 15

    def test_noise_cov_xyz_shape(self, mojito_h5_file):
        data = load_mojito_l1(mojito_h5_file)
        assert data.noise_cov_xyz.shape == (2, 15, 3, 3)

    def test_aet_cov_loaded_when_present(self, mojito_h5_file):
        assert load_mojito_l1(mojito_h5_file).noise_cov_aet is not None

    def test_aet_cov_none_when_absent(self, mojito_h5_file_no_aet):
        assert load_mojito_l1(mojito_h5_file_no_aet).noise_cov_aet is None

    def test_aet_psds_computed_from_diagonal(self, mojito_h5_file):
        data = load_mojito_l1(mojito_h5_file)
        for ch in ["A", "E", "T"]:
            assert ch in data.noise_psds
            assert len(data.noise_psds[ch]) == 15

    def test_metadata_pipeline_name(self, mojito_h5_file):
        data = load_mojito_l1(mojito_h5_file)
        assert data.metadata["pipeline_name"] == "test_pipeline"


# =============================================================================
# load_orbits
# =============================================================================


class TestLoadOrbits:
    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_orbits("/nonexistent/path/orbits.h5")

    def test_string_path_accepted(self, orbit_h5_file):
        assert isinstance(load_orbits(str(orbit_h5_file)), OrbitData)

    def test_returns_orbit_data(self, orbit_h5_file):
        assert isinstance(load_orbits(orbit_h5_file), OrbitData)

    def test_filepath_stored(self, orbit_h5_file):
        assert load_orbits(orbit_h5_file).filepath == orbit_h5_file

    def test_all_six_ltt_links_loaded(self, orbit_h5_file):
        data = load_orbits(orbit_h5_file)
        for link in ["12", "13", "21", "23", "31", "32"]:
            assert link in data.ltts

    def test_all_six_ltt_derivatives_loaded(self, orbit_h5_file):
        data = load_orbits(orbit_h5_file)
        for link in ["12", "13", "21", "23", "31", "32"]:
            assert link in data.ltt_derivatives

    def test_ltt_array_lengths(self, orbit_h5_file):
        data = load_orbits(orbit_h5_file)
        for ltt in data.ltts.values():
            assert len(ltt) == data.ltt_n_samples

    def test_ltt_n_samples_correct(self, orbit_h5_file):
        assert load_orbits(orbit_h5_file).ltt_n_samples == 150

    def test_ltt_dt_correct(self, orbit_h5_file):
        assert load_orbits(orbit_h5_file).ltt_dt == pytest.approx(0.25)

    def test_ltt_fs_correct(self, orbit_h5_file):
        assert load_orbits(orbit_h5_file).ltt_fs == pytest.approx(4.0)

    def test_all_spacecraft_positions_loaded(self, orbit_h5_file):
        data = load_orbits(orbit_h5_file)
        for sc in ["1", "2", "3"]:
            assert sc in data.sc_positions

    def test_all_spacecraft_velocities_loaded(self, orbit_h5_file):
        data = load_orbits(orbit_h5_file)
        for sc in ["1", "2", "3"]:
            assert sc in data.sc_velocities

    def test_orbit_n_samples_correct(self, orbit_h5_file):
        assert load_orbits(orbit_h5_file).orbit_n_samples == 20

    def test_orbit_dt_correct(self, orbit_h5_file):
        assert load_orbits(orbit_h5_file).orbit_dt == pytest.approx(3600.0)

    def test_t_ltt_length(self, orbit_h5_file):
        data = load_orbits(orbit_h5_file)
        assert len(data.t_ltt) == data.ltt_n_samples

    def test_metadata_pipeline_name(self, orbit_h5_file):
        data = load_orbits(orbit_h5_file)
        assert data.metadata["pipeline_name"] == "orbit_pipeline"


# =============================================================================
# truncate_mojito_data
# =============================================================================


class TestTruncateMojitoData:
    def test_raises_if_both_params_given(self, sample_mojito_data):
        with pytest.raises(ValueError, match="not both"):
            truncate_mojito_data(
                sample_mojito_data, duration_seconds=100.0, n_samples=50
            )

    def test_returns_same_object_if_no_params(self, sample_mojito_data):
        result = truncate_mojito_data(sample_mojito_data)
        assert result is sample_mojito_data

    def test_returns_new_object(self, sample_mojito_data):
        result = truncate_mojito_data(sample_mojito_data, n_samples=100)
        assert result is not sample_mojito_data

    def test_truncate_by_n_samples_count(self, sample_mojito_data):
        result = truncate_mojito_data(sample_mojito_data, n_samples=100)
        assert result.n_samples == 100

    def test_truncate_by_n_samples_array_lengths(self, sample_mojito_data):
        result = truncate_mojito_data(sample_mojito_data, n_samples=100)
        for arr in result.tdis.values():
            assert len(arr) == 100

    def test_truncate_by_duration(self, sample_mojito_data):
        result = truncate_mojito_data(sample_mojito_data, duration_seconds=50.0)
        expected = int(50.0 / sample_mojito_data.dt)
        assert result.n_samples == expected

    def test_raises_if_n_samples_exceeds_original(self, sample_mojito_data):
        with pytest.raises(ValueError):
            truncate_mojito_data(
                sample_mojito_data, n_samples=sample_mojito_data.n_samples + 1
            )

    def test_raises_if_n_samples_zero(self, sample_mojito_data):
        with pytest.raises(ValueError):
            truncate_mojito_data(sample_mojito_data, n_samples=0)

    def test_raises_if_n_samples_negative(self, sample_mojito_data):
        with pytest.raises(ValueError):
            truncate_mojito_data(sample_mojito_data, n_samples=-5)

    def test_tdi_values_are_prefix_of_original(self, sample_mojito_data):
        n = 50
        result = truncate_mojito_data(sample_mojito_data, n_samples=n)
        for ch in result.tdis:
            assert_array_equal(result.tdis[ch], sample_mojito_data.tdis[ch][:n])

    def test_duration_updated(self, sample_mojito_data):
        n = 100
        result = truncate_mojito_data(sample_mojito_data, n_samples=n)
        assert result.duration == pytest.approx(n * sample_mojito_data.dt)

    def test_t0_unchanged(self, sample_mojito_data):
        result = truncate_mojito_data(sample_mojito_data, n_samples=100)
        assert result.t0 == pytest.approx(sample_mojito_data.t0)

    def test_dt_unchanged(self, sample_mojito_data):
        result = truncate_mojito_data(sample_mojito_data, n_samples=100)
        assert result.dt == pytest.approx(sample_mojito_data.dt)

    def test_fs_unchanged(self, sample_mojito_data):
        result = truncate_mojito_data(sample_mojito_data, n_samples=100)
        assert result.fs == pytest.approx(sample_mojito_data.fs)

    def test_t_tdi_updated(self, sample_mojito_data):
        result = truncate_mojito_data(sample_mojito_data, n_samples=100)
        assert len(result.t_tdi) == 100

    def test_noise_freqs_copied(self, sample_mojito_data):
        result = truncate_mojito_data(sample_mojito_data, n_samples=100)
        assert_array_equal(result.noise_freqs, sample_mojito_data.noise_freqs)

    def test_noise_freqs_is_independent_copy(self, sample_mojito_data):
        result = truncate_mojito_data(sample_mojito_data, n_samples=100)
        result.noise_freqs[0] = -999.0
        assert sample_mojito_data.noise_freqs[0] != -999.0

    def test_noise_cov_xyz_shape_preserved(self, sample_mojito_data):
        result = truncate_mojito_data(sample_mojito_data, n_samples=100)
        assert result.noise_cov_xyz.shape == sample_mojito_data.noise_cov_xyz.shape

    def test_metadata_copied(self, sample_mojito_data):
        result = truncate_mojito_data(sample_mojito_data, n_samples=100)
        assert result.metadata == sample_mojito_data.metadata

    def test_filepath_preserved(self, sample_mojito_data):
        result = truncate_mojito_data(sample_mojito_data, n_samples=100)
        assert result.filepath == sample_mojito_data.filepath

    def test_sampling_object_updated(self, sample_mojito_data):
        n = 100
        result = truncate_mojito_data(sample_mojito_data, n_samples=n)
        assert result.sampling.size == n
        assert result.sampling.dt == pytest.approx(sample_mojito_data.dt)


# =============================================================================
# truncate_orbit_data
# =============================================================================


class TestTruncateOrbitData:
    def test_raises_if_both_params_given(self, sample_orbit_data):
        with pytest.raises(ValueError, match="not both"):
            truncate_orbit_data(sample_orbit_data, duration_seconds=10.0, n_samples=20)

    def test_returns_same_object_if_no_params(self, sample_orbit_data):
        result = truncate_orbit_data(sample_orbit_data)
        assert result is sample_orbit_data

    def test_returns_new_object(self, sample_orbit_data):
        result = truncate_orbit_data(sample_orbit_data, n_samples=100)
        assert result is not sample_orbit_data

    def test_truncate_by_n_samples_count(self, sample_orbit_data):
        result = truncate_orbit_data(sample_orbit_data, n_samples=100)
        assert result.ltt_n_samples == 100

    def test_truncate_by_n_samples_array_lengths(self, sample_orbit_data):
        result = truncate_orbit_data(sample_orbit_data, n_samples=100)
        for arr in result.ltts.values():
            assert len(arr) == 100

    def test_truncate_by_duration(self, sample_orbit_data):
        dur = 10.0
        result = truncate_orbit_data(sample_orbit_data, duration_seconds=dur)
        expected = int(dur / sample_orbit_data.ltt_dt)
        assert result.ltt_n_samples == expected

    def test_raises_if_n_samples_exceeds_ltt(self, sample_orbit_data):
        with pytest.raises(ValueError):
            truncate_orbit_data(
                sample_orbit_data, n_samples=sample_orbit_data.ltt_n_samples + 1
            )

    def test_raises_if_n_samples_zero(self, sample_orbit_data):
        with pytest.raises(ValueError):
            truncate_orbit_data(sample_orbit_data, n_samples=0)

    def test_raises_if_n_samples_negative(self, sample_orbit_data):
        with pytest.raises(ValueError):
            truncate_orbit_data(sample_orbit_data, n_samples=-1)

    def test_ltt_values_are_prefix_of_original(self, sample_orbit_data):
        n = 50
        result = truncate_orbit_data(sample_orbit_data, n_samples=n)
        for link in result.ltts:
            assert_array_equal(result.ltts[link], sample_orbit_data.ltts[link][:n])

    def test_ltt_derivative_values_are_prefix_of_original(self, sample_orbit_data):
        n = 50
        result = truncate_orbit_data(sample_orbit_data, n_samples=n)
        for link in result.ltt_derivatives:
            assert_array_equal(
                result.ltt_derivatives[link],
                sample_orbit_data.ltt_derivatives[link][:n],
            )

    def test_orbit_n_samples_unchanged(self, sample_orbit_data):
        result = truncate_orbit_data(sample_orbit_data, n_samples=100)
        assert result.orbit_n_samples == sample_orbit_data.orbit_n_samples

    def test_orbit_dt_unchanged(self, sample_orbit_data):
        result = truncate_orbit_data(sample_orbit_data, n_samples=100)
        assert result.orbit_dt == pytest.approx(sample_orbit_data.orbit_dt)

    def test_spacecraft_positions_copied(self, sample_orbit_data):
        result = truncate_orbit_data(sample_orbit_data, n_samples=100)
        for sc in ["1", "2", "3"]:
            assert_array_equal(
                result.sc_positions[sc], sample_orbit_data.sc_positions[sc]
            )

    def test_spacecraft_velocities_copied(self, sample_orbit_data):
        result = truncate_orbit_data(sample_orbit_data, n_samples=100)
        for sc in ["1", "2", "3"]:
            assert_array_equal(
                result.sc_velocities[sc], sample_orbit_data.sc_velocities[sc]
            )

    def test_ltt_duration_updated(self, sample_orbit_data):
        n = 100
        result = truncate_orbit_data(sample_orbit_data, n_samples=n)
        assert result.ltt_duration == pytest.approx(n * sample_orbit_data.ltt_dt)

    def test_t_ltt_updated(self, sample_orbit_data):
        n = 100
        result = truncate_orbit_data(sample_orbit_data, n_samples=n)
        assert len(result.t_ltt) == n

    def test_metadata_copied(self, sample_orbit_data):
        result = truncate_orbit_data(sample_orbit_data, n_samples=100)
        assert result.metadata == sample_orbit_data.metadata


# =============================================================================
# investigate_mojito_file_structure
# =============================================================================


class TestInvestigateMojitoFileStructure:
    def test_runs_without_error(self, mojito_h5_file):
        investigate_mojito_file_structure(mojito_h5_file)

    def test_prints_file_structure_header(self, mojito_h5_file, capsys):
        investigate_mojito_file_structure(mojito_h5_file)
        out = capsys.readouterr().out
        assert "HDF5 File Structure" in out

    def test_prints_file_path(self, mojito_h5_file, capsys):
        investigate_mojito_file_structure(mojito_h5_file)
        out = capsys.readouterr().out
        assert str(mojito_h5_file) in out

    def test_prints_top_level_attributes(self, mojito_h5_file, capsys):
        investigate_mojito_file_structure(mojito_h5_file)
        out = capsys.readouterr().out
        assert "pipeline_name" in out
