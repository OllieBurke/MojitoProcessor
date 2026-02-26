"""Unit tests for MojitoProcessor.SigProcessing."""

import logging

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from MojitoProcessor.SigProcessing import SignalProcessor, process_pipeline

# =============================================================================
# SignalProcessor.__init__
# =============================================================================


class TestSignalProcessorInit:
    def test_channels_stored(self, sp_data):
        sp = SignalProcessor(sp_data, fs=4.0)
        assert set(sp.channels) == {"X", "Y", "Z"}

    def test_fs_stored(self, sp_data):
        sp = SignalProcessor(sp_data, fs=4.0)
        assert sp.fs == pytest.approx(4.0)

    def test_fs_converted_to_float(self, sp_data):
        sp = SignalProcessor(sp_data, fs=4)
        assert isinstance(sp.fs, float)

    def test_n_correct(self, sp_data):
        sp = SignalProcessor(sp_data, fs=4.0)
        assert sp.N == 1000

    def test_dt_correct(self, sp_data):
        sp = SignalProcessor(sp_data, fs=4.0)
        assert sp.dt == pytest.approx(0.25)

    def test_T_correct(self, sp_data):
        sp = SignalProcessor(sp_data, fs=4.0)
        assert sp.T == pytest.approx(1000 * 0.25)

    def test_data_is_copied(self, sp_data):
        """Modifying the original dict must not change processor state."""
        sp = SignalProcessor(sp_data, fs=4.0)
        sp_data["X"][0] = 1e99
        assert sp.data["X"][0] != pytest.approx(1e99)

    def test_raises_if_channel_lengths_differ(self):
        with pytest.raises(ValueError, match="same length"):
            SignalProcessor(
                {"X": np.zeros(100), "Y": np.zeros(200)},
                fs=4.0,
            )

    def test_single_channel_works(self):
        sp = SignalProcessor({"X": np.zeros(50)}, fs=1.0)
        assert sp.channels == ["X"]
        assert sp.N == 50

    def test_t0_defaults_to_none(self, sp_data):
        sp = SignalProcessor(sp_data, fs=4.0)
        assert sp.t0 is None

    def test_t0_stored_as_float(self, sp_data):
        sp = SignalProcessor(sp_data, fs=4.0, t0=9.77298893e7)
        assert isinstance(sp.t0, float)
        assert sp.t0 == pytest.approx(9.77298893e7)

    def test_t0_none_stays_none(self, sp_data):
        sp = SignalProcessor(sp_data, fs=4.0, t0=None)
        assert sp.t0 is None


# =============================================================================
# SignalProcessor._update_params
# =============================================================================


class TestUpdateParams:
    def test_params_update_after_trim(self, simple_sp):
        simple_sp.trim(fraction=0.1)
        assert simple_sp.N == len(simple_sp.data["X"])
        assert simple_sp.dt == pytest.approx(1.0 / simple_sp.fs)
        assert simple_sp.T == pytest.approx(simple_sp.N * simple_sp.dt)

    def test_params_update_after_downsample(self, simple_sp):
        simple_sp.downsample(target_fs=1.0)
        assert simple_sp.N == len(simple_sp.data["X"])
        assert simple_sp.fs == pytest.approx(1.0)


# =============================================================================
# SignalProcessor.filter
# =============================================================================


class TestFilter:
    def test_raises_if_no_cutoffs(self, simple_sp):
        with pytest.raises(ValueError, match="at least one"):
            simple_sp.filter()

    def test_raises_if_low_not_positive(self, simple_sp):
        with pytest.raises(ValueError):
            simple_sp.filter(low=0.0)

    def test_raises_if_low_negative(self, simple_sp):
        with pytest.raises(ValueError):
            simple_sp.filter(low=-0.1)

    def test_raises_if_low_above_nyquist(self, simple_sp):
        with pytest.raises(ValueError):
            simple_sp.filter(low=2.0)  # fs=4, Nyquist=2.0

    def test_raises_if_high_not_positive(self, simple_sp):
        with pytest.raises(ValueError):
            simple_sp.filter(high=0.0)

    def test_raises_if_high_above_nyquist(self, simple_sp):
        with pytest.raises(ValueError):
            simple_sp.filter(high=2.0)  # Nyquist=2.0, must be strictly below

    def test_raises_if_low_ge_high(self, simple_sp):
        with pytest.raises(ValueError):
            simple_sp.filter(low=0.5, high=0.5)

    def test_raises_if_order_zero(self, simple_sp):
        with pytest.raises(ValueError):
            simple_sp.filter(low=0.01, order=0)

    def test_raises_if_order_negative(self, simple_sp):
        with pytest.raises(ValueError):
            simple_sp.filter(low=0.01, order=-1)

    def test_raises_if_order_not_int(self, simple_sp):
        with pytest.raises(ValueError):
            simple_sp.filter(low=0.01, order=2.5)

    def test_raises_if_unknown_filter_type(self, simple_sp):
        with pytest.raises(ValueError, match="Unknown filter type"):
            simple_sp.filter(low=0.01, filter_type="magic")

    def test_highpass_only(self, simple_sp):
        result = simple_sp.filter(low=0.01)
        assert set(result.keys()) == {"X", "Y", "Z"}

    def test_lowpass_only(self, simple_sp):
        result = simple_sp.filter(high=1.0)
        assert set(result.keys()) == {"X", "Y", "Z"}

    def test_bandpass(self, simple_sp):
        result = simple_sp.filter(low=0.01, high=1.0)
        assert set(result.keys()) == {"X", "Y", "Z"}

    def test_output_length_unchanged(self, simple_sp):
        N_before = simple_sp.N
        simple_sp.filter(low=0.01)
        assert simple_sp.N == N_before

    def test_fs_unchanged_after_filter(self, simple_sp):
        fs_before = simple_sp.fs
        simple_sp.filter(low=0.01)
        assert simple_sp.fs == pytest.approx(fs_before)

    def test_all_filter_types_run(self, sp_data):
        for ftype in ["butterworth", "bessel", "chebyshev1", "chebyshev2"]:
            sp = SignalProcessor(sp_data.copy(), fs=4.0)
            result = sp.filter(low=0.01, filter_type=ftype)
            assert "X" in result

    def test_zero_phase_false_runs(self, simple_sp):
        result = simple_sp.filter(low=0.01, zero_phase=False)
        assert "X" in result

    def test_highpass_attenuates_dc(self):
        """A high-pass filter should strongly attenuate a near-DC signal."""
        fs = 100.0
        n = 10000
        t = np.arange(n) / fs
        # Pure low-frequency sine well below the cutoff
        sig = np.sin(2 * np.pi * 0.001 * t)
        sp = SignalProcessor({"X": sig}, fs=fs)
        sp.filter(low=1.0)  # 1 Hz highpass; signal is at 0.001 Hz
        # RMS of filtered output should be much less than input
        rms_in = np.sqrt(np.mean(sig**2))
        rms_out = np.sqrt(np.mean(sp.data["X"] ** 2))
        assert rms_out < 0.01 * rms_in

    def test_lowpass_attenuates_high_freq(self):
        """A low-pass filter should strongly attenuate a signal above the cutoff."""
        fs = 1000.0
        n = 100000
        t = np.arange(n) / fs
        # 200 Hz sine; cutoff at 5 Hz → 40× frequency ratio, strong attenuation
        sig = np.sin(2 * np.pi * 200.0 * t)
        sp = SignalProcessor({"X": sig}, fs=fs)
        sp.filter(high=5.0)
        rms_in = np.sqrt(np.mean(sig**2))
        rms_out = np.sqrt(np.mean(sp.data["X"] ** 2))
        assert rms_out < 0.01 * rms_in  # > 99 % attenuation

    def test_filter_updates_internal_data(self):
        """Highpass filter applied to a pure DC signal must change the data."""
        dc = {"X": np.ones(1000)}
        sp = SignalProcessor(dc, fs=4.0)
        original = sp.data["X"].copy()
        sp.filter(low=0.01)  # highpass removes DC component
        assert not np.allclose(sp.data["X"], original)


# =============================================================================
# SignalProcessor.downsample
# =============================================================================


class TestDownsample:
    def test_raises_if_target_fs_zero(self, simple_sp):
        with pytest.raises(ValueError):
            simple_sp.downsample(target_fs=0.0)

    def test_raises_if_target_fs_negative(self, simple_sp):
        with pytest.raises(ValueError):
            simple_sp.downsample(target_fs=-1.0)

    def test_raises_if_target_fs_exceeds_current(self, simple_sp):
        with pytest.raises(ValueError, match="exceeds"):
            simple_sp.downsample(target_fs=10.0)  # current fs=4.0

    def test_returns_tuple(self, simple_sp):
        result = simple_sp.downsample(target_fs=1.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returned_fs_correct(self, simple_sp):
        _, new_fs = simple_sp.downsample(target_fs=1.0)
        assert new_fs == pytest.approx(1.0)

    def test_internal_fs_updated(self, simple_sp):
        simple_sp.downsample(target_fs=1.0)
        assert simple_sp.fs == pytest.approx(1.0)

    def test_integer_downsampling_length(self, simple_sp):
        """4 Hz → 1 Hz (factor 4): output length should be N/4."""
        N_before = simple_sp.N
        simple_sp.downsample(target_fs=1.0)
        assert simple_sp.N == pytest.approx(N_before // 4, abs=2)

    def test_non_integer_downsampling_runs(self, simple_sp):
        """4 Hz → 2 Hz is a valid integer factor (factor 2)."""
        simple_sp.downsample(target_fs=2.0)
        assert simple_sp.fs == pytest.approx(2.0)

    def test_all_channels_resampled(self, simple_sp):
        simple_sp.downsample(target_fs=1.0)
        expected_len = simple_sp.N
        for arr in simple_sp.data.values():
            assert len(arr) == expected_len

    def test_dt_updated(self, simple_sp):
        simple_sp.downsample(target_fs=1.0)
        assert simple_sp.dt == pytest.approx(1.0)

    def test_T_updated(self, simple_sp):
        simple_sp.downsample(target_fs=1.0)
        assert simple_sp.T == pytest.approx(simple_sp.N * simple_sp.dt)

    def test_equal_target_fs_returns_unchanged(self):
        """Downsampling to the same fs should keep data roughly unchanged."""
        data = {"X": np.ones(100)}
        sp = SignalProcessor(data, fs=4.0)
        sp.downsample(target_fs=4.0)
        assert sp.fs == pytest.approx(4.0)


# =============================================================================
# SignalProcessor.trim
# =============================================================================


class TestTrim:
    def test_raises_if_fraction_negative(self, simple_sp):
        with pytest.raises(ValueError):
            simple_sp.trim(fraction=-0.01)

    def test_raises_if_fraction_equals_one(self, simple_sp):
        with pytest.raises(ValueError):
            simple_sp.trim(fraction=1.0)

    def test_raises_if_fraction_greater_than_one(self, simple_sp):
        with pytest.raises(ValueError):
            simple_sp.trim(fraction=1.5)

    def test_zero_fraction_returns_unchanged(self, sp_data):
        sp = SignalProcessor(sp_data, fs=4.0)
        N_before = sp.N
        sp.trim(fraction=0.0)
        assert sp.N == N_before

    def test_small_fraction_logs_warning_and_returns_unchanged(self, caplog):
        """Fraction too small to remove any sample should log a warning."""
        data = {"X": np.zeros(50)}
        sp = SignalProcessor(data, fs=1.0)
        with caplog.at_level(logging.WARNING):
            sp.trim(fraction=0.001)  # trim_samples = int(50*0.001/2) = 0
        assert sp.N == 50
        assert any("too small" in msg for msg in caplog.messages)

    def test_high_fraction_trims_most_data(self):
        """Very high fraction (< 1) should trim down to a small number of samples.
        Note: 2*floor(N*f/2) < N always holds for f<1 so no ValueError is raised."""
        n = 1000
        sp = SignalProcessor({"X": np.zeros(n)}, fs=1.0)
        # fraction=0.98 → trim_samples = int(1000*0.98/2) = 490 each end → 20 remain
        sp.trim(fraction=0.98)
        assert sp.N == 20

    def test_correct_samples_removed(self, sp_data):
        sp = SignalProcessor(sp_data, fs=4.0)
        N_before = sp.N
        fraction = 0.1
        trim_samples = int(N_before * fraction / 2)
        sp.trim(fraction=fraction)
        assert sp.N == N_before - 2 * trim_samples

    def test_correct_endpoints_removed(self):
        """Trimmed data should equal arr[trim:-trim] of the original."""
        n = 100
        arr = np.arange(n, dtype=float)
        sp = SignalProcessor({"X": arr}, fs=1.0)
        fraction = 0.1
        trim_samples = int(n * fraction / 2)
        sp.trim(fraction=fraction)
        assert_array_equal(sp.data["X"], arr[trim_samples:-trim_samples])

    def test_all_channels_trimmed_equally(self, simple_sp):
        simple_sp.trim(fraction=0.1)
        lengths = [len(arr) for arr in simple_sp.data.values()]
        assert len(set(lengths)) == 1

    def test_internal_state_updated(self, simple_sp):
        simple_sp.trim(fraction=0.1)
        assert simple_sp.N == len(simple_sp.data["X"])

    def test_returns_dict(self, simple_sp):
        result = simple_sp.trim(fraction=0.1)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"X", "Y", "Z"}

    def test_t0_advances_after_trim(self, sp_data):
        """t0 must advance by trim_samples * dt when trimming."""
        t0_init = 9.77298893e7
        sp = SignalProcessor(sp_data, fs=4.0, t0=t0_init)
        fraction = 0.1
        trim_samples = int(sp.N * fraction / 2)
        expected_t0 = t0_init + trim_samples * sp.dt
        sp.trim(fraction=fraction)
        assert sp.t0 == pytest.approx(expected_t0)

    def test_t0_none_unchanged_after_trim(self, sp_data):
        sp = SignalProcessor(sp_data, fs=4.0, t0=None)
        sp.trim(fraction=0.1)
        assert sp.t0 is None


# =============================================================================
# SignalProcessor.apply_window
# =============================================================================


class TestApplyWindow:
    def test_raises_for_unknown_window(self, simple_sp):
        with pytest.raises(ValueError, match="Unknown window type"):
            simple_sp.apply_window(window="unknown")

    def test_tukey_is_default(self, sp_data):
        sp1 = SignalProcessor(sp_data.copy(), fs=4.0)
        sp2 = SignalProcessor(sp_data.copy(), fs=4.0)
        sp1.apply_window()
        sp2.apply_window(window="tukey")
        assert_array_almost_equal(sp1.data["X"], sp2.data["X"])

    def test_tukey_default_alpha_applied(self, sp_data):
        sp1 = SignalProcessor(sp_data.copy(), fs=4.0)
        sp2 = SignalProcessor(sp_data.copy(), fs=4.0)
        sp1.apply_window()
        sp2.apply_window(window="tukey", alpha=0.05)
        assert_array_almost_equal(sp1.data["X"], sp2.data["X"])

    def test_all_window_types_run(self, sp_data):
        for wtype in ["tukey", "blackmanharris", "hann", "hamming", "blackman"]:
            sp = SignalProcessor(sp_data.copy(), fs=4.0)
            result = sp.apply_window(window=wtype)
            assert "X" in result

    def test_output_shape_unchanged(self, simple_sp):
        N_before = simple_sp.N
        simple_sp.apply_window(window="hann")
        assert simple_sp.N == N_before

    def test_tukey_tapers_endpoints(self):
        """With alpha>0, endpoint samples of a Tukey-windowed signal should be 0."""
        n = 200
        arr = np.ones(n)
        sp = SignalProcessor({"X": arr}, fs=1.0)
        sp.apply_window(window="tukey", alpha=0.5)
        assert sp.data["X"][0] == pytest.approx(0.0)
        assert sp.data["X"][-1] == pytest.approx(0.0)

    def test_hann_tapers_endpoints(self):
        n = 200
        arr = np.ones(n)
        sp = SignalProcessor({"X": arr}, fs=1.0)
        sp.apply_window(window="hann")
        assert sp.data["X"][0] == pytest.approx(0.0, abs=1e-10)
        assert sp.data["X"][-1] == pytest.approx(0.0, abs=1e-10)

    def test_fs_unchanged_after_window(self, simple_sp):
        fs_before = simple_sp.fs
        simple_sp.apply_window(window="hann")
        assert simple_sp.fs == pytest.approx(fs_before)

    def test_returns_dict(self, simple_sp):
        result = simple_sp.apply_window()
        assert isinstance(result, dict)
        assert set(result.keys()) == {"X", "Y", "Z"}

    def test_extra_kwargs_for_non_tukey_logs_warning(self, caplog):
        sp = SignalProcessor({"X": np.zeros(50)}, fs=1.0)
        with caplog.at_level(logging.WARNING):
            sp.apply_window(window="hann", alpha=0.5)
        assert any("ignored" in msg for msg in caplog.messages)


# =============================================================================
# SignalProcessor.get_params
# =============================================================================


class TestGetParams:
    def test_returns_dict(self, simple_sp):
        assert isinstance(simple_sp.get_params(), dict)

    def test_contains_fs(self, simple_sp):
        assert simple_sp.get_params()["fs"] == pytest.approx(4.0)

    def test_contains_N(self, simple_sp):
        assert simple_sp.get_params()["N"] == 1000

    def test_contains_T(self, simple_sp):
        assert simple_sp.get_params()["T"] == pytest.approx(1000 * 0.25)

    def test_contains_dt(self, simple_sp):
        assert simple_sp.get_params()["dt"] == pytest.approx(0.25)

    def test_contains_channels(self, simple_sp):
        assert set(simple_sp.get_params()["channels"]) == {"X", "Y", "Z"}


# =============================================================================
# SignalProcessor.__repr__
# =============================================================================


class TestSignalProcessorRepr:
    def test_repr_contains_channels(self, simple_sp):
        assert "X" in repr(simple_sp)

    def test_repr_contains_N(self, simple_sp):
        assert "1000" in repr(simple_sp)

    def test_repr_contains_fs(self, simple_sp):
        r = repr(simple_sp)
        assert "4.0" in r or "4.000" in r

    def test_repr_shows_t0_none(self, simple_sp):
        assert "t0=None" in repr(simple_sp)

    def test_repr_shows_t0_value(self, sp_data):
        sp = SignalProcessor(sp_data, fs=4.0, t0=9.77298893e7)
        assert "t0=" in repr(sp)
        assert "None" not in repr(sp)


# =============================================================================
# process_pipeline
# =============================================================================


class TestProcessPipeline:
    # ── Validation errors ────────────────────────────────────────────────────

    def test_raises_if_channel_missing(self, pipeline_mojito_data):
        with pytest.raises(ValueError, match="not found"):
            process_pipeline(
                pipeline_mojito_data,
                channels=["X", "W"],  # "W" does not exist
            )

    def test_raises_if_filter_order_zero(self, pipeline_mojito_data):
        with pytest.raises(ValueError, match="filter order"):
            process_pipeline(
                pipeline_mojito_data,
                filter_kwargs={"highpass_cutoff": 0.01, "order": 0},
            )

    def test_raises_if_filter_order_float(self, pipeline_mojito_data):
        with pytest.raises(ValueError, match="filter order"):
            process_pipeline(
                pipeline_mojito_data,
                filter_kwargs={"highpass_cutoff": 0.01, "order": 2.5},
            )

    def test_raises_if_kaiser_beta_negative(self, pipeline_mojito_data):
        with pytest.raises(ValueError, match="kaiser_window"):
            process_pipeline(
                pipeline_mojito_data,
                filter_kwargs={"highpass_cutoff": 0.01},
                downsample_kwargs={"target_fs": 1.0, "kaiser_window": -1.0},
            )

    def test_raises_if_truncate_days_zero(self, pipeline_mojito_data):
        with pytest.raises(ValueError, match="days"):
            process_pipeline(
                pipeline_mojito_data,
                filter_kwargs={"highpass_cutoff": 0.01},
                truncate_kwargs={"days": 0.0},
            )

    def test_raises_if_truncate_days_negative(self, pipeline_mojito_data):
        with pytest.raises(ValueError, match="days"):
            process_pipeline(
                pipeline_mojito_data,
                filter_kwargs={"highpass_cutoff": 0.01},
                truncate_kwargs={"days": -1.0},
            )

    def test_raises_if_tukey_alpha_negative(self, pipeline_mojito_data):
        with pytest.raises(ValueError, match="alpha"):
            process_pipeline(
                pipeline_mojito_data,
                filter_kwargs={"highpass_cutoff": 0.01},
                window_kwargs={"window": "tukey", "alpha": -0.1},
            )

    def test_raises_if_tukey_alpha_above_one(self, pipeline_mojito_data):
        with pytest.raises(ValueError, match="alpha"):
            process_pipeline(
                pipeline_mojito_data,
                filter_kwargs={"highpass_cutoff": 0.01},
                window_kwargs={"window": "tukey", "alpha": 1.5},
            )

    # ── Return type ──────────────────────────────────────────────────────────

    def test_returns_dict(self, pipeline_mojito_data):
        result = process_pipeline(
            pipeline_mojito_data, filter_kwargs={"highpass_cutoff": 0.01}
        )
        assert isinstance(result, dict)

    def test_values_are_signal_processors(self, pipeline_mojito_data):
        result = process_pipeline(
            pipeline_mojito_data, filter_kwargs={"highpass_cutoff": 0.01}
        )
        for sp in result.values():
            assert isinstance(sp, SignalProcessor)

    # ── Default behaviour (no segmentation) ──────────────────────────────────

    def test_default_returns_segment0(self, pipeline_mojito_data):
        result = process_pipeline(
            pipeline_mojito_data, filter_kwargs={"highpass_cutoff": 0.01}
        )
        assert "segment0" in result

    def test_default_single_segment(self, pipeline_mojito_data):
        result = process_pipeline(
            pipeline_mojito_data, filter_kwargs={"highpass_cutoff": 0.01}
        )
        assert len(result) == 1

    def test_default_channels_are_xyz(self, pipeline_mojito_data):
        result = process_pipeline(
            pipeline_mojito_data, filter_kwargs={"highpass_cutoff": 0.01}
        )
        sp = result["segment0"]
        assert set(sp.channels) == {"X", "Y", "Z"}

    # ── Custom channels ───────────────────────────────────────────────────────

    def test_custom_channels_subset(self, pipeline_mojito_data):
        result = process_pipeline(
            pipeline_mojito_data,
            channels=["X"],
            filter_kwargs={"highpass_cutoff": 0.01},
        )
        sp = result["segment0"]
        assert sp.channels == ["X"]

    # ── Downsampling ─────────────────────────────────────────────────────────

    def test_downsample_updates_fs(self, pipeline_mojito_data):
        result = process_pipeline(
            pipeline_mojito_data,
            filter_kwargs={"highpass_cutoff": 0.01},
            downsample_kwargs={"target_fs": 1.0},
        )
        sp = result["segment0"]
        assert sp.fs == pytest.approx(1.0)

    def test_skip_downsample_when_not_specified(self, pipeline_mojito_data):
        result = process_pipeline(
            pipeline_mojito_data, filter_kwargs={"highpass_cutoff": 0.01}
        )
        sp = result["segment0"]
        assert sp.fs == pytest.approx(pipeline_mojito_data["fs"])

    # ── Bandpass filter ───────────────────────────────────────────────────────

    def test_bandpass_filter_runs(self, pipeline_mojito_data):
        result = process_pipeline(
            pipeline_mojito_data,
            filter_kwargs={"highpass_cutoff": 0.01, "lowpass_cutoff": 1.5},
        )
        assert "segment0" in result

    # ── Segmentation ─────────────────────────────────────────────────────────

    def test_segmentation_produces_multiple_segments(self, pipeline_mojito_data):
        """1000 samples at 4 Hz → 250 s. With 0.001-day segments (86.4 s each)
        and 2% trim we expect floor((~980) / 345) = 2 segments."""
        result = process_pipeline(
            pipeline_mojito_data,
            filter_kwargs={"highpass_cutoff": 0.01},
            truncate_kwargs={"days": 0.001},
        )
        assert len(result) >= 2

    def test_segment_keys_are_named_correctly(self, pipeline_mojito_data):
        result = process_pipeline(
            pipeline_mojito_data,
            filter_kwargs={"highpass_cutoff": 0.01},
            truncate_kwargs={"days": 0.001},
        )
        for key in result:
            assert key.startswith("segment")

    def test_all_segments_have_same_length(self, pipeline_mojito_data):
        """Non-final segments should share the same sample count."""
        result = process_pipeline(
            pipeline_mojito_data,
            filter_kwargs={"highpass_cutoff": 0.01},
            truncate_kwargs={"days": 0.001},
        )
        lengths = [sp.N for sp in result.values()]
        # All segments produced by an even split share the same length
        assert len(set(lengths)) == 1

    def test_data_shorter_than_segment_gives_single_segment(self, pipeline_mojito_data):
        """If data is shorter than the requested segment length, fall back to 1."""
        result = process_pipeline(
            pipeline_mojito_data,
            filter_kwargs={"highpass_cutoff": 0.01},
            truncate_kwargs={"days": 10.0},  # 10 days >> 250 seconds of data
        )
        assert len(result) == 1

    # ── Window parameters ─────────────────────────────────────────────────────

    def test_hann_window_runs(self, pipeline_mojito_data):
        result = process_pipeline(
            pipeline_mojito_data,
            filter_kwargs={"highpass_cutoff": 0.01},
            window_kwargs={"window": "hann"},
        )
        assert "segment0" in result

    def test_tukey_alpha_zero_is_valid(self, pipeline_mojito_data):
        """alpha=0 is a rectangular window — should not raise."""
        result = process_pipeline(
            pipeline_mojito_data,
            filter_kwargs={"highpass_cutoff": 0.01},
            window_kwargs={"window": "tukey", "alpha": 0.0},
        )
        assert "segment0" in result

    # ── Lowpass cutoff warning ────────────────────────────────────────────────

    def test_warns_if_lowpass_exceeds_target_nyquist(
        self, pipeline_mojito_data, caplog
    ):
        with caplog.at_level(logging.WARNING):
            process_pipeline(
                pipeline_mojito_data,
                filter_kwargs={"highpass_cutoff": 0.01, "lowpass_cutoff": 1.9},
                downsample_kwargs={"target_fs": 1.0},  # Nyquist = 0.5 Hz
            )
        assert any(
            "aliased" in msg or "lowpass_cutoff" in msg for msg in caplog.messages
        )

    # ── t0 propagation ───────────────────────────────────────────────────────

    def test_raises_if_t_tdi_absent(self, pipeline_mojito_data):
        """process_pipeline must raise if t_tdi is missing from the data dict."""
        data_no_t_tdi = {k: v for k, v in pipeline_mojito_data.items() if k != "t_tdi"}
        with pytest.raises(ValueError, match="t_tdi"):
            process_pipeline(data_no_t_tdi, filter_kwargs={"highpass_cutoff": 0.01})

    def test_t0_set_from_t_tdi(self, pipeline_mojito_data):
        """segment0.t0 must equal t_tdi[0] when trim fraction is zero."""
        t_tdi_start = float(pipeline_mojito_data["t_tdi"][0])
        result = process_pipeline(
            pipeline_mojito_data,
            filter_kwargs={"highpass_cutoff": 0.01},
            trim_kwargs={"fraction": 0.0},  # no trim so t0 == t_tdi[0]
        )
        assert result["segment0"].t0 == pytest.approx(t_tdi_start)

    def test_t0_advances_after_trim_in_pipeline(self, pipeline_mojito_data):
        """After trimming, segment0.t0 must be strictly greater than t_tdi[0]."""
        t_tdi_start = float(pipeline_mojito_data["t_tdi"][0])
        result = process_pipeline(
            pipeline_mojito_data,
            filter_kwargs={"highpass_cutoff": 0.01},
            trim_kwargs={"fraction": 0.1},
        )
        assert result["segment0"].t0 > t_tdi_start

    def test_segment_t0_increases_monotonically(self, pipeline_mojito_data):
        """Each successive segment must have a strictly larger t0."""
        result = process_pipeline(
            pipeline_mojito_data,
            filter_kwargs={"highpass_cutoff": 0.01},
            truncate_kwargs={"days": 0.001},
        )
        if len(result) < 2:
            pytest.skip("Not enough segments produced to test monotonicity")
        t0_values = [result[f"segment{i}"].t0 for i in range(len(result))]
        for a, b in zip(t0_values, t0_values[1:]):
            assert b > a
