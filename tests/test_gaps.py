"""Unit tests for MojitoProcessor.gaps."""

import numpy as np
import pytest

from MojitoProcessor.gaps import (
    apply_mask_to_processor,
    apply_raw_mask,
    compute_extended_mask,
    extract_clean_segments,
    taper_mask,
)
from MojitoProcessor.process.sigprocess import SignalProcessor

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

FS_RAW = 4.0  # Hz — raw sampling rate
FS_DS = 1.0  # Hz — target (downsampled) rate
N_RAW = 4000  # samples at raw rate
TRIM_FRAC = 0.02  # fraction trimmed from each end


def _make_sp(n: int = 200, fs: float = FS_DS) -> SignalProcessor:
    """Minimal SignalProcessor with XYZ channels."""
    rng = np.random.default_rng(0)
    data = {ch: rng.standard_normal(n) for ch in ["X", "Y", "Z"]}
    return SignalProcessor(data, fs=fs, t0=0.0)


def _make_raw_data(n_tdi: int = N_RAW) -> dict:
    """Minimal raw data dict (like load_file output)."""
    rng = np.random.default_rng(1)
    return {
        "tdis": {
            ch: rng.standard_normal(n_tdi) for ch in ["X", "Y", "Z", "A", "E", "T"]
        },
        "fs": FS_RAW,
        "t_tdi": np.arange(n_tdi) / FS_RAW,
    }


def _make_smoothed_mask(n: int = N_RAW, gap_fraction: float = 0.1) -> np.ndarray:
    """Binary mask with a single central gap."""
    mask = np.ones(n, dtype=float)
    gap_start = int(n * 0.45)
    gap_end = int(n * 0.45 + n * gap_fraction)
    mask[gap_start:gap_end] = 0.0
    return mask


def _pipeline_kwargs():
    return (
        {"highpass_cutoff": 0.05, "lowpass_cutoff": 0.4, "order": 2},
        {"target_fs": FS_DS},
        {"fraction": TRIM_FRAC},
    )


# ─────────────────────────────────────────────────────────────────────────────
# apply_raw_mask
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyRawMask:
    def test_returns_dict(self):
        data = _make_raw_data()
        mask = np.ones(N_RAW)
        result = apply_raw_mask(data, mask)
        assert isinstance(result, dict)

    def test_does_not_mutate_original(self):
        data = _make_raw_data()
        original_x = data["tdis"]["X"].copy()
        mask = np.zeros(N_RAW)
        apply_raw_mask(data, mask)
        np.testing.assert_array_equal(data["tdis"]["X"], original_x)

    def test_zero_mask_zeros_channels(self):
        data = _make_raw_data()
        mask = np.zeros(N_RAW)
        result = apply_raw_mask(data, mask)
        for ch in ["X", "Y", "Z", "A", "E", "T"]:
            np.testing.assert_array_equal(result["tdis"][ch], 0.0)

    def test_ones_mask_preserves_channels(self):
        data = _make_raw_data()
        mask = np.ones(N_RAW)
        result = apply_raw_mask(data, mask)
        for ch in ["X", "Y", "Z", "A", "E", "T"]:
            np.testing.assert_array_equal(result["tdis"][ch], data["tdis"][ch])

    def test_mask_applied_element_wise(self):
        data = _make_raw_data()
        mask = _make_smoothed_mask()
        result = apply_raw_mask(data, mask)
        np.testing.assert_array_equal(result["tdis"]["X"], data["tdis"]["X"] * mask)

    def test_non_tdi_keys_preserved(self):
        data = _make_raw_data()
        data["metadata"] = {"laser_frequency": 2.816e14}
        mask = np.ones(N_RAW)
        result = apply_raw_mask(data, mask)
        assert result["metadata"] is not data["metadata"]  # deep copy
        assert result["metadata"]["laser_frequency"] == pytest.approx(2.816e14)

    def test_partial_channel_dict(self):
        """Only channels present in 'tdis' are masked; missing ones are ignored."""
        data = {"tdis": {"X": np.ones(10), "Y": np.ones(10)}, "fs": 1.0}
        mask = np.zeros(10)
        result = apply_raw_mask(data, mask)
        assert "X" in result["tdis"]
        assert "Y" in result["tdis"]


# ─────────────────────────────────────────────────────────────────────────────
# apply_mask_to_processor
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyMaskToProcessor:
    def test_returns_signal_processor(self):
        sp = _make_sp()
        mask = np.ones(sp.N)
        result = apply_mask_to_processor(sp, mask)
        assert isinstance(result, SignalProcessor)

    def test_does_not_mutate_original(self):
        sp = _make_sp()
        original_x = sp._data["X"].copy()
        apply_mask_to_processor(sp, np.zeros(sp.N))
        np.testing.assert_array_equal(sp._data["X"], original_x)

    def test_zero_mask_zeros_all_channels(self):
        sp = _make_sp()
        result = apply_mask_to_processor(sp, np.zeros(sp.N))
        for ch in result.channels:
            np.testing.assert_array_equal(result._data[ch], 0.0)

    def test_ones_mask_preserves_data(self):
        sp = _make_sp()
        result = apply_mask_to_processor(sp, np.ones(sp.N))
        for ch in sp.channels:
            np.testing.assert_array_equal(result._data[ch], sp._data[ch])

    def test_fs_preserved(self):
        sp = _make_sp(fs=2.0)
        result = apply_mask_to_processor(sp, np.ones(sp.N))
        assert result.fs == pytest.approx(2.0)

    def test_t0_preserved(self):
        sp = _make_sp()
        sp.t0 = 1234.5
        result = apply_mask_to_processor(sp, np.ones(sp.N))
        assert result.t0 == pytest.approx(1234.5)

    def test_channels_preserved(self):
        sp = _make_sp()
        result = apply_mask_to_processor(sp, np.ones(sp.N))
        assert result.channels == sp.channels

    def test_length_mismatch_raises(self):
        sp = _make_sp(n=100)
        with pytest.raises(ValueError, match="mask length"):
            apply_mask_to_processor(sp, np.ones(99))

    def test_mask_applied_element_wise(self):
        sp = _make_sp(n=50)
        mask = np.random.default_rng(7).uniform(0, 1, 50)
        result = apply_mask_to_processor(sp, mask)
        np.testing.assert_allclose(result._data["X"], sp._data["X"] * mask)


# ─────────────────────────────────────────────────────────────────────────────
# compute_extended_mask
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeExtendedMask:
    """Integration-style tests: run the full leakage calculation."""

    def _run(self, **kwargs):
        filter_kw, ds_kw, trim_kw = _pipeline_kwargs()
        sp = _make_sp(n=int(N_RAW / (FS_RAW / FS_DS) * (1 - TRIM_FRAC)))
        mask = _make_smoothed_mask()
        return compute_extended_mask(
            mask, sp, filter_kw, ds_kw, trim_kw, fs_raw=FS_RAW, **kwargs
        )

    def test_returns_tuple_of_two(self):
        result = self._run()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_extended_mask_length_matches_sp(self):
        sp = _make_sp(n=int(N_RAW / (FS_RAW / FS_DS) * (1 - TRIM_FRAC)))
        filter_kw, ds_kw, trim_kw = _pipeline_kwargs()
        mask = _make_smoothed_mask()
        ext, _ = compute_extended_mask(
            mask, sp, filter_kw, ds_kw, trim_kw, fs_raw=FS_RAW
        )
        assert len(ext) == sp.N

    def test_gap_contamination_length_matches_sp(self):
        sp = _make_sp(n=int(N_RAW / (FS_RAW / FS_DS) * (1 - TRIM_FRAC)))
        filter_kw, ds_kw, trim_kw = _pipeline_kwargs()
        mask = _make_smoothed_mask()
        _, cont = compute_extended_mask(
            mask, sp, filter_kw, ds_kw, trim_kw, fs_raw=FS_RAW
        )
        assert len(cont) == sp.N

    def test_extended_mask_binary(self):
        ext, _ = self._run()
        unique = np.unique(ext)
        assert set(unique).issubset({0.0, 1.0})

    def test_extended_mask_has_ones(self):
        """There must be clean samples somewhere."""
        ext, _ = self._run()
        assert ext.sum() > 0

    def test_extended_mask_has_zeros(self):
        """There must be excluded samples (gap + leakage).

        Use min_clean_hours=0 so binary closing does not merge the entire
        (short) test dataset into a single clean region.
        """
        ext, _ = self._run(min_clean_hours=0.0)
        assert (ext == 0.0).sum() > 0

    def test_high_threshold_larger_mask(self):
        """A very high threshold should exclude fewer samples than a low one."""
        sp = _make_sp(n=int(N_RAW / (FS_RAW / FS_DS) * (1 - TRIM_FRAC)))
        filter_kw, ds_kw, trim_kw = _pipeline_kwargs()
        mask = _make_smoothed_mask()
        ext_tight, _ = compute_extended_mask(
            mask,
            sp,
            filter_kw,
            ds_kw,
            trim_kw,
            fs_raw=FS_RAW,
            contamination_threshold=1e-10,
        )
        ext_loose, _ = compute_extended_mask(
            mask,
            sp,
            filter_kw,
            ds_kw,
            trim_kw,
            fs_raw=FS_RAW,
            contamination_threshold=1.0,
        )
        # tight threshold → more zeros (smaller mask sum)
        assert ext_tight.sum() <= ext_loose.sum()

    def test_no_gap_mask_all_ones(self):
        """A flat all-ones mask should produce an all-ones extended mask."""
        sp = _make_sp(n=int(N_RAW / (FS_RAW / FS_DS) * (1 - TRIM_FRAC)))
        filter_kw, ds_kw, trim_kw = _pipeline_kwargs()
        all_ones = np.ones(N_RAW)
        ext, cont = compute_extended_mask(
            all_ones,
            sp,
            filter_kw,
            ds_kw,
            trim_kw,
            fs_raw=FS_RAW,
            contamination_threshold=1e-4,
        )
        np.testing.assert_array_equal(ext, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# taper_mask
# ─────────────────────────────────────────────────────────────────────────────


class TestTaperMask:
    def _make_binary_mask(self, n: int = 1000, fs: float = 1.0) -> tuple:
        """Return (mask, sp) with a central gap region."""
        mask = np.ones(n, dtype=float)
        mask[400:600] = 0.0
        sp = _make_sp(n=n, fs=fs)
        return mask, sp

    def test_returns_array(self):
        mask, sp = self._make_binary_mask()
        result = taper_mask(mask, sp, taper_hours=0.0, edge_taper_hours=0.0)
        assert isinstance(result, np.ndarray)

    def test_same_length_as_input(self):
        mask, sp = self._make_binary_mask(n=500)
        result = taper_mask(mask, sp, taper_hours=0.0, edge_taper_hours=0.0)
        assert len(result) == 500

    def test_does_not_mutate_input(self):
        mask, sp = self._make_binary_mask()
        original = mask.copy()
        taper_mask(mask, sp, taper_hours=10.0 / 3600, edge_taper_hours=0.0)
        np.testing.assert_array_equal(mask, original)

    def test_zero_taper_preserves_ones(self):
        """With both tapers=0, the output equals the input."""
        mask, sp = self._make_binary_mask()
        result = taper_mask(mask, sp, taper_hours=0.0, edge_taper_hours=0.0)
        np.testing.assert_array_equal(result, mask)

    def test_gap_edges_tapered(self):
        """Samples immediately adjacent to the gap must be < 1.0 after tapering."""
        n, fs = 1000, 1.0
        mask = np.ones(n, dtype=float)
        mask[400:600] = 0.0
        sp = _make_sp(n=n, fs=fs)
        # 10 samples of taper = 10 seconds at 1 Hz
        result = taper_mask(mask, sp, taper_hours=10.0 / 3600, edge_taper_hours=0.0)
        # Just before the gap: falling taper
        assert result[390] < 1.0
        # Just after the gap: rising taper
        assert result[605] < 1.0
        # Far from gaps and edges: still 1.0
        assert result[200] == pytest.approx(1.0)

    def test_edge_taper_applied(self):
        """First and last samples must be zero when edge_taper is non-zero."""
        mask, sp = self._make_binary_mask(n=1000, fs=1.0)
        # 1 sample = 1/3600 hours
        result = taper_mask(mask, sp, taper_hours=0.0, edge_taper_hours=10.0 / 3600)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(0.0)

    def test_gap_region_remains_zero(self):
        """The gap interior should remain zero after tapering."""
        mask, sp = self._make_binary_mask(n=1000, fs=1.0)
        # taper of 10 samples, gap is 200 samples wide — interior stays zero
        result = taper_mask(mask, sp, taper_hours=10.0 / 3600, edge_taper_hours=0.0)
        np.testing.assert_array_equal(result[450:550], 0.0)

    def test_result_non_negative(self):
        mask, sp = self._make_binary_mask()
        result = taper_mask(
            mask, sp, taper_hours=50.0 / 3600, edge_taper_hours=50.0 / 3600
        )
        assert np.all(result >= 0.0)

    def test_result_at_most_one(self):
        mask, sp = self._make_binary_mask()
        result = taper_mask(
            mask, sp, taper_hours=50.0 / 3600, edge_taper_hours=50.0 / 3600
        )
        assert np.all(result <= 1.0 + 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# extract_clean_segments
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractCleanSegments:
    """Tests for gaps.extract_clean_segments."""

    @staticmethod
    def _make_mask_with_gap(n=1000, gap_start=400, gap_end=600):
        """Binary mask with a single central gap."""
        mask = np.ones(n, dtype=int)
        mask[gap_start:gap_end] = 0
        return mask

    def test_returns_dict_of_signal_processors(self):
        sp = _make_sp(n=1000, fs=1.0)
        mask = self._make_mask_with_gap(1000)
        result = extract_clean_segments(sp, mask, min_clean_hours=0.0)
        assert isinstance(result, dict)
        for name, seg in result.items():
            assert isinstance(seg, SignalProcessor)
            assert name.startswith("segment")

    def test_segment_naming(self):
        sp = _make_sp(n=1000, fs=1.0)
        mask = self._make_mask_with_gap(1000)
        result = extract_clean_segments(sp, mask, min_clean_hours=0.0)
        assert "segment0" in result
        assert "segment1" in result
        assert len(result) == 2

    def test_segment_lengths_match_mask(self):
        sp = _make_sp(n=1000, fs=1.0)
        mask = self._make_mask_with_gap(1000, gap_start=400, gap_end=600)
        result = extract_clean_segments(sp, mask, min_clean_hours=0.0)
        assert result["segment0"].N == 400
        assert result["segment1"].N == 400

    def test_data_matches_original(self):
        sp = _make_sp(n=1000, fs=1.0)
        mask = self._make_mask_with_gap(1000, gap_start=400, gap_end=600)
        result = extract_clean_segments(sp, mask, min_clean_hours=0.0)
        for ch in sp.channels:
            np.testing.assert_array_equal(
                result["segment0"].data[ch], sp.data[ch][:400]
            )
            np.testing.assert_array_equal(
                result["segment1"].data[ch], sp.data[ch][600:]
            )

    def test_does_not_mutate_original(self):
        sp = _make_sp(n=1000, fs=1.0)
        original_x = sp.data["X"].copy()
        mask = self._make_mask_with_gap(1000)
        result = extract_clean_segments(sp, mask, min_clean_hours=0.0)
        # Mutate the segment data
        result["segment0"].data["X"][:] = 999.0
        np.testing.assert_array_equal(sp.data["X"], original_x)

    def test_min_clean_hours_filters_short_segments(self):
        sp = _make_sp(n=1000, fs=1.0)
        # Gap from 100-900 → two short segments (100 s each)
        mask = self._make_mask_with_gap(1000, gap_start=100, gap_end=900)
        # min_clean_hours=1.0 → need 3600 samples at 1 Hz
        result = extract_clean_segments(sp, mask, min_clean_hours=1.0)
        assert len(result) == 0

    def test_all_ones_mask_gives_single_segment(self):
        sp = _make_sp(n=1000, fs=1.0)
        mask = np.ones(1000, dtype=int)
        result = extract_clean_segments(sp, mask, min_clean_hours=0.0)
        assert len(result) == 1
        assert result["segment0"].N == 1000

    def test_all_zeros_mask_gives_empty(self):
        sp = _make_sp(n=1000, fs=1.0)
        mask = np.zeros(1000, dtype=int)
        result = extract_clean_segments(sp, mask, min_clean_hours=0.0)
        assert len(result) == 0

    def test_fs_preserved(self):
        sp = _make_sp(n=1000, fs=0.2)
        mask = self._make_mask_with_gap(1000)
        result = extract_clean_segments(sp, mask, min_clean_hours=0.0)
        for seg in result.values():
            assert seg.fs == 0.2

    def test_t0_set_correctly(self):
        sp = _make_sp(n=1000, fs=1.0)
        mask = self._make_mask_with_gap(1000, gap_start=400, gap_end=600)
        result = extract_clean_segments(sp, mask, min_clean_hours=0.0)
        assert result["segment0"].t0 == sp.t[0]
        assert result["segment1"].t0 == sp.t[600]

    def test_t0_nonzero_origin(self):
        """Verify t0 is correct when the source SP has a large t0 offset."""
        sp = _make_sp(n=1000, fs=0.2)
        sp.t0 = 1_000_000.0  # realistic TCB-like offset
        mask = self._make_mask_with_gap(1000, gap_start=400, gap_end=600)
        result = extract_clean_segments(sp, mask, min_clean_hours=0.0)

        # segment0 starts at sample 0 → t0 = sp.t0
        assert result["segment0"].t0 == pytest.approx(sp.t0)
        # segment1 starts at sample 600 → t0 = sp.t0 + 600 * dt
        expected_t0 = sp.t0 + 600 * sp.dt
        assert result["segment1"].t0 == pytest.approx(expected_t0)

        # Time arrays should match the original's samples exactly
        np.testing.assert_allclose(result["segment0"].t, sp.t[:400])
        np.testing.assert_allclose(result["segment1"].t, sp.t[600:])

    def test_mask_length_mismatch_raises(self):
        sp = _make_sp(n=1000, fs=1.0)
        mask = np.ones(500, dtype=int)
        with pytest.raises(ValueError, match="does not match"):
            extract_clean_segments(sp, mask)

    def test_multiple_gaps(self):
        sp = _make_sp(n=1000, fs=1.0)
        mask = np.ones(1000, dtype=int)
        mask[200:300] = 0
        mask[600:700] = 0
        result = extract_clean_segments(sp, mask, min_clean_hours=0.0)
        assert len(result) == 3
        assert result["segment0"].N == 200
        assert result["segment1"].N == 300
        assert result["segment2"].N == 300
