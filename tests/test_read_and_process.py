"""Kept for backward compatibility — tests have moved to test_mojito_pipeline.py."""

import numpy as np
import pytest

from MojitoProcessor.pipelines.mojito.pipeline import pipeline as read_and_process
from MojitoProcessor.process.sigprocess import SignalProcessor

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_data(n: int = 1000, fs: float = 4.0) -> dict:
    """Minimal data dict compatible with process_pipeline."""
    rng = np.random.default_rng(42)
    return {
        "tdis": {ch: rng.standard_normal(n) for ch in ["X", "Y", "Z"]},
        "fs": fs,
        "t_tdi": np.arange(n) / fs,
        "metadata": {"laser_frequency": 2.816e14},
    }


def _make_segment() -> SignalProcessor:
    """Single-channel SignalProcessor for use as a mock pipeline return value."""
    return SignalProcessor(
        {"X": np.zeros(100), "Y": np.zeros(100), "Z": np.zeros(100)},
        fs=1.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: return type and basic plumbing
# ─────────────────────────────────────────────────────────────────────────────


class TestReadAndProcessReturnType:
    """Return-type and basic structural checks."""

    def test_returns_dict(self, mocker):
        """Result must be a dict."""
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        result = read_and_process("fake.h5", filter_kwargs={"highpass_cutoff": 0.01})
        assert isinstance(result, dict)

    def test_values_are_signal_processors(self, mocker):
        """Every value in the result dict must be a SignalProcessor."""
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        result = read_and_process("fake.h5", filter_kwargs={"highpass_cutoff": 0.01})
        for sp in result.values():
            assert isinstance(sp, SignalProcessor)

    def test_segment0_present(self, mocker):
        """Default pipeline with no segmentation must return 'segment0'."""
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        result = read_and_process("fake.h5", filter_kwargs={"highpass_cutoff": 0.01})
        assert "segment0" in result


# ─────────────────────────────────────────────────────────────────────────────
# Tests: load_file is called correctly
# ─────────────────────────────────────────────────────────────────────────────


class TestReadAndProcessLoadFile:
    """Checks that load_file is invoked with the right arguments."""

    def test_load_file_called_with_path(self, mocker):
        """load_file must be called with the path passed to read_and_process."""
        mock_load = mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        read_and_process("myfile.h5", filter_kwargs={"highpass_cutoff": 0.01})
        mock_load.assert_called_once()
        assert mock_load.call_args[0][0] == "myfile.h5"

    def test_load_days_none_by_default(self, mocker):
        """load_days must default to None."""
        mock_load = mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        read_and_process("f.h5", filter_kwargs={"highpass_cutoff": 0.01})
        assert mock_load.call_args[1].get("load_days") is None

    def test_load_days_passed_through(self, mocker):
        """A non-None load_days must be forwarded to load_file."""
        mock_load = mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        read_and_process("f.h5", load_days=3.0, filter_kwargs={"highpass_cutoff": 0.01})
        assert mock_load.call_args[1]["load_days"] == pytest.approx(3.0)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: process_pipeline kwargs are forwarded
# ─────────────────────────────────────────────────────────────────────────────


class TestReadAndProcessPipelineKwargs:
    """Checks that pipeline kwargs are passed through to process_pipeline."""

    def test_filter_kwargs_forwarded(self, mocker):
        """filter_kwargs must be passed verbatim to process_pipeline."""
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        mock_pipeline = mocker.patch(
            "MojitoProcessor.pipelines.pipeline.process_pipeline",
            return_value={"segment0": _make_segment()},
        )
        filter_kwargs = {"highpass_cutoff": 0.01, "order": 4}
        read_and_process("f.h5", filter_kwargs=filter_kwargs)
        assert mock_pipeline.call_args[1]["filter_kwargs"] == filter_kwargs

    def test_downsample_kwargs_forwarded(self, mocker):
        """downsample_kwargs must be passed verbatim to process_pipeline."""
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        mock_pipeline = mocker.patch(
            "MojitoProcessor.pipelines.pipeline.process_pipeline",
            return_value={"segment0": _make_segment()},
        )
        ds_kwargs = {"target_fs": 1.0}
        read_and_process("f.h5", downsample_kwargs=ds_kwargs)
        assert mock_pipeline.call_args[1]["downsample_kwargs"] == ds_kwargs

    def test_channels_forwarded(self, mocker):
        """channels must be forwarded as a keyword arg to process_pipeline."""
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        mock_pipeline = mocker.patch(
            "MojitoProcessor.pipelines.pipeline.process_pipeline",
            return_value={"segment0": _make_segment()},
        )
        read_and_process("f.h5", channels=["X", "Y"])
        assert mock_pipeline.call_args[1].get("channels") == ["X", "Y"]

    def test_window_kwargs_forwarded(self, mocker):
        """window_kwargs must be passed verbatim to process_pipeline."""
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        mock_pipeline = mocker.patch(
            "MojitoProcessor.pipelines.pipeline.process_pipeline",
            return_value={"segment0": _make_segment()},
        )
        win_kwargs = {"window": "hann"}
        read_and_process("f.h5", window_kwargs=win_kwargs)
        assert mock_pipeline.call_args[1]["window_kwargs"] == win_kwargs


# ─────────────────────────────────────────────────────────────────────────────
# Tests: optional write
# ─────────────────────────────────────────────────────────────────────────────


class TestReadAndProcessWrite:
    """Checks that write() is called only when output_path is provided."""

    def test_write_not_called_without_output_path(self, mocker):
        """write must not be called when output_path is None."""
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        mock_write = mocker.patch("MojitoProcessor.pipelines.pipeline.write")
        read_and_process("f.h5", filter_kwargs={"highpass_cutoff": 0.01})
        mock_write.assert_not_called()

    def test_write_called_when_output_path_given(self, mocker, tmp_path):
        """write must be called exactly once when output_path is set."""
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        mock_write = mocker.patch("MojitoProcessor.pipelines.pipeline.write")
        out = tmp_path / "out.h5"
        read_and_process(
            "f.h5", filter_kwargs={"highpass_cutoff": 0.01}, output_path=out
        )
        mock_write.assert_called_once()

    def test_write_receives_output_path(self, mocker, tmp_path):
        """write must receive the output_path as its first positional arg."""
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        mock_write = mocker.patch("MojitoProcessor.pipelines.pipeline.write")
        out = tmp_path / "out.h5"
        read_and_process(
            "f.h5", filter_kwargs={"highpass_cutoff": 0.01}, output_path=out
        )
        assert mock_write.call_args[0][0] == out

    def test_write_receives_raw_data(self, mocker, tmp_path):
        """write must receive the same dict that load_file returned."""
        data = _make_data()
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=data,
        )
        mock_write = mocker.patch("MojitoProcessor.pipelines.pipeline.write")
        out = tmp_path / "out.h5"
        read_and_process(
            "f.h5", filter_kwargs={"highpass_cutoff": 0.01}, output_path=out
        )
        assert mock_write.call_args[1]["raw_data"] is data

    def test_write_receives_filter_kwargs(self, mocker, tmp_path):
        """filter_kwargs must be forwarded to write for metadata storage."""
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        mock_write = mocker.patch("MojitoProcessor.pipelines.pipeline.write")
        filter_kwargs = {"highpass_cutoff": 5e-6}
        out = tmp_path / "out.h5"
        read_and_process("f.h5", filter_kwargs=filter_kwargs, output_path=out)
        assert mock_write.call_args[1]["filter_kwargs"] == filter_kwargs
