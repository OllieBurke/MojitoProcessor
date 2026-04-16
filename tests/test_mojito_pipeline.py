"""Unit tests for MojitoProcessor.pipelines.pipeline."""

import numpy as np
import pytest

from MojitoProcessor.pipelines.pipeline import pipeline
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


class TestPipelineReturnType:
    """Return-type and basic structural checks."""

    def test_returns_dict(self, mocker):
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        result = pipeline("fake.h5", filter_kwargs={"highpass_cutoff": 0.01})
        assert isinstance(result, dict)

    def test_values_are_signal_processors(self, mocker):
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        result = pipeline("fake.h5", filter_kwargs={"highpass_cutoff": 0.01})
        for sp in result.values():
            assert isinstance(sp, SignalProcessor)

    def test_segment0_present(self, mocker):
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        result = pipeline("fake.h5", filter_kwargs={"highpass_cutoff": 0.01})
        assert "segment0" in result


# ─────────────────────────────────────────────────────────────────────────────
# Tests: load_file is called correctly
# ─────────────────────────────────────────────────────────────────────────────


class TestPipelineLoadFile:
    """Checks that load_file is invoked with the right arguments."""

    def test_load_file_called_with_path(self, mocker):
        mock_load = mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        pipeline("myfile.h5", filter_kwargs={"highpass_cutoff": 0.01})
        mock_load.assert_called_once()
        assert mock_load.call_args[0][0] == "myfile.h5"

    def test_load_days_none_by_default(self, mocker):
        mock_load = mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        pipeline("f.h5", filter_kwargs={"highpass_cutoff": 0.01})
        assert mock_load.call_args[1].get("load_days") is None

    def test_load_days_passed_through(self, mocker):
        mock_load = mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        pipeline("f.h5", load_days=3.0, filter_kwargs={"highpass_cutoff": 0.01})
        assert mock_load.call_args[1]["load_days"] == pytest.approx(3.0)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: process_pipeline kwargs are forwarded
# ─────────────────────────────────────────────────────────────────────────────


class TestPipelineKwargs:
    """Checks that pipeline kwargs are passed through to process_pipeline."""

    def test_filter_kwargs_forwarded(self, mocker):
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        mock_pipeline = mocker.patch(
            "MojitoProcessor.pipelines.pipeline.process_pipeline",
            return_value={"segment0": _make_segment()},
        )
        filter_kwargs = {"highpass_cutoff": 0.01, "order": 4}
        pipeline("f.h5", filter_kwargs=filter_kwargs)
        assert mock_pipeline.call_args[1]["filter_kwargs"] == filter_kwargs

    def test_downsample_kwargs_forwarded(self, mocker):
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        mock_pipeline = mocker.patch(
            "MojitoProcessor.pipelines.pipeline.process_pipeline",
            return_value={"segment0": _make_segment()},
        )
        ds_kwargs = {"target_fs": 1.0}
        pipeline("f.h5", downsample_kwargs=ds_kwargs)
        assert mock_pipeline.call_args[1]["downsample_kwargs"] == ds_kwargs

    def test_channels_forwarded(self, mocker):
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        mock_pipeline = mocker.patch(
            "MojitoProcessor.pipelines.pipeline.process_pipeline",
            return_value={"segment0": _make_segment()},
        )
        pipeline("f.h5", channels=["X", "Y"])
        assert mock_pipeline.call_args[1].get("channels") == ["X", "Y"]

    def test_window_kwargs_forwarded(self, mocker):
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        mock_pipeline = mocker.patch(
            "MojitoProcessor.pipelines.pipeline.process_pipeline",
            return_value={"segment0": _make_segment()},
        )
        win_kwargs = {"window": "hann"}
        pipeline("f.h5", window_kwargs=win_kwargs)
        assert mock_pipeline.call_args[1]["window_kwargs"] == win_kwargs


# ─────────────────────────────────────────────────────────────────────────────
# Tests: optional write
# ─────────────────────────────────────────────────────────────────────────────


class TestPipelineWrite:
    """Checks that write() is called only when output_path is provided."""

    def test_write_not_called_without_output_path(self, mocker):
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        mock_write = mocker.patch("MojitoProcessor.pipelines.pipeline.write")
        pipeline("f.h5", filter_kwargs={"highpass_cutoff": 0.01})
        mock_write.assert_not_called()

    def test_write_called_when_output_path_given(self, mocker, tmp_path):
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        mock_write = mocker.patch("MojitoProcessor.pipelines.pipeline.write")
        out = tmp_path / "out.h5"
        pipeline("f.h5", filter_kwargs={"highpass_cutoff": 0.01}, output_path=out)
        mock_write.assert_called_once()

    def test_write_receives_output_path(self, mocker, tmp_path):
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        mock_write = mocker.patch("MojitoProcessor.pipelines.pipeline.write")
        out = tmp_path / "out.h5"
        pipeline("f.h5", filter_kwargs={"highpass_cutoff": 0.01}, output_path=out)
        assert mock_write.call_args[0][0] == out

    def test_write_receives_raw_data(self, mocker, tmp_path):
        data = _make_data()
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=data,
        )
        mock_write = mocker.patch("MojitoProcessor.pipelines.pipeline.write")
        out = tmp_path / "out.h5"
        pipeline("f.h5", filter_kwargs={"highpass_cutoff": 0.01}, output_path=out)
        assert mock_write.call_args[1]["raw_data"] is data

    def test_write_receives_filter_kwargs(self, mocker, tmp_path):
        mocker.patch(
            "MojitoProcessor.pipelines.pipeline.load_file",
            return_value=_make_data(),
        )
        mock_write = mocker.patch("MojitoProcessor.pipelines.pipeline.write")
        filter_kwargs = {"highpass_cutoff": 5e-6}
        out = tmp_path / "out.h5"
        pipeline("f.h5", filter_kwargs=filter_kwargs, output_path=out)
        assert mock_write.call_args[1]["filter_kwargs"] == filter_kwargs
