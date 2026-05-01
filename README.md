# mojito-processor

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18718620.svg)](https://doi.org/10.5281/zenodo.18718620)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://ollieburke.github.io/MojitoProcessor/)

Postprocessing tools for LISA Mojito L01 data for use with L2D noise analysis.

## Goal of package

The goal of this package is to provide a simple, modular, and well-documented set of tools for processing LISA Mojito L1 data. The package applies a signal processing pipeline (filtering, downsampling, trimming, windowing) to data loaded via the [`mojito`](https://pypi.org/project/mojito/) package. The design emphasizes ease of use and flexibility, allowing users to customize the processing steps as needed for their specific analysis tasks.

## Dependencies

This package depends on [`mojito`](https://pypi.org/project/mojito/), the official LISA L1 file reader. All dependencies are installed automatically via pip or uv.

## Installation

```bash
pip install mojito-processor
# or
uv pip install mojito-processor
```

## Development Setup

```bash
# Clone the repository
git clone https://github.com/OllieBurke/MojitoProcessor.git
cd MojitoProcessor

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install core package only
uv sync

# Install with gap-handling support (lisaglitch + lisa-gap)
uv sync --group gaps

# Install all dependency groups (dev, docs, notebooks, gaps)
uv sync --all-groups

# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit on all files (optional)
uv run pre-commit run --all-files
```

> **Note:** `torch` (a transitive dependency of `lisaglitch`) is excluded by
> default via `tool.uv.no-install-package` in `pyproject.toml`. It is not
> needed by MojitoProcessor.

## Quick Start

```python
from MojitoProcessor import load_file, load_processed, process_pipeline, write, read_and_process

# ── Load Mojito L1 data ───────────────────────────────────────────────────────
data = load_file("mojito_data.h5")

# ── Pipeline parameters ───────────────────────────────────────────────────────

downsample_kwargs = {
    "target_fs": 0.2,      # Hz — target sampling rate (None = no downsampling)
    "kaiser_window": 31.0, # Kaiser window beta (higher = more aggressive anti-aliasing)
}

filter_kwargs = {
    "highpass_cutoff": 5e-6,                                # Hz — high-pass cutoff (always applied)
    "lowpass_cutoff": 0.8 * downsample_kwargs["target_fs"], # Hz — low-pass cutoff (None for high-pass only)
    "order": 2,                                             # Butterworth filter order
}

trim_kwargs = {
    "fraction": 0.02,  # Total fraction of data trimmed symmetrically from both ends
}

truncate_kwargs = {
    "days": 7.0,  # Segment length in days (splits dataset into non-overlapping chunks)
}

window_kwargs = {
    "window": "tukey",  # Window type: 'tukey', 'hann', 'hamming', 'blackman', 'blackmanharris', 'planck'
    "alpha": 0.0125,    # Taper fraction for Tukey/Planck windows
}

# ─────────────────────────────────────────────────────────────────────────────

processed_segments = process_pipeline(
    data,
    downsample_kwargs=downsample_kwargs,
    filter_kwargs=filter_kwargs,
    trim_kwargs=trim_kwargs,
    truncate_kwargs=truncate_kwargs,
    window_kwargs=window_kwargs,
)

# Access processed data
sp = processed_segments["segment0"]
print(f"Sampling rate: {sp.fs} Hz")
print(f"Duration:      {sp.T / 86400:.2f} days")
print(f"TCB start:     {sp.t0:.6g} s")

# ── Write to HDF5 ─────────────────────────────────────────────────────────────
# Use segment_ids to write only a subset, e.g. segment_ids=[0, 1, 2]
write(
    "processed.h5",
    processed_segments,
    raw_data=data,
    filter_kwargs=filter_kwargs,
    downsample_kwargs=downsample_kwargs,
    trim_kwargs=trim_kwargs,
    truncate_kwargs=truncate_kwargs,
    window_kwargs=window_kwargs,
)

# ── Reload from HDF5 (deferred analysis, select specific segments) ────────────
segments, raw_data = load_processed(
    "processed.h5",
    segment_ids=[0, 1, 2],   # omit to load all segments
)
# raw_data["orbits"]["sc_position_1"]  → full-span spacecraft positions [m]
# raw_data["noise_estimates"]["xyz"]   → frequency-domain noise covariance cubes
# raw_data["metadata"]                 → laser frequency and pipeline names
```

Alternatively, load, process, and write in a single call using the high-level pipeline:

```python
from MojitoProcessor import read_and_process

segments = read_and_process(
    "mojito_data.h5",
    filter_kwargs=filter_kwargs,
    downsample_kwargs=downsample_kwargs,
    trim_kwargs=trim_kwargs,
    truncate_kwargs=truncate_kwargs,
    window_kwargs=window_kwargs,
    output_path="processed.h5",
)
```

## Features

- **Load** — `load_file` reads LISA Mojito L1 HDF5 files via the [`mojito`](https://gitlab.esa.int/lisa-commons/mojito) package
- **Process** — `process_pipeline` applies filtering, downsampling, trimming, segmentation, and windowing in a single call
- **Gaps** — `MojitoProcessor.gaps` handles gapped data: apply raw masks, compute extended masks accounting for filter leakage, and extract contiguous clean segments for gap-free spectral analysis
- **Write** — `write` saves processed segments and raw auxiliary data (orbits, noise estimates, LTTs) to HDF5; use `segment_ids` to write only a subset of segments
- **Reload** — `load_processed` reads a written file back into a `dict[str, SignalProcessor]` plus a `raw_data` dict (orbits, noise estimates, metadata); use `segment_ids` to load only the segments you need
- **Pipeline** — `read_and_process` combines load, process, and write into one function, with `segment_ids` support and an optional CLI interface
- **TCB time tracking** — `t0` is propagated through every processing step, including segmentation
- **TDI channel support** — XYZ and AET (via `SignalProcessor.to_aet()`)

## Building the Documentation

First, install [Pandoc](https://pandoc.org/installing.html) (required by nbsphinx to render the example notebooks):

```bash
# macOS
brew install pandoc

# Linux (Debian/Ubuntu)
sudo apt-get install pandoc
```

Then install the docs dependencies and build:

```bash
uv sync --group docs
uv run sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in your browser to view the result.

## License

MIT License — see LICENSE file for details.
