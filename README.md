# mojito-processor

Postprocessing tools for LISA Mojito L01 data for use with L2D noise analysis.

## Installation

```bash
pip install mojito-processor
```

Or with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv pip install mojito-processor
```

## Development Setup

For development with pre-commit hooks and linting using uv:
```bash
# Clone the repository
git clone https://github.com/OllieBurke/MojitoProcessor.git
cd mojito-processor

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install in editable mode with dev dependencies
uv pip install -e .

# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit on all files (optional)
uv run pre-commit run --all-files
```

## Quick Start

```python
from MojitoProcessor import load_mojito_l1, process_pipeline

# Load Mojito L1 data
data_file = "mojito_data.h5"  # Path to your Mojito L1 HDF5 file
data = load_mojito_l1(data_file)

# ── Pipeline parameters ───────────────────────────────────────────────────────

# Downsampling parameters
downsample_kwargs = {
    "target_fs": 0.2,  # Hz — target sampling rate (None = no downsampling).
    "kaiser_window": 31.0,  # Kaiser window beta parameter (higher = more aggressive anti-aliasing)
}

# Filter parameters
filter_kwargs = {
    "highpass_cutoff": 5e-6,  # Hz — high-pass cutoff (always applied)
    "lowpass_cutoff": 0.8
    * downsample_kwargs[
        "target_fs"
    ],  # Hz — low-pass cutoff (set None for high-pass only)
    "order": 2,  # Butterworth filter order
}

# Trim parameters
trim_kwargs = {
    "fraction": 0.02,  # Fraction of post-downsample duration trimmed from each end.
    # Total amount of data remaining is (1 - fraction) * N, for N
    # the number of samples after downsampling.
}

# Segmentation parameters
truncate_kwargs = {
    "days": 7.0,  # Segment length in days (splits dataset into 7-day chunks)
}

# Window parameters
window_kwargs = {
    "window": "tukey",  # Window type: 'tukey', 'hann', 'hamming', 'blackman'
    "alpha": 0.0125,  # Taper fraction for Tukey window
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
print(f"Sampling rate: {sp.fs} Hz")
print(f"Duration: {sp.T/86400:.2f} days")
```

## Features

- Load LISA Mojito L1 HDF5 data files
- Signal processing pipeline (filtering, downsampling, trimming, windowing)
- TDI channel transformations (XYZ ↔ AET)
- Noise analysis utilities

## License

MIT License - see LICENSE file for details.
