# mojito-processor

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18718620.svg)](https://doi.org/10.5281/zenodo.18718620)[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://ollieburke.github.io/MojitoProcessor/)

Postprocessing tools for LISA Mojito L01 data for use with L2D noise analysis.

## Goal of package
The goal of this package is to provide a simple, modular, and well-documented set of tools for processing LISA Mojito L1 data. The package applies a signal processing pipeline (filtering, downsampling, trimming, windowing) to data loaded via the [`mojito`](https://gitlab.esa.int/lisa-commons/mojito) package. The design emphasizes ease of use and flexibility, allowing users to customize the processing steps as needed for their specific analysis tasks.

## Dependencies

This package depends on [`mojito`](https://gitlab.esa.int/lisa-commons/mojito), the official LISA L1 file reader, which is distributed via the ESA GitLab package registry. You must configure this index before installing `mojito-processor`.

With [uv](https://docs.astral.sh/uv/), add the following to your `pyproject.toml` or `uv.toml`:

```toml
[[tool.uv.index]]
name = "gitlab-esa-commons"
url = "https://gitlab.esa.int/api/v4/groups/29349/-/packages/pypi/simple"

[tool.uv.sources]
mojito = { index = "gitlab-esa-commons" }
```

Or pass the extra index directly on the command line:

```bash
uv pip install mojito-processor \
    --extra-index-url https://gitlab.esa.int/api/v4/groups/29349/-/packages/pypi/simple
```

## Installation

### From Test PyPI (Development)

This package is currently available on Test PyPI for testing:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mojito-processor
```

Or with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mojito-processor
```

### From PyPI (Coming Soon)

Once stable, the package will be available on PyPI for simpler installation:

```bash
pip install mojito-processor
# or
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
from mojito.reader import MojitoL1File
from MojitoProcessor import process_pipeline

# ── Load Mojito L1 data ───────────────────────────────────────────────────────
mojito_data_file = "mojito_data.h5"  # Path to your Mojito L1 HDF5 file

with MojitoL1File(mojito_data_file) as f:
    tdi_sampling = f.tdis.time_sampling
    data = {
        "tdis": {
            "X": f.tdis.x2[:],
            "Y": f.tdis.y2[:],
            "Z": f.tdis.z2[:],
        },
        "fs":    tdi_sampling.fs,
        "t_tdi": tdi_sampling.t()[:],
    }

# ── Pipeline parameters ───────────────────────────────────────────────────────

# Downsampling parameters
downsample_kwargs = {
    "target_fs": 0.2,      # Hz — target sampling rate (None = no downsampling)
    "kaiser_window": 31.0, # Kaiser window beta (higher = more aggressive anti-aliasing)
}

# Filter parameters
filter_kwargs = {
    "highpass_cutoff": 5e-6,                               # Hz — high-pass cutoff (always applied)
    "lowpass_cutoff": 0.8 * downsample_kwargs["target_fs"],  # Hz — low-pass cutoff (None for high-pass only)
    "order": 2,                                            # Butterworth filter order
}

# Trim parameters
trim_kwargs = {
    "fraction": 0.02,  # Total fraction of data trimmed symmetrically from both ends
}

# Segmentation parameters
truncate_kwargs = {
    "days": 7.0,  # Segment length in days (splits dataset into non-overlapping chunks)
}

# Window parameters
window_kwargs = {
    "window": "tukey",  # Window type: 'tukey', 'hann', 'hamming', 'blackman'
    "alpha": 0.0125,    # Taper fraction for Tukey window
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
print(f"Duration:      {sp.T/86400:.2f} days")
print(f"TCB start:     {sp.t0:.6g} s")
```

## Features

- Signal processing pipeline (filtering, downsampling, trimming, windowing) via `process_pipeline`
- Integrates with the [`mojito`](https://gitlab.esa.int/lisa-commons/mojito) package for reading LISA L1 HDF5 files
- TCB start-time (`t0`) tracking through every processing step, including segmentation — essential for time-dependent LISA transfer functions
- TDI channel support (XYZ and AET)

## Building the Documentation

First, install [Pandoc](https://pandoc.org/installing.html) (required by nbsphinx to render the example notebook):

```bash
# macOS
brew install pandoc

# Linux (Debian/Ubuntu)
sudo apt-get install pandoc
```

Then install the docs dependencies and run `sphinx-build` from the repo root:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Or with uv:

```bash
uv pip install -e ".[docs]"
uv run sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in your browser to view the result.

## License

MIT License - see LICENSE file for details.
