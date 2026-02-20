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
git clone https://github.com/YourUsername/mojito-processor.git
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
data = load_mojito_l1("mojito_data.h5")

# Process with signal pipeline
sp = process_pipeline(
    data,
    channels=['X', 'Y', 'Z'],
    highpass_cutoff=5e-6,
    target_fs=0.4,
    trim_fraction=0.022,
)

# Access processed data
print(f"Sampling rate: {sp.fs} Hz")
print(f"Duration: {sp.T/86400:.2f} days")
```

## Features

- Load LISA Mojito L1 HDF5 data files
- Signal processing pipeline (filtering, downsampling, windowing)
- TDI channel transformations (XYZ ↔ AET)
- Noise analysis utilities

## License

MIT License - see LICENSE file for details.
