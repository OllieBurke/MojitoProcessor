# mojito-noise-sprint

A comprehensive toolkit for LISA noise characterization combining educational tutorials and real Mojito L1 data analysis. This repository provides tools for loading LISA Time-Delay Interferometry (TDI) data, modeling instrument noise with spline-based power spectral densities, and performing Bayesian parameter estimation via MCMC sampling.

## Overview

This repository is organized into two main components:

1. **Basic_Sprint**: Educational tutorials demonstrating noise parameter estimation with synthetic data
2. **Mojito_Sprint**: Production pipeline for analyzing real LISA Mojito L1 simulation data

Both sprints share common utility modules ([BasicUtils](#basicutils-module) and [MojitoUtils](#mojitoutils-module)) that provide noise modeling, signal processing, and data loading capabilities.

## Installation

Install `uv`:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or with Homebrew
brew install uv

# or with pipx
pipx install uv
```

Clone and setup:
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/OllieBurke/mojito-noise-sprint.git
cd mojito-noise-sprint
# Install Python dependencies
uv sync
```

### Download Required Data Files

Before running the notebooks, download the LISA simulation data files (recommended to avoid WiFi issues during the workshop):

1. Navigate to the [Mojito data repository](https://nextcloud-dcc-fi-csc-okd-exchange1.2.rahtiapp.fi/apps/files/files/15081?dir=/dcc-fi-csc-okd-exchange1-globalstorage/brickmarket_processed/mojito_light_v1_0_0)
2. Open the `L1_customforL2D_1` folder
3. Download these two files:
  - `NOISE_731d_2.5s_L1_AETXYZ_source0_0_20251206T220508924302Z.h5` (1.2 GB)
  - `Orbits_LTTs.h5` (1.4 GB)

4. Place the downloaded files in their respective directories:
  ```bash
  # From the repository root
  mv ~/Downloads/NOISE_731d_2.5s_L1_AETXYZ_source0_0_20251206T220508924302Z.h5 Data/Mojito_Data/
  mv ~/Downloads/Orbits_LTTs.h5 Data/Instrument/
  ```

If working within VS code, activate the virtual environment:
```bash
source .venv/bin/activate
```

If Visual Studio Code is not your choice of poison, please use this command instead to launch
a jupyter notebook. 

```bash
uv run --with jupyter jupyter lab
```

## Repository Structure

```
mojito-noise-sprint/
├── BasicUtils/              # Noise modeling & likelihood
│   ├── noise_utils.py       # PSD & noise generation
│   ├── spline_utils.py      # Non-parametric PSD fitting
│   ├── likelihood.py        # MCMC likelihood functions
│   └── mcmc_utils.py        # MCMC backend management
│
├── MojitoUtils/             # Data loading & processing
│   ├── mojito_loader.py     # HDF5 data loading
│   ├── SigProcessing.py     # Signal processing pipeline
│   └── plot_utils.py        # Visualization utilities
│
├── Basic_Sprint/            # Educational tutorials
│   ├── noise_spline_params_tutorial.ipynb
│   └── lisa_noise_estimation.ipynb
│
├── Mojito_Sprint/           # Real data pipeline
│   ├── read_data.ipynb      # Data loading example
│   ├── Post_Process_Data.ipynb  # Manipulating Mojito data
│   └── mojito_psd_csd_model.ipynb # Analytical models for Mojito noise
└── Data/                    # LISA datasets (Git LFS)
    ├── Mojito_Data/         # 4 Hz Mojito L1 simulations
    └── Instrument/          # Orbits & ephemeris
```

## Basic_Sprint

**Purpose**: Educational tutorials for learning LISA noise parameter estimation techniques.

### Notebooks

#### [noise_spline_params_tutorial.ipynb](Basic_Sprint/noise_spline_params_tutorial.ipynb)
Basic tutorial (Ollie) on MCMC-based LISA noise parameter estimation:
- Generates synthetic LISA noise with known parameters (optical noise Ao, acceleration noise Aa)
- Demonstrates spline-based PSD modeling using knot points
- Sets up parallel tempering MCMC with [Eryn](https://github.com/mikekatz04/Eryn) sampler
- Analyzes posterior distributions with corner plots
- Reconstructs full TDI PSD from fitted spline parameters. Second generation assuming static constellation. 

**Key concepts covered:**
- Separating smooth noise components (S_pm, S_op) from oscillatory TDI transfer functions
- PSD estimation via cubic splines
- Whittle likelihood for frequency-domain inference

#### [lisa_noise_estimation.ipynb](Basic_Sprint/lisa_noise_estimation.ipynb)
Advanced tutorial (Martina) focusing on multi-channel correlated noise:
- Models correlations between TDI channels (X, Y, Z)
- Demonstrates covariance matrix handling
- Uses parallel tempering for challenging posterior geometries

## Mojito_Sprint

**Purpose**: Production pipeline for processing real LISA Mojito L1 simulation data. Goal is to understand the post-processing steps and work together to come up with genius solutions to fit the noise. 

### Notebooks

#### [read_data.ipynb](Mojito_Sprint/read_data.ipynb)
Data loading and exploration:
- Loads 2-year Mojito noise simulations using [mojito_loader.py](MojitoUtils/mojito_loader.py)
- Visualizes TDI time series (X, Y, Z or A, E, T channels)
- Plots light travel times between spacecraft
- Basic data quality checks

#### [Post_Process_Data.ipynb](Mojito_Sprint/Post_Process_Data.ipynb)
Complete signal processing pipeline:
- **High-pass filtering**: Removes DC trends (default: 5e-5 Hz cutoff)
- **Trimming**: Removes edge artifacts from filtering (default: 10% from each end)
- **Windowing**: Applies Tukey/Hann windows to reduce spectral leakage
- **FFT analysis**: Converts time domain to frequency domain
- **Consistency checking**: Compares periodogram against Mojito's provided noise covariance
- Supports both XYZ and AET channel transformations

#### [mojito_psd_csd_model.ipynb](Mojito_Sprint/mojito_psd_csd_model.ipynb)
Analytical noise modeling and validation:
- **TDI transfer functions**: Implements analytical PSD/CSD functions for all TDI channel combinations (XX, XY, XZ, YY, YZ, ZZ). This is for unequal and non-constant arm-legnths. 
- **Physical noise modeling**: Uses SCIRD noise parameters (3×10⁻¹⁵ m/s²/√Hz acceleration, 15×10⁻¹² m/√Hz optical readout).
- **Light travel time dependence**: Models noise as function of 6 LISA arm lengths and their time variations.
- **Model validation**: Compares analytical predictions against L01 estimates and processed Mojito data.


## Dependencies

Key packages (see [pyproject.toml](pyproject.toml) for full list):
- **corner**: MCMC posterior visualization
- **eryn**: Bayesian inference with parallel tempering
- **h5py**: HDF5 file handling
- **numpy**, **scipy**, **matplotlib**: Scientific computing
- **ipykernel**: Jupyter notebook support

## License

See repository for license details.

