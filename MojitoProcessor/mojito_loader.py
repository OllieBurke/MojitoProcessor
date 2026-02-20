"""Data loading utilities for LISA Mojito L1 HDF5 files."""

from pathlib import Path

import h5py
import numpy as np

# =============================================================================
# Simple replacement classes for lolipops
# =============================================================================


class TimeSampling:
    """Replacement for lolipops.series.Sampling

    All the sampling information is stored as HDF5 attributes in the file:
    - t0: start time (s)
    - dt: time step (s)
    - size: number of samples
    """

    def __init__(self, t0, dt, size):
        self.t0 = float(t0)
        self.dt = float(dt)
        self.size = int(size)
        self.fs = 1.0 / dt  # Sampling frequency (Hz)
        self.duration = size * dt  # Total duration (s)

    def t(self):
        """Generate time array"""
        return self.t0 + np.arange(self.size) * self.dt

    def __repr__(self):
        return (
            f"SimpleSampling(t0={self.t0}, dt={self.dt}, "
            f"size={self.size}, fs={self.fs} Hz)"
        )


class LogFrequencySampling:
    """Replacement for lolipops.io.LogFrequencySampling

    Frequency sampling information stored as HDF5 attributes:
    - fmin: minimum frequency (Hz)
    - fmax: maximum frequency (Hz)
    - size: number of frequency bins
    """

    def __init__(self, fmin, fmax, size):
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.size = int(size)

    def f(self):
        """Generate logarithmically spaced frequency array"""
        return np.logspace(np.log10(self.fmin), np.log10(self.fmax), self.size)

    def __repr__(self):
        return (
            f"SimpleLogFrequencySampling(fmin={self.fmin}, "
            f"fmax={self.fmax}, size={self.size})"
        )


# =============================================================================
# Data container class
# =============================================================================


class MojitoData:
    """Container for Mojito L1 data"""

    def __init__(self):
        # TDI time series
        self.tdis = {}  # Dict with X, Y, Z, A, E, T arrays

        # Sampling information
        self.sampling = None  # SimpleSampling object
        self.t_tdi = None  # Time array
        self.fs = None  # Sampling frequency (Hz)
        self.dt = None  # Sampling period (s)
        self.t0 = None  # Start time (TCB)
        self.duration = None  # Duration (s)
        self.n_samples = None  # Number of samples

        # Pre-computed noise estimates (from mojito pipeline)
        self.noise_freqs = None  # Frequency array
        self.noise_psds = {}  # Dict with X, Y, Z PSDs (diagonal of covariance)
        self.noise_cov_xyz = None  # Full (n_seg, n_freq, 3, 3) covariance
        self.noise_cov_aet = (
            None  # Full (n_seg, n_freq, 3, 3) covariance (if available)
        )

        # Metadata
        self.metadata = {}
        self.filepath = None

    def __repr__(self):
        s = "MojitoData:\n"
        s += f"  File: {self.filepath}\n"
        s += f"  TDI channels: {list(self.tdis.keys())}\n"
        s += (
            f"  Samples: {self.n_samples:,} @ {self.fs} Hz "
            f"({self.duration/86400:.2f} days)\n"
        )
        s += (
            f"  Noise estimates: {len(self.noise_freqs)} freq bins from "
            f"{self.noise_freqs[0]:.2e} to {self.noise_freqs[-1]:.2e} Hz\n"
        )
        s += f"  Metadata: {list(self.metadata.keys())}\n"
        return s


class OrbitData:
    """Container for LISA orbit data and light travel times"""

    def __init__(self):
        # Light travel times
        self.ltts = {}  # Dict with link keys ('12', '13', etc.) -> arrays
        self.ltt_derivatives = {}  # Dict with link keys -> derivative arrays

        # Spacecraft orbits
        self.sc_positions = {}  # Dict with SC keys ('1', '2', '3') -> (N, 3) arrays
        self.sc_velocities = {}  # Dict with SC keys ('1', '2', '3') -> (N, 3) arrays

        # Sampling information for LTTs
        self.ltt_sampling = None  # TimeSampling object for LTTs
        self.t_ltt = None  # Time array for LTTs
        self.ltt_fs = None  # LTT sampling frequency (Hz)
        self.ltt_dt = None  # LTT sampling period (s)
        self.ltt_t0 = None  # LTT start time (TCB)
        self.ltt_duration = None  # LTT duration (s)
        self.ltt_n_samples = None  # Number of LTT samples

        # Sampling information for orbits
        self.orbit_sampling = None  # TimeSampling object for orbits
        self.t_orbit = None  # Time array for orbits
        self.orbit_fs = None  # Orbit sampling frequency (Hz)
        self.orbit_dt = None  # Orbit sampling period (s)
        self.orbit_t0 = None  # Orbit start time (TCB)
        self.orbit_duration = None  # Orbit duration (s)
        self.orbit_n_samples = None  # Number of orbit samples

        # Metadata
        self.metadata = {}
        self.filepath = None

    def __repr__(self):
        s = "OrbitData:\n"
        s += f"  File: {self.filepath}\n"
        s += f"  LTT links: {list(self.ltts.keys())}\n"
        s += (
            f"  LTT samples: {self.ltt_n_samples:,} @ {self.ltt_fs} Hz "
            f"({self.ltt_duration/86400:.2f} days)\n"
        )
        s += f"  Spacecraft: {list(self.sc_positions.keys())}\n"
        s += (
            f"  Orbit samples: {self.orbit_n_samples:,} @ "
            f"{self.orbit_fs:.2e} Hz ({self.orbit_duration/86400:.2f} days)\n"
        )
        s += f"  Metadata: {list(self.metadata.keys())}\n"
        return s


# =============================================================================
# Standalone loader function (NO lolipops dependency!)
# =============================================================================


def load_mojito_l1(filepath):
    """
    Load LISA L1 data from mojito pipeline HDF5 file WITHOUT lolipops.

    Parameters
    ----------
    filepath : str or Path
        Path to mojito L1 HDF5 file

    Returns
    -------
    data : MojitoData
        Container with all loaded data

    Example
    -------
    >>> data = load_mojito_l1_standalone("mojito_light.h5")
    >>> tdi_x = data.tdis['X']
    >>> freqs = data.noise_freqs
    >>> psd_x = data.noise_psds['X']
    """

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    data = MojitoData()
    data.filepath = filepath

    print(f"Loading: {filepath}")
    print(f"File size: {filepath.stat().st_size / 1e6:.2f} MB")

    with h5py.File(filepath, "r") as f:
        # ===== Metadata =====
        data.metadata["pipeline_name"] = f.attrs.get("pipeline_name", "unknown")
        data.metadata["laser_frequency"] = f.attrs.get("laser_frequency", None)
        data.metadata["lolipops_version"] = f.attrs.get("lolipops_version", "unknown")

        print(f"\nMetadata:")
        print(f"  Pipeline: {data.metadata['pipeline_name']}")
        print(f"  Laser frequency: {data.metadata['laser_frequency']:.3e} Hz")

        # ===== TDI Sampling =====
        # Read sampling attributes directly from HDF5
        tdi_sampling_group = f["tdis/sampling"]
        data.sampling = TimeSampling(
            t0=tdi_sampling_group.attrs["t0"],
            dt=tdi_sampling_group.attrs["dt"],
            size=tdi_sampling_group.attrs["size"],
        )

        # Extract sampling parameters
        data.t0 = data.sampling.t0
        data.dt = data.sampling.dt
        data.fs = data.sampling.fs
        data.duration = data.sampling.duration
        data.n_samples = data.sampling.size
        data.t_tdi = data.sampling.t()

        print(f"\nTDI Sampling:")
        print(f"  t0 = {data.t0:.3f} s (TCB)")
        print(f"  fs = {data.fs:.3f} Hz, dt = {data.dt:.3f} s")
        print(f"  Duration: {data.duration:.1f} s ({data.duration/86400:.2f} days)")
        print(f"  Samples: {data.n_samples:,}")

        # ===== TDI Time Series =====
        tdi_group = f["tdis"]

        # Load TDI-2 Michelson channels (X2, Y2, Z2)
        for ch in ["X", "Y", "Z"]:
            ch_name = ch + "2"  # TDI-2 generation
            if ch_name in tdi_group:
                data.tdis[ch] = tdi_group[ch_name][:]
                print(f"  ✓ Loaded TDI-{ch}: shape={data.tdis[ch].shape}")

        # Load TDI-2 Orthogonal channels (A2, E2, T2) if available
        for ch in ["A", "E", "T"]:
            ch_name = ch + "2"
            if ch_name in tdi_group:
                data.tdis[ch] = tdi_group[ch_name][:]
                print(f"  ✓ Loaded TDI-{ch}: shape={data.tdis[ch].shape}")

        # ===== Pre-computed Noise Estimates =====
        noise_group = f["noise_estimates"]

        # Load frequency sampling - read attributes directly from HDF5
        freq_group = noise_group["log_frequency_sampling"]
        noise_freq_sampling = LogFrequencySampling(
            fmin=freq_group.attrs["fmin"],
            fmax=freq_group.attrs["fmax"],
            size=freq_group.attrs["size"],
        )
        data.noise_freqs = noise_freq_sampling.f()

        print(f"\nPre-computed Noise Estimates:")
        print(
            f"  Frequency range: {data.noise_freqs[0]:.2e} to "
            f"{data.noise_freqs[-1]:.2e} Hz"
        )
        print(f"  Frequency bins: {len(data.noise_freqs)}")

        # Load XYZ noise covariance matrices
        # Shape: (time epoch, n_freqs, 3, 3)
        data.noise_cov_xyz = noise_group["XYZ"][:]
        n_seg, n_freq, _, _ = data.noise_cov_xyz.shape
        print(f"  XYZ covariance: {n_seg} segments x {n_freq} freqs x 3x3")

        # Load AET if available
        if "AET" in noise_group:
            data.noise_cov_aet = noise_group["AET"][:]
            print(f"  AET covariance: {data.noise_cov_aet.shape}")

            # Extract AET PSDs
            mean_cov_aet = np.mean(data.noise_cov_aet, axis=0)
            data.noise_psds["A"] = mean_cov_aet[:, 0, 0].real
            data.noise_psds["E"] = mean_cov_aet[:, 1, 1].real
            data.noise_psds["T"] = mean_cov_aet[:, 2, 2].real

    print(f"\n✓ Successfully loaded mojito data")
    return data


def load_orbits(filepath):
    """
    Load LISA orbit data and light travel times from HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Path to orbits HDF5 file (e.g., Orbits_LTTs.h5)

    Returns
    -------
    data : OrbitData
        Container with orbit positions, velocities, LTTs, and derivatives

    Example
    -------
    >>> orbit_data = load_orbits("Orbits_LTTs.h5")
    >>> ltt_12 = orbit_data.ltts['12']
    >>> sc1_pos = orbit_data.sc_positions['1']
    """

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    data = OrbitData()
    data.filepath = filepath

    print(f"Loading: {filepath}")
    print(f"File size: {filepath.stat().st_size / 1e6:.2f} MB")

    with h5py.File(filepath, "r") as f:
        # ===== Metadata =====
        data.metadata["pipeline_name"] = f.attrs.get("pipeline_name", "unknown")
        data.metadata["laser_frequency"] = f.attrs.get("laser_frequency", None)
        data.metadata["lolipops_version"] = f.attrs.get("lolipops_version", "unknown")

        print(f"\nMetadata:")
        print(f"  Pipeline: {data.metadata['pipeline_name']}")
        if data.metadata["laser_frequency"]:
            print(f"  Laser frequency: {data.metadata['laser_frequency']:.3e} Hz")

        # ===== Light Travel Times (LTTs) =====
        if "ltts" in f:
            ltt_group = f["ltts"]

            # Read LTT sampling
            if "sampling" in ltt_group:
                ltt_sampling_group = ltt_group["sampling"]
                data.ltt_sampling = TimeSampling(
                    t0=ltt_sampling_group.attrs["t0"],
                    dt=ltt_sampling_group.attrs["dt"],
                    size=ltt_sampling_group.attrs["size"],
                )

                # Extract sampling parameters
                data.ltt_t0 = data.ltt_sampling.t0
                data.ltt_dt = data.ltt_sampling.dt
                data.ltt_fs = data.ltt_sampling.fs
                data.ltt_duration = data.ltt_sampling.duration
                data.ltt_n_samples = data.ltt_sampling.size
                data.t_ltt = data.ltt_sampling.t()

                print(f"\nLTT Sampling:")
                print(f"  t0 = {data.ltt_t0:.3f} s (TCB)")
                print(f"  fs = {data.ltt_fs:.3f} Hz, dt = {data.ltt_dt:.3f} s")
                print(
                    f"  Duration: {data.ltt_duration:.1f} s "
                    f"({data.ltt_duration/86400:.2f} days)"
                )
                print(f"  Samples: {data.ltt_n_samples:,}")

            # Load LTT data
            links = ["12", "13", "21", "23", "31", "32"]

            for link in links:
                ltt_name = f"ltt_{link}"
                ltt_deriv_name = f"ltt_derivative_{link}"

                if ltt_name in ltt_group:
                    data.ltts[link] = ltt_group[ltt_name][:]
                if ltt_deriv_name in ltt_group:
                    data.ltt_derivatives[link] = ltt_group[ltt_deriv_name][:]

            if data.ltts:
                print(f"\nLight Travel Times:")
                print(f"  ✓ Loaded {len(data.ltts)} LTTs: {list(data.ltts.keys())}")
                print(f"  ✓ Loaded {len(data.ltt_derivatives)} LTT derivatives")

        # ===== Spacecraft Orbits =====
        if "orbits" in f:
            orbit_group = f["orbits"]

            # Read orbit sampling
            if "sampling" in orbit_group:
                orbit_sampling_group = orbit_group["sampling"]
                data.orbit_sampling = TimeSampling(
                    t0=orbit_sampling_group.attrs["t0"],
                    dt=orbit_sampling_group.attrs["dt"],
                    size=orbit_sampling_group.attrs["size"],
                )

                # Extract sampling parameters
                data.orbit_t0 = data.orbit_sampling.t0
                data.orbit_dt = data.orbit_sampling.dt
                data.orbit_fs = data.orbit_sampling.fs
                data.orbit_duration = data.orbit_sampling.duration
                data.orbit_n_samples = data.orbit_sampling.size
                data.t_orbit = data.orbit_sampling.t()

                print(f"\nOrbit Sampling:")
                print(f"  t0 = {data.orbit_t0:.3f} s (TCB)")
                print(f"  fs = {data.orbit_fs:.2e} Hz, dt = {data.orbit_dt:.1f} s")
                print(
                    f"  Duration: {data.orbit_duration:.1f} s "
                    f"({data.orbit_duration/86400:.2f} days)"
                )
                print(f"  Samples: {data.orbit_n_samples:,}")

            # Load spacecraft positions and velocities
            spacecraft = ["1", "2", "3"]

            for sc in spacecraft:
                pos_name = f"sc_position_{sc}"
                vel_name = f"sc_velocity_{sc}"

                if pos_name in orbit_group:
                    data.sc_positions[sc] = orbit_group[pos_name][:]
                if vel_name in orbit_group:
                    data.sc_velocities[sc] = orbit_group[vel_name][:]

            if data.sc_positions:
                print(f"\nSpacecraft Orbits:")
                print(f"  ✓ Loaded {len(data.sc_positions)} spacecraft positions")
                print(f"  ✓ Loaded {len(data.sc_velocities)} spacecraft velocities")

    print(f"\n✓ Successfully loaded orbit data")
    return data


def truncate_mojito_data(data, duration_seconds=None, n_samples=None):
    """
    Truncate a loaded Mojito dataset to a shorter time segment.

    This function takes a MojitoData object and truncates all time-domain data
    (TDI channels only) to the specified duration, then updates all metadata
    to be consistent with the truncated dataset.

    Parameters
    ----------
    data : MojitoData
        Loaded Mojito data object (from load_mojito_l1)
    duration_seconds : float, optional
        Duration to truncate to in seconds (mutually exclusive with n_samples)
    n_samples : int, optional
        Number of samples to keep (mutually exclusive with duration_seconds)

    Returns
    -------
    truncated_data : MojitoData
        New MojitoData object with truncated time series and updated metadata

    Notes
    -----
    - If both duration_seconds and n_samples are None, returns copy of original
    - If both are specified, raises ValueError
    - Truncation always starts from t0 (beginning of dataset)
    - All TDI channel arrays are truncated
    - Sampling metadata (duration, n_samples, etc.) is updated
    - Noise estimates (frequency-domain) are NOT modified as they're from full dataset

    Examples
    --------
    >>> data_full = load_mojito_l1("mojito_light.h5")
    >>> # Truncate to first 1 day
    >>> data_1day = truncate_mojito_data(data_full, duration_seconds=86400)
    >>>
    >>> # Truncate to first 10000 samples
    >>> data_short = truncate_mojito_data(data_full, n_samples=10000)
    """

    # Input validation
    if duration_seconds is not None and n_samples is not None:
        raise ValueError("Specify either duration_seconds OR n_samples, not both")

    if duration_seconds is None and n_samples is None:
        print(
            "Warning: No truncation parameters specified. "
            "Returning copy of original data."
        )
        return data

    # Calculate n_samples from duration if needed
    if duration_seconds is not None:
        n_samples_truncated = int(duration_seconds / data.dt)
        print(
            f"Truncating to {duration_seconds:.1f} seconds = "
            f"{n_samples_truncated:,} samples"
        )
    else:
        n_samples_truncated = n_samples
        duration_truncated = n_samples_truncated * data.dt
        print(
            f"Truncating to {n_samples_truncated:,} samples = "
            f"{duration_truncated:.1f} seconds "
            f"({duration_truncated/86400:.3f} days)"
        )

    # Validate truncation length
    if n_samples_truncated > data.n_samples:
        raise ValueError(
            f"Requested {n_samples_truncated} samples but data only has "
            f"{data.n_samples}"
        )

    if n_samples_truncated <= 0:
        raise ValueError(f"Invalid truncation: n_samples must be > 0")

    # Create new MojitoData object for truncated data
    truncated = MojitoData()

    # Copy filepath and metadata
    truncated.filepath = data.filepath
    truncated.metadata = data.metadata.copy()

    # Update sampling information
    truncated.t0 = data.t0  # Start time stays the same
    truncated.dt = data.dt  # Sampling period unchanged
    truncated.fs = data.fs  # Sampling frequency unchanged
    truncated.n_samples = n_samples_truncated
    truncated.duration = n_samples_truncated * data.dt

    # Create new TimeSampling object with truncated parameters
    truncated.sampling = TimeSampling(
        t0=truncated.t0, dt=truncated.dt, size=n_samples_truncated
    )
    truncated.t_tdi = truncated.sampling.t()

    print(f"\nTruncated sampling:")
    print(
        f"  Original: {data.n_samples:,} samples, {data.duration:.1f} s "
        f"({data.duration/86400:.3f} days)"
    )
    print(
        f"  Truncated: {truncated.n_samples:,} samples, "
        f"{truncated.duration:.1f} s ({truncated.duration/86400:.3f} days)"
    )
    print(f"  Kept: {100*n_samples_truncated/data.n_samples:.1f}% of " f"original data")

    # Truncate TDI time series
    print(f"\nTruncating TDI channels:")
    for ch, tdi_array in data.tdis.items():
        truncated.tdis[ch] = tdi_array[:n_samples_truncated]
        print(f"  {ch}: {data.tdis[ch].shape} → {truncated.tdis[ch].shape}")

    # Copy noise estimates (frequency-domain, independent of time segment length)
    # NOTE: Pre-computed noise estimates are from the FULL dataset analysis
    # For proper noise estimation on truncated data, recompute PSDs
    truncated.noise_freqs = (
        data.noise_freqs.copy() if data.noise_freqs is not None else None
    )
    truncated.noise_psds = data.noise_psds.copy()
    truncated.noise_cov_xyz = (
        data.noise_cov_xyz.copy() if data.noise_cov_xyz is not None else None
    )
    truncated.noise_cov_aet = (
        data.noise_cov_aet.copy() if data.noise_cov_aet is not None else None
    )

    print(f"\nNote: Noise estimates copied from original (computed on full dataset).")
    print(
        f"      For accurate noise characterization of truncated data, recompute PSDs."
    )

    print(f"\n✓ Truncation complete!")

    return truncated


def truncate_orbit_data(data, duration_seconds=None, n_samples=None):
    """
    Truncate a loaded OrbitData dataset to a shorter time segment.

    This function takes an OrbitData object and truncates all time-domain data
    (LTTs and derivatives) to the specified duration, then updates all metadata
    to be consistent with the truncated dataset.

    Parameters
    ----------
    data : OrbitData
        Loaded orbit data object (from load_orbits)
    duration_seconds : float, optional
        Duration to truncate to in seconds (mutually exclusive with n_samples)
    n_samples : int, optional
        Number of samples to keep (mutually exclusive with duration_seconds)

    Returns
    -------
    truncated_data : OrbitData
        New OrbitData object with truncated time series and updated metadata

    Notes
    -----
    - If both duration_seconds and n_samples are None, returns copy of original
    - If both are specified, raises ValueError
    - Truncation always starts from ltt_t0 (beginning of LTT dataset)
    - All LTT arrays and derivatives are truncated
    - Sampling metadata (duration, n_samples, etc.) is updated
    - Orbit data (positions/velocities) are NOT truncated as they have
      different sampling

    Examples
    --------
    >>> orbit_data = load_orbits("Orbits_LTTs.h5")
    >>> # Truncate to first 1 day
    >>> orbit_1day = truncate_orbit_data(orbit_data, duration_seconds=86400)
    >>>
    >>> # Truncate to first 10000 samples
    >>> orbit_short = truncate_orbit_data(orbit_data, n_samples=10000)
    """

    # Input validation
    if duration_seconds is not None and n_samples is not None:
        raise ValueError("Specify either duration_seconds OR n_samples, not both")

    if duration_seconds is None and n_samples is None:
        print(
            "Warning: No truncation parameters specified. "
            "Returning copy of original data."
        )
        return data

    # Calculate n_samples from duration if needed
    if duration_seconds is not None:
        n_samples_truncated = int(duration_seconds / data.ltt_dt)
        print(
            f"Truncating to {duration_seconds:.1f} seconds = "
            f"{n_samples_truncated:,} samples"
        )
    else:
        n_samples_truncated = n_samples
        duration_truncated = n_samples_truncated * data.ltt_dt
        print(
            f"Truncating to {n_samples_truncated:,} samples = "
            f"{duration_truncated:.1f} seconds "
            f"({duration_truncated/86400:.3f} days)"
        )

    # Validate truncation length
    if n_samples_truncated > data.ltt_n_samples:
        raise ValueError(
            f"Requested {n_samples_truncated} samples but LTT data only "
            f"has {data.ltt_n_samples}"
        )

    if n_samples_truncated <= 0:
        raise ValueError(f"Invalid truncation: n_samples must be > 0")

    # Create new OrbitData object for truncated data
    truncated = OrbitData()

    # Copy filepath and metadata
    truncated.filepath = data.filepath
    truncated.metadata = data.metadata.copy()

    # Update LTT sampling information
    truncated.ltt_t0 = data.ltt_t0  # Start time stays the same
    truncated.ltt_dt = data.ltt_dt  # Sampling period unchanged
    truncated.ltt_fs = data.ltt_fs  # Sampling frequency unchanged
    truncated.ltt_n_samples = n_samples_truncated
    truncated.ltt_duration = n_samples_truncated * data.ltt_dt

    # Create new TimeSampling object with truncated parameters
    truncated.ltt_sampling = TimeSampling(
        t0=truncated.ltt_t0, dt=truncated.ltt_dt, size=n_samples_truncated
    )
    truncated.t_ltt = truncated.ltt_sampling.t()

    print(f"\nTruncated LTT sampling:")
    print(
        f"  Original: {data.ltt_n_samples:,} samples, "
        f"{data.ltt_duration:.1f} s ({data.ltt_duration/86400:.3f} days)"
    )
    print(
        f"  Truncated: {truncated.ltt_n_samples:,} samples, "
        f"{truncated.ltt_duration:.1f} s "
        f"({truncated.ltt_duration/86400:.3f} days)"
    )
    print(
        f"  Kept: {100*n_samples_truncated/data.ltt_n_samples:.1f}% of "
        f"original data"
    )

    # Truncate LTT time series
    print(f"\nTruncating LTTs:")
    for link, ltt_array in data.ltts.items():
        truncated.ltts[link] = ltt_array[:n_samples_truncated]
        print(f"  ltt_{link}: {ltt_array.shape} → {truncated.ltts[link].shape}")

    print(f"\nTruncating LTT derivatives:")
    for link, ltt_deriv_array in data.ltt_derivatives.items():
        truncated.ltt_derivatives[link] = ltt_deriv_array[:n_samples_truncated]
        print(
            f"  ltt_derivative_{link}: {ltt_deriv_array.shape} → "
            f"{truncated.ltt_derivatives[link].shape}"
        )

    # Copy orbit data unchanged (different sampling rate, typically much coarser)
    truncated.orbit_sampling = data.orbit_sampling
    truncated.t_orbit = data.t_orbit
    truncated.orbit_fs = data.orbit_fs
    truncated.orbit_dt = data.orbit_dt
    truncated.orbit_t0 = data.orbit_t0
    truncated.orbit_duration = data.orbit_duration
    truncated.orbit_n_samples = data.orbit_n_samples

    for sc, pos_array in data.sc_positions.items():
        truncated.sc_positions[sc] = pos_array.copy()

    for sc, vel_array in data.sc_velocities.items():
        truncated.sc_velocities[sc] = vel_array.copy()

    print(
        f"\nNote: Spacecraft orbit data copied unchanged "
        f"(different sampling: {data.orbit_fs:.2e} Hz)"
    )
    print(f"\n✓ Truncation complete!")

    return truncated


def investigate_mojito_file_structure(filepath):
    """Recursively print HDF5 file structure with attributes and shapes"""

    def print_item(name, obj, indent=0):
        prefix = "  " * indent

        if isinstance(obj, h5py.Group):
            print(f"{prefix}📁 Group: {name}")
            # Print group attributes
            if obj.attrs:
                for attr_name, attr_val in obj.attrs.items():
                    print(f"{prefix}  └─ attr: {attr_name} = {attr_val}")

        elif isinstance(obj, h5py.Dataset):
            print(f"{prefix}Dataset: {name}")
            print(f"{prefix}  └─ shape: {obj.shape}, dtype: {obj.dtype}")
            # Print dataset attributes
            if obj.attrs:
                for attr_name, attr_val in obj.attrs.items():
                    print(f"{prefix}  └─ attr: {attr_name} = {attr_val}")
            # Show first few values for small arrays
            if obj.size < 10:
                print(f"{prefix}  └─ values: {obj[:]}")

    print(f"\n{'='*80}")
    print(f"HDF5 File Structure: {filepath}")
    print(f"{'='*80}\n")

    with h5py.File(filepath, "r") as f:
        # Print top-level attributes
        print("Top-level attributes:")
        for attr_name, attr_val in f.attrs.items():
            print(f"  └─ {attr_name} = {attr_val}")
        print()

        # Recursively visit all items
        f.visititems(print_item)
