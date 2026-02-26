"""
Signal Processing utilities for LISA TDI data

Provides a SignalProcessor class for filtering, decimating, trimming, and windowing
multi-channel time series data with automatic state tracking.
"""

import logging
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.signal.windows import blackman, blackmanharris, hamming, hann, tukey

__all__ = [
    "SignalProcessor",
    "process_pipeline",
]

# Default Kaiser window beta parameter for anti-aliasing filter
KAISER_BETA_DEFAULT = 31.0

logger = logging.getLogger(__name__)


class SignalProcessor:
    """
    Signal processor for multi-channel time series data.

    Handles filtering, downsampling, trimming, and windowing while automatically
    tracking sampling parameters (fs, N, T, dt).

    Parameters
    ----------
    data : dict
        Dictionary of channel data, e.g., {'X': array, 'Y': array, 'Z': array}
    fs : float
        Sampling frequency in Hz

    Attributes
    ----------
    data : dict
        Current processed data (updated after each operation)
    fs : float
        Current sampling frequency in Hz
    N : int
        Current number of samples per channel
    T : float
        Current duration in seconds
    dt : float
        Current sampling period in seconds
    channels : list
        List of channel names

    Example
    -------
    >>> sp = SignalProcessor({'X': x_data, 'Y': y_data}, fs=4.0)
    >>> filtered = sp.filter(low=1e-4, high=1.0, order=6)
    >>> trimmed = sp.trim(fraction=0.02)  # Trim 2% total (1% each end)
    >>> windowed = sp.apply_window(window='tukey', alpha=0.05)
    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        fs: float,
        t0: Optional[float] = None,
    ):
        """
        Initialize SignalProcessor with multi-channel data.

        Parameters
        ----------
        data : dict
            Dictionary mapping channel names to 1D numpy arrays
        fs : float
            Sampling frequency in Hz
        t0 : float, optional
            TCB timestamp of the first sample in seconds. Should be set from
            the L1 data file (``data["t_tdi"][0]``). Defaults to ``None``
            when working outside of a full Mojito pipeline.
        """
        self.data = {ch: arr.copy() for ch, arr in data.items()}
        self.fs = float(fs)
        self.t0 = float(t0) if t0 is not None else None
        self.channels = list(data.keys())

        # Validate all channels have same length
        lengths = [len(arr) for arr in self.data.values()]
        if len(set(lengths)) != 1:
            raise ValueError(f"All channels must have same length. Got: {lengths}")

        self._update_params()

    def _update_params(self):
        """Update derived parameters N, T, dt based on current data and fs."""
        self.N = len(self.data[self.channels[0]])
        self.dt = 1.0 / self.fs
        self.T = self.N * self.dt

    def filter(
        self,
        *,
        low: Optional[float] = None,
        high: Optional[float] = None,
        order: int = 2,
        filter_type: str = "butterworth",
        zero_phase: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Apply filter to all channels (auto-detects highpass/lowpass/bandpass).

        Automatically determines filter type based on provided cutoff frequencies:
        - Only `low` set: highpass filter
        - Only `high` set: lowpass filter
        - Both `low` and `high` set: bandpass filter

        Parameters
        ----------
        low : float, optional
            Lower cutoff frequency in Hz (highpass)
        high : float, optional
            Upper cutoff frequency in Hz (lowpass)
        order : int, optional
            Filter order (default: 6)
        filter_type : str, optional
            Filter type: 'butterworth', 'chebyshev1', 'chebyshev2',
            'bessel' (default: 'butterworth')
        zero_phase : bool, optional
            Use zero-phase filtering (filtfilt) if True, else single-pass
            (default: True)

        Returns
        -------
        filtered_data : dict
            Dictionary of filtered channel data

        Raises
        ------
        ValueError
            If neither `low` nor `high` is provided

        Examples
        --------
        >>> # Highpass only
        >>> sp.filter(low=5e-6, order=2)
        >>> # Lowpass only
        >>> sp.filter(high=0.1, order=2)
        >>> # Bandpass
        >>> sp.filter(low=1e-4, high=0.1, order=6)
        """
        if low is None and high is None:
            raise ValueError("Must specify at least one of 'low' or 'high' cutoff")

        nyquist = self.fs / 2

        # Validate cutoff frequencies
        if low is not None:
            if low <= 0:
                raise ValueError(f"low cutoff must be positive, got {low}")
            if low >= nyquist:
                raise ValueError(
                    f"low cutoff ({low} Hz) must be below Nyquist ({nyquist} Hz)"
                )
        if high is not None:
            if high <= 0:
                raise ValueError(f"high cutoff must be positive, got {high}")
            if high >= nyquist:
                raise ValueError(
                    f"high cutoff ({high} Hz) must be below Nyquist ({nyquist} Hz)"
                )
        if low is not None and high is not None and low >= high:
            raise ValueError(
                f"low cutoff ({low} Hz) must be less than high cutoff ({high} Hz)"
            )
        if order <= 0 or not isinstance(order, int):
            raise ValueError(f"Filter order must be a positive integer, got {order}")

        # Auto-detect filter type
        if low is not None and high is not None:
            btype = "bandpass"
            Wn = [low, high]
        elif low is not None:
            btype = "highpass"
            Wn = low
        else:  # high is not None
            btype = "lowpass"
            Wn = high

        # Design filter — cheby1/cheby2 require extra ripple/attenuation args
        # butterworth/bessel called via partial to unify the interface
        def _butter(n, wn, **kw):
            return signal.butter(n, wn, **kw)

        def _bessel(n, wn, **kw):
            return signal.bessel(n, wn, **kw)

        def _cheby1(n, wn, **kw):
            return signal.cheby1(n, 1.0, wn, **kw)

        def _cheby2(n, wn, **kw):
            return signal.cheby2(n, 40.0, wn, **kw)

        filter_funcs = {
            "butterworth": _butter,
            "bessel": _bessel,
            "chebyshev1": _cheby1,
            "chebyshev2": _cheby2,
        }

        if filter_type not in filter_funcs:
            raise ValueError(
                f"Unknown filter type: {filter_type}. "
                f"Choose from {list(filter_funcs.keys())}"
            )

        sos = filter_funcs[filter_type](
            order, Wn, btype=btype, fs=self.fs, output="sos"
        )

        # Apply filter to all channels
        filtered_data = {}
        for ch in self.channels:
            if zero_phase:
                filtered_data[ch] = signal.sosfiltfilt(sos, self.data[ch])
            else:
                filtered_data[ch] = signal.sosfilt(sos, self.data[ch])

        # Update internal state
        self.data = filtered_data
        # fs, N, T, dt remain unchanged after filtering

        return filtered_data

    def downsample(
        self,
        target_fs: float,
        window: tuple = ("kaiser", KAISER_BETA_DEFAULT),
        padtype: str = "line",
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Resample all channels to a target sampling rate using polyphase filtering.

        Uses ``scipy.signal.resample_poly`` which applies a zero-phase FIR
        anti-aliasing filter via polyphase decomposition. Accepts arbitrary
        rational target rates (e.g., 4 Hz -> 0.4 Hz), unlike ``decimate``
        which requires an integer factor.

        Parameters
        ----------
        target_fs : float
            Desired output sampling frequency in Hz. Must be positive and
            less than or equal to the current sampling frequency (this method
            is a downsampler).
        window : tuple or array_like, optional
            Window specification passed to ``scipy.signal.resample_poly`` for
            FIR anti-aliasing filter design. Default ``('kaiser', 5.0)`` is
            scipy's own default and gives good stopband attenuation.
        padtype : str, optional
            Edge-padding strategy. Options: ``'line'`` (default), ``'constant'``,
            ``'mean'``, ``'median'``, ``'maximum'``, ``'minimum'``.
            ``'line'`` extends the signal linearly from each end, reducing
            edge transients for slowly-varying data such as LISA TDI channels.

        Returns
        -------
        resampled_data : dict
            Dictionary mapping channel names to resampled 1D arrays.
        new_fs : float
            Actual output sampling frequency in Hz (exact rational result
            ``self.fs * up / down``).

        Raises
        ------
        ValueError
            If ``target_fs`` is not positive.
        ValueError
            If ``target_fs`` exceeds the current sampling frequency.
        ValueError
            If the rational approximation of ``target_fs / self.fs`` produces
            ``up == 0``.

        Notes
        -----
        The up/down integers are computed via::

            ratio = Fraction(target_fs / self.fs).limit_denominator(10000)
            up, down = ratio.numerator, ratio.denominator

        Common use cases from 4 Hz source data:

        * 4 Hz -> 1 Hz:    up=1, down=4
        * 4 Hz -> 0.4 Hz:  up=1, down=10
        * 4 Hz -> 2 Hz:    up=1, down=2
        * 4 Hz -> 3 Hz:    up=3, down=4

        Examples
        --------
        >>> sp = SignalProcessor({'X': x_data, 'Y': y_data}, fs=4.0)
        >>> sp.filter(low=5e-6, order=2)
        >>> sp.trim(fraction=0.022)  # Trim 2.2% total
        >>> resampled, new_fs = sp.downsample(target_fs=1.0)
        >>> print(new_fs)   # 1.0
        """
        if target_fs <= 0:
            raise ValueError(f"target_fs must be positive, got {target_fs}")
        if target_fs > self.fs:
            raise ValueError(
                f"target_fs ({target_fs} Hz) exceeds current sampling frequency "
                f"({self.fs} Hz). resample_poly is a downsampler; for upsampling "
                f"use scipy.signal.resample_poly directly."
            )

        ratio = Fraction(target_fs / self.fs).limit_denominator(10000)
        up, down = ratio.numerator, ratio.denominator

        if up == 0:
            raise ValueError(
                f"Cannot represent target_fs={target_fs} Hz as a rational "
                f"fraction of {self.fs} Hz within limit_denominator=10000. "
                f"The target rate is too far below the source rate."
            )

        resampled_data = {}
        for ch in self.channels:
            resampled_data[ch] = signal.resample_poly(
                self.data[ch], up, down, window=window, padtype=padtype
            )

        self.data = resampled_data
        self.fs = self.fs * up / down
        self._update_params()

        return resampled_data, self.fs

    def trim(self, fraction: float) -> Dict[str, np.ndarray]:
        """
        Trim data by removing a fraction of the dataset.

        Parameters
        ----------
        fraction : float
            Total fraction of data to remove (e.g., 0.01 = 1%).

        Returns
        -------
        trimmed_data : dict
            Dictionary of trimmed channel data

        Raises
        ------
        ValueError
            If fraction is not in range [0, 1] or would remove all data

        Examples
        --------
        >>> # Trim 2% total (1% from each end)
        >>> sp.trim(fraction=0.02)
        >>> # Trim 5% from start only
        >>> sp.trim(fraction=0.05)
        """
        if not 0 <= fraction < 1:
            raise ValueError(f"fraction must be in [0, 1), got {fraction}")

        # No trimming needed
        if fraction == 0:
            return dict(self.data)

        # Split fraction equally between both ends
        trim_samples = int(self.N * fraction / 2)
        if trim_samples == 0:
            logger.warning(
                "trim: fraction %.2e too small to remove any samples "
                "(need at least %d samples per end). Skipping trim.",
                fraction,
                1,
            )
            return dict(self.data)
        if 2 * trim_samples >= self.N:
            raise ValueError(
                f"Cannot trim {fraction*100:.1f}% from both ends "
                f"({2*trim_samples} samples). Would remove all data."
            )
        trimmed_data = {
            ch: arr[trim_samples:-trim_samples] for ch, arr in self.data.items()
        }

        # Update internal state
        self.data = trimmed_data
        if self.t0 is not None:
            self.t0 += trim_samples * self.dt
        self._update_params()

        return trimmed_data

    def apply_window(
        self, window: str = "tukey", **window_params
    ) -> Dict[str, np.ndarray]:
        """
        Apply window function to all channels.

        Parameters
        ----------
        window : str, optional
            Window type: 'tukey', 'blackmanharris', 'hann', 'hamming', 'blackman'
            (default: 'tukey')
        **window_params :
            Additional parameters for window function.
            For 'tukey': alpha (default: 0.05)
            Other windows typically don't need parameters.

        Returns
        -------
        windowed_data : dict
            Dictionary of windowed channel data

        Examples
        --------
        >>> sp.apply_window('tukey', alpha=0.05)
        >>> sp.apply_window('blackmanharris')
        >>> sp.apply_window('hann')
        """
        # Define available windows
        window_funcs = {
            "tukey": lambda N, _: tukey(N),
            "blackmanharris": lambda N, _: blackmanharris(N),
            "hann": lambda N, _: hann(N),
            "hamming": lambda N, _: hamming(N),
            "blackman": lambda N, _: blackman(N),
        }

        if window not in window_funcs:
            raise ValueError(
                f"Unknown window type: {window}. "
                f"Choose from {list(window_funcs.keys())}"
            )

        # Set default alpha for tukey
        if window == "tukey" and "alpha" not in window_params:
            window_params["alpha"] = 0.05

        # Warn if extra kwargs passed to non-tukey windows (they will be ignored)
        if window != "tukey" and window_params:
            logger.warning(
                "apply_window: extra parameters %s ignored for '%s' window",
                list(window_params.keys()),
                window,
            )

        # For tukey, only pass 'alpha' — any other kwargs would crash scipy
        if window == "tukey":
            win = tukey(self.N, alpha=window_params.get("alpha", 0.05))
        else:
            win = window_funcs[window](self.N, {})

        # Apply window to all channels
        windowed_data = {ch: arr * win for ch, arr in self.data.items()}

        # Update internal state
        self.data = windowed_data
        # fs, N, T, dt remain unchanged after windowing

        return windowed_data

    def get_params(self) -> dict:
        """
        Get current signal parameters.

        Returns
        -------
        params : dict
            Dictionary containing fs, N, T, dt, and channels
        """
        return {
            "fs": self.fs,
            "N": self.N,
            "T": self.T,
            "dt": self.dt,
            "channels": self.channels,
        }

    def __repr__(self):
        t0_str = f"{self.t0:.6g} s" if self.t0 is not None else "None"
        return (
            f"SignalProcessor(channels={self.channels}, "
            f"N={self.N}, fs={self.fs:.3f} Hz, T={self.T:.2f} s, t0={t0_str})"
        )


def process_pipeline(
    data: dict,
    channels: Optional[List[str]] = None,
    *,
    filter_kwargs: Optional[Dict] = None,
    downsample_kwargs: Optional[Dict] = None,
    trim_kwargs: Optional[Dict] = None,
    truncate_kwargs: Optional[Dict] = None,
    window_kwargs: Optional[Dict] = None,
) -> Dict[str, SignalProcessor]:
    """
    Run the full TDI data processing pipeline on a MojitoData object.

    Applies the following steps in order:

    1. **Filter** — band-pass (if ``lowpass_cutoff`` given) or high-pass only
    2. **Downsample** — polyphase resampling to ``target_fs`` (optional)
    3. **Trim** — removes edge artefacts introduced by the filter from both ends
    4. **Truncate** — selects the first ``truncate_days`` of the processed data
    5. **Window** — tapers edges to reduce spectral leakage

    Pipeline progress is emitted at ``logging.INFO`` level via the
    ``MojitoUtils.SigProcessing`` logger.

    Parameters
    ----------
    data : MojitoData
        Loaded LISA L1 data object (from ``load_mojito_l1``). Must have
        ``data.tdis`` (dict of channel arrays) and ``data.fs`` (sampling rate).
    channels : list of str, optional
        TDI channels to process. Default ``['X', 'Y', 'Z']``.
    filter_kwargs : dict, optional
        Filter parameters. Keys:
        - ``highpass_cutoff`` (float): High-pass cutoff in Hz (default: 5e-6)
        - ``lowpass_cutoff`` (float, optional): Low-pass cutoff for band-pass
        - ``order`` (int): Filter order (default: 2)
        - ``filter_type`` (str): Filter type (default: 'butterworth')
    downsample_kwargs : dict, optional
        Downsampling parameters. Keys:
        - ``target_fs`` (float): Target sampling rate in Hz
        - ``kaiser_window`` (float): Kaiser window beta parameter (default: 31.0)
    trim_kwargs : dict, optional
        Trimming parameters. Keys:
        - ``fraction`` (float): Fraction to trim from each end (default: 0.022)
    truncate_kwargs : dict, optional
        Segmentation parameters. Keys:
        - ``days`` (float): Segment length in days (default: 4.0)
        Dataset is split into non-overlapping segments of this length.
        Each segment is independently windowed. Set to ``None`` to disable
        segmentation (returns single segment with full dataset).
        Note: Remainder samples shorter than a full segment are discarded.
    window_kwargs : dict, optional
        Windowing parameters. Keys:
        - ``window`` (str): Window type - 'tukey', 'hann', etc. (default: 'tukey')
        - ``alpha`` (float): Taper fraction for Tukey window (default: 0.025)

    Returns
    -------
    segments : dict of SignalProcessor
        Dictionary mapping segment names ('segment0', 'segment1', ...) to
        SignalProcessor objects. Each segment contains windowed data ready
        for FFT analysis. Access via ``segments['segment0'].data``,
        ``segments['segment0'].fs``, etc.

    """
    # Set defaults
    if channels is None:
        channels = ["X", "Y", "Z"]

    if filter_kwargs is None:
        filter_kwargs = {}
    if downsample_kwargs is None:
        downsample_kwargs = {}
    if trim_kwargs is None:
        trim_kwargs = {}
    if truncate_kwargs is None:
        truncate_kwargs = {}
    if window_kwargs is None:
        window_kwargs = {}

    # Extract filter parameters with defaults
    highpass_cutoff = filter_kwargs.get("highpass_cutoff", 5e-6)
    lowpass_cutoff = filter_kwargs.get("lowpass_cutoff", None)
    filter_order = filter_kwargs.get("order", 2)

    # Extract downsample parameters
    target_fs = downsample_kwargs.get("target_fs", None)
    kaiser_window = downsample_kwargs.get("kaiser_window", KAISER_BETA_DEFAULT)

    # Extract trim parameters
    trim_fraction = trim_kwargs.get("fraction", 0.02)

    # Extract truncate parameters
    truncate_days = truncate_kwargs.get("days", 4.0) if truncate_kwargs else None

    # Extract window parameters
    window = window_kwargs.get("window", "tukey")
    window_alpha = window_kwargs.get("alpha", 0.025)

    # Validate Tukey window alpha
    if window == "tukey" and not 0 <= window_alpha <= 1:
        raise ValueError(f"Tukey window alpha must be in [0, 1], got {window_alpha}")

    # Validate filter order
    filter_order_raw = filter_kwargs.get("order", 2)
    if not isinstance(filter_order_raw, int) or filter_order_raw <= 0:
        raise ValueError(
            f"filter order must be a positive integer, got {filter_order_raw}"
        )

    # Validate kaiser_window beta
    if kaiser_window < 0:
        raise ValueError(
            f"kaiser_window beta must be non-negative, got {kaiser_window}"
        )

    # Validate truncate_days
    if truncate_days is not None and truncate_days <= 0:
        raise ValueError(
            f"truncate_kwargs 'days' must be positive, got {truncate_days}"
        )

    # Warn if lowpass_cutoff exceeds Nyquist of the target sampling rate
    if lowpass_cutoff is not None and target_fs is not None:
        target_nyquist = target_fs / 2
        if lowpass_cutoff > target_nyquist:
            logger.warning(
                "lowpass_cutoff (%.4g Hz) exceeds Nyquist of target_fs "
                "(%.4g Hz). Frequencies above %.4g Hz will be aliased after "
                "downsampling. Consider setting lowpass_cutoff <= %.4g Hz.",
                lowpass_cutoff,
                target_nyquist,
                target_nyquist,
                target_nyquist,
            )

    missing = [ch for ch in channels if ch not in data["tdis"]]
    if missing:
        raise ValueError(
            f"Channels {missing} not found in data. "
            f"Available: {list(data['tdis'].keys())}"
        )

    if "t_tdi" not in data:
        raise ValueError(
            "'t_tdi' is required in the data dict. "
            "Set data['t_tdi'] to the array of TCB timestamps for the TDI samples "
            "(e.g. tdi_sampling.t() from MojitoL1File)."
        )

    # ------------------------------------------------------------------ #
    # Step 1 — initialise with the full dataset
    # ------------------------------------------------------------------ #
    sp = SignalProcessor(
        {ch: data["tdis"][ch] for ch in channels},
        fs=data["fs"],
        t0=float(data["t_tdi"][0]),
    )
    logger.info(
        "Step 1/5 | Init: %d samples @ %.4g Hz (%.2f days), channels=%s",
        sp.N,
        sp.fs,
        sp.T / 86400,
        channels,
    )

    # ------------------------------------------------------------------ #
    # Step 2 — filter (band-pass or high-pass)
    # ------------------------------------------------------------------ #
    if lowpass_cutoff is not None:
        sp.filter(
            low=highpass_cutoff,
            high=lowpass_cutoff,
            order=filter_order,
            zero_phase=True,
        )
        logger.info(
            "Step 2/5 | Band-pass: [%.1e, %.1e] Hz, order=%d (zero-phase Butterworth)",
            highpass_cutoff,
            lowpass_cutoff,
            filter_order,
        )
    else:
        sp.filter(
            low=highpass_cutoff,
            order=filter_order,
            zero_phase=True,
        )
        logger.info(
            "Step 2/5 | High-pass: cutoff=%.1e Hz, order=%d (zero-phase Butterworth)",
            highpass_cutoff,
            filter_order,
        )

    # ------------------------------------------------------------------ #
    # Step 3 — downsample (optional)
    # ------------------------------------------------------------------ #
    if target_fs is not None and target_fs != sp.fs:
        pre_fs, pre_N = sp.fs, sp.N
        sp.downsample(target_fs=target_fs, window=("kaiser", kaiser_window))
        logger.info(
            "Step 3/5 | Resample: %.4g Hz → %.4g Hz, %d → %d samples "
            "(Nyquist = %.4g Hz)",
            pre_fs,
            sp.fs,
            pre_N,
            sp.N,
            sp.fs / 2,
        )
    else:
        logger.info("Step 3/5 | Resample: skipped (fs = %.4g Hz)", sp.fs)

    # ------------------------------------------------------------------ #
    # Step 4 — trim edge artefacts
    # ------------------------------------------------------------------ #
    sp.trim(fraction=trim_fraction)
    logger.info(
        "Step 4/5 | Trim: %.2f%% from each end → %d samples (%.2f days)",
        trim_fraction,
        sp.N,
        sp.T / 86400,
    )

    # ------------------------------------------------------------------ #
    # Step 5 — segment into chunks and window each independently
    # ------------------------------------------------------------------ #
    if truncate_days is not None and truncate_days > 0:
        segment_samples = int(truncate_days * 86400 * sp.fs)
        n_segments = sp.N // segment_samples

        if n_segments == 0:
            logger.warning(
                "Step 5/5 | Segment: data (%.2f days) shorter than segment "
                "length (%.2f days) - creating single segment",
                sp.T / 86400,
                truncate_days,
            )
            n_segments = 1
            segment_samples = sp.N

        segments = {}
        for i in range(n_segments):
            start_idx = i * segment_samples
            end_idx = min(start_idx + segment_samples, sp.N)

            # Create segment — t0 advances by i full segment lengths after trimming
            segment_data = {ch: arr[start_idx:end_idx] for ch, arr in sp.data.items()}
            segment_t0 = (
                sp.t0 + i * segment_samples * sp.dt if sp.t0 is not None else None
            )
            seg_sp = SignalProcessor(segment_data, fs=sp.fs, t0=segment_t0)

            # Apply window to this segment
            seg_sp.apply_window(window=window, alpha=window_alpha)

            segments[f"segment{i}"] = seg_sp

        logger.info(
            "Step 5/5 | Segment: created %d segments × %.2f days each | "
            "Window: %s (alpha=%.4g)",
            n_segments,
            truncate_days,
            window,
            window_alpha,
        )

        return segments

    # No segmentation - apply window to full dataset
    sp.apply_window(window=window, alpha=window_alpha)
    logger.info(
        "Step 5/5 | Window: %s (alpha=%.4g) applied to full dataset | "
        "Ready — N=%d, fs=%.4g Hz, dt=%.4g s, T=%.4f days",
        window,
        window_alpha,
        sp.N,
        sp.fs,
        sp.dt,
        sp.T / 86400,
    )

    return {"segment0": sp}
