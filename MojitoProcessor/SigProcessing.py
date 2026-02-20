"""
Signal Processing utilities for LISA TDI data

Provides a SignalProcessor class for filtering, decimating, trimming, and windowing
multi-channel time series data with automatic state tracking.
"""

import logging
import numpy as np
from fractions import Fraction
from scipy import signal
from scipy.signal.windows import tukey, blackmanharris, hann, hamming, blackman
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .mojito_loader import MojitoData

logger = logging.getLogger(__name__)


class SignalProcessor:
    """
    Signal processor for multi-channel time series data.
    
    Handles filtering, decimation, trimming, and windowing while automatically
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
    >>> filtered = sp.bandpass_filter(low=1e-4, high=1.0, order=6)
    >>> decimated, new_fs = sp.decimate(factor=2)
    >>> trimmed = sp.trim(duration=3600)
    >>> windowed = sp.apply_window(window='tukey', alpha=0.05)
    """
    
    def __init__(self, data: Dict[str, np.ndarray], fs: float):
        """
        Initialize SignalProcessor with multi-channel data.
        
        Parameters
        ----------
        data : dict
            Dictionary mapping channel names to 1D numpy arrays
        fs : float
            Sampling frequency in Hz
        """
        self.data = {ch: arr.copy() for ch, arr in data.items()}
        self.fs = float(fs)
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
    
    def bandpass_filter(
        self,
        low: float,
        high: float,
        order: int = 6,
        filter_type: str = 'butterworth',
        zero_phase: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Apply bandpass filter to all channels.
        
        Parameters
        ----------
        low : float
            Lower cutoff frequency in Hz
        high : float
            Upper cutoff frequency in Hz
        order : int, optional
            Filter order (default: 6)
        filter_type : str, optional
            Filter type: 'butterworth', 'chebyshev1', 'chebyshev2', 'bessel'
            (default: 'butterworth')
        zero_phase : bool, optional
            Use zero-phase filtering (filtfilt) if True, else single-pass (default: True)
        
        Returns
        -------
        filtered_data : dict
            Dictionary of filtered channel data
        """
        return self._apply_filter(low, high, 'bandpass', order, filter_type, zero_phase)
    
    def lowpass_filter(
        self,
        cutoff: float,
        order: int = 6,
        filter_type: str = 'butterworth',
        zero_phase: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Apply lowpass filter to all channels.
        
        Parameters
        ----------
        cutoff : float
            Cutoff frequency in Hz
        order : int, optional
            Filter order (default: 6)
        filter_type : str, optional
            Filter type (default: 'butterworth')
        zero_phase : bool, optional
            Use zero-phase filtering if True (default: True)
        
        Returns
        -------
        filtered_data : dict
            Dictionary of filtered channel data
        """
        return self._apply_filter(cutoff, None, 'lowpass', order, filter_type, zero_phase)
    
    def highpass_filter(
        self,
        cutoff: float,
        order: int = 6,
        filter_type: str = 'butterworth',
        zero_phase: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Apply highpass filter to all channels.
        
        Parameters
        ----------
        cutoff : float
            Cutoff frequency in Hz
        order : int, optional
            Filter order (default: 6)
        filter_type : str, optional
            Filter type (default: 'butterworth')
        zero_phase : bool, optional
            Use zero-phase filtering if True (default: True)
        
        Returns
        -------
        filtered_data : dict
            Dictionary of filtered channel data
        """
        return self._apply_filter(cutoff, None, 'highpass', order, filter_type, zero_phase)
    
    def _apply_filter(
        self,
        low_or_cutoff: float,
        high: Optional[float],
        btype: str,
        order: int,
        filter_type: str,
        zero_phase: bool
    ) -> Dict[str, np.ndarray]:
        """Internal method to apply filter to all channels."""
        # Determine critical frequencies
        if btype == 'bandpass':
            Wn = [low_or_cutoff, high]
        else:
            Wn = low_or_cutoff
        
        # Design filter
        filter_funcs = {
            'butterworth': signal.butter,
            'chebyshev1': signal.cheby1,
            'chebyshev2': signal.cheby2,
            'bessel': signal.bessel
        }
        
        if filter_type not in filter_funcs:
            raise ValueError(f"Unknown filter type: {filter_type}. "
                           f"Choose from {list(filter_funcs.keys())}")
        
        sos = filter_funcs[filter_type](order, Wn, btype=btype, fs=self.fs, output='sos')
        
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
        window: tuple = ('kaiser', 5.0),
        padtype: str = 'line'
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Resample all channels to a target sampling rate using polyphase filtering.

        Uses ``scipy.signal.resample_poly`` which applies a zero-phase FIR
        anti-aliasing filter via polyphase decomposition. Accepts arbitrary
        rational target rates (e.g., 4 Hz -> 0.4 Hz), unlike ``decimate``
        which requires an integer factor.

        Recommended pipeline order: filter → trim → truncate → resample_poly
        → window → FFT. Apply signal conditioning (e.g., high-pass) before
        downsampling so the polyphase filter sees clean data; ``resample_poly``
        handles the anti-aliasing low-pass internally.

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
        >>> sp.highpass_filter(cutoff=5e-6, order=2)
        >>> sp.trim(duration=trim_duration)
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
                self.data[ch],
                up,
                down,
                window=window,
                padtype=padtype
            )

        self.data = resampled_data
        self.fs = self.fs * up / down
        self._update_params()

        return resampled_data, self.fs

    def trim(
        self,
        duration: float,
        from_each_end: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Trim data by removing specified duration.
        
        Parameters
        ----------
        duration : float
            Duration to remove in seconds
        from_each_end : bool, optional
            If True, removes `duration` from both start and end.
            If False, removes `duration` only from the start (default: True)
        
        Returns
        -------
        trimmed_data : dict
            Dictionary of trimmed channel data
        """
        trim_samples = int(duration / self.dt)
        
        if from_each_end:
            if 2 * trim_samples >= self.N:
                raise ValueError(
                    f"Cannot trim {2*duration}s from {self.T}s data. "
                    f"Total trim exceeds data length."
                )
            trimmed_data = {ch: arr[trim_samples:-trim_samples] 
                          for ch, arr in self.data.items()}
        else:
            if trim_samples >= self.N:
                raise ValueError(
                    f"Cannot trim {duration}s from {self.T}s data. "
                    f"Trim exceeds data length."
                )
            trimmed_data = {ch: arr[trim_samples:] 
                          for ch, arr in self.data.items()}
        
        # Update internal state
        self.data = trimmed_data
        self._update_params()
        
        return trimmed_data
    
    def apply_window(
        self,
        window: str = 'tukey',
        **window_params
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
            'tukey': lambda N, p: tukey(N, **p),
            'blackmanharris': lambda N, p: blackmanharris(N),
            'hann': lambda N, p: hann(N),
            'hamming': lambda N, p: hamming(N),
            'blackman': lambda N, p: blackman(N)
        }
        
        if window not in window_funcs:
            raise ValueError(f"Unknown window type: {window}. "
                           f"Choose from {list(window_funcs.keys())}")
        
        # Set default parameters
        if window == 'tukey' and 'alpha' not in window_params:
            window_params['alpha'] = 0.05
        
        # Generate window
        win = window_funcs[window](self.N, window_params)
        
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
            'fs': self.fs,
            'N': self.N,
            'T': self.T,
            'dt': self.dt,
            'channels': self.channels
        }
    
    def __repr__(self):
        return (f"SignalProcessor(channels={self.channels}, "
                f"N={self.N}, fs={self.fs:.3f} Hz, T={self.T:.2f} s)")


# ============================================================================
# Standalone utility functions (for quick one-off operations)
# ============================================================================

def bandpass_filter(
    data: np.ndarray,
    fs: float,
    low: float,
    high: float,
    order: int = 6,
    filter_type: str = 'butterworth',
    zero_phase: bool = True
) -> np.ndarray:
    """
    Apply bandpass filter to single-channel data.
    
    Parameters
    ----------
    data : ndarray
        Input signal
    fs : float
        Sampling frequency in Hz
    low : float
        Lower cutoff frequency in Hz
    high : float
        Upper cutoff frequency in Hz
    order : int, optional
        Filter order (default: 6)
    filter_type : str, optional
        Filter type: 'butterworth', 'chebyshev1', 'chebyshev2', 'bessel'
    zero_phase : bool, optional
        Use zero-phase filtering (default: True)
    
    Returns
    -------
    filtered : ndarray
        Filtered signal
    """
    filter_funcs = {
        'butterworth': signal.butter,
        'chebyshev1': signal.cheby1,
        'chebyshev2': signal.cheby2,
        'bessel': signal.bessel
    }
    
    sos = filter_funcs[filter_type](order, [low, high], btype='bandpass', fs=fs, output='sos')
    
    if zero_phase:
        return signal.sosfiltfilt(sos, data)
    else:
        return signal.sosfilt(sos, data)


def apply_window(
    data: np.ndarray,
    window: str = 'tukey',
    **window_params
) -> np.ndarray:
    """
    Apply window function to single-channel data.
    
    Parameters
    ----------
    data : ndarray
        Input signal
    window : str, optional
        Window type: 'tukey', 'blackmanharris', 'hann', 'hamming', 'blackman'
    **window_params :
        Additional parameters (e.g., alpha=0.05 for Tukey)
    
    Returns
    -------
    windowed : ndarray
        Windowed signal
    """
    N = len(data)
    
    window_funcs = {
        'tukey': lambda N: tukey(N, alpha=window_params.get('alpha', 0.05)),
        'blackmanharris': lambda N: blackmanharris(N),
        'hann': lambda N: hann(N),
        'hamming': lambda N: hamming(N),
        'blackman': lambda N: blackman(N)
    }
    
    win = window_funcs[window](N)
    return data * win


def downsample(
    data: np.ndarray,
    fs: float,
    target_fs: float,
    window: tuple = ('kaiser', 31.0),
    padtype: str = 'line'
) -> Tuple[np.ndarray, float]:
    """
    Resample a single-channel signal to a target sampling rate.

    Standalone convenience function mirroring ``SignalProcessor.downsample``,
    for quick one-off operations without constructing a full processor.

    Parameters
    ----------
    data : ndarray
        1D input signal.
    fs : float
        Current sampling frequency in Hz.
    target_fs : float
        Desired output sampling frequency in Hz. Must satisfy
        ``0 < target_fs <= fs``.
    window : tuple or array_like, optional
        Window for FIR anti-aliasing filter design (default: ``('kaiser', 31.0)``).
    padtype : str, optional
        Edge-padding strategy (default: ``'line'``). See
        ``scipy.signal.resample_poly`` for full options.

    Returns
    -------
    resampled : ndarray
        Resampled signal.
    new_fs : float
        Actual output sampling frequency in Hz.

    Raises
    ------
    ValueError
        If ``target_fs <= 0`` or ``target_fs > fs``.

    Examples
    --------
    >>> from MojitoUtils import downsample
    >>> x_ds, new_fs = downsample(x, fs=4.0, target_fs=1.0)
    >>> print(new_fs)   # 1.0
    """
    if target_fs <= 0:
        raise ValueError(f"target_fs must be positive, got {target_fs}")
    if target_fs > fs:
        raise ValueError(
            f"target_fs ({target_fs} Hz) exceeds fs ({fs} Hz). "
            f"This function is for downsampling only."
        )

    ratio = Fraction(target_fs / fs).limit_denominator(10000)
    up, down = ratio.numerator, ratio.denominator

    if up == 0:
        raise ValueError(
            f"Cannot represent target_fs={target_fs} Hz as a rational "
            f"fraction of {fs} Hz within limit_denominator=10000."
        )

    resampled = signal.resample_poly(data, up, down, window=window, padtype=padtype)
    new_fs = fs * up / down
    return resampled, new_fs


def process_pipeline(
    data: 'MojitoData',
    channels: Optional[List[str]] = None,
    highpass_cutoff: float = 5e-6,
    lowpass_cutoff: Optional[float] = None,
    filter_order: int = 2,
    target_fs: Optional[float] = None,
    kaiser_window: Optional[float] = 31.0,
    trim_fraction: float = 0.022,
    truncate_days: Optional[float] = 4.0,
    window: str = 'tukey',
    window_alpha: float = 0.025,
) -> SignalProcessor:
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
    highpass_cutoff : float, optional
        High-pass cutoff frequency in Hz. Default ``5e-6`` Hz.
    lowpass_cutoff : float, optional
        If given, a band-pass filter ``[highpass_cutoff, lowpass_cutoff]`` is
        applied instead of a high-pass only. Must be less than the Nyquist of
        the input data and, if ``target_fs`` is also set, should be no greater
        than ``target_fs / 2``. Default ``None`` (high-pass only).
    filter_order : int, optional
        Zero-phase Butterworth filter order. Default ``2``.
    target_fs : float, optional
        Downsample to this sampling rate in Hz using polyphase resampling
        after filtering. Must be <= current ``fs``. Default ``None`` (no
        downsampling). For a band-pass with ``lowpass_cutoff=f_high``, set
        ``target_fs >= 2 * f_high`` to avoid aliasing.
    trim_fraction : float, optional
        Fraction of the post-downsample duration to remove from each end to
        eliminate filter ringing. Default ``0.022`` (≈2.2%).
    truncate_days : float, optional
        Keep only the first ``truncate_days`` days of the trimmed data.
        Default ``4.0``. Pass ``None`` to keep the full trimmed dataset.
    window : str, optional
        Window function: ``'tukey'``, ``'hann'``, ``'hamming'``,
        ``'blackman'``, ``'blackmanharris'``. Default ``'tukey'``.
    window_alpha : float, optional
        Taper fraction for the Tukey window (ignored for other types).
        Default ``0.025``.

    Returns
    -------
    sp : SignalProcessor
        Processed data ready for FFT analysis. Access windowed arrays via
        ``sp.data``, and sampling parameters via ``sp.fs``, ``sp.dt``,
        ``sp.N``, ``sp.T``.

    Examples
    --------
    >>> import logging
    >>> logging.basicConfig(level=logging.INFO)
    >>> from MojitoUtils import load_mojito_l1, process_pipeline
    >>> data = load_mojito_l1("mojito.h5")
    >>> # High-pass only, downsample to science-band rate
    >>> sp = process_pipeline(data, target_fs=0.4)
    >>> # Band-pass 5e-6 to 0.02 Hz, downsample to 0.1 Hz
    >>> sp = process_pipeline(data, lowpass_cutoff=0.02, target_fs=0.1)
    """
    if channels is None:
        channels = ['X', 'Y', 'Z']

    missing = [ch for ch in channels if ch not in data.tdis]
    if missing:
        raise ValueError(
            f"Channels {missing} not found in data. "
            f"Available: {list(data.tdis.keys())}"
        )

    # ------------------------------------------------------------------ #
    # Step 1 — initialise with the full dataset
    # ------------------------------------------------------------------ #
    sp = SignalProcessor({ch: data.tdis[ch] for ch in channels}, fs=data.fs)
    logger.info(
        "Step 1/5 | Init: %d samples @ %.4g Hz (%.2f days), channels=%s",
        sp.N, sp.fs, sp.T / 86400, channels,
    )

    # ------------------------------------------------------------------ #
    # Step 2 — filter (band-pass or high-pass)
    # ------------------------------------------------------------------ #
    if lowpass_cutoff is not None:
        sp.bandpass_filter(
            low=highpass_cutoff, high=lowpass_cutoff,
            order=filter_order, zero_phase=True,
        )
        logger.info(
            "Step 2/5 | Band-pass: [%.1e, %.1e] Hz, order=%d (zero-phase Butterworth)",
            highpass_cutoff, lowpass_cutoff, filter_order,
        )
    else:
        sp.highpass_filter(
            cutoff=highpass_cutoff, order=filter_order, zero_phase=True,
        )
        logger.info(
            "Step 2/5 | High-pass: cutoff=%.1e Hz, order=%d (zero-phase Butterworth)",
            highpass_cutoff, filter_order,
        )

    # ------------------------------------------------------------------ #
    # Step 3 — downsample (optional)
    # ------------------------------------------------------------------ #
    if target_fs is not None:
        pre_fs, pre_N = sp.fs, sp.N
        sp.downsample(target_fs=target_fs, window=('kaiser', kaiser_window))
        logger.info(
            "Step 3/5 | Resample: %.4g Hz → %.4g Hz, %d → %d samples "
            "(Nyquist = %.4g Hz)",
            pre_fs, sp.fs, pre_N, sp.N, sp.fs / 2,
        )
    else:
        logger.info("Step 3/5 | Resample: skipped (fs = %.4g Hz)", sp.fs)

    # ------------------------------------------------------------------ #
    # Step 4 — trim edge artefacts
    # ------------------------------------------------------------------ #
    trim_duration = trim_fraction * sp.N * sp.dt
    sp.trim(duration=trim_duration, from_each_end=True)
    logger.info(
        "Step 4/5 | Trim: %.1f h from each end → %d samples (%.2f days)",
        trim_duration / 3600, sp.N, sp.T / 86400,
    )

    # ------------------------------------------------------------------ #
    # Step 5 — truncate to working length
    # ------------------------------------------------------------------ #
    if truncate_days is not None:
        n_samples = min(int(truncate_days * 86400 * sp.fs), sp.N)
        sp = SignalProcessor(
            {ch: arr[:n_samples] for ch, arr in sp.data.items()}, fs=sp.fs
        )
        logger.info(
            "Step 5/5 | Truncate: %d samples (%.2f days)",
            sp.N, sp.T / 86400,
        )
    else:
        logger.info("Step 5/5 | Truncate: skipped (%.2f days)", sp.T / 86400)

    # ------------------------------------------------------------------ #
    # Step 6 — window
    # ------------------------------------------------------------------ #
    sp.apply_window(window=window, alpha=window_alpha)
    logger.info(
        "Step 6/6 | Window: %s (alpha=%.4g) | "
        "Ready — N=%d, fs=%.4g Hz, dt=%.4g s, T=%.4f days",
        window, window_alpha, sp.N, sp.fs, sp.dt, sp.T / 86400,
    )

    return sp