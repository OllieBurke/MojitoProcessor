"""
Extended gap mask computation.

Butterworth filters leak energy into the data on either side of every gap.
:func:`compute_extended_mask` quantifies this leakage by running the
complement of the gap mask (``1 − mask``) through the *same* filter that the
processing pipeline applied to the data, and marks any region where the
leakage exceeds a configurable threshold as excluded.  Short clean stretches
between excluded regions are merged via binary closing so that the result is
a smooth, conservative mask.

The function returns **both** the final binary mask and the raw filter-leakage
array so the caller can inspect the contamination level before committing to a
threshold.
"""

from fractions import Fraction
from typing import Tuple

import numpy as np
from scipy import signal as scipy_signal
from scipy.ndimage import binary_closing

__all__ = ["compute_extended_mask"]


def compute_extended_mask(
    smoothed_mask: np.ndarray,
    sp,
    filter_kwargs: dict,
    downsample_kwargs: dict,
    trim_kwargs: dict,
    *,
    fs_raw: float,
    contamination_threshold: float = 1e-4,
    min_clean_hours: float = 12.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute an extended binary mask that accounts for Butterworth filter leakage.

    Mirrors the downsampling and trimming steps applied to the data, then
    filters ``1 − smoothed_mask`` with the same Butterworth filter used in the
    pipeline.  Samples where the absolute leakage exceeds
    *contamination_threshold* are flagged as contaminated and excluded from the
    mask alongside the original gap samples.  Short clean intervals (shorter
    than *min_clean_hours*) between excluded regions are merged via binary
    closing.

    Parameters
    ----------
    smoothed_mask : ndarray
        1-D float array at the **raw** (pre-downsample) sampling rate.  Values
        in ``[0, 1]``:  ``1`` = data present, ``0`` = gap.  Typically the
        output of ``lisagap.GapWindowGenerator``.
    sp : SignalProcessor
        Processed segment (post-downsample, post-trim) whose length ``sp.N``
        defines the target output length.  The segment may represent only part
        of the full dataset (e.g. segment0 of many).
    filter_kwargs : dict
        Must contain ``"highpass_cutoff"`` (Hz).  Optionally ``"lowpass_cutoff"``
        and ``"order"``.  Must match the kwargs passed to
        :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.
    downsample_kwargs : dict
        Must contain ``"target_fs"`` (Hz).  Must match the kwargs passed to
        :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.
    trim_kwargs : dict
        Must contain ``"fraction"`` (trimmed fraction from each end).  Must
        match the kwargs passed to
        :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.
    fs_raw : float
        Sampling frequency (Hz) of *smoothed_mask* — the **raw** rate before
        downsampling.  Pass ``data["fs"]`` from the output of
        :func:`~MojitoProcessor.io.read.load_file`.
    contamination_threshold : float, optional
        Absolute amplitude of the filtered gap indicator above which a sample
        is considered contaminated.  Default: ``1e-4``.
    min_clean_hours : float, optional
        Short clean gaps (in hours) between excluded regions that are shorter
        than this value will be merged into the surrounding excluded region via
        binary closing.  Default: ``12.0``.

    Returns
    -------
    extended_mask_binary : ndarray
        Boolean-valued float array (``0.0`` or ``1.0``) of length ``sp.N``.
        ``1.0`` indicates samples that are clean and not contaminated.
    gap_contamination : ndarray
        Float array of length ``sp.N`` containing the raw filter-leakage
        signal ``filtered(1 − smoothed_mask)`` after downsampling and trimming.
        Inspect this to tune *contamination_threshold*.

    Examples
    --------
    >>> extended_mask, gap_contamination = compute_extended_mask(
    ...     smoothed_mask, sp_0,
    ...     filter_kwargs, downsample_kwargs, trim_kwargs,
    ...     fs_raw=data["fs"],
    ...     contamination_threshold=1e-4,
    ...     min_clean_hours=12.0,
    ... )
    """
    target_fs: float = downsample_kwargs["target_fs"]
    trim_fraction: float = trim_kwargs.get("fraction", 0.0)

    ratio = Fraction(target_fs / fs_raw).limit_denominator(10000)

    # ── Determine trim sample count at the *downsampled* rate ────────────────
    n_ds_total = scipy_signal.resample_poly(
        np.zeros(len(smoothed_mask)), ratio.numerator, ratio.denominator
    ).shape[0]
    trim_n = int(round(n_ds_total * trim_fraction / 2))

    def _downsample_trim(arr: np.ndarray) -> np.ndarray:
        ds = scipy_signal.resample_poly(arr, ratio.numerator, ratio.denominator)
        return ds[trim_n:-trim_n] if trim_n > 0 else ds

    # ── Design the same Butterworth filter as the pipeline ───────────────────
    highpass: float = filter_kwargs["highpass_cutoff"]
    lowpass = filter_kwargs.get("lowpass_cutoff", None)
    order: int = int(filter_kwargs.get("order", 2))

    if lowpass is not None:
        sos = scipy_signal.butter(
            order, [highpass, lowpass], btype="bandpass", fs=fs_raw, output="sos"
        )
    else:
        sos = scipy_signal.butter(
            order, highpass, btype="highpass", fs=fs_raw, output="sos"
        )

    # ── Filter the gap indicator ──────────────────────────────────────────────
    gap_indicator = 1.0 - np.asarray(smoothed_mask, dtype=float)
    gap_filtered = scipy_signal.sosfiltfilt(sos, gap_indicator)

    # ── Downsample and trim ───────────────────────────────────────────────────
    gap_contamination = _downsample_trim(gap_filtered)
    smoothed_mask_ds = _downsample_trim(smoothed_mask.astype(float))

    # ── Length-match to sp.N ─────────────────────────────────────────────────
    # resample_poly rounding and segmentation can cause a small offset;
    # clip or pad to sp.N so masks broadcast correctly against sp.data[ch].
    def _match_length(arr: np.ndarray, n: int) -> np.ndarray:
        if len(arr) >= n:
            return arr[:n]
        return np.pad(arr, (0, n - len(arr)), constant_values=arr[-1])

    gap_contamination = _match_length(gap_contamination, sp.N)
    smoothed_mask_ds = _match_length(smoothed_mask_ds, sp.N)

    # ── Build extended mask ───────────────────────────────────────────────────
    original_gap = smoothed_mask_ds < 0.5
    contaminated = np.abs(gap_contamination) > contamination_threshold
    excluded = original_gap | contaminated

    # Merge short clean stretches between excluded regions
    min_clean_samples = max(1, int(min_clean_hours * 3600.0 * target_fs))
    excluded = binary_closing(excluded, structure=np.ones(min_clean_samples))

    extended_mask_binary = (~excluded).astype(float)

    return extended_mask_binary, gap_contamination
