"""
Load, mask, and process a MojitoL1 HDF5 file through the full gaps pipeline.

Steps applied in order:

1. Load raw L1 data from an HDF5 file.
2. Apply a pre-computed smoothed gap mask to the raw TDI channels.
3. Run the standard processing pipeline (filter → downsample → trim).
   Truncation and windowing are intentionally omitted here — gap-based
   segmentation replaces fixed-length chunking, and each clean segment is
   windowed independently.
4. Compute an extended binary mask that accounts for Butterworth filter
   leakage around each gap.
5. Extract contiguous clean segments defined by the extended mask.
6. Apply a window to each clean segment (optional).
7. Write clean segments to HDF5 (optional).

Returns the clean segments and the extended mask so the caller can inspect
contamination levels or plot the result before further analysis.

Can be run as a script::

    python -m MojitoProcessor.pipelines.gapspipeline path/to/data.h5 \\
        --mask-path smoothed_mask.npy \\
        --output processed_gaps.h5 \\
        --target-fs 0.2
"""

import argparse
import logging
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..gaps import apply_raw_mask, compute_extended_mask, extract_clean_segments
from ..io.read import load_file
from ..io.write import write
from ..process.sigprocess import SignalProcessor, process_pipeline

__all__ = ["gapspipeline"]

logger = logging.getLogger(__name__)


def gapspipeline(
    path: str | pathlib.Path,
    smoothed_mask: np.ndarray,
    channels: Optional[List[str]] = None,
    *,
    load_days: Optional[float] = None,
    filter_kwargs: Optional[dict] = None,
    downsample_kwargs: Optional[dict] = None,
    trim_kwargs: Optional[dict] = None,
    window_kwargs: Optional[dict] = None,
    contamination_threshold: float = 1e-4,
    min_clean_hours: float = 12.0,
    output_path: Optional[str | pathlib.Path] = None,
) -> Tuple[Dict[str, SignalProcessor], np.ndarray]:
    """
    Load a MojitoL1 file and run the full gap-aware processing pipeline.

    Parameters
    ----------
    path : str or Path
        Path to the MojitoL1 ``.h5`` input file.
    smoothed_mask : ndarray
        1-D float array at the **raw** sampling rate (length must match the
        number of TDI samples in the file).  Values in ``[0, 1]``: ``1``
        means data present, ``0`` means gap.  Typically produced by
        ``lisagap.GapWindowGenerator``.
    channels : list of str, optional
        TDI channels to process. Default ``['X', 'Y', 'Z']``.
    load_days : float, optional
        Number of days to load from the file (lazy slicing).
        ``None`` loads the full dataset.
    filter_kwargs : dict, optional
        Filter parameters forwarded to
        :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.
        Keys: ``highpass_cutoff`` (Hz), ``lowpass_cutoff`` (Hz, optional),
        ``order`` (int).
    downsample_kwargs : dict, optional
        Downsampling parameters forwarded to
        :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.
        Keys: ``target_fs`` (Hz), ``kaiser_window`` (float).
    trim_kwargs : dict, optional
        Trimming parameters forwarded to
        :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.
        Keys: ``fraction`` (float).
    window_kwargs : dict, optional
        Window applied independently to each clean segment after extraction.
        Omit (or pass ``None``) to skip windowing.
        Keys: ``window`` (str), ``alpha`` (float).
    contamination_threshold : float, optional
        Absolute leakage amplitude above which a sample is flagged as
        contaminated by the filter ringing around a gap.  Default ``1e-4``.
    min_clean_hours : float, optional
        Minimum duration (hours) a contiguous clean stretch must have to be
        kept as a segment.  Shorter stretches are discarded.  Default ``12.0``.
    output_path : str or Path, optional
        If given, write clean segments and raw auxiliary data to this HDF5
        file via :func:`~MojitoProcessor.io.write.write`.

    Returns
    -------
    clean_segments : dict of SignalProcessor
        Contiguous clean segments keyed by ``'segment0'``, ``'segment1'``,
        etc.  Each segment has had the window applied (if *window_kwargs* was
        provided) and is ready for FFT / Whittle analysis.
    extended_mask : ndarray
        Binary float array (``0.0`` / ``1.0``) aligned to the full processed
        time series (before gap extraction), length ``sp.N``.  ``1.0``
        indicates samples that are clean and uncontaminated.  Useful for
        inspection and plotting.
    """
    # ── Step 1: load ─────────────────────────────────────────────────────────
    logger.info("Loading %s", path)
    data = load_file(path, load_days=load_days)

    # ── Step 2: apply gap mask to raw TDI channels ───────────────────────────
    logger.info(
        "Applying gap mask (gap fraction: %.4f%%)",
        (1.0 - float(np.mean(smoothed_mask))) * 100,
    )
    data_masked = apply_raw_mask(data, smoothed_mask)

    # ── Step 3: process pipeline (no truncation, no window) ──────────────────
    # Truncation is replaced by gap-based segmentation; each clean segment is
    # windowed independently below.
    logger.info("Running processing pipeline")
    processed = process_pipeline(
        data_masked,
        channels=channels,
        filter_kwargs=filter_kwargs,
        downsample_kwargs=downsample_kwargs,
        trim_kwargs=trim_kwargs,
        truncate_kwargs=None,
        window_kwargs=None,
    )
    sp = processed["segment0"]

    # ── Step 4: compute extended mask (filter leakage around gaps) ───────────
    # compute_extended_mask requires target_fs; default to the raw rate if
    # no downsampling was requested (identity resampling, ratio = 1).
    _dkw = dict(downsample_kwargs or {})
    if "target_fs" not in _dkw:
        _dkw["target_fs"] = data["fs"]

    _tkw = trim_kwargs or {}
    _fkw = filter_kwargs or {}

    logger.info(
        "Computing extended mask (contamination_threshold=%.2e, min_clean_hours=%.1f h)",
        contamination_threshold,
        min_clean_hours,
    )
    extended_mask, _ = compute_extended_mask(
        smoothed_mask,
        sp,
        _fkw,
        _dkw,
        _tkw,
        fs_raw=data["fs"],
        contamination_threshold=contamination_threshold,
        min_clean_hours=min_clean_hours,
    )
    logger.info(
        "Extended gap fraction: %.4f%%", (1.0 - float(np.mean(extended_mask))) * 100
    )

    # ── Step 5: extract contiguous clean segments ─────────────────────────────
    logger.info("Extracting clean segments (min_clean_hours=%.1f h)", min_clean_hours)
    clean_segments = extract_clean_segments(
        sp,
        extended_mask,
        min_clean_hours=min_clean_hours,
    )
    logger.info("Extracted %d clean segment(s)", len(clean_segments))

    # ── Step 6: window each clean segment independently ───────────────────────
    if window_kwargs:
        window = window_kwargs.get("window", "tukey")
        alpha = window_kwargs.get("alpha", 0.025)
        for name, seg in clean_segments.items():
            seg.apply_window(window=window, alpha=alpha)
            logger.info(
                "  %s: windowed (%s alpha=%.4g), N=%d (%.2f h)",
                name,
                window,
                alpha,
                seg.N,
                seg.T / 3600,
            )

    # ── Step 7: optional HDF5 write ───────────────────────────────────────────
    if output_path is not None:
        write(
            output_path,
            clean_segments,
            raw_data=data,
            filter_kwargs=filter_kwargs,
            downsample_kwargs=downsample_kwargs,
            trim_kwargs=trim_kwargs,
            window_kwargs=window_kwargs,
        )
        logger.info("Written to %s", output_path)

    return clean_segments, extended_mask


# ── CLI entry point ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Load a MojitoL1 HDF5 file and run the full gap-aware processing pipeline. "
            "The smoothed gap mask must be supplied as a .npy file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=pathlib.Path, help="Path to the MojitoL1 .h5 file")
    p.add_argument(
        "--mask-path",
        type=pathlib.Path,
        required=True,
        metavar="NPY",
        help="Path to a .npy file containing the smoothed gap mask",
    )
    p.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output .h5 path for clean segments (optional)",
    )
    p.add_argument(
        "--load-days",
        type=float,
        default=None,
        metavar="DAYS",
        help="Number of days to load from the file (default: all)",
    )
    p.add_argument(
        "--channels",
        nargs="+",
        default=None,
        metavar="CH",
        help="TDI channels to process (default: X Y Z)",
    )
    p.add_argument(
        "--target-fs",
        type=float,
        default=0.2,
        metavar="HZ",
        help="Target sampling frequency in Hz",
    )
    p.add_argument(
        "--highpass",
        type=float,
        default=5e-6,
        metavar="HZ",
        help="High-pass cutoff frequency in Hz",
    )
    p.add_argument(
        "--lowpass",
        type=float,
        default=None,
        metavar="HZ",
        help="Low-pass cutoff in Hz (default: 0.8 * target_fs)",
    )
    p.add_argument("--filter-order", type=int, default=2)
    p.add_argument(
        "--trim-fraction",
        type=float,
        default=0.02,
        metavar="FRAC",
        help="Fraction of data to trim from each end after filtering",
    )
    p.add_argument(
        "--window",
        type=str,
        default="tukey",
        choices=["tukey", "hann", "hamming", "blackman", "blackmanharris", "planck"],
        help="Window applied to each clean segment",
    )
    p.add_argument(
        "--window-alpha",
        type=float,
        default=0.05,
        metavar="ALPHA",
        help="Taper fraction for tukey/planck windows",
    )
    p.add_argument(
        "--contamination-threshold",
        type=float,
        default=1e-4,
        metavar="EPS",
        help="Filter leakage amplitude above which a sample is flagged as contaminated",
    )
    p.add_argument(
        "--min-clean-hours",
        type=float,
        default=12.0,
        metavar="HOURS",
        help="Minimum clean segment duration in hours",
    )
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    args = _build_parser().parse_args()

    smoothed_mask = np.load(args.mask_path)
    lowpass = args.lowpass if args.lowpass is not None else 0.8 * args.target_fs

    clean_segments, extended_mask = gapspipeline(
        args.input,
        smoothed_mask,
        channels=args.channels,
        load_days=args.load_days,
        filter_kwargs={
            "highpass_cutoff": args.highpass,
            "lowpass_cutoff": lowpass,
            "order": args.filter_order,
        },
        downsample_kwargs={"target_fs": args.target_fs},
        trim_kwargs={"fraction": args.trim_fraction},
        window_kwargs={"window": args.window, "alpha": args.window_alpha},
        contamination_threshold=args.contamination_threshold,
        min_clean_hours=args.min_clean_hours,
        output_path=args.output,
    )

    print(f"\nExtracted {len(clean_segments)} clean segment(s):")
    for name, sp in clean_segments.items():
        print(f"  {name}: {sp}")
    print(f"\nExtended mask: {extended_mask.mean():.4%} clean")
