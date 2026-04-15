"""
Read and process a MojitoL1 HDF5 file in one call.

Can be run as a script::

    python -m MojitoProcessor.pipelines.read_and_process path/to/data.h5 \\
        --output processed.h5 \\
        --target-fs 0.2 \\
        --segment-days 7.0
"""

import argparse
import logging
import pathlib
from typing import Dict, List, Optional

from ..io.read import load_file
from ..io.write import write
from ..process.sigprocess import SignalProcessor, process_pipeline

__all__ = ["read_and_process"]

logger = logging.getLogger(__name__)


def read_and_process(
    path: str | pathlib.Path,
    channels: Optional[List[str]] = None,
    *,
    load_days: Optional[float] = None,
    filter_kwargs: Optional[dict] = None,
    downsample_kwargs: Optional[dict] = None,
    trim_kwargs: Optional[dict] = None,
    truncate_kwargs: Optional[dict] = None,
    window_kwargs: Optional[dict] = None,
    output_path: Optional[str | pathlib.Path] = None,
) -> Dict[str, SignalProcessor]:
    """
    Load a MojitoL1 file and run the full processing pipeline in one call.

    This is a thin wrapper around :func:`~MojitoProcessor.io.read.load_file`
    and :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.  When
    *output_path* is provided the result is also written to an HDF5 file via
    :func:`~MojitoProcessor.io.write.write`.

    Parameters
    ----------
    path : str or Path
        Path to the MojitoL1 ``.h5`` input file.
    channels : list of str, optional
        TDI channels to process. Default ``['X', 'Y', 'Z']``.
    load_days : float, optional
        Number of days to load from the file (lazy slicing).
        ``None`` loads the full dataset.
    filter_kwargs : dict, optional
        Passed to :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.
    downsample_kwargs : dict, optional
        Passed to :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.
    trim_kwargs : dict, optional
        Passed to :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.
    truncate_kwargs : dict, optional
        Passed to :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.
    window_kwargs : dict, optional
        Passed to :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.
    output_path : str or Path, optional
        If given, write processed segments and raw auxiliary data to this
        HDF5 file.

    Returns
    -------
    segments : dict of SignalProcessor
        Processed segments keyed by ``'segment0'``, ``'segment1'``, etc.
    """
    logger.info("Loading %s", path)
    data = load_file(path, load_days=load_days)

    logger.info("Running processing pipeline")
    segments = process_pipeline(
        data,
        channels=channels,
        filter_kwargs=filter_kwargs,
        downsample_kwargs=downsample_kwargs,
        trim_kwargs=trim_kwargs,
        truncate_kwargs=truncate_kwargs,
        window_kwargs=window_kwargs,
    )

    if output_path is not None:
        write(
            output_path,
            segments,
            raw_data=data,
            filter_kwargs=filter_kwargs,
            downsample_kwargs=downsample_kwargs,
            trim_kwargs=trim_kwargs,
            truncate_kwargs=truncate_kwargs,
            window_kwargs=window_kwargs,
        )

    return segments


# ── CLI entry point ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Load and process a MojitoL1 HDF5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=pathlib.Path, help="Path to the MojitoL1 .h5 file")
    p.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output .h5 path for processed data (optional)",
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
        "--segment-days",
        type=float,
        default=7.0,
        metavar="DAYS",
        help="Segment length in days",
    )
    p.add_argument(
        "--window",
        type=str,
        default="tukey",
        choices=["tukey", "hann", "hamming", "blackman", "blackmanharris", "planck"],
        help="Window function to apply to each segment",
    )
    p.add_argument(
        "--window-alpha",
        type=float,
        default=0.0125,
        metavar="ALPHA",
        help="Taper fraction for tukey/planck windows",
    )
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    args = _build_parser().parse_args()

    lowpass = args.lowpass if args.lowpass is not None else 0.8 * args.target_fs

    segments = read_and_process(
        args.input,
        channels=args.channels,
        load_days=args.load_days,
        filter_kwargs={
            "highpass_cutoff": args.highpass,
            "lowpass_cutoff": lowpass,
            "order": args.filter_order,
        },
        downsample_kwargs={"target_fs": args.target_fs},
        trim_kwargs={"fraction": args.trim_fraction},
        truncate_kwargs={"days": args.segment_days},
        window_kwargs={"window": args.window, "alpha": args.window_alpha},
        output_path=args.output,
    )

    print(f"\nProcessed {len(segments)} segment(s):")
    for name, sp in segments.items():
        print(f"  {name}: {sp}")
