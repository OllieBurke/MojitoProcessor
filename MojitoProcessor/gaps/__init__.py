"""
MojitoProcessor gaps — gap-handling utilities for LISA TDI data.

This subpackage provides a non-mutating, modular interface for applying and
investigating the effect of data gaps on the processing pipeline.

The typical workflow is:

1. :func:`apply_raw_mask` — zero-out gapped samples in the raw data dict
   before passing it to :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.
2. :func:`compute_extended_mask` — quantify Butterworth filter leakage and
   produce a conservative binary mask at the processed sampling rate.
3. :func:`taper_mask` — apply smooth half-cosine tapers to gap edges and
   dataset endpoints.
4. :func:`apply_mask_to_processor` — multiply the final mask onto a
   :class:`~MojitoProcessor.process.sigprocess.SignalProcessor`, returning a
   **new** object (original is never modified).
"""

from .extend import compute_extended_mask
from .mask import apply_mask_to_processor, apply_raw_mask
from .segment import extract_clean_segments
from .taper import taper_mask

__all__ = [
    "apply_raw_mask",
    "compute_extended_mask",
    "taper_mask",
    "apply_mask_to_processor",
    "extract_clean_segments",
]
