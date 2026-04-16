"""
Gap masking utilities.

Two functions are provided:

* :func:`apply_raw_mask` — multiplies a scalar mask onto every TDI channel in
  a raw data dictionary, returning a deep copy so the original is untouched.

* :func:`apply_mask_to_processor` — returns a **new**
  :class:`~MojitoProcessor.process.sigprocess.SignalProcessor` whose channel
  arrays have been element-wise multiplied by *mask*.  The original
  ``SignalProcessor`` is never modified.
"""

from copy import deepcopy
from typing import Dict

import numpy as np

from ..process.sigprocess import SignalProcessor

__all__ = ["apply_raw_mask", "apply_mask_to_processor"]

# Channels on which the mask is applied when working with a raw data dict.
_TDI_CHANNELS = ("X", "Y", "Z", "A", "E", "T")


def apply_raw_mask(data: dict, mask: np.ndarray) -> dict:
    """Apply a gap mask to all TDI channels in a raw data dictionary.

    Creates a deep copy of *data* before multiplying, so the original dict is
    never modified.  Only channels present in the dict's ``"tdis"`` sub-dict
    are touched; all other entries (ltts, orbits, metadata, …) are preserved
    unchanged.

    Parameters
    ----------
    data : dict
        Raw data dict as returned by
        :func:`~MojitoProcessor.io.read.load_file`.  Must contain a ``"tdis"``
        key whose value is a dict of 1-D numpy arrays.
    mask : ndarray
        1-D float array of length ``data["tdis"]["X"].size`` (or whichever
        channel is present).  Values should be in ``[0, 1]``.

    Returns
    -------
    data_masked : dict
        Deep copy of *data* with every TDI channel multiplied by *mask*.

    Examples
    --------
    >>> data_masked = apply_raw_mask(data, smoothed_mask)
    """
    mask = np.asarray(mask, dtype=float)
    data_masked = deepcopy(data)
    for ch in _TDI_CHANNELS:
        if ch in data_masked["tdis"]:
            data_masked["tdis"][ch] = data_masked["tdis"][ch] * mask
    return data_masked


def apply_mask_to_processor(
    sp: SignalProcessor,
    mask: np.ndarray,
) -> SignalProcessor:
    """Return a new :class:`~MojitoProcessor.process.sigprocess.SignalProcessor` with *mask* applied.

    The original *sp* is **not** mutated.  The returned object has the same
    ``fs`` and ``t0`` as *sp*, with each channel array multiplied element-wise
    by *mask*.

    Parameters
    ----------
    sp : SignalProcessor
        Source processor.  All channels must have the same length as *mask*.
    mask : ndarray
        1-D float array of length ``sp.N``.  Values should be in ``[0, 1]``.

    Returns
    -------
    sp_masked : SignalProcessor
        New ``SignalProcessor`` with masked channel data.

    Raises
    ------
    ValueError
        If ``len(mask) != sp.N``.

    Examples
    --------
    >>> sp_masked = apply_mask_to_processor(sp_0, gap_mask)
    >>> sp_masked_aet = sp_masked.to_aet()
    """
    mask = np.asarray(mask, dtype=float)
    if len(mask) != sp.N:
        raise ValueError(
            f"mask length ({len(mask)}) does not match SignalProcessor length ({sp.N})"
        )
    masked_data: Dict[str, np.ndarray] = {ch: sp._data[ch] * mask for ch in sp.channels}
    return SignalProcessor(masked_data, fs=sp.fs, t0=sp.t0)
