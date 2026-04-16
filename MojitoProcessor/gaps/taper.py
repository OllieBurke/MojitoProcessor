"""
Gap-mask tapering utilities.

:func:`taper_mask` applies two independent half-cosine tapers to a binary gap
mask:

1. **Gap-edge tapers** — a rising ramp at every ``0 → 1`` transition and a
   falling ramp at every ``1 → 0`` transition so that the data envelope
   approaches zero smoothly at each gap boundary.

2. **Dataset-endpoint tapers** — independent ramps applied to the very first
   and last samples of the array so the endpoints are brought to zero
   regardless of whether a gap is present there.

Both taper lengths are independently tunable in hours.
"""

import numpy as np

__all__ = ["taper_mask"]


def taper_mask(
    extended_mask: np.ndarray,
    sp,
    *,
    taper_hours: float = 12.0,
    edge_taper_hours: float = 12.0,
) -> np.ndarray:
    """Apply half-cosine tapers to gap edges and dataset endpoints.

    The output is a copy of *extended_mask* whose transitions between clean
    (``1``) and excluded (``0``) regions are smoothed by a raised-cosine ramp
    of length *taper_hours*.  Additionally, the first and last
    *edge_taper_hours* of the array are multiplied by an independent
    endpoint ramp so the dataset boundaries reach zero.

    Parameters
    ----------
    extended_mask : ndarray
        Binary float array (``0.0`` or ``1.0``) as returned by
        :func:`~MojitoProcessor.gaps.extend.compute_extended_mask`.
        Length must equal ``sp.N``.
    sp : SignalProcessor
        Processed segment used to read ``fs`` and ``N``.
    taper_hours : float, optional
        Length of the half-cosine ramp applied to each gap edge (hours).
        Default: ``12.0``.
    edge_taper_hours : float, optional
        Length of the half-cosine ramp applied to the very start and end of
        the dataset (hours).  Default: ``12.0``.

    Returns
    -------
    mask_tapered : ndarray
        Float array of length ``sp.N`` with all tapers applied.

    Examples
    --------
    >>> gap_mask = taper_mask(
    ...     extended_mask, sp_0, taper_hours=12.0, edge_taper_hours=12.0
    ... )
    >>> sp_masked = apply_mask_to_processor(sp_0, gap_mask)
    """
    mask = np.asarray(extended_mask, dtype=float).copy()
    N = len(mask)

    # ── Gap-edge tapers ───────────────────────────────────────────────────────
    taper_n = int(taper_hours * 3600.0 * sp.fs)
    if taper_n > 0:
        # Raised-cosine ramp: 0 → 1 over taper_n samples
        half_cosine = 0.5 * (1.0 - np.cos(np.pi * np.arange(taper_n) / taper_n))

        # Pad with boundary-matching sentinels so that dataset edges are NOT
        # treated as gap transitions (that is the job of edge_taper_hours).
        mask_int = mask.astype(int)
        padded = np.concatenate([[mask_int[0]], mask_int, [mask_int[-1]]])
        diff = np.diff(padded)

        for r in np.where(diff == 1)[0]:  # rising edge: gap → data
            end = min(r + taper_n, N)
            n = end - r
            mask[r:end] *= half_cosine[:n]

        for f in np.where(diff == -1)[0]:  # falling edge: data → gap
            start = max(f - taper_n, 0)
            n = f - start
            mask[start:f] *= half_cosine[:n][::-1]

    # ── Dataset endpoint tapers ───────────────────────────────────────────────
    edge_n = int(edge_taper_hours * 3600.0 * sp.fs)
    if edge_n > 0:
        edge_ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(edge_n) / edge_n))
        mask[:edge_n] *= edge_ramp
        mask[-edge_n:] *= edge_ramp[::-1]

    return mask
