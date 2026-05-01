"""
Write processed Mojito data to HDF5.
"""

import json
import logging
import pathlib
from typing import Dict, Optional

import h5py
import numpy as np

from ..process.sigprocess import SignalProcessor

logger = logging.getLogger(__name__)

__all__ = ["write"]


def write(
    output_path: str | pathlib.Path,
    segments: Dict[str, "SignalProcessor"],
    raw_data: Optional[dict] = None,
    *,
    segment_ids: Optional[list] = None,
    filter_kwargs: Optional[dict] = None,
    downsample_kwargs: Optional[dict] = None,
    trim_kwargs: Optional[dict] = None,
    truncate_kwargs: Optional[dict] = None,
    window_kwargs: Optional[dict] = None,
) -> None:
    """
    Write processed segments and raw auxiliary data to an HDF5 file.

    File layout
    -----------
    /processed/
        <segment_name>/          one group per segment
            <channel>            processed time-domain array (e.g. X, Y, Z)
            t                    time array in seconds (TCB)
            attrs: fs, dt, N, T, t0, channels
    /pipeline_params/
        filter/      attrs: highpass_cutoff, lowpass_cutoff, order, ...
        downsample/  attrs: target_fs, kaiser_window
        trim/        attrs: fraction
        truncate/    attrs: days
        window/      attrs: window, alpha
    /raw/                        only written when *raw_data* is provided
        orbits/                  full-span orbit arrays
            sc_position_1        spacecraft 1 positions (n_orbit, 3) [m]
            sc_position_2
            sc_position_3
            sc_velocity_1        spacecraft 1 velocities (n_orbit, 3) [m/s]
            sc_velocity_2
            sc_velocity_3
            times                orbit sample timestamps (TCB) [s]
        noise_estimates/
            xyz                  noise covariance cube (freq, ch, ch)
            aet                  noise covariance cube (freq, ch, ch)
        metadata/
            attrs: laser_frequency, pipeline_names
        <segment_name>/          one group per segment
            ltts/
                <link>           light travel times sliced to segment window
                derivatives/
                    <link>       LTT time-derivatives
                times            LTT sample timestamps for this segment

    LTT data are sliced to each segment's time window using the segment's
    ``t0`` and time array.  Segments without a valid ``t0`` (stored as NaN)
    are skipped for per-segment raw data.

    Parameters
    ----------
    output_path : str or Path
        Destination file path. Created (or overwritten) by this function.
    segments : dict of SignalProcessor
        Output of :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.
    raw_data : dict, optional
        Raw data dict returned by :func:`~MojitoProcessor.io.read.load_file`.
        When provided, noise estimates and metadata are written under ``/raw/``,
        and orbit/LTT data are written per-segment under ``/raw/<segment_name>/``.
    segment_ids : list of int, optional
        Indices of segments to write, e.g. ``[0, 17]`` writes ``segment0``
        and ``segment17``. ``None`` (default) writes all segments.
    filter_kwargs, downsample_kwargs, trim_kwargs, truncate_kwargs, \
window_kwargs : dict, optional
        Pipeline parameter dicts, stored verbatim under ``/pipeline_params/``.
    """
    output_path = pathlib.Path(output_path)

    sp_segments = {k: v for k, v in segments.items() if isinstance(v, SignalProcessor)}
    if segment_ids is not None:
        ids = set(int(i) for i in segment_ids)
        sp_segments = {
            k: v
            for k, v in sp_segments.items()
            if k.startswith("segment") and int(k[len("segment") :]) in ids
        }

    with h5py.File(output_path, "w") as f:
        # ── Pipeline parameters ───────────────────────────────────────────────
        pp = f.create_group("pipeline_params")
        _write_attrs(pp.create_group("filter"), filter_kwargs or {})
        _write_attrs(pp.create_group("downsample"), downsample_kwargs or {})
        _write_attrs(pp.create_group("trim"), trim_kwargs or {})
        _write_attrs(pp.create_group("truncate"), truncate_kwargs or {})
        _write_attrs(pp.create_group("window"), window_kwargs or {})

        # ── Processed segments ────────────────────────────────────────────────
        segs_grp = f.create_group("processed")
        for seg_name, sp in sp_segments.items():
            seg = segs_grp.create_group(seg_name)
            for ch in sp.channels:
                seg.create_dataset(ch, data=sp._data[ch], compression="gzip")
            seg.create_dataset("t", data=sp.t, compression="gzip")
            seg.attrs["fs"] = sp.fs
            seg.attrs["dt"] = sp.dt
            seg.attrs["N"] = sp.N
            seg.attrs["T"] = sp.T
            seg.attrs["t0"] = sp.t0 if sp.t0 is not None else np.nan
            seg.attrs["channels"] = json.dumps(sp.channels)

        if raw_data is None:
            logger.info("Wrote %d segment(s) to %s", len(sp_segments), output_path)
            return

        # ── Raw / auxiliary data ──────────────────────────────────────────────
        raw = f.create_group("raw")

        # Full-span orbit arrays (written once at top level)
        if "orbits" in raw_data and "orbit_times" in raw_data:
            orb_grp = raw.create_group("orbits")
            orb = raw_data["orbits"]  # (n, 3, 3), axis 1 = spacecraft
            for i in range(3):
                orb_grp.create_dataset(
                    f"sc_position_{i + 1}", data=orb[:, i, :], compression="gzip"
                )
            orb_grp.create_dataset(
                "times", data=raw_data["orbit_times"], compression="gzip"
            )
            if "velocities" in raw_data:
                vel = raw_data["velocities"]
                for i in range(3):
                    orb_grp.create_dataset(
                        f"sc_velocity_{i + 1}", data=vel[:, i, :], compression="gzip"
                    )

        # Noise estimates (not time-varying, written once at top level)
        if "noise_estimates" in raw_data:
            ne = raw.create_group("noise_estimates")
            for key, arr in raw_data["noise_estimates"].items():
                ne.create_dataset(key, data=arr, compression="gzip")

        # Metadata (written once at top level)
        if "metadata" in raw_data:
            meta = raw.create_group("metadata")
            md = raw_data["metadata"]
            if "laser_frequency" in md:
                meta.attrs["laser_frequency"] = float(md["laser_frequency"])
            if "pipeline_names" in md:
                meta.attrs["pipeline_names"] = json.dumps(list(md["pipeline_names"]))

        # Per-segment LTT data (sliced to each segment's time window)
        for seg_name, sp in sp_segments.items():
            if sp.t0 is None:
                continue  # can't slice by absolute time without t0
            t_start, t_end = float(sp.t[0]), float(sp.t[-1])
            seg_raw = raw.create_group(seg_name)

            if "ltts" in raw_data and "ltt_times" in raw_data:
                sl = _time_slice(raw_data["ltt_times"], t_start, t_end)
                ltt_t = raw_data["ltt_times"][sl]
                if len(ltt_t) > 0:
                    ltt = seg_raw.create_group("ltts")
                    for link, arr in raw_data["ltts"].items():
                        ltt.create_dataset(link, data=arr[sl], compression="gzip")
                    ltt.create_dataset("times", data=ltt_t, compression="gzip")
                    if "ltt_derivatives" in raw_data:
                        deriv = ltt.create_group("derivatives")
                        for link, arr in raw_data["ltt_derivatives"].items():
                            deriv.create_dataset(link, data=arr[sl], compression="gzip")

    logger.info("Wrote %d segment(s) + raw data to %s", len(sp_segments), output_path)


def _time_slice(times: np.ndarray, t_start: float, t_end: float) -> slice:
    """Return a slice covering [t_start, t_end] within a sorted *times* array."""
    i0 = int(np.searchsorted(times, t_start, side="left"))
    i1 = int(np.searchsorted(times, t_end, side="right"))
    return slice(i0, i1)


def _write_attrs(group: h5py.Group, kwargs: dict) -> None:
    """Serialize a kwargs dict as HDF5 attributes on *group*."""
    for k, v in kwargs.items():
        if v is None:
            group.attrs[k] = "None"
        elif isinstance(v, (bool, int, float, str)):
            group.attrs[k] = v
        else:
            group.attrs[k] = json.dumps(v)
