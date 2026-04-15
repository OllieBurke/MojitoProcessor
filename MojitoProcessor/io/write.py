"""
Write processed Mojito data to HDF5.
"""

import json
import logging
import pathlib
from typing import TYPE_CHECKING, Dict, Optional

import h5py
import numpy as np

if TYPE_CHECKING:
    from ..process.sigprocess import SignalProcessor

logger = logging.getLogger(__name__)

__all__ = ["write"]


def write(
    output_path: str | pathlib.Path,
    segments: Dict[str, "SignalProcessor"],
    raw_data: Optional[dict] = None,
    *,
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
        noise_estimates/
            xyz                  noise covariance cube (freq, ch, ch)
            aet                  noise covariance cube (freq, ch, ch)
        ltts/
            <link>               light travel times (12, 13, …)
            derivatives/
                <link>           LTT time-derivatives
            times                LTT sample timestamps
        orbits/
            positions            spacecraft positions (n, 3, 3)
            velocities           spacecraft velocities (n, 3, 3)
            times                orbit sample timestamps
        metadata/
            attrs: laser_frequency, pipeline_names

    Parameters
    ----------
    output_path : str or Path
        Destination file path. Created (or overwritten) by this function.
    segments : dict of SignalProcessor
        Output of :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.
    raw_data : dict, optional
        Raw data dict returned by :func:`~MojitoProcessor.io.read.load_file`.
        When provided, noise estimates, LTTs, orbits, and metadata are written
        under ``/raw/``.
    filter_kwargs, downsample_kwargs, trim_kwargs, truncate_kwargs, window_kwargs : dict, optional
        Pipeline parameter dicts, stored verbatim under ``/pipeline_params/``.
    """
    output_path = pathlib.Path(output_path)

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
        for seg_name, sp in segments.items():
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
            logger.info("Wrote %d segment(s) to %s", len(segments), output_path)
            return

        # ── Raw / auxiliary data ──────────────────────────────────────────────
        raw = f.create_group("raw")

        # Noise estimates
        if "noise_estimates" in raw_data:
            ne = raw.create_group("noise_estimates")
            for key, arr in raw_data["noise_estimates"].items():
                ne.create_dataset(key, data=arr, compression="gzip")

        # Light travel times
        if "ltts" in raw_data:
            ltt = raw.create_group("ltts")
            for key, arr in raw_data["ltts"].items():
                ltt.create_dataset(key, data=arr, compression="gzip")
            if "ltt_derivatives" in raw_data:
                deriv = ltt.create_group("derivatives")
                for key, arr in raw_data["ltt_derivatives"].items():
                    deriv.create_dataset(key, data=arr, compression="gzip")
            if "ltt_times" in raw_data:
                ltt.create_dataset(
                    "times", data=raw_data["ltt_times"], compression="gzip"
                )

        # Spacecraft orbits
        if "orbits" in raw_data:
            orb = raw.create_group("orbits")
            orb.create_dataset("positions", data=raw_data["orbits"], compression="gzip")
            if "velocities" in raw_data:
                orb.create_dataset(
                    "velocities", data=raw_data["velocities"], compression="gzip"
                )
            if "orbit_times" in raw_data:
                orb.create_dataset(
                    "times", data=raw_data["orbit_times"], compression="gzip"
                )

        # Metadata
        if "metadata" in raw_data:
            meta = raw.create_group("metadata")
            md = raw_data["metadata"]
            if "laser_frequency" in md:
                meta.attrs["laser_frequency"] = float(md["laser_frequency"])
            if "pipeline_names" in md:
                meta.attrs["pipeline_names"] = json.dumps(list(md["pipeline_names"]))

    logger.info("Wrote %d segment(s) + raw data to %s", len(segments), output_path)


def _write_attrs(group: h5py.Group, kwargs: dict) -> None:
    """Serialize a kwargs dict as HDF5 attributes on *group*."""
    for k, v in kwargs.items():
        if v is None:
            group.attrs[k] = "None"
        elif isinstance(v, (bool, int, float, str)):
            group.attrs[k] = v
        else:
            group.attrs[k] = json.dumps(v)
