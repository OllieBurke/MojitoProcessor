"""
Load data for the Mojito Processor.
"""

import pathlib

import mojito


def load_file(
    paths: str | pathlib.Path | list[str | pathlib.Path],
    *,
    load_days: float | None = None,
):
    """Load a file using the mojito library.

    Args:
        paths: A single file path or a list of file paths to load.
        load_days: Optional number of days from the beginning to load (lazy slicing). If None, loads the full dataset.
    """
    with mojito.MojitoL1File(paths) as f:
        tdi_sampling = f.tdis.time_sampling
        ltt_sampling = f.ltts.time_sampling
        orbit_sampling = f.orbits.time_sampling

        # Consistent sample counts across all data streams
        n_tdi = (
            int(load_days * 86400 * tdi_sampling.fs) if load_days else tdi_sampling.size
        )
        n_ltt = (
            int(load_days * 86400 * ltt_sampling.fs) if load_days else ltt_sampling.size
        )
        n_orbit = (
            int(load_days * 86400 * orbit_sampling.fs)
            if load_days
            else orbit_sampling.size
        )

        data = {
            # ── TDI observables ──────────────────────────────────────────────────
            "tdis": {
                "X": f.tdis.x2[:n_tdi],
                "Y": f.tdis.y2[:n_tdi],
                "Z": f.tdis.z2[:n_tdi],
                "A": f.tdis.a2[:n_tdi],
                "E": f.tdis.e2[:n_tdi],
                "T": f.tdis.t2[:n_tdi],
            },
            "fs": tdi_sampling.fs,
            "dt": tdi_sampling.dt,
            "t_tdi": tdi_sampling.t(slice(None, n_tdi)),
            # ── Light travel times ───────────────────────────────────────────────
            "ltts": {
                "12": f.ltts.ltt_12[:n_ltt],
                "13": f.ltts.ltt_13[:n_ltt],
                "21": f.ltts.ltt_21[:n_ltt],
                "23": f.ltts.ltt_23[:n_ltt],
                "31": f.ltts.ltt_31[:n_ltt],
                "32": f.ltts.ltt_32[:n_ltt],
            },
            "ltt_derivatives": {
                "12": f.ltts.ltt_derivative_12[:n_ltt],
                "13": f.ltts.ltt_derivative_13[:n_ltt],
                "21": f.ltts.ltt_derivative_21[:n_ltt],
                "23": f.ltts.ltt_derivative_23[:n_ltt],
                "31": f.ltts.ltt_derivative_31[:n_ltt],
                "32": f.ltts.ltt_derivative_32[:n_ltt],
            },
            "ltt_times": ltt_sampling.t(slice(None, n_ltt)),
            # ── Spacecraft orbits ────────────────────────────────────────────────
            "orbits": f.orbits.positions[:n_orbit],  # (n_orbit, 3, 3)
            "velocities": f.orbits.velocities[:n_orbit],  # (n_orbit, 3, 3)
            "orbit_times": orbit_sampling.t(slice(None, n_orbit)),
            # ── Noise estimates (frequency-domain, not truncated) ────────────────
            "noise_estimates": {
                "xyz": f.noise_estimates.xyz[:],
                "aet": f.noise_estimates.aet[:],
            },
            # ── Metadata ─────────────────────────────────────────────────────────
            "metadata": {
                "laser_frequency": f.laser_frequency,
                "pipeline_names": f.pipeline_names,
            },
        }
    return data
