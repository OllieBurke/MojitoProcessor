"""MojitoProcessor - LISA Mojito L1 data loading and analysis utilities"""

from .__version__ import __version__
from .mojito_loader import (
    MojitoData,
    OrbitData,
    investigate_mojito_file_structure,
    load_mojito_l1,
    load_orbits,
    truncate_mojito_data,
    truncate_orbit_data,
)
from .SigProcessing import (
    SignalProcessor,
    apply_window,
    bandpass_filter,
    downsample,
    process_pipeline,
)

__all__ = [
    "__version__",
    "load_mojito_l1",
    "load_orbits",
    "truncate_mojito_data",
    "truncate_orbit_data",
    "investigate_mojito_file_structure",
    "MojitoData",
    "OrbitData",
    "SignalProcessor",
    "bandpass_filter",
    "apply_window",
    "downsample",
    "process_pipeline",
]
