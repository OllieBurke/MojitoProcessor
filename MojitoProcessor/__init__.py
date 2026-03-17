"""MojitoProcessor - Signal processing utilities for LISA Mojito L1 data"""

from .__version__ import __version__
from .SigProcessing import SignalProcessor, process_pipeline

__all__ = [
    "__version__",
    "SignalProcessor",
    "process_pipeline",
]
