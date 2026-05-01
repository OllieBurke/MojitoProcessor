"""MojitoProcessor - Signal processing utilities for LISA Mojito L1 data"""

from . import gaps, io
from .__version__ import __version__
from .io.read import load_file, load_processed, report_info_from_loaded_file
from .io.write import write
from .pipelines import read_and_process
from .process.sigprocess import SignalProcessor, process_pipeline

__all__ = [
    "__version__",
    "SignalProcessor",
    "process_pipeline",
    "load_file",
    "load_processed",
    "report_info_from_loaded_file",
    "write",
    "gaps",
    "io",
    "read_and_process",
]
