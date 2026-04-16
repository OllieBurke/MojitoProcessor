MojitoProcessor Documentation
==============================

Goal of Package
---------------

The goal of this package is to provide a simple, modular, and well-documented
set of tools for processing LISA Mojito L1 data. The package applies a signal
processing pipeline (filtering, downsampling, trimming, windowing) to data
loaded via the `mojito <https://gitlab.esa.int/lisa-commons/mojito>`_ package.
The design emphasizes ease of use and flexibility, allowing users to customize
the processing steps as needed for their specific analysis tasks.

The ``gaps`` subpackage extends the pipeline to **gapped data**: it applies raw
masks before filtering, computes extended masks that account for Butterworth
filter leakage, and extracts contiguous clean segments suitable for standard
Whittle-likelihood spectral analysis.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   api
   examples
   citing
