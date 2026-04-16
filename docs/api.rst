API Reference
=============

Data loading is handled by the `mojito <https://pypi.org/project/mojito/>`_
package. This package provides utilities for loading, processing, and writing
LISA Mojito L1 data.

I/O
---

.. autofunction:: MojitoProcessor.io.read.load_file

.. autofunction:: MojitoProcessor.io.read.load_processed

.. autofunction:: MojitoProcessor.io.write.write

Signal Processing
-----------------

.. autoclass:: MojitoProcessor.process.sigprocess.SignalProcessor
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: MojitoProcessor.process.sigprocess.process_pipeline

Pipelines
---------

.. autofunction:: MojitoProcessor.pipelines.read_and_process.read_and_process

Gap Handling
------------

.. autofunction:: MojitoProcessor.gaps.mask.apply_raw_mask

.. autofunction:: MojitoProcessor.gaps.mask.apply_mask_to_processor

.. autofunction:: MojitoProcessor.gaps.extend.compute_extended_mask

.. autofunction:: MojitoProcessor.gaps.taper.taper_mask

.. autofunction:: MojitoProcessor.gaps.segment.extract_clean_segments
