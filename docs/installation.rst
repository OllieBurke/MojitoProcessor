Installation
============

From PyPI
---------

.. code-block:: bash

   pip install mojito-processor

Or with `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   uv pip install mojito-processor

Requirements
------------

- Python ≥ 3.12
- `mojito <https://pypi.org/project/mojito/>`_ ≥ 0.5.0
- ``numpy >= 2.0``
- ``scipy >= 1.10``
- ``h5py``

All dependencies are installed automatically when you install ``mojito-processor``
via pip or uv.

``matplotlib`` is required only for the example notebooks and is not a
core dependency.

Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/OllieBurke/MojitoProcessor.git
   cd MojitoProcessor

   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install the package and all dependency groups
   uv sync --all-groups

   # Install pre-commit hooks
   uv run pre-commit install

   # Run pre-commit on all files (optional)
   uv run pre-commit run --all-files
