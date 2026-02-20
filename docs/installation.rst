Installation
============

Requirements
------------

- Python ≥ 3.10
- ``h5py >= 3.0``
- ``numpy >= 2.0``
- ``scipy >= 1.10``
- ``matplotlib >= 3.5``

Install
-------

.. note::

   ``mojito-processor`` is currently available on **TestPyPI**. Use the
   commands below to install from there.

.. code-block:: bash

   pip install --index-url https://test.pypi.org/simple/ \
               --extra-index-url https://pypi.org/simple/ \
               mojito-processor

Or with `uv <https://docs.astral.sh/uv/>`_ (recommended):

.. code-block:: bash

   uv pip install --index-url https://test.pypi.org/simple/ \
                  --extra-index-url https://pypi.org/simple/ \
                  mojito-processor

Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/OllieBurke/MojitoProcessor.git
   cd MojitoProcessor

   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install in editable mode with dev dependencies
   uv pip install -e .

   # Install pre-commit hooks
   uv run pre-commit install

   # Run pre-commit on all files (optional)
   uv run pre-commit run --all-files
