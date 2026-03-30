Installation
============

Requirements
------------

- Python ≥ 3.12
- `mojito <https://gitlab.esa.int/lisa-commons/mojito>`_ ≥ 0.4.0
- ``numpy >= 2.0``
- ``scipy >= 1.10``
- ``matplotlib >= 3.5``

.. note::

   Please remember to install `mojito` from source by locating the `mojito` root directory and running

.. code-block:: bash

   uv pip install .


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
