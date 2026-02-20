Installation
============

Requirements
------------

- Python ≥ 3.10
- ``h5py >= 3.0``
- ``numpy >= 2.0``
- ``scipy >= 1.10``
- ``matplotlib >= 3.5``

Install from PyPI
-----------------

.. code-block:: bash

   pip install mojito-processor

Install with uv (recommended)
------------------------------

.. code-block:: bash

   uv pip install mojito-processor

Development install
-------------------

.. code-block:: bash

   git clone https://github.com/OllieBurke/MojitoProcessor.git
   cd MojitoProcessor

   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install in editable mode with dev dependencies
   uv pip install -e .

   # Install pre-commit hooks
   uv run pre-commit install

Optional: Jupyter notebooks
----------------------------

.. code-block:: bash

   pip install "mojito-processor[notebooks]"
