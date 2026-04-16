Installation
============

Requirements
------------

- Python ≥ 3.12
- `mojito <https://gitlab.esa.int/lisa-commons/mojito>`_ ≥ 0.5.0
- ``numpy >= 2.0``
- ``scipy >= 1.10``
- ``h5py``

.. note::

   ``mojito`` must be installed from source. Locate the ``mojito`` root
   directory and run:

   .. code-block:: bash

      uv pip install .

``matplotlib`` is required only for the example notebooks and is not a
core dependency.

Gap-Handling Extras
-------------------

The ``gaps`` dependency group adds support for processing gapped LISA data.
It pulls in ``lisaglitch`` and ``lisa-gap`` (the latter from Test PyPI):

.. code-block:: bash

   uv sync --group gaps

.. note::

   ``torch`` (a transitive dependency of ``lisaglitch``) is excluded by
   default in ``pyproject.toml`` via ``tool.uv.no-install-package``.
   It is not needed by MojitoProcessor.

Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/OllieBurke/MojitoProcessor.git
   cd MojitoProcessor

   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install core package only
   uv sync

   # Install with gap-handling support
   uv sync --group gaps

   # Install all dependency groups (dev, docs, notebooks, gaps)
   uv sync --all-groups

   # Install pre-commit hooks
   uv run pre-commit install

   # Run pre-commit on all files (optional)
   uv run pre-commit run --all-files
