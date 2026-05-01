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
