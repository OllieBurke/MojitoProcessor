Installation
============

Requirements
------------

- Python ≥ 3.12
- `mojito <https://gitlab.esa.int/lisa-commons/mojito>`_ ≥ 0.2.3
- ``numpy >= 2.0``
- ``scipy >= 1.10``
- ``matplotlib >= 3.5``

.. note::

   ``mojito`` is distributed via the ESA GitLab package registry, not PyPI.
   You must configure this extra index before installing ``mojito-processor``
   (see below).

ESA GitLab Index Setup
----------------------

With `uv <https://docs.astral.sh/uv/>`_, add the following to your
``pyproject.toml`` or ``uv.toml``:

.. code-block:: toml

   [[tool.uv.index]]
   name = "gitlab-esa-commons"
   url = "https://gitlab.esa.int/api/v4/groups/29349/-/packages/pypi/simple"

   [tool.uv.sources]
   mojito = { index = "gitlab-esa-commons" }

Or pass the extra index directly on the command line:

.. code-block:: bash

   uv pip install mojito-processor \
       --extra-index-url https://gitlab.esa.int/api/v4/groups/29349/-/packages/pypi/simple

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
