"""Sphinx configuration for MojitoProcessor documentation."""

import sys
from pathlib import Path

# Add the repo root to sys.path so autodoc can import the package.
# Compute it relative to this conf.py file, not the current working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# -- Project information -----------------------------------------------------
project = "MojitoProcessor"
author = "Ollie Burke"
release = "0.2.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"

# -- nbsphinx options --------------------------------------------------------
nbsphinx_execute = "never"
