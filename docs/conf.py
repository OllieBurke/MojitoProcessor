"""Sphinx configuration for MojitoProcessor documentation."""

import os
import sys

# Add the repo root to sys.path so autodoc can import the package
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "MojitoProcessor"
author = "Ollie Burke"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"

# -- nbsphinx options --------------------------------------------------------
nbsphinx_execute = "never"
