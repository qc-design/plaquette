# Copyright 2023, It'sQ GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parent.parent / "src"))


# -- Project information -----------------------------------------------------

project = "plaquette"
copyright = "2022, It'sQ GmbH"
author = "It'sQ GmbH"
url = "https://docs.plaquette.design"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
]

intersphinx_mapping = {
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

autodoc_member_order = "groupwise"
show_authors = True

# -- BiBTeX configuration --------------------------------------

import pybtex
from pybtex.style import template as bibtpl
from pybtex.style.formatting.alpha import Style


class PlaquetteStyle(Style):
    """Custom bibliography style for pybtex"""

    def format_web_refs(self, e):
        # based on urlbst output.web.refs
        return bibtpl.sentence[
            bibtpl.optional[
                self.format_url(e),
                # We do not want to show "visited on (date of adding entry to
                # bibliography)".
                # bibtpl.optional[" (visited on ", field("urldate"), ")"],
            ],
            bibtpl.optional[self.format_eprint(e)],
            bibtpl.optional[self.format_pubmed(e)],
            bibtpl.optional[self.format_doi(e)],
        ]


pybtex.plugin.register_plugin("pybtex.style.formatting", "plaquette", PlaquetteStyle)

bibtex_default_style = "plaquette"
bibtex_bibfiles = ["references.bib"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "furo"
html_theme_options = {
    "source_repository": "https://github.com/qc-design/plaquette/",
    "source_branch": "main",
    "source_directory": "docs/",
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_logo = "logo.png"

# Pin MathJax version to enable SRI check
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"
mathjax_options = {
    "crossorigin": "anonymous",
    "integrity": "MASABpB4tYktI2Oitl4t+78w/lyA+D7b/s9GEP0JOGI=",
    "async": "async",
}

# See https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-nitpicky
nitpicky = True
nitpick_ignore = []
nitpick_ignore_regex = [
    ("py:class", r"(enum|galois|matplotlib|numbers|numpy|pandas|pymatching|stim)\..*"),
    # The following are due to bugs in the automatic generation of reference docs.
    ("py:obj", r"plaquette\.codes\.latticebase\.PosType\..*"),
    ("py:obj", r"plaquette\.lib\.clifford\.PauliListListDesc\..*"),
]

# Show inheritance and private and special members.
autodoc_default_options = {
    # Note: To disable an option, just comment the line. Changing `None` to something
    # else will not have an effect.
    "members": True,
    "undoc-members": None,
    # "private-members": None,  # _private
    "special-members": True,  # __special__
    "show-inheritance": None,
    # The following attributes and methods are typically built-ins which are not
    # overriden. Therefore, we do not document them.
    # If more fine-grained control over showing or hiding methods is necessary,
    # https://stackoverflow.com/a/5599712 may help.
    #
    # Note: __init__ is excluded here because a class docstring
    # should contain ".. automethod:: __init__". This way, __init__ is shows above all
    # other methods.
    # If excluded methods should be shown for a specific class, this can be accomplished
    # by adding e.g. ".. automethod:: __match_args__" to the class docstring.
    "exclude-members": (
        "__abstractmethods__,__annotations__,"
        "__dataclass_fields__,__dataclass_params__,__post_init__,"
        "__dict__,__hash__,__init__,__match_args__,__module__,__slots__,__weakref__"
    ),
}

# copybutton customisations - exclude line-numbers, prompt characters, and outpus
copybutton_exclude = ".linenos, .gp, .go"

# This option does not seem to have any effect.
# autoclass_content = "both"

# We don't use autosummary's autogen feature anymore.
autosummary_generate = False

html_show_sourcelink = True
todo_include_todos = True
