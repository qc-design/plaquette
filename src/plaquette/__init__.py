# Copyright 2023, It'sQ GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""A featured and accessible quantum error-correction simulation library.

``plaquette`` is structured to make it effortless to start running first error
correction simulations while also providing all the functionality needed for deeper
investigation of error correction.

For a quick walk-through of the most important parts of the API, see
:doc:`/quickstart`.

The frontend comprises :mod:`plaquette.codes` and :mod:`plaquette.circuit`, which allow
designing new error correction codes or specifying built-in ones,
and :mod:`plaquette.errors` which provides a fine-grained interface for specifying
hardware imperfections. The quantum device is mimicked by :mod:`plaquette.device`
and the classical control for finding and correcting errors is provided by
:mod:`plaquette.decoders`, which internally uses :mod:`plaquette.syngraph`. A
set of helper tools for visualising codes and simulation results can be found
in :mod:`plaquette.visualizer`. Finally, the underlying
"tableau-representation" is implemented by the :mod:`plaquette.pauli`.
"""

import sys

import numpy as np

from plaquette.device import Device  # noqa: F401

__version__ = "0.0.1a2"

#: Random number generator (specifically :func:`numpy.random.Generator.default_rng`)
#:
#: To use your own, or to make your simulations deterministic (by using a fixed
#: seed), you can replace this module variable **before** calling any other function
#: or class in the package.
rng = np.random.default_rng()

# Avoid surprises
assert sys.version_info >= (3, 10), "Please upgrade Python to at least 3.10"
