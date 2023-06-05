# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Decoders.

A decoder receives the syndrome, which is given by measurement outcomes of all
stabilizer generators, and returns a correction for suspected errors. In addition,
our "union find" decoder can also process heralding information on erased or lost
qubits.

There are two different interfaces related to decoders:

* For use in QEC simulations, the high-level interface provided by
  :class:`.decoderbase.DecoderInterface` is most suitable.

  The high-level interface is used by instantiating decoders from
  :mod:`plaquette.decoders.interfaces`.
  The classes can also be imported directly from :mod:`plaquette.decoders`. Example::

    from plaquette import decoders

    my_decoder = decoders.UnionFindDecoder.from_code(code, errordata, weighted=True)
    correction = my_decoder.decode(qubit_erased, syndrome)

  For more information, see :meth:`.decoderbase.DecoderInterface.from_code` and
  :meth:`.decoderbase.DecoderInterface.decode`.

* For low-level decoder tests, the low-level interface provided by
  :class:`.decoderbase.DecoderBackendInterface` can also be used.

  The low-level interface is used by instantiating subclasses of that class.
  They can be found in individual decoder submodules (e.g. :mod:`.unionfind_interfaces`,
  :mod:`.matching`).


The main difference between the high- and low-level interface is in the syndrome
graph on which the interfaces operate. The high-level interface operates on a
syndrome graph composed of ``n_rounds`` of identical repetitions of the same set of
stabilizer generators. The low-level interface operates on an arbitrary graph. In
the current implementation, the low-level interface always operates on one connected
component of the total syndrome graph.

In particular, the high- and low-level interfaces to decoders have the following
differences:

* Syndrome:

  In the high-level interface, the syndrome is an array of shape
  ``(n_rounds, n_stabgens)`` and it contains information on all measurement rounds
  and stabilizer generators.

  In the low-level interface, the syndrome is a flat array of shape ``(n_vertices,)``
  which contains information all vertices in an arbitrary graph.

* Erasure:

  In the high-level interface, the erasure information is an array of shape
  ``(n_rounds, n_qubits)`` and it specifies whether a given data qubit was erased in
  a particular round. Typically, both X and Z errors can act on a single qubit such
  that a single qubit corresponds to two edges in the syndrome graph.

  In the low-level interface, the erasure information is a flat array of shape
  ``(n_edges,)``. It specifies which edges in an arbitrary graph were erased.

* Decoding result:

  In the high-level interface, the decoding result is converted to a Pauli frame
  update.

  In the low-level interface, the decoding result is returned as selection of edges
  of an arbitrary graph.


Imports from submodules
-----------------------

This package imports the following classes from submodules:

* :class:`decoderbase.DecoderBackendInterface`
* :class:`decoderbase.DecoderInterface`
* :class:`interfaces.PyMatchingDecoder`
* :class:`interfaces.UnionFindDecoder`
* :class:`interfaces.UnionFindNoWeights`
* :class:`interfaces.FusionBlossomDecoder`
"""

from plaquette.decoders.decoderbase import DecoderBackendInterface, DecoderInterface
from plaquette.decoders.interfaces import (
    FusionBlossomDecoder,
    PyMatchingDecoder,
    UnionFindDecoder,
    UnionFindNoWeights,
)
