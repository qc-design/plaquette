# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Interface to MWPM decoder from PyMatching.

MWPM is short for min-weight perfect matching.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pymatching  # type: ignore

from plaquette import syngraph
from plaquette.decoders import decoderbase


class PyMatchingDecoder(decoderbase.DecoderBackendInterface):
    """Min-weight perfect matching decoder provided by PyMatching.

    This is a low-level interface which interacts with an arbitrary graph defined
    in :class:`.SyndromeGraphComponent`. If you are working :mod:`plaquette.codes` and
    :mod:`plaquette.errors`, or if you are using :class:`.SyndromeGraph`,
    it is easier to use :class:`plaquette.decoders.interfaces.PyMatchingDecoder`
    instead of this class.
    """

    #: Reference to syndrome graph component (we retrieve the syndrome from there)
    sgraph: Optional[syngraph.SyndromeGraphComponent] = None
    #: PyMatching's matching graph
    mgraph: Optional[pymatching.Matching] = None
    #: PyMatching merge strategy for duplicate edges. Change this only if you know
    #: what you are doing.
    merge_strategy: str = "disallow"

    def set_syngraph(self, sgraph: syngraph.SyndromeGraphComponent):
        """Update syndrome graph and weights.

        This method receives the syndrome graph from the decoder interface. The received
        graph was created using the tools from the plaquette library, so this method
        rewrites the graph according to the definitions used in the PyMatching
        library. The graph and weights are inseparable within its definition,
        and when the graph is created this information must already be given.
        """
        self.sgraph = sgraph
        self.mgraph = pymatching.Matching()
        weighted = self.sgraph.is_weighted(support_mixed=False)
        if weighted:
            decoderbase.check_weights_sane(self.sgraph.edge_weights)
        boundary_idx = sgraph.n_vertices
        # Number of connections to boundary vertices for each vertex.
        boundary_edges = np.zeros([sgraph.n_vertices], dtype=int)
        for edge_idx, edge in enumerate(sgraph.edges):
            # In principle, we can omit fault_ids for stabilizer error edges. (Reason:
            # We do not at all use the information about which stab error edges were
            # selected.)
            kw = dict(fault_ids=edge_idx, merge_strategy=self.merge_strategy)
            if weighted:
                kw["weight"] = self.sgraph.edge_weights[edge_idx]
            match edge:
                case (a, b):
                    self.mgraph.add_edge(a, b, **kw)
                case (a,):
                    # Handle each dangling edge on vertex `a` by adding a connection
                    # to a *different* boundary vertex.
                    self.mgraph.add_edge(a, boundary_idx + boundary_edges[a], **kw)
                    boundary_edges[a] += 1
                case _:
                    raise ValueError("Decoder supports 2-edges and 1-edges only")
        if boundary_edges.any():
            boundary = {boundary_idx + i for i in range(boundary_edges.max())}
            self.mgraph.set_boundary_nodes(boundary)

    def update_weights(self):
        """Update weights.

        The PyMatching library receives weights and syndrome graph altogether, so one
        has to set anew the graph with the new weights and feed it to PyMatching.

        .. note::

            Although this method calls for a new set of the syndrome graph, consistency
            dictates that it should check for the existence of a graph within the class
            and create a new graph to be fed into the PyMatching library which is equal
            to the previous, but with a new variable weights.
        """
        assert self.sgraph is not None
        self.set_syngraph(self.sgraph)

    def decode(self):
        """Decode erasure and syndrome."""
        assert self.sgraph is not None
        assert self.mgraph is not None
        if self.sgraph.edge_erased.any():
            raise ValueError("Decoding erasures is not supported yet")
        result = self.mgraph.decode(self.sgraph.syndrome).astype(bool)
        self.sgraph.set_edge_decoder_results(result)
