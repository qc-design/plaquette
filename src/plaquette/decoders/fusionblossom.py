# Copyright 2023, It'sQ GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Interface to MWPM decoder from Fusion-Blossom.

MWPM is short for min-weight perfect matching.
"""

from __future__ import annotations

from typing import Optional

import fusion_blossom as fb  # type: ignore
import numpy as np

from plaquette import syngraph
from plaquette.decoders import decoderbase


class FusionBlossomDecoder(decoderbase.DecoderBackendInterface):
    """Min-weight perfect matching decoder provided by Fusion Blossom.

    This is a low-level interface which interacts with an arbitrary graph defined
    in :class:`.SyndromeGraphComponent`. If you are working :mod:`plaquette.codes`
    and :mod:`plaquette.errors`, or if you are using :class:`.SyndromeGraph`, it
    is easier to use :class:`plaquette.decoders.interfaces.FusionBlossomDecoder`
    instead of this class.
    """

    #: Reference to syndrome graph component (we retrieve the syndrome from there)
    sgraph: Optional[syngraph.SyndromeGraphComponent] = None
    #: Fusion Blossom's initializer graph
    solver: Optional[fb.SolverSerial] = None

    def set_syngraph(self, sgraph: syngraph.SyndromeGraphComponent):
        """Update syndrome graph and weights.

        This method receives the syndrome graph from the decoder interface. The received
        graph was created using the tools from the library, so this method rewrites the
        graph according to the definitions used in the FusionBlossom library. The graph
        and weights are inseparable within its definition, and when the graph is created
        this information must already be given.
        """
        self.sgraph = sgraph
        if self.sgraph.is_weighted(support_mixed=False):
            decoderbase.check_weights_sane(self.sgraph.edge_weights)
            weights = (
                100 * self.sgraph.edge_weights / self.sgraph.edge_weights.max()
            ).astype(int)
        else:
            weights = np.ones(len(self.sgraph.edges)).astype(int)
        n_vertices = self.sgraph.n_vertices
        edges = []
        open_vertices: list[int] = []
        for i, edge in enumerate(self.sgraph.edges):
            weight = 2 * weights[i]
            if len(edge) == 1:
                open_vertices.append(n_vertices + len(open_vertices))
                edge = (edge[0], open_vertices[-1])
            edges.append((edge[0], edge[1], weight))
        n_vertices += len(open_vertices)
        initializer = fb.SolverInitializer(n_vertices, edges, open_vertices)
        self.solver = fb.SolverSerial(initializer)

    def update_weights(self):
        """Update weights.

        The Fusion-Blossom library receives weights and syndrome graph altogether. The
        has to be reset with the new weights and be fed it to Fusion-Blossom.

        .. note::

            Although this method calls for a new solver, consistency dictates that it
            should check for the existence of a graph within the class and create a new
            graph to be fed into the PyMatching library which is equal to the previous,
            but with a new variable weights.
        """
        assert self.sgraph is not None
        self.set_syngraph(self.sgraph)

    def get_syndrome_pattern(self):
        """Create the :class:`fb.SyndromePattern` used by the decoder.

        The solver receives the syndrome and erasure from a specific class called the
        :class:`fb.SyndromePattern`. In order to create such a class, a small
        modification must be made to the :mod:`itsqec` representation of syndromes and
        erasures.
        """
        syndrome = np.where(self.sgraph.syndrome)[0]
        erasure = np.where(self.sgraph.edge_erased)[0]
        return fb.SyndromePattern(syndrome_vertices=syndrome, erasures=erasure)

    def decode(self):
        """Decode erasure and syndrome."""
        assert self.sgraph is not None
        assert self.solver is not None
        syndrome_pattern = self.get_syndrome_pattern()
        self.solver.solve(syndrome_pattern)
        result = self.solver.subgraph(None)
        result = np.array([(i in result) for i in range(len(self.sgraph.edges))])
        self.solver.clear()
        self.sgraph.set_edge_decoder_results(result)
