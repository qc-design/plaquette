# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Interface to union find decoders.

.. note::

    This module provides low-level interfaces to the union find decoder. If you only
    need high-level functions, it is easier to use the high-level interfaces from
    :mod:`plaquette.decoders` instead.
"""

from __future__ import annotations

import abc
import warnings
from typing import Optional

import numpy as np

from plaquette import syngraph
from plaquette.decoders import decoderbase
from plaquette.decoders import unionfind_decoder as unionfind_dec
from plaquette.decoders import unionfind_erasure as erasure_dec


class UnionFindBase(decoderbase.DecoderBackendInterface, metaclass=abc.ABCMeta):
    """Base class for plaquette unionfind decoders."""

    #: Reference to syndrome graph component (we retrieve syndrome and erasure from
    #: there)
    sgraph: Optional[syngraph.SyndromeGraphComponent] = None
    #: The decoder implementation
    decoder: Optional[erasure_dec.ErasureDecoderInterface] = None

    @staticmethod
    @abc.abstractmethod
    def _get_union_find(
        graph: erasure_dec.Graph,
    ) -> erasure_dec.ErasureDecoderInterface:
        raise NotImplementedError

    def set_syngraph(self, sgraph: syngraph.SyndromeGraphComponent):
        """Update syndrome graph and weights.

        This method receives the syndrome graph on which the correction will be
        performed.
        """
        self.sgraph = sgraph
        graph = erasure_dec.Graph(self.sgraph.n_vertices, self.sgraph.edges)
        self.decoder = self._get_union_find(graph)
        if self.sgraph.is_weighted(support_mixed=False):
            weights = self.sgraph.edge_weights
            decoderbase.check_weights_sane(weights)
        else:
            weights = None
        self.decoder.set_weights(weights)

    def update_weights(self):
        """Update weights.

        Given an already existing decoder, one can give a new weight distribution to be
        used during the decoding. This function updates the weight information in the
        decoder's already existing Graph. The new weights are taken from the previously
        supplied syndrome graph.
        """
        assert self.sgraph is not None
        assert self.decoder is not None
        self.decoder.set_weights(self.sgraph.edge_weights)

    def decode(self):
        """Decode erasure and syndrome."""
        assert self.sgraph is not None
        assert self.decoder is not None
        erased_edge_indices = np.where(self.sgraph.edge_erased)[0]
        edge_selected = self.decoder.decode(erased_edge_indices, self.sgraph.syndrome)
        edge_sel_array = np.array(edge_selected)
        self.sgraph.set_edge_decoder_results(edge_sel_array)


class UnionFindNoWeights(UnionFindBase):
    """plaquette union find decoder (without support for weights).

    This is a low-level interface which interacts with an arbitrary graph defined
    in :class:`.SyndromeGraphComponent`. If you are working :mod:`plaquette.codes` and
    :mod:`plaquette.errors`, or if you are using :class:`.SyndromeGraph`,
    it is easier to use :class:`plaquette.decoders.interfaces.UnionFindNoWeights`
    instead of this class.

    This is an interface to :class:`.unionfind_decoder.UnionFindNoWeights`.
    """

    @staticmethod
    def _get_union_find(graph):
        return unionfind_dec.UnionFindDecoder(unionfind_dec.UnionFindNoWeights, graph)


class UnionFindDecoder(UnionFindBase):
    """plaquette union find decoder.

    This is a low-level interface which interacts with an arbitrary graph defined
    in :class:`.SyndromeGraphComponent`. If you are working :mod:`plaquette.codes` and
    :mod:`plaquette.errors`, or if you are using :class:`.SyndromeGraph`,
    it is easier to use :class:`plaquette.decoders.interfaces.UnionFindDecoder`
    instead of this class.

    This is an interface to :class:`.unionfind_decoder.UnionFindDecoder`.
    """

    def __init__(self, *args, **kwargs):  # noqa: D107
        super().__init__(*args, **kwargs)
        warnings.warn(
            "Use the UnionFindDecoderInterface class from the plaquette_unionfind "
            "package instead",
            DeprecationWarning,
            stacklevel=2,
        )

    @staticmethod
    def _get_union_find(graph):
        return unionfind_dec.UnionFindDecoder(unionfind_dec.UnionFind, graph)
