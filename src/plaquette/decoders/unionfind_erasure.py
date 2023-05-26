# Copyright 2023, It'sQ GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Erasure decoder implementation.

This is the backend implementation of the erasure decoder. It uses a syndrome graph
where edges are defined as tuples of integers. For the union find decoder, which uses
the erasure decoder as its last stage, two higher-level interfaces are
available:

* The interface :class:`.unionfind_interfaces.UnionFindDecoder`
  interacts with an arbitrary graph defined
  in :class:`.SyndromeGraphComponent`.
* The high-level interface :class:`.interfaces.UnionFindDecoder` is most convenient if
  you use :mod:`plaquette.codes` and :mod:`plaquette.errors`, or if you use
  :class:`.SyndromeGraph`.

.. todo::

   The following module docstring is a bit outdated. It will be improved once
   ``unionfind_v2`` is ready.


Reference for the erasure decoder: :cite:`delfosse_linear-time_2020`

The code in this module contains the following classes:

Graph

    Generic graph. It is used to represent the syndrome graph and the
    spanning forest in the erasure decoder.

    This class supports working with subgraphs of a given graph by enabling
    or disabling individual edges and vertices. This is used to build the
    spanning forest (edges are added successively) and to peel the spanning
    forest (edges are removed successively).

ErasureDecoder

    Implementation of the actual decoder. The syndrome is given as set of sites
    on the syndrome graph. The set of erased sites is given as set of edges on
    the syndrome graph. The correction is returned as set of edges on the
    syndrome graph.

    The correction is converted to a Pauli operator by DecoderInterface.

.. todo::

    In some places, we use e.g. lists of Python bool objects. This may be much
    less efficient than using numpy arrays and may be changed in the future.

    This package uses `assert` statements in many places. Many of them
    should be replaced by raising a `ValueError` or a custom error class.

.. todo::

    I was expecting that an independent set of stabilizer generators would be
    used at all times. For the toric code, where the syndrome graph is
    basically a periodic, rectangular grid, this would remove one (arbitrary)
    vertex and make the grid non-translation-invariant. Should we work with a
    independent set of generators and an irregular syndrome graph or should we
    prefer a regular syndrome graph and perform one extra stabilizer generator
    measurement?
"""

import abc
from typing import Optional

import numpy as np


class Graph:
    """Generic graph with support for dangling edges and subgraphs.

    This graph class supports the operations required by the erasure decoder.
    It may be expanded as necessary for other decoders.

    Operations can be restricted to a subgraph by selecting or deselecting edges
    and vertices.

    .. todo::

        "selecting/deselecting" means "enabling/disabling" or
        "adding/removing" and should probably be renamed?

    Attributes on this class should not be accessed directly because some of them
    contain all the vertices/edges, whether selected or not. Users of this class
    should only access the methods but not the attributes of this class. The only
    exception are the attributes required for the Union Find Decoder, being them
    ``edge_progress``, ``clusters`` and ``vertex_in_cluster``.

    The number of vertices in the graph is ``n = self.n_vertices`` vertices.
    The graphs is represented in the following way:

    * Vertices = integers 0 ... n-1
    * List of edges (one edge = tuple containing one or two vertices)
    * List of neighbours of each vertex (derived from list of edges)

    .. todo::

        Representing each vertex and edge by an object may have made the code simpler.
        See e.g. :mod:`.unionfind_decoder`. This can be considered in the future.

    Vertices are identified and represented by non-negative integers 0, ..., n - 1.

    Edges are identified by their position in the list of all edges. This
    index must be supplied if an edge should be selected or deselected.

    Edges can connect two vertices (regular edge) or one vertex (dangling
    edge). Edges which connect more than two vertices are not supported.

    A dangling edge is an edge which connects one vertex to nothing (i.e.
    a dangling edge does *not* connect a vertex to itself).

    In neighbour lists, a dangling edge is marked by specfying -1 as neighbour.

    This class can keep a list of leaf vertices while disabling edges and
    vertices.

    Note: The list of edges supplied to the constructor must not change after
          an instance of the class is created!
    """

    # This implementation works with "leaf vertices" instead of "leaf edges".
    # Determining whether an edge is a "leaf edge" requires looking at all the
    # edges of all the vertices contained in the edge. Determining whether
    # a vertex is a leaf requires looking only at the edges of the vertex.
    # I like leaf vertices better that leaf edges because they are simpler.

    def __init__(self, n_vertices: int, edges: list[tuple[int, int]]):  # noqa: D107
        #: Number of vertices (includes disabled vertices)
        self.n_vertices = n_vertices
        #: Number of edges (includes disabled edges)
        self.n_edges = len(edges)
        #: List of edges (includes disabled edges); `edges[i] = (v0, v1)`
        #: specifies that edge `i` connects vertices `v0` and `v1`.
        self.edges: list[tuple[int, int]] = edges
        #: List storing the growth progress of all edges.
        #: If edge_progress[i]==1.0, then it is already enabled
        self.edge_progress: Optional[list[float]] = None
        #: Vertex `i` is enabled if `vertex_enabled[i]` is True
        self.vertex_enabled: Optional[list[bool]] = None
        #: Edge `i` is enabled if `edge_enabled[i]` is True
        self.edge_enabled: Optional[list[bool]] = None
        #: Neighbour list of each vertex (-1 indicates presence of a dangling edge)
        self.neighbours: Optional[list[list[int]]] = None
        #: List of the neighbors of each vertex which are not connected by an
        #: enabled edge
        self.missing_neighbours: Optional[list[list[int]]] = None
        #: List of leaf vertices (vertices with zero or one edges, including
        #: dangling edges)
        self.leaf_vertices: Optional[list[int]] = None
        #: List of clusters (edges and vertices corresponding to connected graphs).
        #: Makes use of the subclass Cluster.
        self.clusters: Optional[list[Cluster]] = None
        #: Cluster ID of each vertex. `-1` indicates that vertex is not in any cluster
        #: yet.
        self.vertex_in_cluster: Optional[list[int]] = None

    def copy(self):
        """Return a copy of the graph."""
        return Graph(self.n_vertices, self.edges)

    def reset_subgraph(self, default_vertex=True, default_edge=True):
        """Reset vertex/edge status for subgraph.

        You have to call `update_neighbours()` yourself if necessary
        (to update `self.neighbours`).
        """
        assert default_vertex or not default_edge
        self.vertex_enabled = self.n_vertices * [default_vertex]
        self.edge_enabled = len(self.edges) * [default_edge]
        self.edge_progress = len(self.edges) * [0]
        self.vertex_in_cluster = self.n_vertices * [-1]

    def enable_vertex(self, vertex, enable):
        """Enable or disable a vertex.

        Disabling a vertex requires that all connecting edges are disabled.
        If `update_neighbours()` was called at least once, this is enforced.
        """
        if (not enable) and self.neighbours is not None:
            assert len(self.neighbours[vertex]) == 0
        self.vertex_enabled[vertex] = bool(enable)

    def enable_edge(self, edge_idx, enable):
        """Enable or disable an edge.

        This enables or disables the edge `self.edges[edge_idx]`.

        Enabling an edge requires that all referenced vertices are enabled.

        This method keeps `self.neighbours` updated if `update_neighbours()`
        was called at least once.
        """
        edge = self.edges[edge_idx]
        for v in edge:
            assert self.vertex_enabled[v]
        old, new = bool(self.edge_enabled[edge_idx]), bool(enable)
        self.edge_enabled[edge_idx] = new
        self.edge_progress[edge_idx] = int(new)
        if self.neighbours is not None and old != new:
            pairs = ((edge[0], -1),) if len(edge) == 1 else (edge, reversed(edge))
            for v, n in pairs:
                entry = (edge_idx, n)
                nlist = self.neighbours[v]
                was_leaf = len(nlist) <= 1
                if enable:
                    assert entry not in nlist
                    nlist.append(entry)
                else:
                    nlist.remove(entry)
                if self.leaf_vertices is not None:
                    assert not new, "Adding edges while tracking leaves not implemented"
                    if len(nlist) <= 1 and not was_leaf:
                        self.leaf_vertices.append(v)

    def enable_edges(self, edges, enable, select_vertices=False):
        """Enable a list of edges and (optionally) all referenced vertices."""
        for edge_idx in edges:
            if select_vertices:
                for v in self.edges[edge_idx]:
                    self.enable_vertex(v, enable)
            self.enable_edge(edge_idx, enable)

    def iter_vertices(self):
        """Iterate over all enabled vertices."""
        if self.vertex_enabled is None:
            return range(self.n_vertices)
        return (i for i, s in enumerate(self.vertex_enabled) if s)

    def iter_edges(self):
        """Iterate over `(edge_idx, edge)` for all enabled edges."""
        if self.edge_enabled is None:
            return self.edges.items()
        return ((i, edge) for i, edge in enumerate(self.edges) if self.edge_enabled[i])

    def is_vertex_enabled(self, vertex):
        """Return whether `edge` is enabled."""
        return self.vertex_enabled[vertex]

    def is_edge_enabled(self, edge):
        """Return whether `edge` is enabled."""
        return self.edge_enabled[edge]

    def get_neighbours(self, vertex):
        """Get neighbours of a vertex.

        Requires that you to called `update_neighbours()` at least once.
        """
        assert self.vertex_enabled is None or self.vertex_enabled[vertex]
        return self.neighbours[vertex]

    def update_neighbours(self):
        """Update neighbour lists (`self.neighbours`)."""
        self.neighbours = [[] for _ in range(self.n_vertices)]
        self.missing_neighbours = [[] for _ in range(self.n_vertices)]
        for i, edge in enumerate(self.edges):
            if not self.edge_enabled[i]:
                self._update_neighbours(self.missing_neighbours, i, edge)
                continue
            for v in edge:
                assert self.vertex_enabled[v]
            self._update_neighbours(self.neighbours, i, edge)

    def _update_neighbours(self, neighbour_type, i, edge):
        """Update the existing neighbours or the missing neighbours."""
        if len(edge) == 1:
            # A dangling edge is represented as -1.
            assert (i, -1) not in neighbour_type[edge[0]]
            neighbour_type[edge[0]].append((i, -1))
        else:
            a, b = edge
            assert (i, b) not in neighbour_type[a]
            neighbour_type[a].append((i, b))
            assert (i, a) not in neighbour_type[b]
            neighbour_type[b].append((i, a))

    def clear_neighbours(self):
        """Clear neighbour lists."""
        self.neighbours = None

    def update_leaves(self):
        """Update list of leaf vertices.

        A vertex is considered a leaf it has zero or one (enabled) edges,
        dangling edges included.
        """
        self.leaf_vertices = [
            v
            for v, sel in enumerate(self.vertex_enabled)
            if sel and len(self.neighbours[v]) <= 1
        ]

    def pop_leaf(self):
        """Disable a leaf vertex and its edge.

        Note that disabling the leaf vertex and its edge updates the list of leaf
        vertices.

        Returns `(leaf_vertex, edge_idx, other_vertex)` where `leaf_vertex` and
        `edge_idx` were already disabled and `other_vertex` is the other vertex
        of the disabled edge.

        If the disabled edge was a dangling edge, `other_vertex` is `-1`.

        If the vertex did not have any edges, `edge_idx` and `other_vertex` are both
        `-1` and only the vertex is disabled.
        """
        leaf_vertex = self.leaf_vertices.pop()
        assert self.vertex_enabled[leaf_vertex]
        neighs = self.neighbours[leaf_vertex]
        if len(neighs) == 0:
            edge_idx = other_vertex = -1
        else:
            edge_idx, other_vertex = neighs[0]
            self.enable_edge(edge_idx, False)
        self.enable_vertex(leaf_vertex, False)
        return leaf_vertex, edge_idx, other_vertex

    def pop_leaves(self):
        """Call `pop_leaf()` until the graph contains zero vertices."""
        while self.leaf_vertices:
            yield self.pop_leaf()


class Cluster:
    """Connected Subgraph.

    Contains information of the vertices and edges of a connected graph
    within the main graph. It also contains information required to perform
    the growth of clusters to be merged within the Union Find Decoder.

    It contains no methods.
    """

    def __init__(self):  # noqa: D107
        #: Vertices lying in the boundary of the cluster.
        self.boundary = []
        #: Set containing the matches found for the cluster after its (and
        #: other clusters') growth.
        self.found = set()
        #: Parity of the cluster. 0 if it's even, 1 if it's odd and
        #: 2 if contains a dangling edge.
        self.parity = 0


class ErasureDecoderInterface(metaclass=abc.ABCMeta):
    """Internal interface definition for erasure and union find decoders."""

    @abc.abstractmethod
    def decode(self, erasure, syndrome):
        """Decode erasure and syndrome.

        :param erasure:
            Indices of erased edges
        :param syndrome:
            List of syndrome bits for each vertex; ``syndrome[i]`` is for vertex ``i``

        :return:
            List of true/false for each edge; ``corr[i]`` is for edge ``i``
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_weights(self, weights: Optional[np.ndarray]):
        """Set weights.

        Args:
            weights: Array containing one weight for each edge (optional).
        """
        raise NotImplementedError


class ErasureDecoder(ErasureDecoderInterface):
    """The maximum likelihood erasure decoder.

    Reference: :cite:`delfosse_linear-time_2020`
    """

    def __init__(self, graph):  # noqa: D107
        self.graph = graph

    def set_weights(self, weights: Optional[np.ndarray]):
        """Set weights.

        Raises:
            ValueError
                This method always raises ``ValueError`` because the decoder does not
                support weights.
        """
        raise ValueError("This decoder does not support weights")

    def decode(self, sel_edges, syndrome):
        """Decode a given erasure and syndrome.

        Implements :meth:`ErasureDecoderInterface.decode`.
        """
        # Apply edge selection
        self.graph.clear_neighbours()
        self.graph.reset_subgraph(False, False)
        self.graph.enable_edges(sel_edges, True, select_vertices=True)
        self.graph.update_neighbours()
        # Construct spanning forest
        self.construct_forest()
        # Peel the forest
        return self.peel_forest(syndrome.copy())

    def peel_forest(self, syndrome):
        """Peel the forest.

        A given edge is considered a "leaf" edge if at least one of its vertices
        is a leaf. A given vertex is a leaf if it has zero or one edges. This makes
        tracking leaf edges a bit more difficult than tracking leaf vertices.

        In this implementation, we choose to keep track of leaf vertices instead
        of tracking leaf edges. However, this means than we will sometimes encounter
        a leaf with zero remaining edges (the last vertex of any connected component
        which does not contain a dangling edge).
        """
        # An "end vertex" is a vertex with zero edges. In the beginning, there
        # should not be any end vertices.
        end_vertex = [False] * self.graph.n_vertices
        corr = [False] * self.graph.n_edges
        self.forest.update_neighbours()
        # Create a list of all leaf vertices which is updated as we peel the tree.
        self.forest.update_leaves()

        for leaf_vertex, edge_idx, other_vertex in self.forest.pop_leaves():
            if end_vertex[leaf_vertex]:
                # We know that `leaf_vertex` is expected to have zero edges.
                end_vertex[leaf_vertex] = False
                # edge_idx == -1 indicates that there is no connecting edge.
                assert edge_idx == -1
                continue
            # We expect `leaf_vertex` to have exactly one edge.
            assert edge_idx != -1
            assert leaf_vertex != -1  # This should be impossible
            if (
                other_vertex != -1
                and len(self.forest.get_neighbours(other_vertex)) == 0
            ):
                # After removing edge_idx, other_vertex now has zero edges
                # -> it is an end vertex.
                end_vertex[other_vertex] = True
            if syndrome[leaf_vertex]:
                assert not corr[edge_idx], "each edge can be peeled at most once?"
                corr[edge_idx] = True
                syndrome[leaf_vertex] = False
                # other_vertex == -1 indicates a dangling edge. In case of a dangling
                # edge, we cannot update the syndrome. Is this correct?
                if other_vertex != -1:
                    syndrome[other_vertex] = not syndrome[other_vertex]
        # No end vertices should remain.
        assert sum(end_vertex) == 0
        return corr

    def construct_forest(self):
        """Construct spanning forest of selected sub-graph."""
        self.forest = self.graph.copy()
        self.forest.reset_subgraph(False, False)
        rem_vertices = list(self.graph.iter_vertices())
        new_vertices = []
        # Start by selecting all dangling edges (if there are any).
        # The forest will have one connected component for each dangling edge.
        for edge_idx, edge in self.graph.iter_edges():
            if len(edge) == 1:
                if self.forest.vertex_enabled[edge[0]]:
                    self.graph.enable_edge(edge_idx, False)
                    continue
                self.forest.enable_vertex(edge[0], True)
                self.forest.enable_edge(edge_idx, True)
                new_vertices.append(edge[0])
        while rem_vertices or new_vertices:
            # If new_vertices is empty, we have to start a new connected component.
            while rem_vertices and not new_vertices:
                # Start new connected component at arbitrary vertex which is not yet
                # part of the tree.
                candidate = rem_vertices.pop()
                if not self.forest.is_vertex_enabled(candidate):
                    self.forest.enable_vertex(candidate, True)
                    new_vertices.append(candidate)
            if not new_vertices:
                break  # We are done.
            # Continue by checking the edges of some vertex `cur`.
            #
            # TODO Does it make a difference whether we remove the first or
            # last element of new_vertices? (i.e. pop(0) versus pop()
            # If we keep pop(0), we should use containers.deque (refer to
            # the awesome https://wiki.python.org/moin/TimeComplexity for more info).
            cur = new_vertices.pop(0)
            for edge_idx, neigh in self.graph.get_neighbours(cur):
                if self.forest.is_edge_enabled(edge_idx):
                    # Ignore the edge if it is already part of the forest.
                    continue
                assert neigh != -1, "Unexpected dangling edge"
                if not self.forest.is_vertex_enabled(neigh):
                    # If the neighbouring vertex is not part of the forest, add the
                    # edge and the vertex.
                    self.forest.enable_vertex(neigh, True)
                    self.forest.enable_edge(edge_idx, True)
                    new_vertices.append(neigh)
