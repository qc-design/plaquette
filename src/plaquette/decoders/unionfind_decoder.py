# Copyright 2023, It'sQ GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""``plaquette`` union find decoder implementation.

Reference: :cite:`delfosse_almost-linear_2021`

This is the backend implementation of the union find decoder. It uses a syndrome graph
where edges are defined as tuples of integers. Two higher-level interfaces are
available:

* The interface :class:`.unionfind_interfaces.UnionFindDecoder`
  interacts with an arbitrary graph defined
  in :class:`.SyndromeGraphComponent`.
* The high-level interface :class:`.interfaces.UnionFindDecoder` is most convenient if
  you use :mod:`plaquette.codes` and :mod:`plaquette.errors`, or if you use
  :class:`.SyndromeGraph`.

Basic ideas:

* The underlying graph is the full matching graph (aka syndrome graph), where
  nodes are stabilizer measurements and edges are errors which toggle adjacent
  stabilizers.

* A :class:`Vertex` is either unused, in the boundary of a cluster or in the
  interior of a cluster.

* If a vertex is in the boundary of a cluster, :attr:`Vertex.cluster` holds the
  :class:`Cluster` object to which the vertex belongs. Otherwise,
  :attr:`Vertex.cluster` is ``None``.

* An :class:`Edge` can be not grown at all, partially grown or complete. This
  is stored in :attr:`Edge.cur_growth` and :attr:`Edge.max_growth`.

* A :class:`Cluster` stores only boundary vertices and the cluster's parity.

* A :class:`Graph` instance creates vertices, edges and clusters.

* After the union find step is done, the maximum likelihood erasure decoder
  from :class:`plaquette.decoders.unionfind_erasure.ErasureDecoder` is used. The
  erasure decoder is run on the unmodified syndrome and on all edges which are
  fully grown after the union find step.

.. todo::

    The :class:`Graph` implemented in this module is very different from the
    :class:`plaquette.decoders.unionfind_erasure.Graph` used in the erasure decoder.
    It may be worthwhile to refactor the erasure decoder.

.. todo::

    Clusters are merged immediately, without identifying groups of to-be-merged
    clusters. This could degrade the algorithm's complexity. Identifying groups of
    to-be-merged clusters is not trivial; it could be done by looking for connected
    components in some abstract graph (each cluster is a vertex and to-be-merged
    clusters are connected).

    Suppose that each cluster holds a list of other clusters with which it
    should be merged. The list is shared across all clusters which should be
    merged, with the intention that one list append is sufficient to add a new
    cluster to the group. This creates a problem if the new cluster is already
    part of some other merge group which has its own list. Therefore, the
    connected component search mentioned above appears better suited for the
    task.
"""

from __future__ import annotations

from typing import Optional, Type

import numpy as np

from plaquette.decoders import unionfind_erasure as erasure_dec

# TODO Use some kind of enum type definition.
_V_UNUSED = 0
_V_BOUNDARY = 1
_V_INTERIOR = 2


class Vertex:
    """Vertex in a matching graph.

    This class tracks clusters using a tree structure. The tree is a directed graph
    in which edges go from childs to parents but not in the converse direction.

    Actually, the directed graph can be viewed as a forest containing multiple trees
    (one tree for each cluster).

    If a vertex is not part of any cluster, both :attr:`parent` and
    :attr:`cluster` are ``None``.

    If a vertex is the root node of a tree, :attr:`cluster` is set but
    :attr:`parent` is ``None``.

    If a vertex is a non-root node in a tree, :attr:`parent` is set but
    :attr:`cluster` is ``None``.
    """

    #: State is `unused`, `boundary` or `interior`.
    state = _V_UNUSED
    #: Parent within the cluster tree.
    parent: Optional[Vertex] = None
    #: Cluster to which this vertex belongs.
    cluster: Optional[Cluster] = None
    #: Syndrome bit on this vertex
    syndrome: bool = False

    def __init__(self, idx: int):  # noqa: D107
        #: Vertex ID
        self.idx: int = idx
        #: List of adjacent edges
        self.edges: list[Edge] = list()
        #: Neighbouring vertices (for each edge); contains `None` for dangling edge
        self.neigh_vertices: list[Optional[Vertex]] = list()

    def reset(self):
        """Reset vertex to ``unused`` state (for new decoding run)."""
        self.state = _V_UNUSED
        self.parent = None
        self.cluster = None

    def set_syndrome(self, syndrome: bool):
        """Update syndrome bit on vertex."""
        self.syndrome = syndrome

    def neigh_edges(self):
        """Iterate over neighouring edges."""
        return self.edges

    def neigh_edges_vertices(self):
        """Iterate over neighbouring edges and vertices."""
        return zip(self.edges, self.neigh_vertices)

    def add_edge(self, edge: Edge):
        """Add an adjacent edge."""
        if len(edge.vertices) == 2:
            if edge.vertices[0] is self:
                other = edge.vertices[1]
            else:
                assert edge.vertices[1] is self
                other = edge.vertices[0]
        else:
            assert len(edge.vertices) == 1
            other = None
        self.edges.append(edge)
        self.neigh_vertices.append(other)

    def any_edge_incomplete(self):
        """Check whether there is at least one incomplete, adjacent edge."""
        return any(not edge.is_complete() for edge in self.edges)

    def all_edges_complete(self):
        """Check whether all adjacent edges are complete."""
        return all(edge.is_complete() for edge in self.edges)

    def is_unused(self):
        """Check whether state is `unused` (not in any cluster)."""
        return self.state == _V_UNUSED

    def is_boundary(self):
        """Check whether state is `boundary` (of a cluster)."""
        return self.state == _V_BOUNDARY

    def is_interior(self):
        """Check whether state is `interior` (of a cluster)."""
        return self.state == _V_INTERIOR

    def find_root(self) -> Vertex:
        """Find cluster tree root.

        Updates :attr:`parent` to reduces the tree depth.
        """
        if self.parent is None:
            assert self.cluster is not None
            return self
        else:
            root = self.parent.find_root()
            self.parent = root
            return root

    @property
    def r_cluster(self) -> Cluster:
        """Cluster object from cluster tree root."""
        r_cluster = self.find_root().cluster
        assert r_cluster is not None  # for mypy
        return r_cluster

    def set_parent(self, root: Vertex):
        """Set new cluster tree parent."""
        assert (self.cluster is None) != (self.parent is None)
        if self.parent is None:
            assert self.cluster is not None  # Redundant check for mypy
        self.cluster = None
        self.parent = root

    def to_boundary(self, root: Cluster | Vertex):
        """Change state from `unused` to `boundary`."""
        assert self.state == _V_UNUSED
        assert root is not None
        if isinstance(root, Cluster):
            self.cluster = root
        else:
            self.parent = root
        self.state = _V_BOUNDARY

    def to_interior(self):
        """Change state from `boundary` to `interior`."""
        assert self.state == _V_BOUNDARY
        self.state = _V_INTERIOR


class Edge:
    """Edge in a matching graph.

    Normal edges connect two vertices. Dangling edges connect one vertex to
    nothing.
    """

    #: Growth value for "fully grown" / "complete" edge
    max_growth: float = 2
    #: Increment for each growth step
    growth_increment: float = 1
    #: Current growth value (0 = not grown at all)
    cur_growth: float = 0

    def __init__(self, idx: int, vertices: tuple[Vertex, ...]):  # noqa: D107
        assert len(vertices) in (1, 2)
        #: Edge ID
        self.idx = idx
        #: Connected vertices
        self.vertices: tuple[Vertex, ...] = vertices
        #: Whether edge is a dangling edge
        self.is_dangling: bool = len(self.vertices) == 1
        for v in self.vertices:
            v.add_edge(self)

    def set_weight(self, max_growth: float, growth_increment: float):
        """Update weight-related parameters ``max_growth`` and ``growth_increment``."""
        self.max_growth = max_growth
        self.growth_increment = growth_increment

    def reset(self):
        """Reset edge to "ungrown" state (for new decoding run)."""
        self.cur_growth = 0

    def is_complete(self):
        """Check whether edge is complete (fully grown)."""
        return self.cur_growth >= self.max_growth

    def set_complete(self):
        """Set edge to complete (fully grown) state."""
        assert self.cur_growth < self.max_growth
        self.cur_growth = self.max_growth

    def grow(self, growth_increment: Optional[float]):
        """Grow edge one step (only allowed if edge is incomplete).

        If the ``growth_argument`` is not ``None``, it is used instead of
        :attr:`growth_increment`.

        Return true if edge is fully grown after the growth step.
        """
        assert self.cur_growth < self.max_growth
        if growth_increment is None:
            growth_increment = self.growth_increment
        self.cur_growth += growth_increment
        return self.cur_growth >= self.max_growth


class Graph:
    """A matching graph.

    :param n_vertices:
        Number of vertices in the graph
    :param edges:
        List of edges. Each edge is specified by a tuple of length 2 or 1
        (normal or dangling edge). Tuple elements are integers from
        `range(n_vertices)`.
    """

    #: Number of additional vertices (e.g. from split edges)
    n_add_vertices: int = 0

    def __init__(self):  # noqa: D107
        #: All vertices in the graph
        self.vertices: list[Vertex] = list()
        #: All edges in the graph
        self.edges: list[Edge] = list()
        #: All (open and closed) clusters in the graph
        self.clusters: list[Cluster] = list()
        #: Open clusters in the graph (open means parity 1)
        self.open_clusters: list[Cluster] = list()

    def set_vertices_edges(self, n_vertices: int, edges: list[tuple[int]]):
        """Fill in vertices and edges from external definition."""
        assert len(self.vertices) == 0, "Can only set vertices and edges once"
        self._edges = edges
        for idx in range(n_vertices):
            self.vertices.append(Vertex(idx))
        for idx, v_indices in enumerate(self._edges):
            assert all(v >= 0 for v in v_indices)
            vertices = tuple(self.vertices[v] for v in v_indices)
            self.edges.append(Edge(idx, vertices))

    def set_vertices_split_edges(self, n_vertices: int, edges: list[tuple[int]]):
        """Split all edges from external definition."""
        assert len(self.vertices) == 0, "Can only set vertices and edges once"
        self._edges = edges
        # self.vertices[:n_vertices]:  Actual vertices
        # self.vertices[n_vertices:]:  Additional vertices from split edges
        self.n_add_vertices = len(edges)
        for idx in range(n_vertices + self.n_add_vertices):
            self.vertices.append(Vertex(idx))
        # self.edges[:n_add_vertices]:  First half of each edge
        # self.edges[n_add_vertices:]:  Second half of each edge
        edges2 = list()
        for edge_idx, v_indices in enumerate(self._edges):
            edge2_idx = self.n_add_vertices + edge_idx
            assert all(v >= 0 for v in v_indices)
            vertices = tuple(self.vertices[idx] for idx in v_indices)
            # Additional vertex for splitting the edge into two parts
            add_v = self.vertices[n_vertices + edge_idx]
            first_half = (vertices[0], add_v)
            second_half: tuple[Vertex, ...]
            if len(vertices) == 2:
                second_half = (add_v, vertices[1])
            else:
                second_half = (add_v,)
            self.edges.append(Edge(edge_idx, first_half))
            edges2.append(Edge(edge2_idx, second_half))
        self.edges.extend(edges2)

    def get_complete_edges(self) -> list[int]:
        """Return (external) indices of all complete edges.

        If split edges are used, this returns the state of external edges. A
        split edge is considered complete if both halfes are complete.
        """
        if self.n_add_vertices == 0:
            return [
                edge_idx
                for edge_idx, edge in enumerate(self.edges)
                if edge.is_complete()
            ]
        else:
            complete_edge_idx = list()
            for edge_idx in range(self.n_add_vertices):
                edge1 = self.edges[edge_idx]
                edge2 = self.edges[self.n_add_vertices + edge_idx]
                if edge1.is_complete() and edge2.is_complete():
                    complete_edge_idx.append(edge_idx)
            return complete_edge_idx

    def refresh_open_clusters(self):
        """Refresh list of open clusters (open = parity 1).

        Drops and closes all clusters of parity 0 and -1 (dangling edge
        present).
        """
        new_list = list()
        for cluster in self.open_clusters:
            if cluster.parity == 1:
                new_list.append(cluster)
            else:
                cluster.close()
        self.open_clusters = new_list

    def reopen_cluster(self, cluster):
        """Reopen a cluster.

        Reopening a cluster can be necessary when a parity-1 cluster grows
        until it merges with a bigger parity-0 cluster. In this case, the
        parity-0 cluster is retained but it then has parity 1 and needs to
        be reopened.
        """
        cluster.reopen()
        self.open_clusters.append(cluster)

    def reset(self):
        """Reset graph state for new decoding run."""
        self.clusters = list()
        self.open_clusters = list()
        for vertex in self.vertices:
            vertex.reset()
        for edge in self.edges:
            edge.reset()

    def set_syndrome(self, syndrome: list[bool]):
        """Set syndrome on vertices."""
        assert len(syndrome) == len(self.vertices) - self.n_add_vertices
        for vertex, value in zip(self.vertices, syndrome):
            vertex.set_syndrome(value)

    def add_cluster(self) -> Cluster:
        """Add a new (empty) cluster and return it."""
        cluster = Cluster(self, len(self.clusters))
        self.clusters.append(cluster)
        self.open_clusters.append(cluster)
        return cluster

    def init(self, erasure, syndrome):
        """Initialize graph for union find."""
        self.reset()
        # Set syndrome on vertices
        self.set_syndrome(syndrome)
        # Mark erased edges as complete
        for edge_idx in erasure:
            assert edge_idx >= 0
            # Edges with weight zero are complete already.
            if not self.edges[edge_idx].is_complete():
                self.edges[edge_idx].set_complete()
            if self.n_add_vertices > 0:
                if not self.edges[edge_idx + self.n_add_vertices].is_complete():
                    self.edges[edge_idx + self.n_add_vertices].set_complete()
        self.init_erasure_clusters()
        self.init_syndrome_clusters()

    def init_erasure_clusters(self):
        """Determine connected components of erased edges with a DFS.

        DFS = Depth-first search
        """
        edge_visited = [False] * len(self.edges)
        for edge_idx, edge in enumerate(self.edges):
            if edge_visited[edge_idx] or not edge.is_complete():
                continue
            cluster = self.add_cluster()
            start_vertex = edge.vertices[0]
            cluster.add_vertex(start_vertex, edge)
            todo = [start_vertex]
            while todo:
                start_vertex = todo.pop()
                for next_edge, next_vertex in start_vertex.neigh_edges_vertices():
                    if edge_visited[next_edge.idx]:
                        continue
                    edge_visited[next_edge.idx] = True
                    if not next_edge.is_complete():
                        continue
                    if next_vertex is None:
                        cluster.add_dangling_edge(next_edge)
                    elif next_vertex.is_unused():
                        cluster.add_vertex(next_vertex, next_edge)
                        todo.append(next_vertex)
            cluster.update_after_growth()

    def init_syndrome_clusters(self):
        """Create single-vertex clusters for unused vertices with syndrome bit."""
        for vertex in self.vertices:
            if vertex.syndrome and vertex.is_unused():
                cluster = self.add_cluster()
                cluster.add_vertex(vertex, None)
                cluster.update_after_growth()


class Cluster:
    """Cluster in union find on matching graph."""

    #: Root vertex of the cluster's tree
    root_vertex: Optional[Vertex] = None
    #: List of boundary vertices. `None` indicates that the cluster was deleted.
    boundary: Optional[list[Vertex]]
    #: Total number of vertices (interior + boundary)
    n_vertices: int = 0
    #: Syndrome parity (-1 indicates presence of a dangling edge)
    parity: int = 0
    #: Tracks whether the cluster is in :attr:`Graph.open_clusters`
    open: bool = True

    def __init__(self, graph: Graph, idx: int):  # noqa: D107
        self.graph = graph
        self.idx = idx
        self.boundary = list()

    def close(self):
        """Close a cluster.

        Called from :class:`Graph` when a cluster is removed from
        :attr:`Graph.open_clusters`.
        """
        assert self.open
        self.open = False

    def reopen(self):
        """Reopen a cluster.

        Called from :class:`Graph` when a cluster is re-added to
        :attr:`Graph.open_clusters`.
        """
        assert not self.open
        self.open = True

    def delete(self):
        """Delete the cluster."""
        self.root_vertex = None
        self.boundary = None
        self.n_vertices = 0
        self.parity = 0

    def add_dangling_edge(self, edge: Edge):
        """Add a dangling edge to the cluster."""
        self.parity = -1

    def merge_with(self, other: Cluster, edge: Edge):
        """Merge cluster with another cluster."""
        if other is self:
            return  # Nothing to do
        # TODO Should we decide based on `len(self.boundary)` or `self.n_vertices`?
        # The paper suggests deciding according to the latter.
        assert self.boundary is not None  # asserts for mypy - len() would fail
        assert other.boundary is not None
        if len(self.boundary) < len(other.boundary):
            # The other cluster has a larger boundary. We should keep `other`
            # instead of `self`.
            return other.merge_with(self, edge)
        self.n_vertices += other.n_vertices
        # Update parity using other.parity
        if self.parity == -1 or other.parity == -1:
            self.parity = -1
        else:
            self.parity = (self.parity + other.parity) % 2
        # Take boundary vertices from other cluster
        assert self.root_vertex is not None  # Asserts for mypy
        assert other.root_vertex is not None
        other.root_vertex.set_parent(self.root_vertex)
        for vertex in other.boundary:
            self.boundary.append(vertex)
        other.delete()
        if not self.open and self.parity == 1:
            # Request that the cluster is re-added to :attr:`Graph.open_clusters`
            self.graph.reopen_cluster(self)

    def add_vertex(self, vertex: Vertex, from_edge: Optional[Edge] = None):
        """Add vertex from edge or syndrome bit (``from_edge = None``)."""
        assert self.boundary is not None  # Assert for mypy
        self.n_vertices += 1
        # from_edge can be used for debugging or printing.
        if self.parity >= 0:
            self.parity = (self.parity + vertex.syndrome) % 2
        if len(self.boundary) == 0:
            self.root_vertex = vertex
            vertex.to_boundary(self)
        else:
            assert self.root_vertex is not None  # Assert for mypy
            vertex.to_boundary(self.root_vertex)
        self.boundary.append(vertex)

    def update_after_growth(self):
        """Remove vertices from boundary if all their edges are complete."""
        if self.boundary is None:
            return
        new_boundary = list()
        for vertex in self.boundary:
            if vertex.all_edges_complete():
                vertex.to_interior()
            else:
                new_boundary.append(vertex)
        self.boundary = new_boundary


class UnionFindNoWeights:
    """Union find on matching graph.

    This class only implements union find, but not the final erasure decoder.
    For that, see :class:`UnionFindDecoder`.
    """

    #: Update edge weights during growth. Used by some weighted implementations.
    choose_growth_increment_enabled: bool = False
    #: Whether any weights have been set at all
    weighted: bool = False
    #: Whether split edges should be used
    split_edges: bool = False

    def __init__(self, n_vertices, edges):  # noqa: D107
        self.graph: Graph = Graph()
        #: Weights normalization coefficient
        self._c = 2
        if self.split_edges:
            self.graph.set_vertices_split_edges(n_vertices, edges)
        else:
            self.graph.set_vertices_edges(n_vertices, edges)

    @property
    def normalization_coefficient(self) -> float:
        """Return the weights normalization coefficient."""
        return self._c

    @normalization_coefficient.setter
    def normalization_coefficient(self, val: float | int):
        """Change the value of the weights normalization coefficient."""
        if not isinstance(val, (float, int)):
            raise TypeError("Coefficient can only be float or int")
        if val < 0:
            raise ValueError("Coefficient must be non-negative")
        self._c = float(val)

    def set_weights(self, weights: Optional[np.ndarray]):
        """Set weights.

        Args:
            weights: Array containing one weight for each edge (optional).
        """
        if weights is None:
            if self.weighted:
                self.weighted = False
                for edge in self.graph.edges:
                    edge.set_weight(2, 1)
        else:
            weights = self._c * weights / np.min(weights[weights != 0])
            self._set_weights(weights)
            self.weighted = True

    def _set_weights(self, weights: np.ndarray):
        """Set weights (to be overridden by subclass)."""
        raise ValueError("This decoder does not support weights")

    def choose_growth_increment(self, cluster: Cluster) -> Optional[float]:
        """Choose cluster's growth increment as minimum for growing at least one edge.

        .. note::

           This method is only used by those subclasses which set
           :attr:`choose_growth_increment_enabled` to ``True``.

        Following the process described in :cite:`huang_fault-tolerant_2020`,
        the weight of each edge
        determines the amount of times one has to iterate over its growth until it is
        complete. However, a further consideration is made: the increment at each edge
        of a cluster during a step is given by the smallest increment that can be
        assigned to every edge to-be-grown to complete the growth of at least one edge
        at each growth step.
        """
        if not (self.choose_growth_increment_enabled and self.weighted):
            return None
        assert cluster.boundary is not None
        increments = list()
        for vertex in cluster.boundary:
            for edge in vertex.neigh_edges():
                if not edge.is_complete():
                    increments.append(edge.max_growth - edge.cur_growth)
        if len(increments) == 0:
            return None
        return min(increments)

    def union_find(self, erasure, syndrome):
        """Perform union find and return new, extended erasure."""
        # Initialize clusters from erasure and syndrome
        self.graph.init(erasure, syndrome)
        # Perform union find steps until there are no odd-parity clusters.
        while len(self.graph.open_clusters) > 0:
            self.step()
        # Return indices of all complete edges after the union find.
        return self.graph.get_complete_edges()

    def step(self):
        """Perform a union find step."""
        self.graph.refresh_open_clusters()
        if len(self.graph.open_clusters) == 0:
            return
        min_size = min(len(c.boundary) for c in self.graph.open_clusters)
        edges_grown = 0
        new_edges = list()
        for cluster in self.graph.open_clusters:
            if len(cluster.boundary) > min_size:
                continue
            growth_increment = self.choose_growth_increment(cluster)
            for vertex in cluster.boundary:
                for edge in vertex.neigh_edges():
                    # Should we only grow edges to unused vertices?
                    if not edge.is_complete():
                        edge.grow(growth_increment)
                        edges_grown += 1
                        if edge.is_complete():
                            new_edges.append(edge)
        assert edges_grown > 0, "Could not find any edges to grow"
        if new_edges:
            self.handle_new_edges(new_edges)
            for cluster in self.graph.clusters:
                cluster.update_after_growth()

    def handle_new_edges(self, new_edges: list[Edge]):
        """Handle new completed edges."""
        for edge in new_edges:
            if len(edge.vertices) == 1:
                # Add dangling edge to cluster.
                v = edge.vertices[0]
                assert v.is_boundary()
                v.r_cluster.add_dangling_edge(edge)
            else:
                # At least one vertex must be a boundary vertex, but we do not
                # know which one.
                v0, v1 = edge.vertices
                if not v0.is_boundary():
                    v0, v1 = v1, v0
                # v0 should be a boundary vertex now.
                assert v0.is_boundary()
                if v1.is_boundary():
                    # Both vertices are part of some cluster --> merge.
                    #
                    # If v0 and v1 are in the same cluster, this is handled by
                    # :meth:`Cluster.merge_with()`.
                    v0.r_cluster.merge_with(v1.r_cluster, edge)
                else:
                    # Add vertex to cluster
                    assert v1.is_unused()
                    v0.r_cluster.add_vertex(v1, edge)


class UnionFind(UnionFindNoWeights):
    """Union find with floating-point weights."""

    def _set_weights(self, weights: np.ndarray):
        """Set weights."""
        assert weights.shape == (len(self.graph.edges),)
        increments = 1 / (2 * weights)
        for edge, incr in zip(self.graph.edges, increments):
            edge.set_weight(1.0, incr)


class UnionFindDecoder(erasure_dec.ErasureDecoderInterface):
    """Union find decoder.

    Invokes union find from :class:`UnionFindNoWeights` (or a subclass thereof)
    and the erasure decoder from
    :class:`plaquette.decoders.unionfind_erasure.ErasureDecoder`.
    """

    def __init__(  # noqa: D107
        self,
        uf_cls: Type[UnionFindNoWeights],
        graph: erasure_dec.Graph,
        skip_erasure_decoder=False,
    ):
        self.uf = uf_cls(graph.n_vertices, graph.edges)
        if skip_erasure_decoder:
            self.elib = None
        else:
            self.elib = erasure_dec.ErasureDecoder(graph)

    def decode(self, erasure, syndrome):
        """Decode erasure and syndrome.

        :param erasure:
            Indices of erased edges
        :param syndrome:
            List of syndrome bits for each vertex; ``syndrome[i]`` is for vertex ``i``

        :return:
            List of true/false for each edge; ``corr[i]`` is for edge ``i``
        """
        erasure2 = self.uf.union_find(erasure, syndrome)
        if self.elib is None:
            return erasure2
        else:
            return self.elib.decode(erasure2, syndrome)

    def set_weights(self, weights: Optional[np.ndarray]):
        """Set weights.

        Args:
            weights: Array containing one weight for each edge (optional).
        """
        return self.uf.set_weights(weights)
