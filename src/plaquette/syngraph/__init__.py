# Copyright 2023, It'sQ GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
r"""Syndrome graph (used by decoders).

The syndrome graph describes which stabilizer measurement outcome is toggled by
which possible error. This information is encoded in a graph because this is the format
used by most decoders.
The syndrome graph can also include information on error probabilities
(aka weights), the actual syndrome (stabilizer measurement results) and
decoding results. Further details are discussed in :mod:`plaquette.syngraph`.

In most cases, you can use :mod:`plaquette.decoders` and you do not need to interact
with the syndrome graph.

Implementation
--------------

The syndrome graph as implemented here manages the following information:

* Stabilizer generator measurements (or change thereof) at a time step,
  including the measurement result (vertices in the graph)

  The set of all measurement results is usually called *the syndrome*.

* Possible errors (data qubit and measurement errors; edges in the graph)

* A-priori weights of detectable errors

* Erasure information on qubits (mapped to possible errors)

* Decoding results (a selection of possible errors)

Currently, a syndrome graph can be generated from a code and error data. In the future,
it should also become possible to generate a syndrome graph with weights from a circuit
with errors.

There is the following basic correspondence between vertices/edges in the graph and
other information:

* Graph vertex = Stabilizer generator measurement result at time step (see
  :class:`Vertex`).

* Edge = a possible error (see :class:`Edge`).

  * Data qubit error edge: Corresponds to an error on a data qubit at a time step (see
    :class:`QubitErrorEdge`).

  * Measurement error edge: Corresponds to a flip of a stabilizer generator
    measurement at a time step (see :class:`StabilizerErrorEdge` and additional
    information in :class:`Vertex`).

The basic structure of this module is as follows:

* :class:`SyndromeGraph` implements all high-level functions related to constructing
  the syndrome graph and handling related information.

* :class:`Vertex`, :class:`QubitErrorEdge` and :class:`StabilizerErrorEdge` represent
  vertices and edges within :class:`SyndromeGraph`.

* Connected components within the complete graph are represented by instances of
  :class:`SyndromeGraphComponent`.

  There is a mapping between vertices/edges in a connected component and corresponding
  vertices/edges in the complete graph.

* Edges within :class:`SyndromeGraphComponent` are represented by specifying
  integer indices of involved vertices. This simple format is most suitable for passing
  the graph to different decoders.

Examples:
* For a simple planar code, the syndrome graph contains two connected components:
  One contains X errors and Z stabilizers whereas the other contains Z errors and
  X stabilizers.

* For a repetition code which detects X errors but does not detect Z errors, the
  syndrome graph has one connected component. It contains all X errors and all
  stabilizers.

* Consider a planar code and an error model where all Z errors have zero probability.
  Here, edges of Z errors have infinite weight. Since infinite weight are removed from
  the graph, each X stabilizer has its own connected component, containing only
  one vertex and no edges. If an X stabilizer is toggled while Z errors are excluded
  a priori by setting their probability to zero, the syndrome cannot be decoded
  (unless weights are disabled).

.. todo::

    Integrate or delete the following old material. It is older than the current
    implementation of the syndrome graph.

    Syndrome graph (stabilizers are nodes and edges are detectable errors)

    .. warning::

       The information which follows may be outdated and/or inaccurate.

    .. todo::

       Integrate this docstring into :class:`SyndromeGraph` or
       :mod:`plaquette.syngraph`.

    The following pieces of information are related to the syndrome graph and may
    or may not be stored within the graph:

    * Weights of detectable errors (= edge weights)
    * Results of stabilizer measurements (aka "the syndrome",
      = one bit on each vertex) [Not implemented yet]
    * Information on erased qubits (= a selection of edges) [Not implemented yet]

    A vertex in the syndrome graph corresponds to one stabilizer generator, i.e. to one
    measurement for error detection. An edge in the syndrome graph corresponds to a
    detectable error (currently either X or Z on a single qubit). An edge connects two
    (or more) stabilizers if the corresponding error toggles the corresponding
    stabilizer generators.

    This class represents the syndrome graph for one round of (repeated)
    measurements (usually 2D). The syndrome graph for multiple rounds of
    measurements is constructed within glue code to decoders (for multiple
    rounds, the syndrome graph is usually 3D).

    .. todo::

       The syndrome graph for multiple rounds of measurements should not be
       constructed within decoder interfaces.

    Vertices in the syndrome graph correspond to stabilizer operators.

    Edges in the syndrome graph correspond to possible single-qubit errors (X
    or Z on any qubit).

    An error connects those stabilizers whose measurement outcomes it toggles.
    Within this class, there are 2-edges, 1-edges and 0-edges.

    * A 2-edge connects two stabilizers to each other; it corresponds to an
      error which toggles two stabilizers.

    * A 1-edge connects one stabilizer to nothing; it corresponds to an error
      which toggles only a single stabilizer. A 1-edge is also called a
      "dangling edge".

    * A 0-edge does not connect any stabilizers; it corresponds to an error
      which does not toggle any stabilizer (this happens e.g. in the repetition
      code).

    .. note::

       On 0-edges: If we process the syndrome graph, we ignore 0-edges
       completely because they do not make any difference (a 0-edge does not
       connect to anything, so it does not introduce any new paths). If we
       store the list of all edges in the syndrome graph, we include 0-edges
       because edges may be identified by their position in the list. Omitting
       0-edges from the list of all edges would changes these positions.

       In the repetition code example, there are several undetected errors,
       i.e. errors which do not toggle any stabilizer, i.e. errors represented
       by 0-edges. This would turn the graph into a multigraph since the edge
       which connects no stabilizers exists several times. By leaving out
       0-edges, we avoid turning the syndrome graph into a multigraph (in a
       multigraph, a given edge can exist multiple times).

    .. todo::

       Rather than keeping track of 0-edges, we should consider removing them
       actively if this is possible.

    In conclusion, the syndrome graph as implemented here is a hypergraph where
    edges are restricted to connecting either one or two stabilizers. All
    0-edges are ignored when working with the syndrome graph.

    The attribute :attr:edges contains only edges which correspond to Pauli
    errors on (physical) data qubits. In case of multiple measurement rounds,
    there are additional edges which correspond to errors on measurement
    outcome (toggled outcome). (See todo on measurements above.)

    Weights for decoding are defined on syndrome graph edges. See
    :meth:plaquette.decoders.decoderbase.DecoderInterface.set_syngraph for more
    information on how to supply weights.

    .. todo::

       Eventually, the syndrome graph should be divided into connected
       components. Shreya: Unclear what the pros are.

    .. note::

       On connected components in the syndrome graph:

       Consider the most basic planar code. People often use one syndrome graph
       containing only X errors (and Z stabilizers) and then another syndrome
       graph containing only Z errors (and X stabilizers). Decoding is
       performed separately on both syndrome graphs. In this implementation, we
       only have one syndrome graph which contains everything. Usually, the
       syndrome graph has two connected components (corresponding to the two
       independent syndrome graphs used elsewhere). This simplifies the picture
       e.g. for the XZZX code. In the future, we should determine the two
       connected components (within our current graph) and performing decoding
       separately on each. This is what the above todo refers to.

    .. todo::
        The following additions need to be integrated with the rest of this docstring:

        Let $\{g_i\}_{i = 0,...,n_stabilizers}$ be the stabilizers in a Code. The
        graph contains vertices $\{u_i\}_{i = 0,...,n_stabilizers}$, whose indices are
        in correspondence with the indices of the stabilizers. The $n$-th edge on the
        graph is obtained by applying an $X_n$ ($Z_n$) gate to the $n$-th data qubit,
        the edge is defined by the vertices corresponding to the stabilizers which
        anti-commute with the gate, i.e., $\{u_j, u_l\}$ where $\{X_n, g_j\}=0$ and
        $\{X_n, g_jl\}=0$.

        .. note::

            On general graphs: if one defines a graph consisting of stabilizers
            containing a $Y$ operator (e.g., the XY Surface Code from
            :cite:`higgott_fragile_2022`), the Syndrome Graph will
            have hyper-edges (be them 3-edges or 4-edges).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from dataclasses import field as dfield
from typing import Final, Iterable, Optional, Sequence

import numpy as np

from plaquette import codes
from plaquette.errors import ErrorDataDict
from plaquette.pauli import (
    Tableau,
    commutator_sign,
    multiply,
    single_qubit_pauli_operator,
)
from plaquette.syngraph import weights as weights_


@dataclass(slots=True, kw_only=True)
class Vertex:
    """Syndrome graph vertex (syndrome bit at a time step).

    For a high-level overview, see :mod:`plaquette.syngraph`.

    * One round of measurements: The syndrome bit is given by one measurement
      of a stabilizer generator.
    * Multiple rounds of measurements: The syndrome bit is given by the XOR of two
      consecutive measurements of the same stabilizer generator (at times ``time_step``
      and ``time_step + 1``).

    Multiple time steps are illustrated in the following scheme:

    .. code::

            0     1     2     3     4     5     6     7       Vertex time step
         0     1     2     3     4     5     6     7     8    Edge time step
       -----O-----O-----O-----O-----O-----O-----O-----O-----
         0     0     1     1     0     0     1     0     0    Meas. result
            0     1     0     1     0     1     1     0       XOR of results

    The scheme contains multiple stabilizer generator results at times ``0...8``.
    Syndrome bits are located at times ``0...7``.

    The dangling edges for the first and last time steps are currently not implemented.
    They may be added later on.

    This is a dataclass.

    .. automethod:: __init__
    """

    #: Linear index of the vertex
    vertex_idx: int
    #: Connected component to which the vertex belongs
    connected_component_idx: Optional[int] = None
    #: Adjacent edges (including disabled edges)
    all_edges: list[Edge] = dfield(default_factory=list)
    #: Neighbouring vertices (including neighbours behind disabled edges)
    all_neighbours: list[tuple[Edge, Vertex]] = dfield(default_factory=list)
    #: Linear index of the corresponding stabilizer generator
    stabgen_idx: int
    #: Time step of the stabilizer generator or syndrome bit
    time_step: int
    #: Syndrome bit on this vertex
    syndrome: bool = False

    def add_edge(self, edge: Edge):
        """Add an adjacent edge."""
        assert all(e != edge for e in self.all_edges), "Duplicate edges not supported"
        self.all_edges.append(edge)
        for vertex in edge.vertices:
            if vertex != self:
                self.all_neighbours.append((edge, vertex))

    @property
    def edges(self) -> Iterable[Edge]:
        """Iterate over adjacent edges (only enabled ones)."""
        for edge in self.all_edges:
            if edge.enabled:
                yield edge

    @property
    def neighbours(self) -> Iterable[Vertex]:
        """Iterate over neighbouring vertices (only enabled ones)."""
        for edge, vertex in self.all_neighbours:
            if edge.enabled:
                yield vertex

    def set_connected_component(self, comp_idx: int):
        """Set connected component of vertex and all adjacent edges."""
        assert self.connected_component_idx is None
        self.connected_component_idx = comp_idx
        for edge in self.edges:
            if edge.connected_component_idx is None:
                edge.set_connected_component(comp_idx)
            else:
                assert edge.connected_component_idx == comp_idx

    def set_syndrome(self, syndrome_bit: bool):
        """Set syndrome bit on vertex."""
        self.syndrome = syndrome_bit

    def __repr__(self):  # noqa: D105
        return f"{self.__class__.__name__}({self.vertex_idx=})"


@dataclass(slots=True, kw_only=True)
class Edge:
    """Syndrome graph edge (data qubit error or stab. measurement error).

    For a high-level overview, see :mod:`plaquette.syngraph`.

    An edge can be disabled (see :attr:`enabled`). If an edge is disabled, it remains
    in :class:`SyndromeGraph` but it is not used for finding connected components
    (:class:`SyndromeGraphComponent`).

    This class is not used directly; instead its subclass :class:`QubitErrorEdge`
    and :class:`StabilizerErrorEdge` are used.

    This is a dataclass.

    .. automethod:: __init__
    """

    #: Linear index of the edge
    edge_idx: int
    #: Connected component to which the edge belongs
    connected_component_idx: Optional[int] = None
    #: Vertices connected by the edge (currently zero, one or two)
    vertices: tuple[Vertex, ...]
    #: Weight of the edge
    weight: float = np.nan
    #: Erasure
    erased: bool = False
    #: Decoder result (selected or not selected)
    decoder_result: bool = False
    #: Edges can be disabled (e.g. if they are duplicate or have infinite weight)
    enabled: bool = True

    def __post_init__(self):  # noqa: D105
        for vertex in self.vertices:
            vertex.add_edge(self)

    def set_connected_component(self, comp_idx: int):
        """Set connected component of the edge.

        This method does not change any vertices.
        """
        assert self.connected_component_idx is None
        self.connected_component_idx = comp_idx

    def set_weight(self, weight: float):
        """Set weight of edge."""
        self.weight = weight

    def set_erased(self, erased: bool):
        """Set erasure status of edge."""
        self.erased = erased

    def set_decoder_result(self, result: bool):
        """Set decoder result for an edge."""
        self.decoder_result = result

    def disable(self) -> None:
        """Disable edge."""
        self.enabled = False

    def enable(self) -> None:
        """Enable edge."""
        self.enabled = True

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}({self.edge_idx=})"


@dataclass(slots=True, kw_only=True)
class QubitErrorEdge(Edge):
    """Edge representing a data qubit error (spacelike edge).

    For a high-level overview, see :mod:`plaquette.syngraph`.

    This is a dataclass.

    .. automethod:: __init__
    """

    #: The edge connects two vertices at time `time_step`
    time_step: int
    #: Index of the corresponding error operator (see :attr:`SyndromeGraph.error_ops`)
    error_op_idx: int
    #: Qubit affected by this error
    qubit_idx: int
    #: Whether it is an X or Z error (``"X"`` or ``"Z"``)
    pauli_str: str


@dataclass(slots=True, kw_only=True)
class StabilizerErrorEdge(Edge):
    """Edge representing a stabilizer measurement error (timelike edge).

    For a high-level overview, see :mod:`plaquette.syngraph`.

    For regular (non-dangling) timelike edges, the following applies (see
    :class:`Vertex`):

    The edge connects vertices at times ``time_step - 1`` and ``time_step``.
    The edge can be identified with toggling a stabilizer generator measurement
    at time ``time_step`` (see :class:`Vertex` for details).

    This is a dataclass.

    .. automethod:: __init__
    """

    #: The edge connects vertices at times ``time_step`` and ``time_step + 1``
    time_step: int
    #: Linear index of the corresponding stabilizer generator
    stabgen_idx: int


class SyndromeGraphComponent:
    """Connected component of a syndrome graph.

    For a high-level overview, see :mod:`plaquette.syngraph`.

    .. warning::

       Decoders expect that the vertex and edge structure within an existing
       ``SyndromeGraphComponent`` instance does not change. If they change, decoders
       may crash or return incorrect results.

    .. automethod:: __init__
    """

    #: Number of vertices in this connected component
    #: (vertices are identified by integers `0 ... (n_vertices-1)`)
    n_vertices: int
    #: List of edges (one edge = tuple of one or two vertex indices)
    edges: list[tuple[int, int]]
    #: Whether the component has at least one dangling edge (aka 1-edge)
    has_dangling_edges: bool
    #: Weight of each edge (float array, NaN indicates "no weight")
    edge_weights: np.ndarray
    #: Whether each edge was erased (boolean array)
    edge_erased: np.ndarray
    #: Whether each edge was selected in the decoding result (boolean array)
    edge_decoder_result: np.ndarray
    #: ``True`` if parent graph did not read the decoder results. Set to ``False``
    #: by :class:`SyndromeGraph`.
    decoder_results_new: bool = True
    #: Syndrome bit for each vertex (boolean array)
    syndrome: np.ndarray
    #: Vertex indices in the parent syndrome graph
    parent_vertex_idx: np.ndarray
    #: Edge indices in the parent syndrome graph
    parent_edge_idx: np.ndarray

    def __init__(
        self, vertices: list[Vertex], edges: list[Edge], edge_weights: np.ndarray
    ):
        """Create a new connected component of a syndrome graph.

        Connected components are generally created by :meth:`SyndromeGraph.__init__`.

        Args:
            vertices: List of vertices in the connected component
            edges: List of edges in the connected component
            edge_weights: List of edge weights (same length as ``edges``)
        """
        self.n_vertices = len(vertices)
        # Lookup table: component vertex index --> parent vertex index.
        self.parent_vertex_idx = np.array([v.vertex_idx for v in vertices])
        self.parent_edge_idx = np.array([e.edge_idx for e in edges])
        # Reverse lookup table: Parent vertex index --> component vertex index.
        rev_vertex_idx = -np.ones([self.parent_vertex_idx.max() + 1], dtype=int)
        rev_vertex_idx[self.parent_vertex_idx] = np.arange(self.n_vertices)
        self.edges = [
            tuple(rev_vertex_idx[v.vertex_idx] for v in edge.vertices)  # type:ignore
            for edge in edges
        ]
        self.has_dangling_edges = any(len(edge) == 1 for edge in self.edges)
        assert all(vertex_idx != -1 for edge in self.edges for vertex_idx in edge)
        self.set_edge_weights(edge_weights)
        self.set_syndrome(None)
        self.set_edge_decoder_results(np.zeros([len(self.edges)], dtype=bool))

    def set_edge_weights(self, weights: np.ndarray):
        """Set edge weights from array."""
        assert weights.dtype == float
        assert weights.shape == (len(self.edges),)
        self.edge_weights = weights

    def set_erasure(self, edge_erased: np.ndarray):
        """Set erasure from array."""
        assert edge_erased.dtype == bool
        assert edge_erased.shape == (len(self.edges),)
        self.edge_erased = edge_erased

    def set_syndrome(self, syndrome: Optional[np.ndarray]):
        """Set syndrome bits from array."""
        if syndrome is None:
            syndrome = np.zeros([self.n_vertices], dtype=bool)
        assert syndrome.dtype == bool
        assert syndrome.shape == (self.n_vertices,)
        self.syndrome = syndrome

    def set_edge_decoder_results(self, result: np.ndarray):
        """Set decoder results for all edges.

        The decoder returns a selection of edges.

        Args:
            result: Boolean array with one entry for each edge. The entry specifies
                whether the edge is included in the decoder's selection.

        .. warning::

           If you set :attr:`edge_decoder_result` directly instead of via this method,
           you also have to set :attr:`decoder_results_new` to ``True``. Otherwise,
           incorrect decoder results may be returned by methods of
           :class:`SyndromeGraph`.
        """
        assert result.dtype == bool
        assert result.shape == (len(self.edges),)
        self.edge_decoder_result = result
        self.decoder_results_new = True

    def is_weighted(self, support_mixed: bool = False) -> bool:
        """Determine whether there are any weights.

        Args:
            support_mixed: If ``False`` and some edges have weights while others do
                not, then a ``ValueError`` is raised.
        """
        if np.isnan(self.edge_weights).all():
            return False
        if (not support_mixed) and np.isnan(self.edge_weights).any():
            raise ValueError(
                "Some edges have weights while other edges do not have weights. "
                "This is not supported yet."
            )
        return True


class SyndromeGraph:
    """Syndrome graph.

    For a high-level overview, see :mod:`plaquette.syngraph`.

    .. warning::

       Decoders expect that the vertex and edge structure within an existing
       ``SyndromeGraphComponent`` instance does not change. If they change, decoders
       may crash or return incorrect results.

    .. automethod:: __init__
    """

    #: Number of rounds of stabilizer measurements
    n_rounds: int
    #: Number of physical data qubits
    n_qubits: int
    #: Number of stabilizer generators
    n_stabgens: int
    #: List of all error operators
    error_ops: Sequence[Tableau]
    #: Vertices in the graph (indexed by :attr:`Vertex.vertex_idx`)
    vertices: list[Vertex]
    #: Vertices, at given time step (indexed by ``[time_step][stabgen_idx]``)
    vertices_at_time: list[list[Vertex]]
    #: Edges in the graph (indexed by :attr:`Edge.edge_idx`)
    edges: list[Edge]
    #: Connected components in the graph (indexed by ``connected_component_idx``)
    components: list[SyndromeGraphComponent]
    #: Connected components which include at least one edge. The entries are not
    #: indexed in a particular way.
    components_with_edges: list[SyndromeGraphComponent]

    #: Whether duplicate edges should be disabled.
    _disable_duplicate_edges: Final[bool] = True
    #: Whether infinite-weight edges should be disabled. DO NOT CHANGE unless you know
    #: what you are doing.
    _disable_infweight_edges: Final[bool] = True
    #: Whether object initialization is complete. DO NOT CHANGE THIS.
    _init_complete: bool = False
    #: Edge weights in an array. Used by :meth:`_set_edge_weights_on_components`.
    _edge_weights: np.ndarray

    def __init__(
        self, n_rounds: int, stabgens: Sequence[Tableau], weights: Optional[dict]
    ):
        """Create a syndrome graph.

        Args:
            n_rounds: Number of rounds of stabilizer measurements
            stabgens: The stabilizer generators
            weights: Edge weights, see :meth:`set_edge_weights` for details.

        To create a syndrome graph from a code object, use :meth:`from_code`.

        .. note::
            This class is currently restricted to a syndrome graph whose edges
            correspond to single-qubit X and Z errors.
        """
        self.n_rounds = n_rounds
        self.n_qubits = stabgens[0].size // 2
        self.n_stabgens = len(stabgens)
        self.vertices = []
        self.vertices_at_time = []
        self.edges = []
        self.components = []
        self._init_error_ops()
        self._init_vertices()
        self._init_qubit_error_edges(stabgens)
        self._init_measurement_error_edges()
        self._edge_weights = np.nan * np.zeros([len(self.edges)])
        if weights is not None:
            self.set_edge_weights(**weights)
        if self._disable_infweight_edges:
            self._check_edges_infweight()
        if self._disable_duplicate_edges:
            self._check_edges_duplicate()
        self._init_conn_components()
        self._set_edge_weights_on_components()
        self._init_complete = True

    @classmethod
    def from_code(cls, code: codes.LatticeCode, weights: Optional[dict]):
        """Create syndrome graph from code.

        Args:
            code: The code
            weights: Edge weights, see :meth:`set_edge_weights` for details.
        """
        return cls(code.n_rounds, code.stabilisers, weights)

    def _init_error_ops(self) -> None:
        """Initialize error operators used for defining graph edges.

        Currently, this is restricted to single-qubit Pauli X and Z on each qubit.
        """
        self.error_ops = [
            single_qubit_pauli_operator(o, i, self.n_qubits)
            for o in "XZ"
            for i in range(self.n_qubits)
        ]

    def _init_vertices(self) -> None:
        """Add vertices."""
        # It is not necessary that the order in which vertices are created here matches
        # the order of vertices in the syndrome - see set_syndrome() below.
        # (set_syndrome() uses vertex.time_step and vertex.stabgen_idx instead of
        # vertex.vertex_idx, such that the order of vertices can be arbitrary.)
        for time_step in range(self.n_rounds):
            self.vertices_at_time.append([])
            for idx in range(self.n_stabgens):
                vertex = Vertex(
                    vertex_idx=len(self.vertices), time_step=time_step, stabgen_idx=idx
                )
                self.vertices.append(vertex)
                self.vertices_at_time[-1].append(vertex)

    def _init_qubit_error_edges(self, stabgens: Sequence[Tableau]):
        """Add data qubit error edges."""
        comm_signs = commutator_sign(self.error_ops, stabgens)
        for oplist in self.error_ops:
            op = oplist[:-1]
            op = np.where(op)[0]
            if len(op) == 2:
                if op[1] - op[0] != self.n_qubits:
                    raise ValueError("Only single-qubit errors are supported")
            elif len(op) > 2:
                raise ValueError("Only single-qubit errors are supported")
        for time_step in range(self.n_rounds):
            for error_op_idx, stabilizers in enumerate(comm_signs):
                stabgen_indices = tuple(np.where(stabilizers)[0])
                if len(stabgen_indices) not in (0, 1, 2):
                    # 2-edge = edge which connects two stabilizers.
                    # n-edge = edge which connects n stabilizers (a hypergraph!).
                    raise ValueError(
                        "The syndrome hypergraph supports only 0-edges, 1-edges and "
                        "2-edges."
                    )
                edge_vertices = tuple(
                    self.vertices_at_time[time_step][idx] for idx in stabgen_indices
                )
                oplist = self.error_ops[error_op_idx][:-1]
                x_op = np.where(oplist[: self.n_qubits])[0]
                z_op = np.where(oplist[self.n_qubits :])[0]
                pauli_str = ""
                if len(x_op) == 1:
                    qubit_idx = x_op
                    pauli_str += "X"
                if len(z_op) == 1:
                    qubit_idx = z_op
                    pauli_str += "Z"
                if len(pauli_str) == 2:
                    pauli_str = "Y"
                self.edges.append(
                    QubitErrorEdge(
                        edge_idx=len(self.edges),
                        vertices=edge_vertices,
                        time_step=time_step,
                        error_op_idx=error_op_idx,
                        qubit_idx=qubit_idx,
                        pauli_str=pauli_str,
                    )
                )

    def _init_measurement_error_edges(self) -> None:
        """Add stabilizer measurement error edges."""
        for time_step in range(1, self.n_rounds):
            for stabgen_idx in range(self.n_stabgens):
                edge_vertices = (
                    self.vertices_at_time[time_step - 1][stabgen_idx],
                    self.vertices_at_time[time_step][stabgen_idx],
                )
                self.edges.append(
                    StabilizerErrorEdge(
                        edge_idx=len(self.edges),
                        vertices=edge_vertices,
                        time_step=time_step,
                        stabgen_idx=stabgen_idx,
                    )
                )

    def _check_edges_infweight(self) -> None:
        """Disable any edges with infinite weight.

        .. note::

           This method does not enable any edges.
        """
        for edge in self.edges:
            if np.isinf(edge.weight):
                edge.disable()

    def _check_edges_duplicate(self) -> None:
        """Disable any duplicate edges, leaving the smallest-weight edge enabled.

        .. note::

           This method does not enable any edges.
        """
        for vertex in self.vertices:
            edge_dict: dict[tuple[int, ...], list[Edge]] = {}
            for edge in vertex.all_edges:
                key = tuple(v.vertex_idx for v in edge.vertices)
                try:
                    edge_dict[key].append(edge)
                except KeyError:
                    edge_dict[key] = [edge]
                for _, edges in edge_dict.items():
                    if len(edges) > 1:
                        self._select_lightest_edge(edges)

    @staticmethod
    def _select_lightest_edge(edges: Sequence[Edge]):
        """Disable all but the lightest edge from a sequence.

        If all edges have weight NaN, disable all edges except the first one.

        If some edges have NaN weigth and other edges have non-NaN weigth, raise an
        error.
        """
        weights = [edge.weight for edge in edges]
        if np.isnan(weights).any():
            if not np.isnan(weights).all():
                raise ValueError("Cannot mix NaN weights with non-NaN weights")
            selected_edge = edges[0]
        else:
            min_weight = min(weights)
            selected_edge = next(edge for edge in edges if edge.weight == min_weight)
        for edge in edges:
            if edge is not selected_edge:
                edge.disable()

    def _init_conn_components(self) -> None:
        """Find connected components."""
        next_vertex_idx = 0
        comp_idx = 0
        while next_vertex_idx < len(self.vertices):
            vertex = self.vertices[next_vertex_idx]
            if vertex.connected_component_idx is not None:
                next_vertex_idx += 1
                continue
            self._find_connected_component(vertex, comp_idx)
            comp_idx += 1
        self._init_components_objects(comp_idx)

    def _init_components_objects(self, n_components: int):
        """Construct a SyndromeGraphComponent object for each connected component."""
        comp_vertices: list[list[Vertex]] = [[] for _ in range(n_components)]
        comp_edges: list[list[Edge]] = [[] for _ in range(n_components)]
        for vertex in self.vertices:
            assert vertex.connected_component_idx is not None
            comp_vertices[vertex.connected_component_idx].append(vertex)
        for edge in self.edges:
            if edge.connected_component_idx is not None:
                comp_edges[edge.connected_component_idx].append(edge)
        for vertices, edges in zip(comp_vertices, comp_edges):
            # Start without weights
            weights = np.nan * np.zeros([len(edges)])
            self.components.append(SyndromeGraphComponent(vertices, edges, weights))
        self.components_with_edges = [
            comp for comp in self.components if len(comp.edges) > 0
        ]

    @staticmethod
    def _find_connected_component(vertex: Vertex, comp_idx: int):
        """Find connected component of given vertex."""
        vertex.set_connected_component(comp_idx)
        todo = [vertex]
        while todo:
            vertex = todo.pop()
            for neigh in vertex.neighbours:
                if neigh.connected_component_idx is None:
                    neigh.set_connected_component(comp_idx)
                    todo.append(neigh)
                else:
                    assert neigh.connected_component_idx == comp_idx

    def set_edge_weights(
        self,
        pauli_x: float | np.ndarray,
        pauli_z: float | np.ndarray,
        measurement: float | np.ndarray,
    ):
        """Set edge weights.

        Args:
            pauli_x: Weight for Pauli X error. Single number or array with one entry
                for each data qubit.
            pauli_z: Weight for Pauli Z error. Single number or array with one entry
                for each data qubit.
            measurement: Weight for measurement errors. Single number or array with
                one entry for each stabilizer generator.

        Pass ``np.nan`` for all arguments to disable weights completely.

        .. note::

           Currently, weights cannot be updated after constructing the
           :class:`SyndromeGraph` object. This may change in the future. See the source
           for details.

           To set weights, pass them to :meth:`the constructor <__init__>`.
        """
        if self._disable_infweight_edges and self._init_complete:
            # We could check that structure does not change or we could handle
            # structural changes. At the moment, neither is implemented.
            raise ValueError(
                "Cannot change weights after init if inf-weight edges are disabled"
            )
        pauli_x = np.ones([self.n_qubits]) * pauli_x
        pauli_z = np.ones([self.n_qubits]) * pauli_z
        measurement = np.ones([self.n_stabgens]) * measurement
        for edge in self.edges:
            if isinstance(edge, QubitErrorEdge):
                if edge.pauli_str == "X":
                    edge.set_weight(pauli_x[edge.qubit_idx])
                elif edge.pauli_str == "Z":
                    edge.set_weight(pauli_z[edge.qubit_idx])
                else:
                    raise ValueError(f"Edge has unsupported {edge.pauli_str=}.")
            elif isinstance(edge, StabilizerErrorEdge):
                edge.set_weight(measurement[edge.stabgen_idx])
            else:
                raise ValueError(f"Unsupported edge type: {edge.__class__.__name__}")
            self._edge_weights[edge.edge_idx] = edge.weight
        if self._init_complete:
            self._set_edge_weights_on_components()

    def _set_edge_weights_on_components(self) -> None:
        """Set edge weights on connected components."""
        for comp in self.components_with_edges:
            comp.set_edge_weights(self._edge_weights[comp.parent_edge_idx])

    def set_erasure(self, qubit_erased: Optional[np.ndarray]):
        """Set edge erasure from qubit erasure.

        Args:
            qubit_erased:
                Boolean array of shape ``(n_rounds, n_qubits)``. Each entry specifies
                whether the given data qubit was erased in a particular round.

        Currently, only edges for data qubit errors can be marked as erased. It is not
        possible to mark edges for stabilizer generator measurement errors as erased.
        """
        if qubit_erased is None:
            qubit_erased = np.zeros([self.n_rounds, self.n_qubits], dtype=bool)
        elif qubit_erased.dtype == "u1":
            qubit_erased = qubit_erased.astype(bool)
        assert qubit_erased.dtype == bool
        assert qubit_erased.shape == (self.n_rounds, self.n_qubits)
        edge_erased = np.zeros([len(self.edges)], dtype=bool)
        for edge in self.edges:
            if isinstance(edge, QubitErrorEdge):
                edge.set_erased(qubit_erased[edge.time_step, edge.qubit_idx])
                edge_erased[edge.edge_idx] = edge.erased
        for comp in self.components_with_edges:
            comp.set_erasure(edge_erased[comp.parent_edge_idx])

    def set_syndrome(self, syndrome: Optional[np.ndarray]):
        """Set syndrome (stabilizer generator measurement results).

        See :class:`Vertex` for details on the syndrome with multiple rounds of
        measurements.

        Args:
            syndrome:
                Boolean array of shape ``(n_rounds, n_stabgens)``. Order of entries
                must be from first to last round and from first to last stabilizer
                generator.
        """
        if syndrome is None:
            syndrome = np.zeros([self.n_rounds, self.n_stabgens], dtype=bool)
        elif syndrome.dtype == "u1":
            syndrome = syndrome.astype(bool)
        assert syndrome.dtype == bool
        assert syndrome.shape == (self.n_rounds, self.n_stabgens)
        v_syndrome = np.zeros([len(self.vertices)], dtype=bool)
        syndrome_done = np.zeros([self.n_rounds, self.n_stabgens], dtype=bool)
        for vertex in self.vertices:
            pos = (vertex.time_step, vertex.stabgen_idx)
            assert not syndrome_done[pos]
            vertex.set_syndrome(syndrome[pos])
            syndrome_done[pos] = True
            v_syndrome[vertex.vertex_idx] = vertex.syndrome
        assert syndrome_done.all(), "Not all syndrome bits were used"
        # Set syndrome on each component
        for comp in self.components:
            comp.set_syndrome(v_syndrome[comp.parent_vertex_idx])

    def combine_decoding_results(self, results: list[np.ndarray]) -> np.ndarray:
        """Combine decoding results from connected components.

        The decoding result is an edge selection on a connected component or on
        the complete graph, respectively. It is passed as boolean array with one entry
        for each edge (in the component or in the complete graph).

        Args:
            results: Edge selection on each connected component (list with one
                ``ndarray`` per component with edges; order corresponds to
                :attr:`components_with_edges`).
        """
        assert len(results) == len(self.components_with_edges)
        combined_result = np.zeros([len(self.edges)], dtype=bool)
        for comp, comp_result in zip(self.components_with_edges, results):
            assert comp_result.dtype == bool
            assert comp_result.shape == comp.parent_edge_idx.shape
            combined_result[comp.parent_edge_idx] = comp_result
        return combined_result

    def update_decoder_results(self):
        """Update decoder results in edges with new results from components."""
        for comp in self.components_with_edges:
            for idx, result in zip(comp.parent_edge_idx, comp.edge_decoder_result):
                self.edges[idx].set_decoder_result(result)
            comp.decoder_results_new = False

    def result_as_error_op_selection(self) -> np.ndarray:
        """Determine which error operators are selected by the current result.

        Returns:
            Boolean array with one entry for each operator in :attr:`error_ops`.
        """
        if any(comp.decoder_results_new for comp in self.components_with_edges):
            # In principle, we could make two changes:
            # - Call update_decoder_results() here if necessary
            # - Update only those components which have `decoder_results_new == True`.
            # We could change it and it should work - but I like the explicit solution
            # better for now.
            raise ValueError("You forgot to call update_decoder_results()")
        error_op_selection = np.zeros([len(self.error_ops)], dtype=bool)
        for edge in self.edges:
            if isinstance(edge, QubitErrorEdge):
                error_op_selection[edge.error_op_idx] ^= edge.decoder_result
        return error_op_selection

    def result_as_pauli_frame(self) -> Tableau:
        """Convert the current decoding result to a Pauli frame correction.

        Returns:
            Decoding result as Pauli frame.
        """
        selection = np.where(self.result_as_error_op_selection())[0]
        operators = [self.error_ops[i] for i in selection]
        if not operators:
            return np.zeros(2 * self.n_qubits + 1, dtype="u1")
        return multiply(*operators)[0]  # discard complex phase

    def get_statistics(self) -> dict[str, int | list[int]]:
        """Get statistics on syndrome graph structure."""
        return {
            "n_vertices": len(self.vertices),
            "n_edges": len(self.edges),
            "n_2_edges": sum(len(edge.vertices) == 2 for edge in self.edges),
            "n_1_edges": sum(len(edge.vertices) == 1 for edge in self.edges),
            "n_0_edges": sum(len(edge.vertices) == 0 for edge in self.edges),
            "n_edges_qubit": sum(
                isinstance(edge, QubitErrorEdge) for edge in self.edges
            ),
            "n_edges_stab": sum(
                isinstance(edge, StabilizerErrorEdge) for edge in self.edges
            ),
            "n_edges_disabled": sum(not edge.enabled for edge in self.edges),
            "n_vertices_in_comp": [
                comp.n_vertices for comp in self.components_with_edges
            ],
            "n_edges_in_comp": [len(comp.edges) for comp in self.components_with_edges],
            "n_comp": len(self.components),
            "n_comp_nontrivial": len(self.components_with_edges),
            "n_comp_trivial": len(self.components) - len(self.components_with_edges),
        }

    def format_statistics(self) -> str:
        """Format statistics of syndrome graph structure as string."""
        stat = self.get_statistics()
        return (
            "Vertices:            {n_vertices}\n"
            "Edges:               {n_edges}\n"
            "Disabled edges:      {n_edges_disabled}\n"
            "\n"
            "2-edges:             {n_2_edges}\n"
            "1-edges (dangling):  {n_1_edges}\n"
            "0-edges (empty):     {n_0_edges}\n"
            "\n"
            "Qubit edges:         {n_edges_qubit}\n"
            "Stabil. edges:       {n_edges_stab}\n"
            "\n"
            "Vertices in non-triv. components:  {n_vertices_in_comp}\n"
            "Edges in non-triv. components:     {n_edges_in_comp}\n"
            "\n"
            "Connected components:    {n_comp}\n"
            "Non-trivial components:  {n_comp_nontrivial}\n"
            "Trivial components:      {n_comp_trivial}\n"
        ).format(**stat)


class SyndromeGraphUpdate(enum.IntEnum):
    """Different levels of changes in the syndrome graph.

    This is an enum.

    Used to tell a decoder what was changed (e.g. weights) before an individual
    decoding run.
    """

    #: Syndrome graph and weights unchanged
    Nothing = 0
    #: Weights changed, syndrome graph structure unchanged
    Weights = 1
    #: Syndrome graph structure and weights changed
    GraphAndWeights = 2


class SyndromeGraphManager:
    """Manage the interaction between a decoder and a syndrome graph.

    If syndrome graph weights and/or structure depend on erasure information (as is
    the case for super-stabilizers), a decoder can request updated syndrome
    graph weights and/or structure from this class.

    This class implements a constant syndrome graph structure and weights (i.e. it
    does not implement super-stabilizers). A subclass of this class could implement
    super-stabilizers by overriding :meth:`update_for_decoding`.

    .. automethod:: __init__
    """

    #: Code for which syndrome graph is generated
    code: codes.LatticeCode
    #: Error data for which syndrome graph weights are computed
    errordata: Optional[ErrorDataDict]
    #: Computed syndrome graph (includes optional weights)
    sgraph: SyndromeGraph

    def __init__(
        self,
        code: codes.LatticeCode,
        errordata: Optional[ErrorDataDict],
        *,
        weighted: Optional[bool] = None,
        weights: Optional[dict] = None,
    ):
        """Create a new instance.

        Args:
            code: the code from which to derive the decoding graph.
            errordata: information about weights and errors.

        Keyword Args:
            weighted: whether this syndrome should use weights or not.
            weights: user-supplied weights, if ``errordata`` was not specified.
        """
        self.code = code
        self.errordata = errordata
        if errordata is not None and weights is not None:
            raise ValueError("Cannot specify both `errordata` and `weights`")
        if weighted is None:
            weighted = errordata is not None or weights is not None
        if weighted:
            if weights is None:
                if errordata is None:
                    raise ValueError("weighted=True requires that you supply errordata")
                comp = weights_.WeightComputation(code, errordata)
                weights = comp.get_edge_weights()
        else:
            weights = None
        self.sgraph = SyndromeGraph.from_code(code, weights)

    def update_for_decoding(
        self, erasure: Optional[np.ndarray], syndrome: np.ndarray
    ) -> SyndromeGraphUpdate:
        """Updates for decoder state before decoding.

        Can be overridden by subclass. E.g. for implementing
        super-stabilizers one can indicate new weights and/or a new graph.

        Returns:
            One of :class:`SyndromeGraphUpdate`'s values
        """
        return SyndromeGraphUpdate.Nothing
