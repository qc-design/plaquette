# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Quantum error-correcting codes."""

import typing as t
from dataclasses import InitVar, dataclass, field
from enum import IntEnum
from functools import cached_property

import numpy as np
import rustworkx as rx
from rustworkx.visualization import mpl_draw

from plaquette import pauli
from plaquette.codes import graph
from plaquette.pauli import Factor, PauliDict, Tableau, count_qubits, sort_operators_ref
from plaquette_graph import MultiGraph, SparseGraph


class QubitType(IntEnum):
    """Type, or role, of a particular qubit."""

    data = 0
    stabilizer = 1
    gauge = 2
    flag = 3
    virtual = 4

    def __str__(self) -> str:  # noqa: D105
        return self.name


@dataclass
class EdgeMetadata:
    """Metadata associated to a single edge.

    Mostly used to generate a circuit from a graph.
    """

    type: Factor


@dataclass
class NodeMetadata:
    """Metadata associate with a qubit.

    Mostly used to help with visualization.
    """

    type: QubitType
    coords: t.Optional[list[int]] = None


class AnnotatedSparseGraph(SparseGraph):
    """A sparse graph supporting annotation/metadata of its nodes and edges."""

    def __init__(
        self,
        nodes_data: t.MutableSequence[t.Optional[NodeMetadata]],
        edges: t.Sequence[tuple[int, int]],
        edges_data: t.Sequence[t.Optional[EdgeMetadata]],
    ):
        """Create a new sparse graph with additional info attached.

        Args:
            nodes_data: A sequence of information about the nodes. Its length must
                match the number of given nodes (extracted from the edges).
            edges: a tuple of ints representing the edges. Directly passed to
                :class:`SparseGraph`.
            edges_data: A sequence of information about the edges. Its length must
                match the number of given edges.
        """
        self._n_nodes = np.max(edges) + 1
        """Number of nodes in the graph, including disconnected nodes."""

        if self._n_nodes != len(nodes_data):
            raise ValueError(
                f"Number of nodes ({self._n_nodes}) and metadata elements "
                f"({len(nodes_data)}) must match."
            )

        self._nodes = np.unique(edges)
        """Nodes involved in edges."""

        if len(edges_data) != len(edges):
            raise ValueError(
                f"Number of edges ({len(edges)}) and metadata elements "
                f"({len(edges_data)}) must match."
            )

        super().__init__(self._n_nodes, edges)
        self.nodes_data = nodes_data
        """Nodes metadata.

        This sequence is indexed in lock-step with the nodes themselves. The
        first element is the metadata of the first node, and so on.
        """
        self.edges_data = edges_data
        """Edges metadata.

        This sequence is indexed in lock-step with the edges themselves. The
        first element is the metadata of the first edge, and so on.
        """

        self._rxg = rx.PyGraph()
        self._rxg.add_nodes_from([i for i in range(self._n_nodes)])
        self._rxg.add_edges_from_no_data(edges)

    def to_pydantic_model(self) -> graph.BaseGraph:
        """Generate a Pydantic model based on current code properties.

        Returns:
            a suitable model for export as JSON and for consumption by
            the 3D visualizer.
        """
        nodes = list()
        for q in range(self.get_num_vertices()):
            if (nd := self.nodes_data[q]) is not None and nd.coords is not None:
                x, y, z = nd.coords
                nodes.append(
                    graph.Node(
                        pos=graph.Position(x=x, y=y, z=z),
                        type=nd.type.name,
                    )
                )
        edges = list()
        for e in range(self.get_num_edges()):
            if (ed := self.edges_data[e]) is not None:
                a, b = self.get_vertices_connected_by_edge(e)
                edges.append(graph.Edge(a=a, b=b, type=ed.type.name))
        return graph.BaseGraph(nodes=nodes, edges=edges)


class AnnotatedMultiGraph(MultiGraph):
    """A multigraph supporting annotation/metadata of its nodes and edges."""

    def __init__(
        self,
        edges: t.Sequence[tuple[int, int]],
        nodes_data: t.MutableSequence[t.Optional[NodeMetadata]],
        edges_data: t.MutableSequence[EdgeMetadata],
    ):
        """Create a new multigraph with additional info attached.

        Args:
            edges: a tuple of ints representing the edges. Directly passed to
                :class:`MultiGraph`.
            nodes_data: A sequence of information about the nodes. Its length must
                match the number of given nodes (extracted from the edges).
            edges_data: A sequence of information about the edges. Its length must
                match the number of given edges.
        """
        self._n_nodes = np.max(edges) + 1
        """Number of nodes in the graph, including disconnected nodes."""

        if self._n_nodes != len(nodes_data):
            raise ValueError(
                f"Number of nodes ({self._n_nodes}) and metadata elements "
                f"({len(nodes_data)}) must match."
            )

        self._nodes = np.unique(edges)
        """Nodes involved in edges."""

        if len(edges_data) != len(edges):
            raise ValueError(
                f"Number of edges ({len(edges)}) and metadata elements "
                f"({len(edges_data)}) must match."
            )

        weights = [edge.type.value for edge in edges_data]
        super().__init__(edges, weights)
        self.nodes_data = nodes_data
        """Nodes metadata.

        This sequence is indexed in lock-step with the nodes themselves. The
        first element is the metadata of the first node, and so on.
        """
        self.edges_data = edges_data
        """Edges metadata.

        This sequence is indexed in lock-step with the edges themselves. The
        first element is the metadata of the first edge, and so on.
        """

    @property
    def num_nodes(self):
        """Number of nodes in the embedded graph."""
        return self._n_nodes

    def get_node_data(self, idx: int) -> t.Optional[NodeMetadata]:
        """Get node data at given index.

        Args:
            idx: The index of the node.

        Returns:
            The :class:`~NodeMetadata` object corresponding to the node.
        """
        if idx >= self._n_nodes:
            # sanity check for the index.
            raise IndexError(
                f"Index out of bounds. Must be between 0 and {self._n_nodes-1}"
            )

        return self.nodes_data[idx]


@dataclass(kw_only=True)
class Code:
    """Class to represent subsystem codes.

    This class has no lattice associated with it. This contains simply a list of
    stabilizers, gauges, and logical operators.

    The generated code and associated graph structures are immutable: if you need a
    slightly modified version of a code you need to make a new one from scratch.
    """

    stabilizers: t.Sequence[Tableau]
    """The stabiliser generators of the code.

    Each stabiliser has length equal to the number of data qubits in the non sparse
    representation.
    """
    logical_ops: t.Sequence[Tableau]
    """The logical operators of the code.

    Each logical operator has length equal to the number of data qubits in the non
    sparse representation. This only stores a pair of logical operators per logical
    qubit. If not user provided, the default only will be the least weight logical
    operators respectively.

    Logical operators need to be ordered in the following way: for each logical
    qubit, first list all logical :math:`X` operators, then the :math:`Z`.
    """
    gauge_ops: t.Sequence[Tableau] | None = None
    """The gauge generators of the code.

    Each gauge operators has length equal to the number of data qubits in the non
    sparse representation. If constructing stabilizers from gauges, it should at least
    be a minimum generating set of the gauge group.
    """
    factorized_checks: list[list[int]] = field(default_factory=list)
    """The factors of the stabilizers that are measured.

    In some cases, stabilizers can be factorized into gauge operators and those can
    be measured to infer the measurement outcome of the stabilizer checks. Here,
    we keep track of factors. The integer refer to indexes of factors in
    :attr:`SubsystemCodes.gauge_ops`. If you dont want it to be factorized, make an
    empty
    list.
    """
    compact: bool = False
    """Flag to determine where multiple measured ops can share the same ancilla.

    If ``True``, then each ancillas can support more than one measurement per round.
    If ``False``, then each ancilla can only support one measurement per round.
    Defaults to ``False``.
    """
    ancilla_supports: list[list[tuple[int, int] | int]] = field(init=False)
    """The checks that each ancilla supports.

    Here, by support we mean the measurement of the check is mediated through an
    ancilla. The outermost list cycles through the ancillas. For each ancilla,
    the checks are supported by it are kept track. We uniquely identify each check
    by the two dimensional index corresponding to it in
    :attr:`SubsystemCodes.factorized_checks`.
    """
    _distance: list[int] = field(default_factory=list)
    """The distance of the code.

    The list stores the distances as X and Z distances, when possible. Otherwise it
    just stores one distance as list of length one. When there are multiple logical
    qubits involved the list has the following ordering: ``[X1, X2, ...., Z1, X2
    ...]`` where ``X1, Z1`` represent the X and Z distances of the first logical qubit.
    """
    _num_gauge_gens: int = field(init=False)
    """The number of gauge generators of the code."""
    _tanner_graph: AnnotatedSparseGraph = field(init=False)
    """The Tanner Graph defining the code."""
    _embedded_graph: AnnotatedMultiGraph = field(init=False)

    measure_gauges: InitVar[bool] = False
    """Bool to determine whether to measure gauges or stabilisers. Defaults to False."""
    dist: InitVar[list[int]] = -1
    """Init variable to bypass distance calculation.

    Likely used when the logical provided by the user is not smallest weight Pauli
    string. See :meth:`Code.make_shor` for an example.
    """

    def __post_init__(self, measure_gauges=False, dist=-1):
        """Construct the graphs from the given user data and fill properties."""
        # TODO: the tanner graphs needs the stabilisers, so they need to be set
        #  ABOVE this line
        self.stabilizers = sort_operators_ref(self.stabilizers)
        if self.gauge_ops is not None:
            self.gauge_ops = sort_operators_ref(self.gauge_ops)
        if not self.factorized_checks:
            self.factorized_checks = self._generate_factorized_checks(measure_gauges)
        if dist == -1:
            self._distance = self._calculate_distance()
        else:
            self._distance = dist
        self._generate_ancilla_supports(compact=self.compact)
        self._generate_embedded_graph()
        self._generate_tanner_graph()

        # TODO: replace this and .plot() with a more interactive visualizer
        self._rxg = rx.PyGraph()
        self._rxg.add_nodes_from(
            [(i, self.tanner_graph.nodes_data[i]) for i in range(self.num_qubits)]
        )
        self._rxg.add_edges_from(
            [
                (
                    *self.tanner_graph.get_vertices_connected_by_edge(i),
                    self._tanner_graph.edges_data[i],
                )
                for i in range(self._tanner_graph.get_num_edges())
            ]
        )

    def __repr__(self):
        """Make dataclass representation more straightforward."""
        return f"Code<[{list(self.code_parameters)}]>"

    def plot_tanner(self):
        """Plot a simple visualization of this code.

        Notes:
            This is only for debugging purposes and **must be removed before
            merging this branch**. It assumes that data qubits can be laid out
            in a square grid, and the ancillas are placed in the center of mass
            of the positions of the data qubits which participate in the
            stabilizer the ancilla is responsible for.
        """
        # Now we place stuff in 2D

        node_colors = [
            "red" if i < self.num_data_qubits else "black"
            for i in range(self.num_qubits)
        ]
        edge_colors = [
            "green"
            if self.tanner_graph.edges_data[i].type == pauli.Factor.X
            else "blue"
            for i in range(self.tanner_graph.get_num_edges())
        ]
        f = mpl_draw(
            self._rxg,
            with_labels=True,
            # pos={i: n.coords for i, n in enumerate(self.tanner_graph.nodes_data)},
            node_color=node_colors,
            font_color="white",
            edge_color=edge_colors,
        )
        f.show()

    @property
    def num_qubits(self) -> int:
        """Number of physical qubits in the code."""
        return self._embedded_graph.num_nodes

    @cached_property
    def num_data_qubits(self) -> int:
        """Number of data qubits in the code."""
        return pauli.count_qubits(self.stabilizers[0])[0]

    @cached_property
    def num_stabilizers(self) -> int:
        """The number of stabilizer generators of the code.

        By default, we always assume that it's the minimum generating set and the
        stabilizers are independent of each other.

        Notes:
            Using a cached property because it is a possible variable that will be
            accessed multiple times during calculations.
        """
        return len(self.stabilizers)

    @cached_property
    def num_logical_qubits(self) -> int:
        """Number of logical qubits in the code."""
        if self.gauge_ops is None:
            return self.num_data_qubits - self.num_stabilizers
        else:
            return (
                self.num_data_qubits
                - (
                    np.linalg.matrix_rank(self.stabilizers)
                    + np.linalg.matrix_rank(self.gauge_ops)
                )
                // 2
            )

    @property
    def num_gauge_qubits(self) -> int:
        """Number of the gauge qubits in the code.

        Notes:
            rank H = n - k - g
            rank G = n - k + g
            Pryadko 2019, eq 2

        Returns:
            The number of gauge qubits in the code.

        Todo:
            Fix reference for the Pryadko Paper.
        """
        if self.gauge_ops is None:
            return 0
        else:
            return (
                np.linalg.matrix_rank(self.gauge_ops)
                - np.linalg.matrix_rank(self.stabilizers)
            ) // 2

    def _calculate_distance(self) -> list[int]:
        distances = [
            count_qubits(logical, include_identities=False)[0]
            for logical in self.logical_ops
        ]

        return distances

    @property
    def distance(self) -> int:
        """The distance of the given Code.

        By definition, the distance of a code is the least weight logical operator
        of the code.
        """
        return min(self._distance)

    @property
    def code_parameters(self) -> tuple[int, int, int, int]:
        """Parameters of the given subsystem code as [[n,k,r,d]].

        Returns:
            A length-4 tuple of number of data qubits(n), number of logical qubits(k),
            number of gauge qubits(r) and the distance of the code(d).
        """
        return (
            self.num_data_qubits,
            self.num_logical_qubits,
            self.num_gauge_qubits,
            self.distance,
        )

    @property
    def is_stabiliser_code(self) -> bool:
        """Determine whether code is stabiliser or subsystem code."""
        return self.num_gauge_qubits == 0

    # @property
    # def logical_operators(self):
    #     # TODO: Because this calculation is not trivial should I cache the property?
    #     return self._logical_operator

    @property
    def tanner_graph(self) -> AnnotatedSparseGraph:
        """The graph with edges between stabiliser checks and data qubits.

        Each stabiliser check defines a new node (ancilla/stabiliser node) and
        each qubit involved in the check is a data-qubit node.
        """
        return self._tanner_graph

    @property
    def embedded_graph(self) -> AnnotatedMultiGraph:
        """The graph with multi edges between ancilla qubits and data qubits.

        Each edge carries the information of Pauli factor involved in the checks.
        """
        return self._embedded_graph

    def validate(self):
        """Check that this is a valid subsystem code.

        It will validate if the stabilizers, gauge, and logical operators make
        a valid subsystem code according to some criteria.

        .. todo:: add criteria.
        """
        raise NotImplementedError

    @classmethod
    def from_embedded_graph(cls, graph: AnnotatedMultiGraph):
        """Create a subsystem code from an annotated multigraph.

        Args:
            graph: An annotated multigraph of the code to be implemented.

        Returns:
            A :class:`Code` corresponding to the provided graph.

        Notes:
            There is a problem here when writing the inverse function. When there are
            multiple edges going out of ancilla node, how do we decide to which
            operator the does the edge belong to.
        """
        raise NotImplementedError

    @cached_property
    def measured_operators(self) -> list[list[PauliDict]]:
        """Nested list of measured ops in the structure of :attr:`factorized_checks`.

        Returns:
            A nested list of PauliDicts.
            If the inner list of length 1, then operator is stabiliser.
            If the inner list of greater length, then operator is gauge operators.

        Todo:
            - Add example with Steane Code later

        """
        measured_ops: list[list[PauliDict]] = list()
        for i, factors in enumerate(self.factorized_checks):
            if not factors:
                measured_ops.append([pauli.pauli_to_dict(self.stabilizers[i])])
                continue

            if self.gauge_ops is None:
                raise ValueError("attr: gauge_ops is None, but index is referenced.")
            measured_ops.append(
                [pauli.pauli_to_dict(self.gauge_ops[f]) for f in factors]
            )

        return measured_ops

    @cached_property
    def data_supports(self) -> list[list[int | tuple[int, int]]]:
        """Nested list with the indices of the checks each data qubit is involved in.

        The index of outer list corresponds to the data qubit index.
        The inner list contains the index corresponding to factorized checks.

        Returns:
            Nested list as described above.

        Todo:
            - Add example with Steane Code later.
        """
        data_supports: list[list[int | tuple[int, int]]] = [
            [] for _ in range(self.num_data_qubits)
        ]

        for i, factors in enumerate(self.measured_operators):
            if len(factors) == 1:
                for qubit_idx in sorted(factors[0]):
                    data_supports[qubit_idx].append(i)
            else:
                for j, f in enumerate(factors):
                    for qubit_idx in sorted(f):
                        data_supports[qubit_idx].append((i, j))

        return data_supports

    def _generate_factorized_checks(self, measure_gauges: bool) -> list[list[int]]:
        """Generate the factorized checks for the measurement operators.

        Args:
            measure_gauges: Bool to determine whether to measure gauge opes.

        Returns:
            The value for :attr:`~SubsystemCodes.factorized_checks`
        """
        factorized_checks: list[list[int]] = [
            list() for _ in range(len(self.stabilizers))
        ]
        if not measure_gauges:
            return factorized_checks

        else:
            stabs = [
                pauli.pauli_to_dict(t.cast(Tableau, op)) for op in self.stabilizers
            ]
            if self.gauge_ops is None:
                raise ValueError("Gauge operators is empty.")
            gauges = [pauli.pauli_to_dict(t.cast(Tableau, op)) for op in self.gauge_ops]

            # list to keep track of matched gauge op
            matched_gauges: list[int] = []

            for s, stab in enumerate(stabs):
                to_match = set(stab.items())
                factors: list[int] = []  # list of factors for a given stabiliser.
                for g, gauge in enumerate(gauges):
                    if set(gauge.items()).issubset(to_match):
                        factors.append(g)
                        to_match.difference_update(gauge.items())
                        if len(to_match) == 0:
                            break

                matched_gauges.extend(factors)
                factorized_checks[s] = factors

            return factorized_checks

    def _generate_ancilla_supports(self, compact: bool = False) -> None:
        """Generate the supports for each ancilla.

        Args:
            compact: Flag to determine if multiple checks can use the same ancilla.
                If ``False``, each checks gets assigned a single ancilla.
                If ``True``, an algorithm assigns ancillas with at most 2 checks per
                ancilla.

        Notes:
            * By "support" we mean which checks does each ancilla involved. For
              instance in the steane code, X1X2X3X4 and Z1Z2Z3Z4 are supported by the
              same ancilla. In the worst resource usage case, each ancilla only supports
              one check.
            * We keep to track of the checks by referring to indices of the
              :attr:`SubsystemCode.factorized_checks`
            * The length of the outer list gives the total number of ancillas used in
              the implementation of the code.

            The algorithm for compact generation:

                1. For each round, pick one check to compare against.
                2. Go through the list of checks. If the following occurs, store the
                  indices into ancilla supports and pop from the search list.

                    - If both checks are involved in the same set of data qubits,
                    assign them to the same ancilla and pop from list.
                    - If one check is a complete subset of another, assign them to
                    the same ancilla and pop from list.
                    - If no matches found for current selected check, assign a
                    separate ancilla and pop from list.
                    - Repeat from step 1 until list is empty.

        Todo:
            - Add example/doctest with Steane / BaconShor code
        """
        ancilla_supports: list[list[tuple[int, int] | int]] = list()

        if compact:
            # flattened list of keys for the values in
            # :attr:`Code.factorized_checks`
            idxs_list: list[int | tuple[int, int]] = []

            # populate idxs_list
            for i, factors in enumerate(self.factorized_checks):
                if not factors:
                    # if stabilizer directly measured, store the index which
                    # corresponds to outer list of
                    # :attr:`Code.factorized_checks`
                    idxs_list.append(i)
                    continue
                for j in range(len(factors)):
                    # if factors is measured, store the 2d index to get index of the
                    # op from :attr:`Code.factorized_checks`
                    idxs_list.append((i, j))

            def get_footprint_of_op(idx: int | tuple[int, int]) -> set[int]:
                """Get indices of data qubits the operator is involved in.

                The footprint of :math:`X_1 X_2` is {1,2}.

                Args:
                    idx: The index of the operator in correspondence to
                         :attr:`Code.measured_ops`
                """
                if isinstance(idx, int):
                    return set(self.measured_operators[idx][0].keys())
                elif isinstance(idx, tuple) and len(idx) == 2:
                    return set(self.measured_operators[idx[0]][idx[1]].keys())
                else:
                    raise ValueError("Invalid indices!")

            # run algorithm while list is not empty.
            while idxs_list:
                # current op to compare against and it's footprint. by default,
                # we use the top of the stack as current and pop it always before the
                # next round.
                current, footprint = idxs_list[0], get_footprint_of_op(idxs_list[0])

                # flag to see if current is matched with another for
                # sharing.
                matched = False

                for i, idx in enumerate(idxs_list):
                    if idx == current:
                        # skip comparing against current.
                        continue
                    # footprint of the op to compare against
                    to_match = get_footprint_of_op(idx)

                    # check for both subset and superset, we don't if op is larger
                    if to_match.issubset(footprint):
                        ancilla_supports.append([current, idx])
                        # need to pop matched first and then current later otherwise,
                        # indexing of the list gets' altered.
                        idxs_list.pop(i)  # pop matched.
                        idxs_list.pop(0)  # pop current
                        matched = True  # set to true when matched.
                        break
                    elif to_match.issuperset(footprint):
                        ancilla_supports.append([current, idx])
                        idxs_list.pop(i)  # pop matched
                        idxs_list.pop(0)  # pop current
                        matched = True
                        break
                    else:
                        continue

                if not matched:
                    # assign single check for ancilla when no match found.
                    ancilla_supports.append([current])
                    idxs_list.pop(0)

            self.ancilla_supports = ancilla_supports

        else:
            # When compact is false, assign one ancilla per check in factorized_checks.
            # Examples is Surface Code
            for i, factors in enumerate(self.factorized_checks):
                if not factors:
                    ancilla_supports.append([i])
                    continue
                for j in range(len(factors)):
                    ancilla_supports.append([(i, j)])

            self.ancilla_supports = ancilla_supports

    def _generate_embedded_graph(self) -> None:
        """Generate the graph of how the Tanner graph is embedded on a real device.

        The current assumption where we don't deal with flags is that
        factorized_checks also define the device connectivity. This is the graph
        which will be used to generate the circuits.

        """
        edges: list[tuple[int, int]] = list()
        # Indexing is done data-qubits first.
        edges_data: list[EdgeMetadata] = list()

        nodes_data: list[NodeMetadata | None] = [
            None for _ in range(self.num_data_qubits + len(self.ancilla_supports))
        ]

        for i, checks in enumerate(self.ancilla_supports):
            ancilla_idx = i + self.num_data_qubits
            for check_idx in checks:
                if isinstance(check_idx, int):
                    # direct measurement of stabilizer
                    support = self.measured_operators[check_idx][0]
                    for qubit_idx in sorted(support):
                        edges.append((ancilla_idx, qubit_idx))
                        edges_data.append(EdgeMetadata(support[qubit_idx]))
                        nodes_data[qubit_idx] = NodeMetadata(QubitType.data)
                        nodes_data[ancilla_idx] = NodeMetadata(QubitType.stabilizer)
                elif isinstance(check_idx, tuple):
                    # facto
                    if self.gauge_ops is None:
                        raise ValueError(
                            "Cannot access the gauge ops factors as it is empty!"
                        )

                    support = self.measured_operators[check_idx[0]][check_idx[1]]
                    for qubit_idx in sorted(support):
                        edges.append((ancilla_idx, qubit_idx))
                        edges_data.append(EdgeMetadata(support[qubit_idx]))
                        nodes_data[qubit_idx] = NodeMetadata(QubitType.data)
                        nodes_data[ancilla_idx] = NodeMetadata(QubitType.gauge)
                else:
                    raise ValueError("Invalid indexing!")

        self._embedded_graph = AnnotatedMultiGraph(edges, nodes_data, edges_data)

    def _generate_tanner_graph(self):
        """Generate the Tanner graph of data and ancilla qubits."""
        stabiliser_supports = [pauli.pauli_to_dict(s) for s in self.stabilizers]
        # Indexing of qubits is done data-qubits first, then ancillas
        edges: list[tuple[int, int]] = list()
        edges_data: list[EdgeMetadata] = list()
        # We need to populate this immediately because we later need to access/set
        # elements half-way through the list.
        nodes_data: list[t.Optional[NodeMetadata]] = [
            None for _ in range(self.num_qubits)
        ]
        for i, support in enumerate(stabiliser_supports):
            ancilla_idx = i + self.num_data_qubits
            for qubit_idx in sorted(support):
                edges.append((ancilla_idx, qubit_idx))
                edges_data.append(EdgeMetadata(support[qubit_idx]))
                nodes_data[qubit_idx] = NodeMetadata(QubitType.data)
                nodes_data[ancilla_idx] = NodeMetadata(QubitType.stabilizer)
        self._tanner_graph = AnnotatedSparseGraph(nodes_data, edges, edges_data)

    @classmethod
    def make_rotated_planar(
        cls, distance: int | tuple[int, int], xzzx: bool = False
    ) -> "Code":
        """Generate a :class:`Code` object for rotated planar code.

        Args:
            distance: The distance of the code.
                If ``int``, both X and Z distances are considered to be the same.
                If ``tuple`` of length 2, then interpreted as (X, Z) distances
                respectively.
            xzzx: Bool to determine whether to use xxzx checks.

        Returns:
             A :class:`Code` object corresponding to the rotated planar code.
        """
        if isinstance(distance, int):
            x_distance, z_distance = distance, distance
        elif isinstance(distance, tuple) and len(distance) == 2:
            x_distance, z_distance = distance
        else:
            raise ValueError(
                "distance of must be an integer or tuple of ints of len 2."
            )
        qubit_number = x_distance * z_distance
        if xzzx:
            stabs = []
            for row in range(x_distance - 1):
                for col in range(z_distance - 1):
                    stabs.append(
                        pauli.string_to_pauli(
                            f"X{z_distance * row + col}"
                            f"Z{z_distance * row + col + 1}"
                            f"Z{z_distance * (row + 1) + col+1}"
                            f"X{z_distance * (row + 1) + col}",
                            qubits=z_distance * x_distance,
                        )
                    )

                    if row == 0 and (col + row) % 2 == 0:
                        stabs.append(
                            pauli.string_to_pauli(
                                f"Z{col}X{col+1}", qubits=z_distance * x_distance
                            )
                        )
                        stabs.append(
                            pauli.string_to_pauli(
                                f"Z{z_distance * (x_distance - 1) + col + 1}"
                                f"X{z_distance * (x_distance - 1) + col + 2}",
                                qubits=z_distance * x_distance,
                            )
                        )
                    elif col == 0 and (col + row) % 2 != 0:
                        stabs.append(
                            pauli.string_to_pauli(
                                f"Z{z_distance * row}X{z_distance * (row + 1)}",
                                qubits=z_distance * x_distance,
                            )
                        )
                        stabs.append(
                            pauli.string_to_pauli(
                                f"Z{z_distance * row + x_distance - 1}"
                                f"X{z_distance * (row - 1) + x_distance - 1}",
                                qubits=z_distance * x_distance,
                            )
                        )

        else:
            x_stabs = []
            z_stabs = []

            # X boundaries are on top and bottom
            # Z boundaries are on left and right.
            for row in range(x_distance - 1):
                for col in range(z_distance - 1):
                    if (col + row) % 2 == 0:
                        z_stabs.append(
                            pauli.string_to_pauli(
                                f"Z{z_distance * row + col}"
                                f"Z{z_distance * row + col + 1}"
                                f"Z{z_distance * (row + 1) + col}"
                                f"Z{z_distance * (row + 1) + col + 1}",
                                qubit_number,
                            )
                        )
                        if row == 0:
                            x_stabs.append(
                                pauli.string_to_pauli(f"X{col}X{col + 1}", qubit_number)
                            )
                            # FIXME: extra qubit count for even indices.
                            x_stabs.append(
                                pauli.string_to_pauli(
                                    f"X{z_distance * (x_distance - 1) + col + 1}"
                                    f"X{z_distance * (x_distance - 1) + col + 2}",
                                    qubit_number,
                                )
                            )

                    else:
                        x_stabs.append(
                            pauli.string_to_pauli(
                                f"X{z_distance * row + col}"
                                f"X{z_distance * row + col + 1}"
                                f"X{z_distance * (row + 1) + col}"
                                f"X{z_distance * (row + 1) + col + 1}",
                                qubit_number,
                            )
                        )
                        if col == 0:
                            z_stabs.append(
                                pauli.string_to_pauli(
                                    f"Z{z_distance * row}Z{z_distance * (row + 1)}",
                                    qubit_number,
                                )
                            )
                            z_stabs.append(
                                pauli.string_to_pauli(
                                    f"Z{z_distance * row + x_distance -1}"
                                    f"Z{z_distance * (row - 1) + x_distance -1}",
                                    qubit_number,
                                )
                            )

            stabs = x_stabs + z_stabs

        logical_x = pauli.string_to_pauli(
            "".join([f"X{z_distance * j}" for j in range(x_distance)]),
            qubit_number,
        )
        logical_z = pauli.string_to_pauli(
            "".join([f"Z{i}" for i in range(z_distance)]),
            qubit_number,
        )

        return cls(
            stabilizers=stabs,
            logical_ops=[logical_x, logical_z],
            compact=False,
        )

    @classmethod
    def make_bacon_shor(
        cls, distance: int | tuple[int, int], measure_gauges: bool = False
    ) -> "Code":
        """Generate a :class:`~Code object for Bacon-Shor Code.

        The Bacon Shor code is defined on a square lattice with data qubits on
        vertices of the lattice. When measuring through gauges, the ancilla qubits
        lie on the edges, with X gauges on vertical edges and Z gauges on horizontal
        edges.

        Args:
            distance: The distance of the code.
                If ``int``, both X and Z distances are considered to be the same.
                If ``tuple`` of length 2, then interpreted as (X, Z) distances
                respectively.
            measure_gauges: Flag to determine measure gauges or stabilizer directly.

        Returns:
            A :class:`Code` object corresponding to the 2D Bacon-Shor code.
        """
        if isinstance(distance, int):
            x_distance, z_distance = distance, distance
        elif isinstance(distance, tuple) and len(distance) == 2:
            x_distance, z_distance = distance
        else:
            raise ValueError(
                "distance of must be an integer or tuple of ints of len 2."
            )

        qubit_number = x_distance * z_distance

        stabs = pauli.sort_operators_ref(
            [
                pauli.string_to_pauli(
                    "".join(
                        [
                            f"X{j * x_distance + i}X{(j + 1) * x_distance + i}"
                            for i in range(x_distance)
                        ]
                    ),
                    qubit_number,
                )
                for j in range(z_distance - 1)
            ]
            + [
                pauli.string_to_pauli(
                    "".join(
                        [
                            f"Z{j * x_distance + i}Z{j * x_distance + i + 1}"
                            for j in range(z_distance)
                        ]
                    ),
                    qubit_number,
                )
                for i in range(x_distance - 1)
            ]
        )

        gauges = pauli.sort_operators_ref(
            [
                pauli.string_to_pauli(
                    f"X{j * x_distance + i}X{(j + 1) * x_distance + i}",
                    qubit_number,
                )
                for i in range(x_distance)
                for j in range(z_distance - 1)
            ]
            + [
                pauli.string_to_pauli(
                    f"Z{j * x_distance + i}Z{j * x_distance + i + 1}",
                    qubit_number,
                )
                for i in range(x_distance - 1)
                for j in range(z_distance)
            ]
        )

        logical_x = pauli.string_to_pauli(
            "".join([f"X{i}" for i in range(x_distance)]),
            qubit_number,
        )
        logical_z = pauli.string_to_pauli(
            "".join([f"Z{j + x_distance}" for j in range(z_distance)]),
            qubit_number,
        )

        return cls(
            stabilizers=stabs,
            gauge_ops=gauges,
            logical_ops=[logical_x, logical_z],
            measure_gauges=measure_gauges,
        )

    @classmethod
    def make_planar(cls, distance: int | tuple[int, int]) -> "Code":
        """Generate a :class:`Code` object for planar code.

        Args:
            distance: The distance of the code.
                    If ``int``, both X and Z distances are considered to be the same.
                    If ``tuple`` of length 2, then interpreted as (X, Z) distances
                    respectively.

        Returns:
            A :class:`Code` object corresponding to the planar code.

        """
        if isinstance(distance, int):
            x_distance, z_distance = distance, distance
        elif isinstance(distance, tuple) and len(distance) == 2:
            x_distance, z_distance = distance
        else:
            raise ValueError(
                "distance of must be an integer or tuple of ints of len 2."
            )

        data_indices = []  # list containing the indices of the data qubits.
        nodes_coords: list[list[float]] = []
        num_data = 0  # counter for the data qubit index
        # populate data qubit list.
        for row in range(2 * z_distance - 1):
            if row % 2 == 0:
                data_indices.append(list(range(num_data, num_data + x_distance)))
                num_data = num_data + x_distance
            else:
                data_indices.append(list(range(num_data, num_data + x_distance - 1)))
                num_data = num_data + x_distance - 1

            for col in range(2 * x_distance - 1):
                # add coords
                if ((row % 2) and (col % 2)) or (not ((row % 2) or (col % 2))):
                    nodes_coords.append([col, row, 0])

        z_stabs, x_stabs = [], []

        for r, qubits in enumerate(data_indices):
            # use even row for Z stabilisers
            if r % 2 == 0:
                for qubit_index in qubits[:-1]:
                    if r == 0:
                        # if bottom boundary add 3 body stabilisers pointing up.
                        z_stabs.append(
                            pauli.string_to_pauli(
                                f"Z{qubit_index}"
                                f"Z{qubit_index+1}"
                                f"Z{qubit_index+x_distance}",
                                qubits=num_data,
                            )
                        )
                    elif r == len(data_indices) - 1:
                        # if top boundary add 3 body stabilisers pointing down.
                        z_stabs.append(
                            pauli.string_to_pauli(
                                f"Z{qubit_index}"
                                f"Z{qubit_index+1}"
                                f"Z{qubit_index-x_distance+1}",
                                qubits=num_data,
                            )
                        )
                    else:
                        z_stabs.append(
                            pauli.string_to_pauli(
                                f"Z{qubit_index}"
                                f"Z{qubit_index+1}"
                                f"Z{qubit_index+x_distance}"
                                f"Z{qubit_index-x_distance+1}",
                                qubits=num_data,
                            )
                        )
            # use odd rows or X stabilisers
            else:
                for c, qubit_index in enumerate(qubits):
                    if c == len(qubits) - 1:
                        # if right boundary, add 3 body stabiliser pointing left.
                        x_stabs.append(
                            pauli.string_to_pauli(
                                f"X{qubit_index}"
                                f"X{qubit_index-x_distance+1}"
                                f"X{qubit_index+x_distance}",
                                qubits=num_data,
                            )
                        )
                        continue
                    elif c == 0:
                        # if left boundary, add 3 body stabiliser pointing right.
                        # there is no continue in this elif block, because num
                        # qubits in this row is one less than number of
                        # stabiliser added in this row. We are adding twice for
                        # the first entry of the row.
                        x_stabs.append(
                            pauli.string_to_pauli(
                                f"X{qubit_index}"
                                f"X{qubit_index- x_distance}"
                                f"X{qubit_index+x_distance-1}",
                                qubits=num_data,
                            )
                        )

                    x_stabs.append(
                        pauli.string_to_pauli(
                            f"X{qubit_index}"
                            f"X{qubit_index+1}"
                            f"X{qubit_index+x_distance}"
                            f"X{qubit_index-x_distance+1}",
                            qubits=num_data,
                        )
                    )

        logicals = [
            pauli.string_to_pauli(
                "".join([f"X{i}" for i in range(x_distance)]), qubits=num_data
            )
        ] + [
            pauli.string_to_pauli(
                "".join(
                    [
                        f"Z{data_indices[i][0]}"
                        for i in range(len(data_indices))
                        if i % 2 == 0
                    ]
                ),
                qubits=num_data,
            )
        ]
        ssc = Code(stabilizers=z_stabs + x_stabs, logical_ops=logicals, compact=False)
        # TODO: this was done for debugging, but it should be done for all
        #  pre-defined codes
        # Add data coords
        for node in range(ssc.num_data_qubits):
            ssc.tanner_graph.nodes_data[node] = NodeMetadata(
                QubitType.data, nodes_coords[node]  # type: ignore
            )
        # Add ancilla coords
        for anc in ssc.ancilla_qubit_indices:
            neigh = list()
            for v in ssc.tanner_graph.get_vertices_touching_vertex(anc):
                x, y, z = ssc.tanner_graph.nodes_data[v].coords
                neigh.append(graph.Position(x=x, y=y, z=z))
            ssc.tanner_graph.nodes_data[anc].coords = list(graph.center_of_mass(neigh))

        return ssc

    @classmethod
    def make_repetition(cls, distance: int, phase_flip: bool = False) -> "Code":
        """Generate a :class:`Code` object for bit(phase) flip repetition code.

        By default, it makes the code the corrects bit flips.

        Args:
            distance: The distance of the code.
            phase_flip: Bool to determine if the correct bit flips or phase flips.
                Defaults to ``False``, which correct bit flips.
                If ``True``, generates stabilisers to correct for phase flips.

        Returns:
            A :class:`Code` object corresponding to the bit(phase) flip
            repetition code.
        """
        if not phase_flip:
            stab_factor = "Z"
            logicals = [
                pauli.string_to_pauli("X" * distance),
                pauli.string_to_pauli("Z0", qubits=distance),
            ]

        else:
            stab_factor = "X"
            logicals = [
                pauli.string_to_pauli("X0", qubits=distance),
                pauli.string_to_pauli("Z" * distance),
            ]

        stabs = [
            pauli.string_to_pauli(
                f"{stab_factor}{i}{stab_factor}{i + 1}", qubits=distance
            )
            for i in range(distance - 1)
        ]
        return Code(stabilizers=stabs, logical_ops=logicals, compact=False)

    @classmethod
    def make_steane(cls, compact=False):
        """Generate a :class:`Code` object for the Steane code.

        Args:
            compact: Bool to determine whether any ancilla is used across multiple
                measurements. For more details, see :attr:`~Code.compact`

        Returns:
            A :class:`Code` object corresponding to the Steane code.
        """
        stabs = [
            pauli.string_to_pauli("X0X1X3X4", 7),
            pauli.string_to_pauli("X1X2X4X5", 7),
            pauli.string_to_pauli("X3X4X5X6", 7),
        ] + [
            pauli.string_to_pauli("Z0Z1Z3Z4", 7),
            pauli.string_to_pauli("Z1Z2Z4Z5", 7),
            pauli.string_to_pauli("Z3Z4Z5Z6", 7),
        ]
        logicals = [
            pauli.string_to_pauli("X0X1X2", qubits=7),
            pauli.string_to_pauli("Z0Z1Z2", qubits=7),
        ]

        return Code(stabilizers=stabs, logical_ops=logicals, compact=compact)

    @classmethod
    def make_heavy_hex_without_flag(
        cls, distance: int | tuple[int, int], measure_gauges=True, compact=False
    ):
        """Class method to make the heavy hexagon code.

        This implementation does not add the flag qubits.

        Args:
            distance: The distance of the code.
                If ``int``, both X and Z distances are considered to be the same.
                If ``tuple`` of length 2, then interpreted as (X, Z) distances
                respectively.
            compact: Bool to determine whether ancilla are shared across stabilisers.
            measure_gauges: Bool to determine whether to measure gauges of stabilisers.

        Returns:
            A :class:`~Code` corresponding to the heavy hexagon code without
            the flag qubits.

        TODO: Bug fixes are required for the correct edges. Currently it generates a
        wrong list.
        """
        if isinstance(distance, int):
            x_distance, z_distance = distance, distance
        elif isinstance(distance, tuple) and len(distance) == 2:
            x_distance, z_distance = distance
        else:
            raise ValueError(
                "distance of must be an integer or tuple of ints of len 2."
            )

        # X boundaries lie on top and bottom.
        # Z boundaries lie on left and right.

        x_gauges, z_gauges = [], []
        z_stabs = []

        x_stabs = [
            pauli.string_to_pauli(
                "".join(
                    [
                        f"X{x_distance*row + col}X{x_distance*row + col+1}"
                        for row in range(x_distance)
                    ]
                ),
                qubits=z_distance * x_distance,
            )
            for col in range(z_distance - 1)
        ]

        for row in range(x_distance - 1):
            for col in range(z_distance - 1):
                if (row + col) % 2 == 0:
                    x_gauges.append(
                        pauli.string_to_pauli(
                            f"X{z_distance * row + col}"
                            f"X{z_distance * row + col + 1}"
                            f"X{z_distance * (row + 1) + col}"
                            f"X{z_distance * (row + 1) + col + 1}",
                            qubits=z_distance * x_distance,
                        )
                    )
                    if col == 0:
                        z_gauges.append(
                            pauli.string_to_pauli(
                                f"Z{z_distance * (row+2) + col}"
                                f"Z{z_distance * (row +1) + col}",
                                qubits=z_distance * x_distance,
                            )
                        )
                        z_gauges.append(
                            pauli.string_to_pauli(
                                f"Z{z_distance * (row +2) -1 }"
                                f"Z{z_distance * (row +3) -1}",
                                qubits=z_distance * x_distance,
                            )
                        )
                        z_stabs.append(
                            pauli.string_to_pauli(
                                f"Z{z_distance * (row + 2) - 1}"
                                f"Z{z_distance * (row + 3) - 1}",
                                qubits=z_distance * x_distance,
                            )
                        )

                        z_stabs.append(
                            pauli.string_to_pauli(
                                f"Z{z_distance * row + col}"
                                f"Z{z_distance * (row +1) + col}",
                                qubits=z_distance * x_distance,
                            )
                        )

                else:
                    z_stabs.append(
                        pauli.string_to_pauli(
                            f"Z{z_distance * row + col}"
                            f"Z{z_distance * row + col + 1}"
                            f"Z{z_distance * (row + 1) + col}"
                            f"Z{z_distance * (row + 1) + col + 1}",
                            qubits=z_distance * x_distance,
                        )
                    )
                    z_gauges.append(
                        pauli.string_to_pauli(
                            f"Z{z_distance * row + col}"
                            f"Z{z_distance * (row +1) + col}",
                            qubits=z_distance * x_distance,
                        )
                    )
                    z_gauges.append(
                        pauli.string_to_pauli(
                            f"Z{z_distance * row  + col + 1}"
                            f"Z{z_distance * (row + 1) + col + 1}",
                            qubits=z_distance * x_distance,
                        )
                    )
                    if row == 0:
                        x_gauges.append(
                            pauli.string_to_pauli(
                                f"X{col}X{col+1}", qubits=z_distance * x_distance
                            )
                        )
                        x_gauges.append(
                            pauli.string_to_pauli(
                                f"X{col + z_distance *(x_distance-1)}"
                                f"X{col + z_distance * (x_distance - 1) - 1}",
                                qubits=z_distance * x_distance,
                            )
                        )

        logical_x = pauli.string_to_pauli(
            "".join([f"X{z_distance * j}" for j in range(x_distance)]),
            qubits=z_distance * x_distance,
        )
        logical_z = pauli.string_to_pauli(
            "".join([f"Z{i}" for i in range(z_distance)]),
            qubits=z_distance * x_distance,
        )

        return Code(
            stabilizers=x_stabs + z_stabs,
            gauge_ops=x_gauges + z_gauges,
            logical_ops=[logical_x, logical_z],
            measure_gauges=measure_gauges,
            compact=compact,
        )

    @cached_property
    def data_qubit_indices(self):
        """Indices of the data qubits of this code."""
        return [
            i
            for i, d in enumerate(self.tanner_graph.nodes_data)
            if d.type == QubitType.data
        ]

    @cached_property
    def ancilla_qubit_indices(self):
        """Indices of the ancilla qubits of this code."""
        return [
            i
            for i, d in enumerate(self.tanner_graph.nodes_data)
            if d.type == QubitType.stabilizer
        ]

    @classmethod
    def make_five_qubit(cls) -> "Code":
        """Generate :class:`Code` object for the five qubit code."""
        stabiliser = [
            pauli.string_to_pauli(s, qubits=5)
            for s in "XZZXI IXZZX XIXZZ ZXIXZ".split()
        ]
        logical_ops = [
            pauli.string_to_pauli(log, qubits=5) for log in "XXXXX ZZZZZ".split()
        ]

        return cls(
            stabilizers=stabiliser, logical_ops=logical_ops, compact=False, dist=[3]
        )

    @classmethod
    def make_shor(cls) -> "Code":
        """Generate :class:`Code` object for the 9-qubit Shor code."""
        stabilisers = [
            pauli.string_to_pauli(s, qubits=9)
            for s in """
            ZZIIIIIII
            IZZIIIIII
            XXXXXXIII
            IIIZZIIII
            IIIIZZIII
            IIIXXXXXX
            IIIIIIZZI
            IIIIIIIZZ
            """.split()
        ]
        logicals = [pauli.string_to_pauli(log, qubits=9) for log in [9 * "Z", 9 * "X"]]
        return cls(stabilizers=stabilisers, logical_ops=logicals, dist=[3])
