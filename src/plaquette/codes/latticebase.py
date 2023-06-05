# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Basic tools for defining codes using a 2D lattice.

In plaquette, a stabilizer code can be defined in terms of qubits located on a
two-dimensional lattice. Identifying both qubits and stabilizer generators by lattice
coordinates helps to define a code easily.

A code is defined by creating vertices within the lattice; each vertex represents
a data qubit, stabilizer generator or logical operator. Edges connect those operators
to those data qubits on which they act non-trivially. Edges need not be
nearest-neighbour, but they can be arbitrary, allowing for the implementation of
codes with arbitrary shapes and connectivities.

How to define a custom code using the lattice is explained in :ref:`codes-guide`.
A concise summary of the constituents of the lattice is available in
:class:`CodeLattice`.
"""

from __future__ import annotations

import enum
import itertools as it
import sys
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from dataclasses import field as dfield
from typing import ClassVar, Final, Literal, Optional, cast, overload

import numpy as np

from plaquette.pauli import Tableau, string_to_pauli

#: Integer value to signal "index not assigned yet"
NO_INDEX: Final[int] = -sys.maxsize


# TODO: this must go and we only keep binary representation
class Pauli(enum.IntEnum):
    """Pauli operator (X, Y, Z).

    This is an enum.
    """

    #: Pauli X
    X = 1
    #: Pauli Y
    Y = 2
    #: Pauli Z
    Z = 3


class StabGroup(enum.IntEnum):
    """Group of a stabilizer (indicates connected component in matching graph).

    This is an enum.

    Stabilizer generators from different groups belong to different connected
    components of the matching graph. Are two groups always sufficient? At this
    point, I don't know the answer.
    """

    #: Unknown connected component
    U = 0
    #: First connected component (elsewhere called "primal" syndrome graph or lattice)
    A = 1
    #: Second connected component (elsewhere called "dual" syndrome graph or lattice)
    B = 2


class VertexType(enum.Enum):
    """Type of a vertex in the lattice (data qubit or operator).

    This is an enum.

    .. todo::

       The attribute :attr:`Vertex.type` as well as this class are unnecessary.
       It better to use ``isinstance()`` checks instead, because these are
       recognized by mypy and don't need the attribute.

       The attribute and this class should be removed.
    """

    #: Data qubit
    data = "data"
    #: Stabilizer operator
    stab = "stab"
    #: Logical operator
    log = "log"


#: Lattice position (tuple of integers)
PosType = tuple[int, ...]


@dataclass(slots=True, kw_only=True)
class Vertex:
    """Code lattice vertex (represents a data qubit or an operator).

    A vertex can represent one of three things:

    * A physical data qubit -- subclass :class:`DataVertex`
    * A stabilizer generator -- subclass :class:`StabGenVertex`
    * A logical operator -- subclass :class:`LogicalVertex`

    Correspondingly, physical qubits can be used for three different purposes:

    * Data qubit within the code
    * Ancilla qubit used to measure a stabilizer generator
    * Ancilla qubit used to measure a logical operator

    The "extended" set of qubits contains all these qubits. The value of
    :attr:`equbit_idx` identifies qubits within the set of "extended" qubits.

    This is a dataclass.

    .. automethod:: __init__
    """

    #: Vertex type (data qubit or stabilizer generator)
    type: ClassVar[VertexType]
    #: Position in the lattice
    pos: PosType
    #: Neighbouring vertices
    neighbours: Sequence[Vertex] = dfield(default_factory=list)
    #: Adjacent edges
    edges: list[Edge] = dfield(default_factory=list)
    #: Extended qubit index (data + all ancilla qubits, see :class:`Vertex`).
    #: The value :data:`NO_INDEX` indicates that the index was not assigned yet.
    equbit_idx: int = NO_INDEX

    def __repr__(self):  # noqa: D105
        return f"{self.__class__.__name__}(pos = {self.pos}, index = {self.equbit_idx})"


@dataclass(slots=True, kw_only=True)
class DataVertex(Vertex):
    """Data qubit vertex.

    This is a dataclass.

    .. automethod:: __init__
    """

    #: Vertex type (data qubit)
    type: ClassVar[VertexType] = dfield(default=VertexType.data, init=False)
    #: Neighbouring vertices (stabilizer and logical operators)
    neighbours: list[OpVertex] = dfield(default_factory=list)
    #: Data qubit index.
    #: The value :data:`NO_INDEX` indicates that the index was not assigend yet.
    dataqubit_idx: int = NO_INDEX

    def __repr__(self):  # noqa: D105
        return (
            f"{self.__class__.__name__}(pos = {self.pos}, ext_idx ="
            f" {self.equbit_idx}, data_idx = {self.dataqubit_idx})"
        )


@dataclass(slots=True, kw_only=True)
class OpVertex(Vertex):
    """Operator vertex (stabilizer or logical).

    This is a dataclass.

    .. automethod:: __init__
    """

    # TODO-Sphinx(5.1.1): The following declaration silences Sphinx warnings
    #: Vertex type (data qubit or stabilizer generator)
    type: ClassVar[VertexType]
    #: Neighbouring vertices (only data qubits)
    neighbours: list[DataVertex] = dfield(default_factory=list)

    def __repr__(self):  # noqa: D105
        return (
            f"{self.__class__.__name__}(pos = {self.pos}, ext_idx = {self.equbit_idx})"
        )


@dataclass(slots=True, kw_only=True)
class StabGenVertex(OpVertex):
    """Stabilizer generator vertex.

    This is a dataclass.

    .. automethod:: __init__
    """

    #: Vertex type (stabilizer generator)
    type: ClassVar[VertexType] = dfield(default=VertexType.stab, init=False)
    #: Stabilizer group (identifies connected component in matching graph)
    group: StabGroup
    #: Stabilizer generator index.
    #: The value :data:`NO_INDEX` indicates that the index was not assigned yet.
    stabgen_idx: int = NO_INDEX

    def __repr__(self):  # noqa: D105
        return (
            f"{self.__class__.__name__}(pos = {self.pos}, ext_idx ="
            f" {self.equbit_idx}, stab_idx = {self.stabgen_idx})"
        )


@dataclass(slots=True, kw_only=True)
class LogicalVertex(OpVertex):
    """Logical operator vertex.

    This is a dataclass.

    .. automethod:: __init__
    """

    #: Name of the logical operator (e.g. "X", "Z2")
    name: str
    #: Vertex type (logical operator)
    type: ClassVar[VertexType] = dfield(default=VertexType.log, init=False)
    #: Logical operator index.
    #: The value :data:`NO_INDEX` indicates that the index was not assigned yet.
    logical_idx: int = NO_INDEX

    def __repr__(self):  # noqa: D105
        return (
            f"{self.__class__.__name__}(pos = {self.pos}, ext_idx ="
            f" {self.equbit_idx}, log_idx = {self.logical_idx})"
        )


@dataclass(slots=True)
class Edge:
    """Code lattice edge (= a single-qubit factor in an operator).

    An edge connects a data qubit to an operator. The edge has information on
    the tensor product factor (Pauli X, Y or Z) which acts on the data qubit.

    This is a dataclass.

    .. automethod:: __init__
    """

    #: Operator in which this single-qubit factor occurs
    op: OpVertex
    #: Data qubit on which this operator acts
    data: DataVertex
    #: The single-qubit Pauli operator
    factor: Pauli
    #: Define temporal ordering of all measurements.
    measurement_time_step: Optional[int] = None


class NoLogicalOperatorError(ValueError):
    """No logical operator available.

    This typically happens after too many data qubits were removed from a code.

    .. automethod:: __init__
    """

    def __init__(self, pauli: Pauli, data: DataVertex):
        """Create a new instance of the error."""
        self.pauli = pauli
        self.data = data

    def __str__(self):  # noqa: D105
        return (
            f"Cannot update logical operator because there is no remaining "
            f"{self.pauli.name} stabilizer on data qubit {self.data.pos}"
        )


class CodeLattice:
    """Lattice used to define stabilizer codes.

    The lattice is a colored graph on a finite hyper-cubic lattice (typically a
    two-dimensional square lattice is used).

    Each lattice site is either empty or occupied by one vertex.

    A vertex can represent one of three things (this provides the "color" of the
    vertex):

    * A physical data qubit -- :class:`DataVertex`
    * A stabilizer generator -- :class:`StabGenVertex`
    * A logical operator -- :class:`LogicalVertex`

    Edges connect one operator to one data qubit and indicate that
    the given operator acts on the given data qubit. The Pauli operator
    which acts on the data qubit is stored in the edge. Usually,
    edges connect operators and data qubits in a nearest-neighbour-only
    fashion, but this restriction is not mandatory. Edges are represented
    by the class :class:`Edge`.

    .. note::

       Options marked as *experimental* may destroy error correction or cause undefined
       behaviour.

    .. automethod:: __init__
    """

    #: Vertex of each lattice site (or ``None`` if empty)
    lattice: np.ndarray
    #: List of all vertices
    vertices: list[Vertex]
    #: List of all data qubits
    dataqubits: list[DataVertex]
    #: List of all stabilizer generators as *vertices*.
    stabgens: list[StabGenVertex]
    #: List of all logical operators
    #:
    #: The order of logical operators is ``[X1, Z1, X2, Z2, ...]`` where
    #: ``X1`` and ``Z2`` are ``X`` on first and ``Z`` on second logical qubit,
    #: respectively.
    logical_ops: list[LogicalVertex]
    #: List of all "extended" qubits (data + all ancilla qubits, see :class:`Vertex`)
    equbits: list[Vertex]
    #: List of all edges
    edges: list[Edge]

    #: In fabrication errors, keep gauge operators (*experimental*).
    #:
    #: * ``"no"`` - When a data qubit is removed, two super-stabilizers (X and Z) are
    #:   created.
    #: * ``"half"`` (experimental) - When a data qubit is removed, one super-stabilizer
    #:   is created and gauge operators are kept for the second super-stabilizer. This
    #:   avoids creating an additional logical qubit.
    #: * ``"all"`` (experimental) - When a data qubit is removed, gauge operators are
    #:   kept for all super-stabilizers. This leads to non-commuting stabilizers!
    fabrication_err_keep_gauge: str = "none"
    #: In fabrication errorns, join boundary stabilizers (*experimental*).
    #:
    #: * ``False`` - When removing a data qubit at the boundary, one stabilizer is
    #:   disabled (as proposed in :cite:`auger_fault-tolerance_2017`).
    #: * ``True`` (experimental) - When removing a data qubit at the boundary, one
    #:   super-stabilizer is created and the third stabilizer is kept.
    fabrication_err_boundary_join: bool = False

    def __init__(self, shape: Iterable[int]):
        """Create a new code lattice.

        Args:
            shape: Shape of the lattice.
        """
        self.lattice = np.zeros(shape, dtype=object)
        self.lattice[:] = None
        self.vertices = []
        self.dataqubits = []
        self.stabgens = []
        self.logical_ops = []
        self.equbits = []
        self.edges = []
        self.n_time_steps: int | None = None

    def assign_indices(self):
        """Assign various indices.

        This function assigns values to the following indices:

        * :attr:`.DataVertex.dataqubit_idx`
        * :attr:`.StabGenVertex.stabgen_idx`
        * :attr:`.LogicalVertex.logical_idx`
        * :attr:`.Vertex.equbit_idx`

        Existing values will be changed if anything has been removed since the last call
        to this function.

        This function also rebuilds :attr:`equbits`.

        .. note::

           This function should be called by the constructor of any subclass. It is
           also called by :meth:`apply_fabrication_errors`. All other methods do *not*
           call this function automatically.
        """
        for idx, qubit in enumerate(self.dataqubits):
            qubit.dataqubit_idx = idx
        for idx, stab in enumerate(self.stabgens):
            stab.stabgen_idx = idx
        for idx, logical in enumerate(self.logical_ops):
            logical.logical_idx = idx
        self.equbits = []
        for v in it.chain(self.dataqubits, self.stabgens, self.logical_ops):
            v.equbit_idx = len(self.equbits)
            self.equbits.append(v)

    def assign_stab_edge_order(self, order: Sequence[tuple[int, int]]):
        """Assign edge order using the relative position of ancilla and data qubit.

        .. note:: As the name of the function indicates, this function assigns a
           temporal order to an edge only if it belongs to a stabilizer generator. Edges
           belonging to logical operators are ignored.

        Args:
            order:
                E.g. ``((0, 1), (1, 0), (0, -1), (-1, 0))`` specifies a "NESW" order.
                Each item specifies the value of the coordinate difference between
                data and ancilla qubit. Edges with the same coordinate difference
                are assigned the same :attr:`Edge.measurement_time_step`.
        """
        self.n_time_steps = len(order)
        for edge in self.edges:
            if not isinstance(edge.op, StabGenVertex):
                continue
            d0 = edge.data.pos[0] - edge.op.pos[0]
            d1 = edge.data.pos[1] - edge.op.pos[1]
            if abs(d0) > 1:
                d0 = self._periodic_boundary_time_step(d0)
            if abs(d1) > 1:
                d1 = self._periodic_boundary_time_step(d1)
            edge.measurement_time_step = order.index((d0, d1))

    def _periodic_boundary_time_step(self, dn: int):
        """Correct the time step for lattices with periodic conditions.

        The time step of an edge is determined by the direction of that edge. The
        ``edge.measurement_time step`` is equal to the corresponding index in the
        variable ``order``. This method corrects against a mistaken interpretation
        of the direction due to the periodicity in the lattice, where a unit vector
        pointing in one direction is being seen as a vector with the same length as the
        lattice's size pointing in the opposite direction.
        """
        if dn > 0:
            return -1
        else:
            return 1

    def add_data(self, pos: PosType):
        """Add data qubit."""
        assert self.lattice[pos] is None
        v = DataVertex(pos=pos)
        self.vertices.append(v)
        self.dataqubits.append(v)
        self.lattice[pos] = v

    def add_stabgen(self, pos: PosType, group: StabGroup):
        """Add stabilizer generator."""
        assert self.lattice[pos] is None
        v = StabGenVertex(pos=pos, group=group)
        self.vertices.append(v)
        self.stabgens.append(v)
        self.lattice[pos] = v

    def add_logical(self, pos: PosType, name: str) -> LogicalVertex:
        """Add logical operator."""
        assert self.lattice[pos] is None
        v = LogicalVertex(pos=pos, name=name)
        self.vertices.append(v)
        self.logical_ops.append(v)
        self.lattice[pos] = v
        return v

    def add_edge(self, data: DataVertex, op: OpVertex, factor: Pauli):
        """Add edge (tensor product factor in operator)."""
        assert data not in op.neighbours
        assert op not in data.neighbours
        e = Edge(data=data, op=op, factor=factor)
        data.edges.append(e)
        data.neighbours.append(op)
        op.edges.append(e)
        op.neighbours.append(data)
        self.edges.append(e)

    def remove_edge(self, edge: Edge):
        """Remove an edge from the lattice."""
        assert edge in self.edges
        data = edge.data
        op = edge.op
        data.edges.remove(edge)
        data.neighbours.remove(op)
        op.edges.remove(edge)
        op.neighbours.remove(data)
        self.edges.remove(edge)

    def remove_vertex(self, vertex: Vertex):
        """Remove a vertex from the lattice.

        This function raises an error if the vertex is still connected to other
        vertices.
        """
        assert vertex in self.vertices
        assert self.lattice[vertex.pos] is vertex
        assert len(vertex.neighbours) == len(vertex.edges)
        if len(vertex.edges) > 0:
            raise ValueError("Cannot remove vertex with neighbours")
        match vertex:
            case DataVertex():
                self.dataqubits.remove(vertex)
            case StabGenVertex():
                self.stabgens.remove(vertex)
            case LogicalVertex():
                self.logical_ops.remove(vertex)
            case _:
                raise TypeError(f"Using {type(vertex)} directly not supported")
        if vertex.equbit_idx != NO_INDEX:
            self.equbits.remove(vertex)
        self.vertices.remove(vertex)
        self.lattice[vertex.pos] = None

    # FIXME: possible its own function, or a method of Vertex
    def _vertices_to_operators(
        self, ops: Sequence[Vertex], n_qubits: int
    ) -> list[Tableau]:
        """Transform a list of lattice vertices to a list of Pauli operators.

        Args:
            ops: vertices to transform.
            n_qubits: the total number of qubits on which the operators will act.

        Returns:
            a list of "tableau" representations of the operators linked to the vertices.
        """
        # TODO: check if this is actually correct. This is the old behaviour of
        #  PauliList, which would produce an empty PauliList if you passed an empty
        #  list of things in one of its static methods.
        if len(ops) == 0:
            return list()
        ops_as_tableau = []
        for op in ops:
            op_string = ""
            # Gather all single-qubit operators so that we can compose a single,
            # multi-qubit Pauli operator
            operators = {edge.data.dataqubit_idx: edge.factor.name for edge in op.edges}
            for qubit_index in sorted(operators.keys()):
                # There could be "holes" in the operator list, which should be
                # replaced by identities. If the next qubit index is beyond the length
                # of the string, then start adding identities
                actual_index = min(len(op_string), qubit_index)
                while actual_index < qubit_index:
                    op_string += "I"
                    actual_index += 1
                # Afterwards, add the actual operator where it was meant to be
                op_string += operators[qubit_index]
            ops_as_tableau.append(string_to_pauli(op_string, n_qubits))
        return ops_as_tableau

    @property
    def stabilisers(self) -> list[Tableau]:
        """The stabiliser generators operators that make this code."""
        return self._vertices_to_operators(self.stabgens, len(self.dataqubits))

    @property
    def logical_operators(self) -> list[Tableau]:
        """The logical operators of this code."""
        return self._vertices_to_operators(self.logical_ops, len(self.dataqubits))

    @overload
    def get_vertex_at_pos(
        self, pos: PosType, *, allow_empty: Literal[True]
    ) -> Optional[Vertex]:
        ...

    @overload
    def get_vertex_at_pos(self, pos: PosType, *, allow_empty: Literal[False]) -> Vertex:
        ...

    def get_vertex_at_pos(self, pos: PosType, *, allow_empty: bool = True):
        """Get vertex at lattice position.

        This is equivalent to `some_lattice.lattice[pos]` but raises a nicer error
        message if the position is invalid.

        Args:
            pos: The lattice position (tuple of integers)
            allow_empty:
                If `False` and the position does not contain a vertex, then
                raise a `ValueError`.
        """
        try:
            vertex = self.lattice[pos]
        except IndexError as e:
            raise IndexError(f"Lattice position {pos} is invalid") from e
        if vertex is None and not allow_empty:
            raise ValueError(f"Lattice position {pos} is empty")
        return vertex

    def check_equbit_is_data(self, equbit_idx: int, **info):
        """Verify that extended qubit index specifies a physical data qubit.

        Args:
            equbit_idx: The extended qubit index which should be checked.
            **info: Additional information for exception messages.

        Raises:
            ValueError: If the index does not specify a physical data qubit.
        """
        prefix = f"For {info!r}: " if info else ""
        if not isinstance(equbit_idx, int):
            raise ValueError(f"{prefix}Need int, got {equbit_idx!r}")
        if equbit_idx < 0 or equbit_idx >= len(self.equbits):
            raise ValueError(f"{prefix}{equbit_idx=} is out of bounds")
        vertex = self.equbits[equbit_idx]
        if not isinstance(vertex, DataVertex):
            raise ValueError(f"{prefix}Not a data qubit: {vertex!r}")

    def check_equbit_is_stabgen(self, equbit_idx: int, **info):
        """Verify that extended qubit index specifies a stabilizer generator.

        If an extended qubit index refers to a stabilizer generator, it also identifies
        the ancilla qubit which is used to measure the stabilizer generator.

        Args:
            equbit_idx: The extended qubit index which should be checked.
            **info: Additional information for exception messages.

        Raises:
            ValueError: If the index does not specify a stabilizer generator.
        """
        prefix = f"For {info!r}: " if info else ""
        if not isinstance(equbit_idx, int):
            raise ValueError(f"{prefix}Need int, got {equbit_idx!r}")
        if equbit_idx < 0 or equbit_idx >= len(self.equbits):
            raise ValueError(f"{prefix}{equbit_idx=} is out of bounds")
        vertex = self.equbits[equbit_idx]
        if not isinstance(vertex, StabGenVertex):
            raise ValueError(f"{prefix}Not a stabilizer generator: {vertex!r}")

    def create_superstab(self, keep: StabGenVertex, remove: StabGenVertex):
        """Turn two stabilizers into one super-stabilizer.

        Args:
            keep:
                The stabilizer vertex which should become the new super-stabilizer.
            remove:
                The stabilizer vertex which should be removed after joining the two
                stabilizer generators.
        """
        assert isinstance(keep, StabGenVertex)
        assert isinstance(remove, StabGenVertex)
        for edge in tuple(remove.edges):
            self.remove_edge(edge)
            if edge.data in keep.neighbours:
                edge2 = next(e for e in keep.edges if e.data == edge.data)
                assert edge2.factor == edge.factor
                self.remove_edge(edge2)
            else:
                self.add_edge(edge.data, keep, edge.factor)
        self.remove_vertex(remove)

    def create_superstab_from_edges(self, edges: Sequence[Edge]):
        """Create a super-stabilizer from two edges referencing the stabilizers.

        Args:
            edges: Length-2 sequence of edges.

        This essentially calls
        ``self.create_superstab(keep=edges[0].op, remove=edges[1].op).``
        """
        assert len(edges) == 2
        stab1, stab2 = edges[0].op, edges[1].op
        assert isinstance(stab1, StabGenVertex)
        assert isinstance(stab2, StabGenVertex)
        return self.create_superstab(stab1, stab2)

    def multiply_logical_with_stab(
        self, logical_op: LogicalVertex, stab: StabGenVertex
    ):
        """Update a logical operator by multiplying it with a stabilizer generator."""
        for s_edge in stab.edges:
            if s_edge.data in logical_op.neighbours:
                edge = next(e for e in logical_op.edges if e.data == s_edge.data)
                assert edge.factor == s_edge.factor
                self.remove_edge(edge)
            else:
                self.add_edge(s_edge.data, logical_op, s_edge.factor)

    def remove_stabilizer(self, stab: StabGenVertex):
        """Remove a stabilizer including all its edges."""
        for edge in tuple(stab.edges):
            self.remove_edge(edge)
        self.remove_vertex(stab)

    def _fabrication_error_warning(self, stacklevel):
        """Warn if fabrication errors are applied on an untested code."""
        if self.__class__.__name__ != "PlanarCodeLattice":
            warnings.warn(
                "Fabrication errors and removing qubits not tested for "
                f"{type(self)}",
                stacklevel=stacklevel,
            )

    def remove_dataqubit(self, qubit: DataVertex):
        """Remove a data qubit and update the code as needed.

        This function updates both stabilizers generators and logical operators.

        By default, this function proceeds according to the proposal
        :cite:`auger_fault-tolerance_2017`.

        The behaviour of this method is affected by (they can cause deviations from
        :cite:`auger_fault-tolerance_2017`):

        * :attr:`fabrication_err_keep_gauge`
        * :attr:`fabrication_err_boundary_join`

        .. note::

           This function does *not* call :meth:`assign_indices`. You have to reassign
           indices by calling that method yourself.
        """
        self._fabrication_error_warning(stacklevel=3)
        stab_edges: dict[Pauli, list[Edge]] = {}
        logical_edges: list[Edge] = []
        for edge in qubit.edges:
            if isinstance(edge.op, LogicalVertex):
                logical_edges.append(edge)
            elif isinstance(edge.op, StabGenVertex):
                if edge.factor not in stab_edges:
                    stab_edges[edge.factor] = []
                stab_edges[edge.factor].append(edge)
            else:
                raise TypeError(f"{type(edge.op)} not supported")
        # Update logical operators to remove given data qubit.
        for edge in logical_edges:
            if edge.factor not in stab_edges:
                raise NoLogicalOperatorError(edge.factor, edge.data)
            stab = stab_edges[edge.factor][0].op
            assert isinstance(stab, StabGenVertex)
            assert isinstance(edge.op, LogicalVertex)
            self.multiply_logical_with_stab(edge.op, stab)
        self._handle_dataqubit_edgegroups(tuple(stab_edges.values()))
        self.remove_vertex(qubit)

    def _handle_dataqubit_edgegroups(self, edgegroups: Sequence[Sequence[Edge]]):
        """Handle removal of edges before removing a data qubit."""
        if all(len(edges) == 2 for edges in edgegroups):
            # Non-boundary data qubit: Can form super-stabilizers from pairs in all
            # cases.
            if self.fabrication_err_keep_gauge == "none":
                for edges in edgegroups:
                    self.create_superstab_from_edges(edges)
            elif self.fabrication_err_keep_gauge == "half":
                self.create_superstab_from_edges(edgegroups[0])
                for edges in edgegroups[1:]:
                    self.remove_edge(edges[0])
                    self.remove_edge(edges[1])
            elif self.fabrication_err_keep_gauge == "all":
                for edges in edgegroups:
                    self.remove_edge(edges[0])
                    self.remove_edge(edges[1])
            else:
                raise ValueError(f"{self.fabrication_err_keep_gauge=} unsupported")
        else:
            # Boundary data qubit: Cannot form super-stabilizers from pairs in some
            # cases.
            if self.fabrication_err_boundary_join:
                self._boundary_qubit_join_stabilizers(edgegroups)
            else:
                self._boundary_qubit_remove_stabilizer(edgegroups)

    def _boundary_qubit_remove_stabilizer(self, edgegroups: Iterable[Sequence[Edge]]):
        """Handle removal of boundary data qubit by disabling a stabilizer.

        This is the behaviour proposed in :cite:`auger_fault-tolerance_2017`.
        """
        for edges in edgegroups:
            if len(edges) == 1:
                # Disable the entire stabilizer (also remoles edges[0]).
                self.remove_stabilizer(cast(StabGenVertex, edges[0].op))
            elif len(edges) == 2:
                # Remove edges but keep the stabilizers.
                self.remove_edge(edges[0])
                self.remove_edge(edges[1])
            else:
                raise ValueError(f"Edge group had unsupported length {len(edges)}")

    def _boundary_qubit_join_stabilizers(self, edgegroups: Iterable[Sequence[Edge]]):
        """Handle removal of a boundary data qubit by forming one super-stabilizer.

        The third stabilizer is kept.
        """
        for edges in edgegroups:
            if len(edges) == 1:
                self.remove_edge(edges[0])
            elif len(edges) == 2:
                self.create_superstab_from_edges(edges)
            else:
                raise ValueError(f"Edge group had unsupported length {len(edges)}")

    def apply_fabrication_errors(
        self,
        *,
        faulty_qubits: Sequence[int] = (),
        faulty_gates: Sequence[tuple[Tableau, int]] = (),
        faulty_stabilizers: Sequence[Tableau] = (),
        keep_gauge: Optional[str] = None,
        boundary_join: Optional[bool] = None,
    ):
        """Change the code lattice by applying fabrication errors.

        This function proceeds mostly according to the proposal
        :cite:`auger_fault-tolerance_2017` (assuming that the following flags have
        their default values: :attr:`fabrication_err_keep_gauge`,
        :attr:`fabrication_err_boundary_join`).

        Data qubits and stabilizers generators can be specified as lattice positions
        (tuples) or via extended qubit indices (integers, see :class:`Vertex
        <plaquette.codes.latticebase.Vertex>`).

        Args:
            faulty_qubits: Sequence of data qubits.
            faulty_gates: Sequence of ``(stabilizer_generator, dataqubit)`` pairs.
            faulty_stabilizers: Sequence of stabilizer generators.
            keep_gauge: Updates :attr:`fabrication_err_keep_gauge`.
            boundary_join: Updates :attr:`fabrication_err_boundary_join`.

        .. note::

           Applying this function changes extended qubit indices. After applying this
           function, extended qubit indices updated to be consecutive without gaps.

        .. note::

           Using this function on codes other than the planar code is experimental
           and will cause a warning.
        """
        self._fabrication_error_warning(stacklevel=3)
        if keep_gauge is not None:
            self.fabrication_err_keep_gauge = keep_gauge
        if boundary_join is not None:
            self.fabrication_err_boundary_join = boundary_join
        remove_qubits = self._fabrication_errors_get_remove_qubits(
            faulty_qubits, faulty_gates, faulty_stabilizers
        )
        for _, qubit in sorted(remove_qubits.items()):
            assert isinstance(qubit, DataVertex)
            # NB: disable_qubit() removes entries from self.equbits. All the indices
            # are no longer valid after the first call to remove_dataqubit().
            self.remove_dataqubit(qubit)
        self.assign_indices()

    def _fabrication_errors_get_remove_qubits(
        self,
        faulty_qubits: Sequence = (),
        faulty_gates: Sequence = (),
        faulty_stabilizers: Sequence = (),
    ) -> dict[PosType, DataVertex]:
        """Convert fabrication errors to dict containing data qubits to be removed.

        This is a super-boring function which does nothing than parameter
        verification and parameter conversion.
        """
        remove_qubits: dict[PosType, DataVertex] = {}
        # Disable faulty qubits.
        for pos in faulty_qubits:
            qubit = self.lattice[pos] if isinstance(pos, tuple) else self.equbits[pos]
            if not isinstance(qubit, DataVertex):
                raise ValueError(f"{pos} is not a data qubit")
            remove_qubits[qubit.pos] = qubit
        # Handle faulty gate by disabling the affected data qubit.
        for stabgen_pos, data_pos in faulty_gates:
            if isinstance(stabgen_pos, tuple):
                stabgen = self.lattice[stabgen_pos]
            else:
                stabgen = self.equbits[stabgen_pos]
            if isinstance(data_pos, tuple):
                qubit = self.lattice[data_pos]
            else:
                qubit = self.equbits[data_pos]
            if not isinstance(stabgen, StabGenVertex):
                raise ValueError(f"{stabgen_pos} is not a stabilizer generator")
            if not isinstance(qubit, DataVertex):
                raise ValueError(f"{data_pos} is not a data qubit")
            if stabgen not in qubit.neighbours:
                raise ValueError(
                    f"There is no controlled gate between ancilla {stabgen_pos} and "
                    f"data qubit {data_pos}"
                )
            remove_qubits[qubit.pos] = qubit
        # Handle faulty stabilizer by disabling all related data qubits.
        for pos in faulty_stabilizers:
            stab = self.lattice[pos] if isinstance(pos, tuple) else self.equbits[pos]
            if not isinstance(stab, StabGenVertex):
                raise ValueError(f"{pos} is not a stabilizer generator")
            for edge in stab.edges:
                remove_qubits[edge.data.pos] = edge.data
        return remove_qubits


# TODO: most probably candidate for static method
class CodeLatticeFromStab(CodeLattice):
    """Define code using string descriptions of stabilizers and logicals.

    .. automethod:: __init__
    """

    # TODO: Merge this class with CodeLattice?

    #: Number of data qubits
    n_data_qubits: int
    #: Stabilizer generator definition (sequence of strings)
    def_stabgens: Sequence[str]
    #: Logical operator definition (sequence of strings). For the order, see
    #: :attr:`.latticebase.CodeLattice.logical_ops`.
    def_logical_ops: Sequence[str]

    def __init__(
        self,
        n_data_qubits: Optional[int] = None,
        def_stabgens: Optional[Sequence[str]] = None,
        def_logical_ops: Optional[Sequence[str]] = None,
    ):
        """Create new instance of the code.

        If you use a subclass of ``CodeLatticeFromStab``, it is not necessary to supply
        any arguments to the constructor.

        If you use ``CodeLatticeFromStab`` directly, you have to supply the following
        arguments:

        Args:
            n_data_qubits: Number of data qubits on which operators act.
            def_stabgens: Sequence of strings describing stabilizer generators.
            def_logical_ops: Sequence of strings describing logical operators.
                The order must conform to
                :attr:`.latticebase.CodeLattice.logical_ops`.

        For example values, see identically named attributes of
        :class:`.latticeinstances.FiveQubitCodeLattice`.
        """
        if not hasattr(self, "n_data_qubits"):
            if n_data_qubits is None:
                raise TypeError("n_data_qubits is required")
            self.n_data_qubits = n_data_qubits
        if not hasattr(self, "def_stabgens"):
            if def_stabgens is None:
                raise TypeError("def_stabgens is required")
            self.def_stabgens = def_stabgens
        if not hasattr(self, "def_logical_ops"):
            if def_logical_ops is None:
                raise TypeError("def_logical_ops is required")
            self.def_logical_ops = def_logical_ops
        super().__init__((self.n_data_qubits, 3))
        for i in range(self.n_data_qubits):
            self.add_data((i, 1))
        for i, stab in enumerate(self.def_stabgens):
            self.add_stabgen((i, 0), StabGroup.U)
            self._add_edges_from_str(self.lattice[i, 0], stab)
        for i, log in enumerate(self.def_logical_ops):
            name = "XZ"[i % 2] + str(i // 2)
            self.add_logical((i, 2), name)
            self._add_edges_from_str(self.lattice[i, 2], log)
        self.assign_indices()

    def _add_edges_from_str(self, op: OpVertex, pauli_str: str):
        # ignore sign in operator string
        if pauli_str[0] in "+-":
            pauli_str = pauli_str[1:]
        assert len(pauli_str) == self.n_data_qubits
        for qi, pauli in enumerate(pauli_str):
            if pauli != "I":
                self.add_edge(self.lattice[qi, 1], op, getattr(Pauli, pauli))
