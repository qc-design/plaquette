# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
r"""Decoders and related functions.

All decoders in ``plaquette`` have the same interface, and adding new
ones should be straightforward. Here we detail the inner workings of
the base class that all decoder inherit from: :class:`AbstractDecoder`.

This class takes care of some common tasks that all decoder
subclasses can take advantage of.


Decoding graph initialization
-----------------------------

First, it creates the **decoding** graph from the code given as an
input. This decoding graph is stored in :attr:`._decoding_graph`,
and it's marked private because in principle no user should ever need
to interact with it, **but** decoder subclasses will find it useful
to translate its structure to whatever is necessary for them to work.
The decoding graph makes no distinction between "X" and "Z" components.

.. seealso::

   :class:`.AnnotatedSparseGraph`.

.. important::

   If decoders cannot deal with multiple components at once or with
   isolated vertices they then they MUST deal with this fact on its own.

Since decoders can deal with multiple round of measurements of the
same code, each decoder needs to know the number of rounds, which is
given during initialization (see :meth:`__init__`). At each round
:math:`r`, starting at 0, nodes are indexed always in the same way,
but with an offset :math:`\delta = rN`, where :math:`N` is the
number of both real and *virtual* ancilla qubits.

The decoding graph will also add **a new virtual node** (i.e. a node
that does not represent a physical qubit, neither data nor ancilla)
when any error mechanism would corrupt only one syndrome value
instead of a pair. These virtual nodes are collected into the
attribute :attr:`_virtual_ancillas`. A similar attribute exists for
:attr:`_virtual_edges`.

Syndrome and erasure information
--------------------------------

Syndrome (erasure) information is an array of booleans which
indicate whether the corresponding ancilla (data) qubit
was affected by an error (erasure). Each syndrome bit corresponds to the
stabilizer measurement defined in the :class:`.Code` used to initialize
the decoder, such that ``syndrome[i]`` corresponds to the measurement
information of ``Code.stabilizers[i]``.
"""
import abc
import copy
import pathlib
import typing as t
import warnings

import fusion_blossom as fb
import numpy as np
import pymatching

from plaquette import codes, errors, pauli
from plaquette.codes import graph
from plaquette_graph import DecodingGraph
from plaquette_unionfind import UnionFindDecoder as CppUF

NodeIndex = t.NewType("NodeIndex", int)
"""Typing helper to mark specific ints as node indices."""
EdgeIndex = t.NewType("EdgeIndex", int)
"""Typing helper to mark specific ints as edge indices."""


def check_success(
    code: codes.Code,
    correction: t.Sequence[pauli.Tableau],
    logical_op_toggle: np.ndarray,
    logical_op: str,
):
    """Compare measured with predicted logical operator results.

    This function compares measured logical operators with the prediction from the
    decoder. If they match, this test of QEC can be considered successful.

    Args:
        code: Code definition containing logical operators
        correction: Correction Pauli operator from the decoder
        logical_op_toggle: XOR between logical op. measurement result
            before and after QEC
        logical_op: Specify which logical operators were measured

    Notes:
        This function does the following:

        * Flip logical operators according to the correction Pauli
          frame from the decoder.
        * Check whether the predicted signs of logical operators agree
          with measurement results.

        For more details, see :doc:`/quickstart`.
    """
    # TODO: this needs better handling in the case of codes with multiple logical qubits
    if len(logical_op) != 1:
        raise NotImplementedError("currently only a single logical qubit is supported")
    assert (
        logical_op in "XZ"
    ), "currently only X and Z logical operators are supported on a logical qubit"
    log_op = code.logical_ops[logical_op == "Z"]
    # FIXME: op_toggle comes from MeasurementSample, which needs updating to calculate
    #  stuff only for a single logical operator at a time
    assert pauli.count_qubits(correction)[0] == code.num_data_qubits
    decoder_prediction = (pauli.commutator_sign(log_op, correction)).ravel()
    return (logical_op_toggle == decoder_prediction).all()


class AbstractDecoder(abc.ABC):
    """The base class/interface for all decoders."""

    def __init__(
        self, code: codes.Code, error_data: errors.ErrorDataDict, n_rounds: int
    ):
        """Create a decoder tailored on a specific code and for the given rounds.

        Args:
            code: the code whose results you want to decode.
            error_data: information about the error model used. Used to
                calculate edge weights of the decoding graph.
            n_rounds: number of measurement rounds your code went through.

        Notes:
            **Do not** construct a new decoder each time you want to decode a
            new syndrome. If the underlying code and number of measurements do
            not change, neither does the decoding graph.

            Using error probabilities above 50% will results in negative weights, which
            will most likely cause problems decoders are not equipped to deal with.
        """
        self.selection: t.Sequence[EdgeIndex] = tuple()
        """The cached selection from the decoder.

        After calling :meth:`decode`, this attribute will be set to the
        results of the decoder. Useful mainly for debugging or
        visualisation purposes.
        """
        self._code = code
        self._n_rounds = n_rounds
        self._data_qubit_index_for_edge: dict[EdgeIndex, NodeIndex] = {}
        """A mapping of edges to qubit indices.

        This dict has as many elements as the decoding graph has edges.
        The i-th edge connects two stabilizer checks whose outcomes have been
        flipped by the action of an error on the data qubit whose index is the
        i-th element of this list.
        """
        # Take a list of all possible error operators that could act on a
        # specific data qubit
        error_operators = [
            pauli.single_qubit_pauli_operator(o, i, self._code.num_data_qubits)
            for o in "XZ"
            for i in range(self._code.num_data_qubits)
        ]
        self._virtual_ancillas: list[NodeIndex] = []
        """Ancilla indices which do not belong to the original code graph."""
        self._virtual_edges: list[EdgeIndex] = []
        """Decoding graph edges *indices* containing virtual ancillas."""
        edges: list[tuple[NodeIndex, NodeIndex]] = list()
        """Edges of the decoding graph, both virtual and not."""
        edges_data: list[t.Optional[codes.EdgeMetadata]] = list()
        """Indicates for each edge which Pauli error would flip the connected nodes."""
        self._edges_weights: np.ndarray[t.Any, np.uint32]  # assigned below
        """A list of weights for each edge in the decoding graph."""
        commutator_signs = pauli.commutator_sign(
            error_operators, self._code.stabilizers
        )
        """2D array telling whether the i-th error commutes with the j-th stabilizer.

        For each error operator, we compute the commutator sign with all
        possible stabilizers. An **edge** in the decoding graph can only exist
        between stabilizer measurements that **do not** commute with the given
        error operator.

        This holds the commutator sign of the given operator pointed
        by op_idx with all stabilizers. For a given index ``i``,
        ``commutator_signs[i]`` is 0 if ``code.stabilizers[i]``
        commutes with ``error_operators[op_idx]``.
        """

        # If there are dangling edges, we need to add fake/virtual ancillas
        # but in order to layer the various rounds properly, we need to know
        # in advance how many virtual ancillas we need
        num_virtual_ancilla_vertices: int = np.count_nonzero(
            np.array(list(map(lambda x: len(np.nonzero(x)[0]), commutator_signs))) == 1
        )
        self._nodes_per_round = (
            num_virtual_ancilla_vertices + self._code.num_stabilizers
        )
        r"""Number of nodes per round necessary to build the decoding graph.

        The decoding graph for a code is made by layering the code graph on top
        of it self as many times as there are measurement rounds. The decoding
        graph will clearly have many more nodes than the graph describing the
        code itself, but each node will have the same index per round, assuming
        we take the modulo with the total number of nodes per round.

        As an example, the distance-3 rotated planar code has :math:`A` real
        ancilla nodes, plus :math:`V`
        virtual ancillas (to avoid dangling edges), so there's :math:`N = A+V`
        nodes per round.

        At each round, we offset the next node index in the decoding graph by
        :math:`N r`, where :math:`r` is the current round number. In this way,
        whenever we want to know which ancilla measurement we are referring to
        with a given index :math:`j` in the decoding graph, we can obtain
        this information by :math:`a = j \bmod N`.
        """
        nodes_data: list[codes.NodeMetadata | None] = [None] * (
            self._nodes_per_round * n_rounds
        )
        """Decoding graph metadata, mostly used for visualization."""

        # region Edge weights calculation

        # First we use the information in error_data to construct the
        # edges' weights. Multiple error mechanisms acting on the same
        # qubit will be merged into an effective "fault" probability
        # that will be used to calculate the effective weights.
        #
        # Faults are always reduced to effectively three mechanisms,
        # because this is what decoders understand: X errors, Z errors
        # or measurement errors. Y errors are not included because they
        # are "split" among X and Z errors appropriately. Erasure errors
        # are effectively half the time X errors and half the time Z
        # errors.

        eff_error_probabilities: dict[str, dict[NodeIndex, float]] = {
            "x": {},
            "z": {},
            "measurement": {},
        }
        pauli_errors = error_data.get("pauli", None)
        erasure_errors = error_data.get("erasure", None)
        measurement_errors = error_data.get("measurement", None)
        for v in code.data_qubit_indices:
            # data qubits are affected by either pauli or erasure errors
            p_x = 0.0
            p_z = 0.0
            if pauli_errors and (qubit_errors := pauli_errors.get(v)):
                p_x += qubit_errors.get("x", 0.0) + qubit_errors.get("y", 0.0)
                p_z += qubit_errors.get("z", 0.0) + qubit_errors.get("y", 0.0)
            if erasure_errors and (qubit_erasures := erasure_errors.get(v)):
                # erasure errors erase (duh) the qubit being manipulated
                # so either a normal X/Z error happens OR and erasure
                # happens, substituting this qubit with a fully
                # depolarized one. Both can't happen
                e = qubit_erasures.get("p", 0.0)
                p_x = (1 - p_x) * e / 2 + p_x * (1 - e / 2)
                p_z = (1 - p_z) * e / 2 + p_z * (1 - e / 2)
            eff_error_probabilities["x"][v] = p_x
            eff_error_probabilities["z"][v] = p_z

        for v in range(code.num_stabilizers):
            # and now we add error probabilities to ancilla qubits for
            # the time-like edges' weights
            v = t.cast(NodeIndex, v + code.num_data_qubits)
            p_m = 0.0
            if measurement_errors and (meas_error := measurement_errors.get(v)):
                p_m = meas_error.get("p", 0.0)
            eff_error_probabilities["measurement"][v] = p_m
        # endregion

        edges_weights: list[float] = list()
        # Used later to calculate positions of virtual ancillas, but
        # we cache it here for performance reasons
        data_qubits_coords = [
            code.tanner_graph.nodes_data[n].coords for n in code.data_qubit_indices
        ]
        data_qubits_coords_incomplete = not all(data_qubits_coords)
        data_qubits_center_of_mass = graph.center_of_mass(
            [graph.Position(x=c[0], y=c[1], z=c[2]) for c in data_qubits_coords]
        )

        # The decoding graph takes into account also all measurement rounds, and
        # we only need to create edges between the same data qubits across rounds.
        # This means that you can always recover the data qubit index in the code
        # from the decoding graph by taking its index % code.num_data_qubits.
        for meas_round in range(self._n_rounds):
            # region Single-layer construction
            # We also need to "fix" dangling edges, where an error operator
            # anti-commutes with only one stabilizer instead of two. We use
            # virtual/fake ancillas, whose indices lie out of the range
            # of indices used by the code, which is as follows
            next_virtual_ancilla_idx = self._code.num_stabilizers
            for op_idx, comm_signs in enumerate(commutator_signs):
                non_commuting_stabilizers = np.nonzero(comm_signs)[0]

                if not (n := len(non_commuting_stabilizers)):
                    # We skip error operators that commute with everything, they will
                    # not contribute to the decoding graph
                    continue
                elif n == 1:
                    # This is a "dangling edge", and we connect it with a virtual
                    # node whose index is defined as the total number of qubits of
                    # the code (data + ancilla) such that it can never be mistaken
                    # for a "real" qubit index
                    non_commuting_stabilizers = (
                        non_commuting_stabilizers[0],
                        next_virtual_ancilla_idx,
                    )
                    next_virtual_ancilla_idx += 1
                elif n > 2:
                    # Currently we don't support hyper-edges, so we need to
                    # avoid this case
                    raise ValueError(
                        f"a qubit being shared among {n} stabilizers "
                        "(a hyper-edge in the decoding graph) is not supported"
                    )

                # Check which factor we need to assign to this edge
                factors = [
                    (k, v)
                    for k, v in pauli.pauli_to_dict(error_operators[op_idx]).items()
                ]
                assert len(factors) == 1, "error operator must act only on one qubit"

                # link decoding graph edge with the corresponding data qubit in
                # the code's graph
                data_qubit_index = t.cast(NodeIndex, factors[0][0])
                pauli_factor = factors[0][1]

                # by offsetting the stabilizer index by num_nodes*meas_round we
                # make sure that we can convert from decoding graph index to real qubit
                # (ancilla) index in a simple way:
                #
                #   ancilla_index = dec_graph_index%num_nodes
                vert_a: NodeIndex = (
                    self._nodes_per_round * meas_round + non_commuting_stabilizers[0]
                )
                vert_b: NodeIndex = (
                    self._nodes_per_round * meas_round + non_commuting_stabilizers[1]
                )
                edges.append((vert_a, vert_b))
                edge_idx = t.cast(EdgeIndex, len(edges) - 1)
                self._data_qubit_index_for_edge[edge_idx] = data_qubit_index
                edges_data.append(codes.EdgeMetadata(type=pauli_factor))
                # recover error probability for the error specified by the edge
                # factor and by which data qubit this edge "crosses", or alternatively
                # by which faulty data qubit would "create" this edge
                p = eff_error_probabilities[pauli_factor.name.lower()][
                    self._data_qubit_index_for_edge[edge_idx]
                ]
                edges_weights.append(-np.log(p / (1 - p)))
                # Some decoders like to know which one is a real ancilla and which one
                # is not, so we keep track of them.
                #
                # Since virtual ancillas are added at each round, we need to check
                # that the current vertex index being added exceeds the correct amount
                # of nodes indices to be marked as virtual *in that round*
                if (vert_a % self._nodes_per_round) >= code.num_stabilizers:
                    self._virtual_ancillas.append(vert_a)
                    self._virtual_edges.append(t.cast(EdgeIndex, len(edges) - 1))
                    nodes_data[vert_b] = copy.deepcopy(
                        code.tanner_graph.nodes_data[
                            (vert_b % self._nodes_per_round) + code.num_data_qubits
                        ]
                    )
                elif (vert_b % self._nodes_per_round) >= code.num_stabilizers:
                    self._virtual_ancillas.append(vert_b)
                    self._virtual_edges.append(t.cast(EdgeIndex, len(edges) - 1))
                    nodes_data[vert_a] = copy.deepcopy(
                        code.tanner_graph.nodes_data[
                            (vert_a % self._nodes_per_round) + code.num_data_qubits
                        ]
                    )
                else:
                    nodes_data[vert_a] = copy.deepcopy(
                        code.tanner_graph.nodes_data[
                            (vert_a % self._nodes_per_round) + code.num_data_qubits
                        ]
                    )
                    nodes_data[vert_b] = copy.deepcopy(
                        code.tanner_graph.nodes_data[
                            (vert_b % self._nodes_per_round) + code.num_data_qubits
                        ]
                    )

                if (nd := nodes_data[vert_a]) is not None and nd.coords is not None:
                    nd.coords[2] = meas_round
                if (nd := nodes_data[vert_b]) is not None and nd.coords is not None:
                    nd.coords[2] = meas_round

            # endregion
            # region Time-like layer linking
            # Now we also link the current layer of ancillas to the previous round
            if meas_round > 0:
                for ancilla in range(code.num_stabilizers):
                    ancilla = t.cast(NodeIndex, ancilla)
                    edges.append(
                        t.cast(
                            tuple[NodeIndex, NodeIndex],
                            (
                                self._nodes_per_round * meas_round + ancilla,
                                self._nodes_per_round * (meas_round - 1) + ancilla,
                            ),
                        )
                    )
                    edges_data.append(codes.EdgeMetadata(pauli.Factor.I))
                    # by default this time-like edge has no error probability, because
                    # we assume that it's a virtual ancilla
                    p = 0.0
                    if ancilla < code.num_stabilizers:
                        # if it is a real ancilla instead, it might have a measurement
                        # error probability greater than zero
                        ancilla = t.cast(NodeIndex, ancilla + code.num_data_qubits)
                        p = eff_error_probabilities["measurement"][ancilla]
                    edges_weights.append(-np.log(p / (1 - p)))
            # endregion

        # region Virtual ancilla placements
        # Now the last thing to do is to *place* the virtual ancillas
        # for visualization. This can only happen if all data qubits
        # have an assigned position themselves.
        if not data_qubits_coords_incomplete:
            for v_edge in self._virtual_edges:
                a, b = edges[v_edge]
                # sort which one is the virtual ancilla and which is not
                if a % self._nodes_per_round > code.num_stabilizers:
                    v_anc = a
                    r_anc = b
                else:
                    v_anc = b
                    r_anc = a
                # We find now a radial vector starting from the code
                # center of mass to the real ancilla position. We will
                # place the virtual ancilla along the direction of this
                # vector, with a small offset
                # TODO: make the offset configurable
                nd = code.tanner_graph.nodes_data[
                    r_anc % self._nodes_per_round + code.num_data_qubits
                ]
                assert nd is not None, "node data missing"
                coords = nd.coords
                assert coords is not None, "coordinates missing"

                x, y, z = coords
                r_pos = graph.Position(x=x, y=y, z=z)
                v_pos = (r_pos - data_qubits_center_of_mass).unit() + r_pos
                v_pos.z = r_anc // self._nodes_per_round
                nodes_data[v_anc] = codes.NodeMetadata(
                    codes.QubitType.virtual, list(v_pos)
                )
        # endregion

        self._decoding_graph = codes.AnnotatedSparseGraph(nodes_data, edges, edges_data)
        self._edges_weights = np.array(edges_weights)

    @abc.abstractmethod
    def select_edges(
        self,
        syndrome: t.Sequence[bool],
        erased_nodes: t.Optional[t.Sequence[bool]] = None,
    ) -> t.Sequence[bool]:
        """Given a syndrome, return the selected edges indices given by the decoder.

        Args:
            syndrome: the computed syndrome to decode.
            erased_nodes: an array of booleans indicating which node at
                which round suffered erasure errors. ``None`` if the
                given decoder has no use for this.

        Returns:
            A list of bools, one per edge, indicating which was selected
            during decoding.

        Notes:
            Decoders must override this method in order to implement their logic. Each
            decoder will be given a syndrome, which is a set of vertices which have
            been "flipped", and erasure information, another set of vertices which
            have been "erased" across all rounds.
        """
        raise NotImplementedError

    def decode(
        self,
        syndrome: t.Sequence[bool],
        erased_nodes: t.Optional[t.Sequence[bool]] = None,
    ) -> pauli.Tableau:
        """Decode a given syndrome, calculating the most likely correction operator.

        Args:
            syndrome: the computed syndrome to decode.
            erased_nodes: an array of booleans indicating which node at
                which round suffered erasure errors.

        Returns:
            an operator that, when applied to the state of the system, should
            correct the errors detected by the code. The returned operator acts
            only on the **data** qubits of the state.

        See Also:
            :mod:`~plaquette.decoders` for an explanation of what type of
            syndrome and erasure information is expected here.
        """
        # Before doing anything, the syndrome that internally we deal with is
        # different from the one used by decoders. Then one returned by a Device does
        # not have any information about virtual edges, so we need to pad it
        # appropriately at each round
        assert (len(syndrome) // self._code.num_stabilizers) == self._n_rounds, (
            f"wrong number of syndrome bits ({len(syndrome)}) for the given code "
            f"you want to correct and given rounds ({self._n_rounds})"
        )
        if erased_nodes is None:
            erased_nodes = np.zeros_like(syndrome)

        # for each round of measurements, we need to XOR whatever edge was selected
        # at round r with the result at round r+1. This will give us the final
        # list of error mechanisms that we need to correct.
        # TODO: maybe this needs better explaining?
        errors = np.zeros(2 * self._code.num_data_qubits, dtype=int)
        self.selection = np.flatnonzero(self.select_edges(syndrome, erased_nodes))
        for edge in t.cast(t.Sequence[int], self.selection):
            edge = t.cast(EdgeIndex, edge)
            edge_data = self._decoding_graph.edges_data[edge]
            assert edge_data is not None, "edge data missing"
            match edge_data.type:
                case pauli.Factor.I:
                    # Nothing to do for measurement edges
                    pass
                case pauli.Factor.X:
                    errors[self._data_qubit_index_for_edge[edge]] += 1
                case pauli.Factor.Z:
                    errors[
                        self._data_qubit_index_for_edge[edge]
                        + self._code.num_data_qubits
                    ] += 1
                case pauli.Factor.Y:
                    raise RuntimeError(
                        "encountered Y error in decoder correction, which "
                        "should not happen. This is probably a bug and should be "
                        "reported!"
                    )
        correction = {
            # fmt: off
            faulty_qubit % self._code.num_data_qubits: pauli.Factor[
                "XZ"[faulty_qubit // self._code.num_data_qubits]  # either 0 or 1
            ]
            # fmt: on
            for faulty_qubit in np.flatnonzero(errors % 2)
        }
        if correction:
            return pauli.dict_to_pauli(correction, self._code.num_data_qubits)
        else:
            return pauli.string_to_pauli("I", self._code.num_data_qubits)

    def results_to_json(self, file_name: pathlib.Path | str):
        """Save a JSON representation of this decoding graph.

        The output JSON file can be visualized with
        `plaquette-viz <https://github.com/qc-design/plaquette-viz>`_.

        In the same folder as the one where the given ``file_name``
        lives, a new ``schema.json`` will be written. This is the
        JSON schema of the general graph JSON file that the visualiser
        expects.

        .. todo::

            The schema writing is done for debugging and it should be
            removed before release.

        Args:
            file_name: file where to store the graph. If it exists, it
                will be overwritten without warning!
        """
        file_name = pathlib.Path(file_name).absolute()
        decoding_edges = list()
        for edge, e_data in enumerate(self._decoding_graph.edges_data):
            a, b = self._code.tanner_graph.get_vertices_connected_by_edge(edge)
            edge_type = e_data.type.name if e_data is not None else "Time"
            decoding_edges.append(graph.Edge(a=a, b=b, type=edge_type))
        base = self._decoding_graph.to_pydantic_model()
        dec_graph = graph.DecodingGraph(
            nodes=base.nodes,
            edges=base.edges,
            selection=list(self.selection),
            faults=list(),
            virtual_nodes=self._virtual_ancillas,
        )
        file_name.write_text(dec_graph.json())
        file_name.with_name("schema.json").write_text(dec_graph.schema_json())


class FusionBlossomDecoder(AbstractDecoder):
    """An interface to the ``fusion-blossom`` library decoder.

    See Also:
        https://github.com/yuewuo/fusion-blossom
    """

    def __init__(self, code: codes.Code, error_data: errors.ErrorDataDict, n_rounds=1):
        """Initialize a ``SolverSerial`` for decoding with ``fusion-blossom``."""
        super().__init__(code, error_data, n_rounds)
        # rescale weights and turn them into ints
        w = (1000 * self._edges_weights / np.max(self._edges_weights)).astype(int)
        edges: list[tuple[NodeIndex, NodeIndex, int]] = [
            (
                *self._decoding_graph.get_vertices_connected_by_edge(e),  # type: ignore
                2 * w[e],
            )
            for e in range(self._decoding_graph.get_num_edges())
        ]
        self._solver = fb.SolverSerial(
            fb.SolverInitializer(
                self._decoding_graph.get_num_vertices(), edges, self._virtual_ancillas
            )
        )

    def select_edges(
        self,
        syndrome: t.Sequence[bool],
        erased_nodes: t.Optional[t.Sequence[bool]] = None,
    ) -> t.Sequence[bool]:
        """Return an edge selection after creating a ``SyndromePattern``."""
        pattern = fb.SyndromePattern(
            syndrome_vertices=np.flatnonzero(syndrome),
            erasures=np.flatnonzero(erased_nodes) if erased_nodes is not None else [],
        )
        self._solver.solve(pattern)
        selection = np.zeros(self._decoding_graph.get_num_edges(), dtype=bool)
        selection[self._solver.subgraph(None)] = True
        self._solver.clear()
        return selection


class PyMatchingDecoder(AbstractDecoder):
    """An interface to the ``PyMatching`` (v2) decoder.

    See Also:
        https://github.com/oscarhiggott/PyMatching
    """

    def __init__(self, code: codes.Code, error_data: errors.ErrorDataDict, n_rounds=1):
        """Initialize a new PyMatching decoder instance."""
        super().__init__(code, error_data, n_rounds)
        # Before starting, PyMatching does not like infinite weights,
        # so we need to replace them with the maximum allowed value
        self._edges_weights[self._edges_weights == np.inf] = 2**24 - 1
        # Initialise empty decoder
        self._pym = pymatching.Matching()
        # Copy our graph structure into pymatching
        for edge in range(self._decoding_graph.get_num_edges()):
            a, b = self._decoding_graph.get_vertices_connected_by_edge(edge)
            if a in self._virtual_ancillas and b in self._virtual_ancillas:
                # We ignore time-like edges for virtual ancillas, they will never
                # have errors
                continue
            self._pym.add_edge(a, b, fault_ids=edge, weight=self._edges_weights[edge])

            # PyMatching concept of boundary edges is different from ours
            # if a not in self._virtual_ancillas and b not in self._virtual_ancillas:
            #     self._pym.add_edge(a, b)
            # elif a in self._virtual_ancillas:
            #     self._pym.add_boundary_edge(a)
            # else:
            #     self._pym.add_boundary_edge(b)
        self._pym.set_boundary_nodes(set(self._virtual_ancillas))

    def select_edges(
        self,
        syndrome: t.Sequence[bool],
        erased_nodes: t.Optional[t.Sequence[bool]] = None,
    ) -> t.Sequence[bool]:
        """Return an array stating which edge was selected by PyMatching."""
        if erased_nodes is not None:
            warnings.warn("pymatching decoder does not use erasure", stacklevel=2)
        # fmt: off
        s = np.append(
            np.array(syndrome, dtype=int).reshape((self._n_rounds, self._code.num_stabilizers)),  # noqa: 501
            np.zeros((self._n_rounds, len(self._virtual_ancillas)//self._n_rounds), dtype=int),  # noqa: 501
            axis=1
        ).ravel()
        # fmt: on
        return self._pym.decode(s)


class UnionFindDecoder(AbstractDecoder):
    """A python interface to the C++ UnionFind decoder.

    See Also:
        https://github.com/qc-design/plaquette-unionfind
    """

    def __init__(self, code: codes.Code, error_data: errors.ErrorDataDict, n_rounds=1):
        """Initialize the UnionFind decoder."""
        super().__init__(code, error_data, n_rounds)
        edges = [
            self._decoding_graph.get_vertices_connected_by_edge(e)
            for e in range(self._decoding_graph.get_num_edges())
        ]
        is_boundary = np.zeros(self._decoding_graph.get_num_vertices(), dtype=bool)
        is_boundary[self._virtual_ancillas] = True
        self._dg = DecodingGraph(
            self._decoding_graph.get_num_vertices(), edges, is_boundary
        )
        self._uf = CppUF(self._dg)

    def select_edges(
        self,
        syndrome: t.Sequence[bool],
        erased_nodes: t.Optional[t.Sequence[bool]] = None,
    ) -> t.Sequence[bool]:
        """Return decoder's edge selection."""
        return self._uf.decode(syndrome, erased_nodes)
