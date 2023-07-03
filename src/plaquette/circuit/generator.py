# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Generate Clifford circuit for QEC simulations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, cast

from plaquette import circuit, codes, pauli
from plaquette.errors import (
    ErrorData,
    ErrorDataDict,
    GateErrorsDict,
    QubitErrorsDict,
)

# TODO: The code here is closer in functionality and concept to simulator._circuitsim


class QECCircuitGenerator:
    """Generate Clifford circuit for QEC simulations.

    This class can be used to generate a circuit which performs quantum error
    correction. For this purpose, it needs information about the code, the errors
    and the logical operators of interest (see :mod:`plaquette.codes` and
    :mod:`plaquette.errors`):

    >>> from plaquette import codes
    >>> gen = QECCircuitGenerator(codes.Code.make_planar(4), {}, {}, 1)
    >>> circ = gen.get_circuit("X")

    The above is equivalent to:

    .. code-block:: python

        circ = generate_qec_circuit(code, {}, {}, "X", 1)

    See :func:`generate_qec_circuit` for a complete minimal example.
    """

    #: QEC code which should be implemented
    code: codes.Code
    #: Qubit errors data
    qubit_errors: QubitErrorsDict
    #: Gate error data
    gate_errors: GateErrorsDict

    #: List of names of supported errors
    supported_errors: tuple[str | tuple[str, str], ...] = (
        "pauli",
        "erasure",
        "measurement",
        "CX",
        "CZ",
        "H",
        "M",
        "R",
    )
    #: Error parameters
    e_params: dict[str | tuple[str, str], dict]
    #: Circuit builder object
    cb: circuit.CircuitBuilder

    def __init__(
        self,
        code: codes.Code,
        qubit_errordata: QubitErrorsDict,
        gate_errordata: GateErrorsDict,
        n_rounds: int = 1,
    ):
        """Create a new circuit generator.

        Args:
            code: The code
            qubit_errordata: Information about qubit errors
            gate_errordata: Information about gate errors
            n_rounds: number of measurement rounds
        """
        if not set(qubit_errordata.keys()).issubset(QubitErrorsDict.__optional_keys__):
            raise KeyError("There is an invalid key in the qubit errors dictionary.")

        if not set(gate_errordata.keys()).issubset(GateErrorsDict.__optional_keys__):
            raise KeyError("There is an invalid key in the gate errors dictionary.")

        self.code = code
        self.qubit_errors = qubit_errordata
        for key in QubitErrorsDict.__optional_keys__:
            self.qubit_errors.setdefault(key, {})  # type: ignore

        self.gate_errors = gate_errordata
        for key in GateErrorsDict.__optional_keys__:
            self.gate_errors.setdefault(key, {})  # type: ignore

        if n_rounds < 1:
            raise ValueError(
                "at least one round of measurement is always necessary, "
                f"can't be {n_rounds}"
            )
        self.n_rounds = n_rounds
        # Process errordata
        ErrorData().check_against_code(
            self.code,
            cast(ErrorDataDict, self.qubit_errors)
            | cast(ErrorDataDict, self.gate_errors),  # `|`: merges two dicts.
        )
        self.cb = circuit.CircuitBuilder(circuit.Circuit())

    def apply_dataqubit_errors(self):
        """Apply errors to data qubits."""
        for v in range(self.code.num_data_qubits):
            p = self.qubit_errors["pauli"].get(v, None)
            if p:
                self.cb.e_pauli(
                    p.get("x", 0.0),
                    p.get("y", 0.0),
                    p.get("z", 0.0),
                    v,
                )
            p = self.qubit_errors["erasure"].get(v, None)
            if p:
                self.cb.e_erase(p.get("p", 0.0), v)

    def apply_measurement_error(self, op: int):
        """Apply measurement error to ancilla qubit of given operator.

        Args:
            op: ancilla vertex index.
        """
        params = self.qubit_errors["measurement"].get(op, None)
        if params:
            self.cb.error(params["p"], "X", op)

    def apply_gate_and_error(
        self, gate: str, ancilla: int, dataqubit: Optional[int], with_errors: bool
    ):
        """Apply a gate and corresponding error.

        Args:
            gate: Name of the gate.
            ancilla:
                First qubit on which the gate acts. Currently, this is always an
                ancilla qubit.
            dataqubit:
                Second qubit on which the gate acts (optional). If given, this is
                always a data qubit (currently).
            with_errors: Flag to indicate whether errors should be applied along
                with the gate
        """
        args = [ancilla]
        if dataqubit is not None:
            args.append(dataqubit)
        self.cb.circ.append(gate, *args)
        if with_errors:
            self.apply_gate_error(gate, ancilla, dataqubit)

    def apply_gate_error(self, gate: str, ancilla: int, dataqubit: Optional[int]):
        """Apply error gates for a given gate.

        Arguments: See :meth:`apply_gate_and_error`.
        """
        gate_errors = self.get_gate_error(gate, ancilla, dataqubit)
        if gate_errors is None:
            return
        if gate in ("H", "R", "M"):
            self.cb.e_pauli(
                gate_errors.get("x", 0.0),
                gate_errors.get("y", 0.0),
                gate_errors.get("z", 0.0),
                ancilla,
            )
        elif gate in ("CZ", "CX"):
            probabs = []
            for p1 in "ixyz":
                for p2 in "ixyz":
                    probabs.append(gate_errors.get(p1 + p2, 0.0))
            self.cb.e_pauli2(*probabs[1:], ancilla, dataqubit)

        else:
            raise ValueError(f"Gate error {gate!r} not implemented")

    def get_gate_error(
        self, gate: str, ancilla: int, dataqubit: Optional[int]
    ) -> Optional[dict]:
        """Retrieve error information for gate on specific qubit(s)."""
        try:
            err_gate = self.gate_errors[gate]  # type: ignore
        except KeyError:
            return None
        err = None
        if dataqubit is not None:
            err = err_gate.get((ancilla, dataqubit), None)
        if err is None:
            err = err_gate.get(ancilla, None)
        return err

    def measure_op(self, ancilla: int, with_errors: bool = True):
        """Measure an operator defined on the code.

        .. see-also:: :meth:`measure_stabgens` for the order in which gates
            are laid out for a given operator.

        Args:
            ancilla: index of the ancilla qubit that needs to be measured, which
                indirectly selects the operator to measure.
            with_errors: toggle to indicate if gate errors are to be applied
        """
        self.apply_gate_and_error("R", ancilla, None, with_errors)
        self.apply_gate_and_error("H", ancilla, None, with_errors)

        # Sort the edges, and the respective gates to be applied, according to
        # the index of the data qubit involved, in ascending order.
        # This looks convoluted, suggestions welcome
        edges: dict[int, pauli.Factor] = dict()
        # get all edges touching an ancilla
        for edge in self.code.tanner_graph.get_edges_touching_vertex(ancilla):
            # get the ancilla neighbours from the given edge
            a, b = self.code.tanner_graph.get_vertices_connected_by_edge(edge)
            # and extract the data qubit index and factor
            edge_data = self.code.tanner_graph.edges_data[edge]
            assert edge_data is not None
            edges[a if a in self.code.data_qubit_indices else b] = edge_data.type
        # now get a list of pauli factors, sorted according to data qubit index
        for data_idx in sorted(edges):
            match edges[data_idx]:
                case pauli.Factor.X:
                    self.apply_gate_and_error("CX", ancilla, data_idx, with_errors)
                case pauli.Factor.Y:
                    raise ValueError("Pauli Y not implemented yet")
                case pauli.Factor.Z:
                    self.apply_gate_and_error("CZ", ancilla, data_idx, with_errors)
                case _:
                    raise AssertionError("Case not handled")
        self.apply_gate_and_error("H", ancilla, None, with_errors)
        if with_errors:
            self.apply_measurement_error(ancilla)
        self.apply_gate_and_error("M", ancilla, None, with_errors)

    def measure_logical_ops(self, logop_indices: Sequence[int]):
        """Measure logical operators.

        Args:
            logop_indices:
                Specifies which logical operators should be measured. See
                :attr:`plaquette.new_codes_outline.Code.logical_ops` for
                details.

        Notes:
            Logical operators do not use an ancilla, as such would assume
            creating a separate "global" ancilla not constrained to
            nearest-neighbours interactions. In this case, each physical
            qubit is individually measured instead.
        """
        for i in logop_indices:
            # Logical operators have no "edges" to anything, we need to measure
            # each individual data qubit.
            op = pauli.pauli_to_dict(self.code.logical_ops[i])
            for qubit, factor in op.items():
                match factor:
                    case codes.Factor.Z:
                        self.apply_gate_and_error("M", qubit, None, with_errors=False)
                    case pauli.Factor.X:
                        # Apply Hadamard gate before and after the measurement
                        # because the "M" operator measures in the Z-basis.
                        self.apply_gate_and_error("H", qubit, None, with_errors=False)
                        self.apply_gate_and_error("M", qubit, None, with_errors=False)
                        self.apply_gate_and_error("H", qubit, None, with_errors=False)
                    case pauli.Factor.Y:
                        raise ValueError("Pauli Y not implemented yet")

    def measure_stabgens(self, with_errors: bool):
        """Measure all stabilizer generators.

        Args:
            with_errors:
                If ``False``, no error gates are added. (State preparation and
                verification are currently performed without errors.)

        Notes:
            The order in which the single data qubits are entangled with the
            corresponding ancilla depends on how the data qubits are indexed
            in the code.

            Specifically, the circuit generator will entangle each data qubit
            starting from the one with the lowest index, and continuing in
            ascending order.
        """
        # TODO: add logic to measure qubits in "parallel"
        self._measure_stabgens_seq(with_errors=with_errors)

    def _measure_stabgens_seq(self, with_errors: bool):
        """Measure all stabilizer generators sequentially.

        Args:
            with_errors:
                If ``False``, no error gates are added. (State preparation and
                verification are currently performed without errors.)
        """
        for ancilla in self.code.ancilla_qubit_indices:
            self.measure_op(ancilla, with_errors=with_errors)

    # fmt: off
    # TODO: this needs to be reworked once a way to schedule measurements is finalised
    # def _measure_stabgens_pl(self, with_errors: bool):
    #     """Measure all stabilizer generators in parallel.
    #
    #     Args:
    #         with_errors:
    #             If ``False``, no error gates are added. (State preparation and
    #             verification are currently performed without errors.)
    #     """
    #     for op in self.lat.stabgens:
    #         ancilla = op.equbit_idx
    #         self.apply_gate_and_error("R", ancilla, None, with_errors)
    #         self.apply_gate_and_error("H", ancilla, None, with_errors)
    #     assert isinstance(time_steps, int)
    #     for i in range(time_steps):
    #         edges = [edge for edge in self.lat.edges if edge.measurement_time_step == i]  # noqa: E501
    #         for edge in edges:
    #             ancilla = edge.op.equbit_idx
    #             dataqubit = edge.data.equbit_idx
    #             match edge.factor:
    #                 case codes.Pauli.X:
    #                     self.apply_gate_and_error("CX", ancilla, dataqubit, with_errors)  # noqa: E501
    #                 case codes.Pauli.Y:
    #                     raise NotImplementedError("Pauli Y not implemented yet")
    #                 case codes.Pauli.Z:
    #                     self.apply_gate_and_error("CZ", ancilla, dataqubit, with_errors)  # noqa: E501
    #     for op in self.lat.stabgens:
    #         ancilla = op.equbit_idx
    #         self.apply_gate_and_error("H", ancilla, None, with_errors)
    #         if with_errors:
    #             self.apply_measurement_error(op)
    #         self.apply_gate_and_error("M", ancilla, None, with_errors)
    # fmt: on

    def get_circuit(self, logical_ops: str) -> circuit.Circuit:
        """Build circuit for simulating QEC on the given error model.

        This function returns a circuit which can be simulated (see
        :mod:`plaquette.device`). If the circuit is simulated, it returns the
        sequence of measurement outcomes described in
        :meth:`~plaquette.device.AbstractSimulator.get_sample`.

        Args:
            logical_ops: Specifies which logical operators should be measured before
                and after the QEC simulation. ``"XZ"`` specifies logical ``X`` on the
                first and logical ``Z`` on the second logical qubits.

        Returns:
            A Clifford circuit (to be simulated with :mod:`plaquette.device`)

        Raises:
            ValueError: if an empty string is given, or if too many logical operators
                are given.
        """
        if not logical_ops:
            raise ValueError("at least one logical operator is necessary")
        if len(logical_ops) > self.code.num_logical_qubits:
            raise ValueError("logical operators exceed number of logical qubits")

        logop_indices = list()
        for i, logical_op in enumerate(logical_ops):
            if logical_op == "X":
                logop_indices.append(i)
            elif logical_op == "Z":
                logop_indices.append(i + self.code.num_logical_qubits)
            else:
                raise ValueError(f"logical operator {logical_op}{i} is invalid")
        # Measure logical operators for state preparation
        self.measure_logical_ops(logop_indices)
        # Measure stabilizer generators for state preparation (without errors)
        self.measure_stabgens(with_errors=False)
        for _ in range(self.n_rounds - 1):
            # Apply data qubit errors
            self.apply_dataqubit_errors()
            # Measure one round of stabilizer generators
            self.measure_stabgens(with_errors=True)
        # Apply data qubit errors on last round
        self.apply_dataqubit_errors()
        # Measure last round of stabilizer generators (without errors)
        self.measure_stabgens(with_errors=False)
        # Measure logical operators for state verification
        self.measure_logical_ops(logop_indices)
        return self.cb.circ


def generate_qec_circuit(
    code: codes.Code,
    qubit_errordata: QubitErrorsDict,
    gate_errordata: GateErrorsDict,
    logical_ops: str,
    n_rounds: int = 1,
) -> circuit.Circuit:
    """Shorthand for generating a circuit for QEC simulations.

    This function takes as input: An error correction code, information about errors
    and information about the logical operators of interest. Using this information,
    this function generates a circuit (as described in :ref:`circuits-ref`). This
    circuit can then be simulated by :mod:`plaquette.device`.

    .. important:: This function doesn't actually simulate the circuit! That's
       the job of the device that uses an underlying backend to perform the
       simulation, e.g. :class:`.CircuitSimulator` when choosing the
       ``"clifford"`` backend.

    Args:
        code: Definition of the error correction code
        qubit_errordata: Defintion of the qubit errors
        gate_errordata: Definition of the gate errors
        logical_ops: Specifies which logical operators should be measured before
            and after the QEC simulation.
        n_rounds: number of measurement rounds

    Returns:
        A Clifford circuit (to be simulated with :mod:`plaquette.device`)

    This function is a shorthand for::

       QECCircuitGenerator(code, errordata, n_rounds).get_circuit(logical_ops)

    See :attr:`QECCircuitGenerator.get_circuit` for a description of the measurement
    outcomes obtained if the circuit returned by this function is simulated.

    Example:
        >>> from plaquette import codes
        >>> from plaquette import errors
        >>> c = codes.Code.make_planar(3)
        >>> qed = errors.QubitErrorsDict()
        >>> ged = errors.GateErrorsDict()
        >>> # The planar code has one logical qubit. We choose logical X.
        >>> log_ops = "X"
        >>> circ = generate_qec_circuit(c, qed, ged, log_ops, 1)
        >>> print(type(circ))
        <class 'plaquette.circuit.Circuit'>

        The resulting circuit can be simulated using :mod:`plaquette.device`. The
        example above does not define any errors, see :mod:`plaquette.errors` for that
        matter.
    """
    circ_gen = QECCircuitGenerator(code, qubit_errordata, gate_errordata, n_rounds)
    return circ_gen.get_circuit(logical_ops)
