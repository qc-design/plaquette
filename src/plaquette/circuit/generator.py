# Copyright 2023, It'sQ GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Generate Clifford circuit for QEC simulations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, cast

from plaquette import circuit, codes
from plaquette.codes import latticebase as lattice
from plaquette.errors import (
    ErrorData,
    ErrorDataDict,
    GateErrorsDict,
    QubitErrorsDict,
)
from plaquette.pauli import commutator_sign

# TODO: The code here is closer in functionality and concept to simulator.circuitsim


class QECCircuitGenerator:
    """Generate Clifford circuit for QEC simulations.

    This class can be used to generate a circuit which performs quantum error
    correction. For this purpose, it needs information about the code, the errors
    and the logical operators of interest (see :mod:`plaquette.codes` and
    :mod:`plaquette.errors`):

    >>> from plaquette.codes import LatticeCode
    >>> gen = QECCircuitGenerator(LatticeCode.make_planar(1, 4), {}, {})
    >>> circ = gen.get_circuit("X")

    The above is equivalent to:

    .. code-block:: python

        circ = generate_qec_circuit(code, {}, {}, "X")

    See :func:`generate_qec_circuit` for a complete minimal example.

    .. todo::

       The circuit error model depends on both the code lattice and the
       stabilizer code. Using only the stabilizer code would be sufficient, but
       using the code lattice in addition is a bit more convenient for the
       implementation. In addition, it enables relating qubit numbers to
       drawings (plots). I am not sure whether we should change something here.
    """

    #: QEC code which should be implemented
    code: codes.LatticeCode
    #: Qubit errors data
    qubit_errors: QubitErrorsDict
    #: Gate error data
    gate_errors: GateErrorsDict
    #: Code lattice (from :attr:`code`)
    lat: lattice.CodeLattice
    #: Stabilizer code (from :attr:`code`)
    stabcode: codes.StabilizerCode
    #: Number of "extended" qubits (data + ancilla) from code lattice
    n_equbits: int
    #: Number of logical qubits (from stabilizer code)
    n_lqubits: int

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
        code: codes.LatticeCode,
        qubit_errordata: QubitErrorsDict,
        gate_errordata: GateErrorsDict,
        logical_ancilla: bool = False,
    ):
        """Create a new circuit generator.

        Args:
            code: The code
            qubit_errordata: Information about qubit errors
            gate_errordata: Information about gate errors
            logical_ancilla: Flag for alternative method to measure logical operators
                via additional ancilla that is entangled with several physical qubits
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

        self.lat = self.code.lattice
        self.stabcode = self.code
        self.n_equbits = len(self.lat.equbits)
        self.n_lqubits = self.stabcode.n_logical_qubits
        self.logical_ancilla = logical_ancilla
        # Process errordata

        ErrorData().check_against_code(
            self.code,
            cast(ErrorDataDict, self.qubit_errors)
            | cast(ErrorDataDict, self.gate_errors),  # `|`: merges two dicts.
        )
        self.cb = circuit.CircuitBuilder(circuit.Circuit())

    def apply_dataqubit_errors(self):
        """Apply errors to data qubits."""
        for v in self.lat.dataqubits:
            p = self.qubit_errors["pauli"].get(v.equbit_idx, None)
            if p:
                self.cb.e_pauli(
                    p.get("x", 0.0),
                    p.get("y", 0.0),
                    p.get("z", 0.0),
                    v.equbit_idx,
                )
            p = self.qubit_errors["erasure"].get(v.equbit_idx, None)
            if p:
                self.cb.e_erase(p.get("p", 0.0), v.equbit_idx)

    def apply_measurement_error(self, op: lattice.OpVertex):
        """Apply measurement error to ancilla qubit of given operator."""
        params = self.qubit_errors["measurement"].get(op.equbit_idx, None)
        if params:
            self.cb.error(params["p"], "X", op.equbit_idx)

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

    def measure_op(self, op: lattice.OpVertex, with_errors: bool = True):
        """Measure an operator defined on the code lattice.

        Args:
            op: operator object that is to be measured
            with_errors: toggle to indicate if gate errors are to be applied
        """
        ancilla = op.equbit_idx
        self.apply_gate_and_error("R", ancilla, None, with_errors)
        self.apply_gate_and_error("H", ancilla, None, with_errors)
        for edge in op.edges:
            match edge.factor:
                case lattice.Pauli.X:
                    self.apply_gate_and_error(
                        "CX", ancilla, edge.data.equbit_idx, with_errors
                    )
                case lattice.Pauli.Y:
                    raise ValueError("Pauli Y not implemented yet")
                case lattice.Pauli.Z:
                    self.apply_gate_and_error(
                        "CZ", ancilla, edge.data.equbit_idx, with_errors
                    )
                case _:
                    raise AssertionError("Case not handled")
        self.apply_gate_and_error("H", ancilla, None, with_errors)
        if with_errors:
            self.apply_measurement_error(op)
        self.apply_gate_and_error("M", ancilla, None, with_errors)

    def measure_logical_ops(self, logop_indices: Sequence[int]):
        """Measure logical operators.

        Args:
            logop_indices:
                Specifies which logical operators should be measured. See
                :meth:`plaquette.codes.StabilizerCode.logical_ops_to_indices` for
                details.
        """
        for i in logop_indices:
            op = self.lat.logical_ops[i]

            if not self.logical_ancilla:
                for edge in op.edges:
                    dataqubit = edge.data.equbit_idx
                    match edge.factor:
                        case lattice.Pauli.X:
                            # Apply Hadamard gate, since we measure in Z-basis
                            self.apply_gate_and_error(
                                "H", dataqubit, None, with_errors=False
                            )
                        case lattice.Pauli.Y:
                            raise ValueError("Pauli Y not implemented yet")
                    self.apply_gate_and_error("M", dataqubit, None, with_errors=False)
            else:
                self.measure_op(op, with_errors=False)

    def measure_stabgens(self, with_errors: bool):
        """Measure all stabilizer generators sequentially or in parallel.

        Args:
            with_errors:
                If ``False``, no error gates are added. (State preparation and
                verification are currently performed without errors.)
        """
        time_steps = self.lat.n_time_steps
        if isinstance(time_steps, int):
            self.measure_stabgens_pl(with_errors=with_errors)
        else:
            self.measure_stabgens_seq(with_errors=with_errors)

    def measure_stabgens_seq(self, with_errors: bool):
        """Measure all stabilizer generators sequentially.

        Args:
            with_errors:
                If ``False``, no error gates are added. (State preparation and
                verification are currently performed without errors.)
        """
        for op in self.lat.stabgens:
            self.measure_op(op, with_errors=with_errors)

    def measure_stabgens_pl(self, with_errors: bool):
        """Measure all stabilizer generators in parallel.

        Args:
            with_errors:
                If ``False``, no error gates are added. (State preparation and
                verification are currently performed without errors.)
        """
        time_steps = self.lat.n_time_steps
        for op in self.lat.stabgens:
            ancilla = op.equbit_idx
            self.apply_gate_and_error("R", ancilla, None, with_errors)
            self.apply_gate_and_error("H", ancilla, None, with_errors)
        assert isinstance(time_steps, int)
        for i in range(time_steps):
            edges = [edge for edge in self.lat.edges if edge.measurement_time_step == i]
            for edge in edges:
                ancilla = edge.op.equbit_idx
                dataqubit = edge.data.equbit_idx
                match edge.factor:
                    case lattice.Pauli.X:
                        self.apply_gate_and_error("CX", ancilla, dataqubit, with_errors)
                    case lattice.Pauli.Y:
                        raise NotImplementedError("Pauli Y not implemented yet")
                    case lattice.Pauli.Z:
                        self.apply_gate_and_error("CZ", ancilla, dataqubit, with_errors)
        for op in self.lat.stabgens:
            ancilla = op.equbit_idx
            self.apply_gate_and_error("H", ancilla, None, with_errors)
            if with_errors:
                self.apply_measurement_error(op)
            self.apply_gate_and_error("M", ancilla, None, with_errors)

    def get_circuit(self, logical_ops: str | Sequence[int]) -> circuit.Circuit:
        """Build circuit for simulating QEC on the given error model.

        This function returns a circuit which can be simulated (see
        :mod:`plaquette.simulator`). If the circuit is simulated, it returns the
        sequence of measurement outcomes described in
        :meth:`plaquette.simulator.AbstractSimulator.get_sample`.

        Args:
            logical_ops: Specifies which logical operators should be measured before
                and after the QEC simulation.

                * E.g. ``XZ`` specifies logical ``X`` on the first and logical ``Z``
                  on the second logical qubit.
                * ``[0, 3]`` specifies the same as ``XZ`` because indices refer to
                  ``logical_ops`` in :attr:`stabcode`.

                For details, see
                :meth:`plaquette.codes.StabilizerCode.logical_ops_to_indices`.

        Returns:
            A Clifford circuit (to be simulated with :mod:`plaquette.simulator`)
        """
        logop_indices = self.stabcode.logical_ops_to_indices(logical_ops)
        # Check that different logical operators commute (if there is more than one).
        # (Measuring e.g. X_1 and Z_2 is fine, but we want to prevent e.g.
        # measuring X_1 and X_2.)
        logical_ops_op = [self.stabcode.logical_ops[i] for i in logop_indices]
        lcomm = commutator_sign(logical_ops_op, logical_ops_op)
        if lcomm.any():
            raise ValueError("Some measured logical operators do not commute")
        # Measure logical operators for state preparation
        self.measure_logical_ops(logop_indices)
        # Measure stabilizer generators for state preparation (without errors)
        self.measure_stabgens(with_errors=False)
        for _ in range(self.code.n_rounds - 1):
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
    code: codes.LatticeCode,
    qubit_errordata: QubitErrorsDict,
    gate_errordata: GateErrorsDict,
    logical_ops: str | Sequence[int],
    logical_ancilla: bool = False,
) -> circuit.Circuit:
    """Shorthand for generating a circuit for QEC simulations.

    This function takes as input: An error correction code, information about errors
    and information about the logical operators of interest. Using this information,
    this function generates a circuit (as described in :ref:`circuits-ref`). This
    circuit can then be simulated by :mod:`plaquette.simulator`.

    .. important:: This function doesn't actually simulate the circuit! That's
       the simulator's job, e.g. :class:`.CircuitSimulator`.

    Args:
        code: Definition of the error correction code
        qubit_errordata: Defintion of the qubit errors
        gate_errordata: Definition of the gate errors
        logical_ops: Specifies which logical operators should be measured before
            and after the QEC simulation.
        logical_ancilla: Flag for alternative method to measure logical operators
            via additional ancilla that is entangled with several physical qubits

            * E.g. ``XZ`` specifies logical ``X`` on the first and logical ``Z``
              on the second logical qubit.
            * ``[0, 3]`` specifies the same as ``XZ`` because indices refer to
              :attr:`~plaquette.codes.StabilizerCode.logical_ops`.

            For details, see
            :meth:`plaquette.codes.StabilizerCode.logical_ops_to_indices`.

    Returns:
        A Clifford circuit (to be simulated with :mod:`plaquette.simulator`)

    This function is a shorthand for::

       QECCircuitGenerator(code, errordata).get_circuit(logical_ops)

    See :attr:`QECCircuitGenerator.get_circuit` for a description of the measurement
    outcomes obtained if the circuit returned by this function is simulated.

    Example:
        >>> from plaquette.codes import LatticeCode
        >>> from plaquette import errors
        >>> c = LatticeCode.make_planar(n_rounds=1, size=3)
        >>> qed = errors.QubitErrorsDict()
        >>> ged = errors.GateErrorsDict()
        >>> # The planar code has one logical qubit. We choose logical X.
        >>> log_ops = "X"
        >>> circ = generate_qec_circuit(c, qed, ged, log_ops)
        >>> print(type(circ))
        <class 'plaquette.circuit.Circuit'>

        The resulting circuit can be simulated using :mod:`plaquette.simulator`. The
        example above does not define any errors, see :mod:`plaquette.errors` for that
        matter.
    """
    circ_gen = QECCircuitGenerator(
        code, qubit_errordata, gate_errordata, logical_ancilla
    )
    return circ_gen.get_circuit(logical_ops)
