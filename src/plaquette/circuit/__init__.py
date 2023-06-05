# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
r"""Representation and generation of Clifford circuits.

Clifford circuits are represented by the class :class:`.Circuit`. A circuit
can be built using :class:`.Circuit` itself or using the wrapper class
:class:`.CircuitBuilder`. See :ref:`circuits-guide` for examples.

Starting from a quantum error correction code and error probabilities, a circuit
which implements the given code and error model can be generated using the function
:func:`.generator.generate_qec_circuit`, which is a convenience wrapper around
:class:`.generator.QECCircuitGenerator`.

For general information on circuits as well as a list of supported gates, see
:ref:`circuits-ref`.

.. todo::

   Import OpenQASM circuits (with restrictions).
"""
from __future__ import annotations

from typing import Any


class Circuit:
    """A Clifford circuit.

    This class represents a Clifford circuit by storing a list of Clifford gates.

    Qubits that the circuit acts on are identified by integers ``0, 1, ...``. Clifford
    gates can be applied to these qubits. For a full list of supported gates and more
    information, see :ref:`circuits-ref`.

    The number of qubits is not fixed and you can use :attr:`number_of_qubits`
    to determine how many qubits are necessary to run the circuit.

    A circuit can be defined from strings as follows:

    >>> n_qubits = 6
    >>> circ = Circuit.from_str('''
    ... X 3 4 5
    ... E_ERASE 0.0 4
    ... E_ERASE 1.0 5
    ... R 5
    ... ''')
    >>> circ.append("M", *range(n_qubits))
    >>> print(circ)
    X 3 4 5
    E_ERASE 0.0 4
    E_ERASE 1.0 5
    R 5
    M 0 1 2 3 4 5

    In order to build circuits step-by-step, :class:`CircuitBuilder` can be useful.

    The circuit can be simulated using :mod:`plaquette.device` module to obtain the
    measurement outcomes along with information about errors as follows:

    >>> from plaquette import Device
    >>> dev = Device("clifford")
    >>> dev.run(circ)
    >>> measurements, erasure = dev.get_sample()
    >>> measurements
    array([0, 0, 0, 1, 1, 0], dtype=uint8)
    >>> erasure
    array([False,  True])

    .. todo::

       Only rudimentary correctness checks are performed when gates are added. Improving
       the checks in this class would allow us to obtain more errors when a gate is
       added (instead of much later when the circuit is simulated).

    .. automethod:: __init__
    """

    def __init__(self):
        """Create a new circuit which does not contain any gates."""
        self.gates: list[tuple[str, tuple[Any, ...]]] = []
        """Sequence of gates."""

    def __str__(self) -> str:
        """Convert circuit description to string."""
        return "\n".join(
            f"{name} " + " ".join(map(str, args)) for name, args in self.gates
        )

    #: List of allowed gate names
    @property
    def error_gates(self):
        """Error instructions, applying specific gates with a certain probability."""
        return {
            "E_ERASE",
            "E_PAULI",
            "E_PAULI2",
            "ERROR",
            "ERROR_ELSE",
            "ERROR_CONTINUE",
        }

    @property
    def deterministic_gates(self):
        """Deterministic gates accepting only qubit indices as arguments."""
        return {"X", "Y", "Z", "H", "CX", "CZ", "M", "R", "DEPOLARIZE"}

    @property
    def allowed_gates(self):
        """All possible instructions and gates understood by ``plaquette``."""
        return self.error_gates.union(self.deterministic_gates)

    @property
    def number_measured_qubits(self) -> int:
        """Total number of qubits measured in the circuit.

        For qubits, this is the number of classical bits of output from running the
        quantum circuit on the quantum device or the simulator.

        Note:
            If a qubit is measured multiple times, it is counted multiple
            times.
        """
        total_measured_qubits: int = 0
        for gate, qubits in self.gates:
            if gate == "M":
                total_measured_qubits += len(qubits)
        return total_measured_qubits

    @property
    def number_of_qubits(self) -> int:
        """Get minimal number of qubits necessary to simulate the circuit.

        This function checks the highest qubit index used. It does not check whether
        all the qubits between 0 and the highest index are actually used. E.g. if
        there is a single gate which acts only on qubit index 9, this function will
        return 10.
        """
        n = 0
        # FIXME: gates should have more information about what arguments they carry,
        #  not being simple unstructured tuples
        for _, args in self.gates:
            for arg in args:  # type: str
                if isinstance(arg, int) and arg > n:
                    n = arg
        return n + 1

    @property
    def n_q(self) -> int:
        """Alias of :attr:`number_of_qubits`."""
        return self.number_of_qubits

    @classmethod
    def from_str(cls, s: str) -> Circuit:
        """Create circuit from string.

        >>> circ = Circuit.from_str('''
        ... X 0
        ... X 1
        ... ''')
        >>> circ.append("M", 0, 1)
        >>> print(circ)
        X 0
        X 1
        M 0 1
        """
        c = cls()
        c.append_from_str(s)
        return c

    def append_from_str(self, s: str):
        r"""Append instructions to current gate sequence.

        >>> circ = Circuit.from_str("X 0 1")
        >>> circ.append_from_str("Y 2 3\nZ 4 5")
        >>> print(circ)
        X 0 1
        Y 2 3
        Z 4 5

        Args:
            s: String which contains gates to be appended (see :ref:`circuits-ref`)
        """
        for line in s.strip().split("\n"):
            line = line.split("#", 1)[0].strip()
            if line:
                self.append(*line.split())

    def append(self, name: str, *args):
        """Append one gate to the circuit.

        >>> circ = Circuit.from_str("X 0 1")
        >>> circ.append("Z", 2, 3)
        >>> print(circ)
        X 0 1
        Z 2 3

        Args:
            name: Name of the gate
            *args: Gate arguments (see :ref:`circuits-ref`)
        """
        if name in ("ERROR", "ERROR_ELSE"):
            try:
                p = float(args[0])
            except ValueError as e:
                raise ValueError(f"First argument of {name} must be a float") from e
            if not 0.0 <= p <= 1.0:
                raise ValueError("The probability must be within [0.0, 1.0]")
            assert args[1] in self.allowed_gates, f"{args[1]} gate is not supported"
            args = (p, str(args[1]), *map(int, args[2:]))
        elif name == "ERROR_CONTINUE":
            assert args[0] in self.allowed_gates, f"{args[0]} gate is not supported"
            args = (str(args[0]), *map(int, args[1:]))
        elif name == "E_ERASE":
            assert len(args) == 2, "Wrong number of arguments to E_ERASE"
            args = (float(args[0]), int(args[1]))
        elif name == "E_PAULI":
            assert len(args) == 4, "Wrong number of arguments to E_PAULI"
            args = (*map(float, args[:3]), int(args[3]))
        elif name == "E_PAULI2":
            assert len(args) == 17, "Wrong number of arguments to E_PAULI2"
            args = (*map(float, args[:15]), *map(int, args[15:]))
        elif name in self.deterministic_gates:
            try:
                args = tuple(map(int, args))
            except ValueError as e:
                raise TypeError("Qubit indices must be integers") from e
        else:
            raise ValueError(f"Do not know how to handle gate {name!r}")
        self.gates.append((str(name), args))


class CircuitBuilder:
    """Helper class to build circuits programatically.

    >>> circ = Circuit()
    >>> c = CircuitBuilder(circ)
    >>> c.X(0, 1)
    >>> c.CZ(0, 1, 11, 12)
    >>> c.error(0.5, "X", 1)
    >>> print(circ)
    X 0 1
    CZ 0 1 11 12
    ERROR 0.5 X 1

    It is also possible to define circuits using :class:`Circuit` directly:

    >>> circ = Circuit()
    >>> circ.append("X", 0, 1)
    >>> print(circ)
    X 0 1

    .. seealso:: :ref:`circuits-guide` and :ref:`circuits-ref`.

    .. automethod:: __init__
    """

    #: Circuit to which gates are added
    circ: Circuit

    def __init__(self, circ: Circuit):
        """Create a new circuit builder.

        Args:
            circ: Gates are added to this circuit object.
        """
        self.circ = circ
        # Methods could be autogenerated in the following way, but then mypy
        # cannot check things anymore. Should we create a code generator?
        # Keeping the manual code would also allow for different names in code
        # and in the string description (if we want that).
        #
        # for name in circ.allowed_gates:
        #     setattr(
        #         self,
        #         name,
        #         lambda *args, _name=name: self.circ.append(_name, *args),
        #     )

    def X(self, *args):
        """Append Pauli X gate."""
        return self.circ.append("X", *args)

    def Y(self, *args):
        """Append Pauli Y gate."""
        return self.circ.append("Y", *args)

    def Z(self, *args):
        """Append Pauli Z gate."""
        return self.circ.append("Z", *args)

    def H(self, *args):
        """Append Hadamard gate."""
        return self.circ.append("H", *args)

    def M(self, *args):
        """Append measurement gate."""
        return self.circ.append("M", *args)

    def R(self, *args):
        """Append reset gate."""
        return self.circ.append("R", *args)

    def CX(self, *args):
        """Append controlled-X gate."""
        return self.circ.append("CX", *args)

    def CZ(self, *args):
        """Append controlled-Z gate."""
        return self.circ.append("CZ", *args)

    def depolarize(self, *args):
        """Append probabilistic depolarization channel.

        Notes:
           Supports multi-qubit argument. Note that e.g.
           ``ERROR 0.01 DEPOLARIZE 0 1 2`` depolarizes either all three or none.
           For independent depolarization errors, you have to use separate error
           and depolarize instructions for each qubit. See
           :mod:`plaquette.circuit`.
        """
        return self.circ.append("DEPOLARIZE", *args)

    def e_erase(self, p, qubit):
        """Append probabilistic erasure channel.

        Args:
            p: Erasure probability
            qubit: Target qubit
        """
        return self.circ.append("E_ERASE", p, qubit)

    def e_pauli(self, *args):
        """Append probabilistic single-qubit Pauli channel."""
        return self.circ.append("E_PAULI", *args)

    def e_pauli2(self, *args):
        """Append probabilistic two-qubit Pauli channel."""
        return self.circ.append("E_PAULI2", *args)

    def error(self, *args):
        """Append ERROR gate."""
        return self.circ.append("ERROR", *args)

    def error_continue(self, *args):
        """Append ERROR_CONTINUE gate."""
        return self.circ.append("ERROR_CONTINUE", *args)

    def error_else(self, *args):
        """Append ERROR_ELSE gate."""
        return self.circ.append("ERROR_ELSE", *args)
