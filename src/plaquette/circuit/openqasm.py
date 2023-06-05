# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Quantum Circuit Converter: Plaquette circuits to OpenQASM.

This Python script converts a quantum circuit from the internal string representation
used by plaquette to the OpenQASM format.

The Plaquette internal string representation is a text-based format that defines
quantum circuits using gates and qubits. The OpenQASM format is a more widely adopted
language for representing quantum circuits and is used by many quantum computing
frameworks, such as Qiskit.
"""

from plaquette.circuit import Circuit


class GateNotSupportedError(Exception):
    """Raised for valid ``plaquette`` gates but invalid openQASM ones.

    Attributes:
        gate: The unsupported gate name.
    """

    def __init__(self, gate: str):
        """Create new ``gate`` error instance.

        Args:
            gate: the unsupported gate identifier.
        """
        self.gate = gate
        super().__init__(f"Gate not currently supported in openQASM export: {gate}")


def convert_to_openqasm(
    circuit: str | Circuit, ignore_unsupported: bool = False
) -> str:
    """Convert a quantum circuit represented by a string into the openQASM format.

    This function takes a string representation of a quantum circuit and
    returns the corresponding openQASM code. It supports single-qubit gates
    (H, X, Y, Z) and two-qubit gates (CX, CZ), as well as measurements (M)
    and reset (R).

    Notes:
        The following ``plaquette`` instructions and gates are not supported by
        OpenQASM: ``DEPOLARIZE``, ``ERROR``, ``ERROR_CONTINUE``,
        ``ERROR_ELSE``, ``E_ERASE``, ``E_PAULI``, ``E_PAULI2``.

    Args:
        circuit: The input quantum circuit as a string.
        ignore_unsupported: Whether the unsupported gates should be ignored or
            not.

    Returns:
        The openQASM code corresponding to the input circuit.

    Raises:
        GateNotSupported: If the input circuit contains an unsupported gate.
    """
    if isinstance(circuit, str):
        circuit = Circuit.from_str(circuit)
    elif not isinstance(circuit, Circuit):
        raise TypeError(
            "Input circuit must be a string or a `plaquette` Circuit object."
        )

    gate_mapping = {
        "R": "reset",
        "H": "h",
        "X": "x",
        "Y": "y",
        "Z": "z",
        "CX": "cx",
        "CZ": "cz",
        "M": "measure",
    }

    qasm_lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{circuit.number_of_qubits}];",
        f"creg c[{circuit.number_measured_qubits}];",
    ]

    meas_idx = 0

    for gate, qubits in circuit.gates:
        if gate not in gate_mapping:
            if ignore_unsupported:
                continue
            else:
                raise GateNotSupportedError(gate)

        qasm_gate = gate_mapping[gate]

        if gate == "M":
            for qubit in qubits:
                qasm_lines.append(f"measure q[{qubit}] -> c[{meas_idx}];")
                meas_idx += 1
        else:
            if gate in ["H", "X", "Y", "Z"]:
                for qubit in qubits:
                    qasm_lines.append(f"{qasm_gate} q[{qubit}];")
            elif gate in ["CX", "CZ"]:
                for i in range(0, len(qubits), 2):
                    control_qubit, target_qubit = qubits[i], qubits[i + 1]
                    qasm_lines.append(
                        f"{qasm_gate} q[{control_qubit}],q[{target_qubit}];"
                    )

    return "\n".join(qasm_lines)
