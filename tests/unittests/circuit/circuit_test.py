# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest as pt

import plaquette
from plaquette import Device
from plaquette.circuit import Circuit, CircuitBuilder


def load_strings(file_path):
    with open(file_path, "r") as file:
        contents = file.read()
    return contents.split("\n\n")


circuit_inputs = load_strings("tests/unittests/circuit/circuit_inputs.txt")

circ_template = """
ERROR {}       X 0
ERROR_CONTINUE X 1
ERROR_CONTINUE X 2
ERROR_ELSE {}  X 3
ERROR_CONTINUE X 4
ERROR_CONTINUE X 5
M 0 1 2 3 4 5
"""


@pt.mark.parametrize(
    "params, result",
    [
        ((0.0, 0.0), "000000"),
        ((0.0, 1.0), "000111"),
        ((1.0, 0.0), "111000"),
        ((1.0, 1.0), "111000"),
    ],
)
def test_circuit_error(params, result):
    """Test that circuit error continue/else works as expected."""
    plaquette.rng = np.random.default_rng(seed=94129232)
    circ = Circuit.from_str(circ_template.format(*params))
    dev = Device("clifford")
    dev.run(circ)
    sim_res, unused_erasure = dev.get_sample()
    sim_res2 = "".join(map(str, sim_res))
    assert sim_res2 == result


class TestCircuit:
    @pt.mark.parametrize(
        "input_string, exp_gate_list",
        [
            ("X 1 2 3", [("X", (1, 2, 3))]),
            ("Y 1 2 3", [("Y", (1, 2, 3))]),
            ("Z 1 3", [("Z", (1, 3))]),
            ("H 1", [("H", (1,))]),
            ("R 1 2 3", [("R", (1, 2, 3))]),
            ("M 1 2 3", [("M", (1, 2, 3))]),
            ("CZ 1 2 0 1", [("CZ", (1, 2, 0, 1))]),
            ("CX 1 2", [("CX", (1, 2))]),
            ("ERROR 0.5 X 1", [("ERROR", (0.5, "X", 1))]),
            ("ERROR_CONTINUE X 3", [("ERROR_CONTINUE", ("X", 3))]),
            ("ERROR_ELSE 0.1 Z 2", [("ERROR_ELSE", (0.1, "Z", 2))]),
        ],
    )
    def test_from_str_single_instruction(
        self, input_string: str, exp_gate_list: list[tuple]
    ):
        """This also tests append_instructions indirectly."""
        c = Circuit.from_str(input_string)
        assert c.gates == exp_gate_list

    @pt.mark.parametrize(
        "input_string, exp_gate_list",
        [
            (
                "X 1 2 3\nY 1 2 3\nM 1 2 3",
                [("X", (1, 2, 3)), ("Y", (1, 2, 3)), ("M", (1, 2, 3))],
            ),
            (
                "Z 1 3\nH 1\nR 1 2 3\nM 1 2 3",
                [("Z", (1, 3)), ("H", (1,)), ("R", (1, 2, 3)), ("M", (1, 2, 3))],
            ),
            (
                "CZ 1 2 0 1\nCX 1 2\nERROR 0.5 X 1",
                [("CZ", (1, 2, 0, 1)), ("CX", (1, 2)), ("ERROR", (0.5, "X", 1))],
            ),
        ],
    )
    def test_from_str_multiple_instructions(
        self, input_string: str, exp_gate_list: list[tuple]
    ):
        c = Circuit.from_str(input_string)
        assert c.gates == exp_gate_list

    @pt.mark.parametrize(
        "input_string, exp_gate_list",
        [
            (
                "Z 1 3\nH 1\nR 1 2 3\nM 1 2 3",
                [
                    ("X", (1, 2, 3)),
                    ("Y", (1, 2, 3)),
                    ("M", (1, 2, 3)),
                    ("Z", (1, 3)),
                    ("H", (1,)),
                    ("R", (1, 2, 3)),
                    ("M", (1, 2, 3)),
                ],
            ),
            (
                "CZ 1 2 0 1\nCX 1 2\nERROR 0.5 X 1",
                [
                    ("X", (1, 2, 3)),
                    ("Y", (1, 2, 3)),
                    ("M", (1, 2, 3)),
                    ("CZ", (1, 2, 0, 1)),
                    ("CX", (1, 2)),
                    ("ERROR", (0.5, "X", 1)),
                ],
            ),
        ],
    )
    def test_append_instructions(self, input_string: str, exp_gate_list: list[tuple]):
        c = Circuit.from_str("X 1 2 3\nY 1 2 3\nM 1 2 3")
        c.append_from_str(input_string)
        assert c.gates == exp_gate_list

    @pt.mark.parametrize(
        "name, params, exp_gate_list",
        [
            ("X", (1, 2, 3), [("X", (1, 2, 3))]),
            ("Y", (1, 2, 3), [("Y", (1, 2, 3))]),
            ("Z", (1, 3), [("Z", (1, 3))]),
            ("H", (1,), [("H", (1,))]),
            ("R", (1, 2, 3), [("R", (1, 2, 3))]),
            ("M", (1, 2, 3), [("M", (1, 2, 3))]),
            ("CZ", (1, 2, 0, 1), [("CZ", (1, 2, 0, 1))]),
            ("CX", (1, 2), [("CX", (1, 2))]),
            ("ERROR", (0.5, "X", 1), [("ERROR", (0.5, "X", 1))]),
            ("ERROR_CONTINUE", ("X", 3), [("ERROR_CONTINUE", ("X", 3))]),
            ("ERROR_ELSE", (0.1, "Z", 2), [("ERROR_ELSE", (0.1, "Z", 2))]),
        ],
    )
    def test_append(self, name: str, params: tuple, exp_gate_list: list[tuple]):
        c = Circuit()
        c.append(name, *params)
        assert c.gates == exp_gate_list

    @pt.mark.parametrize(
        "name, params, exc_type, error_msg",
        [
            ("RX", (1, 2), ValueError, "Do not know how to handle gate 'RX'"),
            ("CRX", (1, 2), ValueError, "Do not know how to handle gate 'CRX'"),
            (
                "ERROR_CONTINUE",
                ("CRX", 1, 2),
                AssertionError,
                "CRX gate is not supported",
            ),
            (
                "ERROR",
                ("Y", 1, 2),
                ValueError,
                "First argument of ERROR must be a float",
            ),
            (
                "ERROR",
                (0.1, "RY", 1, 2),
                AssertionError,
                "RY gate is not supported",
            ),
            (
                "ERROR_ELSE",
                (1.3, "Y", 1, 2),
                ValueError,
                "The probability must be within [0.0, 1.0]",
            ),
            ("X", ("0.5", 2), TypeError, "Qubit indices must be integers"),
        ],
    )
    def test_append_failure(
        self, name: str, params: tuple, exc_type: Exception, error_msg: str
    ):
        # TODO: add failure tests for all gates we plan to support,
        #  but not yet supported
        c = Circuit()
        with pt.raises(exc_type) as excinfo:
            c.append(name, *params)

        assert str(excinfo.value) == error_msg


class TestCircuitGenerator:
    @pt.mark.parametrize(
        "func_name, params, exp_gate_list",
        [
            ("X", (1, 2, 3), [("X", (1, 2, 3))]),
            ("Y", (1, 2, 3), [("Y", (1, 2, 3))]),
            ("Z", (1, 3), [("Z", (1, 3))]),
            ("H", (1,), [("H", (1,))]),
            ("R", (1, 2, 3), [("R", (1, 2, 3))]),
            ("M", (1, 2, 3), [("M", (1, 2, 3))]),
            ("CZ", (1, 2, 0, 1), [("CZ", (1, 2, 0, 1))]),
            ("CX", (1, 2), [("CX", (1, 2))]),
        ],
    )
    def test_append_gate(self, func_name, params, exp_gate_list):
        circ = Circuit()
        c = CircuitBuilder(circ)
        gate_func = getattr(c, func_name)
        gate_func(*params)
        assert circ.gates == exp_gate_list

    @pt.mark.parametrize(
        "func_name, params, exp_gate_list",
        [  # first test possible ambiguous behavior
            ("depolarize", (1, 2, 3), [("DEPOLARIZE", (0.2, 1, 2, 3))]),
            ("error", (0.5, "X", 1), [("ERROR", (0.5, "X", 1))]),
            ("error_continue", ("X", 3), [("ERROR_CONTINUE", ("X", 3))]),
            ("error_else", (0.1, "Z", 2), [("ERROR_ELSE", (0.1, "Z", 2))]),
        ],
    )
    def test_append_error(self, func_name, params, exp_gate_list):
        circ = Circuit()
        c = CircuitBuilder(circ)
        gate_func = getattr(c, func_name)
        gate_func(*params)

    @pt.mark.parametrize(
        "func_name, params, exp_error, error_msg",
        [
            (
                "error",
                ("X", 1),
                ValueError,
                "First argument of ERROR must be a float",
            ),
            (
                "error",
                (1.2, "Y", 1, 2, 3),
                ValueError,
                "The probability must be within [0.0, 1.0]",
            ),
            (
                "error_else",
                ("X", 1),
                ValueError,
                "First argument of ERROR_ELSE must be a float",
            ),
            (
                "error_else",
                (1.2, "Y", 1, 2, 3),
                ValueError,
                "The probability must be within [0.0, 1.0]",
            ),
        ],
    )
    def test_append_failure(self, func_name, params, exp_error, error_msg):
        circ = Circuit()
        c = CircuitBuilder(circ)
        gate_func = getattr(c, func_name)
        with pt.raises(exp_error) as e:
            gate_func(*params)

        assert str(e.value) == error_msg


@pt.mark.parametrize(
    "input_circuit, expected_count", zip(circuit_inputs, [2, 2, 3, 3, 8])
)
def test_num_measured_qubits(input_circuit, expected_count):
    circ = Circuit.from_str(input_circuit)
    result = circ.number_measured_qubits
    assert result == expected_count, f"Expected: {expected_count}, Got: {result}"


@pt.mark.parametrize(
    "input_circuit, expected_count", zip(circuit_inputs, [2, 2, 3, 3, 6])
)
def test_number_of_qubits(input_circuit, expected_count):
    circ = Circuit.from_str(input_circuit)
    result = circ.number_of_qubits
    assert result == expected_count, f"Expected: {expected_count}, Got: {result}"
