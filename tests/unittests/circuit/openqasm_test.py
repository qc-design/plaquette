# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
from typing import Iterable

import pytest as pt

from plaquette.circuit import Circuit
from plaquette.circuit.openqasm import GateNotSupportedError, convert_to_openqasm


def load_documents_from_file(file_path: str) -> Iterable[str]:
    """Helper to load test data from a file.

    You can put multiple test data "documents" in the same file by separating
    them with two new-line characters.
    """
    with open(file_path, "r") as file:
        contents = file.read()
    yield from contents.split("\n\n")


@pt.fixture
def circuits():
    return load_documents_from_file("tests/unittests/circuit/circuit_inputs.txt")


@pt.fixture
def openqasm_outputs():
    return load_documents_from_file("tests/unittests/circuit/expected_qasm_outputs.txt")


def test_conversion_to_openqasm(circuits, openqasm_outputs):
    """Test the convert_to_openqasm function for string inputs.

    This test function compares the output of the ``convert_to_openqasm``
    function to the expected openQASM code. It removes leading and trailing
    whitespaces on each line and filters out empty lines before comparison to
    ensure that the comparison is not affected by differences in formatting.
    """
    for circuit, openqasm_output in zip(circuits, openqasm_outputs):
        result_qasm_str = convert_to_openqasm(circuit)
        circ = Circuit.from_str(circuit)
        result_qasm_circ = convert_to_openqasm(circ)

        # Remove surrounding whitespaces on each line and filter out empty lines
        expected_qasm_clean = "\n".join(
            line.strip() for line in openqasm_output.split("\n") if line.strip()
        )
        result_qasm_str_clean = "\n".join(
            line.strip() for line in result_qasm_str.split("\n") if line.strip()
        )
        result_qasm_circ_clean = "\n".join(
            line.strip() for line in result_qasm_circ.split("\n") if line.strip()
        )

        assert result_qasm_str_clean == expected_qasm_clean

        assert result_qasm_circ_clean == expected_qasm_clean


@pt.mark.parametrize(
    "input_circuit, expected_error_message",
    [
        (
            """
            ERROR 0.01 X 1
            """,
            "Gate not currently supported in openQASM export: ERROR",
        ),
    ],
)
def test_unknown_gate_error(input_circuit, expected_error_message):
    with pt.raises(GateNotSupportedError) as exc_info:
        convert_to_openqasm(input_circuit)

    assert str(exc_info.value) == expected_error_message


@pt.fixture
def circuits_ignore_unsupported():
    return load_documents_from_file(
        "tests/unittests/circuit/circuit_inputs_with_unsupported.txt"
    )


@pt.fixture
def openqasm_outputs_ignore_unsupported():
    return load_documents_from_file(
        "tests/unittests/circuit/expected_qasm_outputs_with_unsupported.txt"
    )


def test_unknown_gate_no_error_when_ignoring(
    circuits_ignore_unsupported, openqasm_outputs_ignore_unsupported
):
    """Test the convert_to_openqasm function for string inputs when unsupported
    gates are ignored.

    This test function compares the output of the ``convert_to_openqasm``
    function to the expected openQASM code when unsupported gates are ignored.
    It removes leading and trailing whitespaces on each line and filters out
    empty lines before comparison to ensure that the comparison is not affected
    by differences in formatting.
    """
    for circuit, openqasm_output in zip(
        circuits_ignore_unsupported, openqasm_outputs_ignore_unsupported
    ):
        result_qasm_str = convert_to_openqasm(circuit, ignore_unsupported=True)
        circ = Circuit.from_str(circuit)
        result_qasm_circ = convert_to_openqasm(circ, ignore_unsupported=True)

        # Remove surrounding whitespaces on each line and filter out empty lines
        expected_qasm_clean = "\n".join(
            line.strip() for line in openqasm_output.split("\n") if line.strip()
        )
        result_qasm_str_clean = "\n".join(
            line.strip() for line in result_qasm_str.split("\n") if line.strip()
        )
        result_qasm_circ_clean = "\n".join(
            line.strip() for line in result_qasm_circ.split("\n") if line.strip()
        )

        assert result_qasm_str_clean == expected_qasm_clean

        assert result_qasm_circ_clean == expected_qasm_clean
