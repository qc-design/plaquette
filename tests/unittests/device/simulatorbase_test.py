# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit-tests for plaquette's CircuitSimulator."""

import numpy as np
import pytest as pt
import yaml  # type: ignore

import plaquette
from plaquette.circuit import Circuit
from plaquette.device import QuantumState
from plaquette.device._circuitsim import CircuitSimulator


def yaml_circuit_to_pytest_params(
    yaml_path: str, circuit_suite_key: str
) -> list[tuple]:
    """Parse non-parametrized circuit to pytests params.

    Args:
        yaml_path: str
        circuit_suite_key: str

    Returns:
        params_list : parameter list that is passed onto pytest.mark.parametrize
    """
    circuit_suite: dict[str, dict[str, str]]
    try:
        circuit_suite = yaml.load(open(yaml_path), yaml.SafeLoader)[circuit_suite_key]
    except FileNotFoundError:
        yaml_path = "tests/unittests/device/" + yaml_path
        circuit_suite = yaml.load(open(yaml_path), yaml.SafeLoader)[circuit_suite_key]
    params_list: list[tuple] = []
    for params in circuit_suite.values():
        params_list.append(tuple(params.values()))

    return params_list


def yaml_parametrized_circuit_to_pytest_params(
    yaml_path: str, circuit_suite_key: str
) -> list[tuple]:
    """Parse non-parametrized circuit to pytests params.

    Args:
        yaml_path: str
        circuit_suite_key: str

    Returns:
        params_list: parameter list that is passed onto
            ``pytest.mark.parametrize``.
    """
    try:
        circuit_suite = yaml.load(open(yaml_path), yaml.SafeLoader)[circuit_suite_key]
    except FileNotFoundError:
        yaml_path = "tests/unittests/device/" + yaml_path
        circuit_suite = yaml.load(open(yaml_path), yaml.SafeLoader)[circuit_suite_key]

    assert isinstance(
        circuit_suite, list
    ), "Please make sure, yaml is correctly specified"
    params_list: list[tuple] = []

    for dict_ in circuit_suite:
        assert isinstance(dict_, dict)
        key = list(dict_.keys())[0]  # this is of length 1 only
        circ_template: str = dict_[key]["circuit-template"]
        params: list = dict_[key]["params"]
        expected_output: list = dict_[key]["expected-output"]
        assert len(params) == len(
            expected_output
        ), "Please make sure number of parameters and expected outputs are same"
        for index in range(len(params)):
            params_list.append(
                (circ_template.format(*params[index]), expected_output[index])
            )

    return params_list


class TestCircuitSimulatorBase:
    """Group testing of basic CircuitSimulator functionality."""

    @pt.mark.skip
    def test__handle_error(self):
        pass

    @pt.mark.parametrize(
        "name, params, error_message",
        [
            ("RX", (1, 2), "Unknown gate 'RX'"),
            ("CRX", (1, 2), "Unknown gate 'CRX'"),
            ("ERROR_CONTINUES", ("X", 1, 2), "Unknown gate 'ERROR_CONTINUES'"),
            ("ERROR_IF", ("RY", 1, 2), "Unknown gate 'ERROR_IF'"),
        ],
    )
    def test__handle_error_failures(
        self, name: str, params: tuple, error_message: str, stable_rgen
    ):
        """Test that the handling of an unknown error gate fails."""
        plaquette.rng = stable_rgen
        circ = Circuit()
        circ.gates.append((name, params))
        c = CircuitSimulator(circ)
        with pt.raises(ValueError) as error:
            c._handle_error(name, params)

        assert str(error.value) == error_message

    # unclear how to write tests as it invokes tableausim
    @pt.mark.skip
    def test__handle_gate(self):
        pass

    @pt.mark.parametrize(
        "name, params, error_message, err_t",
        [
            ("RX", (1, 2), "Unknown gate 'RX' (this should not happen)", ValueError),
            ("CRX", (1, 2), "Unknown gate 'CRX' (this should not happen)", ValueError),
            (
                "ERROR_CONTINUE",
                ("CRX", 1, 2),
                "Unknown gate 'ERROR_CONTINUE' (this should not happen)",
                ValueError,
            ),
            (
                "ERROR",
                ("RY", 1, 2),
                "Unknown gate 'ERROR' (this should not happen)",
                ValueError,
            ),
        ],
    )
    def test__handle_gate_failures(
        self,
        name: str,
        params: tuple,
        error_message: str,
        stable_rgen,
        err_t: Exception,
    ):
        """Test that the handling of an unknown, non-error gate fails."""
        plaquette.rng = stable_rgen
        circ = Circuit()
        circ.gates.append((name, params))
        c = CircuitSimulator(circ)
        with pt.raises(err_t) as error:
            c._handle_gate(name, params)
        assert str(error.value) == error_message

    # unlcear how to write tests as it invokes tableausim
    @pt.mark.skip
    def test__run_gate(self):
        """TBA."""
        pass

    @pt.mark.parametrize(
        "name, params, error_message",
        [
            ("RX", (1, 2), "Unknown gate 'RX' (this should not happen)"),
            ("CRX", (1, 2), "Unknown gate 'CRX' (this should not happen)"),
            ("ERROR_CONTINUE", ("CRX", 1, 2), "ERROR_CONTINUE not valid here"),
            ("ERROR_ELSE", ("CRX", 1, 2), "ERROR_ELSE not valid here"),
            ("ERROR", (1.0, "RY", 1, 2), "Unknown gate 'RY' (this should not happen)"),
            # 1.0 is so that self.rng.random() is less the error prob always to see if
            # the right error is being raised
            ("ERROR_CONTINUES", ("X", 1, 2), "Unknown gate 'ERROR_CONTINUES'"),
            ("ERROR_IF", ("RY", 1, 2), "Unknown gate 'ERROR_IF'"),
        ],
    )
    def test__run_gate_failures(
        self,
        name: str,
        params: tuple,
        error_message: str,
        stable_rgen: np.random.Generator,
    ):
        """Make sure we catch weird gate names."""
        plaquette.rng = stable_rgen
        circ = Circuit()
        circ.gates.append((name, params))
        c = CircuitSimulator(circ)
        with pt.raises(ValueError) as error:
            c._run_gate(name, params)
        assert str(error.value) == error_message

    @pt.mark.parametrize(
        "input_circuit, exp_result",
        yaml_circuit_to_pytest_params(
            "sample_circuits/sample_circuits.yaml", "small-codes-without-error"
        ),
    )
    def test_run_circuit(self, input_circuit: str, exp_result: str, stable_rgen):
        """Run some simple circuits and compare with the expected outputs."""
        plaquette.rng = stable_rgen
        circ = Circuit.from_str(input_circuit)
        sim = CircuitSimulator(circ)
        sim_res, unused_erasure = sim.get_sample()
        assert "".join(map(str, sim_res)) == exp_result

    @pt.mark.parametrize(
        "param_circuit, result",
        yaml_parametrized_circuit_to_pytest_params(
            "sample_circuits/parametrized_circuits.yaml", "error-parametrize-circuits"
        ),
    )
    def test_run_circuit_error_parametrized_tests(
        self, param_circuit: str, result: str, stable_rgen
    ):
        """Run a bunch of circuits with error info and parametric gates.

        The input parameters to this test case is a "packaged" version of a
        very long and convoluted series of ``pytest.mark.parametrize`` options.
        """
        plaquette.rng = stable_rgen
        circ = Circuit.from_str(param_circuit)
        sim = CircuitSimulator(circ)
        sim_res, unused_erasure = sim.get_sample()
        sim_res2 = "".join(map(str, sim_res))
        assert sim_res2 == result


class TestQuantumState:
    """Tests related to the QuantumState object."""

    def test_string_representation(self):
        """Make sure that turning a state into a string of stabilizers works."""
        state = QuantumState(10)
        state.x(1)
        assert str(state).split("\n")[state.n_q + 2].startswith("-IZ")
        state.z(1)
        assert str(state).split("\n")[1].startswith("-IX")
        state.hadamard(5)
        assert str(state).split("\n")[state.n_q + 6][6] == "X"
        assert str(state).split("\n")[5][6] == "Z"
