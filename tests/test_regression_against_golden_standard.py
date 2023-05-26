# Copyright 2023, It'sQ GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests against values found in the literature.

The tests here are mean to blindly check that whatever happens in the codebase will
always result in data that matches what's being published in the literature. They
depend only on data that is common among simulation and experimental studies made by
others, and must not depend on our own implementation of things.
"""
import sys
from typing import Type, cast

import joblib as jl
import numpy as np
import pytest as pt

import plaquette
from plaquette.circuit.generator import generate_qec_circuit
from plaquette.codes import LatticeCode
from plaquette.decoders import (
    FusionBlossomDecoder,
    PyMatchingDecoder,
    UnionFindDecoder,
    decoderbase,
)
from plaquette.errors import (
    ErrorDataDict,
    ErrorValueDict,
    GateErrorsDict,
    QubitErrorsDict,
    SinglePauliChannelErrorValueDict,
)
from plaquette.frontend import ExperimentConfig
from plaquette.simulator import AbstractSimulator, SimulatorSample
from plaquette.simulator.circuitsim import CircuitSimulator
from plaquette.simulator.stimsim import StimSimulator


def calculate_success(
    sim_res: tuple,
    logical_op: str,
    code: LatticeCode,
    decoder: UnionFindDecoder | PyMatchingDecoder | FusionBlossomDecoder,
):
    """Helper function for parallel calculations with joblib."""
    raw_results, erasure = sim_res
    sample = SimulatorSample.from_code_and_raw_results(code, raw_results, erasure)
    correction = decoder.decode(sample.erased_qubits, sample.syndrome)
    return decoderbase.check_success(
        code, correction, sample.logical_op_toggle, logical_op
    )


class TestRegressionAPI:
    @pt.mark.slow
    @pt.mark.parametrize("decoder_class", [PyMatchingDecoder, FusionBlossomDecoder])
    @pt.mark.parametrize(
        "simulator_class,reps",
        [
            (CircuitSimulator, 100),
            (StimSimulator, 10000),
        ],
    )
    def test_pauli_and_measurement_errors_vs_logical_error_rates_in_nickerson_thesis(
        self,
        decoder_class: Type[decoderbase.DecoderInterface],
        simulator_class: Type[AbstractSimulator],
        reps: int,
    ):
        """Test to be matched/compared with Fig. 1.12 in doi:10.25560/31475."""
        code = LatticeCode.make_planar(n_rounds=8, size=7)
        logical_op = "Z"
        qubit_errors = QubitErrorsDict(
            pauli={
                vtx.equbit_idx: SinglePauliChannelErrorValueDict(x=0.026)
                for vtx in code.lattice.dataqubits
            },
            measurement={
                vtx.equbit_idx: ErrorValueDict(p=0.026) for vtx in code.lattice.stabgens
            },
        )

        plaquette.rng = np.random.default_rng(seed=62934814123)
        circ = generate_qec_circuit(code, qubit_errors, GateErrorsDict(), logical_op)
        sim = simulator_class(circ)  # type: ignore

        decoder = decoder_class.from_code(
            code, cast(ErrorDataDict, qubit_errors), weighted=True
        )

        test_success = np.zeros([reps], dtype=bool)

        for i in range(reps):
            raw, erasure = sim.get_sample()
            results = SimulatorSample.from_code_and_raw_results(code, raw, erasure)
            correction = decoder.decode(results.erased_qubits, results.syndrome)
            test_success[i] = decoderbase.check_success(
                code, correction, results.logical_op_toggle, logical_op
            )
        assert 0.01 < 1 - np.count_nonzero(test_success) / reps < 0.1

    @pt.mark.slow
    @pt.mark.parametrize("decoder_class", [PyMatchingDecoder, FusionBlossomDecoder])
    @pt.mark.parametrize(
        "simulator_class,reps", [(CircuitSimulator, 1000), (StimSimulator, 10000)]
    )
    def test_pauli_x_errors_vs_logical_error_rates_in_nickerson_thesis(
        self,
        decoder_class: Type[decoderbase.DecoderInterface],
        simulator_class: Type[AbstractSimulator],
        reps: int,
    ):
        """Test to be matched/compared with Fig. 1.11 in doi:10.25560/31475."""
        code = LatticeCode.make_planar(n_rounds=1, size=7)
        logical_op = "Z"
        qubit_errors = QubitErrorsDict(
            pauli={
                vtx.equbit_idx: SinglePauliChannelErrorValueDict(x=0.1)
                for vtx in code.lattice.dataqubits
            }
        )
        plaquette.rng = np.random.default_rng(seed=62934814123)
        circ = generate_qec_circuit(code, qubit_errors, GateErrorsDict(), logical_op)
        sim = simulator_class(circ)  # type: ignore

        decoder = decoder_class.from_code(
            code, cast(ErrorDataDict, qubit_errors), weighted=True
        )

        test_success = np.zeros([reps], dtype=bool)

        for i in range(reps):
            raw, erasure = sim.get_sample()
            results = SimulatorSample.from_code_and_raw_results(code, raw, erasure)
            correction = decoder.decode(results.erased_qubits, results.syndrome)
            test_success[i] = decoderbase.check_success(
                code, correction, results.logical_op_toggle, logical_op
            )
        assert 0.11 < 1 - np.count_nonzero(test_success) / reps < 0.19

    @pt.mark.slow
    @pt.mark.parametrize(
        "simulator_class,reps", [(CircuitSimulator, 100), (StimSimulator, 10000)]
    )
    @pt.mark.parametrize(
        "p_erasure,err_rate,uncertainty",
        [
            (0.7, 0.49, 0.08),
            (0.4, 0.07, 0.04),
            (0.2, 0.01, 0.02),
        ],
    )
    def test_erasure_errors_vs_logical_error_rates(
        self,
        simulator_class: Type[AbstractSimulator],
        reps: int,
        p_erasure: float,
        err_rate: float,
        uncertainty: float,
    ):
        """This has unfortunately no reference in the literature.

        Reference data is from commit 99ed6afdf2150fe3f6fa9a89e4774d326a1bf24f.
        """
        code = LatticeCode.make_planar(n_rounds=1, size=7)
        logical_op = "Z"
        qubit_errors = QubitErrorsDict(
            pauli={
                vtx.equbit_idx: SinglePauliChannelErrorValueDict(y=1e-15)
                for vtx in code.lattice.dataqubits
            },
            erasure={
                vtx.equbit_idx: ErrorValueDict(p=p_erasure)
                for vtx in code.lattice.dataqubits
            },
        )

        plaquette.rng = np.random.default_rng(seed=62934814123)

        circ = generate_qec_circuit(code, qubit_errors, GateErrorsDict(), logical_op)
        sim = simulator_class(circ)  # type: ignore

        dec = UnionFindDecoder.from_code(
            code, cast(ErrorDataDict, qubit_errors), weighted=True
        )

        succ = np.zeros([reps], dtype=bool)

        for i in range(reps):
            raw, erasure = sim.get_sample()
            results = SimulatorSample.from_code_and_raw_results(code, raw, erasure)
            correction = dec.decode(results.erased_qubits, results.syndrome)
            succ[i] = decoderbase.check_success(
                code, correction, results.logical_op_toggle, logical_op
            )

        assert (
            err_rate - uncertainty / 2
            <= 1 - np.count_nonzero(succ) / reps
            <= err_rate + uncertainty / 2
        )

    @pt.mark.slow
    def test_pauli_x_errors_vs_logical_error_rate_in_mark_hu_thesis(self):
        """Test to be compared with Table 5.2 of Mark Hu `thesis <https://doi.org/10.13140/RG.2.2.13495.96162>`_.

        In particular, we are comparing agains "row SBUF" in that table. The
        significance range in the assert was made by basically averaging over
        100 runs of the inner part of this test case, and it's a 3-sigma
        interval.
        """  # noqa

        plaquette.rng = np.random.default_rng(seed=1234567890)

        sys.setrecursionlimit(10000)

        size = 8
        error_rate = 0.09973
        expected_success_rate = 0.8489
        expected_success_rate_error = 0.0033  # 3-sigma
        logical_op = "Z"
        reps = 96000

        code = LatticeCode.make_planar(n_rounds=1, size=size)
        qed = {
            "pauli": {q.equbit_idx: {"x": error_rate} for q in code.lattice.dataqubits}
        }
        circuit = generate_qec_circuit(code, qed, {}, logical_op)
        simulator = StimSimulator(circ=circuit, batch_size=reps)
        decoder = UnionFindDecoder.from_code(code, qed, weighted=False)
        sim_res = list()
        for _ in range(reps):
            sim_res.append(simulator.get_sample())
        success_rate = (
            np.count_nonzero(
                jl.Parallel(n_jobs=2)(  # GH Actions runners have 2 cores
                    jl.delayed(calculate_success)(sim_re, logical_op, code, decoder)
                    for sim_re in sim_res
                )
            )
            / reps
        )

        assert (
            expected_success_rate - expected_success_rate_error / 2
            <= success_rate
            <= expected_success_rate + expected_success_rate_error / 2
        )


@pt.fixture
def config_nickerson_thesis():
    return ExperimentConfig.load_toml("tests/nickerson_thesis_config.toml")


class TestRegressionFrontend:
    @pt.mark.slow
    @pt.mark.parametrize(
        "simulator_class,reps", [("CircuitSimulator", 100), ("StimSimulator", 10000)]
    )
    def test_pauli_and_measurement_errors_vs_logical_error_rates_in_nickerson_thesis(
        self, simulator_class: str, reps: int, config_nickerson_thesis
    ):
        """Test to be matched/compared with Fig. 1.12 in doi:10.25560/31475."""
        config_nickerson_thesis.simulator_conf.update(name=simulator_class, shots=reps)
        config_nickerson_thesis.build()
        test_success = np.zeros([reps], dtype=bool)

        for i in range(reps):
            raw, erasure = config_nickerson_thesis.simulator.get_sample()
            results = SimulatorSample.from_code_and_raw_results(
                config_nickerson_thesis.code, raw, erasure
            )
            correction = config_nickerson_thesis.decoder.decode(
                results.erased_qubits, results.syndrome
            )
            test_success[i] = decoderbase.check_success(
                config_nickerson_thesis.code,
                correction,
                results.logical_op_toggle,
                config_nickerson_thesis.general_conf["logical_op"],
            )
        assert 0.01 < 1 - np.count_nonzero(test_success) / reps < 0.1

    @pt.mark.slow
    @pt.mark.parametrize(
        "simulator_class,reps", [("CircuitSimulator", 1000), ("StimSimulator", 10000)]
    )
    def test_pauli_x_errors_vs_logical_error_rates_in_nickerson_thesis_frontend(
        self, simulator_class: str, reps: int, config_nickerson_thesis
    ):
        """Test to be matched/compared with Fig. 1.12 in doi:10.25560/31475."""
        config_nickerson_thesis.simulator_conf.update(name=simulator_class, shots=reps)

        config_nickerson_thesis.errors_conf.qubit_errors.X.update(params=[0.1])
        config_nickerson_thesis.errors_conf.qubit_errors.measurement.update(
            enabled=False
        )
        config_nickerson_thesis.code_conf.update(rounds=1)
        config_nickerson_thesis.build()
        test_success = np.zeros([reps], dtype=bool)

        for i in range(reps):
            raw, erasure = config_nickerson_thesis.simulator.get_sample()
            results = SimulatorSample.from_code_and_raw_results(
                config_nickerson_thesis.code, raw, erasure
            )
            correction = config_nickerson_thesis.decoder.decode(
                results.erased_qubits, results.syndrome
            )
            test_success[i] = decoderbase.check_success(
                config_nickerson_thesis.code,
                correction,
                results.logical_op_toggle,
                config_nickerson_thesis.general_conf["logical_op"],
            )
        assert 0.11 < 1 - np.count_nonzero(test_success) / reps < 0.19

    @pt.mark.slow
    @pt.mark.parametrize(
        "simulator_class,reps", [("CircuitSimulator", 100), ("StimSimulator", 10000)]
    )
    @pt.mark.parametrize(
        "p_erasure,err_rate,uncertainty",
        [
            (0.7, 0.49, 0.08),
            (0.4, 0.07, 0.04),
            (0.2, 0.01, 0.02),
        ],
    )
    def test_erasure_errors_vs_logical_error_rates_frontend(
        self,
        simulator_class: str,
        reps: int,
        p_erasure: float,
        err_rate: float,
        uncertainty: float,
        config_nickerson_thesis,
    ):
        """This has unfortunately no reference in the literature.

        Reference data is from commit 99ed6afdf2150fe3f6fa9a89e4774d326a1bf24f.
        """
        config_nickerson_thesis.simulator_conf.update(name=simulator_class, shots=reps)
        config_nickerson_thesis.errors_conf.qubit_errors.X.enabled = False
        config_nickerson_thesis.errors_conf.qubit_errors.Y.update(
            distribution="constant", params=[1e-15], enabled=True
        )
        config_nickerson_thesis.errors_conf.qubit_errors.measurement.enabled = False
        config_nickerson_thesis.errors_conf.qubit_errors.erasure.update(
            distribution="constant", params=[p_erasure], enabled=True
        )
        config_nickerson_thesis.decoder_conf.name = "UnionFindDecoder"
        config_nickerson_thesis.code_conf.rounds = 1
        config_nickerson_thesis.build()
        test_success = np.zeros([reps], dtype=bool)

        for i in range(reps):
            raw, erasure = config_nickerson_thesis.simulator.get_sample()
            results = SimulatorSample.from_code_and_raw_results(
                config_nickerson_thesis.code, raw, erasure
            )
            correction = config_nickerson_thesis.decoder.decode(
                results.erased_qubits, results.syndrome
            )
            test_success[i] = decoderbase.check_success(
                config_nickerson_thesis.code,
                correction,
                results.logical_op_toggle,
                config_nickerson_thesis.general_conf["logical_op"],
            )

        assert (
            err_rate - uncertainty / 2
            <= 1 - np.count_nonzero(test_success) / reps
            <= err_rate + uncertainty / 2
        )
