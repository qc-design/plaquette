# Copyright 2023, It'sQ GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

import numpy as np
import pytest as pt

import plaquette
from plaquette.circuit.generator import QECCircuitGenerator, generate_qec_circuit
from plaquette.codes import LatticeCode
from plaquette.device import Device, MeasurementSample
from plaquette.errors import (
    ErrorValueDict,
    GateErrorsDict,
    QubitErrorsDict,
    SinglePauliChannelErrorValueDict,
    TwoPauliChannelErrorValueDict,
)


@pt.fixture
def qubit_errordata():
    return QubitErrorsDict()


@pt.fixture()
def gate_errordata():
    return GateErrorsDict()


@pt.fixture
def code():
    return LatticeCode.make_repetition(n_rounds=1, size=3)


class TestQECCircuitGenerator:
    @pt.mark.parametrize(
        "qubit_errors, last_instruction",
        [
            (
                QubitErrorsDict(pauli={2: SinglePauliChannelErrorValueDict(x=0.2)}),
                "E_PAULI 0.2 0.0 0.0 2",
            ),
            (
                QubitErrorsDict(
                    pauli={0: SinglePauliChannelErrorValueDict(x=0.2, y=0.1, z=0.9)}
                ),
                "E_PAULI 0.2 0.1 0.9 0",
            ),
            (
                QubitErrorsDict(erasure={1: ErrorValueDict(p=0.2)}),
                "E_ERASE 0.2 1",
            ),
        ],
    )
    def test_apply_dataqubit_errors(
        self,
        code: LatticeCode,
        qubit_errors: QubitErrorsDict,
        last_instruction: str,
    ):
        circgen = QECCircuitGenerator(code, qubit_errors, GateErrorsDict())
        circgen.apply_dataqubit_errors()
        assert str(circgen.cb.circ).split("\n")[-1] == last_instruction

    @pt.mark.parametrize(
        "log_ops, num_gates, gates, logical_ancilla_bool",
        [
            ([0], 7, ["R 5", "H 5", "CX 5 0", "CX 5 1", "CX 5 2", "H 5", "M 5"], True),
            ([1], 5, ["R 6", "H 6", "CZ 6 2", "H 6", "M 6"], True),
            ([0], 9, ["H 0", "M 0", "H 0", "H 1", "M 1", "H 1", "H 2", "M 2", "H 2"], False),
            ([1], 1, ["M 2"], False),
        ],
    )
    def test_measure_logical_ops(
        self,
        qubit_errordata: QubitErrorsDict,
        gate_errordata: GateErrorsDict,
        code: LatticeCode,
        log_ops: Sequence[int],
        num_gates: int,
        gates: Sequence[str],
        logical_ancilla_bool: bool,
    ):
        circgen = QECCircuitGenerator(
            code, qubit_errordata, gate_errordata, logical_ancilla_bool
        )
        circgen.measure_logical_ops(log_ops)
        assert str(circgen.cb.circ).split("\n")[-num_gates:] == gates

    @pt.mark.parametrize(
        "num_gates, gates",
        [
            (
                12,
                [
                    "R 3",
                    "H 3",
                    "CZ 3 0",
                    "CZ 3 1",
                    "H 3",
                    "M 3",
                    "R 4",
                    "H 4",
                    "CZ 4 1",
                    "CZ 4 2",
                    "H 4",
                    "M 4",
                ],
            ),
        ],
    )
    def test_measure_stabgens(
        self,
        qubit_errordata: QubitErrorsDict,
        gate_errordata: GateErrorsDict,
        code: LatticeCode,
        num_gates: int,
        gates: Sequence[str],
    ):
        circgen = QECCircuitGenerator(code, qubit_errordata, gate_errordata)
        circgen.measure_stabgens(with_errors=False)
        assert str(circgen.cb.circ).split("\n")[-num_gates:] == gates

    @pt.mark.parametrize(
        "gate_errs, gate_instr, err_instr",
        [
            (
                GateErrorsDict(
                    H={3: SinglePauliChannelErrorValueDict(z=0.1)},
                ),
                "H 3",
                "E_PAULI 0.0 0.0 0.1 3",
            ),
            (
                GateErrorsDict(
                    R={4: SinglePauliChannelErrorValueDict(y=0.1)},
                ),
                "R 4",
                "E_PAULI 0.0 0.1 0.0 4",
            ),
            (
                GateErrorsDict(
                    M={3: SinglePauliChannelErrorValueDict(x=0.1)},
                ),
                "M 3",
                "E_PAULI 0.1 0.0 0.0 3",
            ),
            (
                GateErrorsDict(CZ={(3, 1): TwoPauliChannelErrorValueDict(zi=0.1)}),
                "CZ 3 1",
                "E_PAULI2 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 "
                "0.0 0.1 0.0 0.0 0.0 3 1",
            ),
        ],
    )
    def test_measure_stabgens_errors(
        self,
        code: LatticeCode,
        qubit_errordata: QubitErrorsDict,
        gate_errs: GateErrorsDict,
        gate_instr: str,
        err_instr: str,
    ):
        # errordata.update_by_index(*params)

        circgen = QECCircuitGenerator(code, qubit_errordata, gate_errs)
        circgen.measure_stabgens(with_errors=True)

        gatelist = circgen.cb.circ.__str__().split("\n")
        indices = [i for i, gate in enumerate(gatelist) if gate == gate_instr]
        e_indices = [index + 1 for index in indices]
        error_list = [gatelist[i] for i in e_indices]

        assert all([x == error_list[0] for x in error_list])
        assert error_list[0] == err_instr

    def test_generated_syndrome(self):
        """Flip all X stabiliser measurements and check relative syndrome."""
        plaquette.rng = np.random.default_rng(seed=62934814123)
        code = LatticeCode.make_planar(n_rounds=1, size=4)
        # Code has same dataqubits/edges/equbit props as `master`.
        qubit_errors = QubitErrorsDict(
            pauli={
                4: SinglePauliChannelErrorValueDict(z=1),
                6: SinglePauliChannelErrorValueDict(z=1),
                11: SinglePauliChannelErrorValueDict(z=1),
                13: SinglePauliChannelErrorValueDict(z=1),
                18: SinglePauliChannelErrorValueDict(z=1),
                20: SinglePauliChannelErrorValueDict(z=1),
            }
        )

        circ = generate_qec_circuit(code, qubit_errors, GateErrorsDict(), "X")

        dev = Device("clifford")
        dev.run(circ)
        raw, erasure = dev.get_sample()
        results = MeasurementSample.from_code_and_raw_results(code, raw, erasure)
        # fmt: off
        assert (results.syndrome == np.array([
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
        ], dtype='u1')).all()
        # fmt: on
