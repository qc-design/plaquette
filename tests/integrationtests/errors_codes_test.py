# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
import pytest as pt

from plaquette.codes import LatticeCode
from plaquette.errors import ErrorData, ErrorDataDict, ErrorValueDict


@pt.fixture
def errordata():
    return ErrorDataDict()


@pt.fixture
def code():
    return LatticeCode.make_rotated_planar(n_rounds=1, size=3)


class TestErrorData:
    @pt.mark.parametrize(
        "params, error_msg",
        [
            (
                ("Erasure", 3, {"p": 0.2}),
                "Unknown errors: {'Erasure'}",
            ),
            (
                ("measurement", 6, {"p": 0.4}),
                "For {'error': 'measurement'}: Not a stabilizer generator: "
                "DataVertex(pos = (5, 1), ext_idx = 6, data_idx = 6)",
            ),
            (
                ("pauli", 4, {"p": 0.5}),
                "pauli qubit 4: Invalid keys {'p'}",
            ),
            # TODO: Add gate error examples
        ],
    )
    def test_check_internal(
        self, errordata: ErrorDataDict, code: LatticeCode, params: tuple, error_msg: str
    ):
        ed_ui = ErrorData()
        errordata[params[0]] = {params[1]: ErrorValueDict(params[2])}  # type: ignore
        with pt.raises(ValueError) as excinfo:
            ed_ui.check_against_code(code, errordata)

        assert str(excinfo.value) == error_msg
