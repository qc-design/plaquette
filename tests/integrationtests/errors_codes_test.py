# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
import pytest as pt

from plaquette import codes
from plaquette.errors import ErrorData, ErrorDataDict, ErrorValueDict


@pt.fixture
def errordata():
    return ErrorDataDict()


@pt.fixture
def code():
    return codes.Code.make_rotated_planar(distance=3)


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
                "Index 6 for gate measurement is not an ancilla",
            ),
            (
                ("pauli", 4, {"p": 0.5}),
                "pauli qubit 4: Invalid keys {'p'}",
            ),
            # TODO: Add gate error examples
        ],
    )
    def test_check_internal(
        self,
        errordata: ErrorDataDict,
        code: codes.Code,
        params: tuple,
        error_msg: str,
    ):
        ed_ui = ErrorData()
        errordata[params[0]] = {params[1]: ErrorValueDict(params[2])}  # type: ignore
        with pt.raises(ValueError) as excinfo:
            ed_ui.check_against_code(code, errordata)

        assert str(excinfo.value) == error_msg
