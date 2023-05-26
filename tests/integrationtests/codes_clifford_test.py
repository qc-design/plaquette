# Copyright 2023, It'sQ GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

import numpy as np
import pytest as pt

from plaquette.codes import StabilizerCode
from plaquette.pauli import Tableau


class TestStabilizerCode:
    """Tests for plaquette.code.StabilizerCode."""

    @pt.mark.parametrize(
        "stabgens, logops",
        [
            (
                [np.array([1, 1, 0, 0, 0, 0, 0]), np.array([0, 1, 1, 0, 0, 0, 0])],
                [np.array([0, 0, 0, 1, 0, 0, 0])],
            ),
            (
                [np.array([1, 1, 0, 0, 0, 0, 0]), np.array([0, 1, 1, 0, 0, 0, 0])],
                [np.ndarray([0, 1, 0]), np.ndarray([1, 0, 0])],
            ),
        ],
    )
    def test_init_error(self, stabgens: list[Tableau], logops: list[Tableau]):
        with pt.raises(AssertionError):
            StabilizerCode(stabgens, logops)

    @pt.mark.parametrize(
        "logical_ops, logop_indices",
        [("XX", [0, 2]), ("ZZ", [1, 3]), ("ZX", [1, 2]), ([0, 3], [0, 3])],
    )
    def test_logical_ops_to_indices(
        self, logical_ops: str | Sequence[int], logop_indices: Sequence[int]
    ):
        stabcode = StabilizerCode(
            [
                np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
                np.array([0, 1, 1, 0, 0, 0, 0, 0, 0]),
            ],
            [
                np.array([0, 0, 1, 1, 0, 0, 0, 0, 0]),
                np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
                np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),
                np.array([0, 0, 0, 0, 1, 1, 1, 1, 0]),
            ],
        )
        assert stabcode.logical_ops_to_indices(logical_ops) == logop_indices

    @pt.mark.parametrize(
        "stabcode, error_msg",
        [
            (
                StabilizerCode(
                    [np.array([1, 1, 0, 0, 0, 0, 0]), np.array([0, 1, 1, 0, 0, 0, 0])],
                    [np.array([0, 0, 0, 1, 1, 0, 0]), np.array([0, 0, 1, 0, 0, 0, 0])],
                ),
                "Logical X and Zs do not satisfy commutation relations",
            ),
            (
                StabilizerCode(
                    [
                        np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
                        np.array([0, 1, 1, 0, 0, 0, 0, 0, 0]),
                    ],
                    [
                        np.array([0, 0, 1, 1, 0, 0, 0, 0, 0]),
                        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),
                        np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
                        np.array([0, 0, 0, 0, 1, 1, 1, 1, 0]),
                    ],
                ),
                "Logical X and Zs do not satisfy commutation relations",
            ),
            (
                StabilizerCode(
                    [
                        np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
                        np.array([0, 1, 1, 0, 0, 0, 0, 0, 0]),
                    ],
                    [
                        np.array([0, 0, 1, 1, 0, 0, 0, 0, 0]),
                        np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
                        np.array([1, 0, 0, 0, 0, 0, 0, 1, 0]),
                        np.array([0, 0, 0, 0, 1, 1, 1, 1, 0]),
                    ],
                ),
                "Logical Xs do not commute with themselves",
            ),
            (
                StabilizerCode(
                    [
                        np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
                        np.array([0, 1, 1, 0, 0, 0, 0, 0, 0]),
                    ],
                    [
                        np.array([0, 0, 1, 1, 0, 0, 0, 0, 0]),
                        np.array([1, 0, 0, 0, 0, 0, 0, 1, 0]),
                        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),
                        np.array([0, 0, 0, 0, 1, 1, 1, 1, 0]),
                    ],
                ),
                "Logical Zs do not commute with themselves",
            ),
            (
                StabilizerCode(
                    [
                        np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
                        np.array([0, 1, 1, 0, 0, 0, 0, 0, 0]),
                        np.array([0, 0, 1, 1, 0, 0, 0, 0, 0]),
                    ],
                    [
                        np.array([0, 0, 0, 0, 1, 0, 0, 1, 0]),
                        np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]),
                    ],
                ),
                "Stabilizer generators do not commute with logical operators",
            ),
            (
                StabilizerCode(
                    [
                        np.array([1, 1, 0, 0, 0, 0, 0]),
                        np.array([0, 1, 1, 1, 0, 0, 0]),
                    ],
                    [
                        np.array([0, 0, 0, 1, 1, 1, 0]),
                        np.array([0, 0, 1, 0, 0, 0, 0]),
                    ],
                ),
                "Stabilizer generators do not commute with themselves",
            ),
        ],
    )
    def test_check(self, stabcode: StabilizerCode, error_msg: str):
        with pt.raises(AssertionError) as excinfo:
            stabcode.check()

        assert str(excinfo.value) == error_msg

    @pt.mark.parametrize(
        "logical_ops, error_msg",
        [
            ("XI", "logical_ops='XI' is invalid"),
            ("Z", "Need one logical operator for each logical qubit"),
            ([1, 4], "Invalid index in `logical_ops`"),
        ],
    )
    def test_logical_ops_to_indices_error(
        self, logical_ops: str | Sequence[int], error_msg: str
    ):
        stabcode = StabilizerCode(
            [
                np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
                np.array([0, 1, 1, 0, 0, 0, 0, 0, 0]),
            ],
            [
                np.array([0, 0, 1, 1, 0, 0, 0, 0, 0]),
                np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
                np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),
                np.array([0, 0, 0, 0, 1, 1, 1, 1, 0]),
            ],
        )
        with pt.raises(ValueError) as excinfo:
            stabcode.logical_ops_to_indices(logical_ops)

        assert str(excinfo.value) == error_msg
