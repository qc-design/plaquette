# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
import typing as t

import numpy as np
import pytest as pt

from plaquette.pauli import (
    Factor,
    Tableau,
    _g,
    commutator_sign,
    count_qubits,
    is_css,
    measure,
    measure_x_base,
    measure_z_base,
    multiply,
    pad_operator,
    pauli_to_dict,
    single_qubit_pauli_operator,
    sort_operators_ref,
    string_to_pauli,
    x,
    z,
)


@pt.mark.parametrize(
    "ops,result",
    [
        [
            [np.array([0, 0, 0, 1, 0]), np.array([0, 0, 0, 1, 0])],
            (np.array([0, 0, 0, 0, 0]), 1),
        ],
        [
            [np.array([0, 0, 0, 1, 1]), np.array([0, 0, 0, 1, 0])],
            (np.array([0, 0, 0, 0, 1]), -1),
        ],
        [
            [np.array([1, 0, 0, 1, 0]), np.array([0, 1, 1, 0, 0])],
            (np.array([1, 1, 1, 1, 0]), 1),
        ],
        [
            [np.array([1, 0, 0, 1, 1]), np.array([0, 1, 1, 0, 0])],
            (np.array([1, 1, 1, 1, 1]), -1),
        ],
        [
            [np.array([1, 0, 0, 1, 0]), np.array([0, 1, 1, 0, 0])],
            (np.array([1, 1, 1, 1, 0]), 1),
        ],
        [
            [np.array([1, 0, 0, 1, 0]), np.array([0, 1, 1, 0, 0])],
            (np.array([1, 1, 1, 1, 0]), 1),
        ],
        [
            [np.array([1, 0, 0, 1, 0]), np.array([0, 1, 0, 0, 0])],
            (np.array([1, 1, 0, 1, 0]), 1j),
        ],
        [
            [np.array([1, 0, 0, 1, 0]), np.array([1, 0, 1, 0, 0])],
            (np.array([0, 0, 1, 1, 0]), 1j),
        ],
        [
            [np.array([1, 0, 0, 1, 0]), np.array([1, 0, 1, 0, 0])],
            (np.array([0, 0, 1, 1, 0]), 1j),
        ],
        [
            [
                np.array([1, 0, 0, 1, 0]),  # ZX
                np.array([1, 0, 0, 1, 0]),  # ZX
                np.array([1, 1, 0, 0, 0]),  # ZZ
            ],
            (np.array([1, 1, 0, 0, 0]), 1),
        ],
        [
            [
                np.array([1, 1, 0, 0, 0]),  # ZZ
                np.array([1, 0, 0, 1, 0]),  # ZX
                np.array([1, 1, 0, 0, 0]),  # ZZ
            ],
            (np.array([1, 0, 0, 1, 1]), -1),
        ],
        [
            [  # Since iI = XYZ = YZX
                np.array([1, 1, 0, 1, 0]),  # XY
                np.array([1, 0, 1, 1, 0]),  # YZ
                np.array([0, 1, 1, 0, 0]),  # ZX
            ],
            (np.array([0, 0, 0, 0, 0]), 1),  # (XYZ) (YZX) = (iI)^2 = -I
        ],
    ],
)
def test_operator_product(ops, result):
    """Ensure that plain multiplication of two or more operators works."""
    data, phase = multiply(*ops)
    assert np.all(data == result[0])
    assert np.all(phase == result[1])


def test_measurements_with_fixed_outcomes():
    """Make sure all fixed outcomes are actually returned."""
    n_tot = 10
    zs = np.eye(2 * n_tot + 1, dtype="u1")[:-1]
    # X1 operator without using stuff from pauli.py
    m_op = np.zeros(2 * n_tot + 1, dtype="u1")
    m_op[0] = 1

    state, res = measure(zs, m_op, forced_outcome=1)

    assert res == 1


def test_base_measurement_with_fixed_outcome():
    """Make sure all fixed outcomes are actually returned for a given base."""
    n_tot = 10
    zs = np.eye(2 * n_tot + 1, dtype="u1")[:-1]
    fo = [0, 0, 0]

    state, res = measure_x_base(zs, [0, 1, 2], forced_outcomes=fo)

    assert all([r == f for r, f in zip(res, fo)])


def test_base_measurement_with_fixed_outcome_failure():
    """Trigger error due to size mismatch between targets and forced outcomes."""
    n_tot = 10
    zs = np.eye(2 * n_tot + 1, dtype="u1")[:-1]
    fo = [0, 0]

    with pt.raises(
        ValueError,
        match="The number of forced outcomes must match the number of targets, "
        "or be a single scalar.",
    ):
        measure_x_base(zs, [0, 1, 2], forced_outcomes=fo)


@pt.mark.parametrize(
    "op,total",
    [
        [np.array([0, 0, 0]), 5],
        [np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 3],
        [np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 1],
    ],
)
def test_operator_padding(op: np.ndarray, total: int):
    """Pad an operator and check its new size."""
    new_op = pad_operator(op, total)
    assert new_op.size == max(2 * total + 1, op.size)


@pt.mark.parametrize(
    "op,res",
    [
        (np.array([0, 0, 0]), {}),
        (np.array([0, 1, 0]), {0: Factor.Z}),
        (np.array([1, 0, 0]), {0: Factor.X}),
        (np.array([1, 1, 0]), {0: Factor.Y}),
        (np.array([0, 0, 0, 0, 0]), {}),
        (np.array([0, 1, 1, 0, 0]), {0: Factor.Z, 1: Factor.X}),
        (np.array([0, 1, 0, 0, 0]), {1: Factor.X}),
        (np.array([1, 1, 1, 1, 0]), {0: Factor.Y, 1: Factor.Y}),
        (np.array([0, 0, 0, 1, 0]), {1: Factor.Z}),
    ],
)
def test_pauli_to_dict(op, res):
    """Test binary-to-dict conversion for operators."""
    assert pauli_to_dict(op) == res


@pt.mark.parametrize(
    "bits,exponent",
    [
        [[0, 0, 0, 0], 0],
        [[0, 0, 0, 1], 0],
        [[0, 0, 1, 0], 0],
        [[0, 0, 1, 1], 0],
        [[0, 1, 0, 0], 0],
        [[0, 1, 0, 1], 0],
        [[0, 1, 1, 0], 1],
        [[0, 1, 1, 1], -1],
        [[1, 0, 0, 0], 0],
        [[1, 0, 0, 1], -1],
        [[1, 0, 1, 0], 0],
        [[1, 0, 1, 1], 1],
        [[1, 1, 0, 0], 0],
        [[1, 1, 0, 1], 1],
        [[1, 1, 1, 0], -1],
        [[1, 1, 1, 1], 0],
    ],
)
def test_phase_exponent_g(bits, exponent):
    assert _g(*bits) == exponent


@pt.mark.parametrize(
    "op,qubit,qubits",
    [
        ("X", 1, 3),
        ("X", 1, 9),
        ("X", 0, 1),
    ],
)
def test_single_qubit_operator_creation(op: str, qubit: int, qubits: int):
    """Test basic functionality to create a single-qubit operator."""
    bin_op = single_qubit_pauli_operator(op, qubit, qubits)
    assert bin_op.size // 2 == qubits
    assert bin_op[qubit] or bin_op[qubit + qubits]  # tests if the right bit was set
    assert not bin_op[~np.array((qubit, qubits + qubit))].any()  # tests all 0


def test_single_qubit_operator_creation_fail():
    """Try to hit all possible problems when creating a single-qubit operator."""
    with pt.raises(
        ValueError, match=r"Total number of qubits less than target qubit position"
    ):
        single_qubit_pauli_operator("X", 1, 1)
    with pt.raises(ValueError, match=r"Total qubit number must be 1 or more"):
        single_qubit_pauli_operator("X", 1, 0)
    with pt.raises(ValueError, match=r"Qubits indices can't be negative"):
        single_qubit_pauli_operator("X", -1, 1)
    with pt.raises(
        ValueError, match=r"Only X, Y, or Z are allowed, optionally with sign"
    ):
        single_qubit_pauli_operator("A", 0, 1)


@pt.mark.parametrize(
    "op1,op2,comm_res",
    [
        # identity
        [np.array([0, 0, 0]), np.array([0, 0, 0]), 0],
        [np.array([0, 0, 0]), np.array([0, 1, 0]), 0],
        [np.array([0, 0, 0]), np.array([1, 0, 0]), 0],
        [np.array([0, 0, 0]), np.array([1, 1, 0]), 0],
        # z
        [np.array([0, 1, 0]), np.array([0, 0, 0]), 0],
        [np.array([0, 1, 0]), np.array([0, 1, 0]), 0],
        [np.array([0, 1, 0]), np.array([1, 0, 0]), 1],
        [np.array([0, 1, 0]), np.array([1, 1, 0]), 1],
        # x
        [np.array([1, 0, 0]), np.array([0, 0, 0]), 0],
        [np.array([1, 0, 0]), np.array([0, 1, 0]), 1],
        [np.array([1, 0, 0]), np.array([1, 0, 0]), 0],
        [np.array([1, 0, 0]), np.array([1, 1, 0]), 1],
        # y
        [np.array([1, 1, 0]), np.array([0, 0, 0]), 0],
        [np.array([1, 1, 0]), np.array([0, 1, 0]), 1],
        [np.array([1, 1, 0]), np.array([1, 0, 0]), 1],
        [np.array([1, 1, 0]), np.array([1, 1, 0]), 0],
        # zero state with Paulis
        [np.eye(5, dtype=np.uint8)[:-1], np.array([0, 0, 0, 0, 0]), [0, 0, 0, 0]],
        [np.eye(5, dtype=np.uint8)[:-1], np.array([0, 1, 0, 0, 0]), [0, 0, 0, 1]],
        [np.eye(5, dtype=np.uint8)[:-1], np.array([1, 0, 0, 0, 0]), [0, 0, 1, 0]],
        [np.eye(5, dtype=np.uint8)[:-1], np.array([1, 1, 0, 0, 0]), [0, 0, 1, 1]],
    ],
)
def test_commutator(op1, op2, comm_res: int):
    """Test that the commutator_sign works for all use-cases."""
    assert np.all(commutator_sign(op1, op2).flatten() == comm_res)


@pt.mark.parametrize(
    "ops,qubits",
    [
        (np.array([0, 0, 0]), [1]),
        (np.array([0, 1, 0]), [1]),
        (np.array([0, 0, 0, 0, 0]), [2]),
        (
            [
                np.array([0, 0, 0]),
                np.array([0, 0, 0]),
                np.array([0, 0, 0]),
            ],
            [1, 1, 1],
        ),
        (
            np.array(
                [
                    np.array([0, 0, 0]),
                    np.array([0, 0, 0]),
                    np.array([0, 0, 0]),
                ]
            ),
            [1],
        ),
        (  # State input
            np.array(
                [
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                ]
            ),
            [2],
        ),
    ],
)
def test_count_qubits_with_identities(ops, qubits):
    """Make sure we count all qubits."""
    assert all(o == q for o, q in zip(count_qubits(ops), qubits))


@pt.mark.parametrize(
    "ops,qubits",
    [
        (np.array([0, 0, 0]), [0]),
        (np.array([0, 1, 0]), [1]),
        (np.array([0, 0, 0, 0, 0]), [0]),
        (
            [
                np.array([0, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, 0, 0]),
            ],
            [0, 1, 0],
        ),
        (
            np.array(
                [
                    np.array([0, 0, 0]),
                    np.array([0, 0, 0]),
                    np.array([0, 0, 0]),
                ]
            ),
            [0],
        ),
        (  # State input
            np.array(
                [
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                ]
            ),
            [1, 1, 1, 1],
        ),
    ],
)
def test_count_qubits_without_identities(ops, qubits):
    """Test that we are not counting identities when we say not to."""
    assert all(o == q for o, q in zip(count_qubits(ops, False), qubits))


def test_z_measurement_on_zero_state_with_binary_op():
    """Make sure the measure() function deals correctly with raw arrays."""
    n_tot = 10
    zs = np.eye(2 * n_tot + 1, dtype="u1")[:-1]
    reference = zs.copy()
    for _i in range(n_tot):
        op = np.zeros(2 * n_tot + 1, dtype="u1")
        op[n_tot + 1] = 1
        after_state, res = measure(zs, op)
        assert res == 0
        assert np.all(after_state == reference)


def test_z_measurement_on_zero_state_with_string_op():
    """Make sure the measure() function deals correctly with strings."""
    n_tot = 10
    zs = np.eye(2 * n_tot + 1, dtype="u1")[:-1]
    reference = zs.copy()
    for i in range(n_tot):
        after_state, res = measure(zs, f"Z{i:d}")
        assert res == 0
        assert np.all(after_state == reference)
        # The weird string is just a Z surrounded by Is
        after_state, res = measure(zs, "I" * i + "Z" + "I" * (n_tot - i - 1))
        assert res == 0
        assert np.all(after_state == reference)


def test_base_measurements_are_flat_with_one_target():
    """Test that given a single target we return a scalar, not a length-1 sequence."""
    n_tot = 10
    zs = np.eye(2 * n_tot + 1, dtype="u1")[:-1]
    _, results = measure_z_base(zs, 0)
    assert isinstance(results, np.uint8)
    _, results = measure_z_base(zs, [0, 1])
    assert len(results) == 2


def test_z_measurement_on_zero_state_failure():
    n_tot = 10
    zs = np.eye(2 * (n_tot - 1) + 1, dtype="u1")[:-1]
    for _i in range(n_tot):
        op = np.zeros(2 * n_tot + 1, dtype="u1")
        op[n_tot + 1] = 1
        with pt.raises(ValueError):
            measure(zs, op)


@pt.mark.parametrize(
    "qubits,sign_row", [(1, [0, 0, 0, 1]), ([0, 1], [0, 0, 1, 1]), ([0], [0, 0, 1, 0])]
)
def test_x_gate(qubits: int | list[int], sign_row: list[int]):
    """Test the X-gate tableau conjugations action."""
    state = np.eye(5, dtype=np.uint8)[:-1]
    state = x(state, qubits)
    assert np.all(state[:, -1] == np.array(sign_row))
    assert np.all(state[:, :-1] == np.eye(4))


@pt.mark.parametrize(
    "qubits,sign_row", [(1, [0, 1, 0, 0]), ([0, 1], [1, 1, 0, 0]), ([0], [1, 0, 0, 0])]
)
def test_z_gate(qubits: int | list[int], sign_row: list[int]):
    """Test the Z-gate tableau conjugations action."""
    state = np.eye(5, dtype=np.uint8)[:-1]
    state = z(state, qubits)
    assert np.all(state[:, -1] == np.array(sign_row))
    assert np.all(state[:, :-1] == np.eye(4))


@pt.mark.parametrize(
    "ops, sorted_ops",
    [
        # Planar Code Distance 2
        (
            [
                string_to_pauli("Z0Z1Z2", 5),
                string_to_pauli("Z2Z3Z4", 5),
                string_to_pauli("X0X2X3", 5),
                string_to_pauli("X1X2X4", 5),
            ],
            [
                string_to_pauli("X0X2X3", 5),
                string_to_pauli("X1X2X4", 5),
                string_to_pauli("Z0Z1Z2", 5),
                string_to_pauli("Z2Z3Z4", 5),
            ],
        ),
        # Rotated Planar XZZX Code
        (
            [
                string_to_pauli("X0Z3", 9),
                string_to_pauli("X1Z2", 9),
                string_to_pauli("X8Z5", 9),
                string_to_pauli("Z6X7", 9),
                string_to_pauli("Z0X1X3Z4", 9),
                string_to_pauli("Z1X2X4Z5", 9),
                string_to_pauli("Z3X4X6Z7", 9),
                string_to_pauli("Z4X5X7Z8", 9),
            ],
            [
                string_to_pauli("X0Z3", 9),
                string_to_pauli("X1Z2", 9),
                string_to_pauli("Z0X1X3Z4", 9),
                string_to_pauli("Z1X2X4Z5", 9),
                string_to_pauli("Z3X4X6Z7", 9),
                string_to_pauli("Z4X5X7Z8", 9),
                string_to_pauli("Z6X7", 9),
                string_to_pauli("X8Z5", 9),
            ],
        ),
    ],
)
def test_sort_operators_ref(ops: t.Sequence[Tableau], sorted_ops: t.Sequence[Tableau]):
    assert all(
        [np.array_equal(r, o) for r, o in zip(sort_operators_ref(ops), sorted_ops)]
    )


@pt.mark.parametrize(
    "ops, ret_val",
    [
        (
            [
                string_to_pauli("Z0Z1Z2", 5),
                string_to_pauli("Z2Z3Z4", 5),
                string_to_pauli("X0X2X3", 5),
                string_to_pauli("X1X2X4", 5),
            ],
            True,
        ),
        (
            [
                string_to_pauli("X0Z3", 9),
                string_to_pauli("X1Z2", 9),
                string_to_pauli("Z0X1X3Z4", 9),
                string_to_pauli("Z1X2X4Z5", 9),
                string_to_pauli("Z3X4X6Z7", 9),
                string_to_pauli("Z4X5X7Z8", 9),
                string_to_pauli("Z6X7", 9),
                string_to_pauli("X8Z5", 9),
            ],
            False,
        ),
    ],
)
def test_is_css(ops: t.Sequence[Tableau], ret_val: bool):
    assert is_css(ops) == ret_val
