# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
from itertools import product

import numpy as np
import pandas as pd
import pytest as pt

from plaquette.codes import LatticeCode
from plaquette.errors import (
    ErrorData,
    assimilate_gate_errors,
    assimilate_qubit_errors,
    delimited_string_list_to_series,
    generate_constant_errors,
    generate_empty_qubit_errors,
    generate_gaussian_errors,
)


@pt.mark.parametrize(
    "input_tuple, exp_opt",
    (
        [(["XX|ZZ", "ZZ|XX"], "str"), pd.Series([["XX", "ZZ"], ["ZZ", "XX"]])],
        [
            (["1|2|3|4", "13|12", "1"], "int"),
            pd.Series(
                [
                    [1, 2, 3, 4],
                    [13, 12],
                    [
                        1,
                    ],
                ]
            ),
        ],
        [
            (["0.1|0.2|1.3|1.4", "1.3|1.2"], "float"),
            pd.Series([[0.1, 0.2, 1.3, 1.4], [1.3, 1.2]]),
        ],
    ),
)
def test_delimited_string_list_to_series(input_tuple: tuple, exp_opt: pd.Series):
    pd.testing.assert_series_equal(
        delimited_string_list_to_series(*input_tuple), exp_opt
    )


@pt.mark.parametrize(
    "input_tuple, exc, exp_opt",
    (
        [
            (["XX,ZZ", "ZZ,XX"], "str"),
            AssertionError,
            pd.Series([["XX", "ZZ"], ["ZZ", "XX"]]),
        ],
        [
            (["1|2|3|4", "13|12", "1"], "str"),
            AssertionError,
            pd.Series(
                [
                    [1, 2, 3, 4],
                    [13, 12],
                    [1],
                ]
            ),
        ],
        [
            (["0.1|0.2|1.3|1.4", "1.3|1.2"], "int"),
            ValueError,
            pd.Series([[0.1, 0.2, 1.3, 1.4], [1.3, 1.2]]),
        ],
    ),
)
def test_delimited_string_list_to_series_fail(
    input_tuple: tuple, exc: Exception, exp_opt: pd.Series
):
    with pt.raises(exc):
        pd.testing.assert_series_equal(
            delimited_string_list_to_series(*input_tuple), exp_opt
        )


@pt.mark.parametrize(
    "mean, std_dev", [(np.linspace(0.01, 1.0, 20), np.linspace(0.01, 1.0, 20))]
)
def test_generate_gaussian_errors(mean, std_dev):
    for m, s in product(mean, std_dev):
        errors = generate_gaussian_errors(m, s, list(np.arange(int(m * 100))))
        assert np.all((errors >= 0.0) & (errors <= 1.0))
        assert len(errors) == int(m * 100)
        assert m - 3 * s < np.mean(errors) < m + 3 * s


@pt.mark.parametrize("val", np.linspace(0.01, 1.0, 20))
def test_generate_constant_errors(val):
    assert np.array_equal(
        generate_constant_errors(val, list(np.arange(int(val * 100)))),
        np.ones(int(val * 100)) * val,
    )


@pt.mark.parametrize(
    "code",
    [
        LatticeCode.make_rotated_planar(size=3, n_rounds=10),
        LatticeCode.make_planar(size=3, n_rounds=10),
        LatticeCode.make_repetition(size=9, n_rounds=3),
        LatticeCode.make_five_qubit(n_rounds=10),
        LatticeCode.make_shor(n_rounds=6),
    ],
)
@pt.mark.parametrize(
    "error_names",
    [
        ["X", "Y", "Z"],
        ["erasure", "measurement", "fabrication"],
        ["erasure", "X", "fabrication"],
        ["measurement", "Z", "fabrication"],
        ["Y"],
        ["X", "Y", "Z", "erasure", "measurement", "fabrication"],
    ],
)
def test_generate_empty_qubit_errors(code, error_names):
    df = generate_empty_qubit_errors(code.lattice, error_names)
    assert not set(df.keys()).difference(set(["qubit_id", "qubit_type"] + error_names))
    for key in error_names:
        if key == "fabrication":
            assert len(df[key]) == code.n_data_qubits + code.n_stabgens
            assert all(
                df[key].array == ["available"] * (code.n_data_qubits + code.n_stabgens)
            )
        else:
            assert len(df[key]) == code.n_data_qubits + code.n_stabgens
            assert np.all(
                np.array_equal(
                    df[key].array, np.zeros(code.n_data_qubits + code.n_stabgens)
                )
            )


@pt.mark.parametrize(
    "qubit_error_config",
    [
        tuple(
            (
                {
                    "X": ("constant", 0.01),
                    "Y": ("constant", 0.1),
                    "Z": ("constant", 0.2),
                },
                "dummy.csv",
            )
        ),
        tuple(
            (
                {
                    "erasure": ("constant", 0.01),
                    "measurement": ("constant", 0.1),
                    "Z": ("constant", 0.2),
                },
                "dummy.csv",
            )
        ),
        tuple(
            (
                {
                    "erasure": ("gaussian", 0.1, 0.1),
                    "measurement": ("gaussian", 0.1, 0.09),
                    "Z": ("gaussian", 0.2, 0.05),
                },
                "dummy.csv",
            )
        ),
    ],
)
@pt.mark.parametrize(
    "code",
    [
        LatticeCode.make_rotated_planar(size=3, n_rounds=10),
        LatticeCode.make_planar(size=3, n_rounds=10),
        LatticeCode.make_repetition(size=9, n_rounds=3),
        LatticeCode.make_five_qubit(n_rounds=10),
        LatticeCode.make_shor(n_rounds=6),
    ],
)
def test_assimilate_qubit_errors_and_no_csv(qubit_error_config: tuple[dict, str], code):
    df = assimilate_qubit_errors(qubit_error_config, code.lattice)
    assert np.array_equal(
        df["qubit_id"], np.arange(code.n_data_qubits + code.n_stabgens)
    )
    assert np.array_equal(
        df["qubit_type"], ["data"] * code.n_data_qubits + ["stab"] * code.n_stabgens
    )
    for key, val in qubit_error_config[0].items():
        if val[0] == "constant":
            assert np.array_equal(
                df[key], [val[1]] * (code.n_data_qubits + code.n_stabgens)
            )
        elif val[0] == "gaussian":
            assert (val[1] - 3 * val[2]) < np.mean(df[key]) < (val[1] + 3 * val[2])


@pt.mark.parametrize(
    "gate_error_config",
    [
        (
            {
                "CX": [("constant", "XY", 0.1), ("gaussian", "ZI", 0.2, 0.05)],
                "H": [("gaussian", "Z", 0.5, 0.2)],
            },
            "dummy.csv",
        ),
        (
            {
                "CZ": [("constant", "XZ", 0.1), ("constant", "IY", 0.2, 0.05)],
                "H": [("constant", "Z", 0.5, 0.2)],
            },
            "dummy.csv",
        ),
        (
            {
                "CZ": [("constant", "XX", 0.1), ("constant", "ZZ", 0.2, 0.05)],
                "R": [
                    ("constant", "X", 0.1),
                    ("constant", "Y", 0.1),
                    ("constant", "Z", 0.1),
                ],
                "M": [("constant", "X", 0.1)],
            },
            "dummy.csv",
        ),
    ],
)
@pt.mark.parametrize(
    "code",
    [
        LatticeCode.make_rotated_planar(size=3, n_rounds=10),
        LatticeCode.make_planar(size=3, n_rounds=10),
        LatticeCode.make_repetition(size=9, n_rounds=3),
        LatticeCode.make_five_qubit(n_rounds=10),
        LatticeCode.make_shor(n_rounds=6),
    ],
)
def test_assimilate_gate_errors_and_no_csv(gate_error_config: tuple[dict, str], code):
    df = assimilate_gate_errors(gate_error_config, code.lattice)
    index = 0
    for gate, val in gate_error_config[0].items():
        gate_subframe = df.loc[df.gate == gate]
        for confs in val:
            if confs[0] == "constant":
                assert gate_subframe.loc[index]["induced_errors"] == [confs[1]]
                assert gate_subframe.loc[index]["probs"] == [confs[2]]
                if gate in {"CX", "CZ"}:
                    assert (
                        len(gate_subframe.iloc[index]["on_qubits"])
                        == sum(
                            [1 for stab in code.lattice.stabgens for edge in stab.edges]
                        )
                        * 2
                    )
                elif gate in {"H", "R", "M"}:
                    assert len(gate_subframe.loc[index]["on_qubits"]) == code.n_stabgens

                index += 1

            elif confs[0] == "gaussian":
                if gate in {"CX", "CZ"}:
                    num_rows = sum(
                        [1 for stab in code.lattice.stabgens for edge in stab.edges]
                    )
                    assert np.array_equal(
                        np.full(num_rows, confs[1]),
                        gate_subframe.loc[index:num_rows]["induced_errors"],
                    )
                    assert (
                        confs[2] - 3 * confs[3]
                        < np.mean(
                            [i[0] for i in gate_subframe.loc[index:num_rows]["probs"]]
                        )
                        < confs[2] + 3 * confs[3]
                    )
                    index = num_rows + index
                elif gate in {"H", "R", "M"}:
                    num_rows = code.n_stabgens
                    assert np.array_equal(
                        np.full(num_rows, confs[1]),
                        [
                            gate_subframe.loc[index + i]["induced_errors"][0]
                            for i in range(num_rows)
                        ],
                    )
                    assert (
                        confs[2] - 2 * confs[3]
                        < np.mean(
                            [
                                gate_subframe.loc[index + i]["probs"][0]
                                for i in range(num_rows)
                            ]
                        )
                        < confs[2] + 2 * confs[3]
                    )
                    index = num_rows + index


class TestErrorDataUI:
    @pt.mark.parametrize(
        "code",
        [
            LatticeCode.make_rotated_planar(size=3, n_rounds=10),
            LatticeCode.make_planar(size=3, n_rounds=10),
            LatticeCode.make_repetition(size=9, n_rounds=3),
            LatticeCode.make_five_qubit(n_rounds=10),
            LatticeCode.make_shor(n_rounds=6),
        ],
    )
    @pt.mark.parametrize(
        "gate_error_config",
        [
            (
                {
                    "CX": [("constant", "XY", 0.1), ("gaussian", "ZI", 0.2, 0.05)],
                    "H": [("gaussian", "Z", 0.5, 0.2)],
                },
                "dummy.csv",
            ),
            (
                {
                    "CZ": [("constant", "XZ", 0.1), ("constant", "IY", 0.2, 0.05)],
                    "H": [("constant", "Z", 0.5, 0.2)],
                },
                "dummy.csv",
            ),
            (
                {
                    "CZ": [("constant", "XX", 0.1), ("constant", "ZZ", 0.2, 0.05)],
                    "R": [
                        ("constant", "X", 0.1),
                        ("constant", "Y", 0.1),
                        ("constant", "Z", 0.1),
                    ],
                    "M": [("constant", "X", 0.1)],
                },
                "dummy.csv",
            ),
        ],
    )
    @pt.mark.parametrize(
        "qubit_error_config",
        [
            tuple(
                (
                    {
                        "X": ("constant", 0.01),
                        "Y": ("constant", 0.1),
                        "Z": ("constant", 0.2),
                    },
                    "dummy.csv",
                )
            ),
            tuple(
                (
                    {
                        "erasure": ("constant", 0.01),
                        "measurement": ("constant", 0.1),
                        "Z": ("constant", 0.2),
                    },
                    "dummy.csv",
                )
            ),
            tuple(
                (
                    {
                        "erasure": ("gaussian", 0.1, 0.1),
                        "measurement": ("gaussian", 0.1, 0.09),
                        "Z": ("gaussian", 0.2, 0.05),
                    },
                    "dummy.csv",
                )
            ),
        ],
    )
    def test_from_lattice_no_csv(
        self,
        code: LatticeCode,
        gate_error_config: tuple[dict, str],
        qubit_error_config: tuple[dict, str],
    ):
        errs = ErrorData.from_lattice(
            lattice=code.lattice,
            qubit_error_config=qubit_error_config,
            gate_error_config=gate_error_config,
        )

        # assert qubit errors
        df = errs.qubit_errors
        assert np.array_equal(
            df["qubit_id"], np.arange(code.n_data_qubits + code.n_stabgens)
        )
        assert np.array_equal(
            df["qubit_type"], ["data"] * code.n_data_qubits + ["stab"] * code.n_stabgens
        )
        for key, val in qubit_error_config[0].items():
            if val[0] == "constant":
                assert np.array_equal(
                    df[key], [val[1]] * (code.n_data_qubits + code.n_stabgens)
                )
            elif val[0] == "gaussian":
                assert (val[1] - 3 * val[2]) < np.mean(df[key]) < (val[1] + 3 * val[2])

        # assert gate errors
        df = errs.gate_errors
        index = 0
        for gate, val in gate_error_config[0].items():
            gate_subframe = df.loc[df.gate == gate]
            for confs in val:
                if confs[0] == "constant":
                    assert gate_subframe.loc[index]["induced_errors"] == [confs[1]]
                    assert gate_subframe.loc[index]["probs"] == [confs[2]]
                    if gate in {"CX", "CZ"}:
                        assert (
                            len(gate_subframe.iloc[index]["on_qubits"])
                            == sum(
                                [
                                    1
                                    for stab in code.lattice.stabgens
                                    for edge in stab.edges
                                ]
                            )
                            * 2
                        )
                    elif gate in {"H", "R", "M"}:
                        assert (
                            len(gate_subframe.loc[index]["on_qubits"])
                            == code.n_stabgens
                        )

                    index += 1

                elif confs[0] == "gaussian":
                    if gate in {"CX", "CZ"}:
                        num_rows = sum(
                            [1 for stab in code.lattice.stabgens for edge in stab.edges]
                        )
                        assert np.array_equal(
                            np.full(num_rows, confs[1]),
                            gate_subframe.loc[index:num_rows]["induced_errors"],
                        )
                        assert (
                            confs[2] - 3 * confs[3]
                            < np.mean(
                                [
                                    i[0]
                                    for i in gate_subframe.loc[index:num_rows]["probs"]
                                ]
                            )
                            < confs[2] + 3 * confs[3]
                        )
                        index = num_rows + index
                    elif gate in {"H", "R", "M"}:
                        num_rows = code.n_stabgens
                        assert np.array_equal(
                            np.full(num_rows, confs[1]),
                            [
                                gate_subframe.loc[index + i]["induced_errors"][0]
                                for i in range(num_rows)
                            ],
                        )
                        # the array is compiled in weird way oin 293-294.
                        # using the same syntax as line
                        # 278 throws an error for some reason
                        assert (
                            confs[2] - 3 * confs[3]
                            < np.mean(
                                [
                                    gate_subframe.loc[index + i]["probs"][0]
                                    for i in range(num_rows)
                                ]
                            )
                            < confs[2] + 3 * confs[3]
                        )
                        index = num_rows + index

    @pt.mark.skip()
    def test_from_csv(self):
        pass

    @pt.mark.parametrize(
        "code",
        [
            LatticeCode.make_rotated_planar(size=3, n_rounds=10),
            LatticeCode.make_planar(size=3, n_rounds=10),
            LatticeCode.make_repetition(size=9, n_rounds=3),
            LatticeCode.make_five_qubit(n_rounds=10),
            LatticeCode.make_shor(n_rounds=6),
        ],
    )
    @pt.mark.parametrize("error_name", ["measurement", "erasure"])
    def test_add_qubit_error(self, code, error_name):
        rng = np.random.default_rng()
        for size in range(
            9
        ):  # stop at 8 cus it is the maximum of ext indices, the 5 qubit code
            errs = ErrorData.from_lattice(code.lattice)  # make new instance every time
            qubit_id = rng.choice(
                range(code.n_data_qubits + code.n_stabgens), size=size, replace=False
            )
            zero_qubits = np.setdiff1d(
                np.arange(code.n_data_qubits + code.n_stabgens), qubit_id
            )
            probs = rng.random(size=size)
            errs.add_qubit_error(
                qubit_id=list(qubit_id), error_name=error_name, probs=probs
            )

            assert np.array_equal(errs.qubit_errors[error_name][qubit_id], probs)
            assert np.array_equal(
                errs.qubit_errors[error_name][zero_qubits], np.zeros(len(zero_qubits))
            )

    @pt.mark.parametrize(
        "code",
        [
            LatticeCode.make_rotated_planar(size=3, n_rounds=10),
            LatticeCode.make_planar(size=3, n_rounds=10),
            LatticeCode.make_repetition(size=9, n_rounds=3),
            LatticeCode.make_five_qubit(n_rounds=10),
            LatticeCode.make_shor(n_rounds=6),
        ],
    )
    @pt.mark.parametrize(
        "error_name, err_msg",
        [
            ("X", "X errors already in dataframe. Use update_qubit_error() instead"),
            ("Y", "Y errors already in dataframe. Use update_qubit_error() instead"),
            ("Z", "Z errors already in dataframe. Use update_qubit_error() instead"),
            ("fabrication", "Use the function update_fab_error() instead!"),
        ],
    )
    def test_add_qubit_error_fail(self, code, error_name, err_msg):
        rng = np.random.default_rng()
        errs = ErrorData.from_lattice(code.lattice)  # make new instance every time
        qubit_id = rng.choice(
            range(code.n_data_qubits + code.n_stabgens), size=3, replace=False
        )
        probs = rng.random(size=3)
        with pt.raises(ValueError) as exc_info:
            errs.add_qubit_error(
                qubit_id=list(qubit_id), error_name=error_name, probs=probs
            )
        assert str(exc_info.value) == err_msg

    @pt.mark.parametrize(
        "code",
        [
            LatticeCode.make_rotated_planar(size=3, n_rounds=10),
            LatticeCode.make_planar(size=3, n_rounds=10),
            LatticeCode.make_repetition(size=9, n_rounds=3),
            LatticeCode.make_five_qubit(n_rounds=10),
            LatticeCode.make_shor(n_rounds=6),
        ],
    )
    @pt.mark.parametrize("error_name", ["X", "Y", "Z"])
    def test_update_qubit_error(self, code, error_name):
        rng = np.random.default_rng()
        for size in range(
            9
        ):  # stop at 8 cus it is the maximum of ext indices, the 5 qubit code
            errs = ErrorData.from_lattice(code.lattice)  # make new instance every time
            qubit_id = rng.choice(
                range(code.n_data_qubits + code.n_stabgens), size=size, replace=False
            )
            other_qubits = np.setdiff1d(
                np.arange(code.n_data_qubits + code.n_stabgens), qubit_id
            )
            probs = rng.random(size=size)
            errs.update_qubit_error(
                qubit_id=list(qubit_id), error_name=error_name, probs=probs
            )

            assert np.array_equal(errs.qubit_errors[error_name][qubit_id], probs)
            assert np.array_equal(
                errs.qubit_errors[error_name][other_qubits],
                np.full(len(other_qubits), 0.1),
            )

    @pt.mark.parametrize(
        "code",
        [
            LatticeCode.make_rotated_planar(size=3, n_rounds=10),
            LatticeCode.make_planar(size=3, n_rounds=10),
            LatticeCode.make_repetition(size=9, n_rounds=3),
            LatticeCode.make_five_qubit(n_rounds=10),
            LatticeCode.make_shor(n_rounds=6),
        ],
    )
    @pt.mark.parametrize(
        "error_name, err_type, err_msg",
        [
            (
                "measurement",
                ValueError,
                "measurement is not an existing column in the dataframe."
                "Use the function add_qubit_error() instead",
            ),
            (
                "erasure",
                ValueError,
                "erasure is not an existing column in the dataframe."
                "Use the function add_qubit_error() instead",
            ),
            ("fabrication", ValueError, "Use the function update_fab_error() instead!"),
        ],
    )
    def test_update_qubit_error_fail_existing_columns(
        self, code, error_name, err_type, err_msg
    ):
        rng = np.random.default_rng()
        errs = ErrorData.from_lattice(code.lattice)  # make new instance every time
        qubit_id = rng.choice(
            range(code.n_data_qubits + code.n_stabgens), size=3, replace=False
        )
        probs = rng.random(size=3)
        with pt.raises(err_type) as exc_info:
            errs.update_qubit_error(
                qubit_id=list(qubit_id), error_name=error_name, probs=probs
            )
        assert str(exc_info.value) == err_msg

    @pt.mark.parametrize(
        "code",
        [
            LatticeCode.make_rotated_planar(size=3, n_rounds=10),
            LatticeCode.make_planar(size=3, n_rounds=10),
            LatticeCode.make_repetition(size=9, n_rounds=3),
            LatticeCode.make_five_qubit(n_rounds=10),
            LatticeCode.make_shor(n_rounds=6),
        ],
    )
    @pt.mark.parametrize("error_name", ["X", "Y", "Z"])
    @pt.mark.parametrize("qubit_id", [[0], [1], [0, 1], [1, 3], [6, 7]])
    @pt.mark.parametrize("probs", [[0.1, 0.2, 0.3], [0.1] * 4, [0.2] * 5])
    def test_update_qubit_error_fail_mismatched_arr_lengths(
        self, code, error_name, qubit_id, probs
    ):
        np.random.default_rng()
        errs = ErrorData.from_lattice(code.lattice)  # make new instance every time
        with pt.raises(ValueError) as exc_info:
            errs.update_qubit_error(
                qubit_id=list(qubit_id), error_name=error_name, probs=probs
            )
        assert str(exc_info.value) == "len of qubit ids and probs must be the same!"

    @pt.mark.skip
    def test_add_gate_error(self):
        pass

    @pt.mark.skip
    def test_update_gate_error(self):
        pass

    @pt.mark.skip
    def test_update(self):
        pass

    @pt.mark.skip
    def test_update_from_csv(self):
        pass

    @pt.mark.skip
    def test__map_qubit_errors_to_IR(self):
        pass

    @pt.mark.skip
    def test__map_gate_errors_to_IR(self):
        pass
