# Copyright 2023, It'sQ GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
import pytest as pt

from plaquette.codes import LatticeCode
from plaquette.frontend import (
    CircuitConfig,
    CodeConfig,
    DecoderConfig,
    DeviceConfig,
    ExperimentConfig,
    GateErrorsConfig,
    QubitErrorsConfig,
    _GateErrorMetadata,
    _QubitErrorMetadata,
)


class TestCodeConfig:
    """Class to test the class `plaquette.frontend.config.CodeConfig`."""

    @pt.mark.parametrize(
        "code_dict, output_obj",
        [
            (
                dict(name="RotatedPlanarCode", size=3, rounds=4),
                CodeConfig(name="RotatedPlanarCode", size=3, rounds=4),
            ),
            (
                dict(name="PlanarCode", size=5, rounds=10),
                CodeConfig(name="PlanarCode", size=5, rounds=10),
            ),
            (
                dict(name="RepetitionCode", size=3, rounds=10),
                CodeConfig(name="RepetitionCode", size=3, rounds=10),
            ),
            (
                dict(name="FiveQubitCode", rounds=100),
                CodeConfig(name="FiveQubitCode", rounds=100),
            ),
            (
                dict(name="ShorCode", rounds=1),
                CodeConfig(name="ShorCode", rounds=1),
            ),
            (
                dict(name="ShorCode"),
                CodeConfig(name="ShorCode", rounds=1),
            ),
        ],
    )
    def test_from_dict(
        self, code_dict: dict[str, str | int], output_obj: CodeConfig
    ) -> None:
        assert CodeConfig.from_dict(code_dict) == output_obj

    @pt.mark.parametrize(
        "output_dict, input_obj",
        [
            (
                dict(name="RotatedPlanarCode", size=3, rounds=4),
                CodeConfig(name="RotatedPlanarCode", size=3, rounds=4),
            ),
            (
                dict(name="PlanarCode", size=5, rounds=10),
                CodeConfig(name="PlanarCode", size=5, rounds=10),
            ),
            (
                dict(name="RepetitionCode", rounds=10, size=3),
                CodeConfig(name="RepetitionCode", size=3, rounds=10),
            ),
            (
                dict(name="FiveQubitCode", rounds=100, size=-1),
                CodeConfig(name="FiveQubitCode", rounds=100),
            ),
            (
                dict(name="ShorCode", rounds=1, size=-1),
                CodeConfig(name="ShorCode", rounds=1),
            ),
        ],
    )
    def test_as_dict(self, output_dict: dict, input_obj: CodeConfig) -> None:
        assert input_obj.as_dict() == output_dict

    @pt.mark.parametrize(
        "input_dict, err_msg",
        [
            (
                dict(name="RotatedPlanarCode", size=3.3, rounds=4),
                "For the given code, size and round must be positive integers!",
            ),
            (
                dict(name="PlanarCode", size=-1, rounds=4),
                "For the given code, size and round must be positive integers!",
            ),
            (
                dict(name="RepetitionCode", size=-1, rounds=-4),
                "For the given code, size and round must be positive integers!",
            ),
            (
                dict(name="PlanerCode", size=1, rounds=4),
                "PlanerCode is not yet supported!",
            ),
            (
                dict(name="ColorCode", size=3, rounds=3),
                "ColorCode is not yet supported!",
            ),
        ],
    )
    def test_from_dict_errors(self, input_dict: dict, err_msg: str) -> None:
        with pt.raises(ValueError) as err:
            CodeConfig.from_dict(input_dict)
        assert str(err.value) == err_msg

    @pt.mark.parametrize(
        "input_conf, output_obj",
        [
            (
                CodeConfig(name="FiveQubitCode", rounds=1),
                LatticeCode.make_five_qubit(n_rounds=1),
            ),
            (CodeConfig(name="ShorCode", rounds=2), LatticeCode.make_shor(n_rounds=2)),
            (
                CodeConfig(name="RepetitionCode", size=3, rounds=3),
                LatticeCode.make_repetition(size=3, n_rounds=3),
            ),
            (
                CodeConfig(name="RepetitionCode", size=9, rounds=1),
                LatticeCode.make_repetition(size=9, n_rounds=1),
            ),
            (
                CodeConfig(name="RotatedPlanarCode", size=3, rounds=4),
                LatticeCode.make_rotated_planar(size=3, n_rounds=4),
            ),
            (
                CodeConfig(name="RotatedPlanarCode", size=17, rounds=1),
                LatticeCode.make_rotated_planar(size=17, n_rounds=1),
            ),
            (
                CodeConfig(name="PlanarCode", size=5, rounds=10),
                LatticeCode.make_planar(size=5, n_rounds=10),
            ),
            (
                CodeConfig(name="PlanarCode", size=11, rounds=10),
                LatticeCode.make_planar(size=11, n_rounds=10),
            ),
        ],
    )
    def test_instantiate(self, input_conf: CodeConfig, output_obj: LatticeCode):
        assert input_conf.instantiate() == output_obj

    @pt.mark.parametrize(
        "init_object, kwargs, expected",
        [
            (
                CodeConfig(name="FiveQubitCode", rounds=1),
                {"rounds": 10},
                CodeConfig(name="FiveQubitCode", rounds=10),
            ),
            (
                CodeConfig(name="ShorCode", rounds=2),
                dict(name="FiveQubitCode", rounds=7),
                CodeConfig(name="FiveQubitCode", rounds=7),
            ),
            (
                CodeConfig(name="RepetitionCode", size=3, rounds=3),
                dict(size=10, rounds=1),
                CodeConfig(name="RepetitionCode", size=10, rounds=1),
            ),
            (
                CodeConfig(name="RepetitionCode", size=9, rounds=1),
                dict(name="RotatedPlanarCode", rounds=10),
                CodeConfig(name="RotatedPlanarCode", size=9, rounds=10),
            ),
            (
                CodeConfig(name="RotatedPlanarCode", size=3, rounds=4),
                dict(name="PlanarCode"),
                CodeConfig(name="PlanarCode", size=3, rounds=4),
            ),
            (
                CodeConfig(name="RotatedPlanarCode", size=17, rounds=1),
                dict(name="FiveQubitCode"),
                CodeConfig(name="FiveQubitCode", size=-1, rounds=1),
            ),
            (
                CodeConfig(name="RotatedPlanarCode", size=17, rounds=1),
                dict(name="FiveQubitCode"),
                CodeConfig(name="FiveQubitCode", size=-1, rounds=1),
            ),
            # (
            #     CodeConfig(name="PlanarCode", size=11, rounds=10),
            #     LatticeCode.make_planar(size=11, n_rounds=10),
            # ),
        ],
    )
    def test_update(self, init_object: CodeConfig, kwargs: dict, expected: CodeConfig):
        init_object.update(**kwargs)
        assert init_object == expected

    @pt.mark.parametrize(
        "update_dict, err_msg",
        [
            (
                dict(name="RotatedPlanarCode", size=3.3, rounds=4),
                "For the given code, size and round must be positive integers!",
            ),
            (
                dict(name="PlanarCode", size=-1, rounds=4),
                "For the given code, size and round must be positive integers!",
            ),
            (
                dict(name="RepetitionCode", size=-1, rounds=-4),
                "For the given code, size and round must be positive integers!",
            ),
            (
                dict(name="PlanerCode", size=1, rounds=4),
                "PlanerCode is not yet supported!",
            ),
            (
                dict(name="ColorCode", size=3, rounds=3),
                "ColorCode is not yet supported!",
            ),
        ],
    )
    def test_update_fail(self, update_dict, err_msg):
        code_conf = CodeConfig(name="FiveQubitCode")
        with pt.raises(ValueError) as exc_info:
            code_conf.update(**update_dict)

        assert str(exc_info.value) == err_msg


class TestCircuitConfig:
    """Class to load CircuitConfiguration."""

    @pt.mark.parametrize(
        "input_dict, output_obj",
        [
            (
                dict(),
                CircuitConfig(
                    circuit_path="circuit.txt", has_errors=False, circuit_provided=False
                ),
            ),
            (
                dict(
                    circuit_provided=True,
                    has_errors=True,
                    circuit_path="tests/unittests/frontend/rep_code_5.txt",
                ),
                CircuitConfig(
                    circuit_provided=True,
                    has_errors=True,
                    circuit_path="tests/unittests/frontend/rep_code_5.txt",
                ),
            ),
            (
                dict(
                    circuit_provided=True,
                    circuit_path="tests/unittests/frontend/rep_code_5.txt",
                ),
                CircuitConfig(
                    circuit_provided=True,
                    has_errors=False,
                    circuit_path="tests/unittests/frontend/rep_code_5.txt",
                ),
            ),
        ],
    )
    def test_from_dict(self, input_dict, output_obj):
        assert CircuitConfig.from_dict(input_dict) == output_obj

    @pt.mark.parametrize(
        "input_conf, exc, err_msg",
        [
            (
                CircuitConfig(
                    circuit_path="tests/unittests/frontend/rep_code_5.txt",
                    circuit_provided=True,
                    has_errors=False,
                ),
                NotImplementedError,
                "Cannot currently insert errors onto a error free circuit",
            )
        ],
    )
    def test_instantiate_err(
        self, input_conf: CircuitConfig, exc: Exception, err_msg: str
    ):
        with pt.raises(exc) as exc_info:
            input_conf.instantiate()
        assert str(exc_info.value) == err_msg

    @pt.mark.parametrize(
        "output_dict, input_obj",
        [
            (
                dict(
                    circuit_path="circuit.txt", has_errors=False, circuit_provided=False
                ),
                CircuitConfig(),
            ),
            (
                dict(
                    circuit_provided=True,
                    has_errors=True,
                    circuit_path="tests/unittests/frontend/rep_code_5.txt",
                ),
                CircuitConfig(
                    circuit_provided=True,
                    has_errors=True,
                    circuit_path="tests/unittests/frontend/rep_code_5.txt",
                ),
            ),
            (
                dict(
                    circuit_provided=True,
                    has_errors=False,
                    circuit_path="tests/unittests/frontend/rep_code_5.txt",
                ),
                CircuitConfig(
                    circuit_provided=True,
                    circuit_path="tests/unittests/frontend/rep_code_5.txt",
                ),
            ),
        ],
    )
    def test_as_dict(self, output_dict, input_obj):
        assert input_obj.as_dict() == output_dict


class TestDecoderConfig:
    """Class to test the loading the Decoder Configuration."""

    @pt.mark.parametrize(
        "input_dict, output_obj",
        [
            (
                dict(name="PyMatchingDecoder", weighted=True),
                DecoderConfig(name="PyMatchingDecoder", weighted=True),
            ),
            (
                dict(name="UnionFindDecoder", weighted=True),
                DecoderConfig(name="UnionFindDecoder", weighted=True),
            ),
            (
                dict(name="UnionFindNoWeights", weighted=False),
                DecoderConfig(name="UnionFindNoWeights", weighted=False),
            ),
            (
                dict(name="FusionBlossomDecoder", weighted=False),
                DecoderConfig(name="FusionBlossomDecoder", weighted=False),
            ),
        ],
    )
    def test_from_dict(self, input_dict: dict, output_obj: DecoderConfig) -> None:
        assert DecoderConfig.from_dict(input_dict) == output_obj

    @pt.mark.parametrize(
        "output_dict, input_obj",
        [
            (
                dict(name="PyMatchingDecoder", weighted=True),
                DecoderConfig(name="PyMatchingDecoder", weighted=True),
            ),
            (
                dict(name="UnionFindDecoder", weighted=True),
                DecoderConfig(name="UnionFindDecoder", weighted=True),
            ),
            (
                dict(name="UnionFindNoWeights", weighted=False),
                DecoderConfig(name="UnionFindNoWeights", weighted=False),
            ),
        ],
    )
    def test_as_dict(self, output_dict: dict, input_obj: DecoderConfig) -> None:
        assert input_obj.as_dict() == output_dict

    @pt.mark.parametrize(
        "input_dict, err_msg",
        [
            (
                dict(name="UnionFind1", weighted=False),
                "UnionFind1 is not yet supported!",
            ),
            (dict(name="Matching", weighted=True), "Matching is not yet supported!"),
            (
                dict(name="UnionFindouble", weighted=True),
                "UnionFindouble is not yet supported!",
            ),
        ],
    )
    def test_from_dict_failures(self, input_dict: dict, err_msg: str) -> None:
        with pt.raises(ValueError) as err:
            DecoderConfig.from_dict(input_dict)
        assert str(err.value) == err_msg


class TestQubitErrorMetadata:
    @pt.mark.parametrize(
        "input_dict, output_tuple",
        [
            (dict(distribution="user"), (True, "user", [])),
            (dict(distribution="user", enabled=False), (False, "user", [])),
            (
                dict(enabled=False, distribution="constant", params=[0.1]),
                (False, "constant", [0.1]),
            ),
            (dict(distribution="constant", params=[0.17]), (True, "constant", [0.17])),
            (
                dict(enabled=True, distribution="gaussian", params=[0.1, 0.01]),
                (True, "gaussian", [0.1, 0.01]),
            ),
            (
                dict(enabled=False, distribution="gaussian", params=[0.1, 0.05]),
                (False, "gaussian", [0.1, 0.05]),
            ),
            (
                dict(distribution="gaussian", params=[0.2, 0.02]),
                (True, "gaussian", [0.2, 0.02]),
            ),
        ],
    )
    def test_from_dict(self, input_dict, output_tuple):
        created_obj = _QubitErrorMetadata.from_dict(input_dict)
        assert created_obj.enabled == output_tuple[0]
        assert created_obj.distribution == output_tuple[1]
        assert created_obj.params == output_tuple[2]

    @pt.mark.parametrize(
        "input_dict, err_msg",
        [
            (
                dict(distribution="user", params=[0.01]),
                "Distribution is set to user, but params are provided."
                "Set distribution to 'gaussian' or 'constant' or don't provide params.",
            ),
            (
                dict(distribution="gaussian", params=[0.01]),
                "The length of params is wrong, please provide a list of length 2",
            ),
            (
                dict(distribution="constant", params=[]),
                "The length of params is wrong, please provide a list of length 1",
            ),
        ],
    )
    def test_init_failures(self, input_dict, err_msg):
        with pt.raises(ValueError) as exc_info:
            _QubitErrorMetadata.from_dict(input_dict)

        assert str(exc_info.value) == err_msg

    @pt.mark.parametrize(
        "input_obj, ret_val",
        [
            (_QubitErrorMetadata.make_empty(), True),
            (_QubitErrorMetadata(distribution="user"), False),
            (_QubitErrorMetadata(distribution="constant", params=[0.1]), False),
            (_QubitErrorMetadata(distribution="", params=[], enabled=False), True),
            (_QubitErrorMetadata(distribution="", params=[]), True),
        ],
    )
    def test_is_empty(self, input_obj, ret_val):
        assert input_obj.is_empty() == ret_val


class TestGateErrorMetadata:
    @pt.mark.parametrize(
        "input_dict, output_tuple",
        [
            (dict(distribution="user"), ("user", [], [], True)),
            (dict(distribution="user", enabled=False), ("user", [], [], False)),
            (
                dict(
                    distribution=["constant", "constant"],
                    induced_errors=["XZ", "IY"],
                    params=[[0.1], [0.2]],
                    enabled=False,
                ),
                (["constant", "constant"], ["XZ", "IY"], [[0.1], [0.2]], False),
            ),
            (
                dict(
                    distribution=["constant", "constant"],
                    induced_errors=["X", "Y"],
                    params=[[0.1], [0.2]],
                ),
                (["constant", "constant"], ["X", "Y"], [[0.1], [0.2]], True),
            ),
            (
                dict(
                    distribution=["gaussian", "constant"],
                    induced_errors=["XZ", "IY"],
                    params=[[0.1, 0.01], [0.2]],
                    enabled=False,
                ),
                (["gaussian", "constant"], ["XZ", "IY"], [[0.1, 0.01], [0.2]], False),
            ),
            (
                dict(
                    distribution=["constant", "gaussian"],
                    induced_errors=["X", "Y"],
                    params=[[0.1], [0.2, 0.05]],
                ),
                (["constant", "gaussian"], ["X", "Y"], [[0.1], [0.2, 0.05]], True),
            ),
        ],
    )
    def test_from_dict(self, input_dict, output_tuple):
        created_obj = _GateErrorMetadata.from_dict(input_dict)
        assert created_obj.distribution == output_tuple[0]
        assert created_obj.induced_errors == output_tuple[1]
        assert created_obj.params == output_tuple[2]
        assert created_obj.enabled == output_tuple[3]

    @pt.mark.parametrize(
        "input_dict, exc_type, err_msg",
        [
            (
                dict(
                    distribution=["constant", "constant"],
                    induced_errors=["X", "Y"],
                    params=[[0.1], [0.2, 0.05]],
                ),
                ValueError,
                "Incorrect length of params, provide a list of length 1",
            ),
            (
                dict(
                    distribution=["gaussian", "constant"],
                    induced_errors=["XX", "YI"],
                    params=[[0.1], [0.2, 0.05]],
                ),
                ValueError,
                "Incorrect length of params, provide a list of length 2",
            ),
            (
                dict(
                    distribution=["user", "constant"],
                    induced_errors=["X", "Y"],
                    params=[[0.1], [0.2, 0.05]],
                ),
                ValueError,
                "Only 'constant' or 'gaussian' are valid distribution",
            ),
            (
                dict(
                    distribution=["constant", "constant"],
                    induced_errors=["X", "Y"],
                    params=[[0.1]],
                ),
                ValueError,
                "Induced errors and params must be of the same length!",
            ),
            (
                dict(
                    distribution=["constant", "constant"],
                    induced_errors=["I", "Y"],
                    params=[[0.1], [0.2]],
                ),
                ValueError,
                "The given induced error I is invalid",
            ),
            (
                dict(
                    distribution="user",
                    induced_errors=["I", "Y"],
                    params=[[0.1], [0.2]],
                ),
                ValueError,
                "Incorrect values of induced_errors or params!",
            ),
        ],
    )
    def test_init_failures(self, input_dict, exc_type, err_msg):
        with pt.raises(exc_type) as exc_info:
            _GateErrorMetadata.from_dict(input_dict)
        assert str(exc_info.value) == err_msg

    @pt.mark.parametrize(
        "input_obj, ret_val",
        [
            (_GateErrorMetadata.make_empty(), True),
            (_GateErrorMetadata(distribution=""), True),
            (_GateErrorMetadata(distribution=[]), True),
            (_GateErrorMetadata(distribution="user"), False),
            (
                _GateErrorMetadata(
                    distribution=["constant"], induced_errors=["X"], params=[[0.1]]
                ),
                False,
            ),
        ],
    )
    def test_is_empty(self, input_obj, ret_val):
        assert input_obj.is_empty() == ret_val


@pt.fixture
def qubit_error_config():
    return {
        "data_path": "tests/unittests/frontend/SpEM.csv",
        "X": {"distribution": "user"},
        "Y": {"distribution": "constant", "params": [0.1]},
        "Z": {"distribution": "gaussian", "params": [0.5, 0.2]},
        "erasure": {"enabled": False, "distribution": "user"},
        "measurement": {
            "enabled": False,
            "distribution": "gaussian",
            "params": [0.1, 0.01],
        },
        "fabrication": {"enabled": False, "distribution": "user"},
    }


class TestQubitErrorsConfig:
    @pt.mark.parametrize(
        "input_dict, output_tuple",
        [
            (
                {
                    "data_path": "tests/unittests/frontend/SpEM.csv",
                    "X": {"distribution": "user"},
                    "Y": {"distribution": "constant", "params": [0.1]},
                    "Z": {"distribution": "gaussian", "params": [0.5, 0.2]},
                    "erasure": {"enabled": False, "distribution": "user"},
                    "measurement": {
                        "enabled": False,
                        "distribution": "gaussian",
                        "params": [0.1, 0.01],
                    },
                },
                (
                    "tests/unittests/frontend/SpEM.csv",
                    True,
                    True,
                    _QubitErrorMetadata(distribution="user"),
                    _QubitErrorMetadata(distribution="constant", params=[0.1]),
                    _QubitErrorMetadata(distribution="gaussian", params=[0.5, 0.2]),
                    _QubitErrorMetadata(distribution="user", enabled=False),
                    _QubitErrorMetadata(
                        distribution="gaussian", params=[0.1, 0.01], enabled=False
                    ),
                    _QubitErrorMetadata(distribution="", params=[], enabled=False),
                ),
            ),
            (
                {
                    "X": {"distribution": "user"},
                    "Y": {"distribution": "constant", "params": [0.1]},
                    "Z": {"distribution": "gaussian", "params": [0.5, 0.2]},
                    "erasure": {"enabled": False, "distribution": "user"},
                    "measurement": {
                        "enabled": False,
                        "distribution": "gaussian",
                        "params": [0.1, 0.01],
                    },
                },
                (
                    "qubit_errors.csv",
                    True,
                    False,
                    _QubitErrorMetadata(distribution="user"),
                    _QubitErrorMetadata(distribution="constant", params=[0.1]),
                    _QubitErrorMetadata(distribution="gaussian", params=[0.5, 0.2]),
                    _QubitErrorMetadata(distribution="user", enabled=False),
                    _QubitErrorMetadata(
                        distribution="gaussian", params=[0.1, 0.01], enabled=False
                    ),
                    _QubitErrorMetadata(distribution="", params=[], enabled=False),
                ),
            ),
        ],
    )
    def test_from_dict(self, input_dict, output_tuple):
        created_obj = QubitErrorsConfig.from_dict(input_dict)
        assert created_obj.data_path == output_tuple[0]
        assert created_obj.sample == output_tuple[1]
        assert created_obj.load_file == output_tuple[2]
        assert created_obj.X == output_tuple[3]
        assert created_obj.Y == output_tuple[4]
        assert created_obj.Z == output_tuple[5]
        assert created_obj.erasure == output_tuple[6]
        assert created_obj.measurement == output_tuple[7]
        assert created_obj.fabrication == output_tuple[8]

    @pt.mark.parametrize(
        "output_tuples",
        [
            (
                (
                    {
                        "X": ("user",),
                        "Y": ("constant", 0.1),
                        "Z": ("gaussian", 0.5, 0.2),
                    },
                    "tests/unittests/frontend/SpEM.csv",
                ),
                (
                    {
                        "X": ("user",),
                        "Y": ("constant", 0.1),
                        "Z": ("gaussian", 0.5, 0.2),
                        "erasure": ("user",),
                    },
                    "tests/unittests/frontend/SpEM.csv",
                ),
                (
                    {
                        "X": ("user",),
                        "erasure": ("user",),
                        "measurement": ("gaussian", 0.1, 0.01),
                    },
                    "tests/unittests/frontend/SpEM.csv",
                ),
            )
        ],
    )
    def test_get_simulated_errors(self, output_tuples, qubit_error_config):
        conf = QubitErrorsConfig.from_dict(qubit_error_config)
        assert conf.simulated_errors == output_tuples[0]
        conf.erasure.enabled = True
        assert conf.simulated_errors == output_tuples[1]
        conf.Y.enabled = False
        conf.Z.enabled = False
        conf.measurement.enabled = True
        assert conf.simulated_errors == output_tuples[2]
        conf.sample = False
        assert conf.simulated_errors is None


class TestGateErrorsConfig:
    @pt.mark.parametrize(
        "input_dict, output_tuple",
        [
            (
                {
                    "data_path": "tests/unittests/frontend/GateErrors.csv",
                    "load_file": True,
                    "CX": {"distribution": "user"},
                    "CZ": {
                        "enabled": False,
                        "distribution": ["constant", "gaussian"],
                        "induced_errors": ["XZ", "IY"],
                        "params": [[0.1], [0.2, 0.05]],
                    },
                    "H": {
                        "distribution": ["gaussian"],
                        "induced_errors": ["Z"],
                        "params": [[0.5, 0.2]],
                    },
                    "R": {"enabled": False, "distribution": "user"},
                    "M": {
                        "enabled": False,
                        "distribution": ["constant"],
                        "induced_errors": ["X"],
                        "params": [[0.1]],
                    },
                },
                (
                    "tests/unittests/frontend/GateErrors.csv",
                    False,
                    True,
                    _GateErrorMetadata(
                        distribution="user", induced_errors=[], params=[], enabled=True
                    ),
                    _GateErrorMetadata(
                        distribution=["constant", "gaussian"],
                        induced_errors=["XZ", "IY"],
                        params=[[0.1], [0.2, 0.05]],
                        enabled=False,
                    ),
                    _GateErrorMetadata(
                        distribution=["gaussian"],
                        induced_errors=["Z"],
                        params=[[0.5, 0.2]],
                        enabled=True,
                    ),
                    _GateErrorMetadata(
                        distribution="user", induced_errors=[], params=[], enabled=False
                    ),
                    _GateErrorMetadata(
                        distribution=["constant"],
                        induced_errors=["X"],
                        params=[[0.1]],
                        enabled=False,
                    ),
                    _GateErrorMetadata(
                        distribution="", induced_errors=[], params=[], enabled=False
                    ),
                ),
            ),
            (
                {
                    "sample": True,
                    "CX": {"distribution": "user"},
                    "CZ": {
                        "enabled": False,
                        "distribution": ["constant", "gaussian"],
                        "induced_errors": ["XZ", "IY"],
                        "params": [[0.1], [0.2, 0.05]],
                    },
                    "H": {
                        "distribution": ["gaussian"],
                        "induced_errors": ["Z"],
                        "params": [[0.5, 0.2]],
                    },
                    "R": {"enabled": False, "distribution": "user"},
                    "M": {
                        "enabled": False,
                        "distribution": ["constant"],
                        "induced_errors": ["X"],
                        "params": [[0.1]],
                    },
                },
                (
                    "gate_errors.csv",
                    True,
                    False,
                    _GateErrorMetadata(
                        distribution="user", induced_errors=[], params=[], enabled=True
                    ),
                    _GateErrorMetadata(
                        distribution=["constant", "gaussian"],
                        induced_errors=["XZ", "IY"],
                        params=[[0.1], [0.2, 0.05]],
                        enabled=False,
                    ),
                    _GateErrorMetadata(
                        distribution=["gaussian"],
                        induced_errors=["Z"],
                        params=[[0.5, 0.2]],
                        enabled=True,
                    ),
                    _GateErrorMetadata(
                        distribution="user", induced_errors=[], params=[], enabled=False
                    ),
                    _GateErrorMetadata(
                        distribution=["constant"],
                        induced_errors=["X"],
                        params=[[0.1]],
                        enabled=False,
                    ),
                    _GateErrorMetadata(
                        distribution="", induced_errors=[], params=[], enabled=False
                    ),
                ),
            ),
        ],
    )
    def test_from_dict(self, input_dict, output_tuple):
        created_obj = GateErrorsConfig.from_dict(input_dict)
        assert created_obj.data_path == output_tuple[0]
        assert created_obj.sample == output_tuple[1]
        assert created_obj.load_file == output_tuple[2]
        assert created_obj.CX == output_tuple[3]
        assert created_obj.CZ == output_tuple[4]
        assert created_obj.H == output_tuple[5]
        assert created_obj.R == output_tuple[6]
        assert created_obj.M == output_tuple[7]
        assert created_obj.fabrication == output_tuple[8]

    @pt.mark.parametrize(
        "output_tuple",
        [
            (
                (
                    {"CX": ("user",), "H": [("gaussian", "Z", 0.5, 0.2)]},
                    "tests/unittests/frontend/GateErrors.csv",
                ),
                (
                    {
                        "CX": ("user",),
                        "CZ": [("constant", "XZ", 0.1), ("gaussian", "IY", 0.2, 0.05)],
                        "H": [("gaussian", "Z", 0.5, 0.2)],
                    },
                    "tests/unittests/frontend/GateErrors.csv",
                ),
                (
                    {
                        "CZ": [("constant", "XZ", 0.1), ("gaussian", "IY", 0.2, 0.05)],
                        "R": ("user",),
                        "M": [("constant", "X", 0.1)],
                    },
                    "tests/unittests/frontend/GateErrors.csv",
                ),
            ),
        ],
    )
    def test_get_simulated_errors(self, output_tuple, gate_error_config):
        conf = GateErrorsConfig.from_dict(gate_error_config)
        assert conf.simulated_errors == output_tuple[0]
        conf.CZ.enabled = True
        assert conf.simulated_errors == output_tuple[1]
        conf.CX.enabled = False
        conf.H.enabled = False
        conf.R.enabled = True
        conf.M.enabled = True
        assert conf.simulated_errors == output_tuple[2]
        conf.sample = False
        assert conf.simulated_errors is None


@pt.fixture
def gate_error_config():
    return {
        "data_path": "tests/unittests/frontend/GateErrors.csv",
        "sample": True,
        "load_file": True,
        "CX": {"distribution": "user"},
        "CZ": {
            "enabled": False,
            "distribution": ["constant", "gaussian"],
            "induced_errors": ["XZ", "IY"],
            "params": [[0.1], [0.2, 0.05]],
        },
        "H": {
            "distribution": ["gaussian"],
            "induced_errors": ["Z"],
            "params": [[0.5, 0.2]],
        },
        "R": {"enabled": False, "distribution": "user"},
        "M": {
            "enabled": False,
            "distribution": ["constant"],
            "induced_errors": ["X"],
            "params": [[0.1]],
        },
        "fabrication": {"enabled": False, "distribution": "user"},
    }


class TestDeviceConfig:
    """Tests for the class the DeviceConfig."""

    @pt.mark.parametrize(
        "input_dict, output_obj",
        [
            (
                dict(name="clifford", shots=1024),
                DeviceConfig(name="clifford", shots=1024),
            ),
            (
                dict(name="clifford", shots=1),
                DeviceConfig(name="clifford", shots=1),
            ),
            (
                dict(name="stim", shots=1231245),
                DeviceConfig(name="stim", shots=1231245),
            ),
        ],
    )
    def test_from_dict(self, input_dict: dict, output_obj: DeviceConfig) -> None:
        assert DeviceConfig.from_dict(input_dict) == output_obj

    @pt.mark.parametrize(
        "output_dict, input_obj",
        [
            (
                dict(name="clifford", shots=1),
                DeviceConfig(name="clifford", shots=1),
            ),
            (
                dict(name="clifford", shots=12),
                DeviceConfig(name="clifford", shots=12),
            ),
            (
                dict(name="stim", shots=1289),
                DeviceConfig(name="stim", shots=1289),
            ),
        ],
    )
    def test_as_dict(self, input_obj: DeviceConfig, output_dict: dict) -> None:
        assert input_obj.as_dict() == output_dict

    @pt.mark.parametrize(
        "input_dict, err_msg",
        [
            (
                dict(name="clifford", shots=-1),
                "device.shots = -1 must be an integer",
            ),
            (
                dict(name="clifford", shots=13.3),
                "device.shots = 13.3 must be an integer",
            ),
            (
                dict(name="CodespaceSimulator", shots=1),
                "CodespaceSimulator is not a valid device name",
            ),
            (
                dict(name="stim-Simulator", shots=1),
                "stim-Simulator is not a valid device name",
            ),
        ],
    )
    def test_from_dict_failures(self, input_dict: dict, err_msg: str) -> None:
        with pt.raises(ValueError) as err:
            DeviceConfig.from_dict(input_dict)
        assert str(err.value) == err_msg


@pt.fixture()
def extensive_toml() -> ExperimentConfig:
    return ExperimentConfig.load_toml("tests/unittests/frontend/extensive.toml")


class TestExperimentConfig:
    @pt.mark.parametrize(
        "input_conf, output_obj",
        [
            (
                CodeConfig(name="FiveQubitCode", rounds=1),
                LatticeCode.make_five_qubit(n_rounds=1),
            ),
            (CodeConfig(name="ShorCode", rounds=2), LatticeCode.make_shor(n_rounds=2)),
            (
                CodeConfig(name="RepetitionCode", size=3, rounds=3),
                LatticeCode.make_repetition(size=3, n_rounds=3),
            ),
            (
                CodeConfig(name="RepetitionCode", size=9, rounds=1),
                LatticeCode.make_repetition(size=9, n_rounds=1),
            ),
            (
                CodeConfig(name="RotatedPlanarCode", size=3, rounds=4),
                LatticeCode.make_rotated_planar(size=3, n_rounds=4),
            ),
            (
                CodeConfig(name="RotatedPlanarCode", size=17, rounds=1),
                LatticeCode.make_rotated_planar(size=17, n_rounds=1),
            ),
            (
                CodeConfig(name="PlanarCode", size=5, rounds=10),
                LatticeCode.make_planar(size=5, n_rounds=10),
            ),
            (
                CodeConfig(name="PlanarCode", size=11, rounds=10),
                LatticeCode.make_planar(size=11, n_rounds=10),
            ),
        ],
    )
    def test_build_code(self, input_conf, output_obj, extensive_toml):
        extensive_toml.update(code_conf=input_conf)
        extensive_toml.build_code()
        assert extensive_toml.code == output_obj

    @pt.mark.skip
    def test_build_errors(self):
        pass

    @pt.mark.skip
    def test_build_circuit(self):
        pass

    @pt.mark.skip
    def test_build_decoder(self):
        pass

    @pt.mark.skip
    def test_build_simulator(self):
        pass
