# Copyright 2023, It'sQ GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Front-end configuration processing for running experiments."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from itertools import product
from pathlib import Path
from pprint import pformat
from typing import Any, ClassVar, Type, TypedDict, Union, cast

import numpy as np
import toml  # type: ignore
from jsonschema import validate
from tqdm import tqdm

import plaquette
from plaquette import Device, decoders, visualizer
from plaquette.circuit import Circuit, generator
from plaquette.codes import LatticeCode
from plaquette.codes.latticebase import CodeLattice
from plaquette.decoders.decoderbase import check_success
from plaquette.device import AbstractSimulator, MeasurementSample
from plaquette.errors import ErrorData, generate_empty_qubit_errors_csv
from plaquette.frontend import schemas


def _validate_filepath(filepath: str, load_file: bool) -> bool:
    """Validate given filepath by checking if file exists or can be created there.

    Args:
        filepath: The path to the file
        load_file: Boolean variable indicating if a file should be loaded or not

    Raises:
        NotADirectoryError: if the directory containing the file is inaccessible
        FileNotFoundError: if the given path to the file is inaccessible

    Returns:
        ``True`` if validated, else raises an Error.
    """
    fp = Path(filepath)
    if not fp.parent.exists():
        raise NotADirectoryError(
            "The given directory containing the file in inaccessible!"
        )
    if not fp.exists() and load_file:
        raise FileNotFoundError("The given file in inaccessible!")

    return True


def _validate_config(config_dict: dict) -> bool:
    """Validate the experiment config using the schema.

    Args:
        config_dict : dictionary of the experiment configuration

    Returns:
        ``True`` if validated, else raises an Error. The errors raised are according to
        :func:`~jsonschema.validate`.

    """
    schema_dict = schemas.get_exp_config_schema()
    validate(config_dict, schema_dict)
    # validate will throw a detailed error message, if not assume to pass
    return True


@dataclass(kw_only=True)
class CodeConfig:
    """Class to handle to store the configuration for Codes.

    Raises:
        ValueError: If code name not in :attr:`~CodeConfig.VALID_CODES`
        ValueError: If size or round are not positive.

    Examples:
        >>> obj = CodeConfig(name="RotatedPlanarCode", size=3)
        >>> print(obj)
        CodeConfig(name='RotatedPlanarCode', rounds=1, size=3)
    """

    name: str
    """Name of the error correcting code.

    Possible values are in :attr:`~CodeConfig.VALID_CODES`
    """
    rounds: int = field(default=1)
    """ Number of measurement rounds.

    A positive integer `>=1`. Defaults to `1`, when not provided.
    """
    size: int = field(default=-1)
    """Size of the error correcting code.

    Defaults `-1` when size is not relevant to the code.
    For codes which require a size, it must be a positive integer `>=1`.
    """
    VALID_CODES: ClassVar[tuple[str, ...]] = (
        "RotatedPlanarCode",
        "PlanarCode",
        "ToricCode",
        "RepetitionCode",
        "FiveQubitCode",
        "ShorCode",
    )
    """The valid codes that are available in :mod:`plaquette`."""

    def __post_init__(self):
        """Checks initialized values.

        :meta private:
        """
        if self.name not in self.VALID_CODES:
            raise ValueError(f"{self.name} is not yet supported!")

        if self.name in self.VALID_CODES[4:]:
            self.size = -1
        elif (
            (self.size < 0 or self.rounds < 0)
            or not (isinstance(self.size, int) and isinstance(self.rounds, int))
            and self.name in self.VALID_CODES[:4]
        ):
            raise ValueError(
                "For the given code, size and round must be positive integers!"
            )
        else:
            self.size = cast(int, self.size)

    @classmethod
    def from_dict(cls, config_dict: dict) -> CodeConfig:
        """Instantiate :class:`~.CodeConfig` from config dictionary.

        For codes where size is irrelevant, like the five-qubit code, it defaults to -1.

        Args:
            config_dict: Dict containing code configurations

        Returns:
            A :class:`.CodeConfig` object.

        Raises:
            ValueError: If code name not in :attr:`.CodeConfig.VALID_CODES`
            ValueError: If size or round are not positive.

        Examples:
            >>> code_conf = CodeConfig.from_dict(dict(name="FiveQubitCode", rounds=1))
            >>> print(code_conf)
            CodeConfig(name='FiveQubitCode', rounds=1, size=-1)
        """
        config_dict.setdefault("rounds", 1)
        config_dict.setdefault("size", -1)

        return cls(
            name=cast(str, config_dict.get("name")),
            rounds=cast(int, config_dict.get("rounds")),
            size=cast(int, config_dict.get("size")),
        )

    def as_dict(self) -> dict[str, Union[str, int]]:
        """Return class attributes as dictionary.

        Returns:
            A dictionary of the configuration for the code.
        """
        return asdict(self)

    def update(self, **kwargs):
        """Update class attributes with given ``**kwargs``.

        Args:
            kwargs: Any class attribute.

        Returns:
            None, updates the config in-place.

        Raises:
            ValueError: If code name not one of the supported ones.
            ValueError: If size or round are not positive.

        Examples:
            >>> code_conf = CodeConfig.from_dict(dict(name="FiveQubitCode", rounds=1))
            >>> code_conf.update(name="ShorCode", rounds= 2)
            >>> print(code_conf)
            CodeConfig(name='ShorCode', rounds=2, size=-1)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.__post_init__()

    def __str__(self) -> str:  # noqa: D105
        return pformat(self)

    def instantiate(self) -> LatticeCode:
        """Instantiate a :class:`~.LatticeCode` object with the objects' config.

        Returns:
            A :class:`~.LatticeCode` object.

        Examples:
            >>> code_conf = CodeConfig.from_dict(dict(name="FiveQubitCode", rounds=1))
            >>> code = code_conf.instantiate()
        """
        match self.name:
            case "ShorCode":
                return LatticeCode.make_shor(n_rounds=self.rounds)
            case "FiveQubitCode":
                return LatticeCode.make_five_qubit(n_rounds=self.rounds)
            case "RepetitionCode":
                return LatticeCode.make_repetition(size=self.size, n_rounds=self.rounds)
            case "PlanarCode":
                return LatticeCode.make_planar(size=self.size, n_rounds=self.rounds)
            case "RotatedPlanarCode":
                return LatticeCode.make_rotated_planar(
                    size=self.size, n_rounds=self.rounds
                )
            case "ToricCode":
                return LatticeCode.make_toric(size=self.size, n_rounds=self.rounds)
            case _:
                raise AttributeError(f"Attribute {self.name} could not be found!")


@dataclass(kw_only=True)
class _QubitErrorMetadata:
    """Class to handle to store the configuration for qubit errors.

    Raises:
         ValueError: If distribution is "user" and params is not empty.
         ValueError: If distribution is "constant" and params is not a list of length 1.
         ValueError: If distribution is "gaussian" and params is not a list of length 2.
         ValueError: If distribution is not one of ("user", "constant", "gaussian")

    Examples:
        >>> conf_1 = _QubitErrorMetadata(distribution='user', enabled = False)
        >>> print(conf_1)
        _QubitErrorMetadata(distribution='user', params=[], enabled=False)
        >>> conf_2 = _QubitErrorMetadata(distribution='constant', params=[0.12])
        >>> print(conf_2)
        _QubitErrorMetadata(distribution='constant', params=[0.12], enabled=True)

    :meta public:
    """

    distribution: str
    """The distribution of the errors.

    Possible values: ``user``, ``constant``, ``gaussian``.
    """
    params: list[float] = field(default_factory=lambda: [])
    """The params for the distribution.

    * Empty list when distribution is "user".
    * List of length 1, when distribution is "constant".
    * List of length 2, when distribution is "gaussian".
    """
    enabled: bool = True
    """Bool to determine where to use errors when running the simulation."""

    def __post_init__(self):
        """Finalise init and run some sanity checks.

        :meta private:
        """
        match self.distribution:
            case "user":
                if self.params:
                    raise ValueError(
                        "Distribution is set to user, but params are provided."
                        "Set distribution to 'gaussian' or 'constant' "
                        "or don't provide params."
                    )
            case "constant":
                if len(cast(list[float], self.params)) != 1:
                    raise ValueError(
                        "The length of params is wrong, "
                        "please provide a list of length 1"
                    )
            case "gaussian":
                if len(cast(list[float], self.params)) != 2:
                    raise ValueError(
                        "The length of params is wrong, "
                        "please provide a list of length 2"
                    )
            case "":  # corresponds to empty object
                pass

            case _:
                raise ValueError(f"Unknown distribution {self.distribution}")

        if self.is_empty():
            self.enabled = False

    @classmethod
    def from_dict(cls, metadata: dict) -> _QubitErrorMetadata:
        """Instantiate :class:`_QubitErrorMetadata` object from config dictionary.

        Args:
            metadata: Dictionary containing qubit error data config.

        Returns:
            A :class:`_QubitErrorMetadata` object.

        Raises:
            ValueError: If distribution is "user" and params is not empty.
            ValueError: If distribution is "constant" and params is not a list of
                length 1.
            ValueError: If distribution is "gaussian" and params is not a list of
                length 2.
            ValueError: If distribution is not one of ("user", "constant", "gaussian").

        Examples:

            >>> conf_1 = {"distribution": "constant", "params": [0.1], "enabled": False}
            >>> qe_meta_1 = _QubitErrorMetadata.from_dict(conf_1)
            >>> print(qe_meta_1)
            _QubitErrorMetadata(distribution='constant', params=[0.1], enabled=False)
            >>> conf_2 = {"distribution": "gaussian", "params": [0.1, 0.04]}
            >>> qe_meta_2 = _QubitErrorMetadata.from_dict(conf_2)
            >>> print(qe_meta_2)
            _QubitErrorMetadata(distribution='gaussian', params=[0.1, 0.04], enabled=True)
        """  # noqa : E501

        metadata.setdefault("enabled", True)
        metadata.setdefault("params", [])

        return cls(
            enabled=cast(bool, metadata.get("enabled")),
            distribution=cast(str, metadata.get("distribution")),
            params=cast(list[float], metadata.get("params")),
        )

    @classmethod
    def make_empty(cls) -> _QubitErrorMetadata:
        """Make an empty :class:`_QubitErrorMetadata` object.

        Examples:
            >>> example =_QubitErrorMetadata.make_empty()
            >>> print(example)
            _QubitErrorMetadata(distribution='', params=[], enabled=False)
        """
        return cls(enabled=False, distribution="", params=[])

    def as_dict(self) -> dict:
        """Return class attributes as dictionary.

        Returns:
            A dictionary of the configuration for the qubit error.
        """
        return asdict(self)

    def update(self, **kwargs):
        """Update class attributes with given ``**kwargs``.

        Args:
            kwargs: Any class attribute.

        Raises:
            ValueError: If distribution is "user" and params is not empty.
            ValueError: If distribution is "constant" and params is not a
                list of length 1.
            ValueError: If distribution is "gaussian" and params is not a
                list of length 2.
            ValueError: If distribution is not one of
                ("user", "constant", "gaussian")

        Returns:
            None, updates in-place.

        Examples:
            >>> obj= _QubitErrorMetadata.make_empty()
            >>> obj.update(distribution="user", enabled = True)
            >>> print(obj)
            _QubitErrorMetadata(distribution='user', params=[], enabled=True)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.__post_init__()

    def is_empty(self) -> bool:
        """Check if the metadata is empty.

        Returns:
            ``True`` if empty, else ``False``.
        """
        if self.distribution == "" and self.params == []:
            return True
        return False


@dataclass(kw_only=True)
class QubitErrorsConfig:
    """Class to store configuration of Qubit Errors.

    Examples:
        >>> from plaquette.frontend import _QubitErrorMetadata, QubitErrorsConfig
        >>> qubit_errors = QubitErrorsConfig(
        ...        data_path='qubit_errors.csv',
        ...        sample=True,
        ...        load_file = False,
        ...        X =  _QubitErrorMetadata(distribution="constant", params=[0.1]),
        ...        Z =  _QubitErrorMetadata(
        ...                 distribution="gaussian", params=[0.1, 0.02]),
        ...        measurement = _QubitErrorMetadata(
        ...                 distribution="constant", params=[0.1]))
        >>> print(qubit_errors)
        QubitErrorsConfig(data_path='qubit_errors.csv',
                          sample=True,
                          load_file=False,
                          X=_QubitErrorMetadata(distribution='constant',
                                                params=[0.1],
                                                enabled=True),
                          Y=_QubitErrorMetadata(distribution='',
                                                params=[],
                                                enabled=False),
                          Z=_QubitErrorMetadata(distribution='gaussian',
                                                params=[0.1, 0.02],
                                                enabled=True),
                          erasure=_QubitErrorMetadata(distribution='',
                                                      params=[],
                                                      enabled=False),
                          measurement=_QubitErrorMetadata(distribution='constant',
                                                          params=[0.1],
                                                          enabled=True),
                          fabrication=_QubitErrorMetadata(distribution='',
                                                          params=[],
                                                          enabled=False))
    """

    data_path: str = str(Path.cwd() / Path("qubit_errors.csv"))
    """The data path to load(save) the qubit errors from(to).

    Defaults to ``$cwd/qubit_errors.csv``
    """
    sample: bool = True
    """Flag to determine whether qubit errors should be sampled into the circuit.

    Defaults to True.
    """
    load_file: bool = False
    """Bool to determine whether the data in the file should be loaded.

    Defaults to False.
    """
    X: _QubitErrorMetadata = field(
        default_factory=lambda: _QubitErrorMetadata.make_empty()
    )
    """:class:`_QubitErrorMetadata` object regarding the pauli X channel errors.

    Defaults to an empty :class:`_QubitErrorMetadata` object.
    """
    Y: _QubitErrorMetadata = field(
        default_factory=lambda: _QubitErrorMetadata.make_empty()
    )
    """:class:`_QubitErrorMetadata` object regarding the Y channel errors.

    Defaults to an empty :class:`_QubitErrorMetadata` object.
    """
    Z: _QubitErrorMetadata = field(
        default_factory=lambda: _QubitErrorMetadata.make_empty()
    )
    """:class:`_QubitErrorMetadata` object regarding the Z channel errors.

    Defaults to an empty :class:`_QubitErrorMetadata` object.
    """
    erasure: _QubitErrorMetadata = field(
        default_factory=lambda: _QubitErrorMetadata.make_empty()
    )
    """:class:`_QubitErrorMetadata` object regarding the erasure channel errors.

    Defaults to an empty :class:`_QubitErrorMetadata` object.
    """
    measurement: _QubitErrorMetadata = field(
        default_factory=lambda: _QubitErrorMetadata.make_empty()
    )
    """:class:`_QubitErrorMetadata` object regarding the measurement errors.

    Defaults to an empty :class:`_QubitErrorMetadata` object.
    """
    fabrication: _QubitErrorMetadata = field(
        default_factory=lambda: _QubitErrorMetadata.make_empty()
    )
    """:class:`_QubitErrorMetadata` object regarding the fabrication errors.

    Defaults to an empty :class:`_QubitErrorMetadata` object.
    """

    NON_QUBIT_KEYS: ClassVar[set[str]] = {"data_path", "sample", "load_file"}
    """
    :meta private:
    """
    QUBIT_KEYS: ClassVar[set[str]] = {
        "X",
        "Y",
        "Z",
        "erasure",
        "measurement",
        "fabrication",
    }
    """
    :meta private:
    """

    def __post_init__(self):  # noqa: D105
        _validate_filepath(self.data_path, self.load_file)

    @classmethod
    def from_dict(cls, configdata: dict):
        """Instantiate :class:`~QubitErrorsConfig` object from config dictionary.

        Args:
            configdata : Dictionary containing the qubit error config schema

        Returns:
            A :class:`~QubitErrorsConfig` object.

        The dictionary schema for the config of qubit errors is shown below.
        All the keys are optional. If a key is not available, it takes the default
        value in the class.

        .. code-block:: python

           {
                "data_path": str,
                "sample": bool,
                "load_file": bool,
                "X": {
                    "distribution": str,
                    "params": list[float]
                    "enabled": bool,
                },
                "Y": {
                    "distribution": str,
                    "params": list[float]
                    "enabled": bool,
                },
                "Z": {
                    "distribution": str,
                    "params": list[float]
                    "enabled": bool,
                },
                "erasure": {
                    "distribution": str,
                    "params": list[float]
                    "enabled": bool,
                },
                "measurement": {
                    "distribution": str,
                    "params": list[float]
                    "enabled": bool,
                },
                "fabrication": {
                    "distribution": str,
                    "params": list[float]
                    "enabled": bool,
                },
           }
        """
        if configdata.get("data_path") is not None:
            if configdata.get("load_file") is not None:
                _validate_filepath(configdata["data_path"], configdata["load_file"])
            elif _validate_filepath(configdata["data_path"], True):
                configdata["load_file"] = True
            else:
                configdata["load_file"] = False
        else:
            configdata.setdefault("data_path", "qubit_errors.csv")
            configdata.setdefault("load_file", False)

        configdata.setdefault("sample", True)
        for key in cls.QUBIT_KEYS:
            if configdata.get(key) is not None:
                configdata[key] = _QubitErrorMetadata.from_dict(
                    cast(dict, configdata.get(key))
                )
            else:
                configdata[key] = _QubitErrorMetadata.make_empty()

        return cls(
            data_path=cast(str, configdata.get("data_path")),
            sample=cast(bool, configdata.get("sample")),
            load_file=cast(bool, configdata.get("load_file")),
            X=cast(_QubitErrorMetadata, configdata.get("X")),
            Y=cast(_QubitErrorMetadata, configdata.get("Y")),
            Z=cast(_QubitErrorMetadata, configdata.get("Z")),
            erasure=cast(_QubitErrorMetadata, configdata.get("erasure")),
            measurement=cast(_QubitErrorMetadata, configdata.get("measurement")),
            fabrication=cast(_QubitErrorMetadata, configdata.get("fabrication")),
        )

    @classmethod
    def make_empty(cls) -> QubitErrorsConfig:
        """Make an empty :class:`~QubitErrorsConfig` object.

        Examples:
            >>> example =QubitErrorsConfig.make_empty()
            >>> print(example)
            QubitErrorsConfig(data_path='qubit_errors.csv',
                              sample=True,
                              load_file=False,
                              X=_QubitErrorMetadata(distribution='',
                                                    params=[],
                                                    enabled=False),
                              Y=_QubitErrorMetadata(distribution='',
                                                    params=[],
                                                    enabled=False),
                              Z=_QubitErrorMetadata(distribution='',
                                                    params=[],
                                                    enabled=False),
                              erasure=_QubitErrorMetadata(distribution='',
                                                          params=[],
                                                          enabled=False),
                              measurement=_QubitErrorMetadata(distribution='',
                                                              params=[],
                                                              enabled=False),
                              fabrication=_QubitErrorMetadata(distribution='',
                                                              params=[],
                                                              enabled=False))
        """
        return cls.from_dict({})

    def as_dict(self) -> dict:
        """Return class attributes as dictionary.

        Returns:
            A dictionary of the configuration for the qubit error.
        """
        return asdict(self)

    def update(self, **kwargs):
        """Update class attributes with given ``**kwargs``.

        Args:
            kwargs: Any class attribute.

        Returns:
            A :class:`~QubitErrorsConfig` object.

        Examples:
            >>> obj = QubitErrorsConfig.make_empty()
            >>> obj.update(
            ...     measurement=_QubitErrorMetadata(distribution='user'),
            ...     sample = True,
            ...     Y = _QubitErrorMetadata(distribution='constant', params = [0.1])
            ... )
            >>> print(obj)
            QubitErrorsConfig(data_path='qubit_errors.csv',
                              sample=True,
                              load_file=False,
                              X=_QubitErrorMetadata(distribution='',
                                                    params=[],
                                                    enabled=False),
                              Y=_QubitErrorMetadata(distribution='constant',
                                                    params=[0.1],
                                                    enabled=True),
                              Z=_QubitErrorMetadata(distribution='',
                                                    params=[],
                                                    enabled=False),
                              erasure=_QubitErrorMetadata(distribution='',
                                                          params=[],
                                                          enabled=False),
                              measurement=_QubitErrorMetadata(distribution='user',
                                                              params=[],
                                                              enabled=True),
                              fabrication=_QubitErrorMetadata(distribution='',
                                                              params=[],
                                                              enabled=False))
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __str__(self):  # noqa: D105
        return pformat(self)

    @property
    def simulated_errors(self) -> tuple[dict, str] | None:
        """Simulated qubit errors as tuple.

        Usually given as input to use in :func:`~.assimilate_qubit_errors`

        Returns:
            A length-2 ``tuple`` of ``dict`` and ``str``.
            The string corresponds to the :attr:`.data_path`.
            The dictionary has keys as the names of the qubit errors that are enabled.
            The value for each key corresponds to a tuple of the values of the
            attributes of the :class:`_QubitErrorMetadata` object the corresponding
            key. See example below.

        Examples:

            >>> from plaquette.frontend import _QubitErrorMetadata, QubitErrorsConfig
            >>> qubit_errors = QubitErrorsConfig(
            ...    data_path='qubit_errors.csv',
            ...    sample=True,
            ...    load_file = False,
            ...    X =  _QubitErrorMetadata(distribution="constant", params=[0.1]),
            ...    Z =  _QubitErrorMetadata(
            ...             distribution="gaussian", params=[0.1, 0.02]),
            ...    measurement = _QubitErrorMetadata(
            ...             distribution="constant", params=[0.1]),
            ...    fabrication = _QubitErrorMetadata(distribution="user")
            ... )
            >>> print(sorted(qubit_errors.simulated_errors[0].items()))
            [('X', ('constant', 0.1)), ('Z', ('gaussian', 0.1, 0.02)), ('fabrication', ('user',)), ('measurement', ('constant', 0.1))]
            >>> print(qubit_errors.simulated_errors[1])
            qubit_errors.csv
        """  # noqa : E501

        if not self.sample:
            return None
        sim_qub_errors_distr: dict = dict()
        for key in self.QUBIT_KEYS:
            qubit_metadata = cast(_QubitErrorMetadata, getattr(self, key))
            if qubit_metadata.enabled and not qubit_metadata.is_empty():
                if qubit_metadata.distribution == "user":
                    sim_qub_errors_distr[key] = (qubit_metadata.distribution,)
                else:
                    sim_qub_errors_distr[key] = (
                        qubit_metadata.distribution,
                        *qubit_metadata.params,
                    )
        return sim_qub_errors_distr, self.data_path


@dataclass(kw_only=True)
class _GateErrorMetadata:
    """Class to handle to store the configuration for gate errors.

    A single gate error can induce multiple error channels. For example,
    an imperfect ``H`` gate can induce an ``X`` error and a ``Z`` error with
    probabilities :math:`0.01, 0.02` respectively.

    Raises:
        ValueError: If :attr:`.distribution`="user" and ``params``, ``induced_errors``
            are not empty.
        ValueError: If params and induced_errors are not of the same length.
        ValueError: If induced_errors is not a valid error channel.
        ValueError: If distribution is "constant" and the corresponding params is not
            of length 1.
        ValueError: If distribution is "gaussian" and the corresponding params is not
            of length 2.
        ValueError: If distribution is not "user", "gaussian" or "constant".

    Examples:
            >>> conf_1 = _GateErrorMetadata(distribution= ["constant", "gaussian"],
            ...           induced_errors= ["XX", "IY"],
            ...           params= [[0.1],[0.2, 0.1]],
            ...           enabled= True)
            >>> print(conf_1)
            _GateErrorMetadata(distribution=['constant', 'gaussian'],
                               induced_errors=['XX', 'IY'],
                               params=[[0.1], [0.2, 0.1]],
                               enabled=True)
            >>> conf_2 = _GateErrorMetadata(distribution= "user",
            ...           enabled= True)
            >>> print(conf_2)
            _GateErrorMetadata(distribution='user',
                               induced_errors=[],
                               params=[],
                               enabled=True)

    :meta public:
    """

    distribution: list[str] | str
    """The distribution for the gate errors.

    Possible values are chosen from ``("user", "constant", "gaussian")``.
    Takes the type ``str`` when ``distribution="user"``.
    'user' cannot be included inside a list currently.
    """
    induced_errors: list[str] = field(default_factory=lambda: [])
    """The error channels induced by the gate.

    When providing two qubit channels, the induced errors must also be strings of
    length 2. For instance, ``CX`` gate can have the ``induced_errors`` as ``['XI',
    'IX']``. However ``['X', 'Z']`` is an invalid  string for the same gate.
    Trivially, single qubit gates can also only induce  single qubit errors.
    """

    params: list[list[float]] = field(default_factory=lambda: [])
    """The params for the different error channels induced.

    Defaults to an empty list.

    * If ``distribution="constant"``, corresponds to a list of length 1.
    * If ``distribution="gaussian"``, corresponds to a list of length 2.
    """

    enabled: bool = True
    """Bool to determine whether to simulate the errors."""

    VALID_INDUCED_ERRORS: ClassVar[set[str]] = (
        set(["".join(i) for i in product("XYZI", repeat=2)])
        .difference("II")
        .union(["X", "Y", "Z"])
    )
    """
    :meta private:
    """

    def __post_init__(self):
        if isinstance(self.distribution, str):
            if len(self.induced_errors) != 0 and len(self.params) != 0:
                raise ValueError("Incorrect values of induced_errors or params!")

        elif isinstance(self.distribution, list):
            if len(self.induced_errors) != len(self.params):
                raise ValueError(
                    "Induced errors and params must be of the same length!"
                )

            for i, dist in enumerate(self.distribution):
                if self.induced_errors[i] not in self.VALID_INDUCED_ERRORS:
                    raise ValueError(
                        f"The given induced error {self.induced_errors[i]} is invalid"
                    )
                match dist:
                    case "constant":
                        if len(self.params[i]) != 1:
                            raise ValueError(
                                "Incorrect length of params, provide a list of length 1"
                            )
                    case "gaussian":
                        if len(self.params[i]) != 2:
                            raise ValueError(
                                "Incorrect length of params, provide a list of length 2"
                            )
                    case _:
                        raise ValueError(
                            "Only 'constant' or 'gaussian' are valid distribution"
                        )

    def __str__(self) -> str:
        return pformat(self)

    @classmethod
    def from_dict(cls, metadata: dict) -> _GateErrorMetadata:
        """Instantiate :class:`_GateErrorMetadata` object from config dictionary.

        Args:
            metadata: Dictionary containing gate error data config.

        Returns:
            A :class:`_GateErrorMetadata` object.

        Raises:
            ValueError: If ``distribution="user"`` and ``params``, ``induced_errors``
                are not empty.
            ValueError: If ``params`` and ``induced_errors`` are not of the same length.
            ValueError: If ``induced_errors`` is not a valid error channel.
            ValueError: If ``distribution="constant"`` and the corresponding
                ``params`` is not of length 1.
            ValueError: If ``distribution="gaussian"`` and the corresponding
                ``params`` is not of length 2.
            ValueError: If ``distribution`` is not "user", "gaussian" or "constant".

        Examples:
            >>> conf_1 = {"distribution": ["constant", "gaussian"],
            ...           "induced_errors": ["XX", "IY"],
            ...           "params": [[0.1],[0.2, 0.1]],
            ...           "enabled": True}
            >>> ge_meta_1 = _GateErrorMetadata.from_dict(conf_1)
            >>> print(ge_meta_1)
            _GateErrorMetadata(distribution=['constant', 'gaussian'],
                               induced_errors=['XX', 'IY'],
                               params=[[0.1], [0.2, 0.1]],
                               enabled=True)
            >>> conf_2 = {"distribution": "user",
            ...           "enabled": True}
            >>> ge_meta_2 = _GateErrorMetadata.from_dict(conf_2)
            >>> print(ge_meta_2)
            _GateErrorMetadata(distribution='user',
                               induced_errors=[],
                               params=[],
                               enabled=True)
        """
        metadata.setdefault("enabled", True)
        metadata.setdefault("induced_errors", [])
        metadata.setdefault("params", [])

        return cls(
            distribution=metadata.get("distribution"),  # type: ignore
            induced_errors=cast(list, metadata.get("induced_errors")),
            params=cast(list, metadata.get("params")),
            enabled=cast(bool, metadata.get("enabled")),
        )

    @classmethod
    def make_empty(cls) -> _GateErrorMetadata:
        """Make an empty :class:`_GateErrorMetadata` object.

        Returns:
            An empty :class:`_GateErrorMetadata` object.

        Examples:

            >>> example =_GateErrorMetadata.make_empty()
            >>> print(example)
            _GateErrorMetadata(distribution='', induced_errors=[], params=[], enabled=False)
        """  # noqa : E501
        return cls(distribution="", induced_errors=[], enabled=False, params=[])

    def as_dict(self) -> dict:
        """Return class attributes as dictionary.

        Returns:
            A dictionary of the configuration for the gate error.
        """
        return asdict(self)

    def update(self, **kwargs):
        """Update class attributes with given ``**kwargs``.

        Args:
            kwargs: Any class attribute.

        Returns:
            None, updates in-place.

        Examples:
            >>> obj= _GateErrorMetadata.make_empty()
            >>> obj.update(distribution="user", enabled = True)
            >>> print(obj)
            _GateErrorMetadata(distribution='user',
                               induced_errors=[],
                               params=[],
                               enabled=True)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def is_empty(self):
        """Check if the metadata is void.

        Returns:
            ``True`` if empty, else ``False``.
        """
        if self.distribution == "" or self.distribution == []:
            return True
        return False


@dataclass(kw_only=True)
class GateErrorsConfig:
    """Class to store configuration of gate errors.

    Examples:
        >>> from plaquette.frontend import _GateErrorMetadata, GateErrorsConfig
        >>> gate_errors = GateErrorsConfig(
        ...         data_path = 'gate_errors.csv',
        ...         sample = True,
        ...         load_file = False,
        ...         CZ = _GateErrorMetadata(
        ...             distribution = ['constant', 'gaussian'],
        ...             induced_errors = ['XZ', 'IX'],
        ...             params = [[0.1], [0.2, 0.05]], enabled = True
        ...         ),
        ...         H = _GateErrorMetadata(
        ...             distribution = 'user', enabled = True
        ...         ),
        ... )
        >>> print(gate_errors)
        GateErrorsConfig(data_path='gate_errors.csv',
                         sample=True,
                         load_file=False,
                         CZ=_GateErrorMetadata(distribution=['constant', 'gaussian'],
                                               induced_errors=['XZ', 'IX'],
                                               params=[[0.1], [0.2, 0.05]],
                                               enabled=True),
                         CX=_GateErrorMetadata(distribution='',
                                               induced_errors=[],
                                               params=[],
                                               enabled=False),
                         H=_GateErrorMetadata(distribution='user',
                                              induced_errors=[],
                                              params=[],
                                              enabled=True),
                         R=_GateErrorMetadata(distribution='',
                                              induced_errors=[],
                                              params=[],
                                              enabled=False),
                         M=_GateErrorMetadata(distribution='',
                                              induced_errors=[],
                                              params=[],
                                              enabled=False),
                         fabrication=_GateErrorMetadata(distribution='',
                                                        induced_errors=[],
                                                        params=[],
                                                        enabled=False))

    """

    data_path: str = str(Path.cwd() / Path("gate_errors.csv"))
    """The data path to load(save) the gate errors from(to).

    Defaults to gate_errors.csv in the current working directory.
    """

    sample: bool = True
    """Bool to determine whether to sample gate errors in the simulation.

    Defaults to ``True``.
    """

    load_file: bool = True
    """Bool to determine whether the data in the file should be loaded.

    Defaults to ``True``.
    """

    CZ: _GateErrorMetadata = field(
        default_factory=lambda: _GateErrorMetadata.make_empty()
    )
    """:class:`_GateErrorMetadata` object regarding the CZ gate errors.

     Defaults to an empty :class:`_GateErrorMetadata` object. Valid ``induced_errors``
     are from :math:`{I,X,Y,Z} \\times  {I,X,Y,Z}` where :math:`\\times`
     represents the cartesian product.
    """

    CX: _GateErrorMetadata = field(
        default_factory=lambda: _GateErrorMetadata.make_empty()
    )
    """:class:`_GateErrorMetadata` object regarding the CX gate errors.

     Defaults to an empty :class:`_GateErrorMetadata` object. Valid ``induced_errors``
     are from :math:`{I,X,Y,Z} \\times  {I,X,Y,Z}` where :math:`\\times`
     represents the cartesian product.
    """

    H: _GateErrorMetadata = field(
        default_factory=lambda: _GateErrorMetadata.make_empty()
    )
    """:class:`_GateErrorMetadata` object regarding the ``H`` gate errors.

    Defaults to an empty :class:`_GateErrorMetadata` object. Valid
    ``induced_errors`` are from :math:`X,Y,Z`.
    """

    R: _GateErrorMetadata = field(
        default_factory=lambda: _GateErrorMetadata.make_empty()
    )
    """:class:`_GateErrorMetadata` object regarding the ``R`` gate errors.

    Defaults to an empty :class:`_GateErrorMetadata` object. Valid
    ``induced_errors`` are from :math:`X,Y,Z`.
    """

    M: _GateErrorMetadata = field(
        default_factory=lambda: _GateErrorMetadata.make_empty()
    )
    """:class:`_GateErrorMetadata` object regarding the ``M`` gate errors.

    Defaults to an empty :class:`_GateErrorMetadata` object. Valid
    ``induced_errors`` are from :math:`X,Y,Z`.
    """

    fabrication: _GateErrorMetadata = field(
        default_factory=lambda: _GateErrorMetadata.make_empty()
    )
    """:class:`_GateErrorMetadata` object regarding the gate ``fabrication`` errors.

    Defaults to an empty :class:`_GateErrorMetadata` object. Valid
    ``induced_errors`` are from :math:`X,Y,Z`.
    """
    NON_GATE_KEYS: ClassVar[set[str]] = {"data_path", "sample", "load_file"}
    """
    :meta private:
    """
    GATE_KEYS: ClassVar[set[str]] = {"CZ", "CX", "H", "R", "M", "fabrication"}
    """
    :meta private:
    """

    def __post_init__(self):  # noqa: D105
        _validate_filepath(self.data_path, self.load_file)

    def __str__(self) -> str:  # noqa: D105
        return pformat(self)

    @classmethod
    def from_dict(cls, configdata: dict) -> GateErrorsConfig:
        """Instantiate :class:`_GateErrorMetadata` object from config dictionary.

        Keyword Args:
            configdata : The dictionary of the gate error configuration data

        Returns:
            A :class:`GateErrorsConfig` object.

        The dictionary schema for the config of qubit errors is shown below.
        All the keys are optional. If a key is not available, it takes the default
        value in the class.

         .. code-block:: python

           {
                "data_path": str,
                "sample": bool,
                "load_file": bool,
                "CZ": {
                    "distribution": list[str] | str,
                    "induced_errors": list[str]
                    "params": list[list[float]],
                    "enabled": bool,
                },
                "CX": {
                    "distribution": list[str] | str,
                    "induced_errors": list[str]
                    "params": list[list[float]],
                    "enabled": bool,
                },
                "H": {
                    "distribution": list[str] | str,
                    "induced_errors": list[str]
                    "params": list[list[float]],
                    "enabled": bool,
                },
                "R": {
                    "distribution": list[str] | str,
                    "induced_errors": list[str]
                    "params": list[list[float]],
                    "enabled": bool,
                },
                "M": {
                    "distribution": list[str] | str,
                    "induced_errors": list[str]
                    "params": list[list[float]],
                    "enabled": bool,
                },
                "fabrication": {
                    "distribution": list[str] | str,
                    "induced_errors": list[str]
                    "params": list[list[float]],
                    "enabled": bool,
                },
           }

        """
        if configdata.get("data_path") is not None:
            if configdata.get("load_file") is not None:
                _validate_filepath(configdata["data_path"], configdata["load_file"])
            elif _validate_filepath(configdata["data_path"], True):
                configdata["load_file"] = True
            else:
                configdata["load_file"] = False
        else:
            configdata.setdefault("data_path", "gate_errors.csv")
            configdata.setdefault("load_file", False)

        configdata.setdefault("sample", False)

        for label in cls.GATE_KEYS:
            if configdata.get(label) is not None:
                configdata[label] = _GateErrorMetadata.from_dict(
                    cast(dict, configdata.get(label))
                )
            else:
                configdata[label] = _GateErrorMetadata.make_empty()

        return cls(
            data_path=cast(str, configdata.get("data_path")),
            sample=cast(bool, configdata.get("sample")),
            load_file=cast(bool, configdata.get("load_file")),
            CZ=cast(_GateErrorMetadata, configdata.get("CZ")),
            CX=cast(_GateErrorMetadata, configdata.get("CX")),
            H=cast(_GateErrorMetadata, configdata.get("H")),
            M=cast(_GateErrorMetadata, configdata.get("M")),
            R=cast(_GateErrorMetadata, configdata.get("R")),
        )

    @classmethod
    def make_empty(cls):
        """Make an empty :class:`GateErrorsConfig` object.

        Examples:
            >>> example = GateErrorsConfig.make_empty()
            >>> print(example)
            GateErrorsConfig(data_path='gate_errors.csv',
                             sample=False,
                             load_file=False,
                             CZ=_GateErrorMetadata(distribution='',
                                                   induced_errors=[],
                                                   params=[],
                                                   enabled=False),
                             CX=_GateErrorMetadata(distribution='',
                                                   induced_errors=[],
                                                   params=[],
                                                   enabled=False),
                             H=_GateErrorMetadata(distribution='',
                                                  induced_errors=[],
                                                  params=[],
                                                  enabled=False),
                             R=_GateErrorMetadata(distribution='',
                                                  induced_errors=[],
                                                  params=[],
                                                  enabled=False),
                             M=_GateErrorMetadata(distribution='',
                                                  induced_errors=[],
                                                  params=[],
                                                  enabled=False),
                             fabrication=_GateErrorMetadata(distribution='',
                                                            induced_errors=[],
                                                            params=[],
                                                            enabled=False))

        """
        return cls.from_dict({})

    def as_dict(self) -> dict:
        """Return class attributes as dictionary.

        Returns:
            A dictionary of the configuration for the qubit error.
        """
        return asdict(self)

    def update(self, **kwargs) -> None:
        """Update class attributes with given ``**kwargs``.

        Args:
            kwargs: Any class attribute.

        Returns:
            None, updates the object in-place.

        Examples:

            >>> obj = GateErrorsConfig.make_empty()
            >>> obj.update(
            ...     data_path = 'gate_errors.csv', sample =True,
            ...     CZ = _GateErrorMetadata(
            ...             distribution=['constant', 'gaussian'],
            ...             induced_errors=['XZ', 'IX'],
            ...             params=[[0.1], [0.2, 0.05]], enabled = True
            ...         ),
            ...     H = _GateErrorMetadata(
            ...             distribution='user', enabled = True
            ...         ),
            ... )
            >>> print(obj)
            GateErrorsConfig(data_path='gate_errors.csv',
                             sample=True,
                             load_file=False,
                             CZ=_GateErrorMetadata(distribution=['constant', 'gaussian'],
                                                   induced_errors=['XZ', 'IX'],
                                                   params=[[0.1], [0.2, 0.05]],
                                                   enabled=True),
                             CX=_GateErrorMetadata(distribution='',
                                                   induced_errors=[],
                                                   params=[],
                                                   enabled=False),
                             H=_GateErrorMetadata(distribution='user',
                                                  induced_errors=[],
                                                  params=[],
                                                  enabled=True),
                             R=_GateErrorMetadata(distribution='',
                                                  induced_errors=[],
                                                  params=[],
                                                  enabled=False),
                             M=_GateErrorMetadata(distribution='',
                                                  induced_errors=[],
                                                  params=[],
                                                  enabled=False),
                             fabrication=_GateErrorMetadata(distribution='',
                                                            induced_errors=[],
                                                            params=[],
                                                            enabled=False))
        """  # noqa : E501
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @property
    def simulated_errors(self) -> tuple[dict, str] | None:
        """Simulated gate errors as tuple.

        Usually given as input to use in :func:`~.assimilate_gate_errors`

        Returns:
            A length-2 tuple of ``dict`` and ``str``.
            The string corresponds to the :attr:`~data_path`.
            The dictionary has keys as the names of the gate errors that are enabled.
            The value for each key corresponds to a tuple of the values of the
            attributes of the :class:`_GateErrorMetadata` object the corresponding key.
            See example below.


        Examples:

            >>> from plaquette.frontend import _GateErrorMetadata, GateErrorsConfig
            >>> gate_errors = GateErrorsConfig(
            ...    data_path='gate_errors.csv',
            ...    sample=True,
            ...    load_file = False,
            ...    CZ =  _GateErrorMetadata(
            ...             distribution=["constant", "constant"],
            ...             induced_errors = ["XX", "YY"],
            ...             params=[[0.1], [0.13]],
            ...         ),
            ...    H =  _GateErrorMetadata(
            ...             distribution=["gaussian", "gaussian"],
            ...             induced_errors = ["X", "Y"],
            ...             params=[[0.1, 0.02], [0.13, 0.07]],
            ...         ),
            ...    M =  _GateErrorMetadata(
            ...             distribution="user",
            ...         ),
            ...    fabrication = _GateErrorMetadata(distribution="user", enabled = False)
            ... )
            >>> print(sorted(gate_errors.simulated_errors[0].items()))
            [('CZ', [('constant', 'XX', 0.1), ('constant', 'YY', 0.13)]), ('H', [('gaussian', 'X', 0.1, 0.02), ('gaussian', 'Y', 0.13, 0.07)]), ('M', ('user',))]
            >>> print(gate_errors.simulated_errors[1])
            gate_errors.csv
        """  # noqa : E501
        if not self.sample:
            return None

        sim_gate_errors_distr: dict = dict()
        for key in self.GATE_KEYS:
            gate_metadata = cast(_GateErrorMetadata, getattr(self, key))
            if not gate_metadata.is_empty() and gate_metadata.enabled:
                if isinstance(gate_metadata.distribution, str):
                    sim_gate_errors_distr[key] = (gate_metadata.distribution,)
                elif isinstance(gate_metadata.distribution, list):
                    lst = list()
                    for i, ind_err in enumerate(gate_metadata.induced_errors):
                        lst.append(
                            (
                                gate_metadata.distribution[i],
                                ind_err,
                                *(gate_metadata.params[i]),
                            )
                        )
                    sim_gate_errors_distr[key] = lst
        return sim_gate_errors_distr, self.data_path


@dataclass(kw_only=True)
class ErrorsConfig:
    """Class to store the configuration for Errors."""

    qubit_errors: QubitErrorsConfig = field(
        default_factory=lambda: QubitErrorsConfig.make_empty()
    )
    """The configuration of qubit errors being simulated.

    Defaults to an empty :class:`~QubitErrorsConfig` object.
    """

    gate_errors: GateErrorsConfig = field(
        default_factory=lambda: GateErrorsConfig.make_empty()
    )
    """The configuration of gate errors being simulated.

    Defaults to an empty :class:`~GateErrorsConfig` object.
    """

    @classmethod
    def from_dict(cls, errors_config: dict) -> ErrorsConfig:
        """Instantiate :class:`~ErrorsConfig` object from config dictionary.

        For schemas for ``qubit_errors``, see :func:`~QubitErrorsConfig.from_dict`.
        For schemas for ``gate_errors``, see :func:`~GateErrorsConfig.from_dict`.

        Args:
            errors_config : The dictionary of the error configuration data

        Returns:
            An :class:`~ErrorsConfig` object.
        """
        if errors_config.get("qubit_errors"):
            qubit_errors = QubitErrorsConfig.from_dict(errors_config["qubit_errors"])
        else:
            qubit_errors = QubitErrorsConfig.make_empty()

        if errors_config.get("gate_errors"):
            gate_errors = GateErrorsConfig.from_dict(errors_config["gate_errors"])
        else:
            gate_errors = GateErrorsConfig.make_empty()

        return cls(
            qubit_errors=qubit_errors,
            gate_errors=gate_errors,
        )

    @property
    def simulated_errors(self) -> tuple:
        """Simulated qubit and gate errors as tuple.

        Returns:
            A length-2 tuple of outputs from
            :func:`~QubitErrorsConfig.simulated_errors` and
            :func:`~GateErrorsConfig.simulated_errors`
        """
        return (
            self.qubit_errors.simulated_errors,
            self.gate_errors.simulated_errors,
        )

    def instantiate(self, lattice: CodeLattice) -> ErrorData:
        """Instantiate a :class:`~.ErrorData` object with the objects' config.

        Args:
            lattice : The lattice of the code being simulated.

        Returns:
            An :class:`~plaquette.errors.ErrorData` object.
        """
        ed = ErrorData.from_lattice(
            lattice,
            self.qubit_errors.simulated_errors,
            self.gate_errors.simulated_errors,
        )
        return ed

    def update(self, **kwargs):
        """Update class attributes with given ``**kwargs``.

        Args:
            kwargs: Any class attribute.

        Returns:
              None, updates in-place

        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass(kw_only=True)
class CircuitConfig:
    """Class to store the configurations for a circuit.

    Examples:
        >>> conf = CircuitConfig(circuit_path='test.txt')
        >>> print(conf)
        CircuitConfig(circuit_provided=False, has_errors=False, circuit_path='test.txt')
    """

    circuit_provided: bool = field(default=False)
    """Flag to determine if circuit is provided by the user.

    ``True`` if a circuit is provided, ``False`` otherwise. Defaults to ``False``.
    """
    has_errors: bool = field(default=False)
    """Flag to determine if provided circuit has error gates in it.

    ``True`` if the provided circuit has errors, ``False`` otherwise. Defaults to
    ``False``.
    """
    circuit_path: str = field(default="circuit.txt")
    """The path to the circuit.

    Defaults to ``"circuit.txt"``.
    """

    def __post_init__(self) -> None:  # noqa: D105
        _validate_filepath(self.circuit_path, self.circuit_provided)

    def __str__(self):  # noqa: D105
        return pformat(self)

    @classmethod
    def from_dict(cls, config_dict: dict | None = None) -> CircuitConfig:
        """Instantiate :class:`~CircuitConfig` object from config dictionary.

        Args:
            config_dict : dictionary containing circuit configurations

        Returns:
            A :class:`~CircuitConfig` object.
        """
        config_dict = cast(dict, config_dict)
        for f in fields(cls):
            match f.name:
                case "has_errors" | "circuit_provided":
                    config_dict.setdefault(f.name, False)
                case "circuit_path":
                    config_dict.setdefault(f.name, "circuit.txt")

        return cls(
            circuit_provided=cast(bool, config_dict.get("circuit_provided")),
            has_errors=cast(bool, config_dict.get("has_errors")),
            circuit_path=cast(str, config_dict.get("circuit_path")),
        )

    def as_dict(self) -> dict[str, Union[bool, str]]:
        """Return class attributes as dictionary.

        Returns:
            A dictionary of the configuration for the code.
        """
        return asdict(self)

    def update(self, **kwargs):
        """Update class attributes with given ``**kwargs``.

        Args:
            kwargs: Any class attribute.

        Returns:
            None, update in-place.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def instantiate(
        self,
        code: LatticeCode | None = None,
        errors: ErrorData | None = None,
        logop_indices=None,
    ) -> Circuit:
        """Instantiate a :class:`.LatticeCode` object with the objects' config.

        Args:
            code : The specified code to simulate
            errors : The error on the qubits in the code
            logop_indices : Specifies which logical operators should be measured
                before and after the QEC simulation.
                See :func:`.generate_qec_circuit` for more details.

        Returns:
            A :class:`~.Circuit` object.

        Examples:
            >>> conf =  CircuitConfig.from_dict(dict(name="FiveQubitCode", rounds=1))
            >>> c = LatticeCode.make_five_qubit(n_rounds=1)
            >>> errs = ErrorData.from_lattice(c.lattice)
            >>> circ = conf.instantiate(code=c, errors=errs, logop_indices='X')
        """
        if self.circuit_provided:
            if self.has_errors:
                with open(self.circuit_path, "r") as f:
                    ret_val = Circuit.from_str(f.read())
            else:
                raise NotImplementedError(
                    "Cannot currently insert errors onto a error free circuit"
                )
        else:
            if code is None or errors is None:
                raise ValueError(
                    "Please pass non-None code and errors to the function."
                )
            ret_val = generator.generate_qec_circuit(
                cast(LatticeCode, code),
                errors.qubit_errors_dict,
                errors.gate_errors_dict,
                logop_indices,
            )
        return ret_val


@dataclass(kw_only=True)
class DeviceConfig:
    """Class to store the configurations for the device.

    Raises:
        ValueError: The device name not one of the supported ones.
        ValueError: Shots is not a positive integer.

    Examples:
        >>> conf = DeviceConfig(name = "clifford", shots =1024)
        >>> print(conf)
        DeviceConfig(name='clifford', shots=1024)
    """

    name: str = field(default="clifford")
    """ The name of the backend to use with the device.

    Valid names for local simulators are ``"clifford"`` and ``"stim"``.
    Defaults to ``"clifford"`` that uses :class:`~.CircuitSimulator`.
    """

    shots: int = field(default=1024)
    """Number of shots for the simulation.

    Must be a positive integer. Defaults to 1024.
    """

    VALID_SIMULATORS: ClassVar[tuple[str, ...]] = tuple(
        plaquette.device.recognized_devices
    )
    """
    :meta private:
    """

    def __post_init__(self):
        """Make sure we are not using an unsupported device."""
        if self.name not in self.VALID_SIMULATORS:
            raise ValueError(f"{self.name} is not a valid device name")
        if not (isinstance(self.shots, int) and self.shots > 0):
            raise ValueError(f"device.shots = {self.shots} must be an integer")

    def __str__(self):  # noqa: D105
        return pformat(self)

    def update(self, **kwargs):
        """Update class attributes with given ``**kwargs``.

        Args:
            kwargs: Any class attribute.

        Returns:
            None, updates in-place.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def from_dict(cls, config_dict: dict) -> DeviceConfig:
        """Instantiate :class:`~DeviceConfig` object from config dictionary.

        Args:
            config_dict: Any class attribute.

        Returns:
            A :class:`~DeviceConfig` object.
        """
        config_dict.setdefault("name", "clifford")
        config_dict.setdefault("shots", 1024)
        return cls(
            name=cast(str, config_dict.get("name")),
            shots=cast(int, config_dict.get("shots")),
        )

    def as_dict(self) -> dict[str, str]:
        """Return class attributes as dictionary.

        Returns:
            A dictionary of the configuration for the code.
        """
        return asdict(self)

    def instantiate(self, **kwargs) -> Device:
        """Instantiate a ``Device`` with the chosen backend using the objects' config.

        Args:
            kwargs: Passed on when the backend is ``"stim"``.

        Returns:
            A :class:`~Device` object.
        """
        if self.name in tuple(plaquette.device.recognized_devices):
            return Device(self.name, **kwargs)
        raise AttributeError(f"{self.name} is an invalid device!")


@dataclass(kw_only=True)
class DecoderConfig:
    """Class to store the config of the decoder.

    Raises:
        ValueError: If name not in :attr:`~.VALID_DECODERS`.

    Examples:
        >>> conf = DecoderConfig(name="UnionFindDecoder")
        >>> print(conf)
        DecoderConfig(name='UnionFindDecoder', weighted=True)

    """

    name: str = field(default="PyMatchingDecoder")
    """The name of the decoder used.

    Defaults to ``PyMatchingDecoder``. Valid names are ``PyMatchingDecoder``,
    ``UnionFindDecoder``,  ``UnionFindNoWeights`` and ``FusionBlossomDecoder``.
    """
    weighted: bool = field(default=True)
    """Flag to determine if weights are used by the decoder.

    Defaults to ``True``.
    """

    VALID_DECODERS: ClassVar[tuple[str, ...]] = (
        "PyMatchingDecoder",
        "UnionFindDecoder",
        "UnionFindNoWeights",
        "FusionBlossomDecoder",
    )
    """
    :meta private:
    """

    def __post_init__(self):
        """Make sure we are not using an unsupported decoder."""
        if self.name not in self.VALID_DECODERS:
            raise ValueError(f"{self.name} is not yet supported!")

    @classmethod
    def from_dict(cls, config_dict: dict) -> DecoderConfig:
        """Instantiate :class:`~DecoderConfig` from config dictionary.

        Args:
            config_dict: Dict containing decoder configurations.
        """
        config_dict.setdefault("name", "PyMatchingDecoder")
        config_dict.setdefault("weighted", False)
        return cls(
            name=cast(str, config_dict.get("name")),
            weighted=cast(bool, config_dict.get("weighted")),
        )

    def as_dict(self) -> dict[str, Union[str, bool]]:
        """Return class attributes as dictionary.

        Returns:
            A dictionary of the configuration for the code.
        """
        return asdict(self)

    def update(self, **kwargs):
        """Update class attributes with given ``**kwargs``.

        Args:
            kwargs: Any class attribute.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.__post_init__()

    def instantiate(self, code: LatticeCode, ed: ErrorData):
        """Instantiate the ``decoder`` with the objects' config.

        Args:
            code : The code being simulated in the experiment.
            ed : The :class:`~plaquette.errors.ErrorData` object used in the
                simulation of the experiment.


        Returns:
            A :class:`~plaquette.decoders.interfaces.PyMatchingDecoder` or
            :class:`~plaquette.decoders.interfaces.UnionFindDecoder` or
            :class:`~plaquette.decoders.interfaces.UnionFindNoWeights` or
            :class:`~plaquette.decoders.interfaces.FusionBlossomDecoder` object.

        Raises:
            ValueError: if weighted is ``True`` and the decoder name is
                ``UnionFindNoWeights``.

        Examples:
            >>> conf =  DecoderConfig.from_dict(dict(name="PyMatchingDecoder"))
            >>> c = LatticeCode.make_five_qubit(n_rounds=1)
            >>> errors = ErrorData.from_lattice(c.lattice)
            >>> dec = conf.instantiate(code =c, ed=errors)
        """
        if self.weighted and self.name == "UnionFindNoWeights":
            raise ValueError(
                "To use UnionFindNoWeights, set decoder_conf.weighted to False"
            )

        ret_val = getattr(decoders, self.name).from_code(
            code, ed.error_data_dict, weighted=self.weighted
        )

        return ret_val


class GeneralConfig(TypedDict):
    """Generic configuration that applies to multiple use-cases."""

    logical_op: str
    """The logical operator to measure.

    .. seealso:: :func:`.generate_qec_circuit`
    """
    qec_property: list[str]
    """The QEC property to estimate.

    Currently only supports ``logical_error_rate``."""
    seed: int
    """The seed for running the simulations."""


@dataclass(kw_only=True)
class ExperimentConfig:
    """Class to handle running a simulation using a configuration file."""

    code_conf: CodeConfig
    """Configuration of the code in the experiment.

    No default value. This is required from the user.
    """

    errors_conf: ErrorsConfig
    """Configuration of the errors in the experiment.

    No default value. This is required from the user.
    """

    general_conf: GeneralConfig = field(
        default_factory=lambda: GeneralConfig(
            logical_op="X", qec_property=["logical_error_rate"], seed=12345
        )
    )
    """Generic configurations for the experiment of the :class:`~GeneralConfig`."""

    device_conf: DeviceConfig = field(default_factory=lambda: DeviceConfig())
    """Configuration of the device in the experiment.

    Defaults to a :class:`~DeviceConfig` object with default values.
    """

    circuit_conf: CircuitConfig = field(default_factory=lambda: CircuitConfig())
    """Configuration of the circuit in the experiment.

    Defaults to a :class:`~CircuitConfig` object with default values.
    """

    decoder_conf: DecoderConfig = field(default_factory=lambda: DecoderConfig())
    """Configuration of the decoder in the experiment.

    Defaults to an :class:`~DecoderConfig` object with default values.
    """

    _code: LatticeCode | None = None
    _errors: ErrorData | None = None
    _device: Type[AbstractSimulator] | None = None
    _circuit: Circuit | None = None
    _decoder: Type[decoders.DecoderInterface] | None = None

    def update_rng(self):  # noqa: D102
        plaquette.rng = np.random.default_rng(seed=self.general_conf["seed"])

    @property
    def code(self) -> LatticeCode:
        """The built :class:`.LatticeCode` object in the experiment."""
        if self._code is None:
            self.build_code()
        return cast(LatticeCode, self._code)

    @property
    def errors(self) -> ErrorData:
        """The built :class:`~.ErrorData` object in the experiment."""
        if self._errors is None:
            raise ValueError(
                "Errors has not been instantiated yet, use build() or "
                "build_errors() method to do so!"
            )
        return cast(ErrorData, self._errors)

    @property
    def circuit(self) -> Circuit:
        """The built :class:`~.Circuit` object in the experiment."""
        if self._circuit is None:
            raise ValueError(
                "Circuit has not been instantiated yet, use build() or "
                + "build_circuit() method to do so!"
            )
        return cast(Circuit, self._circuit)

    @property
    def device(self):
        """The built :class:`~.AbstractSimulator` object in the experiment."""
        if self._circuit is None:
            raise ValueError(
                "Simulator has not been instantiated yet, use build() or "
                + "build_device() method to do so!"
            )
        return self._device

    @property
    def decoder(self):
        """The built ``Decoder`` object in the experiment."""
        if self._decoder is None:
            raise ValueError(
                "Decoder has not been instantiated yet, use build() or "
                + "build_decoder() method to do so!"
            )
        return self._decoder

    @classmethod
    def load_toml(cls, toml_path: str) -> ExperimentConfig:
        """Instantiate an :class:`~.ExperimentConfig` object from a ``toml`` file.

        Args:
            toml_path : The path to the ``toml`` config file

        Returns:
           An :class:`~.ExperimentConfig` object containing the experiment config.

        The example of the ``toml`` file is shown below. The schema can be retrieved
        with :func:`~plaquette.frontend.schemas.get_exp_config_schema`


        .. code-block:: toml

            [general]
            logical_op = "Z" # the logical operator to measure
            qec_property = ["logical_error_rate"] # The QEC property to measure, now
                                                  # only logical_error_rate is possible
            seed = 123124 # the seed for the random number generator

            [device]
            name = "stim" # The device to use,
            shots = 10000 # the number of shots to run the device for

            [code]
            name = "RotatedPlanarCode" # The code to use.
            size = 3 # The size of the code
            rounds = 10 # The number rounds of syndrome measurement per QEC cycle

            [circuit]
            circuit_provided = false
            has_errors = false
            circuit_path = "plaquette/assets/config_tests/surface17_circuit.txt"

            [errors.qubit_errors]
            data_path = "/path/to/errors.csv"
            sample = true

            [errors.qubit_errors.X]
            enabled = false
            distribution = "user"

            [errors.qubit_errors.Y]
            enabled = false
            distribution = "constant"
            params = [0.1]

            [errors.qubit_errors.Z]
            enabled = true
            distribution = "gaussian"
            params = [0.1, 0.01]

            [errors.qubit_errors.erasure]
            enabled = false
            distribution = "gaussian"
            params = [0.1, 0.01]

            [errors.qubit_errors.fabrication]
            distribution = "user"
            enabled = false

            [errors.qubit_errors.measurement]
            enabled = false
            distribution = "constant"
            params = [0.1]

            [errors.gate_errors]
            data_path = "path/to/gate/errors.csv"
            sample = false
            load_file = false

            [errors.gate_errors.CZ]
            induced_errors= ["XX", "ZI"]
            distribution = ["constant", "constant"]
            params =[[0.01], [0.01]]

            [errors.gate_errors.CX]
            induced_errors=["ZZ", "XI"]
            distribution = ["constant", "constant"]
            params =[[0.01], [0.01]]

            [errors.gate_errors.H]
            induced_errors=["X", "Z"]
            distribution=["constant", "constant"]
            params = [[0.01], [0.01]]

            [errors.gate_errors.fabrication]
            distribution = "user"
            enabled = false

            [decoder]
            name = "PyMatchingDecoder"
            weighted = false
        """
        config_dict: dict[str, Any] = toml.load(toml_path)
        _validate_config(config_dict)
        return cls.from_dict(config_dict)

    @classmethod
    def load_json(cls, json_path: str) -> ExperimentConfig:
        """Instantiate :class:`~ExperimentConfig` from a ``json`` file.

        For the schema, see :func:`~plaquette.frontend.schemas.get_exp_config_schema`.

        Args:
            json_path :The path to the ``json`` config file.

        Returns:
           An :class:`~ExperimentConfig` object containing the experiment config.
        """
        config_dict = json.load(open(json_path))
        _validate_config(config_dict)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> ExperimentConfig:
        """Instantiate an :class:`~ExperimentConfig` from a :class:`dict`.

        Args:
            config_dict : The configuration dictionary of the experiment.

        Returns:
           An :class:`~ExperimentConfig` object containing the experiment config.


        An example of the dictionary is shown below. The schema can be retrieved with
        :func:`~plaquette.frontend.schemas.get_exp_config_schema`

        .. code-block:: python

            {
                'circuit': {
                    'circuit_path': 'path/to/circuit.txt',
                    'circuit_provided': False,
                    'has_errors': False
                },
                'code': {
                    'name': 'RotatedPlanarCode',
                    'rounds': 10,
                    'size': 3
                },
                'decoder': {
                    'name': 'PyMatchingDecoder', 'weighted': False
                },
                'errors': {
                    'gate_errors': {
                        'CX': {
                            'distribution': ['constant', 'constant'],
                            'induced_errors': ['ZZ', 'XI'],
                            'params': [[0.01], [0.01]]
                        },
                        'CZ': {
                            'distribution': ['constant', 'constant'],
                            'induced_errors': ['XX', 'ZI'],
                            'params': [[0.01], [0.01]]
                        },
                        'H': {
                            'distribution': ['constant', 'constant'],
                            'induced_errors': ['X', 'Z'],
                            'params': [[0.01], [0.01]]
                        },
                        'data_path': 'path/to/gate/errors.csv',
                        'fabrication': {
                            'distribution': 'user',
                            'enabled': False
                        },
                        'load_file': False,
                        'sample': False
                    },
                    'qubit_errors': {
                        'X': {'distribution': 'user', 'enabled': False},
                        'Y': {
                            'distribution': 'constant',
                            'enabled': False,
                            'params': [0.1]
                        },
                        'Z': {
                            'distribution': 'gaussian',
                            'enabled': True,
                            'params': [0.1, 0.01]
                        },
                        'data_path': '/path/to/errors.csv',
                        'erasure': {
                            'distribution': 'gaussian',
                            'enabled': False,
                            'params': [0.1, 0.01]
                        },
                        'fabrication': {
                            'distribution': 'user',
                            'enabled': False
                        },
                        'measurement': {
                            'distribution': 'constant',
                            'enabled': False,
                            'params': [0.1]
                        },
                        'sample': True
                    }
                },
                'general': {
                    'logical_op': 'Z',
                    'qec_property': ['logical_error_rate'],
                    'seed': 123124
                },
                'device': {'name': 'stim', 'shots': 10000}
            }
        """  # noqa
        config_dict.setdefault("device", {})
        config_dict.setdefault("errors", {})
        config_dict.setdefault("decoder", {})
        config_dict.setdefault("circuit", {})
        config_dict.setdefault("code", {})
        return cls(
            general_conf=cast(GeneralConfig, config_dict.get("general")),
            device_conf=DeviceConfig.from_dict(cast(dict, config_dict.get("device"))),
            circuit_conf=CircuitConfig.from_dict(
                cast(dict, config_dict.get("circuit"))
            ),
            code_conf=CodeConfig.from_dict(cast(dict, config_dict.get("code"))),
            errors_conf=ErrorsConfig.from_dict(cast(dict, config_dict.get("errors"))),
            decoder_conf=DecoderConfig.from_dict(
                cast(dict, config_dict.get("decoder"))
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return all class attributes as dictionary."""
        return_dict: dict[str, Any] = dict()
        for dfield in fields(self):
            if dfield.name == "exp_objects":
                continue
            return_dict[dfield.name] = self.__getattribute__(dfield.name).as_dict()

        return return_dict

    def update(self, **kwargs):
        """Update class attributes with given ``**kwargs``.

        Useful to update multiple attributes at once with a :class:`dict`.

        Args:
            kwargs: Any class attribute.

        Returns:
            None, updates in-place.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def validate(self) -> bool:
        """Validate the current state of configurations."""
        return _validate_config(self.as_dict())

    def __str__(self):
        """Return a slightly better string representation of this object."""
        return pformat(self, compact=False)

    def build(self) -> None:
        """Build the objects required for the experiment's simulation.

        This method will call the relevant classes from :mod:`plaquette` and generate
        the different objects required for performing the simulation.
        """
        self.update_rng()
        self.build_code()
        self.build_errors()
        self.build_circuit()
        self.build_device()
        self.build_decoder()

    def build_code(self) -> None:
        """Build the :class:`.LatticeCode` object.

        Built object is accessible through :class:`ExperimentConfig.code`

        Returns:
            None, updates :attr:`ExperimentConfig.code`.
        """
        self._code = self.code_conf.instantiate()

    def build_errors(self) -> None:
        """Build the :class:`~ErrorData` object.

        Built object is accessible through :class:`~ExperimentConfig.errors`

        Returns:
            None, updates :attr:`~ExperimentConfig.errors`.
        """
        self._errors = self.errors_conf.instantiate(self._code.lattice)  # type: ignore
        # self._errors.to_ErrorData().check_against_code(self._code)
        # FIXME: move check function to ErrorDataUI from ErrorData

    def build_circuit(self) -> None:
        """Build the `Circuit` object.

        Built object is accessible through :class:`~ExperimentConfig.circuit`

        Returns:
            None, updates :attr:`~ExperimentConfig.circuit`.
        """
        self._circuit = self.circuit_conf.instantiate(
            self._code,
            self._errors,
            self.general_conf["logical_op"],
        )

    def build_device(self, **kwargs) -> None:
        """Build the `Circuit` object.

        Built object is accessible through :class:`~ExperimentConfig.circuit`

        Keyword Args:
            batch_size: if using Stim as the backend, defaults to 1024.
            kwargs: keyword arguments to :meth:`.DeviceConfig.instantiate`.

        Returns:
            None, updates :attr:`~ExperimentConfig.circuit`.
        """
        self._device = self.device_conf.instantiate(**kwargs)  # type: ignore

    def build_decoder(self) -> None:
        """Build the `Decoder` object.

        Built object is accessible through :class:`~ExperimentConfig.decoder`

        Returns:
            None, updates :attr:`~ExperimentConfig.decoder`.
        """
        if (
            self.decoder_conf.weighted
            and self.decoder_conf.name == "UnionFindNoWeights"
        ):
            raise ValueError(
                "To use UnionFindNoWeights, set decoder_conf.weighted to False"
            )

        self._decoder = self.decoder_conf.instantiate(
            cast(LatticeCode, self._code),
            cast(ErrorData, self._errors),
        )  # type: ignore

    def generate_empty_error_csv(self, path: str):
        """Generate empty error csv.

        Args:
            path : The path to save the csv to

        Returns:
            None, creates the `csv` at the given path.
        """
        return generate_empty_qubit_errors_csv(
            lattice=cast(LatticeCode, self._code).lattice, csv_path=path
        )

    def dump_toml(self, toml_path: str) -> None:
        """Save the config as a ``toml`` file at the given path."""
        with open(toml_path, "w") as file:
            toml.dump(self.__dict__, file)

    def dump_json(self, json_path: str):
        """Save the config as a ``json`` file at the given path."""
        with open(json_path, "w") as file:
            json.dump(self.__dict__, file)

    def run(self):
        """Run an experiment to get a logical error rate.

        In a threshold plot, a single :class:`ExperimentConfig` gives a single data
        point on the plot. See :ref:`declarative-guide` for an example.
        """
        match self.general_conf["qec_property"]:
            case ["logical_error_rate"]:
                if self._device._backend_class in plaquette.device.local_simulators:
                    test_success = np.zeros([self.device_conf.shots], dtype=bool)
                    for i in tqdm(range(self.device_conf.shots)):
                        self._device.run(cast(Circuit, self._circuit))
                        raw, erasure = self._device.get_sample()
                        results = MeasurementSample.from_code_and_raw_results(
                            self._code, raw, erasure
                        )
                        correction = self._decoder.decode(
                            results.erased_qubits, results.syndrome
                        )
                        test_success[i] = check_success(
                            self._code,
                            correction,
                            results.logical_op_toggle,
                            self.general_conf["logical_op"],
                        )
                    return 1.0 - np.count_nonzero(test_success) / len(test_success)

    @property
    def visualizer(self, code=None) -> visualizer.LatticeVisualizer:
        """Get the `~plaquette.visualizer.LatticeVisualizer` object."""
        if code is None:
            code = self._code
        return visualizer.LatticeVisualizer(code)
