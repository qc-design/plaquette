# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Configuration of physical error sources.

Supported errors include:

* Pauli channels
* Erasure errors
* Measurement errors
* Single- and two-qubit Pauli channels after each gate in a circuit.

.. seealso:: For a description of all the possible types of errors, see
   :ref:`errordata-ref`.

Errors can be defined by creating an instance of :class:`ErrorDataDict`.
:class:`ErrorDataDict` is the union of two dictionaries :class:`QubitErrorsDict` and
:class:`GateErrorsDict`, which are ``TypedDict`` dictionaries specifies the error
configurations of the qubits. To simplify specifying error configurations,
the user can use :class:`.ErrorData` and its convenience functions to
play around with error configuration. There required ``TypedDict`` can be generated
via the different methods of this convenience class.

Starting from a :class:`.LatticeCode` and an :class:`.ErrorData` object, one can
generate a circuit (see :mod:`plaquette.circuit`) which can then be used to simulate
the performance of the code under the specified errors. Afterwards, the decoders from
:mod:`plaquette.decoders` can use the :class:`.ErrorDataDict` object to determine
decoding weights which are ideal for a given configuration of errors.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, TypedDict, cast

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

import plaquette
from plaquette.codes import LatticeCode
from plaquette.codes.latticebase import CodeLattice, LogicalVertex


class ErrorValueDict(TypedDict, total=False):  # noqa: D101
    p: float
    """Probability of the error."""


class SinglePauliChannelErrorValueDict(TypedDict, total=False):  # noqa: D101
    x: float
    """Probability of Pauli X error."""
    y: float
    """Probability of Pauli Y error."""
    z: float
    """Probability of Pauli Z error."""


class TwoPauliChannelErrorValueDict(TypedDict, total=False):  # noqa: D101
    ix: float
    """Probability of Pauli IX error."""
    iy: float
    """Probability of Pauli IY error."""
    iz: float
    """Probability of Pauli IZ error."""
    xi: float
    """Probability of Pauli XI error."""
    xx: float
    """Probability of Pauli XX error."""
    xy: float
    """Probability of Pauli XY error."""
    xz: float
    """Probability of Pauli XZ error."""
    yi: float
    """Probability of Pauli YI error."""
    yx: float
    """Probability of Pauli YX error."""
    yy: float
    """Probability of Pauli YY error."""
    yz: float
    """Probability of Pauli YZ error."""
    zi: float
    """Probability of Pauli ZI error."""
    zx: float
    """Probability of Pauli ZX error."""
    zy: float
    """Probability of Pauli ZY error."""
    zz: float
    """Probability of Pauli ZZ error."""


class QubitErrorsDict(TypedDict, total=False):  # noqa: D101
    pauli: dict[int, SinglePauliChannelErrorValueDict]
    """Dictionary for pauli errors.

    Keys are qubit indices and values are :class:`~SinglePauliChannelErrorValueDict`.
    """
    erasure: dict[int, ErrorValueDict]
    """Dictionary for erasure errors.

    Keys are qubit indices and values are:class:`~ErrorValueDict`.
    """
    measurement: dict[int, ErrorValueDict]
    """Dictionary for measurement errors.

    Keys are qubit indices and values are :class:`~ErrorValueDict`.
    """


class GateErrorsDict(TypedDict, total=False):  # noqa: D101
    CX: dict[tuple[int, int], TwoPauliChannelErrorValueDict]
    """Dictionary for CX gate errors.

    Keys are tuple of qubit indices: (control,target) and values are
    :class:`~TwoPauliChannelErrorValueDict`.
    """
    CZ: dict[tuple[int, int], TwoPauliChannelErrorValueDict]
    """Dictionary for CZ gate errors.

    Keys are tuple of qubit indices: (control, target) and values are
    :class:`~TwoPauliChannelErrorValueDict`.
    """
    H: dict[int, SinglePauliChannelErrorValueDict]
    """Dictionary for H gate errors.

    Keys are qubit indices and values are :class:`~SinglePauliChannelErrorValueDict`.
    """
    R: dict[int, SinglePauliChannelErrorValueDict]
    """Dictionary for R gate errors.

    Keys are qubit indices and values are :class:`~SinglePauliChannelErrorValueDict`.
    """
    M: dict[int, SinglePauliChannelErrorValueDict]
    """Dictionary for M gate errors.

    Keys are qubit indices and values are :class:`~SinglePauliChannelErrorValueDict`.
    """


class ErrorDataDict(QubitErrorsDict, GateErrorsDict):
    """Union dictionary of :class:`QubitErrorsDict` and :class:`GateErrorsDict`.

    For details regarding the attributes see the docs of :class:`QubitErrorsDict` and
    :class:`GateErrorsDict`.
    """

    pass


supported_errors = set(QubitErrorsDict.__optional_keys__).union(
    set(GateErrorsDict.__optional_keys__)
)


def delimited_string_list_to_series(
    values: list[str], exp_type: str = "str", index=None, delimiter: str = "|"
) -> pd.Series:
    """Convert list of delimited strings to :class:`~pandas.Series` of list.

    Args:
        values: A list of delimited strings
        exp_type: The expected type of the elements in the string.
        index: The indexing of the dataframe.
        delimiter : The delimiter in the strings, defaults to Pipe("|")

    Returns:
        A Pandas Series of lists

    Raises:
        ValueError: when the ``exp_type`` is not supported.

    Examples:
        >>> example = ["XX|ZZ", "ZZ|XX"]
        >>> print(delimited_string_list_to_series(example))
        0    [XX, ZZ]
        1    [ZZ, XX]
        dtype: object

        >>> example = ["0.1|0.2", "1.0|0.7"]
        >>> print(delimited_string_list_to_series(example))
        0    [0.1, 0.2]
        1    [1.0, 0.7]
        dtype: object
    """
    data: list[list[str]] | list[list[int]] | list[list[float]]
    match exp_type:
        case "str":
            data = [(val.split(delimiter)) for val in values]
        case "int":
            data = [[int(x) for x in val.split(delimiter)] for val in values]
        case "float":
            data = [[float(x) for x in val.split(delimiter)] for val in values]
        case _:
            raise ValueError(
                f"{exp_type} is not supported. Use one of int, str or float."
            )
    if index:
        return pd.Series(data=data, index=index)
    return pd.Series(data=data)


def generate_gaussian_errors(
    mean: float, std_dev: float, qubit_list: list | np.ndarray
) -> np.ndarray:
    """Generate gaussian distribution of errors for the given qubit list.

    Args:
        mean : The mean of the gaussian distribution
        std_dev: The standard deviation of the gaussian distribution
        qubit_list: The list of qubits for which to be generated

    Returns:
        Array of normally distributed errors

    Raises:
       ValueError: if the ``mean`` or `std_dev` doesn't lie between :math:`0.0, 1.0`.

    Examples:
        >>> import plaquette
        >>> plaquette.rng =  np.random.default_rng(seed=1)
        >>> a= generate_gaussian_errors(mean=0.1, std_dev=0.1, qubit_list=np.arange(10))
        >>> print(a)
        [0.1225677  0.27315472 0.04169897 0.2714663  0.08006974 0.10371536
         0.20582905 0.10073515 0.13082476 0.00916322]
    """
    if not 0.0 <= mean <= 1.0:
        raise ValueError(f"Invalid mean={mean}. Must be between 0.0 and 1.0")
    if not 0.0 <= std_dev <= 1.0:
        raise ValueError(
            f"Invalid std deviation={std_dev}. Must be between 0.0 and 1.0"
        )

    distribution = truncnorm(
        (0.0 - mean) / std_dev, (1.0 - mean) / std_dev, mean, std_dev
    )
    return distribution.rvs(size=len(qubit_list), random_state=plaquette.rng)


def generate_constant_errors(val: float, qubit_list: list | np.ndarray) -> np.ndarray:
    """Generate a constant distribution of errors for the given qubit list.

    Args:
        val: The constant value that is used the generate the errors
        qubit_list: The list of qubits for which to be generated.

    Returns:
        Array of constant errors

    Raises:
        ValueError: if ``val`` is not in between :math:`0.0` and :math:`1.0`.

    Examples:
        >>> generate_constant_errors(val = 0.1, qubit_list= np.arange(10))
        array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    """
    if not 0.0 <= val <= 1.0:
        raise ValueError("The value of probability must lie between 0 and 1")
    return np.full(len(qubit_list), fill_value=val)


def generate_empty_qubit_errors_csv(
    lattice: CodeLattice,
    error_names: list[str] | None = None,
    csv_path: str = "qubit_errors.csv",
) -> None:
    """Generate an empty CSV file with given error columns and CodeLattice.

    Calls :func:`~.generate_empty_qubit_errors()` and saves to given path.

    Args:
        lattice : The lattice of the code that is considered
        error_names : The names of errors to be added to columns to the dataframe
        csv_path : The path of the csv to save

    Returns:
        None, saves the dataframe as csv at the given path

    Examples:
        >>> from plaquette.codes import LatticeCode
        >>> import tempfile
        >>> import pandas as pd
        >>> c = LatticeCode.make_five_qubit(n_rounds=1)
        >>> error_names = ["X", "Y", "Z","erasure", "measurement", "fabrication"]
        >>> with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        ...     generate_empty_qubit_errors_csv(c.lattice, error_names, tmp.name)
        ...     df = pd.read_csv(tmp.name, index_col=False)
        >>> print(df)
           qubit_id qubit_type    X    Y    Z  erasure  measurement fabrication
        0         0       data  0.0  0.0  0.0      0.0          0.0   available
        1         1       data  0.0  0.0  0.0      0.0          0.0   available
        2         2       data  0.0  0.0  0.0      0.0          0.0   available
        3         3       data  0.0  0.0  0.0      0.0          0.0   available
        4         4       data  0.0  0.0  0.0      0.0          0.0   available
        5         5       stab  0.0  0.0  0.0      0.0          0.0   available
        6         6       stab  0.0  0.0  0.0      0.0          0.0   available
        7         7       stab  0.0  0.0  0.0      0.0          0.0   available
        8         8       stab  0.0  0.0  0.0      0.0          0.0   available
    """
    generate_empty_qubit_errors(lattice, error_names).to_csv(csv_path, index=False)


def generate_empty_qubit_errors(
    lattice: CodeLattice, error_names: list[str] | None = None
) -> pd.DataFrame:
    """Generate an empty dataframe with given error columns and CodeLattice.

    The valid error names are
    ``{"X", "Y", "Z", "erasure", "measurement", "fabrication"}``

    Args:
        lattice : The lattice of the code that is considered
        error_names : The names of errors to be added to columns to the dataframe

    Returns:
        A Pandas dataframe with the given error names as columns.
        The values are set to zero for every error expect fabrication.
        For fabrication, it is set to available.

    Examples:
        >>> from plaquette.codes import LatticeCode
        >>> code = LatticeCode.make_five_qubit(n_rounds=1)
        >>> error_names = ["X", "Y", "Z","erasure", "measurement", "fabrication"]
        >>> generate_empty_qubit_errors(code.lattice, error_names)
           qubit_id qubit_type    X    Y    Z  erasure  measurement fabrication
        0         0       data  0.0  0.0  0.0      0.0          0.0   available
        1         1       data  0.0  0.0  0.0      0.0          0.0   available
        2         2       data  0.0  0.0  0.0      0.0          0.0   available
        3         3       data  0.0  0.0  0.0      0.0          0.0   available
        4         4       data  0.0  0.0  0.0      0.0          0.0   available
        5         5       stab  0.0  0.0  0.0      0.0          0.0   available
        6         6       stab  0.0  0.0  0.0      0.0          0.0   available
        7         7       stab  0.0  0.0  0.0      0.0          0.0   available
        8         8       stab  0.0  0.0  0.0      0.0          0.0   available
    """
    if error_names is None:
        error_names = ["X", "Y", "Z"]

    if not set(error_names).issubset(
        {"X", "Y", "Z", "erasure", "measurement", "fabrication"}
    ):
        raise ValueError("Invalid set of error names")

    qubit_info = [
        (qubit.equbit_idx, qubit.type.value)
        for qubit in lattice.equbits
        if not isinstance(qubit, LogicalVertex)
    ]
    qubit_pd = pd.DataFrame(qubit_info, columns=["qubit_id", "qubit_type"])
    for name in error_names:
        if name == "fabrication":
            qubit_pd[name] = ["available"] * len(qubit_info)
        else:
            qubit_pd[name] = np.zeros(len(qubit_info))

    return qubit_pd


@dataclass(kw_only=True)
class ErrorData:  # noqa: D101
    qubit_errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    """Dataframe containing the qubit error specification.

    The dataframe has the columns:
    - ``qubit_id`` (int): The qubit ids for the qubits in the code.
    - ``qubit_type`` (str): The type of the qubit. One of data or stab/ancilla.
    - ``X`` (float): The probabilities of pauli X errors sampling.
    - ``Y`` (float): The probabilities of pauli Y errors sampling.
    - ``Z`` (float): The probabilities of pauli Z errors sampling.
    - ``erasure`` (float): The probabilities for sampling erasure errors.
    - `measurement`` (float): The probabilities for sampling measurement errors.
    - ``fabrication``(str): missing/available indicating qubit is available or not.

    Only ``qubit_id`` and ``qubit_type`` are required columns.
    All the other columns are optional.
    """

    gate_errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    """Dataframe containing the gate error specification. The dataframe has the columns:

    - ``gate`` (str): The gate that induces the error
    - ``on_qubits`` (list[int]):  The qubits afflicted by the gate error
    - ``induced_errors`` (list[str]): The channels effected by the faulty gate
    - ``probs`` (list[float]): The probailities of the effected channels
    """

    #: The columns that are actively used in the simulation
    enabled_qubit_errors: set[str] = field(default_factory=set)
    #: The columns that are actively used in the simulation
    enabled_gate_errors: set[str] = field(default_factory=set)

    __cached_code_indices__: dict = field(default_factory=dict)

    @property
    def _cached_code_indices(self):
        """.. todo:: simply make attribute instead of property."""
        return self.__cached_code_indices__

    def __post_init__(self):  # noqa: D105
        if not self.qubit_errors.empty:
            self.qubit_errors.set_index(["qubit_id"], drop=False, inplace=True)
            if not self.enabled_qubit_errors:
                self.enabled_qubit_errors = set(self.qubit_errors.columns[2:])

        if not self.gate_errors.empty:
            if not self.enabled_gate_errors:
                self.enabled_gate_errors = set(self.gate_errors.gate.unique())

    @classmethod
    def from_csv(
        cls, qubit_error_csv: str | None = None, gate_error_csv: str | None = None
    ) -> ErrorData:
        """Instantiate an ErrorData object from CSV file files.

        Args:
            qubit_error_csv : The path to the qubit error csv
            gate_error_csv : The path to the gate error csv

        Returns:
            A :class:`.ErrorData` object with qubit errors and gate errors loaded
            into the respective dataframe from the provided csvs.

        Warnings:
            - If qubit csv is not provided, assumes qubits to be perfect.
            - If gate csv is not provided, assumes the gates to be perfect.

        Examples:
            .. code-block:: python

                errs = ErrorData.from_csv(
                    qubit_error_csv="qubit.csv", gate_error_csv="gate.csv"
                )
        """
        qubit_errors: pd.DataFrame = pd.DataFrame()
        if qubit_error_csv:
            qubit_errors = pd.read_csv(qubit_error_csv)
        else:
            warnings.warn(
                "Qubit errors are not provided! "
                "The simulator assumes the qubits to be perfect",
                stacklevel=2,
            )

        gate_errors: pd.DataFrame = pd.DataFrame()
        if gate_error_csv:
            gate_errors = pd.read_csv(gate_error_csv)
            exp_types = ("int", "str", "float")
            for key, et in zip(gate_errors.keys()[1:], exp_types):
                gate_errors[key] = delimited_string_list_to_series(gate_errors[key], et)

        else:
            warnings.warn(
                "Gates are assumed to be perfect! Gates Errors are not provided.",
                stacklevel=2,
            )

        return cls(qubit_errors=qubit_errors, gate_errors=gate_errors)

    @classmethod
    def from_lattice(
        cls,
        lattice: CodeLattice,
        qubit_error_config: tuple[dict, str] | None = None,
        gate_error_config: tuple[dict, str] | None = None,
    ) -> ErrorData:
        """Instantiate an :class:`ErrorData` object from lattice and error configs.

        Args:
            lattice : lattice of the code being simulated.
            qubit_error_config : configuration of qubit errors. Defaults to ``None``.
            gate_error_config : configuration of gate errors. Defaults to ``None``.

        Returns:
            An :class:`ErrorData` object containing the information of a given lattice.

        Notes:
            - If no information on the errors to be simulated in provided, it defaults
              to X, Y, Z errors at a constant value of 0.01. If atleast one error
              config is provided then an empty dataframe is initialized for whichever
              error config is ``None``.
            - The parameter ``qubit_error_config`` is obtained as the output of
              :attr:`~plaquette.frontend.QubitErrorsConfig.simulated_errors`.
            - The parameter ``gate_error_config`` is obtained as the output of
              :attr:`~plaquette.frontend.GateErrorsConfig.simulated_errors`.

        Examples:
            >>> from plaquette.codes import LatticeCode
            >>> c = LatticeCode.make_five_qubit(n_rounds=1)
            >>> errs = ErrorData.from_lattice(c.lattice)
            >>> print(errs.gate_errors)
            Empty DataFrame
            Columns: [gate, on_qubits, induced_errors, probs]
            Index: []

            The updated ``qubit_errors`` is shown below.

            .. code-block:: text

                          qubit_id qubit_type    X    Y    Z
                qubit_id
                0                0       data  0.1  0.1  0.1
                1                1       data  0.1  0.1  0.1
                2                2       data  0.1  0.1  0.1
                3                3       data  0.1  0.1  0.1
                4                4       data  0.1  0.1  0.1
                5                5       stab  0.1  0.1  0.1
                6                6       stab  0.1  0.1  0.1
                7                7       stab  0.1  0.1  0.1
                8                8       stab  0.1  0.1  0.1

        """
        ed_ui = cls()
        if qubit_error_config is None and gate_error_config is None:
            ed_ui.qubit_errors = pd.DataFrame(
                [
                    (qubit.equbit_idx, qubit.type.value)
                    for qubit in lattice.equbits
                    if not isinstance(qubit, LogicalVertex)
                ],
                columns=["qubit_id", "qubit_type"],
            )
            ed_ui.qubit_errors.set_index(["qubit_id"], drop=False, inplace=True)
            probs = generate_constant_errors(0.1, ed_ui.qubit_errors["qubit_id"])
            for err in ["X", "Y", "Z"]:
                ed_ui.add_qubit_error(ed_ui.qubit_errors["qubit_id"], err, probs)

            ed_ui.gate_errors = pd.DataFrame(
                columns=["gate", "on_qubits", "induced_errors", "probs"]
            )
            return ed_ui

        if qubit_error_config is not None:
            ed_ui.qubit_errors = assimilate_qubit_errors(qubit_error_config, lattice)
            ed_ui.enabled_qubit_errors = set(qubit_error_config[0].keys())

        else:
            ed_ui.qubit_errors = pd.DataFrame(
                [
                    (qubit.equbit_idx, qubit.type.value)
                    for qubit in lattice.equbits
                    if not isinstance(qubit, LogicalVertex)
                ],
                columns=["qubit_id", "qubit_type"],
            )
            ed_ui.qubit_errors.set_index(["qubit_id"], drop=False, inplace=True)

        if gate_error_config is not None:
            ed_ui.gate_errors = assimilate_gate_errors(gate_error_config, lattice)
            ed_ui.enabled_gate_errors = set(gate_error_config[0].keys())

        else:
            ed_ui.gate_errors = pd.DataFrame(
                columns=["gate", "on_qubits", "induced_errors", "probs"]
            )
        return ed_ui

    def add_qubit_error(
        self,
        qubit_id: list[int] | np.ndarray,
        error_name: str,
        probs: list[float] | np.ndarray,
    ) -> None:
        """Add an error column to qubit_errors dataframe in the object.

        Args:
            qubit_id : The list of qubits to add values, the rest default to 0.0
            error_name : The errors to be added. The valid error names are
                (X, Y, Z, erasure, measurement).
            probs : The list of probabilities corresponding to the list of qubits.
                The length of ``probs`` and ``qubit_id`` has to be equal.

        Returns:
            None, updates the dataframe in-place.

        Raises:
            ValueError: if fabrication error is provided as ``error_name``.
            ValueError: if the ``error_name`` is already a column in the dataframe.

        Examples:
            >>> from plaquette.codes import LatticeCode
            >>> c = LatticeCode.make_five_qubit(n_rounds=1)
            >>> errs = ErrorData.from_lattice(c.lattice)
            >>> errs.add_qubit_error(np.arange(9), "erasure",[0.2]*9)

            .. code-block:: text

                          qubit_id qubit_type    X    Y    Z erasure
                qubit_id
                0                0       data  0.1  0.1  0.1     0.2
                1                1       data  0.1  0.1  0.1     0.2
                2                2       data  0.1  0.1  0.1     0.2
                3                3       data  0.1  0.1  0.1     0.2
                4                4       data  0.1  0.1  0.1     0.2
                5                5       stab  0.1  0.1  0.1     0.2
                6                6       stab  0.1  0.1  0.1     0.2
                7                7       stab  0.1  0.1  0.1     0.2
                8                8       stab  0.1  0.1  0.1     0.2

        """
        if error_name == "fabrication":
            raise ValueError("Use the function update_fab_error() instead!")

        if error_name in self.qubit_errors.columns:
            raise ValueError(
                f"{error_name} errors already in dataframe. "
                "Use update_qubit_error() instead"
            )

        self.enabled_qubit_errors.add(error_name)
        self.qubit_errors[error_name] = None
        for i, p in zip(qubit_id, probs):
            self.qubit_errors.at[i, error_name] = p
        self.qubit_errors[error_name].fillna(0.0, inplace=True)

    def update_qubit_error(
        self,
        qubit_id: int | list[int],
        error_name: str,
        probs: float | list[float] | np.ndarray,
    ):
        """Update the error corresponding to the given qubits and error_name.

        Args:
            qubit_id : single id or list of qubit indices to update
            error_name : the name of the error to update
            probs : single float or list of floats to update

        Returns:
            None, updates the dataframe in-place

        Raises:
            ValueError: If ``error_name`` is a fabrication error.
            ValueError: If ``error_name`` is not an existing column in the
                qubit_errors dataframe.
            ValueError: If the length of ``qubit_id`` and ``probs`` are not the same.
            TypeError: If the ``qubit_id``/``probs`` are not both lists or numbers.


        Examples:
            >>> from plaquette.codes import LatticeCode
            >>> c = LatticeCode.make_five_qubit(n_rounds=1)
            >>> errs = ErrorData.from_lattice(c.lattice)
            >>> errs.update_qubit_error(list(np.arange(8)),"Y", [0.2]*8)

            .. code-block:: text

                          qubit_id qubit_type    X    Y    Z
                qubit_id
                0                0       data  0.1  0.2  0.1
                1                1       data  0.1  0.2  0.1
                2                2       data  0.1  0.2  0.1
                3                3       data  0.1  0.2  0.1
                4                4       data  0.1  0.2  0.1
                5                5       stab  0.1  0.2  0.1
                6                6       stab  0.1  0.2  0.1
                7                7       stab  0.1  0.2  0.1
                8                8       stab  0.1  0.1  0.1
        """
        if error_name == "fabrication":
            raise ValueError("Use the function update_fab_error() instead!")

        if error_name not in self.qubit_errors.columns:
            raise ValueError(
                f"{error_name} is not an existing column in the dataframe."
                f"Use the function add_qubit_error() instead"
            )

        if isinstance(qubit_id, int) and isinstance(probs, float):
            self.qubit_errors.at[qubit_id, error_name] = probs

        elif isinstance(qubit_id, list) and (
            isinstance(probs, list) or (isinstance(probs, np.ndarray))
        ):
            if len(probs) != len(qubit_id):
                raise ValueError("len of qubit ids and probs must be the same!")
            for i, p in zip(qubit_id, probs):
                self.qubit_errors.at[i, error_name] = p
        else:
            raise TypeError(
                "Type mismatch. qubit_id/probs should int/float or list[int]/list["
                "float]"
            )

    def add_gate_error(
        self,
        gate_name: str,
        on_qubits: list[int],
        error_names: list[str],
        probs: list[float],
    ):
        """Add a row to the end of the :attr:`.ErrorData.gate_errors` dataframe.

        Args:
            gate_name : Name of the gate affected
            on_qubits : The qubits on which it acts
            error_names : The error channels it induces
            probs : The probability of the channel

        Returns:
            Updates in-place the dataframe containing gate errors

        Notes:
            If there exists some values already existing for the same input parameters,
            when creating the dataframe the bottom most values will take precedence when
            converting to :class:`ErrorData`.

        .. todo:: add checks to in the input values.
        """
        self.enabled_gate_errors.add(gate_name)
        self.gate_errors = pd.concat(
            [
                self.gate_errors,
                pd.Series(
                    dict(
                        gate=gate_name,
                        on_qubits=on_qubits,
                        induced_errors=error_names,
                        probs=probs,
                    )
                )
                .to_frame()
                .T,
            ]
        )

    def update_gate_error(self):
        """Update gate error.

         Updates if the value already exists in the dataframe, otherwise passes to
         add_gate_error() which adds a new row.

        ..todo .. Finish this convenience function.

        """
        raise NotImplementedError("Use add_gate_error() instead!")

    def update(
        self,
        qubit_errors: pd.DataFrame | None = None,
        gate_errors: pd.DataFrame | None = None,
        overwrite: bool = False,
    ) -> None:
        """Update the dataframes in object from a CSV file.

        Args:
            qubit_errors : Dataframe containing qubit errors
            gate_errors : Dataframe containing gate errors
            overwrite : If ``True``, replaces with the given dataframe
                If ``False``, just updates overlapping locations in the dataframe.
                and missing locations are appended to the dataframe
        """
        if qubit_errors:
            if overwrite:
                self.qubit_errors = qubit_errors
            else:
                qubit_errors.set_index(["qubit_id"], drop=True, inplace=True)
                for col in qubit_errors.colums:
                    if col not in self.qubit_errors.columns:
                        self.qubit_errors[col] = None
                    self.qubit_errors.update(qubit_errors.loc[:, col])

            self.enabled_qubit_errors = set(self.qubit_errors.columns[2:])
        if gate_errors:
            if overwrite:
                self.gate_errors = gate_errors
            else:
                self.gate_errors = pd.concat(
                    [self.gate_errors, gate_errors], ignore_index=True
                )
            self.enabled_gate_errors = set(self.gate_errors.gate.unique())

    def update_from_csv(
        self,
        qubit_error_csv: str | None = None,
        gate_error_csv: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Update the stored errors from a given CSV file.

        Args:
            qubit_error_csv : The path to the qubit error csv
            gate_error_csv : The path to the gate error csv
            overwrite : If ``True``, replaces with data from the csv.
                If ``False``, just updates the locations in the Dataframe

        Returns:
            None, updates in-place.
        """
        if qubit_error_csv:
            qubit_errors = pd.read_csv(qubit_error_csv)
            if overwrite:
                self.qubit_errors = qubit_errors
            else:
                qubit_errors.set_index(["qubit_id"], drop=True, inplace=True)
                for col in qubit_errors.columns:
                    if col not in self.qubit_errors.columns:
                        self.qubit_errors[col] = None
                    self.qubit_errors.update(qubit_errors.loc[:, col])

        if gate_error_csv:
            gate_errors = pd.read_csv(gate_error_csv)
            exp_types = ("int", "str", "float")
            for key, et in zip(gate_errors.keys()[1:], exp_types):
                gate_errors[key] = delimited_string_list_to_series(gate_errors[key], et)
            if overwrite:
                self.gate_errors = gate_errors
            else:
                self.gate_errors = pd.concat(
                    [self.gate_errors, gate_errors], ignore_index=True
                )
            self.enabled_gate_errors = set(self.gate_errors.gate.unique())

    @property
    def error_data_dict(self) -> ErrorDataDict:
        """Convert given dataframes to :class:`~.ErrorDataDict`.

        In other modules like :class:`~plaquette.circuit`, :class:`~.ErrorData`
        object is used to extract information relevant to the errors and not an
        :class:`~.ErrorData` object.

        Returns:
            An :class:`~.ErrorDataDict` object

        Examples:
            >>> from plaquette.circuit.generator import generate_qec_circuit
            >>> from plaquette.codes import LatticeCode
            >>> c = LatticeCode.make_five_qubit(n_rounds=1)
            >>> errs = ErrorData.from_lattice(c.lattice)
            >>> circ = generate_qec_circuit(
            ...     c, errs.qubit_errors_dict,
            ...     errs.gate_errors_dict, "X"
            ... )
            >>> print(circ.gates[:5])
            [('H', (0,)), ('M', (0,)), ('H', (0,)), ('H', (1,)), ('M', (1,))]
        """
        errordata = dict()
        if self.enabled_gate_errors:
            for k, v in self.gate_errors_dict.items():
                if k not in GateErrorsDict.__optional_keys__:
                    raise KeyError(f"Invalid key: {k} for typed-dict GateErrorsDict.")
                errordata[k] = v  # type:ignore

        if self.enabled_qubit_errors:
            for k, v in self.qubit_errors_dict.items():
                if k not in QubitErrorsDict.__optional_keys__:
                    raise KeyError(f"Invalid key: {k} for typed-dict QubitErrorsDict.")
                errordata[k] = v  # type: ignore
        return cast(ErrorDataDict, errordata)

    def save_to_csv(
        self, qubit_error_path: str | None = None, gate_error_path: str | None = None
    ) -> None:
        """Save the errors dataframes to CSV file.

        Args:
            qubit_error_path : The path to the qubit error csv file
            gate_error_path : The path to the gate error csv file

        Returns:
            None
        """
        if qubit_error_path is None:
            qubit_error_path = "./qubit_error.csv"
        if gate_error_path is None:
            gate_error_path = "./gate_error.csv"
        self.qubit_errors.to_csv(qubit_error_path)
        self.gate_errors.to_csv(gate_error_path)

    @property
    def qubit_errors_dict(self) -> QubitErrorsDict:
        """Map the qubit errors dataframe to a :class:`~.QubitErrorsDict` dictionary."""
        qubit_errors: QubitErrorsDict = QubitErrorsDict()

        for e in self.enabled_qubit_errors:
            match e:
                case "X" | "Y" | "Z":
                    if qubit_errors.get("pauli") is None:
                        qubit_errors["pauli"] = {
                            idx: SinglePauliChannelErrorValueDict()
                            for idx in self.qubit_errors["qubit_id"]
                            if self.qubit_errors["qubit_type"][idx] == "data"
                        }
                    for k, v in qubit_errors["pauli"].items():
                        v[f"{e.casefold()}"] = self.qubit_errors[e][k]  # type: ignore

                case "erasure":
                    if qubit_errors.get("erasure") is None:
                        qubit_errors["erasure"] = {
                            idx: ErrorValueDict()
                            for idx in self.qubit_errors["qubit_id"]
                            if self.qubit_errors["qubit_type"][idx] == "data"
                        }

                    for key, val in qubit_errors["erasure"].items():
                        val["p"] = self.qubit_errors[e][key]

                case "measurement":
                    if qubit_errors.get("measurement") is None:
                        qubit_errors["measurement"] = dict(
                            {
                                idx: ErrorValueDict()
                                for idx in self.qubit_errors["qubit_id"]
                                if self.qubit_errors["qubit_type"][idx]
                                in ("stab", "ancilla")
                            }
                        )
                    for key, val in qubit_errors["measurement"].items():
                        val["p"] = self.qubit_errors[e][key]

                case "fabrication":
                    pass

        return qubit_errors

    @property
    def gate_errors_dict(self) -> GateErrorsDict:
        """Map the gate errors dataframe to a :class:`~.GateErrorsDict` dictionary."""
        single_paulis: tuple[str, ...] = ("X", "Y", "Z")

        gate_errors_dict: GateErrorsDict = GateErrorsDict()

        for row in self.gate_errors.itertuples():
            if row.gate in self.enabled_gate_errors:
                match row.gate:
                    # single qubit gate errors
                    case "H" | "M" | "R":
                        if gate_errors_dict.get(row.gate) is None:
                            gate_errors_dict[row.gate] = dict()
                        for q_id in row.on_qubits:
                            gate_errors_dict[row.gate][
                                q_id
                            ] = SinglePauliChannelErrorValueDict(  # type: ignore
                                {
                                    str(k).lower(): v
                                    for k, v in zip(row.induced_errors, row.probs)
                                }
                            )
                    # two qubit gate errors
                    case "CX" | "CZ":
                        if gate_errors_dict.get(row.gate) is None:
                            gate_errors_dict[row.gate] = {}
                        if not set(single_paulis).isdisjoint(set(row.induced_errors)):
                            raise KeyError("Two qubit channels must be provided!")
                        for i in range(0, len(row.on_qubits), 2):
                            gate_errors_dict[row.gate][
                                (row.on_qubits[i], row.on_qubits[i + 1])
                            ] = TwoPauliChannelErrorValueDict(  # type: ignore
                                {
                                    str(k).lower(): v
                                    for k, v in zip(row.induced_errors, row.probs)
                                }
                            )

        return gate_errors_dict

    def check_against_code(
        self, code: LatticeCode, errordata: Optional[ErrorDataDict] = None
    ):
        """Check basic properties of error data dictionary.

        This function checks that the names of all specified errors and error
        parameters are known. Preventing spelling errors in parameters is important
        because spelling errors can cause parameters to not be used at all (e.g.
        ``X`` will be ignored if ``x`` is expected).

        Args:
            code: A code is needed to determine whether qubit indices are valid.
            errordata: The error configuration as :class:`~.ErrorDataDict` dictionary.
                If None, uses the :attr:`~.ErrorData.error_data_dict` to check against
                the code.
        """
        if errordata is None:
            errordata = self.error_data_dict

        if invalid := set(errordata).difference(supported_errors):
            raise ValueError(f"Unknown errors: {invalid!r}")
        if "pauli" in errordata:
            self._check_single_phys_qubit_error(
                errordata["pauli"], code, "pauli", {"x", "y", "z"}
            )
        if "erasure" in errordata:
            self._check_single_phys_qubit_error(
                errordata["erasure"], code, "erasure", {"p"}
            )
        if "measurement" in errordata:
            self._check_single_stab_qubit_error(
                errordata["measurement"], code, "measurement", {"p"}
            )
        for gate in ("CX", "CZ"):
            if gate in errordata:
                self._check_twoqubit_gate_error(
                    errordata[gate], code, gate  # type: ignore
                )
        for gate in ("H", "M", "R"):
            if gate in errordata:
                self._check_onequbit_gate_error(
                    errordata[gate], code, gate  # type: ignore
                )

    def _check_single_phys_qubit_error(
        self, error_dict: dict, code: LatticeCode, name: str, allowed_keys: set
    ):
        """Check physical qubit index and parameter keys.

        Args:
            error_dict: The dictionary corresponding to the gate name from
                :class:`~.QubitErrorsDict`
            code: The code against which to check the provided errors.
            name: The name of the qubit error. Must be a key of
                :class:`~.QubitErrorsDict`.
            allowed_keys: The allowed keys for the error key. Keys of either
                :class:`~.SinglePauliChannelErrorValueDict` or
                :class:`~.ErrorValueDict`.

        Returns:
            None.

        Raises:
            ValueError: If the key(qubit index) doesn't correspond to a data qubit.
            ValueError: If the values(dictionary) doesn't have the keys corresponding to
                its dictionary type.
        """
        for equbit_idx, params in error_dict.items():
            code.lattice.check_equbit_is_data(equbit_idx, error=name)
            if invalid := set(params) - allowed_keys:
                raise ValueError(
                    f"{name} qubit {equbit_idx!r}: Invalid keys {invalid!r}"
                )

    def _check_single_stab_qubit_error(
        self, error_dict: dict, code: LatticeCode, name: str, allowed_keys: set
    ):
        """Check stabilizer ancilla qubit index and parameter keys.

        Args:
            error_dict: The dictionary corresponding to the gate name from
                :class:`~.QubitErrorsDict`
            code: The code against which to check the provided errors.
            name: The name of the qubit error. Must be a key of
                :class:`~.QubitErrorsDict`.
            allowed_keys: The allowed keys for the error key. Keys are from
                :class:`~.ErrorValueDict`.

        Returns:
            None.

        Raises:
            ValueError: If the key(qubit index) doesn't correspond to a stabilizer
                qubit.
            ValueError: If the values(dictionary) doesn't have the keys corresponding to
                its dictionary type.

        """
        for equbit_idx, params in error_dict.items():
            code.lattice.check_equbit_is_stabgen(equbit_idx, error=name)
            if invalid := set(params) - allowed_keys:
                raise ValueError(
                    f"{name} qubit {equbit_idx!r}: Invalid keys {invalid!r}"
                )

    def _check_twoqubit_gate_error(
        self, error_dict: dict, code: LatticeCode, gate: str
    ) -> None:
        """Check two qubit gate errors against the code.

        Args:
            error_dict: The dictionary corresponding to the CX/CZ gate from
                :class:`~.GateErrorsDict`
            code: The code against which to check the provided errors.
            gate: The name of the qubit error. Must be a key of
                :class:`~.GateErrorsDict`.

        Returns:
            None

        Raises:
            ValueError: Qubit keys is not a tuple.
            ValueError: If the values(dictionary) doesn't have the keys corresponding to
                its dictionary type.
        """
        for qubits, errors in error_dict.items():
            if len(qubits) != 2:
                raise ValueError(f"Two-qubit gates needs two arguments, got {qubits!r}")
            code.lattice.check_equbit_is_stabgen(qubits[0], gate=gate)
            code.lattice.check_equbit_is_data(qubits[1], gate=gate)

            invalid = set(errors.keys()).issubset(
                TwoPauliChannelErrorValueDict.__optional_keys__
            )
            if not invalid:
                raise ValueError(f"Invalid keys in gate: {gate!r}")

    def _check_onequbit_gate_error(
        self, error_dict: dict, code: LatticeCode, gate: str
    ) -> None:
        """Check one qubit gate errors against the code.

        Args:
            error_dict: The dictionary corresponding to the CX/CZ gate from
                :class:`~.GateErrorsDict`
            code: The code against which to check the provided errors.
            gate: The name of the qubit error. Must be a key of
                :class:`~.GateErrorsDict`.

        Returns:
            None.

        Raises:
            ValueError: Qubit keys is not a stabilizer qubit.
            ValueError: If the values(dictionary) doesn't have the keys corresponding to
                its dictionary type.
        """
        for equbit_idx, errors in error_dict.items():
            code.lattice.check_equbit_is_stabgen(equbit_idx, gate=gate)
            invalid = set(errors.keys()).issubset(
                SinglePauliChannelErrorValueDict.__optional_keys__
            )

            if not invalid:
                raise ValueError(f"Invalid keys for gate: {gate!r}")

    def cache_full_code_indices(self, code: LatticeCode):
        """Cache the indices of the full code without fabrication errors.

        Args:
            code: the code whose indices we want to cache.

        Returns:
            None, updates the class
        """
        self.__cached_code_indices__ = {
            qubit.pos: qubit.equbit_idx
            for qubit in code.lattice.equbits
            if not isinstance(qubit, LogicalVertex)
        }

    # TODO: implement this
    # def toggle_fabrication_error(self, code: LatticeCode) -> LatticeCode:
    #     """Toggle fabrication error on/off.
    #
    #     Args:
    #         code : The code to which to apply fabrication errors to.
    #
    #     Returns:
    #         None
    #
    #     Notes:
    #
    #         What should this function do?
    #         * save the full code indices without fabrication errors to
    #             __cached_code_indices
    #         * Create new code according to the input fabrication errors.
    #         * update ErrorData dataframes with the new indices of the codes.
    #         * This function should be explicitly called even if the fabrication
    #         errors are in the dataframes. This is required because the code indexing
    #         is different.
    #     """
    #     raise NotImplementedError


def assimilate_qubit_errors(
    qubit_error_config: tuple[dict, str],
    lattice: CodeLattice,
) -> pd.DataFrame:
    """Assimilate qubit errors from config class and CSV file.

    Args:
        qubit_error_config : output from
            :attr:`~plaquette.frontend.QubitErrorsConfig.simulated_errors`
        lattice : lattice of the code being simulated.

    Returns:
        A dataframe of qubit errors, typically used to set
        :attr:`~.ErrorData.qubit_errors`
    """
    qubit_pd = generate_empty_qubit_errors(lattice, list(qubit_error_config[0].keys()))
    qubit_pd.set_index(["qubit_id"], drop=False, inplace=True)

    try:
        qubit_csv_df = pd.read_csv(qubit_error_config[1])
        qubit_csv_df.set_index(["qubit_id"], drop=True, inplace=True)
    except FileNotFoundError:
        warnings.warn(
            "The given path to `csv` was inaccessible, ignoring the errors from csv."
            "To add the errors from csv, "
            "use `update_qubit_errors()` with the right path.",
            stacklevel=2,
        )
        qubit_csv_df = pd.DataFrame()

    for label in qubit_error_config[0].keys():
        if not qubit_error_config[0].get(label):
            qubit_error_config[0][label] = ("constant", 0.01)
        match qubit_error_config[0][label][0]:
            case "constant":
                qubit_pd[label] = np.full(
                    qubit_pd.shape[0], qubit_error_config[0][label][1]
                )
            case "gaussian":
                qubit_pd[label] = np.random.default_rng().normal(
                    loc=qubit_error_config[0][label][1],
                    scale=qubit_error_config[0][label][2],
                    size=qubit_pd.shape[0],
                )
            case "user":
                if qubit_csv_df.empty:
                    continue
                if label not in qubit_pd.columns:
                    qubit_pd[label] = None
                qubit_pd.update(qubit_csv_df.loc[:, label])

    return qubit_pd


def assimilate_gate_errors(
    gate_error_config: tuple[dict, str], lattice: CodeLattice
) -> pd.DataFrame:
    """Assimilate gate errors from config and CSV file.

    Args:
        gate_error_config : The output from
            :attr:`~plaquette.frontend.GateErrorsConfig.simulated_errors`
        lattice : the lattice of the code being simulated

    Returns:
        A dataframe of gate errors, typically used to set
        :attr:`~.ErrorData.gate_errors`
    """
    gate_pd = pd.DataFrame(columns=["gate", "on_qubits", "induced_errors", "probs"])
    try:
        gate_csv = pd.read_csv(gate_error_config[1])
    except FileNotFoundError:
        gate_csv = pd.DataFrame(
            columns=["gate", "on_qubits", "induced_errors", "probs"]
        )
    exp_types = ("int", "str", "float")
    for key, et in zip(gate_csv.keys()[1:], exp_types):
        gate_csv[key] = delimited_string_list_to_series(gate_csv[key], et)

    for label in gate_error_config[0].keys():
        if not gate_error_config[0][label][0] == "user":
            match label:
                case "H" | "M" | "R":
                    for params in gate_error_config[0][label]:
                        if params[0] == "constant":
                            gate_pd = pd.concat(
                                [
                                    gate_pd,
                                    pd.Series(
                                        dict(
                                            gate=label,
                                            on_qubits=[
                                                i.equbit_idx for i in lattice.stabgens
                                            ],
                                            induced_errors=[params[1]],
                                            probs=[params[2]],
                                        )
                                    )
                                    .to_frame()
                                    .T,
                                ],
                                ignore_index=True,
                            )

                        elif params[0] == "gaussian":
                            probs = np.random.default_rng().normal(
                                loc=params[2],
                                scale=params[3],
                                size=len(lattice.stabgens),
                            )
                            for i, p in zip(lattice.stabgens, probs):
                                gate_pd = pd.concat(
                                    [
                                        gate_pd,
                                        pd.Series(
                                            dict(
                                                gate=label,
                                                on_qubits=[i.equbit_idx],
                                                induced_errors=[params[1]],
                                                probs=[p],
                                            )
                                        )
                                        .to_frame()
                                        .T,
                                    ],
                                    ignore_index=True,
                                )

                case "CX" | "CZ":
                    for params in gate_error_config[0][label]:
                        on_qubits: list[int] = list()
                        for stab in lattice.stabgens:
                            for edge in stab.edges:
                                on_qubits.extend(
                                    [stab.equbit_idx, edge.data.equbit_idx]
                                )

                        if params[0] == "constant":
                            gate_pd = pd.concat(
                                [
                                    gate_pd,
                                    pd.Series(
                                        dict(
                                            gate=label,
                                            on_qubits=on_qubits,
                                            induced_errors=[params[1]],
                                            probs=[params[2]],
                                        )
                                    )
                                    .to_frame()
                                    .T,
                                ],
                                ignore_index=True,
                            )

                        elif params[0] == "gaussian":
                            probs = np.random.default_rng().normal(
                                loc=params[2],
                                scale=params[3],
                                size=len(on_qubits) // 2,
                            )

                            for ind, p in enumerate(probs):
                                gate_pd = pd.concat(
                                    [
                                        gate_pd,
                                        pd.Series(
                                            dict(
                                                gate=label,
                                                on_qubits=on_qubits[
                                                    2 * ind : 2 * (ind + 1)
                                                ],
                                                induced_errors=params[1],
                                                probs=[p],
                                            )
                                        )
                                        .to_frame()
                                        .T,
                                    ],
                                    ignore_index=True,
                                )

        elif not gate_csv.empty:
            gate_pd = pd.concat(
                [
                    gate_pd,
                    gate_csv.loc[gate_csv.gate == label],
                ],
                ignore_index=True,
            )
    return gate_pd
