# Copyright 2023, It'sQ GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Quantum error correction codes.

This module defines a number of standard codes for quantum error correction,
such as planar and rotated planar codes, via :class:`.LatticeCode` and its class
methods.

Custom codes can be defined in one of two ways:

1. from operator definitions **or**
2. using a 2D lattice.

Defining e.g. the three-qubit repetition code using only the definition of stabilizer
generators and logical operators can be accomplished as follows:

>>> from plaquette import codes
>>> from plaquette.pauli import string_to_pauli
>>> stabilisers = [string_to_pauli(o, 3) for o in ["ZZI", "IZZ"]]
>>> logical_ops = [string_to_pauli(o, 3) for o in ["XXX", "IIZ"]]
>>> code = codes.LatticeCode.from_operators(
...     stabilisers, logical_ops, n_rounds=1
... )
>>> code  # doctest: +ELLIPSIS
<plaquette.codes.LatticeCode object at ...>

Defining a code using the lattice would start using as follows::

    class IncompleteCode(latticebase.CodeLattice):
        def __init__(self):
            # Initiate a lattice of shape (2, 2)
            super().__init__((2, 2))

            # Define two data qubits at positions (0, 0) and (1, 0)
            self.add_data((0, 0))
            self.add_data((1, 0))

            # Define one single stabilizer generator
            self.add_stabgen((0, 1), latticebase.StabGroup.A)

            # Specify that the stabilizer at (0, 1) acts on both qubits as Z
            l = self.lattice
            self.add_edge(l[0, 0], l[0, 1], latticebase.Pauli.Z)
            self.add_edge(l[1, 0], l[0, 1], latticebase.Pauli.Z)

            # ... add more definitions here ...
            self.assign_indices()

Using the lattice looks more complicated on first sight, but it is generally useful
for defining codes on larger lattices using lattice coordinates instead of linear
indices. The same lattice coordinates are also used to obtain expressive visualizations
using :mod:`plaquette.visualizer`. More details on defining codes can be found in
:ref:`codes-guide`.
"""
from collections.abc import Sequence

import numpy as np

from plaquette.codes import latticebase as lb
from plaquette.codes import latticeinstances
from plaquette.pauli import Tableau, commutator_sign, count_qubits, pauli_to_string


class StabilizerCode:
    """Stores stabilizer and logical operators of a code.

    Physical qubits can be used for the following purposes:

    * Data qubit within the code
    * Ancilla qubit used to measure a stabilizer generator
    * Ancilla qubit used to measure a logical operator

    The "extended" set of qubits contains all these qubits.

    .. automethod:: __init__
    """

    def __init__(self, stabgens: Sequence[Tableau], logical_ops: Sequence[Tableau]):
        """Construct a stabilizer code from stabilizer and logical operators.

        Args:
            stabgens: Stabilizer generators.
            logical_ops: Logical operators (see
                :attr:`~plaquette.codes.latticebase.CodeLattice.logical_ops`
                for expected order).
        """
        logical_operations = len(logical_ops)
        n_qubits = stabgens[0].size // 2
        assert (logical_operations % 2) == 0
        assert n_qubits == logical_ops[0].size // 2
        self.stabilisers = stabgens
        """The stabiliser generators of the code."""
        self.logical_ops = logical_ops
        """The logical operators.

        .. seealso:: :attr:`~plaquette.codes.latticebase.CodeLattice.logical_ops`
           describes the expected order.
        """
        self.logical_x = self.logical_ops[::2]
        r"""The logical :math:`\bar{X}` operators."""
        self.logical_z = self.logical_ops[1::2]
        r"""The logical :math:`\bar{Z}` operators."""
        # self.check() # TODO: Should we check validity while initializing?

    def __eq__(self, other) -> bool:
        """Check that two codes are the same.

        Notes:
            The check is based on equality between the two codes' stabilisers
            and logical operators. If they match across codes, then the two
            codes are deemed the same, irrespective of any additional attribute
            that may differ.
        """
        if not isinstance(other, StabilizerCode):
            return NotImplemented
        return bool(  # mypy complains otherwise
            np.all(
                [
                    (s1 == s2).all()
                    for s1, s2 in zip(self.stabilisers, other.stabilisers)
                ]
            )
            and np.all(
                [
                    (l1 == l2).all()
                    for l1, l2 in zip(self.logical_ops, other.logical_ops)
                ]
            )
        )

    @property
    def n_logical_ops(self) -> int:
        """Number of **logical** operators."""
        return len(self.logical_ops)

    @property
    def n_ext_qubits(self) -> int:
        """Number of all "extended" qubits.

        This includes data and ancilla qubits.

        .. seealso:: :class:`.StabilizerCode`
        """
        return self.n_data_qubits + self.n_stabgens + self.n_logical_ops

    @property
    def n_logical_qubits(self) -> int:
        """Number of **logical** qubits, encoded by the data qubits."""
        return len(self.logical_ops) // 2

    @property
    def n_stabgens(self) -> int:
        """Number of stabilisers generators."""
        return len(self.stabilisers)

    @property
    def n_data_qubits(self) -> int:
        """Number of data/physical qubits."""
        return self.stabilisers[0].size // 2

    @classmethod
    def from_codelattice(cls, lattice: lb.CodeLattice) -> "StabilizerCode":
        """Create stabilizer code from code lattice.

        Args:
            lattice: The code lattice.

        Returns:
            A stabilizer code.
        """
        return cls(lattice.stabilisers, lattice.logical_operators)

    def check(self, *, rank: bool = False):
        """Check that stabilizer code properties are satisfied.

        If this method fails with an AssertionError, the stabilizer code is
        invalid.

        Args:
            rank:
                If ``True``, an AssertionError is raised if the rank of the
                check matrix does not equal its intended value (number of
                physical data qubits minus number of logical qubits).
        """
        # commutator() returns always an array, that's why the [0] everywhere
        assert np.all(
            commutator_sign(self.logical_x, self.logical_z)
            == np.eye(self.n_logical_qubits)
        ), "Logical X and Zs do not satisfy commutation relations"

        assert np.all(
            commutator_sign(self.logical_x, self.logical_x) == 0
        ), "Logical Xs do not commute with themselves"
        assert np.all(
            commutator_sign(self.logical_z, self.logical_z) == 0
        ), "Logical Zs do not commute with themselves"
        assert np.all(
            commutator_sign(self.stabilisers, self.logical_ops) == 0
        ), "Stabilizer generators do not commute with logical operators"
        assert np.all(
            commutator_sign(self.stabilisers, self.stabilisers) == 0
        ), "Stabilizer generators do not commute with themselves"
        # if rank:
        #     assert (
        #         self.stabilisers.check_matrix_rank()
        #         == self.n_data_qubits - self.n_logical_qubits
        #     ), "Rank of the check matrix is incorrect"

    def logical_ops_to_indices(self, logical_ops: str | Sequence[int]) -> Sequence[int]:
        """Convert specification of logical operators to index sequence.

        Args:
            logical_ops:
                Specifies a sequence of logical operators. Possible values:

                * E.g. ``XZ`` specifies logical ``X`` on the first and logical ``Z`` on
                  the second logical qubit.
                * ``[0, 3]`` specifies the same as ``XZ`` because indices refer to
                  :attr:`logical_ops`.

        Returns:
            Sequence of indices referring to logical operators within
            :attr:`logical_ops`.

        .. todo::

            [SPK] We should allow inputs of the form ``XI``.

            [MH] After discussion, the idea of the suggestion was roughly: What if we
            want to determine the index of a single logical qubit on some operator?
            I think in this case it would be better to introduce a separate function,
            e.g. ``get_logical_op_index(logical_qubit=0, logical_op="X")``.
        """
        logop_indices: Sequence[int]
        if isinstance(logical_ops, str):
            try:
                logop_indices = [
                    2 * qubit + {"X": 0, "Z": 1}[op]
                    for qubit, op in enumerate(logical_ops)
                ]
            except KeyError as e:
                raise ValueError(f"{logical_ops=} is invalid") from e
        else:
            logop_indices = logical_ops
        if len(logop_indices) != self.n_logical_qubits:
            raise ValueError("Need one logical operator for each logical qubit")
        if any(i < 0 or i >= self.n_logical_ops for i in logop_indices):
            raise ValueError("Invalid index in `logical_ops`")
        return logop_indices


class LatticeCode(StabilizerCode):
    """Code for QEC, defined using a 2D lattice.

    This class additionally contains a description of the 2D lattice.

    .. note::

       * A :class:`.codes.LatticeCode` contains a :class:`.latticebase.CodeLattice`.
       * A :class:`.LatticeCode` is a variant of :class:`.StabilizerCode`. It
         contains a code which has been defined using a 2D lattice.
       * A :class:`.CodeLattice` is a 2D lattice which is used to define a code.

    .. todo::

       Apologies if the naming is a bit confusing. Suggestions for better names are
       welcome.

    .. automethod:: __init__
    """

    # FIXME: n_rounds should not be a concern for a code
    def __init__(self, lattice: lb.CodeLattice, n_rounds: int):
        """Create a new code.

        Args:
            lattice: Lattice-based code definition.
            n_rounds: Number of rounds of stabilizer measurements.
        """
        self.lattice = lattice
        """The 2D lattice which was used to define this code."""
        self.n_rounds = n_rounds
        super().__init__(lattice.stabilisers, lattice.logical_operators)

    def apply_fabrication_errors(self, **params):
        """Change code by applying fabrication errors.

        All arguments are passed to
        :meth:`plaquette.codes.latticebase.CodeLattice.apply_fabrication_errors`.
        """
        self.lattice._fabrication_error_warning(stacklevel=3)
        self.lattice.apply_fabrication_errors(**params)

    @classmethod
    def from_operators(
        cls,
        stabilisers: Tableau,
        logical_ops: Tableau,
        n_rounds: int,
    ):
        """Define a new code from stabiliser generators and logical operators.

        Args:
            stabilisers: The stabilizer generators defining the code, in binary form.
            logical_ops: The logical operators defining the code, in binary form.
                The order must conform to :attr:`.latticebase.CodeLattice.logical_ops`.
            n_rounds: Number of rounds of stabilizer measurements.
        """
        return cls(
            lb.CodeLatticeFromStab(
                count_qubits(stabilisers[0])[0],
                [pauli_to_string(s) for s in stabilisers],
                [pauli_to_string(lo) for lo in logical_ops],
            ),
            n_rounds,
        )

    @classmethod
    def make_five_qubit(cls, n_rounds: int):
        """Create the five-qubit code.

        Args:
            n_rounds: Number of rounds of stabilizer measurements.
        """
        return cls(latticeinstances.FiveQubitCodeLattice(), n_rounds)

    @classmethod
    def make_toric(cls, n_rounds: int, size: int):
        """Create Toric surface code.

        Args:
            n_rounds: Number of rounds of stabilizer measurements.
            size: Size of the code (edge length of the square lattice).
        """
        return cls(latticeinstances.ToricCodeLattice(size), n_rounds)

    @classmethod
    def make_planar(cls, n_rounds: int, size: int):
        """Create Planar surface code.

        Args:
            n_rounds: Number of rounds of stabilizer measurements.
            size: Size of the code (edge length of the square lattice).
        """
        return cls(latticeinstances.PlanarCodeLattice(size), n_rounds)

    @classmethod
    def make_repetition(cls, n_rounds: int, size: int):
        """Create a repetition code.

        Also known as bit-flip code because it can detect bit flips (X)
        but not phase flips (Z).

        Args:
            n_rounds: Number of rounds of stabilizer measurements.
            size: Size of the code (edge length of the square lattice).
        """
        return cls(latticeinstances.RepetitionCodeLattice(size), n_rounds)

    @classmethod
    def make_rotated_planar(cls, n_rounds: int, size: int, xzzx: bool = False):
        """Create a rotated Planar code.

        Args:
            n_rounds: Number of rounds of stabilizer measurements.
            size: Size of the code (edge length of the square lattice).
            xzzx:
                Whether XZZX stabilizers should be used instead of regular X and Z
                stabilizers.
        """
        return cls(latticeinstances.RotatedPlanarCodeLattice(size, xzzx), n_rounds)

    @classmethod
    def make_shor(cls, n_rounds: int):
        """Make the Shor's nine-qubit code.

        Args:
            n_rounds: Number of rounds of stabilizer measurements.
        """
        return cls(latticeinstances.ShorCodeLattice(), n_rounds)
