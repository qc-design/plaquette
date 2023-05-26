# Copyright 2023, It'sQ GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Error correction simulators.

This module provides tools for simulating quantum error correction using suitable
Clifford circuits. A circuit from :mod:`plaquette.circuit` can be simulated as follows:

>>> from plaquette.codes import LatticeCode
>>> from plaquette.circuit.generator import generate_qec_circuit
>>> from plaquette.simulator.circuitsim import CircuitSimulator
>>> circ = generate_qec_circuit(LatticeCode.make_planar(1, 4), {}, {}, logical_ops="X")
>>> sim = CircuitSimulator(circ)
>>> sim  # doctest: +ELLIPSIS
<plaquette.simulator.circuitsim.CircuitSimulator object at ...>

In addition to the built-in pure-Python circuit simulator, the faster Stim simulator
can be used by substituting :class:`stimsim.StimSimulator`:

>>> from plaquette.simulator.stimsim import StimSimulator
>>> sim = StimSimulator(circ)
>>> raw, erasure = sim.get_sample()

``raw`` contains all the measurement results from measurement gates in the
circuit, while ``erasure`` contains information on erased qubits if
the :ref:`Gate E_ERASE` was used. The circuit returns measurement results as a
linear array and the function
:meth:`.SimulatorSample.from_code_and_raw_results` can be used to split this
array into different parts (this assumes that the circuit was generated using
:mod:`plaquette.circuit`).
"""

import abc
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from plaquette.codes import LatticeCode
from plaquette.pauli import (
    count_qubits,
    cx,
    cz,
    hadamard,
    reset_qubits_to_eigenstate,
    state_to_stabiliser_string,
    x,
    y,
    z,
    zero_state,
)


class QuantumState:
    """Quantum state represented by stabilizer and destabilizer generators (tableau).

    The description comprises the full tableau of the state including sign, destabilizer
    and stabilizer as introduced in :cite:`aaronson_improved_2004`.

    .. automethod:: __init__
    """

    def __init__(self, qubits: int):
        """Initialize a new state in the computational 0-state.

        Args:
            qubits: Total number of qubits in the state.
        """
        #: Full tableau of the state including sign, destabilizer and stabiliser
        #: generators in the binary picture
        self.tableau = zero_state(qubits)

    def __array__(self) -> np.ndarray:
        """Support direct casting of this object into a numpy array."""
        return self.tableau

    def __str__(self) -> str:  # noqa: D105
        d, s = state_to_stabiliser_string(self.tableau, show_identities=True)
        # you can't use new-lines in f-strings parameters, so this is necessary
        new_line = "\n"
        return f"{new_line.join(d)}\n{'-'*(len(d)+1)}\n{new_line.join(s)}"

    def cx(
        self, control_qubits: int | Sequence[int], target_qubits: int | Sequence[int]
    ):
        """Perform the CNOT gate on this state.

        Args:
            control_qubits: the control qubits.
            target_qubits: the target qubits.
        """
        # FIXME: I don't know why mypy complains here with
        #  "Argument 2 has incompatible type "Union[int, Sequence[int]]"; expected "int"
        #  CLEARLY the types match...
        self.tableau = cx(self.tableau, control_qubits, target_qubits)  # type: ignore

    def cz(
        self, control_qubits: int | Sequence[int], target_qubits: int | Sequence[int]
    ):
        """Perform the CPHASE gate on this state.

        Args:
            control_qubits: the control qubits.
            target_qubits: the target qubits.
        """
        # FIXME: See FIXME in self.cx
        self.tableau = cz(self.tableau, control_qubits, target_qubits)  # type: ignore

    def hadamard(self, qubits: int | Sequence[int]):
        """Perform the Hadamard gate on this state.

        Args:
            qubits: qubits onto which to apply the gate.
        """
        self.tableau = hadamard(self.tableau, qubits)

    def x(self, qubits: int | Iterable[int]):
        """Perform the X gate on this state.

        Args:
            qubits: qubits onto which to apply the gate.
        """
        self.tableau = x(self.tableau, qubits)

    def y(self, qubits: int | Iterable[int]):
        """Perform the Y gate on this state.

        Args:
            qubits: qubits onto which to apply the gate.
        """
        self.tableau = y(self.tableau, qubits)

    def z(self, qubits: int | Iterable[int]):
        """Perform the Z gate on this state.

        Args:
            qubits: qubits onto which to apply the gate.
        """
        self.tableau = z(self.tableau, qubits)

    @property
    def number_of_qubits(self):
        """Number of qubits in the state."""
        return count_qubits(self.tableau)[0]

    @property
    def n_q(self):
        """Alias for number of qubits.

        See Also:
            :attr:`number_of_qubits`
        """
        return count_qubits(self.tableau)[0]

    def reset_qubits_to_eigenstate(self, basis: str, qubits: Sequence[int]):
        """Reset the given ``qubits`` to the plus-eigenstate of the given basis.

        Args:
            basis: either ``"X"`` or ``"Z"``.
            qubits: qubit indices to reset.
        """
        # TODO: check if this is necessary
        if basis not in "XZ" and len(basis) != 1:
            raise ValueError(f"{basis!r} is not a valid basis.")
        self.tableau = reset_qubits_to_eigenstate(self.tableau, basis, qubits)


@dataclass
class SimulatorSample:
    """One sample from a simulator.

    .. automethod:: __init__
    """

    logical_op_initial: np.ndarray
    """Measurement results for logical operators before QEC.

    Array with one entry for each logical operator (0 or 1).
    """
    logical_op_final: np.ndarray
    """Measurement results for logical operators after QEC.
    Array with one entry for each logical operator (0 or 1).
    """
    logical_op_toggle: np.ndarray
    """XOR between logical operator values before and after QEC.

    Array with one entry for each logical operator (0 or 1).
    """
    stabilizer_gen: np.ndarray
    """Measurement results for stabilizer generators.

    The shape is ``(n_rounds + 1, n_stabgens)``. The ``+1`` is necessary because it
    always includes one round of "initial" stabilizer measurements to prepare the
    initial state.
    """
    syndrome: np.ndarray
    """Syndrome data (derived from measurement results for stabilizer generators).

    The shape is ``(n_rounds, n_stabgens)``. It is obtained from ``stabilizer_gen``
    by taking the XOR of consecutive rounds.

    For details of the relation between measurement results and syndrome for multiple
    rounds of stabilizer measurements, see :class:`plaquette.syngraph.Vertex`.
    """
    erased_qubits: Optional[np.ndarray]
    """Erasure information

    The shape is ``(n_rounds, n_qubits)`` (only data qubits, no ancilla qubits).
    Each entry is a boolean and specifies whether the given qubit was erased in the
    given round.
    """

    @classmethod
    def from_code_and_raw_results(
        cls,
        code: LatticeCode,
        raw_results: np.ndarray,
        erasure: Optional[np.ndarray] = None,
        logical_ancilla: bool = False,
    ):
        """Unpack the results from a simulator into a more convenient format.

        Args:
            code: the :class:`.LatticeCode` used to generate the circuit which produced
                the results you want to unpack.
            raw_results: the measurement results from
                :meth:`AbstractSimulator.get_sample`.
            erasure: erasure information from
                :meth:`AbstractSimulator.get_sample`.
            logical_ancilla: flag for alternative method to measure logical operators
                via additional ancilla that is entangled with several physical qubits
        """
        if not logical_ancilla:
            # Default behavior
            code_distance = count_qubits(code.logical_ops, include_identities=False)[0]

            logical_op_initial = np.array(
                [
                    np.sum(raw_results[i * code_distance : (i + 1) * code_distance]) % 2
                    for i in range(code.n_logical_qubits)
                ],
                dtype=int,
            )
            # fmt: off
            logical_op_final = np.array(
                [
                    np.sum(
                        raw_results[len(raw_results) - (i * code_distance)
                            - code_distance : len(raw_results) - (i * code_distance)]  # noqa
                    ) % 2 for i in range(code.n_logical_qubits)
                ][::-1], dtype=int,
            )
            ancillas = np.array(
                raw_results[code.n_logical_qubits * code_distance : -code.n_logical_qubits * code_distance] # noqa
            )
            # fmt: on
        else:
            # Alternative behavior
            logical_op_initial = np.array(raw_results[: code.n_logical_qubits])
            logical_op_final = np.array(raw_results[-code.n_logical_qubits :])
            ancillas = np.array(
                raw_results[code.n_logical_qubits : -code.n_logical_qubits]
            )

        logical_toggle = logical_op_initial ^ logical_op_final
        stab = ancillas.reshape((code.n_rounds + 1, code.n_stabgens))
        syndrome = stab[1:] ^ stab[:-1]

        if erasure is not None:
            if erasure.shape != (code.n_rounds * code.n_data_qubits,):
                raise ValueError("Wrong number of erasure information pieces")
            erasure = erasure.reshape((code.n_rounds, code.n_data_qubits))
        else:
            erasure = None

        return SimulatorSample(
            logical_op_initial=logical_op_initial,
            logical_op_final=logical_op_final,
            logical_op_toggle=logical_toggle,
            stabilizer_gen=stab,
            syndrome=syndrome,
            erased_qubits=erasure,
        )


class AbstractSimulator(metaclass=abc.ABCMeta):
    """Simulator base class.

    .. automethod:: __init__
    """

    @abc.abstractmethod
    def get_sample(self, *, after_reset=True) -> tuple[np.ndarray, np.ndarray]:
        """Sample from simulator.

        Keyword Args:
            after_reset: if ``True``, the simulator must reset its internal
                state before returning a new sample. Subclasses that cannot
                honor this should raise an exception.

        Returns:
            a tuple whose first item is an array of measurement outcomes
            and whose second item is the erasure information.

        Notes:
            The **measurements** item in the returned tuple is a one-dimensional array
            which contains the following entries, in sequence:

            * ``n_logical_qubits`` results from logical operator measurements for state
              preparation.
            * ``(n_rounds + 1) * n_stabgens`` results from stabilizer generator
              measurements:

              * ``n_stabgens`` results from initial stabilizer generator measurements
                for state preparation.
              * ``n_rounds * n_stabgens`` results from stabilizer generator
                measurements.
            * ``n_logical_qubits`` results from logical operator measurements for state
              verification.

            Here, ``n_logical_qubits`` is the number of logical qubits, ``n_stabgens``
            is the number of stabilizer generators and ``n_rounds`` is the number of
            rounds of stabilizer measurements, all of which are attributes of
            :class:`plaquette.codes.StabilizerCode`.

            The described sequence of measurements is implemented in the circuit
            generator in :meth:`~.QECCircuitGenerator.get_circuit`.

            The **erasure** array specifies, for each data qubit and each round of
            measurements, whether it was erased or not. The shape of this array is
            ``[n_rounds * n_qubits]``.

            To unpack these arrays, you can use
            :meth:`~plaquette.simulator.SimulatorSample.from_code_and_raw_results`.

            .. todo::
                We need to come up with a way to remove this additional step. In theory,
                a simulator class should immediately return a :class:`.SimulatorSample`,
                and not require this class method.
        """
        raise NotImplementedError()
