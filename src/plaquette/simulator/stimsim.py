# Copyright 2023, It'sQ GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Interface to Stim for use as circuit simulator.

Stim is available from https://github.com/quantumlib/Stim (Apache 2.0 license).
"""

from typing import Optional

import numpy as np
import stim  # type: ignore

import plaquette
from plaquette import circuit, simulator


def _append_equiprobable(circ: stim.Circuit, p: float, steps):
    """Helper function for adding equiprobable cases to Stim circuit."""
    for i, step in enumerate(steps):
        gate = ("ELSE_" if i > 0 else "") + "CORRELATED_ERROR"
        p_i = p / (1 - i * p)
        circ.append(gate, step, p_i)


def circuit_to_stim(circ: circuit.Circuit) -> tuple[stim.Circuit, list[bool]]:
    """Convert Clifford circuit in ``plaquette``'s format to Stim's format.

    Args:
        circ: Circuit in ``plaquette``'s format

    Returns:
        a tuple of two elements.

        ``stim_circuit``:
            Circuit in Stim's format
        ``meas_is_erasure``
           For each measurement result, this list contains an entry which specifies
           whether the result signals an erasure (``True``) or a regular
           measurement outcome (:class`False`).
    """
    n_qubits = circ.number_of_qubits
    # One additional qubit is used to herald erasures (if necessary).
    erasure_ancilla = n_qubits
    x_erasure_ancilla = stim.target_x(erasure_ancilla)
    res = stim.Circuit()
    # For each measuremen result, the following list contains an entry which specifies
    # whether the result signals an erasure or a regular measurement outcome.
    meas_is_erasure: list[bool] = []
    for name, args in circ.gates:
        match name:
            case "X" | "Y" | "Z" | "H" | "R" | "CX" | "CZ":
                res.append(name, args)
            case "M":
                res.append(name, args)
                meas_is_erasure.extend([False] * len(args))
            case "DEPOLARIZE":
                res.append(name, args, 1.0)
            case "E_PAULI":
                assert len(args) == 4
                res.append("PAULI_CHANNEL_1", args[3], args[:3])
            case "E_PAULI2":
                assert len(args) == 17
                res.append("PAULI_CHANNEL_2", args[15:], args[:15])
            case "E_ERASE":
                assert len(args) == 2
                p, target = args
                # Reference: https://quantumcomputing.stackexchange.com/a/26583
                res.append("R", erasure_ancilla)
                _append_equiprobable(
                    res,
                    p / 4,
                    (
                        (x_erasure_ancilla, stim.target_x(target)),
                        (x_erasure_ancilla, stim.target_y(target)),
                        (x_erasure_ancilla, stim.target_z(target)),
                        (x_erasure_ancilla,),
                    ),
                )
                res.append("M", erasure_ancilla)
                meas_is_erasure.append(True)
            case "ERROR":
                p, name2, *args2 = args
                if name2 in ("X", "Y", "Z"):
                    if len(args2) != 1:
                        raise ValueError("ERROR ... XYZ only supported on one qubit")
                    res.append(name2 + "_ERROR", args2, p)
                else:
                    raise ValueError(f"ERROR ... {name!r} not supported yet")
            case "ERROR_ELSE" | "ERROR_CONTINUE":
                raise ValueError(f"{name!r} not supported yet")
    return res, meas_is_erasure


class StimSimulator(simulator.AbstractSimulator):
    """Circuit simulator using Stim as backend.

    .. automethod:: __init__
    """

    #: The circuit
    circ: circuit.Circuit
    #: The circuit, converted to Stim's format
    stim_circ: stim.Circuit
    #: Determines which measurement results from the Stim circuit are actually erasure
    #: indicators.
    meas_is_erasure: np.ndarray
    #: Seed used when last rebuilding Stim's sampler.
    stim_seed: int
    #: The last-used sampler.
    stim_sampler: stim.CompiledMeasurementSampler = None
    #: Batch size for retrieving samples from Stim (retrieving single samples is
    #: inefficient).
    batch_size: int
    #: Remaining unused entries from current batch.
    batch_remaining: int = 0
    #: Data from current batch.
    batch: np.ndarray | None = None

    def __init__(
        self,
        circ: circuit.Circuit | circuit.CircuitBuilder,
        *,
        stim_seed: Optional[int] = None,
        batch_size: int = 1024,
    ):
        """Create a new Stim-based circuit simulator.

        Args:
            circ: The circuit (or the builder containing it) to be simulated.
            stim_seed: If omitted, a random seed is generated using
                :attr:`plaquette.rng`.
            batch_size: Number of pre-computed samples.

        Stim is more efficient if we compute multiple samples simultaneously. For this
        reason, :attr:`get_sample()` pre-computes a number of ``batch_size`` samples
        whenever it needs to get new samples from Stim.
        """
        if isinstance(circ, circuit.CircuitBuilder):
            self.circ = circ.circ
        elif isinstance(circ, circuit.Circuit):
            self.circ = circ
        else:
            raise TypeError(
                "Only a Circuit or a CircuitBuilder can be used in a simulator"
            )
        self.batch_size = batch_size
        # Convert the circuit to Stim format. A sampler is not built until an RNG
        # is supplied.
        self.stim_circ, is_erasure = circuit_to_stim(self.circ)
        self.meas_is_erasure = np.array(is_erasure)
        if stim_seed is None:
            stim_seed = plaquette.rng.integers(0, 2**63)
        self._compile_stim_sampler(stim_seed)

    def _compile_stim_sampler(self, stim_seed: int):
        """Compile a stim sampler."""
        self.stim_seed = stim_seed
        self.stim_sampler = self.stim_circ.compile_sampler(seed=stim_seed)
        # Erase current batch (if any)
        self.batch = None
        self.batch_remaining = 0

    def get_sample(  # noqa: D102
        self, *, after_reset=True
    ) -> tuple[np.ndarray, np.ndarray]:
        # Sample a new match if necessary.
        if not after_reset:
            raise ValueError("Stim does not allow sampling without resetting")
        if self.batch_remaining == 0:
            self.batch = self.stim_sampler.sample(shots=self.batch_size)
            self.batch_remaining = self.batch_size
        assert self.batch is not None
        all_meas = self.batch[-self.batch_remaining]
        self.batch_remaining -= 1
        # Split results into actual measurements and erasure indications
        meas = all_meas[~self.meas_is_erasure]
        qubits_erased = all_meas[self.meas_is_erasure]
        if len(qubits_erased) == 0:
            qubits_erased = None
        return meas, qubits_erased
