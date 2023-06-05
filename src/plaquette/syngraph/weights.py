# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Computation of edge weights for syndrome graph."""

from __future__ import annotations

import numpy as np

from plaquette import codes
from plaquette.codes import latticebase
from plaquette.errors import ErrorData, ErrorDataDict


class WeightComputation:
    r"""Internal data structures for computing edge weights.

    This model uses parameters for single-qubit Pauli channels and for measurement
    errors.

    Erasure errors do not contribute to decoding weights because they are heralded.

    .. todo::

        The following todos contain parts of the docstring written when the name of
        the class was "BasicErrorModel". They should be deleted or integrated into
        the actual docstring (if still relevant).

    .. todo::

       Currently, the error model depends on the syndrome graph because it has
       to compute weights for syndrome graph edges. This needs access to the
       edge definition within the SyndromeGraph and to the number of rounds.

       It may be better to have ErrorModel independent from the SyndromeGraph.
       After all, the error model should need information about the physical
       system (number of qubits and lattice structure) but not about the error
       correction code we use (i.e. access to the syndrome graph should not be
       needed).

    .. todo::

        The following is outdated and can be removed:

        An erasure error with probability :math:`p_e` causes a Pauli X or Y error
        with probability :math:`\frac{p_e}2`.

        Given the erasure probability :math:`p_e` and the Pauli X error probability
        :math:`p_x`, the probability for an effective X error is

        :math:`p_{xe} = p_x \left( 1 - \frac{p_e}2 \right) + \frac{p_e}{2} (1 - p_x)`

        .. todo::

           This computation of weights is in the spirit of the Union Find decoder.
           Will it also work for super-stabilizers with a matching decoder?

    .. todo::

        [The following is not currently used - it may be re-added to the docs if it
        is used in the future:]

        Suppose that there is an effective probability :math:`p_1` (for e.g. X
        on some qubit) and we add a new, independent source of an additional X with
        probability :math:`q` on the same qubit. Then, the new effective
        probability :math:`p_\text{new}` is

        :math:`p_\text{new} = (1 - q) p_1 + q (1 - p_1)`

        This allows us to combine an arbitrary number of independent X error
        sources (e.g. Pauli X/Z error, Pauli Y error, depolarize and erasure noise).

    .. automethod:: __init__
    """
    # Note for extending this class:
    #
    # Error models should subclass this interface definition and override the
    # constructor if necessary. In its constructor, an error model can require
    # arbitrary parameters. A more sophisticated error model may e.g. require
    # a ``CodeLattice`` instance to consider nearest-neighbour relations
    # between qubits. For non-uniform error probabilities, e.g.
    # :meth:`get_edge_probab` also has to be overridden.
    #
    # An error model may return approximate edge probabilities or edge weights
    # for the syndrome graph. In any case, an error model must document how it
    # computes probabilities and/or weights.

    #: List of names of supported errors
    #:
    #: Erasure errors are accepted, but it is not necessary to consider them in weight
    #: computations.
    supported_errors: tuple[str, ...] = ("erasure", "pauli", "measurement")
    #: QEC code
    code: codes.LatticeCode
    #: Error data (for computing weights)
    errordata: ErrorDataDict
    #: Effective Pauli X error probability
    p_x_eff: np.ndarray
    #: Effective Pauli Z error probability
    p_z_eff: np.ndarray
    #: This flag decides whether erasure error probabilities contribute to weights.
    #: This should usually be False since erasures are heralded.
    erasure_weights: bool = False

    def __init__(self, code: codes.LatticeCode, errordata: ErrorDataDict):
        """Compute decoding weights.

        Args:
            code: The code
            errordata: Information about error probabilities

        .. note::

           If you get array size mismatch errors, it means that your probability arrays
           do not have the correct size.
        """
        ErrorData().check_against_code(code, errordata)

        if invalid := set(errordata).difference(self.supported_errors):
            raise ValueError(f"Unknown errors: {invalid!r}")
        self.code = code
        self.errordata = errordata
        self.n_qubits = code.n_data_qubits
        self.n_stabgens = code.n_stabgens
        self.n_rounds = code.n_rounds
        self.p_x_eff = np.zeros([self.n_qubits])
        self.p_z_eff = np.zeros([self.n_qubits])
        self.p_meas = np.zeros([self.n_stabgens])
        self._compute_probabs()

    def _compute_probabs(self):
        """Compute p_x_eff, p_z_eff and p_meas."""
        err = self.errordata
        if "pauli" in err:
            self._p_eff_pauli_1()
        if self.erasure_weights and "erasure" in err:
            self._p_eff_erasure()
        if "measurement" in err:
            self._p_meas()

    def _p_meas(self):
        """Set measurement error probabilities according to input data."""
        for equbit_idx, params in self.errordata["measurement"].items():
            vertex = self.code.lattice.equbits[equbit_idx]
            assert isinstance(vertex, latticebase.StabGenVertex)
            p = params.get("p", 0.0)
            self.p_meas[vertex.stabgen_idx] = p

    def _p_eff_pauli_1(self):
        """Update p_x_eff and p_z_eff according to Pauli error channel."""
        q_x = np.zeros_like(self.p_x_eff)
        q_z = np.zeros_like(self.p_z_eff)
        for equbit_idx, params in self.errordata["pauli"].items():
            vertex = self.code.lattice.equbits[equbit_idx]
            assert isinstance(vertex, latticebase.DataVertex)
            p_x = params.get("x", 0.0)
            p_y = params.get("y", 0.0)
            p_z = params.get("z", 0.0)
            q_x[vertex.dataqubit_idx] = p_x + p_y
            q_z[vertex.dataqubit_idx] = p_y + p_z
        self._update_p_eff(q_x, q_z)

    def _p_eff_erasure(self):
        """Update p_x_eff and p_z_eff according to erasure error channel."""
        q_x = np.zeros_like(self.p_x_eff)
        q_z = np.zeros_like(self.p_z_eff)
        for equbit_idx, params in self.errordata["erasure"].items():
            vertex = self.code.lattice.equbits[equbit_idx]
            assert isinstance(vertex, latticebase.DataVertex)
            p = params.get("p", 0.0)
            q_x[vertex.dataqubit_idx] = p / 2  # Erasure causes X half the time
            q_z[vertex.dataqubit_idx] = p / 2  # Erasure causes Z half the time
        self._update_p_eff(q_x, q_z)

    def _update_p_eff(self, q_x: np.ndarray, q_z: np.ndarray):
        """Update effective qubit error probabilities with additional indep. source."""
        self.p_x_eff = (1 - q_x) * self.p_x_eff + q_x * (1 - self.p_x_eff)
        self.p_z_eff = (1 - q_z) * self.p_z_eff + q_z * (1 - self.p_z_eff)

    def get_edge_probab(self) -> dict[str, np.ndarray]:
        """Compute error probabilities for all syndrome graph edges."""
        return dict(pauli_x=self.p_x_eff, pauli_z=self.p_z_eff, measurement=self.p_meas)

    def get_edge_weights(self) -> dict[str, np.ndarray]:
        r"""Convert syndrome graph edge probabilities to edge weights.

        Weights are computed as :math:`w = -\ln\left(\frac{p}{1-p}\right)`
        where :math:`p` is the probability that an error occurs and :math:`w`
        is its weight.

        .. note::

           Weights computed here are between -inf and +inf (both values included).

           For weights supported by decoders, see
           :func:`plaquette.decoders.decoderbase.check_weights_sane`.

        .. todo::

           Find a better reference for the formula used to compute weights.

           This formula appears in, among other places,
           :cite:`pattison_improved_2021`.
        """
        # Allow division by zero and log(0).
        with np.errstate(divide="ignore"):
            return {
                name: -np.log(p / (1 - p)) for name, p in self.get_edge_probab().items()
            }
