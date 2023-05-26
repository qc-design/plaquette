# Copyright 2023, It'sQ GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Basic definitions for decoders."""

from __future__ import annotations

import abc
import warnings
from typing import NoReturn, Optional, Sequence, Type

import numpy as np

from plaquette import codes, syngraph
from plaquette.errors import ErrorDataDict, SinglePauliChannelErrorValueDict
from plaquette.pauli import Tableau, commutator_sign, count_qubits


def _assert_never(value: NoReturn) -> NoReturn:
    """Assert for unreachable code.

    For more information, see ``assert_never`` in the `mypy docs
    <https://mypy.readthedocs.io/en/stable/literal_types.html#exhaustiveness-checking>`_.
    """
    # This also works at runtime as well
    raise AssertionError(f"This code should never be reached, got: {value}")


def check_weights_sane(weights: Optional[np.ndarray], stacklevel: int = 2) -> bool:
    r"""Check weights for possible problems.

    This function is used by decoders to check whether weights look generally
    reasonable.

    Raise errors for NaN and negative weights. Raise warnings for zero and
    :math:`+\infty` weights.

    Rationale:

    * Negative weights are not supported yet.
    * :math:`+\infty` weights are handled by
      :class:`plaquette.syngraph.SyndromeGraph`.
      Therefore, they will not be passed to a decoder and they will not be seen by this
      function.
    * Some decoders may not handle zero weights correctly.

    Args:
        weights: The weights
        stacklevel: Stack level for any warnings raised

    .. todo::

       Shreya: Remove these warnings once we have many tests.
    """
    sane = True
    if weights is None:
        return sane
    # Hard errors can be moved to decoders in the future (if some start supporting
    # negative weights).
    if np.isnan(weights).any():
        raise ValueError("Some edge weights are NaN. This is not supported.")
    if (weights < 0).any():
        raise ValueError("Negative edge weights are currently not supported")
    if (weights == 0).any():
        sane = False
        warnings.warn(
            "Some edge weights are zero. Some decoders may not handle this correctly.",
            stacklevel=stacklevel,
        )
    if np.isinf(weights).any():
        sane = False
        warnings.warn(
            "Some edge weights are +inf. Some decoders may not handle this correctly.",
            stacklevel=stacklevel,
        )
    return sane


def check_success(
    code: codes.StabilizerCode,
    correction: Sequence[Tableau],
    logical_op_toggle: np.ndarray,
    logical_ops: str | Sequence[int] | None = None,
):
    """Compare measured with predicted logical operator results.

    This function compares measured logical operators with the prediction from the
    decoder. If they match, this test of QEC can be considered successful.

    Args:
        code: Code definition containing logical operators
        correction: Correction Pauli frame from the decoder
        logical_op_toggle:
            XOR between logical op. measurement result before and after QEC
        logical_ops: Specify which logical operators were measured (default: all)

    This function does the following:

    * Flip logical operators according to the correction Pauli frame from the decoder.
    * Check whether the predicted signs of logical operators agree with measurement
      results.

    This function can be used for (assuming ``k`` logical qubits):

    * Measurement results from experiment (``k`` logical ops. measured)
    * Measurement results from circuit simulation (``k`` logical ops. simulated)
    * Measurement results from code space simulation (``2*k`` logical ops. simulated)

    Example:
        .. code::

            from plaquette import codes, circuit, decoders, errors, simulator, utils

            code = codes.PlanarCode(n_rounds=1, size=3)
            errordata = errors.ErrorData()
            errordata.update_all(code, {"pauli_1": {"p_x": 0.01, "p_z": 0.08}})

            circ = circuit.generate_qec_circuit(code, errordata, logical_ops="Z")
            sim = simulator.CircuitSimulator(circ)
            sample = sim.get_sample()

            results = utils.split_measurement_results(code, sample)

            dec = decoders.UnionFindDecoder.from_code(code, errordata, weighted=False)
            correction = dec.decode(results.qubits_erased, results.syndrome)
            success = decoders.check_success(
                code, correction, results.logical_op_toggle, logical_ops="Z"
            )

        For an explanation, see :doc:`/quickstart`.
    """
    log_ops = code.logical_ops
    if logical_ops is not None:
        log_ops = [log_ops[i] for i in list(code.logical_ops_to_indices(logical_ops))]
    assert len(log_ops) == len(logical_op_toggle)
    assert count_qubits(correction)[0] == code.n_data_qubits
    decoder_prediction = (commutator_sign(log_ops, correction)).ravel()
    return (logical_op_toggle == decoder_prediction).all()


class DecoderBackendInterface(metaclass=abc.ABCMeta):
    """Low-level interface to decoders (abstract class).

    A decoder interacts with a :class:`plaquette.syngraph.SyndromeGraphComponent`,
    which represents a connected component of the syndrome graph.

    .. automethod:: __init__
    """

    @abc.abstractmethod
    def set_syngraph(self, sgraph: syngraph.SyndromeGraphComponent):
        """Update syndrome graph and weights.

        The decoder should keep a reference to the syndrome graph component because
        it needs to retrieve updated weights, erasure and syndrome from there.
        """

    @abc.abstractmethod
    def update_weights(self):
        """Update weights from syndrome graph.

        Update weights from the syndrome graph component last supplied to
        :meth:`set_syngraph`.
        """

    @abc.abstractmethod
    def decode(self):
        """Decode erasure and syndrome.

        Erasure and syndrome information are retrieved from the syndrome graph
        component last supplied to :meth:`set_syngraph`.
        Specifically, a decoder is expected to read
        :attr:`.SyndromeGraphComponent.syndrome` and
        :attr:`.SyndromeGraphComponent.edge_erased`.

        Returns:
            Nothing.

            The decoder is expected to call
            :meth:`.SyndromeGraphComponent.set_edge_decoder_results` and pass decoding
            results to that function.
        """


class ComponentDecoder:
    """Handle decoder instances for each connected component of the syndrome graph.

    .. automethod:: __init__
    """

    #: Type of decoder to be used
    decoder_cls: Type[DecoderBackendInterface]
    #: List of decoders (one for each connected component of the syndrome graph)
    decoders: Sequence[DecoderBackendInterface]
    #: The syndrome graph used (decoder results are retrieved from it)
    sgraph: Optional[syngraph.SyndromeGraph] = None

    def __init__(self, decoder_cls: Type[DecoderBackendInterface], split: bool = True):
        """Create new instance.

        Args:
            decoder_cls: Type of decoder which should be used
            split:
                Whether decoding connected components should be done separately.
                Currently, only ``True`` is supported.
        """
        assert split, "split=False is not implemented yet"
        self.decoder_cls = decoder_cls
        self.decoders = []

    def set_syngraph(self, sgraph: syngraph.SyndromeGraph):
        """Set new syndrome graph.

        This class keeps a reference to the supplied syndrome graph. It retrieves
        updated weights, erasure and syndrome as well as decoding results from the
        syndrome graph.
        """
        self.sgraph = sgraph
        self.decoders = []
        for comp in sgraph.components_with_edges:
            self.decoders.append(self.decoder_cls())
            self.decoders[-1].set_syngraph(comp)

    def update_weights(self):
        """Update weights from previously supplied syndrome graph."""
        if self.sgraph is None:
            raise ValueError("No syndrome graph set")
        for dec in self.decoders:
            dec.update_weights()

    def _check_decodable(self):
        """Basic sanity checks prior to decoding.

        For details of what is checked, see the exceptions raised by the code.
        """
        # TODO Should this check remain here? Should it be moved to
        #  SyndromeGraph._set_syndrome()?
        for comp in self.sgraph.components:
            # A component with zero edges must have precisely one vertex.
            # (NB: A component with one vertex can have one or more dangling edges.)
            if len(comp.edges) == 0:
                assert comp.n_vertices == 1
                assert comp.syndrome.shape == (1,)
                if comp.syndrome[0]:
                    raise ValueError(
                        "Cannot decode an isolated vertex with syndrome bit"
                    )
            else:
                # Component has at least one edge.
                if (not comp.has_dangling_edges) and (comp.syndrome.sum() % 2) == 1:
                    raise ValueError(
                        "Cannot decode a connected component with no dangling edges "
                        "and odd syndrome parity."
                    )

    def decode(self, qubit_erased: Optional[np.ndarray], syndrome: np.ndarray):
        """Decode erasure and syndrome.

        This function takes the same arguments as :meth:`DecoderInterface.decode`.

        Decoding results can be retrieved from the last set syndrome graph.
        """
        if self.sgraph is None:
            raise ValueError("No syndrome graph set")
        self.sgraph.set_erasure(qubit_erased)
        self.sgraph.set_syndrome(syndrome)
        self._check_decodable()
        for decoder in self.decoders:
            decoder.decode()


class DecoderInterface:
    """Base class for high-level interface to decoders.

    This class defines the user-visible high-level interface to all decoders.

    .. automethod:: __init__
    """

    #: Decoder class (to be defined by subclass)
    _decoder_cls: Type[DecoderBackendInterface]
    #: List of decoders for connected components of the syndrome graph
    decoder: ComponentDecoder
    #: Syndrome graph manager. Can provide updates which depend on erasure information.
    synmgr: syngraph.SyndromeGraphManager

    def __init__(self, synmgr: syngraph.SyndromeGraphManager):
        """Create a new instance of a decoder.

        If you do not want to supply a custom syndrome graph manager, you can use
        :meth:`from_code` to construct an instance.
        """
        self.decoder = ComponentDecoder(self._decoder_cls, split=True)
        self.synmgr = synmgr
        self.decoder.set_syngraph(self.synmgr.sgraph)

    @classmethod
    def from_code(
        cls,
        code: codes.LatticeCode,
        errordata: Optional[ErrorDataDict],
        *,
        weighted: Optional[bool] = None,
        weights: Optional[dict] = None,
    ):
        """Initialize decoder for use with a given code and error data.

        Args:
            code: The error correction code
            errordata: The error data (for computing weights).
            weighted:
                * ``True``: Weights are used.
                * ``False``: Weights are not used.
                * ``None``: Weights are used if ``errordata`` or ``weights`` is given.
            weights:
                * ``None``: Weights are derived from ``errordata``.
                * ``dict``: Used as keyword arguments to :meth:`set_edge_weights()
                  <plaquette.syngraph.SyndromeGraph.set_edge_weights>`.
                  ``errordata`` must be ``None``.

        Notes:
            Passing an :class:`ErrorDataDict` where some of the Pauli errors
            have exactly zero probability has some unintended consequences. To
            avoid these corner cases, this method automatically scans the
            error data and replaces or adds all zeros with ``1e-15``.

            This means that specifying an error model like so::

                ed = {"pauli": {0: {"x": 0.01}}}

            will be turned into::

                ed = {"pauli": {0: {"x": 0.01, "y": 1e-15, "z": 1e-15}, 1: {...}}}

            where also all *unspecified* qubits will have these additional but
            tiny errors applied.

            .. important:: This behavior will be removed in a future version of
                :mod:`plaquette`, and exactly 0-valued probabilities will be
                possible.
        """
        # Avoid using 0 as value for Pauli errors and instead use a tiny value
        _default_pauli_errors: SinglePauliChannelErrorValueDict = {
            "x": 1e-15,
            "y": 1e-15,
            "z": 1e-15,
        }
        if errordata is not None:
            if "pauli" not in errordata:
                errordata["pauli"] = {
                    q: _default_pauli_errors
                    for q in range(len(code.lattice.dataqubits))
                }
            else:
                erasures = "erasure" in errordata
                for q in range(len(code.lattice.dataqubits)):
                    # We give a default value to all qubits, regardless of
                    # whether they were originally specified or not
                    if not errordata["pauli"].get(q):
                        errordata["pauli"][q] = _default_pauli_errors
                    else:
                        # If the qubit has some errors specified, we need to
                        # modify/add only the wrong/missing ones
                        for k in "xyz":
                            if not errordata["pauli"][q].get(k, 0):  # type: ignore
                                warnings.warn(
                                    f"Pauli {k.upper()} error on data qubit {q} set to "
                                    "1e-15 instead of 0. This will change in a future "
                                    "version of plaquette.",
                                    stacklevel=2,
                                )
                                errordata["pauli"][q][k] = 1e-15  # type: ignore
                    if erasures and not errordata["erasure"].get(q):
                        errordata["erasure"][q] = {"p": 0}
        else:
            errordata: ErrorDataDict = {  # type: ignore[no-redef]
                "pauli": {
                    q: _default_pauli_errors
                    for q in range(len(code.lattice.dataqubits))
                }
            }

        synmgr = syngraph.SyndromeGraphManager(
            code, errordata, weighted=weighted, weights=weights
        )
        return cls(synmgr)

    def _update_syngraph(self, erasure: Optional[np.ndarray], syndrome: np.ndarray):
        """Update syngraph and/or weights before decoding if necessary.

        This method is always called by :meth:`decode` before decoding starts.
        """
        update = self.synmgr.update_for_decoding(erasure, syndrome)
        match update:
            case syngraph.SyndromeGraphUpdate.Nothing:
                pass
            case syngraph.SyndromeGraphUpdate.Weights:
                self.decoder.update_weights()
            case syngraph.SyndromeGraphUpdate.GraphAndWeights:
                self.decoder.set_syngraph(self.synmgr.sgraph)
            case _ as unhandled_value:
                _assert_never(unhandled_value)

    def decode(
        self, qubit_erased: Optional[np.ndarray], syndrome: np.ndarray
    ) -> Tableau:
        """Decode erasure and syndrome.

        Args:
            qubit_erased: Passed to
                :meth:`plaquette.syngraph.SyndromeGraph.set_erasure`.
            syndrome: Passed to
                :meth:`plaquette.syngraph.SyndromeGraph.set_syndrome`.

        Returns:
            Decoding result as Pauli frame update (from
            :meth:`plaquette.syngraph.SyndromeGraph.result_as_pauli_frame`.
        """
        self._update_syngraph(qubit_erased, syndrome)
        self.decoder.decode(qubit_erased, syndrome)
        self.synmgr.sgraph.update_decoder_results()
        # TODO: is this always a single pauli operator? Can it ever be a list of
        #  operators?
        return self.synmgr.sgraph.result_as_pauli_frame()
