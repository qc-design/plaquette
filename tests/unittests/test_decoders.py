# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit-test for the :mod:`.decoders` module."""
import typing as t

import numpy as np
import pytest as pt

from plaquette import codes as codes
from plaquette import decoders, pauli


@pt.fixture
def ssc_code():
    """Default code against which all decoders are tested."""
    return codes.Code.make_planar(3)


@pt.fixture
def results_single_round(ssc_code: codes.Code):
    """Fake generated syndrome and erasure info.

    This avoids running a circuit for no reason. It mimics a Z error
    acting on qubits 2 and an X error on qubit 5. This means that
    it breaks both logical operators' parities.
    """
    s = np.zeros(ssc_code.num_stabilizers, dtype=bool)
    s[[2, 8]] = True
    return s, []


@pt.fixture
def results_multi_round(ssc_code, results_single_round):
    """Fake generated syndrome and erasure info.

    Replicates :func:`results_single_round` for **two** rounds.
    """
    s, e = results_single_round
    return np.concatenate((s, s)), np.concatenate((e, e))


@pt.mark.parametrize(
    "dec_class", [decoders.FusionBlossomDecoder, decoders.UnionFindDecoder]
)
def test_decoders_base_props(
    ssc_code: codes.Code, dec_class: t.Type[decoders.AbstractDecoder]
):
    """Validate common properties of the ``AbstractDecoder`` class."""
    dec = dec_class(ssc_code, {}, 2)  # use two rounds to check 3D properties
    # Make sure no edge touches data qubits
    dg = dec._decoding_graph
    tg = ssc_code._tanner_graph
    for e in range(dg.get_num_edges()):
        dg_a, dg_b = dg.get_vertices_connected_by_edge(e)
        # ignore virtual edges from decoding graph. We also need to remove the
        # rounds offset, because the original code does not have these
        # vertices and would rais an IndexError
        if dg_a not in dec._virtual_ancillas:
            assert (
                tg.nodes_data[dg_a % dec._nodes_per_round].type != codes.QubitType.data
            )
        if dg_b not in dec._virtual_ancillas:
            assert (
                tg.nodes_data[dg_b % dec._nodes_per_round].type != codes.QubitType.data
            )


@pt.mark.parametrize(
    "dec_class",
    (
        decoders.FusionBlossomDecoder,
        decoders.PyMatchingDecoder,
        decoders.UnionFindDecoder,
    ),
)
def test_decoders_correction(
    ssc_code: codes.Code,
    results_single_round: tuple[t.Sequence[bool], t.Sequence[bool]],
    results_multi_round: tuple[t.Sequence[bool], t.Sequence[bool]],
    dec_class: t.Type[decoders.AbstractDecoder],
):
    """Test correction operator anti-commuting with "broken" logical operators."""
    syn, er = results_single_round
    dec = dec_class(ssc_code, {}, 1)
    correction = dec.decode(syn, er)
    assert pauli.commutator_sign(correction, ssc_code.logical_ops[0]) == 1
    assert pauli.commutator_sign(correction, ssc_code.logical_ops[1]) == 1

    syn, er = results_multi_round
    dec = dec_class(ssc_code, {}, 2)
    correction_multi = dec.decode(syn, er)
    assert pauli.commutator_sign(correction_multi, ssc_code.logical_ops[0]) == 1
    assert pauli.commutator_sign(correction_multi, ssc_code.logical_ops[1]) == 1
