"""Unit-tests for the :mod:`plaquette.decoders.decoderbase` module."""
import pytest as pt

from plaquette.codes import LatticeCode
from plaquette.decoders import UnionFindDecoder
from plaquette.errors import ErrorDataDict


def test_zero_valued_pauli_errors_turned_into_small_errors():
    """Test guard against infinite-weight edges.

    Make sure that passing values of exactly 0, or no value at all triggers a
    warning and a change in value of the error.
    """
    qed: ErrorDataDict = {"pauli": {0: {"x": 0.1, "y": 0.0}}}

    with pt.warns(UserWarning, match=r"Pauli [XYZ] error on data qubit \d+"):
        d = UnionFindDecoder.from_code(LatticeCode.make_planar(1, 3), qed)
        assert d.synmgr.errordata["pauli"][0]["y"] == 1e-15
        assert d.synmgr.errordata["pauli"][0]["z"] == 1e-15
