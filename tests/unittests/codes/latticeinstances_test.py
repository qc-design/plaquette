# Copyright 2023, It'sQ GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
import pytest as pt

from plaquette.codes import StabilizerCode
from plaquette.codes import latticeinstances as lattice
from plaquette.pauli import pauli_to_string


@pt.mark.parametrize(
    "code_cls, size",
    [
        # (lattice.RepetitionCodeLattice, 1),
        (lattice.RepetitionCodeLattice, 3),
        (lattice.RepetitionCodeLattice, 5),
        (lattice.RepetitionCodeLattice, 7),
        # (lattice.PlanarCodeLattice, 1),
        (lattice.PlanarCodeLattice, 2),
        (lattice.PlanarCodeLattice, 3),
        (lattice.PlanarCodeLattice, 7),
        # (lattice.RotatedPlanarCodeLattice, 1),
        (lattice.RotatedPlanarCodeLattice, 2),
        (lattice.RotatedPlanarCodeLattice, 3),
        (lattice.RotatedPlanarCodeLattice, 7),
    ],
)
def test_codelattice_stab(code_cls, size):
    """Call StabilizerCode.check() for all implemented codes."""
    code = code_cls(size)
    stab = StabilizerCode.from_codelattice(code)
    stab.check(rank=True)


@pt.mark.parametrize(
    "code_cls", [lattice.FiveQubitCodeLattice, lattice.ShorCodeLattice]
)
def test_codelattice_from_stab_str(code_cls):
    """Verify string -> code lattice -> stabilizer -> string conversion.

    Some codes are defined from strings describing the stabilizers and
    logicals. For these codes, we can directly check that creating the
    corresponding stabilizer operators and converting back to a string produces
    the correct results.
    """
    code = code_cls()
    stab = StabilizerCode.from_codelattice(code)
    stab.check(rank=True)
    # The [:1] removes the positive sign
    assert [pauli_to_string(sg)[1:] for sg in stab.stabilisers] == code.def_stabgens
    assert [pauli_to_string(lo)[1:] for lo in stab.logical_ops] == code.def_logical_ops
