# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest as pt

from plaquette.codes import StabilizerCode, latticeinstances


@pt.mark.parametrize(
    "code_cls, size",
    [
        (latticeinstances.RepetitionCodeLattice, 1),
        (latticeinstances.RepetitionCodeLattice, 3),
        (latticeinstances.RepetitionCodeLattice, 5),
        (latticeinstances.RepetitionCodeLattice, 7),
        (latticeinstances.PlanarCodeLattice, 1),
        (latticeinstances.PlanarCodeLattice, 2),
        (latticeinstances.PlanarCodeLattice, 3),
        (latticeinstances.PlanarCodeLattice, 7),
        (latticeinstances.RotatedPlanarCodeLattice, 1),
        (latticeinstances.RotatedPlanarCodeLattice, 2),
        (latticeinstances.RotatedPlanarCodeLattice, 3),
        (latticeinstances.RotatedPlanarCodeLattice, 7),
    ],
)
def test_codelattice_stab(code_cls, size):
    """Call StabilizerCode.check() for all implemented codes."""
    code = code_cls(size)
    stab = StabilizerCode.from_codelattice(code)
    stab.check(rank=True)


@pt.mark.parametrize(
    "code_cls",
    [latticeinstances.FiveQubitCodeLattice, latticeinstances.ShorCodeLattice],
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
    assert str(stab.stabgens).split("\n") == code.def_stabgens
    assert str(stab.logical_ops).split("\n") == code.def_logical_ops


@pt.mark.parametrize(
    "from_lattice, direct",
    [
        (
            StabilizerCode.from_codelattice(latticeinstances.FiveQubitCodeLattice()),
            StabilizerCode(
                [
                    np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0]),
                    np.array([0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0]),
                    np.array([1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0]),
                    np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0]),
                ],
                [
                    np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]),
                ],
            ),
        ),
        (
            StabilizerCode.from_codelattice(latticeinstances.RepetitionCodeLattice(5)),
            StabilizerCode(
                [
                    np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]),
                    np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]),
                ],
                [
                    np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]),
                ],
            ),
        ),
    ],
)
def test_from_codelattice(from_lattice: StabilizerCode, direct: StabilizerCode):
    assert from_lattice == direct
