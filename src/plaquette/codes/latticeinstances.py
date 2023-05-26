# Copyright 2023, It'sQ GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Implementation of several QEC codes."""

from __future__ import annotations

from typing import Sequence

from plaquette.codes import latticebase


class FiveQubitCodeLattice(latticebase.CodeLatticeFromStab):
    """The five-qubit code.

    References: :cite:`laflamme_perfect_1996` and :cite:`bennett_mixed-state_1996`.
    This implementation uses the stabilizer convention from
    :cite:`nielsen_quantum_2010`.

    .. automethod:: __init__
    """

    n_data_qubits = 5
    def_stabgens = "XZZXI IXZZX XIXZZ ZXIXZ".split()
    def_logical_ops = "XXXXX ZZZZZ".split()


class ShorCodeLattice(latticebase.CodeLatticeFromStab):
    """The Shor code on nine qubits.

    Reference: :cite:`shor_scheme_1995`.

    .. automethod:: __init__
    """

    n_data_qubits = 9
    def_stabgens = """
    ZZIIIIIII
    IZZIIIIII
    XXXXXXIII
    IIIZZIIII
    IIIIZZIII
    IIIXXXXXX
    IIIIIIZZI
    IIIIIIIZZ
    """.split()
    def_logical_ops = [9 * "Z", 9 * "X"]


class RepetitionCodeLattice(latticebase.CodeLattice):
    """The repetition code.

    Also known as bit-flip code because it can detect bit flips (X)
    but not phase flips (Z).

    Reference: :cite:`nielsen_quantum_2010` (see three-qubit bit-flip code).

    .. automethod:: __init__
    """

    def __init__(self, size: int):  # noqa: D107
        if (size % 2) != 1:
            raise ValueError(f"Repetition code size must be odd (got {size=})")
        #: Length parameter of the planar code
        self.size: int = size
        super().__init__((2 * size, 2))
        for i in range(0, 2 * size, 2):
            self.add_data((i, 0))
        for i in range(1, 2 * size - 2, 2):
            self.add_stabgen((i, 0), latticebase.StabGroup.A)
            for di in (-1, 1):
                self.add_edge(
                    self.lattice[i + di, 0], self.lattice[i, 0], latticebase.Pauli.Z
                )
        X = self.add_logical((size - 1, 1), "X")
        for i in range(0, 2 * size, 2):
            self.add_edge(self.lattice[i, 0], X, latticebase.Pauli.X)
        Z = self.add_logical((2 * size - 1, 0), "Z")
        self.add_edge(self.lattice[2 * size - 2, 0], Z, latticebase.Pauli.Z)
        self.assign_indices()


class ToricCodeLattice(latticebase.CodeLattice):
    """The toric code.

    References: :cite:`kitaev_quantum_1997`.
    """

    def __init__(
        self,
        size: int,
        edge_order: Sequence[tuple[int, int]] = ((0, 1), (-1, 0), (1, 0), (0, -1)),
    ):
        """Initialise the toric code's lattice.

        Default ``edge_order`` is NWES aka "zig-zag". This order is used e.g. in
        https://journals.aps.org/pra/abstract/10.1103/PhysRevA.86.032324.
        """
        #: Length parameter of the toric code
        self.size: int = size
        super().__init__((2 * size + 2, 2 * size + 2))
        for i in range(2 * self.size):
            for j in range(2 * self.size):
                i_ = i + 1
                j_ = j + 1
                pos = (i_, j_)
                im, jm = (i_ % 2), (j_ % 2)
                if im != jm:
                    self.add_data(pos)
                elif jm == 0:
                    self.add_stabgen(pos, latticebase.StabGroup.A)
                else:
                    self.add_stabgen(pos, latticebase.StabGroup.B)
        max_index = 2 * self.size
        ops = (
            latticebase.Pauli.X,
            latticebase.Pauli.Z,
            latticebase.Pauli.X,
            latticebase.Pauli.Z,
        )
        for stab in self.vertices:
            if not isinstance(stab, latticebase.StabGenVertex):
                continue
            i, j = stab.pos
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                qi, qj = i + di, j + dj
                if qi == 0:
                    qi = max_index
                if qj == 0:
                    qj = max_index
                if qi > max_index:
                    qi = 1
                if qj > max_index:
                    qj = 1
                self.add_edge(self.lattice[qi, qj], stab, ops[int(stab.group) - 1])
        X1 = self.add_logical((0, self.size), "X")
        Z1 = self.add_logical((self.size, max_index + 1), "Z")
        X2 = self.add_logical((self.size, 0), "X")
        Z2 = self.add_logical((max_index + 1, self.size), "Z")
        for i in range(self.size):
            self.add_edge(self.lattice[1, 2 * i + 2], X1, latticebase.Pauli.X)
            self.add_edge(self.lattice[2 * i + 1, max_index], Z1, latticebase.Pauli.Z)
            self.add_edge(self.lattice[2 * i + 2, 1], X2, latticebase.Pauli.X)
            self.add_edge(self.lattice[max_index, 2 * i + 1], Z2, latticebase.Pauli.Z)
        self.assign_stab_edge_order(edge_order)
        self.assign_indices()


class PlanarCodeLattice(latticebase.CodeLattice):
    """The planar code.

    References: :cite:`bravyi_quantum_1998`, :cite:`freedman_projective_2001`.

    .. automethod:: __init__
    """

    def __init__(
        self,
        size: int,
        edge_order: Sequence[tuple[int, int]] = ((0, 1), (-1, 0), (1, 0), (0, -1)),
    ):
        """Initialise the planar code's lattice.

        Default ``edge_order`` is NWES aka "zig-zag". This order is used e.g. in
        https://journals.aps.org/pra/abstract/10.1103/PhysRevA.86.032324.
        """
        #: Length parameter of the planar code
        self.size: int = size
        super().__init__((2 * size, 2 * size))
        for i in range(2 * self.size - 1):
            for j in range(2 * self.size - 1):
                pos = (i, j)
                im, jm = (i % 2), (j % 2)
                if im == jm:
                    self.add_data(pos)
                elif jm == 0:
                    self.add_stabgen(pos, latticebase.StabGroup.A)
                else:
                    self.add_stabgen(pos, latticebase.StabGroup.B)
        max_index = 2 * self.size - 2
        ops = (latticebase.Pauli.X, latticebase.Pauli.Z)
        for stab in self.vertices:
            if not isinstance(stab, latticebase.StabGenVertex):
                continue
            i, j = stab.pos
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                qi, qj = i + di, j + dj
                if qi < 0 or qj < 0 or qi > max_index or qj > max_index:
                    # i.e. if qubit is outside the lattice
                    continue
                self.add_edge(self.lattice[qi, qj], stab, ops[int(stab.group) - 1])
        X = self.add_logical((max_index + 1, self.size - 1), "X")
        Z = self.add_logical((self.size - 1, max_index + 1), "Z")
        for i in range(self.size):
            self.add_edge(self.lattice[max_index, 2 * i], X, latticebase.Pauli.X)
            self.add_edge(self.lattice[2 * i, max_index], Z, latticebase.Pauli.Z)
        self.assign_stab_edge_order(edge_order)
        self.assign_indices()


class RotatedPlanarCodeLattice(latticebase.CodeLattice):
    """Rotated (aka Wen) planar code.

    Reference: E.g. :cite:`horsman_surface_2012`. See also :cite:`wen_quantum_2003`.

    .. automethod:: __init__
    """

    def __init__(
        self,
        size: int,
        xzzx: bool = False,
        edge_order: Sequence[tuple[int, int]] = ((1, 1), (-1, 1), (1, -1), (-1, -1)),
    ):
        """Initialise the rotated planar code's lattice.

        Default ``edge_order`` is (N-E)(N-W)(S-E)(S-W) for "black tiles" and
        (N-E)(S-E)(N-W)(S-W) for "white tiles". This order is used e.g. in
        https://arxiv.org/pdf/2208.01178.
        """
        #: Length parameter of the planar code
        self.size: int = size
        super().__init__((2 * size + 1, 2 * size + 1))
        for i in range(1, 2 * size, 2):
            for j in range(1, 2 * size, 2):
                self.add_data((i, j))
        for i in range(2, 2 * size, 2):
            for j in range(2, 2 * size, 2):
                if i // 2 % 2 == j // 2 % 2:
                    self.add_stabgen((i, j), latticebase.StabGroup.A)
                else:
                    self.add_stabgen((i, j), latticebase.StabGroup.B)
        odd = size % 2
        for i in range(4, 2 * size, 4):
            self.add_stabgen((0, i), latticebase.StabGroup.A)
            self.add_stabgen((2 * size, i - odd * 2), latticebase.StabGroup.A)
        for i in range(2, 2 * size, 4):
            self.add_stabgen((i, 0), latticebase.StabGroup.B)
            self.add_stabgen((i + odd * 2, 2 * size), latticebase.StabGroup.B)
        data_min = 1
        data_max = 2 * size - 1
        ops = X, Z = (latticebase.Pauli.X, latticebase.Pauli.Z)
        for stab in self.vertices:
            if not isinstance(stab, latticebase.StabGenVertex):
                continue
            i, j = stab.pos
            for p, di, dj in [(X, 1, 1), (Z, 1, -1), (Z, -1, 1), (X, -1, -1)]:
                qi, qj = i + di, j + dj
                if not xzzx:
                    p = ops[stab.group - 1]
                if data_min <= qi <= data_max and data_min <= qj <= data_max:
                    self.add_edge(self.lattice[qi, qj], stab, p)
        mid = self.size + odd - 1
        vX = self.add_logical((mid, 0), "X")
        vZ = self.add_logical((0, mid), "Z")
        for i in range(1, 2 * size, 2):
            if (not xzzx) or i // 2 % 2 == 0:
                X, Z = latticebase.Pauli.X, latticebase.Pauli.Z
            else:
                X, Z = latticebase.Pauli.Z, latticebase.Pauli.X
            self.add_edge(self.lattice[i, 1], vX, X)
            self.add_edge(self.lattice[1, i], vZ, Z)
        self.assign_stab_edge_order(edge_order)
        self.inverted_order()
        self.assign_indices()

    def inverted_order(self):
        """Invert gate order for different types of tiles along a diagonal.

        In most papers where the gate order of a rotated planar code is displayed, the
        order of the top-left dataqubit and the bottom-right is inverted for dark tile
        and light tile operators. This method handles this order inversion.
        """
        for edge in self.edges:
            op = edge.op
            if not isinstance(op, latticebase.StabGenVertex):
                continue
            x, y = op.pos
            if (x + y) % 4 != 0:
                continue
            if edge.measurement_time_step == 1:
                edge.measurement_time_step = 2
                continue
            if edge.measurement_time_step == 2:
                edge.measurement_time_step = 1
