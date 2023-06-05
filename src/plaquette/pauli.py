# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Definition of a Pauli operator acting on multiple qubits and associated functions.

This module implements the basic math foundations for the rest of the package, and it's
independent of anything else in the package itself *except* for the random number
generator necessary for the measurement functions.

Notes:
    All calculations in ``plaquette`` are based on the tableau formalism introduced
    by :cite:`aaronson_improved_2004`. The implementation of this is based on
    ``numpy`` for convenience of development and familiarity with the wider
    scientific ecosystem, but it comes with some catches that one need to be
    aware about.

    Numpy arrays, when assigned to multiple Python variables, are not copied.
    You only get a new reference to the same piece of memory that stores you
    data. **Most** functions in this module take an array as input and return
    an array as output. Commonly, functions return a *modified* array that was
    given as the input. Please be aware that this means that in the following
    snippet:

    >>> zs = zero_state(10)
    >>> state_after_measurement, measurement_result = measure_z_base(zs, 0)
    >>> np.all(zs == state_after_measurement)
    True

    ``zs`` and ``state_after_measurement`` **point to the same data**. If you
    want to compare things before and after you pass them in any function of
    this module, you should always take a copy first (remember that operators
    and quantum states in ``plaquette`` a bare Numpy ``ndarray``). For example:

    >>> zs = zero_state(10)
    >>> ref = zs.copy()  # this is going to be left untouched
    >>> state_after_measurement, measurement_result = measure_x_base(zs, 0)
    >>> np.all(zs == state_after_measurement)
    True
    >>> np.all(ref == state_after_measurement)
    False
"""
import collections.abc
import functools
import itertools
import re
import typing
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional, TypeAlias

import numpy as np

import plaquette

Tableau: TypeAlias = np.ndarray[Any, np.dtype[np.uint8]]
PauliDict = Mapping[int, str]


def _g(x1: int, z1: int, x2: int, z2: int) -> int:
    """Help calculate the phase of the operator when multiplying pauli operators.

    This is effectively the exponent (1, 0, or -1) of the :math:`i` factor resulting
    from this multiplication.

    See definition of ``rowsum(h, i)`` in [AG04]_.

    Args:
        x1: x bit of operator 1
        z1: z bit of operator 1
        x2: x bit of operator 2
        z2: z bit of operator 2

    Returns:
        exponent of i when two operators are multiplied
    """
    # TODO: if *really* necessary, this can be sped-up by a dict instead of a
    #  series of if-else statements. But like... only if **really** necessary.
    # this casting is necessary, otherwise unsigned numpy types will loop back
    x1 = int(x1)
    x2 = int(x2)
    z1 = int(z1)
    z2 = int(z2)
    if x1 == 0 and z1 == 0:
        return 0
    elif x1 == 1 and z1 == 1:
        return z2 - x2
    elif x1 == 1 and z1 == 0:
        return z2 * (2 * x2 - 1)
    elif x1 == 0 and z1 == 1:
        return x2 * (1 - 2 * z2)
    else:
        # There's literally no more possible combinations, so...
        raise AssertionError("Inexplicable universe breakdown")


def append_qubit(state: Tableau) -> Tableau:
    r"""Append the qubit :math:`\lvert 0\rangle` to the right side of the ``state``."""
    return insert_qubit_at(state, state.shape[1] // 2)


def append_scratch(state: Tableau) -> Tableau:
    """Append the :math:`2n+1`-th row of "scratch space".

    Returns:
        the modified state matrix.

    References:
        :cite:`aaronson_improved_2004`, p. 4.
    """
    return np.concatenate((state, np.zeros((1, state.shape[1]), dtype="u1")))


def apply_operator(op: Tableau, state: Tableau) -> Tableau:
    """Applies the operator ``op`` to the ``state``.

    ``op`` can be a single operator or a list of operators, whose lenght equals the
    number of stabiliser generators that define/support ``state``.

    Args:
        op: the operator(s) to apply.
        state: the state to modify, in binary form.

    Returns:
        the state after application.

    Notes:
        Applying an operator has a different result than applying a *gate*.

    See Also:
        The pauli *gates* :func:`x`, :func:`y`, :func:`z`, :func:`hadamard`,
        :func:`cx`, :func:`cz`, and :func:`phase_gate`.
    """
    if not isinstance(state, np.ndarray):
        raise TypeError("`state` must be a numpy array")
    if state.ndim != 2:
        raise ValueError("`state` must be a 2D numpy array")

    if isinstance(op, np.ndarray):
        raise TypeError("`op` must be a numpy array")

    if op.ndim == 1:
        # we add an "empty" dimension to normalise the array and use always the same
        # operations
        op = op[np.newaxis, :]
    elif op.shape[0] != state.shape[0]:
        # TODO: how to handle the application of 3 operators to a state with N !=3
        #  stabiliser generators? Is it necessary?
        raise ValueError(
            f"Number of operators in `op` ({op.shape[0]}) does not match number of "
            f"stabilisers in `state` ({state.shape[0]})"
        )

    if op.shape[1] != state.shape[1]:
        raise ValueError(
            "`op` and `state` act on a different number of qubits: "
            f"{op.shape[1]//2} and {state.shape[1]//2}"
        )
    # TODO: all these exceptions should be tested, I hope there's no more wrong paths.
    # casting doesn't work here for some reason, and (o, stabiliser) are not recognised
    return np.array(
        [
            multiply(o, stabiliser)[0]
            for o, stabiliser in itertools.zip_longest(op, state, fillvalue=op[0])
        ],
        dtype="u1",
    )


def commutator_sign(
    a: Tableau | list[Tableau],
    b: Tableau | list[Tableau],
    *,
    ignore_destabilisers=False,
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    r"""Check if the given operators/states (anti)commute.

    If any of the two arguments is a 1D array, a new axis will be *prepended* to the
    array, so that even if you give two single operators as arguments (i.e. 2
    1D-arrays) you will still end up with a 1D array as output, with only one element.

    If one of the arguments is an operator and the other is a state, this function will
    calculate the commutator between that operator each "row" of the state, i.e. all
    stabiliser operators that define the state.

    If **both** arguments are 2D arrays, then the output is going to be a "cartesian
    product" of commutator values. This means that it will return a matrix :math:`M` of
    shape ``(len(a), len(b))`` where each entry is
    :math:`m_{ij} = 0 \iff [a_i, b_j] = 0 \quad\text{else}\quad1`. Here
    :math:`a_i,b_j` are the rows of the states, which in the tableau representation
    are two (de)stabiliser operators.

    If ``ignore_destabilisers`` is ``True``, then the returned array shape is
    not ``(len(a), len(b))`` but rather ``(len(a)/2, len(b)/2)``. Empty axes
    are squeezed out (in the sense of :func:`numpy.squeeze`).

    Args:
        a: first operator/state
        b: second operator/state

    Keyword Args:
        ignore_destabilisers: wether or not to calculate the commutator sign
            also with the destabiliser operators. Defaults to ``False``.

    Returns:
        a 1D or 2D array of 0s (1s), representing which operators did (anti)commute.

    Examples:
        Check the commutator sign between two operators:
        >>> a, b = string_to_pauli("X1X2"), string_to_pauli("Z1Z2")
        >>> commutator_sign(a, b)
        array(0, dtype=uint8)
        >>> a, b = string_to_pauli("X1X2"), string_to_pauli("Z1X2")
        >>> commutator_sign(a, b)
        array(1, dtype=uint8)

        Check the commutator sign between a state and an operator:
        >>> zs = zero_state(5)
        >>> op = single_qubit_pauli_operator("X", 1, 5)
        >>> commutator_sign(op, zs)
        array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=uint8)

        Ignore the destabilisers from the output:
        >>> zs = zero_state(5)
        >>> op = single_qubit_pauli_operator("X", 3, 5)
        >>> commutator_sign(op, zs, ignore_destabilisers=True)
        array([0, 0, 0, 1, 0], dtype=uint8)
    """
    if isinstance(a, list):
        a = np.array(a)
    if a.ndim == 1:
        a = a[np.newaxis, :]

    if isinstance(b, list):
        b = np.array(b)
    if b.ndim == 1:
        b = b[np.newaxis, :]
    ax, az, _ = unpack_tableau(a)
    bx, bz, _ = unpack_tableau(b)

    if ignore_destabilisers:
        ax = ax[ax.shape[0] // 2 :]
        bx = bx[bx.shape[0] // 2 :]
        az = az[az.shape[0] // 2 :]
        bz = bz[bz.shape[0] // 2 :]
    return ((ax @ bz.T + az @ bx.T) % 2).astype("u1").squeeze()


def control_not_gate(
    state: Tableau,
    control_qubits: int | Sequence[int],
    target_qubits: int | Sequence[int],
) -> Tableau:
    r"""Perform a CNOT gate on ``target`` qubit based on ``control`` qubit.

    CNOT transforms stabilizers according to
    :math:`X \otimes I \mapsto X \otimes X`,
    :math:`I \otimes X \mapsto I \otimes X`,
    :math:`Z \otimes I \mapsto Z \otimes I` and
    :math:`I \otimes Z \mapsto Z \otimes Z`.

    Args:
        state: quantum state in binary representation.
        control_qubits: 0-based control qubit index/indices.
        target_qubits: 0-based target qubit index/indices.
    """
    if isinstance(target_qubits, int):
        target_qubits = [target_qubits]
    if isinstance(control_qubits, int):
        control_qubits = [control_qubits]

    if len(target_qubits) != len(control_qubits):
        raise ValueError("Target and control must have the same number of qubits")

    pairs = zip(control_qubits, target_qubits)
    x, z, r = unpack_tableau(state)

    for control, target in pairs:
        # r = r ^ (x_control * z_target *(x_target ^ z_control ^1))
        r ^= x[:, control] * z[:, target] * (x[:, target] ^ z[:, control] ^ 1)
        # x_target = x_target ^ x_control
        x[:, target] ^= x[:, control]
        # z_control = z_target ^ z_control
        z[:, control] ^= z[:, target]
    return state


def control_phase_gate(
    state: Tableau,
    control_qubits: int | Sequence[int],
    target_qubits: int | Sequence[int],
):
    r"""Perform CPHASE or CZ gate on a target qubit based on a control qubit.

    CZ transforms stabilizers according to
    :math:`X \otimes I \mapsto X \otimes Z`,
    :math:`I \otimes X \mapsto Z \otimes X`,
    :math:`Z \otimes I \mapsto Z \otimes I` and
    :math:`I \otimes Z \mapsto I \otimes Z`.

    Args:
        state: quantum state in binary representation.
        control_qubits: 0-based control qubit index/indices.
        target_qubits: 0-based target qubit index/indices.
    """
    if isinstance(target_qubits, int):
        target_qubits = [target_qubits]
    if isinstance(control_qubits, int):
        control_qubits = [control_qubits]

    if len(target_qubits) != len(control_qubits):
        raise ValueError("Target and control must have the same number of qubits")

    pairs = zip(control_qubits, target_qubits)
    x, z, r = unpack_tableau(state)

    for control, target in pairs:
        # r = r ^ (x_control * x_target *(z_target ^ z_control))
        r ^= x[:, control] * x[:, target] * (z[:, target] ^ z[:, control])
        # z_target = z_target ^ x_control
        z[:, target] ^= x[:, control]
        # z_control = z_control ^ x_target
        z[:, control] ^= x[:, target]
    return state


#: Alias for :func:`control_not_gate`.
cx = control_not_gate
#: Alias for :func:`control_phase_gate`.
cz = control_phase_gate


def dict_to_pauli(ops: PauliDict, qubits: Optional[int] = None) -> Tableau:
    """Transform a dictionary into a Pauli operator in tableau form.

    The keys of the dictionary are the qubit **indices** on which the values act.

    Args:
        ops: the operators to "combine".
        qubits: the total number of qubits. If ``None`` (default), then the total
            number of qubits will be the highest index in ``ops`` plus one.

    Returns:
        the total operator in tableau form.

    Raises:
        ValueError: when ``qubits`` is less than the inferred number of qubits taken
            from ``ops``'s keys.
    """
    # TODO: add tests
    max_qubit_index = max(ops.keys())
    if qubits is None:
        qubits = max_qubit_index + 1

    if qubits <= max_qubit_index:
        raise ValueError(
            f"Specified qubits ({qubits}) are less than the "
            f"ones necessary ({max_qubit_index+1}"
        )
    binary_operator = np.zeros(2 * qubits + 1, dtype="u1")
    for qubit_index, operator in ops.items():
        binary_operator[qubit_index] = int(operator in "XY")
        binary_operator[qubit_index + qubits] = int(operator in "ZY")
    return binary_operator


def hadamard(state: Tableau, target_qubits: int | Iterable[int]) -> Tableau:
    r"""Perform the Hadamard gate on a given target qubit.

    H transforms stabilizers according to :math:`Z \mapsto X` and
    :math:`X \mapsto Z`

    Args:
        state: quantum state in binary representation.
        target_qubits: 0-based target qubit index/indices.
    """
    if isinstance(target_qubits, int):
        target_qubits = [target_qubits]
    number_of_qubits = state.shape[1] // 2
    x, z, r = unpack_tableau(state)

    for target in target_qubits:
        # Set r = r ^ (x_target*z_target)
        r ^= x[:, target] * z[:, target]
        # Swap x_target and z_target
        state[:, [target, target + number_of_qubits]] = state[
            :, [target + number_of_qubits, target]
        ]
    return state


#: Alias for :func:`hadamard`.
h = hadamard


def insert_qubit_at(state: Tableau, pos: int | Sequence[int]) -> Tableau:
    r"""Add a qubit in state :math:`|0\rangle` at a specified position.

    Args:
        state: the state to modify, in binary form.
        pos: position(s) of qubit(s) to be added.

    Raises:
        IndexError: if you try to add a qubit at ``pos > number_of_qubits``.
        ValueError: if the ``state`` argument is not a 2D array.
    """
    # TODO: add tests
    if state.ndim != 2:
        raise ValueError(
            "You can only add qubits to a state (2D ndarray). Perhaps "
            "you're looking for pauli.pad_operator()?"
        )
    number_of_qubits = state.shape[1] // 2
    if isinstance(pos, int):
        pos = np.array([pos])
    if (pos > number_of_qubits).any():
        raise IndexError(
            f"Cannot insert a new qubit at position {pos} for a state with "
            f"{number_of_qubits} qubits."
        )
    h_pos = np.concatenate((pos, np.array(pos) + number_of_qubits))
    number_of_missing_qubits = np.max(h_pos) - number_of_qubits
    v_pos = np.concatenate(
        (
            [number_of_qubits] * number_of_missing_qubits,
            [2 * number_of_qubits] * number_of_missing_qubits,
        )
    )

    state = np.insert(state, h_pos, 0, axis=1)
    state = np.insert(state, v_pos, 0, axis=0)
    number_of_qubits += number_of_missing_qubits
    state[v_pos[:number_of_qubits], h_pos[:number_of_qubits]] = 1
    state[v_pos[number_of_qubits:], h_pos[number_of_qubits:-1]] = 1
    return state


@typing.overload
def _measure_base(
    state: Tableau,
    base: str,
    targets: int,
    *,
    destructive: bool = ...,
    forced_outcomes: Optional[int] = ...,
) -> tuple[Tableau, int]:
    ...


@typing.overload
def _measure_base(
    state: Tableau,
    base: str,
    targets: Sequence[int],
    *,
    destructive: bool = ...,
    forced_outcomes: Sequence[Optional[int]] | Optional[int] = ...,
) -> tuple[Tableau, Sequence[int]]:
    ...


def _measure_base(state, base, targets, *, destructive=False, forced_outcomes=None):
    """Measure a series of qubits in the given base, sequentially.

    Args:
        state: the state to measure
        base: either ``X`` or ``Z``.
        targets: a list of qubit indices to measure

    Keyword Args:
        destructive: if ``True``, each measured qubit will be traced out after **all**
            non-destructive measurements using the PTRACE algorithm found in
            `this paper <https://doi.org/10.1088/1367-2630/7/1/170>`_.
        forced_outcomes: same as in :func:`measure`.

    Returns:
        a tuple whose first element is the transformed state, and the second one is the
        actual measurement outcome.

    Raises:
        ValueError: if the list of given ``forced_outcomes`` is not the same
        size as ``targets``.

    Notes:
        If you give a list of targets but a single forced outcome, this is
        interpreted as if you want that to be the forced outcome for each
        target. If you give a sequence of forced outcomes, this needs to be
        the same length as ``targets``.

        Also note that you do not need to give the targets in any particular
        order, but if you asked for destructive measurements, tracing will
        be done highest-index first. The relationship target-forced outcome
        is preserved also at the output.
    """
    n_q = count_qubits(state)[0]

    if isinstance(targets, int):
        targets = [targets]

    if isinstance(forced_outcomes, int) or forced_outcomes is None:
        forced_outcomes = [forced_outcomes]
    elif not isinstance(forced_outcomes, collections.abc.Sequence):
        raise TypeError("forced_outcomes must be None, int, or a sequence of these")

    forced_outcomes = typing.cast(Sequence[Optional[int]], forced_outcomes)

    # Pad the outcomes to match the size of the targets list, so we can zip
    # everything in a for-loop

    _fo: list[Optional[int]] = []
    if len(forced_outcomes) == 1:
        for _ in range(len(targets)):
            _fo.append(forced_outcomes[-1])
    elif len(forced_outcomes) == len(targets):
        for outcome in forced_outcomes:
            _fo.append(outcome)
    else:
        raise ValueError(
            "The number of forced outcomes must match the number of targets, "
            "or be a single scalar."
        )

    measurement_results = np.zeros(len(targets), dtype="u1")
    # In order to not invalidate the indices of the targets, we need to start
    # tracing from the highest index to the lowest. This also means that we
    # need to first sort the targets in descending order, and then reorder
    # the forced outcomes `_fo` accordingly. We cannot assume that the original
    # targets were in any particular order, but we WILL assume that the first
    # forced outcome was referring to the first target and so on.
    #
    # As an example, if I have the targets [2, 3, 1] and the outcomes [0, 1, 0]
    # I need to reshuffle everything to obtain
    #   targets = [3, 2, 1]
    # and
    #   _fo = [0, 0, 1]
    # so we can simply use np.argsort for that, but we need to start from the
    # highest index in order not to shoot ourselves in the foot, so we
    # reverse everything
    for sorted_idx in reversed(np.argsort(targets)):
        state, m = measure(
            state,
            single_qubit_pauli_operator(base, targets[sorted_idx], n_q),
            forced_outcome=_fo[sorted_idx],
        )
        measurement_results[sorted_idx] = m
        if destructive:
            state = _ptrace(state, targets[sorted_idx], base)
    return state, measurement_results[0] if len(targets) == 1 else measurement_results


def _ptrace(state: Tableau, target: int, measured_base: str) -> Tableau:
    """Execute the PTRACE algorithm on ``state``, tracing out the ``target`` qubit.

    Args:
        state: the state to modify
        target: the qubit to trace out
        measured_base: the base in which the qubit was measured, either ``Z`` or ``X``

    Returns:
        the modified state

    Note:
        This trace function is very limited, and will only work on a single
        qubit. Additionally, right now it assumes that you just measured a single-qubit
        Pauli operator, and you want to trace the qubit pertaining to that operator,
        hence why the need to specify ``measured_base``.

    Example:
        >>> state = zero_state(10)
        >>> state, m = measure_z_base(state, 0)
        >>> state = _ptrace(state, 0, 'Z')
        >>> m
        0
        >>> count_qubits(state)
        [9]
    """
    _x, _z, _ = unpack_tableau(state)
    number_of_qubits = count_qubits(state)[0]
    if measured_base == "X":
        ones = np.where(_x[number_of_qubits:, target] > 0)[0]
    elif measured_base == "Z":
        ones = np.where(_z[number_of_qubits:, target] > 0)[0]
    else:
        # Can't do Y yet. Do we even want to?
        raise ValueError("Y-basis is not supported")
    first_stab = ones[0]

    for i in ones[1:]:
        state[i + number_of_qubits], _ = multiply(
            state[i + number_of_qubits], state[first_stab + number_of_qubits]
        )
        state[first_stab], _ = multiply(state[i], state[first_stab])

    a = np.delete(state, [first_stab, first_stab + number_of_qubits], axis=0)
    b = np.delete(a, [target, target + number_of_qubits], axis=1)
    state = b  # should have one less qubit

    return state


@typing.overload
def measure_z_base(
    state: Tableau,
    targets: int,
    *,
    destructive: bool = ...,
    forced_outcomes: Optional[int] = ...,
) -> tuple[Tableau, int]:
    ...


@typing.overload
def measure_z_base(
    state: Tableau,
    targets: Sequence[int],
    *,
    destructive: bool = ...,
    forced_outcomes: Sequence[Optional[int]] | Optional[int] = ...,
) -> tuple[Tableau, Sequence[int]]:
    ...


def measure_z_base(state, targets, *, destructive=False, forced_outcomes=None):
    """Uncorrelated single-qubit Z-operator parity measurement on a series of qubits."""
    return _measure_base(
        state, "Z", targets, destructive=destructive, forced_outcomes=forced_outcomes
    )


@typing.overload
def measure_x_base(
    state: Tableau,
    targets: int,
    *,
    destructive: bool = ...,
    forced_outcomes: Optional[int] = ...,
) -> tuple[Tableau, int]:
    ...


@typing.overload
def measure_x_base(
    state: Tableau,
    targets: Sequence[int],
    *,
    destructive: bool = ...,
    forced_outcomes: Sequence[Optional[int]] | Optional[int] = ...,
) -> tuple[Tableau, Sequence[int]]:
    ...


def measure_x_base(state, targets, *, destructive=False, forced_outcomes=None):
    """Uncorrelated single-qubit X-operator parity measurement on a series of qubits."""
    return _measure_base(
        state, "X", targets, destructive=destructive, forced_outcomes=forced_outcomes
    )


def measure(
    state: Tableau,
    meas_operator: Tableau | str,
    *,
    forced_outcome: Optional[int] = None,
) -> tuple[Tableau, int]:
    """Measure the parity of the given operator non-destructively.

    Args:
        state: the state to measure.
        meas_operator: full measurement operator either in binary form already, or a
            string that can be passed to :func:`string_to_pauli`.

    Keyword Args:
        forced_outcome: if a measurement is such that its outcome would be chosen at
            random, use the provided outcome instead.

            .. warning:: No check is made that this forced outcome makes
                physical/mathematical sense in the context of the measurement.

    Returns:
        a tuple where the first element is the state modified by the measurement,
        and the second is the measurement outcome (0 or 1) itself.

    Raises:
        ValueError: if the given *binary* operator is defined on a different number of
            qubits than the state.
        ValueError: if ``forced_outcome`` is neither ``0`` nor ``1``.
    """
    number_of_qubits = count_qubits(state)[0]
    if isinstance(meas_operator, str):
        meas_op = string_to_pauli(meas_operator, number_of_qubits)
    else:
        if number_of_qubits != (m_q := count_qubits(meas_operator)[0]):
            raise ValueError(
                f"Measurement operator is defined on {m_q} qubits, while the state is "
                f"defined on {number_of_qubits}, but the two must match"
            )
        meas_op = np.copy(meas_operator)
    comm = commutator_sign(state, meas_op)
    # List of indices of destabilizers and stabilizers that don't commute
    # with meas_op
    nonzero = np.nonzero(comm)[0]

    # Outcome is random if one meas_op does not commute with at least one
    # stabilizer, which appear after n_qubits. We don't care about not commuting
    # with the de-stabilisers
    if nonzero[-1] >= number_of_qubits:
        # Find the first non-commuting (with meas_op) stabilizer
        first_nc_stab = nonzero[nonzero >= number_of_qubits][0]
        index_pos = np.where(nonzero == first_nc_stab)[0][0]
        other_rows = np.delete(nonzero, index_pos)

        # Multiply all other non-commuting stabilizers with the first one
        for row_no in other_rows:
            state[row_no], _ = multiply(state[first_nc_stab], state[row_no])

        # Replace destabilizer corresponding to first non-commuting stabilizer
        # with the stabilizer itself
        state[first_nc_stab - number_of_qubits] = state[first_nc_stab]

        # Replace the first non-commuting stabilizer with
        # (-1)ˆmeas_outcome meas_op
        if forced_outcome is not None:
            if forced_outcome not in [0, 1]:
                raise ValueError(
                    f"Measurement outcome can only be 0 or 1, but {forced_outcome} "
                    "was given."
                )
            meas_outcome = forced_outcome
        else:
            meas_outcome = plaquette.rng.choice([0, 1])
        meas_op[-1] ^= meas_outcome
        state[first_nc_stab] = meas_op

    # Outcome is deterministic if all stabilizers commute with meas_op.
    # Refer to Pg. 5 (unnumbered equation) of [AG04]_ for results and
    # handwritten notes Pg.# 7 for explanation.
    # TODO: if the handwritten notes are important to understand the code, then they
    #  belong in the (developer) docs
    else:
        state = append_scratch(state)

        for row_no in nonzero:
            state[2 * number_of_qubits], _ = multiply(
                state[2 * number_of_qubits], state[row_no + number_of_qubits]
            )

        meas_outcome = int(state[-1, -1] != meas_op[-1])
        state = state[:-1]

    return state, meas_outcome


def count_qubits(
    tableau: Tableau | Sequence[Tableau], include_identities=True
) -> Sequence[int]:
    """Return the number of qubits that this tableau "acts" upon.

    Args:
        tableau: the binary representation of the operator/state, or a list of
            operators.
        include_identities: whether to include identities or not.

    Returns:
        if ``include_identities=True`` (default), it will return the total number of
        qubits which support this operator, otherwise return the number of qubits
        on which a non-identity Pauli operator acts.
    """
    if isinstance(tableau, np.ndarray):
        if tableau.ndim == 1:
            tableau = tableau[np.newaxis, :]
        if include_identities:
            return [tableau.shape[-1] // 2]
        _x, _z, _ = unpack_tableau(tableau)
        return np.count_nonzero(_x | _z, axis=1)
    else:
        # Here we assume that tableau is a sequence
        tableau = typing.cast(Sequence[Tableau], tableau)
        active_qubits: list[int] = list()
        for op in tableau:
            _x, _z, _ = unpack_tableau(op)
            active_qubits.append(
                op.size // 2 if include_identities else np.count_nonzero(_x | _z)
            )
        return active_qubits


def pad_operator(op: Tableau, total_qubits: int) -> Tableau:
    """Add identites to ``op`` such that it acts on ``total_qubits``.

    Args:
        op: the operator to pad.
        total_qubits: total number of qubits of the resulting operator

    Notes:
        if ``total_qubits`` is less than the current number of qubits supported by
        ``op``, the latter will be returned unchanged. That is, **you cannot
        truncate** an operator.
    """
    current_qubits = op.size // 2
    if current_qubits >= total_qubits:
        return op
    missing_qubits = total_qubits - current_qubits
    return np.insert(
        op,
        np.concatenate(
            ([current_qubits] * missing_qubits, [2 * current_qubits] * missing_qubits)
        ),
        0,
    )


def pauli_to_dict(op: Tableau) -> PauliDict:
    """Transform a Pauli operator into a mapping of qubit indices and Pauli term.

    .. important:: This transformation discards phase/sign information of the
       original operator.

    Examples:
        >>> pauli_to_dict(np.array([0, 1, 1, 0, 1]))
        {0: 'Z', 1: 'X'}
    """
    x, z, _ = unpack_tableau(op)

    res = {}
    for qubit, (i, j) in enumerate(zip(x, z)):
        binary_index = (i << 1) | j
        if binary_index:
            # We discard the identity case (binary_index == 0)
            res[qubit] = "IZXY"[binary_index]
    return res


def pauli_to_string(op: Tableau, show_identities: bool = True) -> str:
    """Convert an operator's binary form to a Pauli string.

    Args:
        op: One-dimensional binary array
        show_identities: Format output as e.g., ``'+XIX'`` if true. Else as
            ``'+ X0 X3'``.

    Returns:
        Pauli form of the binary array.

    Raises:
        ValueError: if you try to convert a *state* to a string or if the length of
            the operator is even (the internal representation of operators assumes an
            odd number of bits).
    """
    if op.ndim != 1:
        raise ValueError(
            "Only operators (1D ndarrays) can be parsed. Use "
            "`state_to_stabiliser_string` for states (2D ndarrays)"
        )
    elif len(op) % 2 == 0:
        raise ValueError("Length should be odd to include sign bit")

    xs, zs, r = unpack_tableau(op)
    sign = "-" if r else "+"
    arr = xs + 2 * zs

    trans = str.maketrans("0123", "IXZY", "[ ,\n]")
    strarr = str(arr.tolist())
    pauli_op_main = strarr.translate(trans)

    if show_identities:
        pauli_op = sign + pauli_op_main
    else:
        alpha_str = ""
        for location, bin_op_str in enumerate(pauli_op_main):
            if bin_op_str != "I":
                alpha_str = alpha_str + " " + bin_op_str + str(location)
        pauli_op = sign + alpha_str

    return pauli_op


def phase_gate(state: Tableau, targets: int | Sequence[int] | Tableau) -> Tableau:
    r"""Perform the phase gate on a given target qubit.

    Phase transforms stabilizers according to :math:`X \mapsto Y` and
    :math:`Z \mapsto Z`

    Args:
        state: the state to which to apply the gate
        targets: The indices of the target qubit
    """
    if isinstance(targets, int):
        targets = [targets]

    # These are references to the same underlying numpy array, so modifying
    x, z, r = unpack_tableau(state)

    for target in targets:
        # Set r = r ^ (x_target*z_target)
        r ^= x[:, target] * z[:, target]
        # z_target = z_target ^ x_target
        z[:, target] ^= x[:, target]
    return state


def state_to_stabiliser_string(
    state: Tableau, show_identities: bool
) -> tuple[list[str], list[str]]:
    """Convert a state's binary form to a series of Pauli strings.

    Args:
        state: Two-dimensional binary array
        show_identities: Format output as e.g., ``'+XIX'`` if true. Else as
            ``'+ X0 X3'``

    Returns:
        a tuple of two lists. Each list contains the destabilisers and stabilisers,
        respectively, as strings.

    Examples:
        >>> d, s = state_to_stabiliser_string(zero_state(2), True)
        >>> print(d)
        ['+XI', '+IX']
        >>> print(s)
        ['+ZI', '+IZ']
    """
    if state.shape[0] != (state.shape[1] - 1):
        raise ValueError(
            "The given state seems malformed. I'm expecting an (N, N+1) array, "
            f"but got {state.shape!r}"
        )

    operators = list(map(lambda o: pauli_to_string(o, show_identities), state))
    n = len(operators)
    return operators[: n // 2], operators[n // 2 :]


def pprint_state(
    state: Tableau,
    show_identities: bool = True,
    show_destabilisers: bool = False,
    **kwargs,
):
    """Pretty-print the string representation of state.

    Args:
        state: quantum state in binary representation.
        show_identities: include identities in the displayed state.
            .. seealso:: :func:`state_to_stabiliser_string`.
        show_destabilisers: include destabilisers in the displayed state.
        kwargs: passed over to :func:`print`.

    Examples:
        >>> pprint_state(zero_state(2), show_destabilisers=True)
        Destabilisers:
        +XI
        +IX
        Stabilisers:
        +ZI
        +IZ
    """
    d, s = state_to_stabiliser_string(state, show_identities)
    destabilisers = "\n".join(d)
    stabilisers = "\n".join(s)
    if show_destabilisers:
        print(f"Destabilisers:\n{destabilisers}\nStabilisers:\n{stabilisers}", **kwargs)
    else:
        print(f"Stabilisers:\n{stabilisers}", **kwargs)


def string_to_pauli(op_str: str, qubits: Optional[int] = None) -> Tableau:
    """Transform a string such as ``"+XYZI"`` into its binary representation.

    The input string can be of two forms. A **canonical** form specifies all operators
    acting on the qubits, i.e. the identities are specified as part of the tensor
    product and not omitted (e.g. ``XIIZIY``). A **short** form specifies only those
    non-identity operators and on which qubits they act (e.g. ``X2Y44``, would be an
    ``X`` operator acting on qubit 3 and another ``X`` operator acting on qubit 45).

    To help writing long operators, or if you want to group parts of them together
    visually, you can insert spaces and/or underscore wherever you like. They will be
    removed before starting to parse the actual operator string.

    .. important:: Canonical and short form **can't be mixed**.

    Args:
        op_str: the string description of the operator. Spaces and/or underscores are
            removed before parsing.
        qubits: optionally, the total number of qubits. If ``None`` (default)
            the number of qubit is taken as the total number of operators present.

    Raises:
        ValueError: when giving a string containing unrecognised characters, i.e.
            anything but ``[+-IZXY0-9]``.
        ValueError: when specifying also ``qubits`` and if ``qubits < len(op_str)``.
        IndexError: when giving a number of ``qubits`` which is less than the number
            of operators.

    Examples:
        Create an operator with its canonical form.

        >>> op_canon = string_to_pauli("XIIXYZ")
        >>> count_qubits(op_canon) == [6]
        True

        Include sign information

        >>> op_signed = string_to_pauli("-XXIZ")
        >>> count_qubits(op_signed) == [4]
        True
        >>> op_signed[-1] == 1
        True

        Create an operator with its canonical form and specifying a much larger space.

        >>> op_long = string_to_pauli("XIIXYZ", 10)
        >>> count_qubits(op_long) == [10]
        True

        Create an operator with its short form.

        >>> op_short = string_to_pauli("X0X3Y4Z5")
        >>> count_qubits(op_short) == [6]
        True
        >>> np.all(op_short == op_canon)
        True
    """
    # TODO: add tests
    initial_phase = 0
    # Remove visual separators
    op_str = op_str.replace(" ", "")
    op_str = op_str.replace("_", "")
    # Check if there's a digit somewhere
    if any(map(lambda c: c.isdigit(), op_str)):
        # In that case we assume short form and bring this into canonical form
        short_string_re = re.compile(r"([-+])?(([XYZI])(\d+))")
        # Split op_str into single-qubit operator strings
        # for a string like "+X11Z33Y4" the result would be
        #     [('+', 'X11', 'X', '11'), ('', 'I33', 'I', '33'), ('', 'Y4', 'Y', '4')]
        # where each tuple in the list is an "operator", and for each tuple
        # * the first element is the sign (ideally only the first one has it)
        # * the second element is the whole single qubit operator (X11, X on 12th qubit)
        # * the third element is the operator only
        # * the fourth element is the qubit **index**
        op_str_components = short_string_re.findall(op_str)
        # we check only the first sign to apply a global phase
        if op_str_components[0][0] == "-":
            initial_phase = 1
        operator_dict: PauliDict = {int(i): o for _, _, o, i in op_str_components}
    else:
        # Otherwise, this is the canonical representation. We first remove the sign
        # if present at all
        if op_str[0] in "+-":
            initial_phase = 1 if op_str[0] == "-" else 0
            op_str = op_str[1:]
        # and then we compile a dictionary of where non-identity operators are
        operator_dict = {i: o for i, o in enumerate(op_str) if o != "I"}
        # we also need to make sure that we are not truncating identities from
        # the end in the canonical representation
        qubits = max(qubits or 0, len(op_str))

    binary_operator = dict_to_pauli(operator_dict, qubits)
    binary_operator[-1] = initial_phase
    return binary_operator


def string_to_state(
    stabilisers: Sequence[str],
    destabilisers: Optional[Sequence[str]] = None,
    qubits: Optional[int] = None,
) -> Tableau:
    """Transform a state from its string representation to its tableau one.

    Args:
        stabilisers: the list of stabiliser operators.
        destabilisers: if given, the list of destabilisers will be checked for
            consistencies with the stabilisers. If ``None`` (default) then
            an appropriate set of destabilisers will be computed from the
            stabilisers.
        qubits: the total number of qubits that this state should support.

    Returns:
        the state in tableau format.

    Examples:
        >>> string_to_state(["ZI", "IZ"])
        array([[1, 0, 0, 0, 0],
               [0, 1, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 1, 0]], dtype=uint8)

        >>> string_to_state(["XI", "IX"])
        array([[0, 0, 1, 0, 0],
               [0, 0, 0, 1, 0],
               [1, 0, 0, 0, 0],
               [0, 1, 0, 0, 0]], dtype=uint8)

        Giving operators that do not respect the properties of stabilisers
        states will raise an exception:
        >>> string_to_state(["XI", "ZI"])
        Traceback (most recent call last):
          ...
        ValueError: Stabiliser 1 does not commute with stabiliser 0

        If you know both stabiliser and destabilisers, you can give them
        both. The function will still check for commutation relations:
        >>> np.all(string_to_state(["ZI", "IZ"], ["XI", "IX"]) == zero_state(2))
        True
        >>> string_to_state(["ZI", "IZ"], ["XI", "ZX"])
        Traceback (most recent call last):
          ...
        ValueError: Destabiliser 1 does not commute with destabiliser 0
    """
    binary_stabilisers = np.array(
        [string_to_pauli(s, qubits=qubits) for s in stabilisers]
    )
    if destabilisers is not None:
        # If we are given the destabilisers, we should check that they are
        # the same number of stabilisers and also that they respect basic
        # commutation relationships
        binary_destabilisers = np.array(
            [string_to_pauli(s, qubits=qubits) for s in destabilisers]
        )
        if len(destabilisers) != len(stabilisers):
            raise ValueError(
                "when providing both, the number of stabilisers and "
                "destabilisers must match"
            )
    else:
        # If no destabilisers are given, we need to calculate them
        ident_size = count_qubits(binary_stabilisers[0])[0]
        ident = np.eye(ident_size, dtype="u1")
        zero = np.zeros(ident.shape, dtype="u1")
        omega = np.block([[zero, ident], [ident, zero]])
        binary_destabilisers = (
            np.linalg.pinv(binary_stabilisers[:, :-1].T) @ omega
        ).astype("u1")
        # attach an empty column for the sign info
        binary_destabilisers = np.hstack(
            (binary_destabilisers, np.zeros((ident_size, 1), dtype="u1"))
        )

    # No matter how we got the destabilisers, check things for sanity
    for i, s in enumerate(binary_stabilisers):
        for j, d in enumerate(binary_destabilisers):
            if i < j:  # avoid calculating things twice
                continue
            # reuse the indices to also check commutation relations among
            # stabilisers and destabilisers themselves
            if commutator_sign(binary_stabilisers[i], binary_stabilisers[j]):
                raise ValueError(f"Stabiliser {i} does not commute with stabiliser {j}")
            if commutator_sign(binary_destabilisers[i], binary_destabilisers[j]):
                raise ValueError(
                    f"Destabiliser {i} does not commute with destabiliser {j}"
                )

            # otherwise we check the relationships between stabilisers and destab
            if i == j and not commutator_sign(s, d):
                # stab and destab of the same index should anticommute
                raise ValueError(
                    f"Destabiliser {j} does not anticommute with its stabiliser"
                )
            elif j > i and commutator_sign(s, d):
                # different indices should commute instead
                raise ValueError(
                    f"Destabiliser {j} does not commute with stabiliser {i}"
                )
    return np.vstack((binary_destabilisers, binary_stabilisers)).astype("u1")


def tensor_product(*operators: Tableau) -> Tableau:
    """Compute the tensor product among the given operators or states.

    Raises:
        ValueError: when trying to multiply an operator and a state.

    Examples:

        Tensor product of two operators:
        >>> prod = tensor_product(string_to_pauli("XZ"), string_to_pauli("-XY"))
        >>> pauli_to_string(prod)
        '-XZXY'

        Tensor product of two states:
        >>> s1 = zero_state(2)
        >>> s2 = zero_state(4)
        >>> pprint_state(tensor_product(s1, s2))
        Stabilisers:
        +ZIIIII
        +IZIIII
        +IIZIII
        +IIIZII
        +IIIIZI
        +IIIIIZ

        Tensor product of a single operator with a state is not allowed:
        >>> tensor_product(zero_state(2), single_qubit_pauli_operator("X", 0, 1))
        Traceback (most recent call last):
         ...
        ValueError: cannot multiply operators with states (mismatching number of dimensions in input arrays)

    Notes:
        Giving a single operator as input argument will simply return it
        unchanged as output.
    """  # noqa
    if len(operators) == 0:
        raise ValueError("At least one operator is necessary")
    elif len(operators) == 1:
        return operators[0]
    elif len(operators) > 2:
        prod = tensor_product(operators[-2], operators[-1])
        return tensor_product(*operators[:-2], prod)
    else:
        if len(operators[0].shape) != len(operators[1].shape):
            raise ValueError(
                "cannot multiply operators with states (mismatching "
                "number of dimensions in input arrays)"
            )
        # X part, Z part, and sign of the two operators A and B, resulting in
        # operator C
        ax, az, asig = unpack_tableau(operators[0])
        bx, bz, bsig = unpack_tableau(operators[1])

        # If we multiply states, we need to pad the resulting tableau with
        # identities "blocks".
        if len(operators[0].shape) == 2:  # this is a state
            nq_a = count_qubits(operators[0])[0]
            nq_b = count_qubits(operators[1])[0]

            # split the x and z parts of the two operators in destabilisers
            # and stabiliser sections, respectively
            dest_ax, stab_ax = ax[: ax.shape[0] // 2], ax[ax.shape[0] // 2 :]
            dest_az, stab_az = az[: az.shape[0] // 2], az[az.shape[0] // 2 :]
            dest_bx, stab_bx = bx[: bx.shape[0] // 2], bx[bx.shape[0] // 2 :]
            dest_bz, stab_bz = bz[: bz.shape[0] // 2], bz[bz.shape[0] // 2 :]

            # Pad each section with zeroes corresponding to the other state
            c = np.block(
                [
                    [dest_ax, np.zeros((nq_a, nq_b)), dest_az, np.zeros((nq_a, nq_b))],
                    [np.zeros((nq_b, nq_a)), dest_bx, np.zeros((nq_b, nq_a)), dest_bz],
                    [stab_ax, np.zeros((nq_a, nq_b)), stab_az, np.zeros((nq_a, nq_b))],
                    [np.zeros((nq_b, nq_a)), stab_bx, np.zeros((nq_b, nq_a)), stab_bz],
                ]
            ).astype("u1")
            # the 'astype' ensures that we don't fallback to floats, which will make
            # other parts of the library misbehave
            csig = np.hstack((asig[:nq_a], bsig[:nq_b], asig[nq_a:], bsig[nq_b:]))
            # Transpose otherwise numpy will not concatenate correctly
            return np.hstack((c, csig[None, :].T))
        else:
            csig = asig ^ bsig
            return np.hstack((ax, bx, az, bz, csig)).astype("u1")


def reset_qubits_to_eigenstate(
    state: Tableau, pauli: str, targets: int | Iterable[int] | Tableau
) -> Tableau:
    """Reset qubits to +1 eigenstate of given Pauli operator.

    Args:
        state: the state with the qubit(s) to reset.
        pauli: Pauli operator. Eg. :math:`Z` or :math:`-X`.
        targets: The indices of the qubits to be reset.
    """
    if not isinstance(state, np.ndarray):
        raise TypeError(f"`state` must be a numpy array, not '{type(state)}'.")
    # No Ys yet please
    # What have thei done to iou?
    if isinstance(targets, int):
        targets = [targets]

    state_qubits = count_qubits(state)[0]

    for qubit in targets:
        state, meas_outcome = measure(
            state, single_qubit_pauli_operator(pauli, qubit, state_qubits)
        )
        if meas_outcome:
            if pauli == "Z" or pauli == "-Z":
                state = x(state, qubit)
            elif pauli == "X" or pauli == "-X":
                state = z(state, qubit)
    return state


def single_qubit_pauli_operator(op: str, qubit: int, qubits: int) -> Tableau:
    """Create the binary representation of single-qubit Pauli operator.

    Constructs a single qubit operator in binary representation from the pauli operator
    its position and the total number of qubits.

    Args:
        op: Single qubit Pauli operator, e.g. "Z" or "-X" or "+Y". Uppercase only.
        qubit: Zero-based position of the operator.
        qubits: Total number of qubits.

    Returns:
        Operator in binary representation.

    Raises:
        ValueError: if the input string contains anything but characters from
            ``+-IXYZ``.
        IndexError: if the total number of qubits is less than the position of the qubit
            on which this operator should act.
    """
    if qubit < 0:
        raise ValueError("Qubits indices can't be negative")
    if qubits < 1:
        raise ValueError("Total qubit number must be 1 or more")
    if qubits <= qubit:
        raise ValueError("Total number of qubits less than target qubit position")

    meas_op = np.zeros(2 * qubits + 1, dtype="u1")
    meas_op[-1] = 1 if op[0] == "-" else 0

    shift: int | Tableau
    if op[-1] == "X":
        shift = 0
    elif op[-1] == "Z":
        shift = qubits
    elif op[-1] == "Y":
        shift = np.array([0, qubits])
    else:
        raise ValueError("Only X, Y, or Z are allowed, optionally with sign")

    meas_op[shift + qubit] = 1
    return meas_op


def x(state: Tableau, qubits: int | Iterable[int]) -> Tableau:
    r"""Perform the X (bit-flip) gate on a given target qubit.

    X transforms stabilizers according to :math:`X \mapsto X` and
    :math:`Z \mapsto -Z`.

    Args:
        state: The state in the tableau representation
        qubits: The indices of the target qubit
    """
    if isinstance(qubits, int):
        qubits = [qubits]
    number_of_qubits = state.shape[1] // 2

    if any(map(lambda q: q >= number_of_qubits, qubits)):
        # TODO: add test
        raise IndexError("Target qubit index higher than total number of qubits")

    x, z, r = unpack_tableau(state)
    for q in qubits:
        # Set r = r ^ z_target
        state[:, 2 * number_of_qubits] = r ^ z[:, q]
    return state


def y(state: Tableau, qubits: int | Iterable[int]) -> Tableau:
    r"""Perform the Y gate on a given target qubit.

    This is the same as applying the gate :math:`ZX`, and is here for convenience,
    since it calls :func:`z` and :func:`x` internally.

    Args:
        state: The state in the tableau representation
        qubits: The indices of the target qubit
    """
    return z(x(state, qubits), qubits)


def unpack_tableau(operator: Tableau) -> tuple[Tableau, Tableau, Tableau]:
    """Split the operator/state into its X/Z/Sign components.

    Args:
        operator: the operator to unpack

    Raises:
        ValueError: if the function's argument last axis has an even number of elements.
    """
    if operator.ndim > 2 or operator.ndim == 0:
        raise ValueError("Only 1D or 2D arrays are allowed")
    if operator.shape[-1] % 2 == 0:
        # We are expecting an operator/state WITH sign row
        raise ValueError("Argument has wrong shape (are you missing the sign column?)")

    elif operator.ndim == 2:
        return (
            operator[:, : operator.shape[1] // 2],
            operator[:, operator.shape[1] // 2 : -1],
            operator[:, -1],
        )
    else:
        return (
            operator[: operator.size // 2],
            operator[operator.size // 2 : -1],
            operator[-1],
        )


def multiply(
    *operators: Tableau, initial_phase: complex = 1
) -> tuple[Tableau, complex]:
    """Compute the product among two Pauli operators.

    Args:
        *operators: a series of Pauli operators in binary form.
        initial_phase: the initial complex phase that this product has.

    Returns:
        a tuple where the first item is the computed operator in binary form (1D array)
        and the global complex phase resulting from this product.

    Raises:
        TypeError: when anything but a numpy array is given as operators.
        NotImplementedError: when trying to multiply many operators at once (i.e.
            2D numpy arrays)
        ValueError: when not enough operators (2) are supplied **or** when the given
            operators have different shapes.
        RuntimeError: when some internal consistency check fails. This should never
            happen, but when it does (because of course it will) it should be
            reported as bug.
    """
    if any(map(lambda o: not isinstance(o, np.ndarray), operators)):
        raise TypeError("Only operators in binary form (as ndarray) are supported.")
    elif any(map(lambda o: o.ndim != 1, operators)):
        # we handle only the case of single operators here, not full a tableau
        # what about the future?
        raise NotImplementedError(
            "Only products among single operators are currently supported."
        )

    if len(operators) == 0:
        raise ValueError("At least one operator is necessary")
    elif len(operators) == 1:
        # If only one operator is given, treat this product as being with an identity.
        return operators[0], -1 if operators[0][-1] else 1
    elif len(operators) > 2:
        prod, phase = multiply(
            operators[-2], operators[-1], initial_phase=initial_phase
        )
        return multiply(*operators[:-2], prod, initial_phase=phase)
    else:
        if operators[0].shape != operators[1].shape:
            raise ValueError("Operators must have the same shape.")

        ans = operators[0] ^ operators[1]
        phase_exponent = functools.reduce(
            lambda prev, nxt: prev + _g(*nxt),
            zip(*unpack_tableau(operators[0])[:-1], *unpack_tableau(operators[1])[:-1]),
            0,
        )
        # This is a bit more "generic" than the version on the paper this is from.
        # The point is that in order for this function to support *arbitrary* products,
        # and not the ones only in a tableau. The paper claims that the outcome of
        # the routine for the sign can only be 2 or 4. This is demonstrably wrong for
        # single-qubit operator products. Since this function is generic, we take care
        # of calculating the entire phase.
        #
        # The sign information is also entirely included in ``initial_phase`` and copied
        # to the sign bit only to be used in those algorithms that require it. If we
        # don't do this, we will ignore minus-signs from ``i`` factors.
        phase = initial_phase * 1j**phase_exponent * (1 - 2 * ans[-1])
        if phase.imag != 0:
            sign_bit = 1 if phase.imag < 0 else 0
        elif phase.real != 0:
            sign_bit = 1 if phase.real < 0 else 0
        else:
            raise RuntimeError("Internal consistency check failed during product")
        ans[-1] = sign_bit
        return ans, phase


def z(state: Tableau, qubits: int | Iterable[int] | Tableau):
    r"""Perform the Z (phase-flip) gate on a given target qubit.

    Z transforms stabilizers according to :math:`X \mapsto -X` and
    :math:`Z \mapsto Z`

    Args:
        state: The state in the tableau representation
        qubits: The indices of the target qubit
    """
    if isinstance(qubits, int):
        qubits = [qubits]
    number_of_qubits = state.shape[1] // 2

    if any(map(lambda q: q >= number_of_qubits, qubits)):
        # TODO: add test
        raise IndexError("Target qubit index higher than total number of qubits")

    for q in qubits:
        # Set r = r ^ x_target
        state[:, 2 * number_of_qubits] = state[:, 2 * number_of_qubits] ^ state[:, q]
    return state


def zero_state(number_of_qubits: int) -> Tableau:
    """Return the zero state in the computational basis of ``number_of_qubits``.

    Args:
        number_of_qubits: how many qubits to initialise this state with.

    Returns:
        the tableau representation of the state.
    """
    return np.eye(2 * number_of_qubits + 1, dtype="u1")[:-1]
