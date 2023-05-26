# Copyright 2023, It'sQ GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Visualization tools.

When working with a stabilizer code, it can be useful to work with a graphical
representation of the whole code, which describes both data qubits and stabilizer
generators. Such a graphical representation can be obtained as follows::

    from plaquette import codes, visualizer

    code = codes.LatticeCode.make_planar(n_rounds=1, size=3)
    vis = visualizer.LatticeVisualizer(code)
    vis.draw_lattice()

The section :ref:`codes-guide` contains many examples of code visualizations.

If you also want to draw errors, syndrome data ond/or corrections, refer to
:meth:`LatticeVisualizer.draw_latticedata` or :ref:`viz-guide`. Using such a
drawing, you can explore the relation between individual errors and triggered
syndrome bits as well as determine why a given correction succeeded or failed.

You can also visualise circuits that you generate from codes, using the
:class:`CircuitVisualizer`. This will use Qiskit under the hood to render the
circuit into a ``matplotlib`` figure.

.. important::

    When generating a circuit, you can provide error information to model
    hardware-relevant use-cases. These types of errors are usually represented
    as probabilistic gates, which are impossible to represent with standard
    quantum circuit symbols. This means that **circuits containing error
    instructions cannot be rendered**.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, cast

import matplotlib as mpl  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore
import qiskit
from qiskit import qasm3

from plaquette import circuit, codes
from plaquette.circuit import openqasm
from plaquette.codes import latticebase
from plaquette.pauli import Tableau, unpack_tableau

#: Optional Matplotlib subplot axes
OptSubAx = Optional[mpl.axes.Subplot]


class LatticeVisualizer:
    """Create figures related to :class:`plaquette.codes.LatticeCode`.

    .. automethod:: __init__
    """

    #: An error correction code defined using a lattice.
    #:
    #: We use the lattice coordinates for creating figures.
    code: latticebase.CodeLattice
    #: Qubit error probabilities
    qubit_error_probs: Optional[np.ndarray]

    def __init__(
        self,
        code: codes.LatticeCode | latticebase.CodeLattice,
        qubit_error_probs: Optional[np.ndarray] = None,
    ):
        """Create a visualizer.

        Args:
            code: The code
            qubit_error_probs:
                Error probabilities (optional). One entry for each physical data qubit.
        """
        the_code = code if isinstance(code, latticebase.CodeLattice) else code.lattice
        #: The error correction code on a lattice
        self.code = the_code
        self.qubit_error_probs = qubit_error_probs
        if not len(the_code.lattice.shape) == 2:
            raise ValueError("Can only plot two-dimensional code lattices")
        if self.qubit_error_probs is not None:
            if len(self.qubit_error_probs) != len(the_code.dataqubits):
                raise ValueError("q_error_probs array has incorrect length")

    def _mk_df(
        self, syndrome: Optional[np.ndarray] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create DataFrame used for creating plots."""
        if syndrome is not None:
            assert syndrome.shape == (len(self.code.stabgens),)
        v_items = []
        i = 0
        for v in self.code.vertices:
            item: dict[str, Any] = dict(x=v.pos[0], y=v.pos[1], hovertext=None)
            # TODO HTML escape things passed to plotly, such as v.group.name
            if isinstance(v, latticebase.DataVertex):
                item["type"] = "data"
                item["type_group"] = "data"
                if self.qubit_error_probs is not None:
                    item["hovertext"] = (
                        f"Data qubit {v.dataqubit_idx}<br />Ext. qubit {v.equbit_idx}"
                        f"<br />p = {self.qubit_error_probs[i]}"
                    )
                else:
                    item[
                        "hovertext"
                    ] = f"Data qubit {v.dataqubit_idx}<br />Ext. qubit {v.equbit_idx}"
                i += 1
            elif isinstance(v, latticebase.StabGenVertex):
                item["type"] = "stab"
                item["type_group"] = f"stab_{v.group.name}"
                item[
                    "hovertext"
                ] = f"Stabgen {v.stabgen_idx}<br />Ext. qubit {v.equbit_idx}"
                if syndrome is not None and syndrome[v.stabgen_idx]:
                    item["hovertext"] += "<br /><b>Syndrome set</b>"
                    item["type_group"] += "_toggled"
            elif isinstance(v, latticebase.LogicalVertex):
                item["type"] = "log"
                item["type_group"] = "log"
                item["hovertext"] = (
                    f"{v.name}<br />Logical {v.logical_idx}"
                    f"<br />Ext. qubit {v.equbit_idx}"
                )
            else:
                raise TypeError(f"Handling {type(v)} not implemented")
            v_items.append(item)
        e_items = []
        for e in self.code.edges:
            e_items.append(dict(op=e.factor.name, x=e.op.pos[0], y=e.op.pos[1]))
            e_items.append(dict(op=e.factor.name, x=e.data.pos[0], y=e.data.pos[1]))
            e_items.append(dict(op=e.factor.name, x=np.nan, y=np.nan))
        v_df = (
            pd.DataFrame(v_items)
            if v_items
            else pd.DataFrame([], columns=["type_group"])
        )
        e_df = pd.DataFrame(e_items) if e_items else pd.DataFrame([], columns=["op"])
        return v_df, e_df

    #: Offsets for plotting error/correction markers on data qubits
    _frame_offsets = {
        "p_err_X": (0.15, -0.15),
        "p_err_Z": (-0.15, -0.15),
        "p_corr_X": (0.15, 0.15),
        "p_corr_Z": (-0.15, 0.15),
    }

    def _mk_pauliframe_df(
        self,
        err: Optional[Sequence[Tableau]],
        corr: Optional[Sequence[Tableau]],
        only_pauli: Optional[str],
    ) -> pd.DataFrame:
        """Create DataFrame to plot Pauli frames (errors and corrections)."""
        assert err is None or len(err) == 1
        assert corr is None or len(corr) == 1
        assert only_pauli in (None, "X", "Z")
        items = []
        for kind, frame in (("p_err", err), ("p_corr", corr)):
            if frame is None:
                continue

            ops: list[list[tuple[int, str]]] = list()
            for op in frame:
                x, z, _ = unpack_tableau(op)
                ops.append(list())
                for i, (x_bit, z_bit) in enumerate(zip(x, z)):
                    if x_bit:
                        ops[-1].append((i, "X"))
                    if z_bit:
                        ops[-1].append((i, "Z"))
            assert len(ops) == 1
            for pos, pauli in ops[0]:
                assert pauli in ("X", "Z")
                if only_pauli and pauli != only_pauli:
                    continue
                q = self.code.dataqubits[pos]
                item: dict[str, Any] = dict()
                item["type"] = kind
                item["type_group"] = f"{kind}_{pauli}"
                off = self._frame_offsets[item["type_group"]]
                item["x"] = q.pos[0] + off[0]
                item["y"] = q.pos[1] + off[1]
                item["hovertext"] = f"Data qubit {pos}"
                items.append(item)
        return pd.DataFrame(items)

    #: Plotly options for drawing vertices
    vertex_kw = {
        "data": dict(
            name="Data qubit", marker=dict(color="firebrick", symbol="circle", size=25)
        ),
        "stab_U": dict(
            name="Stabilizer", marker=dict(color="midnightblue", symbol="x", size=10)
        ),
        "stab_A": dict(
            name="A stabilizer", marker=dict(color="midnightblue", symbol="x", size=10)
        ),
        "stab_B": dict(
            name="B stabilizer", marker=dict(color="green", symbol="cross", size=10)
        ),
        "stab_U_toggled": dict(
            name="Toggled stabilizer", marker=dict(color="hotpink", symbol="x", size=13)
        ),
        "stab_A_toggled": dict(
            name="Toggled A stabilizer",
            marker=dict(color="deepskyblue", symbol="x", size=10),
        ),
        "stab_B_toggled": dict(
            name="Toggled B stabilizer",
            marker=dict(color="fuchsia", symbol="cross", size=10),
        ),
        "log": dict(
            name="Logical operator", marker=dict(color="orange", symbol="star", size=15)
        ),
        "p_err_X": dict(
            name="X error",
            marker=dict(color="red", symbol="x", size=7),
        ),
        "p_err_Z": dict(
            name="Z error",
            marker=dict(color="red", symbol="cross", size=7),
        ),
        "p_corr_X": dict(
            name="X correction",
            marker=dict(color="limegreen", symbol="x", size=7),
        ),
        "p_corr_Z": dict(
            name="Z correction",
            marker=dict(color="limegreen", symbol="cross", size=7),
        ),
    }

    #: Plotly options for drawing edges
    edge_kw = {
        "X": dict(name="Pauli X", line=dict(color="cornflowerblue", dash="solid")),
        "Z": dict(name="Pauli Z", line=dict(color="yellowgreen", dash="5px 5px")),
    }

    _vertex_kw: Optional[dict] = None
    _edge_kw: Optional[dict] = None

    def _set_grey(self):
        """Set colors to grey except for syndrome bits, errors and corrections."""
        assert self._vertex_kw is None
        assert self._edge_kw is None
        self._vertex_kw = self.vertex_kw
        self._edge_kw = self.edge_kw
        self.vertex_kw = self.vertex_kw.copy()
        self.edge_kw = self.edge_kw.copy()

        for name in self.vertex_kw:
            if name.startswith("p_") or name.endswith("_toggled"):
                continue
            kw = self.vertex_kw[name] = self.vertex_kw[name].copy()
            m = kw["marker"] = kw["marker"].copy()
            if name.startswith("stab_"):
                m["color"] = "darkgrey"
            else:
                m["color"] = "lightgrey"

        for name in self.edge_kw:
            kw = self.edge_kw[name] = self.edge_kw[name].copy()
            m = kw["line"] = kw["line"].copy()
            m["color"] = "lightgrey"

    def _reset_grey(self):
        """Reset grey colors."""
        assert self._vertex_kw is not None
        assert self._edge_kw is not None
        self.vertex_kw = self._vertex_kw
        self.edge_kw = self._edge_kw
        self._vertex_kw = None
        self._edge_kw = None

    def _draw(self, vdf: pd.DataFrame, edf: pd.DataFrame, height: int, margin: int):
        """Internal drawing function (Plotly)."""
        fig = go.Figure()
        fig.update_layout(height=height, margin={k: margin for k in "lrtb"})
        fig.update_yaxes(scaleanchor="x", scaleratio=1.0)
        for key, gr in edf.groupby("op"):
            key = cast(str, key)
            fig.add_trace(
                go.Scatter(x=gr["x"], y=gr["y"], mode="lines", **self.edge_kw[key])
            )
        i = 0
        fig.update_layout(coloraxis_colorbar_x=-0.1)
        for key, gr in vdf.groupby("type_group"):
            # Could supply hovertext=["a", "b", ...] here.
            if key == "data" and self.qubit_error_probs is not None:
                fig.add_trace(
                    go.Scatter(
                        x=gr["x"],
                        y=gr["y"],
                        mode="markers",
                        name="Data qubit",
                        marker=dict(
                            colorscale="Viridis",
                            color=self.qubit_error_probs,
                            symbol="circle",
                            size=15,
                            colorbar=dict(thickness=20, title="Error Probs.", x=-0.2),
                        ),
                        hovertext=gr["hovertext"],
                    )
                )
                i += 1
            else:
                fig.add_trace(
                    go.Scatter(
                        x=gr["x"],
                        y=gr["y"],
                        mode="markers",
                        **self.vertex_kw[cast(str, key)],
                        hovertext=gr["hovertext"],
                    )
                )
        return fig

    def draw_lattice(self, height: int = 600, margin: int = 20):
        """Draw the code with Plotly.

        Args:
            height: Height of the figure in pixels.
            margin: Margin of the figure in pixels.

        Example:
            See :ref:`viz-guide`.
        """
        vdf, edf = self._mk_df()
        return self._draw(vdf, edf, height, margin)

    def draw_latticedata(
        self,
        syndrome: Optional[np.ndarray] = None,
        error: Optional[Tableau] = None,
        correction: Optional[Tableau] = None,
        only_pauli: Optional[str] = None,
        grey: bool = True,
        draw_stabs: bool = True,
        draw_edges: bool = True,
        height: int = 600,
        margin: int = 20,
    ):
        """Draw the code with error and correction Pauli frame (Plotly).

        Args:
            syndrome: The syndrome to be drawn.
            error: The error to be drawn.
            correction: The correction to be drawn.
            only_pauli: Supply ``"X"`` or ``"Z"`` to draw only the given Pauli error.
            grey: Everything except syndrome, error and correction should be grey.
            draw_stabs: Whether stabilizer markers should be drawn.
            draw_edges: Whether edges (from stabilizers to data qubits) should be drawn.
            height: Height of the figure in pixels.
            margin: Margin of the figure in pixels.

        Example:
            See :ref:`viz-guide`.
        """
        vdf, edf = self._mk_df(syndrome)
        if not draw_stabs:
            vdf = vdf[vdf.type == "data"]
        if not draw_edges:
            edf = edf.iloc[:0]
        vdf = pd.concat((vdf, self._mk_pauliframe_df(error, correction, only_pauli)))
        if grey:
            self._set_grey()
        fig = self._draw(vdf, edf, height, margin)
        if grey:
            self._reset_grey()
        return fig

    #: Matplotlib options for drawing vertices
    mpl_vertex_kw = {
        "data": dict(
            label="Data qubit",
            linestyle="None",
            color="firebrick",
            marker="o",
            markersize=10,
        ),
        "stab_U": dict(
            label="Stabilizer",
            linestyle="None",
            color="midnightblue",
            marker="X",
            markersize=8,
        ),
        "stab_A": dict(
            label="A stabilizer",
            linestyle="None",
            color="midnightblue",
            marker="X",
            markersize=8,
        ),
        "stab_B": dict(
            label="B stabilizer",
            linestyle="None",
            color="green",
            marker="P",
            markersize=8,
        ),
        "log": dict(
            label="Logical operator",
            linestyle="None",
            color="orange",
            marker="*",
            markersize=12,
        ),
    }

    #: Matplotlib options for drawing edges
    mpl_edge_kw = {
        "X": dict(label="Pauli X", linestyle="solid", color="cornflowerblue"),
        "Z": dict(label="Pauli Z", linestyle="dashed", color="yellowgreen"),
    }

    def draw_lattice_mpl(self, ax: OptSubAx = None):
        """Draw the code with Matplotlib.

        Args:
            ax: Matplotlib axes on which the code should be drawn (optional).
        """
        ax = self._get_ax(ax)
        vdf, edf = self._mk_df()
        for key, gr in edf.groupby("op"):
            ax.plot(gr["x"], gr["y"], **self.mpl_edge_kw[str(key)])
        for key, gr in vdf.groupby("type_group"):
            ax.plot(gr["x"], gr["y"], **self.mpl_vertex_kw[str(key)])
        ax.legend()

    def _get_ax(self, ax: OptSubAx = None) -> mpl.axes.Subplot:
        """Get default Matplotlib axes and set axis limits for lattice."""
        if ax is None:
            ax = plt.axes()
        ax.set(
            xlim=[-1, 1.4 * self.code.lattice.shape[0]],
            ylim=[-1, self.code.lattice.shape[1]],
        )
        return ax


class CircuitVisualizer:
    """Visualiser for quant circuits.

    Notes:
        The circuit visualiser currently is a very thin wrapper around the
        ``'mpl'`` backend from ``qiskit``.
    """

    def __init__(self, circuit: circuit.Circuit):
        """Load a circuit for visualisation purposes.

        Notes:
            Qiskit can draw OpenQASM 3.0 circuits, but the header we use in our
            :func:`.convert_to_openqasm` is for OpenQASM 2.0. The visualiser
            automatically strips the 2.0 header and substitutes the 3.0 one,
            otherwise circuits would not render.

        Args:
            circuit: a :class:`~.circuit.Circuit` to draw.
        """
        qsm = openqasm.convert_to_openqasm(circuit).split("\n")
        qsm[0] = "OPENQASM 3.0;"
        qsm[1] = 'include "stdgates.inc";'
        qsm = "\n".join(qsm)
        self._circuit: qiskit.circuit.quantumcircuit.QuantumCircuit = qasm3.loads(qsm)

    def draw_circuit(self):
        """Draw a previously-loaded circuit with the Qiskit matplotlib drawer."""
        return self._circuit.draw("mpl")
