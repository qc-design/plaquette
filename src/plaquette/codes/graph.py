"""Pydantic models for the visualizer objects.

The visualizer is based on Web technologies, and as such it uses a
huge amount of JSON for data interchange. This data needs to match
what the JS side of things expect and to do this we use Pydantic
to make proper JSON Schema documents to ease development.
"""
import typing as t

import numpy as np
import pydantic


class Position(pydantic.BaseModel):
    """Qubit position in 3D space.

    For foliated codes, ``Z`` is the "time" directions.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other):
        """Add two vectors component-wise."""
        if not isinstance(other, Position):
            raise TypeError("only Position objects can be added")
        return Position(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)

    def __mul__(self, other):
        """Multiply a vector with a scalar."""
        return Position(x=self.x * other, y=self.y * other, z=self.z * other)

    def __sub__(self, other):
        """Subtract two vectors component-wise."""
        if not isinstance(other, Position):
            raise TypeError("only Position objects can be subtracted")
        return Position(x=self.x - other.x, y=self.y - other.y, z=self.z - other.z)

    def __truediv__(self, other):
        """Divide a vector with by scalar."""
        return Position(x=self.x / other, y=self.y / other, z=self.z / other)

    def __iter__(self):
        """Support unpacking vector components."""
        return iter((self.x, self.y, self.z))

    def unit(self):
        """Normalise a vector.

        Raises:
            ValueError: if you try to normalise the null-vector.
        """
        if norm := np.sqrt(self.x**2 + self.y**2 + self.z**2):
            return self / norm
        raise ValueError("null-vector cannot be normalised")


class Node(pydantic.BaseModel):
    """Node properties."""

    pos: Position
    """Position w.r.t. the origin of the graph."""
    type: str
    """Type of node, for styling purposes.

    This is a free-form "tag", which can be used to select/filter a
    group of nodes out of a graph."""


class Edge(pydantic.BaseModel):
    """An edge between nodes."""

    a: int
    """The index of the first node in the list of the graph nodes."""
    b: int
    """The index of the second node in the list of the graph nodes."""
    type: str
    """Type of edge, for styling purposes.

    This is a free-form "tag", which can be used to select/filter a
    group of edges out of a graph."""

    def __iter__(self):
        """Support unpacking edge nodes."""
        return iter((self.a, self.b))


class BaseGraph(pydantic.BaseModel):
    """Dictionary schema to serialize a graph."""

    nodes: list[Node]
    """All nodes making up this graph."""
    edges: list[Edge]
    """All edges making up this graph."""


class TannerGraph(BaseGraph):
    """The graph linking checks/ancillas and data qubits in QEC code."""

    checks: list[list[int]] | None
    """List of nodes forming a single "check".

    The outer list is the list of all checks. Each check is defined
    by a list of integers referring to node indices.
    """


class DecodingGraph(BaseGraph):
    """A decoding graph."""

    selection: list[int]
    """The edges being selected from a decoder."""
    faults: list[int]
    """The nodes that have been "flipped" by errors."""
    virtual_nodes: list[int]
    """Indices of those nodes which have been artificially added for parity.

    The indices in this list should refer to the list of :attr:`nodes`.
    """


def center_of_mass(
    points: t.Sequence[Position], masses: t.Sequence[int] | None = None
) -> Position:
    """Calculate the center of mass among the given points.

    Args:
        points: the position where the objects are.
        masses: mass of each object. If ``None``, it will assume a mass
            of 1 for each object. Should be integers.
    """
    if masses is None:
        masses = np.ones(len(points), dtype=int)
    if len(masses) != len(points):
        raise ValueError("Points and masses arrays must be of same length")

    return np.sum(np.array(points) * np.array(masses)) / np.sum(masses)
