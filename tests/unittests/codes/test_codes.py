import numpy as np
import pytest as pt

from plaquette import pauli
from plaquette.codes import Code, QubitType


@pt.fixture
def barebone_planar_code():
    """Create stabilisers and logical operators for a size-2 planar code."""
    x_stabs = [pauli.string_to_pauli("X0X2X3", 5), pauli.string_to_pauli("X1X2X4", 5)]
    z_stabs = [pauli.string_to_pauli("Z0Z1Z2", 5), pauli.string_to_pauli("Z2Z3Z4", 5)]
    logical_x = pauli.string_to_pauli("X0X1", 5)
    logical_z = pauli.string_to_pauli("Z0Z3", 5)
    return x_stabs, z_stabs, logical_x, logical_z


class TestSubsystemCodes:
    def test_creation_from_stabilisers(self, barebone_planar_code):
        """Make sure a valid code can be generated from stabilisers alone."""
        xst, zst, xlog, zlog = barebone_planar_code
        sc = Code(stabilizers=xst + zst, logical_ops=[xlog, zlog])

        assert sc.num_data_qubits == 5
        assert sc.num_stabilizers == 4
        assert sc.num_logical_qubits == 1
        assert sc.distance == 2

        assert sc.num_qubits == 9
        assert sc.code_parameters == (5, 1, 0, 2)

        assert sc.is_stabiliser_code

        # The indexing starts with data and ends with ancillas
        assert sc.tanner_graph.nodes_data[3].type == QubitType.data
        assert sc.tanner_graph.nodes_data[3].coords is None  # by default
        assert sc.tanner_graph.nodes_data[5].type == QubitType.stabilizer
        # Edges are created starting from the lowest index in the stabiliser
        # and always (ancilla, data). Since we provide the X stabilisers first
        # in the code constructor, the first edge ever created is going to be
        # an X pauli factor
        assert sc.tanner_graph.edges_data[0].type == pauli.Factor.X
        assert sc.tanner_graph.get_vertices_connected_by_edge(0) == (5, 0)
        assert sc.tanner_graph.edges_data[6].type == pauli.Factor.Z
        assert sc.tanner_graph.get_vertices_connected_by_edge(6) == (7, 0)


class TestCommonCodes:
    def test_planar_code_3_square(self):
        code = Code.make_planar(distance=3)

        # check code paramters
        assert code.num_data_qubits == 13
        assert code.num_stabilizers == 12
        assert code.num_logical_qubits == 1
        assert code.distance == 3

        # check graph parameters
        assert code.embedded_graph.num_nodes == code.num_qubits
        assert (
            code.tanner_graph.get_num_vertices()
            == code.num_data_qubits + code.num_stabilizers
        )

        # check stabilisers
        for op, log in zip(code.logical_ops, ["X0X1X2", "Z0Z5Z10"]):
            assert np.array_equal(op, pauli.string_to_pauli(log, qubits=13))

        correct_stabs = [
            "X0X3X5",
            "X1X3X4X6",
            "X2X4X7",
            "X5X8X10",
            "X6X8X9X11",
            "X7X9X12",
        ] + ["Z0Z1Z3", "Z1Z2Z4", "Z3Z5Z6Z8", "Z4Z6Z7Z9", "Z8Z10Z11", "Z9Z11Z12"]
        for op, stab in zip(code.stabilizers, correct_stabs):
            assert np.array_equal(op, pauli.string_to_pauli(stab, qubits=13))

        assert code.factorized_checks == [[] for _ in range(12)]

        for op, correct in zip(code.measured_operators, correct_stabs):
            assert np.array_equal(
                pauli.dict_to_pauli(op[0]), pauli.string_to_pauli(correct)
            )

        assert code.embedded_graph.num_nodes == code.tanner_graph.get_num_vertices()
        # for the planar code the embedded graph is equal to the tanner graph

        assert code.embedded_graph.get_num_edges() == code.tanner_graph.get_num_edges()

        for i in range(code.embedded_graph.num_nodes):
            assert (
                len(code.embedded_graph.get_vertices_touching_vertex(i))
                == code.tanner_graph.get_vertices_touching_vertex(i).size()
            )

    def test_planar_code_rectangle_3_5(self):
        code = Code.make_planar(distance=(3, 5))

        assert code.num_data_qubits == 23
        assert code.num_stabilizers == 22
        assert code.num_logical_qubits == 1
        assert code.distance == 3

        assert code.embedded_graph.num_nodes == code.num_qubits
        assert (
            code.tanner_graph.get_num_vertices()
            == code.num_data_qubits + code.num_stabilizers
        )

        assert code.embedded_graph.num_nodes == code.tanner_graph.get_num_vertices()
        # for the planar code the embedded graph is equal to the tanner graph

        assert code.embedded_graph.get_num_edges() == code.tanner_graph.get_num_edges()

        for i in range(code.embedded_graph.num_nodes):
            assert (
                len(code.embedded_graph.get_vertices_touching_vertex(i))
                == code.tanner_graph.get_vertices_touching_vertex(i).size()
            )

    def test_planar_code_rectangle_5_5(self):
        code = Code.make_planar(distance=(5, 3))

        assert code.num_data_qubits == 23
        assert code.num_stabilizers == 22
        assert code.num_logical_qubits == 1
        assert code.distance == 3

        assert code.embedded_graph.num_nodes == code.num_qubits
        assert (
            code.tanner_graph.get_num_vertices()
            == code.num_data_qubits + code.num_stabilizers
        )

        assert code.embedded_graph.num_nodes == code.tanner_graph.get_num_vertices()
        # for the planar code the embedded graph is equal to the tanner graph

        assert code.embedded_graph.get_num_edges() == code.tanner_graph.get_num_edges()

        for i in range(code.embedded_graph.num_nodes):
            assert (
                len(code.embedded_graph.get_vertices_touching_vertex(i))
                == code.tanner_graph.get_vertices_touching_vertex(i).size()
            )

    def test_rotated_planar_3_square(self):
        code = Code.make_rotated_planar(distance=3, xzzx=False)

        assert code.num_data_qubits == 9
        assert code.num_stabilizers == 8
        assert code.num_logical_qubits == 1
        assert code.distance == 3

        assert code.embedded_graph.num_nodes == code.num_qubits
        assert (
            code.tanner_graph.get_num_vertices()
            == code.num_data_qubits + code.num_stabilizers
        )

        for op, log in zip(code.logical_ops, ["X0X3XX6", "Z0Z1Z2"]):
            assert np.array_equal(op, pauli.string_to_pauli(log, qubits=9))

        correct_stabs = [
            "X0X1",
            "X1X2X4X5",
            "X3X4X6X7",
            "X7X8",
        ] + ["Z0Z1Z3Z4", "Z2Z5", "Z3Z6", "Z4Z5Z7Z8"]
        for op, stab in zip(code.stabilizers, correct_stabs):
            assert np.array_equal(op, pauli.string_to_pauli(stab, qubits=9))

        assert code.factorized_checks == [[] for _ in range(8)]

        for op, correct in zip(code.measured_operators, correct_stabs):
            assert np.array_equal(
                pauli.dict_to_pauli(op[0]), pauli.string_to_pauli(correct)
            )

        assert code.embedded_graph.num_nodes == code.tanner_graph.get_num_vertices()
        # for the planar code the embedded graph is equal to the tanner graph

        assert code.embedded_graph.get_num_edges() == code.tanner_graph.get_num_edges()

        for i in range(code.embedded_graph.num_nodes):
            assert (
                len(code.embedded_graph.get_vertices_touching_vertex(i))
                == code.tanner_graph.get_vertices_touching_vertex(i).size()
            )

    def test_rotated_planar_3_5_rectangle(self):
        code = Code.make_rotated_planar(distance=(5, 7), xzzx=False)

        assert code.num_data_qubits == 35
        assert code.num_stabilizers == 34
        assert code.num_logical_qubits == 1
        assert code.distance == 5

        assert code.embedded_graph.num_nodes == code.num_qubits
        assert (
            code.tanner_graph.get_num_vertices()
            == code.num_data_qubits + code.num_stabilizers
        )

        assert code.embedded_graph.num_nodes == code.tanner_graph.get_num_vertices()
        # for the planar code the embedded graph is equal to the tanner graph

        assert code.embedded_graph.get_num_edges() == code.tanner_graph.get_num_edges()

        for i in range(code.embedded_graph.num_nodes):
            assert (
                len(code.embedded_graph.get_vertices_touching_vertex(i))
                == code.tanner_graph.get_vertices_touching_vertex(i).size()
            )

    def test_rotated_planar_5_3_rectangle(self):
        code = Code.make_rotated_planar(distance=(7, 5), xzzx=False)

        assert code.num_data_qubits == 35
        assert code.num_stabilizers == 34
        assert code.num_logical_qubits == 1
        assert code.distance == 5

        assert code.embedded_graph.num_nodes == code.num_qubits
        assert (
            code.tanner_graph.get_num_vertices()
            == code.num_data_qubits + code.num_stabilizers
        )

        assert code.embedded_graph.num_nodes == code.tanner_graph.get_num_vertices()
        # for the planar code the embedded graph is equal to the tanner graph

        assert code.embedded_graph.get_num_edges() == code.tanner_graph.get_num_edges()

        for i in range(code.embedded_graph.num_nodes):
            assert (
                len(code.embedded_graph.get_vertices_touching_vertex(i))
                == code.tanner_graph.get_vertices_touching_vertex(i).size()
            )

    def test_steane_code_non_compact(self):
        code = Code.make_steane(compact=False)

        assert code.num_data_qubits == 7
        assert code.num_stabilizers == 6
        assert code.num_logical_qubits == 1
        assert code.distance == 3

        assert code.embedded_graph.num_nodes == code.num_qubits == 13
        assert (
            code.tanner_graph.get_num_vertices()
            == code.num_data_qubits + code.num_stabilizers
        )

        assert code.ancilla_supports == [[i] for i in range(len(code.stabilizers))]

        correct_stabs = [
            "X0X1X3X4",
            "X1X2X4X5",
            "X3X4X5X6",
            "Z0Z1Z3Z4",
            "Z1Z2Z4Z5",
            "Z3Z4Z5Z6",
        ]

        for op, log in zip(code.logical_ops, ["X0X1X2", "Z0Z1Z2"]):
            assert np.array_equal(op, pauli.string_to_pauli(log, qubits=7))

        for op, stab in zip(code.stabilizers, correct_stabs):
            assert np.array_equal(op, pauli.string_to_pauli(stab, qubits=7))

        assert code.factorized_checks == [[] for _ in range(len(code.stabilizers))]

    def test_steane_code_compact(self):
        code = Code.make_steane(compact=True)

        assert code.num_data_qubits == 7
        assert code.num_stabilizers == 6
        assert code.num_logical_qubits == 1
        assert code.distance == 3

        assert code.embedded_graph.num_nodes == code.num_qubits == 10
        assert (
            code.tanner_graph.get_num_vertices()
            == code.num_data_qubits + code.num_stabilizers
        )

        assert code.factorized_checks == [[] for _ in range(len(code.stabilizers))]
        assert code.ancilla_supports == [
            [i, i + 3] for i in range(len(code.stabilizers) // 2)
        ]

        correct_stabs = [
            "X0X1X3X4",
            "X1X2X4X5",
            "X3X4X5X6",
            "Z0Z1Z3Z4",
            "Z1Z2Z4Z5",
            "Z3Z4Z5Z6",
        ]

        for op, log in zip(code.logical_ops, ["X0X1X2", "Z0Z1Z2"]):
            assert np.array_equal(op, pauli.string_to_pauli(log, qubits=7))

        for op, stab in zip(code.stabilizers, correct_stabs):
            assert np.array_equal(op, pauli.string_to_pauli(stab, qubits=7))
