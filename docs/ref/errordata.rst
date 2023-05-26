.. Copyright 2023, It'sQ GmbH and the plaquette contributors
   SPDX-License-Identifier: Apache-2.0

.. _errordata-ref:

Working with errors in plaquette
================================

The module :mod:`~plaquette.errors` provides an extensible framework for the user to
provide detailed information regarding the physical error model on each individual
qubits. This information regarding the errors is packaged in two different
representations: a dictionary based one and a tabular one. The relevant wrapper
classes are

* :class:`~plaquette.errors.ErrorDataDict` - A ``TypedDict`` subclass with two
  keys, one each for qubit type errors and gate type errors.
* :class:`~plaquette.errors.ErrorData` - A :func:`~dataclasses.dataclass` with
  a :class:`~pandas.DataFrame` each for qubit type errors and gate type errors.

How are errors modelled in plaquette?
-------------------------------------

For the purposes of simulation, we consider two broad types of errors, which we refer
to as *qubit errors* and *gate errors*. *Qubit errors* model errors that occur on
individual qubits due to their fragility. The available qubit errors are the Pauli
channels, the erasure channel, and measurement errors. *Gate errors* represent
the modelling of faulty gate. This is modelled by the application of a perfect
gate, followed a probabilistic error channel.

The dictionaries are nested in multiple levels.

* The top most level contains keys differentiating between qubit and gate errors
* The next layer's keys differentiates either between the type of channel of qubit
  errors or the gate that is faulty.
* The next layer keys identifies the qubit.
* The final layer's keys differentiate how the errors for that particular type are
  parametrized. For example, erasure gates are parametrized by the erasure probability,
  whereas two qubit gate like CX is parametrized by the probabilities of each two
  qubit Pauli channel (see :class:`.TwoPauliChannelErrorValueDict`).

The Qubit Error Database and its inputs
---------------------------------------

The qubit error table has the following the columns. Not every columns needs to
be mentioned. The attribute :attr:`~.ErrorData.enabled_qubit_errors` is set to
the names of the columns that are available, barring ``qubit_id`` and
``qubit_type``.

.. table::

   ===========  ==========  ==================================
   Header        Types      Description
   ===========  ==========  ==================================
   qubit_id        int      index of qubit
   qubit_type      str      data or ancilla
   X              float     probability of X error
   Y              float     probability of Y error
   Z              float     probability of Z error
   erasure        float     probability of erasure error
   measurement    float     probability of measurement error
   fabrication     str      missing or available
   ===========  ==========  ==================================

A sample table looks like this.

.. table::

   ========  ==========  ===  ===  ===  =======  ===========  ===========
   qubit_id  qubit_type   X    Y    Z   erasure  measurement  fabrication
   ========  ==========  ===  ===  ===  =======  ===========  ===========
          0  data        0.1  NaN  0.1      0.1          NaN    available
          1  ancilla     0.1  0.1  0.1      0.1         0.03    available
   ========  ==========  ===  ===  ===  =======  ===========  ===========

You can some values are missing in the measurement. The missing values are set
to NaN and ignored during the simulations. To use the data from the csv, one
can use :meth:`.ErrorData.from_csv` to
create a new object. To update an existing object using a csv, one can use the
method :meth:`.ErrorData.update_from_csv`.
Apart from using CSV files, qubit errors can also be added programmatically
using :meth:`.ErrorData.add_qubit_error` if the column is not present, or
:meth:`.ErrorData.update_qubit_error` if the column is present.

.. seealso:: The page :doc:`/advanced/errors/index` has more hands-on examples!

The user also has the ability to load errors from a CSV file. A sample CSV for
the ``RotatedPlanarCode`` of distance 3 is shown below

.. csv-table:: Rotated Planar Code of distance 3
   :file: ../advanced/errors/spem.csv

You can see some values are missing in the measurement. The missing values are
set to NaN and ignored during the simulations. To use the data from the CSV,
one can use  :meth:`.ErrorData.from_csv` to create a new object. To update an
existing object using a CSV file, one can use the method
:meth:`~.ErrorData.update_from_csv`. Apart from using `csv`, qubit errors
can also be added directly through code using: `add_qubit_error()` if the
column is not present and :meth:`~.ErrorData.update_qubit_error()` if the
column is present. See the tutorial for more examples.

The Gate Error Database and its inputs
--------------------------------------

The gate error table has the following the columns. Here every column needs to be mentioned.
The variable ``enabled_gate_errors`` is set to the unique values from the gate column

.. table::

    ===============  ============  ========================================
    Header            Types         Description
    ===============  ============  ========================================
    gate                str         The erroneous gates. {CX,CZ,H,R,M}
    on_qubits         list[int]     the list of qubits
    induced_errors    list[str]     the list of induced errors by the gate
    probs            list[float]    probability of each induced error
    ===============  ============  ========================================

The ``on_qubits`` variable for two qubit gates is specified as:
``[control1, target1, control2, target2 ,..]`` and the ``induced_errors`` are
length-2 Pauli strings.

The ``csv`` spec for the gate is as follows. The qubits and induced_errors are
seperated by the
pipe(|)  instead of a comma(,)

.. csv-table:: Rotated Planar Code of distance 3
    :file: ../advanced/errors/gate_errors.csv


Apart from using the `csv`, the user can also use the to `add_gate_error()` function to add an
error to the gate error table. In the case of this function, the error gets simply append to the
end of the table. If there are two values for the same gate error, the one added last to the
table takes precedence.

Internally within in `plaquette`, the submodules currently interface with
`ErrorDataDict` instead of `ErrorData`. There is method to generate the `ErrorDataDict`
dictionary to pass onto other objects in ``plaquette``









