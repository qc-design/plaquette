.. Copyright 2023, It'sQ GmbH and the plaquette contributors
   SPDX-License-Identifier: Apache-2.0

.. _frontend-ref:

Configuration classes in the frontend
=====================================

This page provides details on how to use the frontend to run simulations for using
:mod:`~plaquette.frontend`. To run an experiment on Quantum Error Correction, there
exist five things that need to be decided.

1. The Error correcting code that needs to be simulated.
2. The errors that the system can experience a.k.a Error Model.
3. The circuit that is being simulated. Here the circuit contains components to
   prepare the logical state, perform the syndrome measurements and measure the
   logical state. In a circuit based error model, errors are added as stochastic
   gates between each timestep.
4. The simulator being used to simulate the circuit.
5. The decoder being used to determine the correction.

In this module, config classes exist to specify each of the above stated requirements namely

- :class:`~plaquette.frontend.CodeConfig`
- :class:`~plaquette.frontend.ErrorsConfig`
- :class:`~plaquette.frontend.CircuitConfig`
- :class:`~plaquette.frontend.SimulatorConfig`
- :class:`~plaquette.frontend.DecoderConfig`

All of these classes have a ``from_dict()`` classmethod and an ``instantiate()``
method. The ``from_dict()`` allows you to load the metadata from dictionary and the
``instantiate()`` method instantiates the appropriate class for the given metadata
from the relevant submodule in :mod:`~plaquette`. For instance, ``CodeConfig
(name="FiveQubitCode").instantiate()`` outputs a ``LatticeCode`` object from
:mod:`~plaquette.codes`.

There exists another config class :class:`~plaquette.frontend.ExperimentConfig` that packages
everything together and has functions that enable the user to run experiments with ease. See the
tutorial for more details on how to use this class.

Before we start, why?
---------------------

We want to make things clear: **everything in the frontend module can be done
by calling the appropriate functions and using the appropriate classes found
elsewhere in ``plaquette``**. The :mod:`.frontend` module exists to enable a
different workflow. This is a *declarative* workflow, in which you only care
about the *what* you want to simulate and let ``plaquette`` take care of the
*how*. You can describe your system with a series of simple configuration files
which ``plaquette`` can understand and it will then run the simulations for
you.

Why would you want to use this?

* a configuration file is a much simpler document to write for non-programmers.
  There is no flow control, no abstract programming concepts, types, etc.
  Understanding the structure of these configuration files is much faster than
  becoming a python programmer.
* these files a very human **and** machine friendly, allowing you to drive
  simulations using ``plaquette`` from outside of a python interpreter.
  Generate all the configuration files you want with the tools you decide
  (JS, Rust, Go, awk)!

Of course there is a price to pay for this flexibility, and you pay in
giving up some freedoms. In particular, you are limited to what ``plaquette``
comes equipped with, so no custom codes or other special utility constructs
that you would otherwise be able to build around ``plaquette`` in Python
itself.

The ``CodeConfig`` class
-------------------------

The code config class takes the following metadata as input

1. :attr:`~plaquette.frontend.CodeConfig.name` - The name of the code being simulated.
   The valid names are in the docstring of :class:`~plaquette.frontend.CodeConfig`.
2. :attr:`~plaquette.frontend.CodeConfig.size` - For some codes, you are able to
   specify the size of the code. If the code does not require a size it defaults to
   `-1`.
3. :class:`~plaquette.frontend.CodeConfig.rounds` - The number of rounds to repeat the
   syndrome measurement in each cycle.


The ``ErrorConfig`` class
--------------------------

The error config class takes as input two dictionaries, one configuration each for the qubit
errors and gate errors. See, ref errordata for the difference between the two

.. parsed-literal::

    qubit_error (dict[str, Union[str, dict]]): Config of qubit errors.
    The Schema for the same:
    {
        "data_path": path to csv load/save the data,
        "sample": Bool to determine whether to simulate qubit errors
                 Optional, if not provided defaults to True.
        "load_file": Bool to determine whether to load errors from the file
                     Optional. If not provided, sets to True if one of the "distribution" is
                     set to user.
        "X":{
            "enabled": bool to determine whether to simulate single qubit paulis
                        Optional, if not provided defaults to True
            "distribution": str # the distribution of the errors across the qubits.
            # Valid Distributions include ["user", "constant", "gaussian"]
            "params": list[float] the same length of paulis
             Optional variable to be provided if the distribution is not "user".
                      constant takes one value. gaussian takes a mean and std deviation
        },
        "Y": {
            "enabled": bool, Optional like aforementioned
            "distribution": str # from one of the valid distributions listed above
            "params": list[float]
                      Used in a similar spirit as above mentioned
                },
        "Z": {
            "enabled": bool, Optional like aforementioned
            "distribution": str # from one of the valid distributions listed above
            "params": list[float]
                      Used in a similar spirit as above mentioned
                },
        "erasure": {
            "enabled": bool, Optional like aforementioned
            "distribution": str # from one of the valid distributions listed above
            "params": list[float]
                      Used in a similar spirit as above mentioned
        },
        "measurement": {
            "enabled": bool, Optional like aforementioned
            "distribution": str # from one of the valid distributions listed above
            "params": list[float]
                      Used in a similar spirit as above mentioned
        },
    }

    gate_error (dict[str, str]): Config of two-qubit errors
    The schema for the same
    {
        "data_path": path to `csv` load/save the data,
        "sample": Bool to determine whether to simulate qubit errors
                 Optional, if not provided defaults to True.
        "load_file": Bool to determine whether to load errors from the file
                     Optional. If not provided, sets to True if one of the "distribution" is
                     set to user.
        "CZ":{
            "enabled": bool to determine whether to simulate CZ gate errors
                       Optional, if not provided defaults to True
            "distribution": list[str] | str
                            "user" is a valid option as string, errors load from the file
                            "constant" or "gaussian" is provided as list[str] and of the length
                            of induced_errors variable below.
            "induced_errors": list[str] The induced errors after the CZ gate.
                              Valid strings are length two from {I,X,Y,Z}, like ["IX", "YZ"]
            "params": list[list[float]] the same length of induced_errors
             Optional variable to be provided if the distribution is not "user".
                      constant takes one value. gaussian takes a mean and std deviation
        },
        "CX":{
            "enabled": bool to determine whether to simulate CX gate errors
                       Optional, if not provided defaults to True
            "distribution": list[str] | str
                            "user" is a valid option as string, errors load from the file
                            "constant" or "gaussian" is provided as list[str] and of the length
                            of induced_errors variable below.
            "induced_errors": list[str] The induced errors after the CX gate.
                              Valid strings are length two from {I,X,Y,Z}, like ["IX", "YZ"]
            "params": list[list[float]] the same length of induced_errors
             Optional variable to be provided if the distribution is not "user".
                      constant takes one value. gaussian takes a mean and std deviation
        },
        "H":{
            "enabled": bool to determine whether to simulate H gate errors
                       Optional, if not provided defaults to True
            "distribution": list[str] | str
                            "user" is a valid option as string, errors load from the file
                            "constant" or "gaussian" is provided as list[str] and of the length
                            of induced_errors variable below.
            "induced_errors": list[str] The induced errors after the H gate.
                              Valid strings are length one from {X,Y,Z}, like ["X", "Y"]
            "params": list[list[float]] the same length of induced_errors
             Optional variable to be provided if the distribution is not "user".
                      constant takes one value. gaussian takes a mean and std deviation
        },
        "R":{
            "enabled": bool to determine whether to simulate R (reset) gate errors
                       Optional, if not provided defaults to True
            "distribution": list[str] | str
                            "user" is a valid option as string, errors load from the file
                            "constant" or "gaussian" is provided as list[str] and of the length
                            of induced_errors variable below.
            "induced_errors": list[str] The induced errors after the R gate.
                              Valid strings are length one from {X,Y,Z}, like ["X", "Y"]
            "params": list[list[float]] the same length of induced_errors
                      Optional variable to be provided if the distribution is not "user".
                      constant takes one value. gaussian takes a mean and std deviation
        },
        "M":{
            "enabled": bool to determine whether to simulate R (reset) gate errors
                       Optional, if not provided defaults to True
            "distribution": list[str] | str
                            "user" is a valid option as string, errors load from the file
                            "constant" or "gaussian" is provided as list[str] and of the length
                            of induced_errors variable below.
            "induced_errors": list[str] The induced errors after the R gate.
                              Valid strings are length one from {X,Y,Z}, like ["X", "Y"]
            "params": list[list[float]] the same length of induced_errors
                      Optional variable to be provided if the distribution is not "user".
                      constant takes one value. gaussian takes a mean and std deviation
        }
    }



The ``CircuitConfig`` class
---------------------------

1. :attr:`~plaquette.frontend.CircuitConfig.circuit_provided` - A boolean to decide
   whether to load the file from disk.
2. :attr:`~plaquette.frontend.CircuitConfig.has_errors` - A boolean to decide
   whether the provided circuit has stochastic error gates
   included or if it must be added using
   :func:`~plaquette.circuit.generator.generate_qec_circuit`
3. :meth:`~plaquette.frontend.CircuitConfig.circuit_path` - The path to circuit. If the
   ``circuit_provided`` is false, this path will be used to save the circuit instead.

The ``SimulatorConfig`` class
------------------------------

1. :attr:`~plaquette.frontend.SimulatorConfig.name` - The name of the
   simulator  being used. Valid simulators can be found in the docstring
2. :attr:`~plaquette.frontend.SimulatorConfig.shots` - The number of shots to run
   the simulation for.


The ``DecoderConfig`` class
----------------------------

The decoder config class has the following two inputs

1. :attr:`~plaquette.frontend.DecoderConfig.name` - The name of the decoder being used.
   See the docstring reference for valid decoders.
2. :attr:`~plaquette.frontend.DecoderConfig.weighted` - A boolean that determines if
   the decoder uses weights or not.

All of these classes be loaded through python. However, at the point, it becomes
easier to just load the respective objects from the relevant submodules. The entire
metadata that is required can instead be provided through a ``toml`` config file, see an example here.


The ``ExperimentConfig`` class
------------------------------

The reference of the toml is provided below. The names follow from the above
descriptions of the config classes. Please refer to for more details.

.. code-block:: toml

    [general]
    logical_op = "Z" # the logical operator to measure
    qec_property = ["logical_error_rate"] # The QEC property to measure, currently only
                                          #  logical error rate is possible.
    seed = 123124 # the seed for the random number generator

    [simulator]
    name = "StimSimulator" # The simulator to use, see docstring for SimulatorConfig for valid names
    shots = 10000 # the number of shots to run the simulator for

    [code]
    name = "RotatedPlanarCode" # The code to use. See docstring of CodeConfig for valid names
    size = 3 # The size of the code
    rounds = 10 # The number rounds of syndrome measurement per QEC cycle

    [circuit]
    circuit_provided = false
    has_errors = false
    circuit_path = "/path/to/circuit.txt"

    [errors.qubit_errors]
    data_path = "/path/to/qubit_errors.csv"
    sample = true

    [errors.qubit_errors.X]
    distribution = "constant"
    params = [0.1]

    [errors.qubit_errors.Z]
    distribution = "user"

    [errors.qubit_errors.erasure]
    enabled = false
    distribution = "gaussian"
    params = [0.1, 0.01]

    [errors.qubit_errors.fabrication]
    distribution = "user"
    enabled = false

    [errors.qubit_errors.measurement]
    enabled = false
    distribution = "constant"
    params = [0.1]

    [errors.gate_errors]
    data_path = "path/to/gate/errors.csv"
    sample = false
    load_file = false

    [errors.gate_errors.CZ]
    induced_errors= ["XX", "ZI"]
    distribution = ["constant", "constant"]
    params =[[0.01], [0.01]]

    [errors.gate_errors.CX]
    induced_errors=["ZZ", "XI"]
    distribution = ["constant", "constant"]
    params =[[0.01], [0.01]]

    [errors.gate_errors.H]
    induced_errors=["X", "Z"]
    distribution=["constant", "constant"]
    params = [[0.01], [0.01]]

    [errors.gate_errors.fabrication]
    distribution = "user"
    enabled = false

    [decoder]
    name = "PyMatchingDecoder"
    weighted = false



Once we have the toml ready, we can load it into `ExperimentConfig` using the `load_toml()` method.
Once this is done, we can instantiate the necessary objects using `instantiate()` and the `run()`
function to get the logical error rate.

.. code-block:: python

    from plaquette.frontend import ExperimentConfig
    conf = ExperimentConfig.load_toml("/path/to/toml")
    conf.instantiate()
    conf.run()

The instantiated objects are properties of the class and
can be accessed as ``conf.code``, ``conf.errors``, ``conf.circuit``, ``conf.simulator``
and ``conf.decoder``. Since these are objects from the ``plaquette`` submodules,
the internal methods are also readily available.
