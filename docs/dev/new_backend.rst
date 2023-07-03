.. Copyright 2023, QC Design GmbH and the plaquette contributors
   SPDX-License-Identifier: Apache-2.0


How to create a new backend?
============================

The error correction pipeline of plaquette involves running the quantum circuit that represents a quantum error correcting code. To run the quantum circuit on a quantum device, plaquette provides the ``Device`` object which can be used with various backends, for example with the Clifford local backend:

.. code-block:: python

    from plaquette import Device
    device = Device("clifford")
    device.run(circuit)

The ``Device`` object maintains a backend object under the hood. Although
plaquette comes with built-in backends and it is also extensible with new
backends which can be developed and integrated into plaquette via the following
steps:

1. Create a new backend class (we suggest naming the backend class such that it has ``Backend`` as a suffix in its name);
2. Implement the set of required (and in addition the optional) backend methods and properties (as detailed below);
3. Package your new device backend into a separate Python package;
4. In the ``pyproject.toml`` file of the Python package

The following section may be placed in the ``pyproject.toml`` file:

.. code-block:: toml

    [project.entry-points."plaquette.device"]
    my_backend_name = "plaquette_my_backend.backend:MyBackend"

where ``my_backend_name`` will be the name that can be passed to ``Device`` to use the new backend, the new backend package name is ``plaquette_my_backend`` and the backend implementation is represented with the ``MyBackend`` class which is placed in the ``backend`` module of the package.

We refer to the `plaquette-ibm-backend <https://github.com/qc-design/plaquette-ibm-backend>`_ package as an example of a backend for exact packaging details.

Backend API
-----------

The set of required backend methods and their signatures are:
* ``run(self, circuit: plaq_circuit.Circuit | plaq_circuit.CircuitBuilder, *,shots=1)``: Run the given circuit.
* ``get_sample(self) -> Tuple[List[Union[List[Any], Any]], Optional[List[None]]]``: Return the samples **after a circuit run**.

Remote backends may also implement the ``is_completed(self) -> List[bool]``
property that determines which jobs have been completed for the list of jobs
submitted using the device.

Simulators that allow obtaining the underlying quantum state of the backend may
also define the ``state`` property to access such a state.

Furthermore, all arguments and keyword arguments passed to ``Device`` upon creation are later passed to the underlying backend object. Therefore, custom methods and properties may also be implemented in new backends that may take arguments passed at the time of creation.

If you have any suggestions or questions related to creating new device backends, feel free to open a new GitHub issue!
