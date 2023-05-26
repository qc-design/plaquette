# %% Import necessary objects and functions   # noqa: D100
import numpy as np

import plaquette
from plaquette.circuit.generator import generate_qec_circuit
from plaquette.codes import LatticeCode
from plaquette.decoders import UnionFindDecoder
from plaquette.decoders.decoderbase import check_success
from plaquette.device import Device, MeasurementSample
from plaquette.errors import QubitErrorsDict
from plaquette.visualizer import LatticeVisualizer

# %% Set fixed RNG seed, so we always get the same results
plaquette.rng = np.random.default_rng(seed=1234567890)

# %% Create a code
code = LatticeCode.make_planar(n_rounds=1, size=4)

# %% We can look at a graphical representation of a code with the
# `LatticeVisualizer` class.
visualizer = LatticeVisualizer(code)
# %% If you're in a Jupyter notebook, try `visualizer.draw_lattice()`, but the
# `matplotlib` (mpl) version is always safe.
visualizer.draw_lattice_mpl()
qed: QubitErrorsDict = {
    "pauli": {
        q: {"x": 0.05, "y": 1e-15, "z": 1e-15}
        for q in range(len(code.lattice.dataqubits))
    },
    "erasure": {q: {"p": 0.01} for q in range(len(code.lattice.dataqubits))},
}

# %% You can automatically create a circuit from a code
circuit = generate_qec_circuit(code, qed, {}, "Z")
# %% and run it via a device using one of the available local backends:
# * `clifford`
# * `stim`
dev = Device("clifford")
dev.run(circuit)

# %% The device can *sample* the circuit by returning the outcomes of all
# the measurement gates and erasure gates you define.
raw_results, erasure = dev.get_sample()

# %% which can be unpacked into a more comfortable object
sample = MeasurementSample.from_code_and_raw_results(code, raw_results, erasure)

# %% We can now **decode** the measurement results and check whether we can
# correct the errors that have appeared
decoder = UnionFindDecoder.from_code(code, qed, weighted=True)  # type: ignore
correction = decoder.decode(sample.erased_qubits, sample.syndrome)
print(
    f"Decoding {'succeeded!' if check_success(code, [correction], sample.logical_op_toggle, 'Z') else 'failed...'}"  # noqa
)
# %% You can also visualise the *corrections* and *errors* on the lattice
# (although right now this only works with the `plotly` backend)
fig = visualizer.draw_latticedata(
    syndrome=sample.syndrome[0], correction=correction
).update_layout(width=500, height=350)

# %% And finally, you can calculate the logical error rate
successes = 0
reps = 1000
for _ in range(1000):
    dev.run(circuit)
    raw, erasure = dev.get_sample()
    results = MeasurementSample.from_code_and_raw_results(code, raw, erasure)
    correction = decoder.decode(results.erased_qubits, results.syndrome)
    if check_success(code, [correction], results.logical_op_toggle, "Z"):
        successes += 1
print(f"Logical error rate: {(1 - successes / reps)*100:.2f}%")
