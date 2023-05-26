# In[1]:
from plaquette.codes import LatticeCode
from plaquette.errors import (
    ErrorData,
    generate_constant_errors,
    generate_gaussian_errors,
)

# In[2]:
code = LatticeCode.make_rotated_planar(size=3, n_rounds=1)
ed = ErrorData.from_lattice(code.lattice)

# In[3]:
data_list = [qubit.equbit_idx for qubit in code.lattice.dataqubits]
ancilla_list = [qubit.equbit_idx for qubit in code.lattice.stabgens]
ed.add_qubit_error(
    qubit_id=data_list,
    error_name="erasure",
    probs=generate_constant_errors(0.1, data_list),
)
ed.add_qubit_error(
    qubit_id=ancilla_list,
    error_name="measurement",
    probs=generate_gaussian_errors(0.1, 0.02, ancilla_list),
)
print(ed.qubit_errors)
# In[18]:
ed.update_qubit_error(9, "measurement", 0.0)

# In[19]:
ed.update_from_csv(gate_error_csv="docs/advanced/errors/gate_errors.csv")
print(ed.gate_errors)
# In[20]:

ed.check_against_code(code)
# In[21]:
ed.update_from_csv(qubit_error_csv="docs/advanced/errors/spem.csv", overwrite=False)
