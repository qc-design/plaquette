# In[1]:
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from plaquette.frontend import ExperimentConfig, _QubitErrorMetadata

# In[ ]:
surf17_base_conf = ExperimentConfig.load_toml(
    "docs/advanced/frontend/surface17_tutorial.toml"
)
surf17_base_conf.build()
pprint(surf17_base_conf)
code_sizes = [3, 5, 7, 9, 11, 13]

# In[ ]:
# Experiment 1: X,Z Pauli Errors
# Simulate X, Z errors linearly growing between [0.001, 0.1] for code distances {3,5,7}
probs = np.linspace(1e-3, 1e-1, 20)

exp1_ler = list()
for size in code_sizes:
    surf17_base_conf.code_conf.size = size
    log_err_rates_code = list()
    for prob in probs:
        print(size, prob)
        surf17_base_conf.errors_conf.qubit_errors.update(
            X=_QubitErrorMetadata(distribution="constant", params=[prob]),
            Z=_QubitErrorMetadata(distribution="constant", params=[prob]),
        )
        surf17_base_conf.build()
        log_err_rates_code.append(surf17_base_conf.run())
    exp1_ler.append(log_err_rates_code)
# In[4]:
# plt.figure()
plt.plot(probs, exp1_ler[0], "-o", label="Code Size 3")
plt.plot(probs, exp1_ler[1], "-o", label="Code Size 5")
plt.plot(probs, exp1_ler[2], "-o", label="Code Size 7")
plt.plot(probs, exp1_ler[3], "-o", label="Code Size 9")
plt.plot(probs, exp1_ler[4], "-o", label="Code Size 11")
plt.plot(probs, exp1_ler[5], "-o", label="Code Size 13")
plt.legend()
plt.show()

# In[5]:
# ## Experiment 2: X,Z Pauli Errors with gate errors
# Simulate X,Z errors like Experiment 1, but add gate errors for H, CZ and CX.

probs = np.linspace(1e-3, 1e-1, 20)
surf17_base_conf.errors_conf.gate_errors.sample = True
surf17_base_conf.errors_conf.gate_errors.simulated_errors
code_sizes = [3, 5, 7]
exp2a_ler = list()
for size in code_sizes:
    surf17_base_conf.code_conf.size = size
    log_err_rates_code = list()
    for prob in probs:
        print(size, prob)
        surf17_base_conf.errors_conf.qubit_errors.update(
            X=_QubitErrorMetadata(distribution="constant", params=[prob]),
            Z=_QubitErrorMetadata(distribution="constant", params=[prob]),
        )
        surf17_base_conf.build()
        log_err_rates_code.append(surf17_base_conf.run())
    exp2a_ler.append(log_err_rates_code)


# In[7]:

plt.figure()
plt.plot(probs, exp2a_ler[0], "-o", label="Code Size 3")
plt.plot(probs, exp2a_ler[1], "-o", label="Code Size 5")
plt.plot(probs, exp2a_ler[2], "-o", label="Code Size 7")
# plt.plot(probs, exp2a_ler[3], '-o', label="Code Size 9")
# plt.plot(probs, exp2a_ler[4], '-o', label="Code Size 11")
# plt.plot(probs, exp2a_ler[5], '-o', label="Code Size 13")
plt.legend()
plt.show()

# In[ ]:

probs = np.linspace(1e-5, 1e-3, 20)
code_sizes = [3, 5, 7]
surf17_base_conf.errors_conf.gate_errors.sample = True
surf17_base_conf.simulator_conf.shots = 100000
exp2b = list()
for size in code_sizes:
    surf17_base_conf.code_conf.size = size
    log_err_rates_code = list()
    for prob in probs:
        print(size, prob)
        surf17_base_conf.errors_conf.qubit_errors.update(
            X=_QubitErrorMetadata(distribution="constant", params=[prob]),
            Z=_QubitErrorMetadata(distribution="constant", params=[prob]),
        )
        surf17_base_conf.build()
        log_err_rates_code.append(surf17_base_conf.run())
    exp2b.append(log_err_rates_code)


# In[9]:
plt.figure()
plt.plot(probs, exp2b[0], "-o", label="Code Size 3")
plt.plot(probs, exp2b[1], "-o", label="Code Size 5")
plt.plot(probs, exp2b[2], "-o", label="Code Size 7")
# plt.plot(probs, exp2b[3], '-o',label="Code Size 9")
# plt.plot(probs, exp2b[4], '-o',label="Code Size 11")
# plt.plot(probs, exp2b[5], '-o',label="Code Size 13")
plt.legend()
plt.show()

# In[]:
probs_combined = np.append(probs, np.linspace(1e-3, 1e-1, 20)[:5])
plt.figure()
plt.plot(probs_combined, exp2b[0][:] + exp2a_ler[0][:5], "-o", label="Code Size 3")
plt.plot(probs_combined, exp2b[1][:] + exp2a_ler[1][:5], "-o", label="Code Size 5")
plt.plot(probs_combined, exp2b[2][:] + exp2a_ler[2][:5], "-o", label="Code Size 7")
plt.legend()
plt.show()

# In[]:
probs = np.linspace(5e-4, 2e-2, 20)
code_sizes = [3, 5, 7]
surf17_base_conf.errors_conf.gate_errors.sample = True
surf17_base_conf.simulator_conf.shots = 100000
exp2c = list()
for size in code_sizes:
    surf17_base_conf.code_conf.size = size
    log_err_rates_code = list()
    for prob in probs:
        print(size, prob)
        surf17_base_conf.errors_conf.qubit_errors.update(
            X=_QubitErrorMetadata(distribution="constant", params=[prob]),
            Z=_QubitErrorMetadata(distribution="constant", params=[prob]),
        )
        surf17_base_conf.build()
        log_err_rates_code.append(surf17_base_conf.run())
    exp2c.append(log_err_rates_code)
# In[]:
plt.figure()
plt.loglog(probs, exp2c[0], "-o", label="Code Size 3")
plt.loglog(probs, exp2c[1], "-o", label="Code Size 5")
plt.loglog(probs, exp2c[2], "-o", label="Code Size 7")
plt.legend()
plt.show()

# In[]:
probs_2a = np.linspace(1e-3, 1e-1, 20)
probs_2b = np.linspace(1e-5, 1e-3, 20)
probs_2c = np.linspace(5e-4, 2e-2, 20)


def combined_array(index):
    return exp2a_ler[index][:] + exp2b[index][:] + exp2c[index][:]


probs_combined = np.append(np.append(probs_2a, probs_2b), probs_2c)
sort_indices = np.argsort(probs_combined)
probs_sorted = probs_combined[sort_indices]
plt.figure()
combined_ler = combined_array(0)
plt.semilogx(
    probs_sorted[3:45],
    [combined_ler[i] for i in sort_indices[3:45]],
    "-",
    label="Code Size 3",
)
combined_ler = combined_array(1)
plt.semilogx(
    probs_sorted[3:45],
    [combined_ler[i] for i in sort_indices[3:45]],
    "-",
    label="Code Size 5",
)
combined_ler = combined_array(2)
plt.semilogx(
    probs_sorted[3:45],
    [combined_ler[i] for i in sort_indices[3:45]],
    "-",
    label="Code Size 7",
)

plt.legend()
plt.show()


# In[10]:
# ## Experiment 3: X,Z Pauli errors with measurement errors
# X,Z errors with measurement errors that also grows along with the pauli errors

surf17_base_conf.errors_conf.gate_errors.sample = False
probs = np.linspace(1e-3, 1e-1, 20)
code_sizes = [3, 5, 7]
surf17_base_conf.errors_conf.qubit_errors.measurement.enabled = True

exp3_ler = list()
for size in code_sizes:
    surf17_base_conf.code_conf.size = size
    surf17_base_conf.code_conf.rounds = size - 1
    log_err_rates_code = list()
    for prob in probs:
        print(size, prob)
        surf17_base_conf.errors_conf.qubit_errors.update(
            X=_QubitErrorMetadata(distribution="constant", params=[prob]),
            Z=_QubitErrorMetadata(distribution="constant", params=[prob]),
            measurement=_QubitErrorMetadata(distribution="constant", params=[prob]),
        )
        surf17_base_conf.build()
        log_err_rates_code.append(surf17_base_conf.run())
        print(log_err_rates_code)
    exp3_ler.append(log_err_rates_code)

# In[12]:

# plt.figure()
plt.plot(probs, exp3_ler[0], "-o", label="Code Size 3")
plt.plot(probs, exp3_ler[1], "-o", label="Code Size 5")
plt.plot(probs, exp3_ler[2], "-o", label="Code Size 7")
plt.plot(probs, exp3_ler[3], "-o", label="Code Size 9")
plt.plot(probs, exp3_ler[4], "-o", label="Code Size 11")
plt.plot(probs, exp3_ler[5], "-o", label="Code Size 13")
plt.legend()
plt.show()

# In[ ]:

probs = np.linspace(1e-4, 1e-2, 20)
surf17_base_conf.errors_conf.qubit_errors.measurement.enabled = True
exp3b_ler = list()
for size in code_sizes:
    surf17_base_conf.code_conf.size = size
    surf17_base_conf.code_conf.rounds = size - 1
    log_err_rates_code = list()
    for prob in probs:
        print(size, prob)
        surf17_base_conf.errors_conf.qubit_errors.update(
            X=_QubitErrorMetadata(distribution="constant", params=[prob]),
            Z=_QubitErrorMetadata(distribution="constant", params=[prob]),
            measurement=_QubitErrorMetadata(distribution="constant", params=[prob]),
        )
        surf17_base_conf.build()
        log_err_rates_code.append(surf17_base_conf.run())
        print(log_err_rates_code)
    exp3b_ler.append(log_err_rates_code)


# In[14]:

# plt.figure()
plt.plot(probs, exp3b_ler[0], "-o", label="Code Size 3")
plt.plot(probs, exp3b_ler[1], "-o", label="Code Size 5")
plt.plot(probs, exp3b_ler[2], "-o", label="Code Size 7")
plt.plot(probs, exp3b_ler[3], "-o", label="Code Size 9")
plt.plot(probs, exp3b_ler[4], "-o", label="Code Size 11")
plt.plot(probs, exp3b_ler[5], "-o", label="Code Size 13")
plt.legend()
plt.show()


# In[ ]:
# Experiment 4 : X,Z Pauli errors with erasures
# X,Z with erasure errors growing the pauli errors  and changing the decoder to
# UnionFindDecoder to accomodate the decoding with erasures.

surf17_base_conf.errors_conf.qubit_errors.measurement.enabled = False
surf17_base_conf.decoder_conf.name = "UnionFindDecoder"
surf17_base_conf.decoder_conf.weighted = True
probs = np.linspace(5e-4, 5e-2, 20)
exp4_ler = list()
for size in code_sizes:
    surf17_base_conf.code_conf.size = size
    surf17_base_conf.code_conf.rounds = size - 1
    log_err_rates_code = list()
    for prob in probs:
        print(size, prob)
        surf17_base_conf.errors_conf.qubit_errors.update(
            X=_QubitErrorMetadata(distribution="constant", params=[prob]),
            Z=_QubitErrorMetadata(distribution="constant", params=[prob]),
            erasure=_QubitErrorMetadata(distribution="constant", params=[prob]),
        )
        surf17_base_conf.build()
        log_err_rates_code.append(surf17_base_conf.run())
    exp4_ler.append(log_err_rates_code)


# In[17]:

# plt.figure()
plt.plot(probs, exp4_ler[0], "-o", label="Code Size 3")
plt.plot(probs, exp4_ler[1], "-o", label="Code Size 5")
plt.plot(probs, exp4_ler[2], "-o", label="Code Size 7")
plt.plot(probs, exp4_ler[3], "-o", label="Code Size 9")
plt.plot(probs, exp4_ler[4], "-o", label="Code Size 11")
plt.plot(probs, exp4_ler[5], "-o", label="Code Size 13")
plt.legend()
plt.show()
