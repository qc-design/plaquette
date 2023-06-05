# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest


@pytest.fixture(scope="module")
def stable_rgen():
    return np.random.RandomState(seed=123123456)
