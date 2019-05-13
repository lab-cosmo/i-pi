"""Tests rotation into the Eckart frame."""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import numpy as np
import pytest
from ipi.utils import mathtools



def test_eckrot():

    # Equilibrium configuration
    qref = np.zeros((3,3))
    qref[0,2] = 0.06615931  # Oz
    qref[1,1] = 0.75813308  # H1y
    qref[1,2] = -0.52499806 # H1z
    qref[2,1] = -qref[1,1] # H2y
    qref[2,2] = qref[1,2]  # H2z

    # Perturbed configuration
    qshift = qref.copy()
    qshift[0,:2] += 0.05

    # Masses
    m = np.zeros_like(qref)
    m[0] = 15.99491502
    m[1:] = 1.00782522

    # Shift configs to Eckart frame
    CoM = np.sum(qref*m, axis=0)/m.sum(axis=0)
    qref -= CoM
    CoM = np.sum(qshift*m, axis=0)/m.sum(axis=0)
    qshift -= CoM

    # Configuration in the Eckart frame
    qans = np.zeros_like(qref)
    qans[0,1] = 0.00362754
    qans[0,2] = 0.06653210
    qans[1,1] = 0.72901492
    qans[1,2] = -0.50551041
    qans[2,1] = -0.78658654
    qans[2,2] = -0.55040211

    q = qshift.copy()
    q = mathtools.eckrot(q, m, qref)
    assert np.allclose(q,qans)

    # Stack of configurations
    n = 5
    qstack = np.stack([qshift.copy() for i in range(n)])
    refstack = np.stack([qref.copy() for i in range(n)])
    qstack = mathtools.eckrot(qstack, m[None,...], refstack)
    assert np.allclose(qstack,qans)