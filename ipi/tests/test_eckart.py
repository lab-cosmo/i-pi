"""Tests rotation into the Eckart frame."""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import numpy as np
import pytest
from ipi.utils import mathtools


def eckgen():
    """Returns a reference configuration and a perturbed geometry
       for testing functions related to the Eckart conditions.

       Output:
           qref: the reference geometry
           qshift: a perturbed configuration
           qans: qshift rotated into the Eckart frame
           m: particle masses

       NOTE: all configurations are expressed in their respective
             centre-of-mass frames.
    """
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

    return qref, qshift, qans, m

def test_eckrot():

    qref, qshift, qans, m = eckgen()
    q = qshift.copy()
    q = mathtools.eckrot(q, m, qref)
    assert q == pytest.approx(qans, rel=1.0e-05, abs=1.0e-08)

    # Stack of configurations
    n = 5
    qstack = np.stack([qshift.copy() for i in range(n)])
    refstack = np.stack([qref.copy() for i in range(n)])
    qstack = mathtools.eckrot(qstack, m[None,...], refstack)
    assert qstack == pytest.approx(
            np.stack([qans for i in range(n)]), rel=1.0e-05, abs=1.0e-08)

def test_eckspin():

    qref, qshift, qans, m = eckgen()
    p = np.random.normal(size=qref.shape)
    L = np.cross(qref, p, axis=-1).sum(axis=-2)
    p = mathtools.eckspin(p, qans, m, qref, L)
    eckprod = np.cross(qref, p, axis=-1).sum(axis=-2)
    assert eckprod == pytest.approx(np.zeros_like(eckprod))

    # Stack of configurations
    n = 5
    pstack = np.random.normal(size=(n,)+qref.shape)
    L = np.cross(qref[None,...], pstack, axis=-1).sum(axis=-2)
    pstack = mathtools.eckspin(pstack, qans[None,...],
                               m[None,...], qref[None,...], L)
    eckprod = np.cross(qref, pstack, axis=-1).sum(axis=-2)
    assert eckprod == pytest.approx(np.zeros_like(eckprod))



