"""Deals with testing specialisations of HolonomicConstraint."""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2019 i-PI developers
# See the "licenses" directory for full license information.


from common import local
from ipi.utils.depend import dstrip, dobject, depend_array
from ipi.utils import nmtransform
from ipi.engine.initializer import init_file
from ipi.engine.beads import Beads
from ipi.engine.constraints import BondLength, BondAngle
import numpy as np
import pytest

class MinimalNM(dobject):
    """A minimal version of NormalModes containing the attributes
    referenced by the HolonomicConstraint class.

    Attributes:
        nbeads(int): number of beads
        natoms(int): number of atoms
        dynm3(2d array): dynamical RP mode masses
        transform: normal mode transform
    """

    def bind(self,beads):
        self.nbeads = beads.nbeads
        self.natoms = beads.natoms
        self.transform = nmtransform.nm_fft(nbeads=self.nbeads, natoms=self.natoms, open_paths=[])
        self.dynm3 = depend_array(name="dynm3",
                                  value=dstrip(beads.m3))

def create_beads(fin):
    """Read atom coordinates from file @fin and return an initialised
       instance of beads.
    """
    mode = "xyz"
    ret = init_file(mode, fin)
    atoms = ret[0]
    beads = Beads(atoms[0].natoms, len(atoms))
    for i in range(len(atoms)):
        beads[i] = atoms[i]
    nm = MinimalNM()
    nm.bind(beads)

    return beads, nm

def bondlength(beads, indices, target):
    """Calculate the mean of the bondlenght over the replicas and
       return the value of the constraint function and its gradient.
    """
    grad = np.empty((beads.nbeads,len(indices)),float)
    for i,idx in enumerate(indices):
        grad[:,i] = dstrip(beads.q[:,idx])
    grad[:,3:] -= grad[:,:3]
    rvec = np.sqrt(np.sum(grad[:,3:]**2, axis=-1))
    sigma = rvec.mean()
    sigma -= target
    grad[:,3:] /= rvec[:,None]
    grad[:,3:] /= rvec.size
    grad[:,:3] = -grad[:,3:]
    return sigma, grad.T

def bondangle(beads, indices, target):
    """Calculate the mean of the bond-angle over the replicas and
       return the value of the constraint function and its gradient.
    """
    grad = np.empty((beads.nbeads,len(indices)),float)
    for i,idx in enumerate(indices):
        grad[:,i] = dstrip(beads.q[:,idx])
    qXA = grad[:,3:6]-grad[:,:3]
    rXA = np.sqrt(np.sum(qXA**2, axis=-1))
    qXA /= rXA[:,None]
    qXB = grad[:,6:]-grad[:,:3]
    rXB = np.sqrt(np.sum(qXB**2, axis=-1))
    qXB /= rXB[:,None]
    ct = np.sum(qXA*qXB, axis=-1)
    st = np.sqrt(1-ct**2)
    sigma = np.arccos(ct).mean()
    sigma -= target
    grad[:,3:6] = (ct[:,None]*qXA-qXB)/(rXA*st*rXA.size)[:,None]
    grad[:,6:] = (ct[:,None]*qXB-qXA)/(rXB*st*rXB.size)[:,None]
    grad[:,:3] = -grad[:,3:6]
    grad[:,:3] -= grad[:,6:]

    return sigma, grad.T

def com(beads, indices, target):
    """Calculate the position of the centre of mass of a system of beads
       and return the corresponding value of the constraint function and
       its gradient.
    """
    natom = len(indices)//3
    grad = np.empty((beads.nbeads,natom),float)
    for i,idx in enumerate(indices[::3]):
        grad[:,i] = beads.m[idx//3]
    mtot = np.sum(grad)
    grad /= mtot
    sigma = np.sum(grad[:,:,None]*np.reshape(
                dstrip(beads.q)[:,indices],
                (-1,natom,3)), axis=(0,1))
    sigma -= target
    return sigma, grad.T

def eckart_rot(beads, indices, reference):
    """Calculate the angular-momentum-like quantity that appears in the
       rotational Eckart conditions and return the corresponding value
       of the constraint function and its gradient.
    """
    # Calculate the total mass of the RP
    natom = len(indices)//3
    m = np.empty((beads.nbeads,natom,3))
    for i,idx in enumerate(indices[::3]):
        m[:,i,:] = dstrip(beads.m)[idx//3]
    mtot = np.sum(m[:,:,0])

    # Calculate the CoM of the reference configuration
    lref = np.asarray([reference[i] for i in indices]).reshape((natom,3))
    ref_com = np.sum(m[0,:,:]*lref, axis=0)/np.sum(m[0,:,0])

    # Calculate the reference configuration in its CoM, weighted
    # by mass/mtot
    mref = lref - ref_com
    mref *= m[0,...]
    mref /= mtot

    #Calculate the gradients of the three Cartesian components of the
    #sum of cross-products
    grad = np.empty((3,beads.nbeads,natom,2))
    temp = np.empty((beads.nbeads,natom,3))
    for i,jk in enumerate([(1,2), (2,0), (0,1)]):
        for ndim,idx in enumerate(jk):
            temp *= 0
            temp[...,idx] = 1
            grad[i,:,:,ndim] = np.cross(mref, temp, axis=-1)[:,:,i]

    #Calculate the sum of cross-products
    temp.shape = (beads.nbeads,-1)
    for i in indices:
        temp[:,i] = dstrip(beads.q)[:,i]
    temp.shape = (beads.nbeads,natom,3)
    temp -= lref
    sigma = np.cross(mref, temp, axis=-1).sum(axis=(0,1))

    # Gradients needs to be transposed to agree with the ordering
    # from EckartRot
    return sigma, np.asarray([
            np.transpose(g,(0,2,1)).reshape(beads.nbeads,-1).T
            for g in grad])

def test_bond():
    beads, nm = create_beads(local("test.ice_Ih.xyz"))
    q = np.transpose(dstrip(beads.q))
    indices = [0,1]
    dofs = np.arange(6)
    constraint = BondLength(indices)
    ref_sigma, ref_grad = bondlength(beads, dofs, 0.0)
    grad = np.empty_like(q)
    sigma, grad = constraint(q, grad)
    assert sigma == pytest.approx(ref_sigma)
    assert grad[:6] == pytest.approx(ref_grad)

def test_angle():
    beads, nm = create_beads(local("test.ice_Ih.xyz"))
    q = np.transpose(dstrip(beads.q))
    indices = list(range(3))
    dofs = list(range(9))
    constraint = BondAngle(indices)
    ref_sigma, ref_grad = bondangle(beads, dofs, 0.0)
    grad = np.empty_like(q)
    sigma, grad = constraint(q, grad)
    assert sigma == pytest.approx(ref_sigma)
    assert grad == pytest.approx(ref_grad)