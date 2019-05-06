"""Deals with testing specialisations of HolonomicConstraint."""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2019 i-PI developers
# See the "licenses" directory for full license information.


from common import local
from ipi.utils.depend import dstrip, dobject, depend_array
from ipi.utils import nmtransform
from ipi.engine.initializer import init_file
from ipi.engine.beads import Beads
from ipi.engine.constraints import Replicas, BondLength, BondAngle, \
                                   EckartTransX, EckartTransY, EckartTransZ, \
                                   EckartRotX, EckartRotY, EckartRotZ
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
    replica_list = []
    for i in range(3*beads.natoms):
        replica_list.append(Replicas(len(atoms), beads, i))
    nm = MinimalNM()
    nm.bind(beads)

    return beads, replica_list, nm

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

def test_replicas():
    beads, replica_list = create_beads(local("test.ice_Ih.xyz"))[:2]
    assert beads.q[:,0] == pytest.approx(replica_list[0].q)
    assert beads.p[:,0] == pytest.approx(replica_list[0].p)
    assert beads.m[0] == pytest.approx(replica_list[0].m)
    beads.q[:,0] += 1
    assert not (beads.q[:,0] == pytest.approx(replica_list[0].q))
    replica_list[0].q += 1
    assert beads.q[:,0] == pytest.approx(replica_list[0].q)

def test_bond():
    beads, replicas, nm = create_beads(local("test.ice_Ih.xyz"))
    dofs = list(range(6))
    bond_constraint = BondLength(dofs)
    bond_constraint.bind(replicas, nm)
    sigma, grad = bondlength(beads, dofs, bond_constraint.targetval)
    assert bond_constraint.sigma == pytest.approx(sigma)
    assert bond_constraint.jac == pytest.approx(grad)
    for i,idof in enumerate(bond_constraint.dofs):
        # Divide by the dynamical mass
        grad[i,:] /= dstrip(nm.dynm3)[:,idof]
    assert bond_constraint.mjac == pytest.approx(grad)
    # vary targetval
    new_target = 2.5
    bond_constraint.targetval = new_target
    sigma, grad = bondlength(beads, dofs, new_target)
    assert bond_constraint.sigma == pytest.approx(sigma)
    assert bond_constraint.jac == pytest.approx(grad)
    # vary configuration
    beads.q[0,dofs[2]] = 1.0
    sigma, grad = bondlength(beads, dofs, new_target)
    assert not (bond_constraint.sigma == pytest.approx(sigma))
    assert not (bond_constraint.jac == pytest.approx(grad))
    replicas[dofs[2]].q[0] = beads.q[0,dofs[2]]
    assert (bond_constraint.sigma == pytest.approx(sigma))
    assert (bond_constraint.jac == pytest.approx(grad))

def test_angle():
    beads, replicas, nm = create_beads(local("test.ice_Ih.xyz"))
    dofs = list(range(9))
    bond_constraint = BondAngle(dofs)
    bond_constraint.bind(replicas, nm)
    sigma, grad = bondangle(beads, dofs, bond_constraint.targetval)
    assert bond_constraint.sigma == pytest.approx(sigma)
    assert bond_constraint.jac == pytest.approx(grad)
    # vary targetval
    new_target = np.pi/4
    bond_constraint.targetval = new_target
    sigma, grad = bondangle(beads, dofs, new_target)
    assert bond_constraint.sigma == pytest.approx(sigma)
    assert bond_constraint.jac == pytest.approx(grad)
    # vary configuration
    beads.q[0,dofs[2]] = 1.0
    replicas[dofs[2]].q[0] = beads.q[0,dofs[2]]
    sigma, grad = bondangle(beads, dofs, new_target)
    assert bond_constraint.sigma == pytest.approx(sigma)
    assert bond_constraint.jac == pytest.approx(grad)

def test_eckart_trans():
    beads, replicas, nm = create_beads(local("test.ice_Ih.xyz"))
    dofs = list(range(9))
    com_constraints = [cls(dofs) for cls in (
                                             EckartTransX,
                                             EckartTransY,
                                             EckartTransZ)]
    nmtrans = nmtransform.nm_trans(nbeads = beads.nbeads)
    for c in com_constraints:
        c.bind(replicas, nm, transform=nmtrans)
    sigma, grad = com(
            beads, dofs, np.asarray(
                    [c.targetval for c in com_constraints]))
    assert np.asarray([c.sigma for c in com_constraints]) == pytest.approx(sigma)
    for c in com_constraints:
        assert c.jac == pytest.approx(grad)
    # vary targetval
    new_target = [2.5, -1.0, 5.0]
    for c,t in zip(com_constraints,new_target):
        c.targetval = t
    sigma, grad = com(beads, dofs, np.asarray(new_target))
    assert np.asarray([c.sigma for c in com_constraints]) == pytest.approx(sigma)
    for c in com_constraints:
        assert c.jac == pytest.approx(grad)
    # vary configuration
    beads.q[0,dofs[2]] = 1.0
    beads.q[1,dofs[4]] =-1.5
    beads.q[3,dofs[8]] = 2.7
    sigma, grad = com(beads, dofs, np.asarray(new_target))
    replicas[dofs[2]].q[0] = beads.q[0,dofs[2]]
    replicas[dofs[4]].q[1] = beads.q[1,dofs[4]]
    replicas[dofs[8]].q[3] = beads.q[3,dofs[8]]
    assert np.asarray([c.sigma for c in com_constraints]) == pytest.approx(sigma)
    for c in com_constraints:
        assert c.jac == pytest.approx(grad)

def test_eckart_rot():
    beads, replicas, nm = create_beads(local("test.ice_Ih.xyz"))
    # Value of constraint should not be affected by re-weighting non-centroid modes
    nm.dynm3[...] *= np.arange(1,beads.nbeads+1)[:,None]
    dofs = list(range(9))
    constraints = [cls(dofs) for cls in (EckartRotX,
                                         EckartRotY,
                                         EckartRotZ)]
    nmtrans = nmtransform.nm_trans(nbeads = beads.nbeads)
    for c in constraints:
        c.bind(replicas, nm, transform=nmtrans)
    # Calculate the centroids
    reference = []
    for rep in replicas:
        reference.append(dstrip(rep.q).mean())
    reference = np.array(reference)
    sigma, grad = eckart_rot(
            beads, dofs, reference)
    for i,c in enumerate(constraints):
        assert c.sigma == pytest.approx(sigma[i])
        assert c.jac == pytest.approx(grad[i])
        for j,idof in enumerate(c.dofs):
            # Divide by the dynamical mass
            grad[i][j,:] /= dstrip(nm.dynm3)[0,idof]
        assert c.mjac == pytest.approx(grad[i])
    # vary reference
    reference[0] += 1
    reference[4] -= 1.56
    reference[8] -= 3.87
    for c in constraints:
        c._ref[...] = reference[c.dofs].reshape((2,-1))
    sigma, grad = eckart_rot(
            beads, dofs, reference)
    for i,c in enumerate(constraints):
        assert c.sigma == pytest.approx(sigma[i])
        assert c.jac == pytest.approx(grad[i])
    # vary configuration
    beads.q[0,dofs[2]] = 1.0
    beads.q[1,dofs[4]] =-1.5
    beads.q[3,dofs[8]] = 2.7
    sigma, grad = eckart_rot(beads, dofs, reference)
    replicas[dofs[2]].q[0] = beads.q[0,dofs[2]]
    replicas[dofs[4]].q[1] = beads.q[1,dofs[4]]
    replicas[dofs[8]].q[3] = beads.q[3,dofs[8]]
    for i,c in enumerate(constraints):
        assert c.sigma == pytest.approx(sigma[i])
        assert c.jac == pytest.approx(grad[i])