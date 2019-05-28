"""Deals with testing specialisations of HolonomicConstraint."""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2019 i-PI developers
# See the "licenses" directory for full license information.


from common import local
from ipi.utils.depend import dstrip, dobject, depend_array, dd
from ipi.utils import nmtransform
from ipi.engine.initializer import init_file
from ipi.engine.beads import Beads
from ipi.engine.constraints import \
        BondLength, BondAngle, Eckart, \
        tri_b2fc, tri_b2qc, tri_cart2ba, tri_ba2cart, \
        tri_internals
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

class MinimalCG(dobject):
    """A minimal version of ConstraintGroup containing the attributes referenced
    by the force/coordinate converter functions.

    Attributes:
        nsets: The number of sets of atoms subject to the same constraint
        natoms: The number of atoms in a constrained set
        nbeads: Number of beads
        external: An object enforcing the external constraints

    Depend objects:
        qc: quasi-centroid coordinates
        mc: quasi-centroid masses
        q: bead coordinates
        dynm3: dynamical bead masses

    Methods:
        b2qc: Extracts the quasi-centroid configuration from the bead geometry
        b2fc: Extracts the quasi-centroid forces from the bead forces
    """

    def __init__(self):
        self.natoms = 3
        self.external = Eckart()
        self.b2qc = tri_b2qc
        self.b2fc = tri_b2fc

    def bind(self, q, m, qc=None, mc=None):
        ncart, self.nbeads = q.shape
        self.nsets = ncart//self.natoms
        dself = dd(self)
        dself.q = depend_array(name="q", value=np.zeros(ncart, self.nbeads))
        dself.dynm3 = depend_array(name="dynm3", value=np.zeros(ncart, self.nbeads))
        dself.qc = depend_array(name="qc", value=np.zeros(ncart,))
        dself.mc = depend_array(name="dynm3", value=np.zeros(ncart,))
        dself.q[:] = q
        dself.dynm3[:] = m
        if qc is None:
            self.b2qc()
        else:
            dself.qc[:] = qc
        if mc is None:
            dself.mc = m[:,0]
        else:
            dself.mc[:] = mc

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

def tri_force(q):
    """
    Returns the negative derivatives of the potential
    U(r1, r2, theta) = Sum(a_{i-2} Dr1**i, i=2..4) +
                       Sum(b_{i-2} Dr2**i, i=2..4) +
                       Sum(c_{i-2} Dtheta**i, i=2..4)
    with respect to r1, r2, theta and the Cartesian
    coordinates q.
    """

    a = np.array([0.5,0.5,0.1])
    b = np.array([0.7,0.2,0.01])
    c = np.array([0.2,0.05,0.003])
    r1eq = 1.78
    r2eq = 2.05
    thetaeq = 1.91

    q1 = q[1]-q[0]
    r1 = np.sqrt(np.sum(q1**2, axis=-1))
    Dr1 = r1-r1eq
    q2 = q[2]-q[0]
    r2 = np.sqrt(np.sum(q2**2, axis=-1))
    Dr2 = r2-r2eq
    ct = np.sum(q1*q2, axis=-1)/r1/r2
    st = np.sqrt(1.0-ct**2)
    theta = np.arccos(ct)
    Dt = theta-thetaeq

    fr1 = np.sum([ -i*a[i-2]*Dr1**(i-1) for i in [2,3,4]])
    fr2 = np.sum([ -i*b[i-2]*Dr2**(i-1) for i in [2,3,4]])
    ftheta = np.sum([-i*c[i-2]*Dt**(i-1) for i in [2,3,4]])

    f = np.zeros_like(q)
    f1 = f[1]
    f1[:] = fr1*q1/r1 - ftheta*(q2/(r1*r2) - q1*ct/r1**2)/st
    f2 = f[2]
    f2[:] = fr2*q2/r2 - ftheta*(q1/(r1*r2) - q2*ct/r2**2)/st
    f[0] -= f1+f2
    return f, fr1, fr2, ftheta

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

def test_tri_cart2ba():
    # Generate sets of random triatomic geometries
    nsets=64
    natoms=3
    ndims=3
    qref = np.random.uniform(low=-1.5, high=1.5, size=(nsets,natoms,ndims))
    q1 = qref[:,1,:]-qref[:,0,:]
    r1ref = np.sqrt(np.sum(q1**2, axis=-1))
    q2 = qref[:,2,:]-qref[:,0,:]
    r2ref = np.sqrt(np.sum(q2**2, axis=-1))
    ctref = np.sum(q1*q2, axis=-1)/r1ref/r2ref
    # Test bond-angle calculation
    r1, r2, ct = tri_internals(qref)[:3]
    assert r1 == pytest.approx(r1ref)
    assert r2 == pytest.approx(r2ref)
    assert ct == pytest.approx(ctref)
    # Get reference forces
    fr1ref = np.zeros_like(r1)
    fr2ref = np.zeros_like(r2)
    ftref = np.zeros_like(ct)
    fref = np.zeros_like(qref)
    for i in range(len(qref)):
        fref[i], fr1ref[i:i+1], fr2ref[i:i+1], ftref[i:i+1] = \
            tri_force(qref[i])
    # Test force conversion
    f = fref.copy()
    fr1, fr2, ftheta = tri_cart2ba(qref, f)
    assert fr1 == pytest.approx(fr1ref)
    assert fr2 == pytest.approx(fr2ref)
    assert ftheta == pytest.approx(ftref)

    fba = np.stack([arr.copy() for arr in (fr1ref, fr2ref, ftref)])
    f[:] = tri_ba2cart(qref, fba)
    assert f == pytest.approx(fref)

def test_b2qc():
    nsets=64
    natoms=3
    ndims=3
    nbeads=128
    # Generate a set of random geometries
    q = np.random.uniform(low=-1.5, high=1.5, size=(nsets*natoms*ndims,nbeads))
    m = np.ones((nsets,natoms,ndims,nbeads)) * \
            np.array([15.999, 1.0078, 1.0078])[:,None,None]
    m.shape = q.shape
    m[...,1:] *= np.random.uniform(low=0.01, high=1.0, size=nbeads-1)
    #!! TODO: continue from here

