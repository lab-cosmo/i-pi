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
        self.b2qc = lambda: tri_b2qc(self)
        self.b2fc = lambda f, fc: tri_b2fc(self, f, fc)

    def bind(self, q, m, qc=None, mc=None):
        ncart, self.nbeads = q.shape
        self.nsets = ncart//(3*self.natoms)
        dself = dd(self)
        rpshape=(self.nsets, 3*self.natoms, self.nbeads)
        dself.q = depend_array(name="q",
                               value=np.zeros(rpshape))
        dself.dynm3 = depend_array(name="dynm3",
                                   value=np.zeros(rpshape))
        dself.qc = depend_array(name="qc",
                                value=np.zeros((self.nsets, 3*self.natoms)))
        dself.mc = depend_array(name="mc",
                                value=np.zeros((self.nsets, 3*self.natoms)))
        dself.q[:] = q.reshape(rpshape)
        dself.dynm3[:] = m.reshape(rpshape)
        if qc is None:
            self.external.bind(dstrip(self.q).mean(axis=-1),
                               dstrip(self.dynm3)[...,0])
            self.b2qc()
        else:
            dself.qc[:] = qc
        if mc is None:
            dself.mc = m.reshape(rpshape)[...,0]
        else:
            dself.mc[:] = mc
        self.external.bind(dstrip(self.qc), dstrip(self.mc))


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

def com(q, m, axes=(-1,-2)):
    """Calculate the position of the com of the system

    Args:
        q (ndarray): at least a 2d array with particle positions
        m (ndarray): at least a 2d array with particle masses
        axes (tuple): length-2 tuple indicating the axes corresponding to
                      the cartesian dimensions and the identities of the atoms
    """
    cart_ax, atom_ax = [ax if ax >= 0 else ax+q.ndim for ax in axes]
    mtot = np.sum(m, axis=atom_ax)
    ans = np.sum(m*q, axis=atom_ax)/mtot
    return ans

def eckart(q, m, qref, axes=(-1,-2)):
    """Calculate the angular-momentum-like quantity that appears in the
       rotational Eckart conditions.

    Args:
        q (ndarray): at least a 2d array with particle positions
        m (ndarray): at least a 2d array with particle masses
        qref (ndarray): at least a 2d array with reference coordinates
        axes (tuple): length-2 tuple indicating the axes corresponding to
                      the cartesian dimensions and the identities of the atoms

    """
    # Calculate the total mass of the RP
    cart_ax, atom_ax = [ax if ax >= 0 else ax+q.ndim for ax in axes]
    mtot = np.sum(m, axis=atom_ax)
    # Calculate the CoM of the reference configuration
    ref_com = np.sum(m*qref, axis=atom_ax)/mtot
    ref_com = np.expand_dims(ref_com, axis=atom_ax)
    mtot = np.expand_dims(mtot, axis=atom_ax)
    # Calculate the reference configuration in its CoM, weighted
    # by mass/mtot
    mref = (qref-ref_com)*m/mtot
    # Calculate the cross-product
    ans = np.sum(np.cross(mref, q-qref, axis=cart_ax), axis=atom_ax)
    return ans

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
    # Initialising a constraint group
    cg = MinimalCG()
    cg.bind(q, m)
    # Test that the quasi-centroid internals have been calculated correctly
    qc = dstrip(cg.qc).reshape((-1,3,3))
    mc = dstrip(cg.mc).reshape((-1,3,3))
    r1_qc, r2_qc, ct_qc = tri_internals(qc)[:3]
    r1, r2, ct = tri_internals(dstrip(cg.q).reshape((-1,3,3,nbeads)), axes=(-2,-3))[:3]
    assert r1.mean(axis=-1) == pytest.approx(r1_qc)
    assert r2.mean(axis=-1) == pytest.approx(r2_qc)
    assert np.arccos(ct).mean(axis=-1) == pytest.approx(np.arccos(ct_qc))
    # Test that the externals have been calculated correctly
    q0 = dstrip(cg.q).mean(axis=-1).reshape((-1,3,3))
    com0 = com(q0, mc)
    comc = com(qc, mc)
    assert com0 == pytest.approx(comc)
    assert eckart(qc, mc, q0) == pytest.approx(0.0)
    # NOTE: to use the following need to comment out the external forces
    #       in tri_b2fc
#    # Test that internal-only forces are reproduced correctly
#    qref = np.transpose(
#            np.reshape(q, (nsets, natoms, ndims, nbeads)),
#            [0,3,1,2]) # nsets, nbeads, natoms, ndims
#    fr1ref = np.zeros((nsets,nbeads))
#    fr2ref = np.zeros_like(fr1ref)
#    ftref = np.zeros_like(fr1ref)
#    fref = np.zeros_like(qref)
#    for i, qbeads in enumerate(qref):
#        for j, qmol in enumerate(qbeads):
#            fref[i,j], fr1ref[i,j], fr2ref[i,j], ftref[i,j] = \
#                tri_force(qmol)
#    fref = np.transpose(fref, [0,2,3,1])
#    fc = np.empty((nsets, 9))
#    fc = cg.b2fc(fref, fc)
#    fr1, fr2, ftheta = tri_cart2ba(qc.reshape((-1,3,3)),
#                                   fc.reshape((-1,3,3)))
#    assert fr1ref.mean(axis=-1) == pytest.approx(fr1)
#    assert fr2ref.mean(axis=-1) == pytest.approx(fr2)
#    assert ftref.mean(axis=-1) == pytest.approx(ftheta)

    # Test forces for a single-bead case -- should be exactly reproduced
    nbeads=1
    # Generate a new set of random geometries
    q = np.random.uniform(low=-1.5, high=1.5, size=(nsets*natoms*ndims,nbeads))
    fref = np.random.uniform(low=-1.5, high=1.5, size=(nsets,natoms*ndims,nbeads))
    fc = np.empty((nsets,natoms*ndims))
    m = np.ones((nsets,natoms,ndims,nbeads)) * \
            np.array([15.999, 1.0078, 1.0078])[:,None,None]
    m.shape = q.shape
    cg = MinimalCG()
    cg.bind(q, m)
    fc = cg.b2fc(fref, fc)
    assert fc.flatten() == pytest.approx(fref.flatten())

