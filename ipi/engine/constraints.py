"""Classes that deal with holonomic constraints.

Contains objects that return the values and the gradients of holonomic constraint
functions.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2019 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np
from ipi.utils.depend import depend_value, dd, dobject, dstrip
from ipi.utils import mathtools

__all__ = ['Constraints','ConstraintGroup','BondLength','BondAngle','_Eckart']

class HolonomicConstraint(object):
    """Base holonomic constraints class.

    Specifies the standard methods and attributes common to all holonomic
    constraint classes.

    Attributes:
        natoms: The number of atoms in the constrained group
        indices: The indices of the atoms involved in the constraint

    Methods:
        get_natoms: Return the number of atoms involved in the constraint
        sigma: Return the value of the constraint function and its gradient.

    """
    # number of degrees of freedom expected by the constraint function
    _natoms = 0

    @classmethod
    def get_natoms(cls):
        """The number of degrees of freedom expected by the constraint function.
        """
        if (cls._natoms == 0):
            raise TypeError("Trying to get the number of atoms "+
                            "from base HolonomicConstraint class")
        return cls._natoms

    def __init__(self, indices, **kwargs):
        """Initialise the holonomic constraint.

        Args:
            indices: a list of integer indices identifying the constrained atoms.
        """
        natoms = self.get_natoms()
        atoms = np.asarray(indices, dtype=int).copy()
        if (atoms.ndim != 1):
            raise ValueError("Shape of constrained atom group incompatible "+
                             "with "+self.__class__.__name__)
        if (atoms.size != natoms and natoms != -1):
            raise ValueError("Size of constrained atom group incompatible "+
                             "with "+self.__class__.__name__)
        self.indices = atoms
        self.natoms = len(self.indices)

    def __call__(self, q, jac):
        """Dummy constraint function calculator that does nothing."""
        pass

#------------- Specialisations of the Holonomic Constraint class -----------#
class BondLength(HolonomicConstraint):
    """Constraint on the length of a bond between two atoms,
       averaged over the replicas.
    """
    _natoms = 2

    def __call__(self, q, jac):
        """Calculate the difference between the mean bond-length and the
           target value and its gradient.
        """
        jac[:] = 0.0
        nbeads = q.shape[-1]
        slcA, slcB = (slice(3*idx,3*idx+3) for idx in self.indices)
        # A->B displacement vector
        jac[...,slcB,:] = q[...,slcB,:]-q[...,slcA,:]
        # Norm
        jac[...,slcA.start,:] = np.sqrt(np.sum(jac[...,slcB,:]**2, axis=-2))
        # Normalised A->B displacement vector
        jac[...,slcB,:] /= nbeads*jac[...,slcA.start:slcA.start+1,:]
        # Mean bond-length
        sigma = np.mean(jac[...,slcA.start,:], axis=-1)
        # Gradient wrt atom A
        jac[...,slcA,:] = -jac[...,slcB,:]
        return sigma, jac

class BondAngle(HolonomicConstraint):
    """Constraint on the A--X--B bond-angle, averaged over the replicas.
       The indices of the coordinates of the central atom X are supplied
       first, followed by A and B.
    """

    # Nine degrees of freedom -- three per atom
    _natoms = 3

    def __call__(self, q, jac):
        """Calculate the difference between the mean angle and the
           target value.
        """
        jac[:] = 0.0
        nbeads = q.shape[-1]
        slcX, slcA, slcB = (slice(3*idx,3*idx+3) for idx in self.indices)
        # X->A displacement vector
        qXA = q[...,slcA,:]-q[...,slcX,:]
        # Vector norm
        jac[...,slcX.start,:] = np.sqrt(np.sum(qXA**2, axis=-2))
        rXA = jac[...,slcX.start:slcX.start+1,:]
        # Normalised X->A
        qXA /= rXA
        # Repeat for B
        jac[...,slcB,:] = q[...,slcB,:]-q[...,slcX,:]
        qXB = jac[...,slcB,:]
        # Norm
        jac[...,slcX.start+1,:] = np.sqrt(np.sum(qXB**2, axis=-2))
        rXB = jac[...,slcX.start+1:slcX.start+2,:]
        # Normalise
        qXB /= rXB
        # Cosine of the angle
        jac[...,slcX.start+2,:] = np.sum(qXA*qXB, axis=-2)
        ct = jac[...,slcX.start+2:slcX.start+3,:]
        # Gradients w.r.t peripheral atoms
        jac[...,slcA,:] = (qXA*ct-qXB)/rXA
        jac[...,slcB,:] = (qXB*ct-qXA)/rXB
        # Calculate mean angle
        sigma = np.mean(np.arccos(jac[...,slcX.start+2,:]), axis=-1)
        # Calculate sine of angle
        ct[...] = np.sqrt(1.0-ct**2)*nbeads
        # Complete gradient calculation
        jac[...,slcA,:] /= ct
        jac[...,slcB,:] /= ct
        jac[...,slcX,:] = -(jac[...,slcA,:]+jac[...,slcB,:])
        return sigma, jac

class ExternalConstraint(object):
    """Base external constraints class.

    This is different from the other holonomic constraints in that
    (a) the constraints are applied to the centroids,
    (b) the number of atoms is flexible, (c) the constraints require
    the masses of the particles and a reference configuration instead
    of a single value to specify the target, and (d) the constraints are
    of a particularly simple form which allows them to be imposed exactly.

    NOTE: using properties instead of depend arrays because the
    constraint is called repeatedly in a loop during SHAKE/RATTLE,
    and it seems to be faster to access the values this way.

    Properties:
        qref: The reference configuration
        mref: The corresponding masses

    Methods:
        enforce_q: return an array indicating which sets of atoms did not
                   initially satisfy the constraint to the specified tolerance
                   and enforce the constraint onto any such non-conforming sets.
        enforce_p: analogous procedure for the time-derivatives of the
                   constraints.
    """

    def bind(self, qref, mref):
        """
        "Binds" the reference configuration and masses to the constraint.
        Args:
            qref (2d ndarray): the reference configuration stored as
                               an ngp-by-ncart array, where ngp is the
                               number of molecules/groups of atoms
                               individually subject to the constraint,
                               and ncart is the number of Cartesian DoFs
                               in each such group
            mref (2d ndarray): a conforming array of atomic masses

        NOTE: this is not really proper "binding" because the configuration
              and the masses have to be updated manually.
        """
        self.__qref = None
        self.mref = mref
        self.qref = qref

    @property
    def mref(self):
        return self.__mref
    @mref.setter
    def mref(self, val):
        init_shape = val.shape
        self.__mref = val.reshape(init_shape[:-1]+(init_shape[-1]//3, 3))
        if self.__qref is not None:
            self._update_config()
    @property
    def qref(self):
        return self.__qref
    @qref.setter
    def qref(self, val):
        init_shape = val.shape
        self.__qref = val.reshape(init_shape[:-1]+(init_shape[-1]//3, 3))
        self._update_config()

    def _update_config(self):
        """Re-calculate the intermediate values used in enforcing the constraints.
        """
        pass

    def enforce_q(self, q, tol, nc, ns):
        """Test which constraints are not satisfied to within the given
           tolerance, record their identities and enforce the constraints
           on any such non-conforming sets of atoms

           Args:
               q (2d ndarray): an ngp-by-ncart array of
                               atomic configurations
               tol (float): convergence parameter
               nc (1d ndarray): an ngp array of booleans indicating which sets
                                of atoms are still not converged and therefore
                                need to be processed
               ns (1d ndarray): an ngp array of boolean for indicating which
                                sets do not currently satisfy the constraints
                                to the level specified by tol.
            All modified in-place.
        """
        pass

    def enforce_p(self, p, q, tol, nc, ns):
        """Analogous to the above but for enforcing constraints on the
           momenta.
        """
        pass

class Eckart(ExternalConstraint):
    """The Eckart constraints, enforcing the coincidence of the CoMs of
       the input and the reference configuration and a particular
       relative orientation.
    """

    def _update_config(self):
        """Re-calculate the CoM of the reference configuration
           and its coordinates relative to the CoM
        """
        self.mtot = np.sum(self.mref[...,0:1], axis=-2)
        # CoM of reference
        try:
            self.qref_com = np.sum(
                    self.qref*self.mref, axis=-2
                    )/self.mtot
        except:
            raise ValueError(self.qref.shape.__repr__()+
                             self.mref.shape.__repr__()+
                             self.mtot.shape.__repr__())
        # Reference coords relative to CoM
        self.qref_rel = self.qref-self.qref_com[...,None,:]
        # The above, mass-weighted
        self.mqref_rel = self.qref_rel*self.mref/self.mtot[...,None,:]

    def _eckart_prod(self, q, nc):
        ans = np.cross(self.mqref_rel[nc], (q[nc]-self.qref_rel[nc]),
                       axis=-1).sum(axis=-2)
        return np.sqrt(np.sum(ans**2, axis=-1))

    def enforce_q(self, q, tol, nc, ns):
        # Get the current CoM of the system and shift to CoM frame
        init_shape = q.shape
        q.shape = self.qref.shape
        CoM = np.sum(self.mref[nc] * q, axis=-2)/self.mtot[nc]
        q[nc] -= CoM[:,None,:]
        # Calculate the Eckart product
        sigmas = self._eckart_prod(q, nc)
        ns[nc] = sigmas > tol
        # Rotate into Eckart frame as required
        q[ns] = mathtools.eckrot(q[ns], self.mref[ns], self.qref_rel[ns])
        # Shift to CoM of reference
        q[nc] += self.qref_com[nc,None,:]
        q.shape = init_shape

    def enforce_p(self, p, q, tol, nc, ns):
        init_shape = p.shape
        for arr in (p, q):
            arr.shape = self.qref.shape
        # Get CoM velocity and set to zero
        v = np.sum(p[nc], axis=-2)/self.mtot[nc]
        p[nc] -= self.mref[nc]*v[nc,None,:]
        L = np.zeros((len(p),3))
        L[nc,:] = np.cross(self.qref_rel[nc], p[nc],axis=-1).sum(axis=-2)
        sdots = np.sqrt(np.sum(L**2, axis=-1))/self.mtot.reshape((-1,))[nc]
        ns[nc] = sdots > tol
        # Enforce the rotational constraint as required
        p[ns] = mathtools.eckspin(p[ns], q[ns], self.mref[ns],
                                  self.qref_rel[ns], L[ns])
        for arr in (p, q):
            arr.shape = init_shape

class FixCoM(ExternalConstraint):
    """Constrain the CoM only
    """

    def _update_config(self):
        self.mtot = np.sum(self.mref[...,0:1], axis=-2)
        # CoM of reference
        self.qref_com = np.sum(
                self.qref*self.mref, axis=-2
                )/self.mtot

    def enforce_q(self, q, tol, nc, ns):
        # Get the current CoM of the system and shift to CoM frame
        init_shape = q.shape
        q.shape = self.qref.shape
        CoM = np.sum(self.mref[nc] * q, axis=-2)/self.mtot[nc]
        q[nc] -= CoM[:,None,:]
        q[nc] += self.qref_com[nc,None,:]
        ns[nc] = False
        q.shape = init_shape

    def enforce_p(self, p, q, tol, nc, ns):
        init_shape = p.shape
        p.shape = self.qref.shape
        v = np.sum(p[nc], axis=-2)/self.mtot[nc]
        p[nc] -= self.mref[nc]*v[nc,None,:]
        ns[nc] = False
        p.shape = init_shape

#!! TODO: This is to be retired and replaced by Eckart defined above.
class _Eckart(HolonomicConstraint):
    """Rotational Eckart constraint.

       NOTE: this is different to the other HolonomicConstraint objects
       in that it does not calculate the Jacobian, and is initialised
       with additional paramters (particle mass and reference config.)
    """

    # Number of DoFs determined upon initialisation
    _ndof = -1

    def __init__(self, indices, qref=None, mref=None):
        """Initialise the holonomic constraint.
           Args:
               indices: a list of integer indices identifying the constrained atoms (this is ignored)
               qref(ndarray): the reference configuration stored as
                              an ngp-by-ncart array, where ngp is the
                              number of molecules/groups of atoms
                              individually subject to the Eckart
                              constraint, and ncart is the number
                              of Cartesian DoFs in each such group
               mref(ndarray): a conforming array of atomic masses

        """
        if qref is None:
            raise ValueError(
                    "EckartRot must be given a reference configuratiom.")
        if mref is None:
            raise ValueError(
                    "EckartRot must be given the atomic masses.")
        self.__qref = None
        self.mref = mref
        self.qref = qref

    @property
    def mref(self):
        return self.__mref
    @mref.setter
    def mref(self, val):
        init_shape = val.shape
        self.__mref = val.reshape(init_shape[:-1]+(init_shape[-1]//3, 3))
        if self.__qref is not None:
            self._update_config()
    @property
    def qref(self):
        return self.__qref
    @qref.setter
    def qref(self, val):
        init_shape = val.shape
        self.__qref = val.reshape(init_shape[:-1]+(init_shape[-1]//3, 3))
        self._update_config()

    def _update_config(self):
        """Re-calculate the CoM of the reference configuration
           and its coordinates relative to the CoM
        """
        mtot = np.sum(self.mref[...,0:1], axis=-2)
        # CoM of reference
        self.qref_com = np.sum(
                self.qref*self.mref, axis=-2
                )/mtot
        # Reference coords relative to CoM
        self.qref_rel = self.qref-self.qref_com[...,None,:]
        # The above, mass-weighted
        self.mqref_rel = self.qref_rel*self.mref/mtot[...,None,:]

    def __call__(self, q, nc):
        """Return the norm of the sum of cross-products
               SUM[m_a*qref_a x (q_a - qref_a), a] / mtot
            Args:
                q(ndarray): an ngp-by-ncart array of atomic configurations
                nc(ndarray): an array of booleans indicating which rows of
                             q are to be used for calculation.
        """
        qarr = q.reshape(self.qref_rel.shape)[nc]
        ans = np.cross(self.mqref_rel[nc], (qarr-self.qref_rel[nc]),
                       axis=-1).sum(axis=-2)
        return np.sqrt(np.sum(ans**2, axis=-1))


#!! TODO: the Constraints is to be retired and replaced by ConstrainGroup
class Constraints(dobject):

    """
    Holds parameters used in cosnstrained propagation.

    Attributes:
       tol: Tolerance threshold for RATTLE
       maxcycle: Maximum number of cycles in RATTLE
       nfree: Number of steps into which ring-polymer propagation
              under spring forces is split at the inner-most mts level.

    """

    def __init__(self, tol=1.0e-06, maxcycle=100, nfree=1):

        dself = dd(self)
        dself.tol = depend_value(name='tol', value=tol)
        dself.maxcycle = depend_value(name='maxcycle', value=maxcycle)
        dself.nfree = depend_value(name='nfree', value=nfree)

#------------ FUNCTIONS FOR CONVERTING TO QUASI-CENTROID COORDS ------------#
## MONATOMIC

## TRIATOMIC
def tri_internals(q, axes=(-1,-2)):
    """
    Calculate the internal coordinates r1, r2, angle of a triatomic

    q (ndarray): at least a 2d array of Cartesian coordinates,
    axes (tuple): specifies the axis running over the Cartesian dimensions
                  and the axis running over the atoms belonging to the same
                  molecule
    NOTE: the returned array contains cosine(angle); the angle itself
          has to be calculated externally
    """
    cart_ax, atom_ax = axes
    if cart_ax == atom_ax:
        raise ValueError("tri_internals product and sum axes cannot be the same")
    if (q.shape[cart_ax],q.shape[atom_ax]) != (3,3):
        raise ValueError("tri_internals active axes must both be of length 3")
    q0 = np.take(q, 0, axis=atom_ax)
    q1 = np.take(q, 1, axis=atom_ax)-q0
    q2 = np.take(q, 2, axis=atom_ax)-q0
    r1 = np.sqrt(np.sum(q1**2, axis=cart_ax))
    r2 = np.sqrt(np.sum(q2**2, axis=cart_ax))
    ct = np.sum(q1*q2, axis=cart_ax)/(r1*r2)
    return r1, r2, ct, q1, q2

def tri_cart2ba(q, f):
    """
    Convert the internal Cartesian forces acting on a set of beads
    representing a triatomic molecule into bond-angle forces.

    Args:
        q (ndarray): at least a 2d array of Cartesian coordinates,
                     with the final dimension running over x,y,z,
                     and the one before running over the atoms in
                     a triatomic
        f (ndarray): a conforming array of forces

    Result:
        fba: array of shape (3,N,K,...) where N,K,... are the leading
             dimensions of q and f (excluding the final two);
             this unpacks to fr1, fr2, ftheta where fri is the force
             on ri and ftheta the force on theta
    """
    r1, r2, ct, q1, q2 = tri_internals(q)
    q1 /= r1[...,None]
    f1 = f[...,1,:]
    q2 /= r2[...,None]
    f2 = f[...,2,:]
    fr1 = np.sum(q1*f1, axis=-1)
    fr2 = np.sum(q2*f2, axis=-1)
    st = np.sqrt(1.0-ct**2)
    r1 /= st
    ftheta = -r1*np.sum((q2 - q1*ct[...,None])*f1, axis=-1)
    return np.stack((fr1, fr2, ftheta))

def tri_ba2cart(q, fba):
    """
    Convert the forces on bond-angle coordinates of a triatomic
    into Cartesians.

    Args:
        q (ndarray): at least a 2d array of Cartesian coordinates,
                     with the final dimension running over x,y,z,
                     and the one before running over the atoms in
                     a triatomic (generally shape = N,K,...,3,3)
        fba (ndarray): a stack of internal forces in the order
                       fr1, fr2, ftheta and shape 3,N,K,...

    Result:
        f: array of shape (N,K,...,3,3) of the Cartesian representation
           of the internal forces.
    """
    f = np.zeros_like(q)
    r1, r2, ct, q1, q2 = tri_internals(q)
    q1 /= r1[...,None]
    f1 = f[...,1,:]
    q2 /= r2[...,None]
    f2 = f[...,2,:]
    fr1, fr2, ftheta = [np.expand_dims(arr, axis=-1) for arr in fba]
    st = np.sqrt(1.0-ct**2)
    r1 *= st
    f1[...] = q1*fr1 - (q2 - q1*ct[...,None])*ftheta/r1[...,None]
    r2 *= st
    f2[...] = q2*fr2 - (q1 - q2*ct[...,None])*ftheta/r2[...,None]
    f[...,0,:] = -(f1+f2)
    return f

def tri_b2qc(self):
    """
    Given a set of ring-polymer bead coordinates, generate a set
    of quasi-centroid geometries describing triatomic molecules, such
    that the bond-lengths are the means of the RP bond-lenghts, the angles
    are the means of the RP angles, and the Eckart conditions are satisfied
    between the centroids and the quasi-centroids

    Args:
        self(ConstraintGroup): an object describing sets of identically-defined
                               quasi-centroids

    """

    q = np.reshape(dstrip(self.q), (self.nsets, 3, 3, self.nbeads))
    r1, r2, ct = tri_internals(q, axes=(-2,-3))[:3]
    theta = np.arccos(ct)
    R1 = np.mean(r1, axis=-1)
    R2 = np.mean(r2, axis=-1)
    Theta = np.mean(theta, axis=-1)
    qc = np.zeros_like(dstrip(self.qc))
    qc.shape = (self.nsets, 3, 3)
    qc[:,1,0] = R1
    qc[:,2,0] = R2*np.cos(Theta)
    qc[:,2,1] = R2*np.sin(Theta)
    nc = np.ones(self.nsets, dtype=np.bool)
    ns = nc.copy()
    self.external.enforce_q(qc, 0.0, nc, ns)
    qc.shape = (self.nsets, -1)
    self.qc[:] = qc

def tri_b2fc(self, f, fc):
    """
    Extract quasi-centroid forces from forces acting on the beads.

    Args:
        self(ConstraintGroup): an object describing sets of identically-defined
                               quasi-centroids
        f (3d-ndarray): an nsets-by-9-by-nbeads array of bead forces
        fc (2d-ndarray): an nsets-by-9 array of quasi-centroid forces

    Return:
        fc (2d-ndarray): modified in-place
    """
    # Swap beads and Cartesians in input and shift to CoM:
    # Axis order: set, bead, atom, dimension
    shape = (self.nsets,3,3,self.nbeads)
    order = [0,3,1,2]
    q = dstrip(self.q)
    q_bead = np.empty((self.nsets,self.nbeads,3,3))
    q_bead[:] = np.transpose(np.reshape(q, shape), order)
    q0 = np.reshape(np.mean(q, axis=-1), (self.nsets,3,3)) # Centroid coordinates
    f_bead = np.empty((self.nsets,self.nbeads,3,3))
    f_bead[:] = np.transpose(np.reshape(f, shape), order)
    f0 = np.reshape(np.mean(f, axis=-1), (self.nsets,3,3)) # Centroid forces
    m = np.transpose( np.reshape(dstrip(self.dynm3), shape), order)
    # Shift to CoM
    q_bead[:] = mathtools.toCoM(m, q_bead)[0]
    # Remove the external forces
    II = mathtools.mominertia(m, q_bead, shift=False, axes=(-1,-2))
    tau_CoM = np.sum(np.cross(q_bead, f_bead, axis=-1), axis=2)
    alpha = np.linalg.solve(II, tau_CoM)
    alpha = np.expand_dims(alpha, axis=2)
    f_CoM = np.sum(f_bead, axis=2)/np.sum(m[:,:,:,0:1], axis=2)
    f_CoM = np.expand_dims(f_CoM, axis=2)
    f_bead -= m*(f_CoM + np.cross(alpha, q_bead, axis=-1))
    # Extract the internal forces on r1, r2, theta
    fba = tri_cart2ba(q_bead, f_bead)
    # Convert to Cartesian representation and add to fc
    shape = (self.nsets,3,3)
    qc = np.reshape(dstrip(self.qc), shape).copy()
    mc = np.reshape(dstrip(self.mc), shape)
    qc[:], CoM = mathtools.toCoM(mc, qc)
    fcba = fba.mean(axis=-1)
    fcview = np.reshape(fc, (self.nsets,3,3))
    fcview[:] = tri_ba2cart(qc, fcba)
    # Extract the external forces on fc
    q0 -= CoM
    rhs = np.cross(fcview, q0, axis=-1).sum(axis=-2)
    rhs -= np.cross(f0, qc, axis=-1).sum(axis=-2)
    lhs = mathtools.mominertia(mc, qc, q2=q0, shift=False)
    alpha = np.linalg.solve(lhs, rhs)
    alpha = np.expand_dims(alpha, axis=1)
    mtot = np.expand_dims(np.sum(mc[...,0:1], axis=-2),1)
    f_CoM = np.expand_dims(np.sum(f0, axis=1), axis=1)/mtot
    fcview += mc*(f_CoM + np.cross(alpha, qc, axis=-1))
    return fc

#---------------------------------------------------------------------------#
#
#class ConstraintGroup(dobject):
#
#    """
#    Gathers constraints that link a subset of atoms and stores parameters
#    for constrained propagation. Handles calculation of constraint functions
#    and gradients, generation of quasi-centroid geometries compatible with a
#    given ring-polymer configuration, conversion between Cartesian and
#    internal forces/coordinates, and enforcement of constraints with SHAKE/RATTLE.
#
#    Attributes:
#        name: Name of the constraint type
#        natoms: The number of atoms in a single constrained set
#        indices: List of atomic indices specifying the atoms subject to the
#                 set of constraints.
#        nsets: The number of sets of atoms subject to the same constraint
#        nbeads: The number of beads
#        tol: Tolerance threshold for RATTLE
#        maxcycle: Maximum number of cycles in RATTLE
#        internal: List of internal constraint objects
#        external: An object enforcing the external constraints
#
#    Depend objects:
#        qc: Current quasi-centroid configuration
#        mc: Quasi-centroid masses
#        q: Current bead configuration
#        p: Current bead momenta
#        dynm3: Dynamic normal-mode masses
#        q0: Cached constrained configuration from the previous converged SHAKE step
#        g0: Corresponding constraint gradients
#        mg0: Gradients pre-multiplied by the inverse mass tensor
#        targets: Targets for the values of the internal constraint functions
#
#    Methods:
#        b2qc: Generates a quasi-centroid configuration compatible with the given
#              bead geometry
#        b2fc: Extracts forces onto the quasi-centroids from bead forces
#              and geometries
#        shake: Enforce the constraints
#        rattle: Enforce that the time-derivative of the constraints be zero
#    """
#
#    def __init__(self, name, indices, tol=1.0e-06, maxcycle=100):
#
#        self.name = name.lower()
#        self.indices = indices
#        self.tol = tol
#        self.maxcycle = maxcycle
#        #!! TODO: implement routines and nonesuch
#        #!! TODO: in future add diatomic and 3-pyramid (ammonia)
#        if self.name == "triatomic":
#            self.natoms = 3
#            self.internal = [BondLength(indices=[0,1]),
#                             BondLength(indices=[0,2]),
#                             BondAngle(indices=[0,1,2])]
#            self.external = Eckart()
#            self.b2qc = lambda: tri_b2qc(self)
#            self.b2fc = lambda: tri_b2fc(self)
#        elif self.name == "monatomic":
#            self.natoms = 1
#            self.internal = []
#            self.external = FixCoM()
#            self.b2qc = lambda: get_centroid(self)
#            self.b2fc = lambda: get_centroid(self)
#        else:
#            raise ValueError(
#                    "Quasi-centroid group of type '{:s}' ".format(self.name) +
#                    "has not been defined.")
#        if (len(self.indices)%self.natoms != 0):
#            raise ValueError(
#                    "Total number of atoms in the {:s} group ".format(self.name)+
#                    "is inconsistent with partitioning into sets "+
#                    "of size {:d}.".format(self.natoms))
#        self.nsets = len(self.indices)//self.natoms
#
#    def bind(self, beads, nm):
#        pass
