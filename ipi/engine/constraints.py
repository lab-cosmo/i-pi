"""Classes that deal with holonomic constraints.

Contains objects that return the values and the gradients of holonomic constraint
functions.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2019 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np
from ipi.utils.depend import depend_value, dd, dobject

__all__ = ['HolonomicConstraint','BondLength','BondAngle','Eckart']

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
        jac *= 0
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
        jac *= 0
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

class Eckart(HolonomicConstraint):
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
               mref(ndarray): a confirming array of atomic masses

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