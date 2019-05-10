"""Classes that deal with holonomic constraints.

Contains objects that return the values and the gradients of holonomic constraint
functions.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2019 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np

from ipi.utils.depend import dstrip
#from ipi.utils import nmtransform

__all__ = ['Replicas','HolonomicConstraint','BondLength','BondAngle',]
#           'EckartTransX', 'EckartTransY', 'EckartTransZ',
#           'EckartRotX', 'EckartRotY', 'EckartRotZ',]

class Replicas(object):

    """Storage of ring-polymer replica positions, masses and momenta
       for a single degree of freedom.

    Positions and momenta are stored as nbeads-sized contiguous arrays,
    and mass is stored as a scalar.

    Attributes:
       nbeads: The number of beads.

    Depend objects:
       p: An array that holds the momenta of all the replicas of this DoF.
       q: An array that holds the positions of all the replicas of this DoF.
       m: The mass associated with this DoF
    """

    def __init__(self, nbeads, beads=None, idof=None):
        """Initialises Replicas.

        Args:
           nbeads: An integer giving the number of beads.
           beads: An instance of Beads from which to copy the initial values
                  of the positions, momenta and masses
           idof: An integer index of the degree of freedom from which to copy
                  these values.
        """
        self.nbeads = nbeads
#        dself = dd(self)
        if beads is None:
            qtemp = np.zeros(nbeads, float)
            ptemp = np.zeros(nbeads, float)
            mtemp = np.zeros(1,float)
        else:
            if idof is None:
                raise ValueError("The index of the degree of freedom must be "+
                                 "specified when initialising Replicas from "+
                                 "beads.")
            qtemp = dstrip(beads.q)[:,idof].copy()
            ptemp = dstrip(beads.p)[:,idof].copy()
            mtemp = dstrip(beads.m3)[0:1,idof].copy()
        self.q = qtemp
        self.p = ptemp
        self.m = mtemp

#        dself.q = depend_array(name="q", value=qtemp)
#        dself.p = depend_array(name="p", value=ptemp)
#        dself.m = depend_value(name="m", value=mtemp)

    def __len__(self):
        """Length function.

        This is called whenever the standard function len(replicas) is used.

        Returns:
           The number of beads.
        """

        return self.nbeads

    def copy(self):
        """Creates a new Replicas object.

        Returns:
           A Replicas object with the same q, p, and m as the original.
        """

        newrep = Replicas(self.nbeads)
        newrep.q[:] = self.q
        newrep.p[:] = self.p
        newrep.m[:] = self.m
        return newrep

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
#
#class EckartTrans(HolonomicConstraint):
#    """Constraint on one of the components of the centre of mass.
#    """
#
#    # Number of DoFs determined upon initialisation
#    _ndof = -1
#
#    def __init__(self, dofs, coord, val=None):
#        """Initialise the holonomic constraint.
#
#           Args:
#               dofs(list): integers indexing *all* the degrees of freedom of the
#                           atoms subject to this Eckart constraint
#               coord(str): 'x', 'y', 'z' -- specifies the component of the CoM
#                           to be constrained
#               val(float): the position at which the component is to be constrained.
#        """
#        q_str = coord.lower()
#        if q_str=="x":
#            idx = 0
#        elif q_str=="y":
#            idx = 1
#        elif q_str=="z":
#            idx = 2
#        else:
#            raise ValueError("Invalid coordinate specification supplied to "+
#                             self.__class__.__name__)
#        super(EckartTrans,self).__init__(dofs[idx::3], val)
#
#    def bind(self, replicas, nm, transform=None, **kwargs):
#        """Bind the appropriate coordinates to the constraint.
#        """
#
#        super(EckartTrans, self).bind(replicas, nm, transform, **kwargs)
#        self.qtaint()
#        if self.targetval is None:
#            # Set to the current position of the CoM
#            self.targetval = 0.0
#            currentval = self.sigma
#            self.targetval = currentval
#
#    def get_sigma(self):
#        mrel = np.asarray(self._m).reshape(-1)
#        mtot = np.sum(mrel)
#        mrel /= mtot
#        qc = np.mean(np.asarray(self._q),axis=-1)
#        sigma = np.dot(mrel, qc)
#        mrel /= self.nbeads
#        return sigma, mrel[:,None]
#
#class EckartTransX(EckartTrans):
#    """Constraint on the x-component of the CoM of a group of atoms.
#    """
#    def __init__(self, dofs, val=None):
#        """Initialise the holonomic constraint.
#
#           Args:
#               dofs(list): integers indexing *all* the degrees of freedom of the
#                           atoms subject to this Eckart constraint
#               val(float): the position at which the component is to be constrained.
#        """
#        super(EckartTransX,self).__init__(dofs, "x", val)
#
#class EckartTransY(EckartTrans):
#    """Constraint on the y-component of the CoM of a group of atoms.
#    """
#    def __init__(self, dofs, val=None):
#        """Initialise the holonomic constraint.
#
#           Args:
#               dofs(list): integers indexing *all* the degrees of freedom of the
#                           atoms subject to this Eckart constraint
#               val(float): the position at which the component is to be constrained.
#        """
#        super(EckartTransY,self).__init__(dofs, "y", val)
#
#class EckartTransZ(EckartTrans):
#    """Constraint on the z-component of the CoM of a group of atoms.
#    """
#    def __init__(self, dofs, val=None):
#        """Initialise the holonomic constraint.
#
#           Args:
#               dofs(list): integers indexing *all* the degrees of freedom of the
#                           atoms subject to this Eckart constraint
#               val(float): the position at which the component is to be constrained.
#        """
#        super(EckartTransZ,self).__init__(dofs, "z", val)
#
#class EckartRot(HolonomicConstraint):
#    """One of the components of the Eckart rotational constraint.
#
#       NOTE: in this definition the usual sum over cross-products is divided by
#             the total mass of the system.
#    """
#
#    # Number of DoFs determined upon initialisation
#    _ndof = -1
#
#    def __init__(self, dofs, coord, val=None):
#        """Initialise the holonomic constraint.
#
#           Args:
#               dofs(list): integers indexing *all* the degrees of freedom of the
#                           atoms subject to this Eckart constraint
#               coord(str): 'x', 'y', 'z' -- specifies the component of the
#                           angular-momentum-like quantity to be constrained
#               val(array-like): this argument is not used
#        """
#        q_str = coord.lower()
#        if q_str=="x":
#            idces = (1,2)
#        elif q_str=="y":
#            idces = (2,0)
#        elif q_str=="z":
#            idces = (0,1)
#        else:
#            raise ValueError("Invalid coordinate specification supplied to "+
#                             self.__class__.__name__)
#        super(EckartRot,self).__init__(dofs[idces[0]::3]+dofs[idces[1]::3], 0.0)
#
#    def bind(self, replicas, nm, transform=None, **kwargs):
#        """Bind the appropriate coordinates to the constraints.
#          Args:
#              replicas(list): List of Replicas
#              nm: A normal modes object used to do the normal modes transformation.
#          **kwargs:
#              ref(array-like): Reference configuration for the constraint; if
#                               absent, taken to be the centroid configuration
#
#        """
#
#
#
#        super(EckartRot, self).bind(replicas, nm, transform, **kwargs)
#        self.qtaint()
#        if "ref" in kwargs:
#            # Reference configuration is provided
#            lref = np.asarray(kwargs["ref"]).flatten() # local copy
#            ref = np.asarray(
#                    [ lref[i] for i in self.dofs ]).reshape((2, self.ndof//2))
#        else:
#            # Initialise to centroid configuration
#            ref = np.asarray(
#                    [ np.mean(q) for q in self._q] ).reshape((2, self.ndof//2))
#        self._ref = ref
#
#    def get_sigma(self):
#        # Individual centroid masses divided by the molecular mass
#        mrel = np.asarray(self._m).reshape((2,self.ndof//2))
#        mrel /= mrel.sum(axis=-1)[:,None]
#        # Reference geometry in its CoM, weighted by mrel
#        mref = mrel*self._ref
#        CoM = mref.sum(axis=-1)
#        mref[...] = self._ref-CoM[:,None]
#        mref *= mrel
#        # Displacement between centroid and reference configs
#        delqc = np.mean(np.asarray(self._q), axis=-1).reshape((2, self.ndof//2))
#        delqc -= self._ref
#        sigma = np.sum(mref[0]*delqc[1] - mref[1]*delqc[0])
#        mref /= self.nbeads
#        return sigma, np.hstack((-mref[1],mref[0]))[:,None]
#
#class EckartRotX(EckartRot):
#    """Constraint on the x-component of the Eckart "angular momentum"
#    """
#    def __init__(self, dofs, val=None):
#        """Initialise the holonomic constraint.
#
#           Args:
#               dofs(list): integers indexing *all* the degrees of freedom of the
#                           atoms subject to this Eckart constraint
#               val(float): the position at which the component is to be constrained.
#        """
#        super(EckartRotX,self).__init__(dofs, "x", val)
#
#class EckartRotY(EckartRot):
#    """Constraint on the y-component of the Eckart "angular momentum"
#    """
#    def __init__(self, dofs, val=None):
#        """Initialise the holonomic constraint.
#
#           Args:
#               dofs(list): integers indexing *all* the degrees of freedom of the
#                           atoms subject to this Eckart constraint
#               val(float): the position at which the component is to be constrained.
#        """
#        super(EckartRotY,self).__init__(dofs, "y", val)
#
#class EckartRotZ(EckartRot):
#    """Constraint on the z-component of the Eckart "angular momentum".
#    """
#    def __init__(self, dofs, val=None):
#        """Initialise the holonomic constraint.
#
#           Args:
#               dofs(list): integers indexing *all* the degrees of freedom of the
#                           atoms subject to this Eckart constraint
#               val(float): the position at which the component is to be constrained.
#        """
#        super(EckartRotZ,self).__init__(dofs, "z", val)
