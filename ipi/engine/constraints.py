"""Classes that deal with holonomic constraints.

Contains objects that return the values and the gradients of holonomic constraint
functions.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2019 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np

from ipi.utils.depend import dstrip
from ipi.utils import nmtransform

__all__ = ['Replicas','HolonomicConstraint','BondLength','BondAngle',
           'EckartTransX', 'EckartTransY', 'EckartTransZ',
           'EckartRotX', 'EckartRotY', 'EckartRotZ',]

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
        nbeads: The number of beads in the constrained ring-polymer DoFs
        ndof: The number of DoFs in the constrained group
        open_paths: the open-path ring-polymer DOFs
                    in the constrained group
        targetval: The target value of the constraint function

    Properties:
        sigma: Difference between the current value of the constraint
               and targetval
        jac: Gradient of the constraint function w.r.t the constrained
             degrees of freedom (ndof-by-nbeads array)
        mjac: Gradient left-multiplied by the inverse of the mass tensor
        sigmadot: Total time derivative of the constraint function

    Methods:
        qtaint: Taint the properties that depend on the configurational
                degrees of freedom
        ptaint: Taint sigmadot only

    """
    # number of degrees of freedom expected by the constraint function
    _ndof = 0

    @classmethod
    def get_ndof(cls):
        """The number of degrees of freedom expected by the constraint function.
        """
        if (cls._ndof == 0):
            raise TypeError("Trying to get the number of degrees of freedom "+
                            "from base HolonomicConstraint class")
        return cls._ndof

    def __init__(self, dofs, val=None):
        """Initialise the holonomic constraint.
        """
        ndof = self.get_ndof()
        dofs = np.asarray(dofs, dtype=int).copy()
        if (dofs.ndim != 1):
            raise ValueError("Shape of constrained DoF group incompatible "+
                             "with "+self.__class__.__name__)
        if (dofs.size != ndof and ndof != -1):
            raise ValueError("Size of constrained DoF group incompatible "+
                             "with "+self.__class__.__name__)
        self.dofs = dofs
        self.ndof = len(self.dofs)
        if val is not None:
            if not np.isscalar(val):
                raise TypeError("Expecting a scalar for target constraint value")
        self.targetval = val

    def bind(self, replicas, nm, transform=None, **kwargs):
        """Bind the appropriate degrees of freedom to the holonomic constraint.
           Args:
               replicas(list): List of Replicas objects containing the
                   ring-polymer data grouped by degree
                   of freedom.
               nm: A normal modes object used to do the normal modes
                   transformation.
               transform: a normal mode transform object that is
                   agnostic to the number of atoms and has no open paths
                   (either no-op or nm_trans)
        """

        self.nbeads = len(replicas[0])
        # Local reference to the normal mode transform
        self._nm = nm
        # Local transform (matrix version, agnostic to number of atoms)
        if transform is not None:
            #!! TODO add checks for type of transform and _open == []
            self.transform = transform
        elif self.nbeads == 1:
            self.transform = nmtransform.nm_noop(nbeads = self.nbeads)
        else:
            self.transform = nmtransform.nm_trans(nbeads = self.nbeads)
        # Local references to array sections involved in the constraint
        self._q = [replicas[i].q for i in self.dofs]
        self._p = [replicas[i].p for i in self.dofs]
        self._m = [replicas[i].m for i in self.dofs]
        # Private attributes for caching
        self.__qtainted = False
        self.__mtainted = False
        self.__ptainted = False
        self.__sigma = 0.0
        self.__sigmadot = 0.0
        self.__jac = np.zeros_like(np.asarray(self._q))
        self.__mjac = np.zeros_like(self.__jac)

    def qtaint(self):
        """Notify the class that the holonomic constraint and its gradient
           have to be recalculated when next referenced.
        """
        self.__qtainted = True
        self.__mtainted = True
        self.ptaint() # It follows that sigmadot is also affected

    def ptaint(self):
        """Notify the class that the time-derivative of the constraint will
           have to be recalculated when next referenced.
        """
        self.__ptainted = True

    @property
    def sigma(self):
        if self.__qtainted:
            self.update_qfxns()
        return self.__sigma - self.targetval

    @property
    def jac(self):
        if self.__qtainted:
            self.update_qfxns()
        return self.__jac

    @property
    def mjac(self):
        if self.__mtainted:
            self.update_mjac()
            self.__mtainted = False
        return self.__mjac

    @property
    def sigmadot(self):
        if self.__ptainted:
            self.__sigmadot = self.get_sigmadot()
            self.__ptainted = False
        return self.__sigmadot

    def update_qfxns(self):
        """Re-calculate the constraint function and its gradient.
        """
        self.__sigma, self.__jac[...] = self.get_sigma()
        self.__qtainted = False
        self.__ptainted = True # Sigmadot now has to be recalculated

    def get_sigma(self):
        """Dummy constraint function calculator that does nothing."""
        pass

    def update_mjac(self):
        """Calculate the constraint gradient, weighted by the inverse mass tensor"""

        self.__mjac[...] = self.jac
        if self.nbeads == 1:
            self.__mjac /= dstrip(self._nm.dynm3)[0,self.dofs,None]
#            for i,idof in enumerate(self.dofs):
#                mjac[i,:] /= dstrip(self._nm.dynm3)[:,idof]
        else:
            gnm = self.transform.b2nm(self.__mjac.T)
            for i,idof in enumerate(self.dofs):
                # Override if DoF in open path
                if idof//3 in self._nm.transform._open:
                    gnm[:,i] = np.dot(self._nm.transform._b2o_nm,
                                      self.__mjac[i,:])
            # Divide by the dynamical mass
            gnm /= dstrip(self._nm.dynm3)[:,self.dofs]
            self.__mjac[...] = self.transform.nm2b(gnm).T
            for i,idof in enumerate(self.dofs):
                if idof//3 in self._nm.transform._open:
                    self.__mjac[i,:] = np.dot(self._nm.transform._o_nm2b,
                                              gnm[:,i])

    def get_sigmadot(self):
        """Calculate the total derivative of the constraint function w.r.t. time."""
        return np.sum(np.asarray(self._p)*self.mjac)

#------------- Specialisations of the Holonomic Constraint class -----------#
class BondLength(HolonomicConstraint):
    """Constraint on the length of a bond between two atoms,
       averaged over the replicas.
    """

    # Six degrees of freedom --- three per atom
    _ndof = 6

    def bind(self, replicas, nm, transform=None, **kwargs):
        """Bind the appropriate degrees of freedom to the holonomic constraint.
        """

        super(BondLength, self).bind(replicas, nm, transform, **kwargs)
        self.qtaint()
        if self.targetval is None:
            # Set to the current mean bond-length
            self.targetval = 0.0
            currentval = self.sigma
            self.targetval = currentval

    def get_sigma(self):
        """Calculate the difference between the mean bond-length and the
           target value and its gradient.
        """
        qAB = np.asarray([self._q[i+3]-self._q[i] for i in range(3)])
        rAB = np.sqrt(np.sum(qAB**2, axis=0))
        qAB /= rAB
        qAB /= self.nbeads
        return np.mean(rAB), np.vstack((-qAB,qAB))

class BondAngle(HolonomicConstraint):
    """Constraint on the A--X--B bond-angle, averaged over the replicas.
       The indices of the coordinates of the central atom X are supplied
       first, followed by A and B.
    """

    # Nine degrees of freedom -- three per atom
    _ndof =  9

    def bind(self, replicas, nm, transform=None, **kwargs):
        """Bind the appropriate degrees of freedom to the holonomic constraint.
        """

        super(BondAngle,self).bind(replicas, nm, transform, **kwargs)
        self.qtaint()
        if self.targetval is None:
            # Set to initial bond angle
            self.targetval = 0.0
            currentval = self.sigma
            self.targetval = currentval

    def get_sigma(self):
        """Calculate the difference between the mean angle and the
           target value.
        """
        qXA = np.asarray([self._q[i+3]-self._q[i] for i in range(3)])
        rXA = np.sqrt(np.sum(qXA**2, axis=0))
        qXA /= rXA
        qXB =  np.asarray([self._q[i+6]-self._q[i] for i in range(3)])
        rXB = np.sqrt(np.sum(qXB**2, axis=0))
        qXB /= rXB
        # Cosine and sine of the angle
        ct = np.sum(qXA*qXB, axis=0)
        st = np.sqrt(1.0-ct**2)
        # Gradients w.r.t peripheral atoms
        jac_A = ct*qXA - qXB
        jac_A /= st
        jac_A /= rXA
        jac_A /= self.nbeads
        jac_B = ct*qXB - qXA
        jac_B /= st
        jac_B /= rXB
        jac_B /= self.nbeads

        return np.mean(np.arccos(ct)), np.vstack((-(jac_A+jac_B), jac_A, jac_B))

class EckartTrans(HolonomicConstraint):
    """Constraint on one of the components of the centre of mass.
    """

    # Number of DoFs determined upon initialisation
    _ndof = -1

    def __init__(self, dofs, coord, val=None):
        """Initialise the holonomic constraint.

           Args:
               dofs(list): integers indexing *all* the degrees of freedom of the
                           atoms subject to this Eckart constraint
               coord(str): 'x', 'y', 'z' -- specifies the component of the CoM
                           to be constrained
               val(float): the position at which the component is to be constrained.
        """
        q_str = coord.lower()
        if q_str=="x":
            idx = 0
        elif q_str=="y":
            idx = 1
        elif q_str=="z":
            idx = 2
        else:
            raise ValueError("Invalid coordinate specification supplied to "+
                             self.__class__.__name__)
        super(EckartTrans,self).__init__(dofs[idx::3], val)

    def bind(self, replicas, nm, transform=None, **kwargs):
        """Bind the appropriate coordinates to the constraint.
        """

        super(EckartTrans, self).bind(replicas, nm, transform, **kwargs)
        self.qtaint()
        if self.targetval is None:
            # Set to the current position of the CoM
            self.targetval = 0.0
            currentval = self.sigma
            self.targetval = currentval

    def get_sigma(self):
        mrel = np.asarray(self._m).reshape(-1)
        mtot = np.sum(mrel)
        mrel /= mtot
        qc = np.mean(np.asarray(self._q),axis=-1)
        sigma = np.dot(mrel, qc)
        mrel /= self.nbeads
        return sigma, mrel[:,None]

class EckartTransX(EckartTrans):
    """Constraint on the x-component of the CoM of a group of atoms.
    """
    def __init__(self, dofs, val=None):
        """Initialise the holonomic constraint.

           Args:
               dofs(list): integers indexing *all* the degrees of freedom of the
                           atoms subject to this Eckart constraint
               val(float): the position at which the component is to be constrained.
        """
        super(EckartTransX,self).__init__(dofs, "x", val)

class EckartTransY(EckartTrans):
    """Constraint on the y-component of the CoM of a group of atoms.
    """
    def __init__(self, dofs, val=None):
        """Initialise the holonomic constraint.

           Args:
               dofs(list): integers indexing *all* the degrees of freedom of the
                           atoms subject to this Eckart constraint
               val(float): the position at which the component is to be constrained.
        """
        super(EckartTransY,self).__init__(dofs, "y", val)

class EckartTransZ(EckartTrans):
    """Constraint on the z-component of the CoM of a group of atoms.
    """
    def __init__(self, dofs, val=None):
        """Initialise the holonomic constraint.

           Args:
               dofs(list): integers indexing *all* the degrees of freedom of the
                           atoms subject to this Eckart constraint
               val(float): the position at which the component is to be constrained.
        """
        super(EckartTransZ,self).__init__(dofs, "z", val)

class EckartRot(HolonomicConstraint):
    """One of the components of the Eckart rotational constraint.

       NOTE: in this definition the usual sum over cross-products is divided by
             the total mass of the system.
    """

    # Number of DoFs determined upon initialisation
    _ndof = -1

    def __init__(self, dofs, coord, val=None):
        """Initialise the holonomic constraint.

           Args:
               dofs(list): integers indexing *all* the degrees of freedom of the
                           atoms subject to this Eckart constraint
               coord(str): 'x', 'y', 'z' -- specifies the component of the
                           angular-momentum-like quantity to be constrained
               val(array-like): this argument is not used
        """
        q_str = coord.lower()
        if q_str=="x":
            idces = (1,2)
        elif q_str=="y":
            idces = (2,0)
        elif q_str=="z":
            idces = (0,1)
        else:
            raise ValueError("Invalid coordinate specification supplied to "+
                             self.__class__.__name__)
        super(EckartRot,self).__init__(dofs[idces[0]::3]+dofs[idces[1]::3], 0.0)

    def bind(self, replicas, nm, transform=None, **kwargs):
        """Bind the appropriate coordinates to the constraints.
          Args:
              replicas(list): List of Replicas
              nm: A normal modes object used to do the normal modes transformation.
          **kwargs:
              ref(array-like): Reference configuration for the constraint; if
                               absent, taken to be the centroid configuration

        """



        super(EckartRot, self).bind(replicas, nm, transform, **kwargs)
        self.qtaint()
        if "ref" in kwargs:
            # Reference configuration is provided
            lref = np.asarray(kwargs["ref"]).flatten() # local copy
            ref = np.asarray(
                    [ lref[i] for i in self.dofs ]).reshape((2, self.ndof//2))
        else:
            # Initialise to centroid configuration
            ref = np.asarray(
                    [ np.mean(q) for q in self._q] ).reshape((2, self.ndof//2))
        self._ref = ref

    def get_sigma(self):
        # Individual centroid masses divided by the molecular mass
        mrel = np.asarray(self._m).reshape((2,self.ndof//2))
        mrel /= mrel.sum(axis=-1)[:,None]
        # Reference geometry in its CoM, weighted by mrel
        mref = mrel*self._ref
        CoM = mref.sum(axis=-1)
        mref[...] = self._ref-CoM[:,None]
        mref *= mrel
        # Displacement between centroid and reference configs
        delqc = np.mean(np.asarray(self._q), axis=-1).reshape((2, self.ndof//2))
        delqc -= self._ref
        sigma = np.sum(mref[0]*delqc[1] - mref[1]*delqc[0])
        mref /= self.nbeads
        return sigma, np.hstack((-mref[1],mref[0]))[:,None]

class EckartRotX(EckartRot):
    """Constraint on the x-component of the Eckart "angular momentum"
    """
    def __init__(self, dofs, val=None):
        """Initialise the holonomic constraint.

           Args:
               dofs(list): integers indexing *all* the degrees of freedom of the
                           atoms subject to this Eckart constraint
               val(float): the position at which the component is to be constrained.
        """
        super(EckartRotX,self).__init__(dofs, "x", val)

class EckartRotY(EckartRot):
    """Constraint on the y-component of the Eckart "angular momentum"
    """
    def __init__(self, dofs, val=None):
        """Initialise the holonomic constraint.

           Args:
               dofs(list): integers indexing *all* the degrees of freedom of the
                           atoms subject to this Eckart constraint
               val(float): the position at which the component is to be constrained.
        """
        super(EckartRotY,self).__init__(dofs, "y", val)

class EckartRotZ(EckartRot):
    """Constraint on the z-component of the Eckart "angular momentum".
    """
    def __init__(self, dofs, val=None):
        """Initialise the holonomic constraint.

           Args:
               dofs(list): integers indexing *all* the degrees of freedom of the
                           atoms subject to this Eckart constraint
               val(float): the position at which the component is to be constrained.
        """
        super(EckartRotZ,self).__init__(dofs, "z", val)
