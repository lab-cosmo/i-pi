"""Definition of constraint classes for constrained MD."""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np

from ipi.utils.depend import *
from ipi.utils.messages import verbosity, warning

try:
    import scipy.linalg as spla
except:
    spla = None

__all__ = ['ConstraintBase', 'ConstraintList', 'AngleConstraint', 
            'RigidBondConstraint', 'EckartConstraint', 'KDEConstraint']
           
           
class ConstraintBase(dobject):
    """ Base constraint class for MD. Takes care of indexing of 
    the atoms that are affected by the constraint, and creates
    depend objects to store the constraint function and gradient."""

    def __init__(self, constrained_indices, ncons=0, domain='centroid'):
        """ Initialize the constraint. 
        
        constrained_indices: Indices of the atoms that are involved in the constraint
        ncons: number of atom groups subject to the constraint
        domain: ['centroid'/'beads'] specifies whether the constraint is 
            calculated as a function of the centroid, or as an average over
            the beads.
        """
        
        self.constrained_indices = constrained_indices
        self.ncons = ncons
        self.domain = domain
        
        # determines the list of unique indices of atoms that are 
        # affected by this constraint. this is necessary because the 
        # same atom may appear multiple times in the definition of the constraint
        self.i_unique = np.unique(self.constrained_indices.flatten())
        self.mk_idmaps()

    def mk_idmaps(self):
        """ Creates arrays of indices that can be used for bookkeeping
        and quickly accessing elements and positions """

        # position of the UID associated with each index (i_reverse[atom_id] 
        # gives the position in the list of unique atoms of atom_id
        self.i_reverse = { value : i for i,value in enumerate(self.i_unique)}
        self.n_unique = len(self.i_unique)
        
        # access of the portion of 3-vectors corresponding to constraint atoms
        self.i3_unique = np.zeros(self.n_unique*3, int)
        for i in range(self.n_unique):
            self.i3_unique[3*i:3*(i+1)] = [3*self.i_unique[i], 3*self.i_unique[i]+1, 3*self.i_unique[i]+2]
        # this can be used to access the position array based on the constrained_indices list
        self.i3_indirect = np.zeros((len(self.constrained_indices.flatten()),3), int)
        for ri, i in enumerate(self.constrained_indices.flatten()):
            rri = self.i_reverse[i]
            self.i3_indirect[ri] = [3*rri, 3*rri+1, 3*rri+2]
            
    def bind(self, beads):
        """ Binds the constraint to a beads object - so that all of the 
        necessary quantities are easily and depend-ably accessable """
        
        self.beads = beads
        dself = dd(self)
        
        # masses of the atoms involved in the constraint - repeated 3 
        # times to vectorize operations on 3-vectors
        dself.m3 = depend_array(name="m3", value=np.zeros(self.n_unique*3),
                    func=(lambda: self.beads.m3[0,self.i3_unique]),
                    dependencies=[dd(self.beads).m3])
        
        # The constraint function is computed at iteratively updated 
        # coordinates during geodesic integration; this array holds such
        # updates. this is useful to avoid messing with the beads q 
        # until they can be updated
        if self.domain == "centroid":
            dself.q = depend_array(
                    name="q", 
                    value=dstrip(beads.qc)[None,self.i3_unique].copy()
                    )
        elif self.domain == "beads":
            dself.q = depend_array(
                    name="q", 
                    value=dstrip(beads.q)[:,self.i3_unique].copy()
                    )
        else:
            raise ValueError("Unknown constraint domain '{:s}'.".format(self.domain))
        dself.g = depend_array(name="g", value=np.zeros(self.ncons),
                       func=self.gfunc, dependencies=[dself.q])                       
        # The gradient of the constraint function is computed at the configuration
        # obtained from the previous converged SHAKE step; this array holds
        # a local copy of that configuration
        dself.qprev = depend_array(
                name="qprev", 
                value=np.zeros_like(dstrip(self.q)))
        dself.Dg = depend_array(
                name="Dg", 
                value=np.zeros((self.ncons,)+dstrip(self.q).shape),
                func=self.Dgfunc, 
                dependencies=[dself.qprev])
        dself.Gram = depend_array(
                name="Gram",
                value=np.zeros((self.ncons,self.ncons)),
                func=self.Gfunc, 
                dependencies=[dself.Dg] )
        if spla is None:
            dself.GramChol = depend_array(
                    name="GramChol",
                    value=np.zeros((self.ncons,self.ncons)),
                    func=self.GCfunc, 
                    dependencies=[dself.Gram])
        else:
            # scipy cho_factor returns both c and lower, so we need to have a depend tuple            
            dself.GramChol = depend_value(
                    name="GramChol",
                    value=(),
                    func=self.GCfunc, 
                    dependencies=[dself.Gram])

    def gfunc(self):
        """ Calculates the value of the constraint(s) """
        raise NotImplementedError()

    def Dgfunc(self):
        """
        Calculates the Jacobian of the constraint.
        """
        raise NotImplementedError()

    def Gfunc(self):
        """ Computes a cholesky decomposition of the mass-scaled Jacobian,
        used in a few places """
        
        dg = dstrip(self.Dg).copy()
        dgm = dg/self.m3
        for arr in [dg, dgm]:
            arr.shape = (self.ncons,-1)
        return np.dot(dg, dgm.T)
        
    if spla is None:
        def GCfunc(self):
            """ Computes a cholesky decomposition of the mass-scaled Jacobian,
            used in a few places """
                    
            return np.linalg.cholesky(self.Gram)
    else:
        def GCfunc(self):
            """ Computes a cholesky decomposition of the mass-scaled Jacobian,
            used in a few places """
                    
            return spla.cho_factor(self.Gram)

class ValueConstraintBase(ConstraintBase):
    """ Base class for a constraint that contains target values. """
    
    def __init__(self, constrained_indices, constraint_values, ncons, domain="centroid"):
        """ Initialize the constraint. 
        
        constrained_indices: Indices of the atoms that are involved in the constraint
        constraint_values: Value or values associated with the constraint
        ncons: number of constraints. Can be used to create a constraint class
               without specifying the values of the constraints 
        """            
        
        # NB: if constraint_values are not provided, it will 
        #     use the *initial* values of the constraints to initialize values
        if len(constraint_values) != 0:
            self.ncons = len(constraint_values)
            self._calc_cons = False
            dd(self).constraint_values = depend_array(
                    name="constraint_values",
                    value=np.asarray(constraint_values).copy())
        elif ncons != 0:
            self.ncons = ncons
            self._calc_cons = True
            dd(self).constraint_values = depend_array(
                    name="constraint_values",
                    value=np.zeros(ncons))                
        else:
            raise ValueError("cannot determine the number of constraints in list")    
        
        super(ValueConstraintBase,self).__init__(constrained_indices, ncons, domain)
                    
    def bind(self, beads):
        
        super(ValueConstraintBase,self).bind(beads)
        
        dself = dd(self)
        dself.g.add_dependency(dself.constraint_values)
        dself.Dg.add_dependency(dself.constraint_values)
        
        
class RigidBondConstraint(ValueConstraintBase):
    """ Constraint class for MD.
        Specialized for rigid bonds. This can actually hold a *list*
        of rigid bonds, i.e. there will be a list of pairs of atoms and
        a list of bond lengths. """

    def __init__(self, constrained_indices, constraint_values, domain="centroid"):

        # this is a bit perverse, but we need to function regardless of 
        # whether this is called with a shaped or flattended index list
        ncons = len(constrained_indices.flatten())/2
        super(RigidBondConstraint,self).__init__(
                constrained_indices, constraint_values, 
                ncons=ncons,domain=domain)
                
        # reshapes accessors to simplify picking individual bonds
        self.constrained_indices.shape = (self.ncons, 2)
        self.i3_indirect.shape = (self.ncons, 2, 3)

    def bind(self, beads):
        
        super(RigidBondConstraint, self).bind(beads)
        # if constraint values are not specified, initializes based on the first frame
        if self._calc_cons:
            self.constraint_values = np.sqrt(dstrip(self.g))
            
    def gfunc(self):
        """
        Calculates the deviation of the constraint from its target 
        """

        q = dstrip(self.q)
        r = np.zeros(self.ncons)
        ndof = len(q) # nbeads if domain == "beads", 1 if "centroid"
        constraint_distances = dstrip(self.constraint_values)
        for i in range(self.ncons):
            c_atoms = self.i3_indirect[i]
            c_dist = constraint_distances[i]
            r[i] = np.sum((q[:,c_atoms[0]]-q[:,c_atoms[1]])**2)/ndof - c_dist**2
        return r
    
    def Dgfunc(self, reduced=False):
        """
        Calculates the Jacobian of the constraint.
        """

        q = dstrip(self.qprev)
        ndof = len(q)
        r = np.zeros((self.ncons, ndof, self.n_unique*3))
        for i in range(self.ncons):
            c_atoms = self.i3_indirect[i]
            inst_position_vector = q[:,c_atoms[0]] - q[:,c_atoms[1]]
            r[i][:,c_atoms[0]] =   2.0 * inst_position_vector
            r[i][:,c_atoms[1]] = - 2.0 * inst_position_vector
        return r/ndof
    
class AngleConstraint(ValueConstraintBase):
    """ Constraint class for MD specialized for angles. 
        This can hold a list of angles, i.e. a list of triples of atoms
        and the corresponding values. We adopt the convention that the 
        middle atom appears first in the list.
    """

    def __init__(self,constrained_indices, constraint_values, domain="centroid"):

        ncons = len(constrained_indices.flatten())/3
        super(AngleConstraint,self).__init__(
                constrained_indices, constraint_values, 
                ncons=ncons, domain=domain)
        self.constrained_indices.shape = (self.ncons, 3)
        self.i3_indirect.shape = (self.ncons, 3, 3)
        
    def bind(self, beads):
        
        super(AngleConstraint, self).bind(beads)
        # if constraint values have not been provided, infers them from the
        # initial geometry
        if self._calc_cons:
            self.constraint_values = np.pi/2 # so that cos(angle) = 0
            self.constraint_values = np.arccos(dstrip(self.g))

    def gfunc(self):
        """
        Calculates the constraint.
        NOTE: for domain == "beads", the constraint is defined
              as the cosine of the average angle, and not as the average
              of angle cosines.
        """

        q = dstrip(self.q)
        r = np.zeros(self.ncons)
        constraint_cosines = np.cos(dstrip(self.constraint_values))
        for i in range(self.ncons):
            c_atoms = self.i3_indirect[i]
            c_cos = constraint_cosines[i]
            q1 = q[:,c_atoms[1]] - q[:,c_atoms[0]]
            r1 = np.sqrt(np.sum(q1**2, axis=-1))[...,None]
            q1 /= r1
            q2 = q[:,c_atoms[2]] - q[:,c_atoms[0]]
            r2 = np.sqrt(np.sum(q2**2, axis=-1))[...,None]
            q2 /= r2
            if self.domain == "centroid":
                r[i] = np.sum(q1*q2) - c_cos
            else:
                theta = np.mean(np.arccos(np.sum(q1*q2, axis=-1)))
                r[i] = np.cos(theta) - c_cos
        return r

    def Dgfunc(self, reduced=False):
        """
        Calculates the Jacobian of the constraint.
        """
        q = dstrip(self.qprev)
        ndof = len(q)
        r = np.zeros((self.ncons, ndof, self.n_unique*3))
        for i in range(self.ncons):
            c_atoms = self.i3_indirect[i]
            # 0-1
            q1 = q[:,c_atoms[1]] - q[:,c_atoms[0]]
            r1 = np.sqrt(np.sum(q1**2, axis=-1))[...,None]
            q1 /= r1
            # 0-2
            q2 = q[:,c_atoms[2]] - q[:,c_atoms[0]]
            r2 = np.sqrt(np.sum(q2**2, axis=-1))[...,None]
            q2 /= r2
            # jacobian
            ct = np.sum(q1*q2, axis=-1)[...,None]
            if self.domain == "centroid":
                r[i][:,c_atoms[1]] = (q2-ct*q1)/r1
                r[i][:,c_atoms[2]] = (q1-ct*q2)/r2
                r[i][:,c_atoms[0]] = -(r[i][:,c_atoms[1]] + 
                                       r[i][:,c_atoms[2]])
            else:
                # sin(theta)
                st = np.sqrt(1.0-ct**2) 
                # sin(mean(theta))/ndof
                smt = np.sin(np.mean(np.arccos(ct.reshape(-1))))/ndof
                r[i][:,c_atoms[1]] = (q2-ct*q1)/(r1*st)*smt
                r[i][:,c_atoms[2]] = (q1-ct*q2)/(r2*st)*smt
                r[i][:,c_atoms[0]] = -(r[i][:,c_atoms[1]] + 
                                       r[i][:,c_atoms[2]])
        return r
    
class EckartConstraint(ConstraintBase):
    """ Constraint class for MD specialized for enforcing the Eckart conditions
        (see E. Bright Wilson et al. 'Molecular Vibrations') 
        Unlike the constraints above, a single instance of this class can only
        describe one set of Eckart conditions.
    """

    def __init__(self, constrained_indices, constraint_values, domain="centroid"):
        
        if domain != "centroid":
            raise ValueError(
            "Eckart constraints not defined for domain {:s}".format(domain))
        super(EckartConstraint,self).__init__(constrained_indices, 6)
        self.constrained_indices.shape = -1
        dd(self).constraint_values = depend_array(name="constraint_values",
                value=np.zeros(self.ncons)) 
                
        # Check that there are no repeats
        if np.any(self.constrained_indices != self.i_unique):
            raise ValueError("repeated atom indices in EckartConstraint")
        self.i3_indirect.shape = (-1, 3)
        if len(constraint_values) == 0:
            self._calc_cons = True
            dd(self).qref = depend_array(
                    name="qref", 
                    value=np.zeros((1,)+self.i3_indirect.shape))
        else:
            self._calc_cons = False
            dd(self).qref = depend_array(
                    name="qref", 
                    value=np.reshape(
                            constraint_values, 
                            (1,)+self.i3_indirect.shape).copy())
        
    def bind(self, beads):
        
        if not self._calc_cons:
            qref = dstrip(self.qref).copy()
        super(EckartConstraint, self).bind(beads)
        if self._calc_cons:
            self.qref = np.reshape(dstrip(self.q).copy(), (1,-1,3))
        else:
            self.qref = qref
        dself = dd(self)
        dself.g.add_dependency(dself.constraint_values)
        dself.Dg.add_dependency(dself.constraint_values)
        
        # Total mass of the group of atoms
        dself.mtot = depend_value(name="mtot", value=1.0, 
            func=(lambda: dstrip(self.m3)[::3].sum()),
            dependencies=[dself.m3]
            )
            
        # Coords of reference centre of mass
        dself.qref_com = depend_array(
                name="qref_com", value=np.zeros(3, float),
                func=(lambda: np.sum(
                      dstrip(self.qref).reshape((-1,3)) *
                      dstrip(self.m3).reshape((-1,3)),
                      axis=0)/self.mtot),
                dependencies=[dself.m3, dself.qref]
                )
        # qref in its centre of mass frame
        dself.qref_rel = depend_array(
                name="qref_rel",
                value=np.zeros(dstrip(self.qref).shape[1:]),
                func=(lambda: dstrip(self.qref).reshape(-1,3) -
                              dstrip(self.qref_com)),
                dependencies=[dself.qref, dself.qref_com]
                )
        # qref in the CoM frame, mass-weighted
        dself.mqref_rel = depend_array(
                name="mqref_rel", 
                value=np.zeros(dstrip(self.qref).shape[1:]),
                func=(lambda: 
                    dstrip(self.qref_rel)*dstrip(self.m3).reshape((-1,3))),
                dependencies=[dself.qref_rel, dself.m3]
                )
        # Make constraint function and gradient depend on the parameters
        dself.g.add_dependency(dself.qref)
        dself.g.add_dependency(dself.m3)
        dself.Dg.add_dependency(dself.qref)
        dself.Dg.add_dependency(dself.m3)
        
    def gfunc(self):
        """
        Calculates the constraint.
        """

        q = dstrip(self.q).reshape((-1,3))
        m = dstrip(self.m3).reshape((-1,3))
        qref = dstrip(self.qref).reshape((-1,3))
        r = np.zeros(self.ncons)
        Delta = q-qref
        r[:3] = np.sum(m*Delta, axis=0)/self.mtot
        r[3:] = np.sum(np.cross(dstrip(self.mqref_rel), Delta), axis=0)/self.mtot
        return r

    def Dgfunc(self, reduced=False):
        """
        Calculates the Jacobian of the constraint.
        """

        r = np.zeros((self.ncons, self.n_unique, 3))
        m = dstrip(self.m3).reshape((-1,3))
        mqref_rel = dstrip(self.mqref_rel).reshape((-1,3))
        for i in range(3):
            r[i,:,i] = m[:,i]
        # Eckart rotation, x-component
        r[3,:,1] =-mqref_rel[:,2]
        r[3,:,2] = mqref_rel[:,1]
        # Eckart rotation, y-component
        r[4,:,0] = mqref_rel[:,2]
        r[4,:,2] =-mqref_rel[:,0]
        # Eckart rotation, z-component
        r[5,:,0] =-mqref_rel[:,1]
        r[5,:,1] = mqref_rel[:,0]
        r /= self.mtot
        r.shape = (self.ncons,1,-1)
        return r

class KDEConstraint(ValueConstraintBase):
    """ Constraint class that fixes the mode of the ring polymer distribution
        estimated based on a kernel density estimation.
    """

    def __init__(self, constrained_indices, constraint_values, domain="beads", sigma = 1.0):
        
        if domain != "beads":
            raise ValueError(
            "KDE constraints not defined for domain {:s}".format(domain))
        
        ncons = len(constrained_indices.flatten())*3 
        super(KDEConstraint,self).__init__(constrained_indices, constraint_values, ncons, domain=domain)                
        self.constraint_values.shape = (ncons)
        self.sigma = sigma
                
        # Check that there are no repeats
        if np.any(self.constrained_indices != self.i_unique):
            raise ValueError("Repeated atom indices in KDEConstraint")
        
    def bind(self, beads):
        
        if not self._calc_cons:
            qref = dstrip(self.qref).copy()
        
        super(KDEConstraint, self).bind(beads)
        
        if self._calc_cons:
            self.qref = dstrip(beads.q[:,self.i3_unique]).copy()      
            # computes the constraint as the linear centroid
            self.constraint_values = self.qref.mean(axis=0)      
        else:
            self.qref = qref
                
        dself = dd(self)
        dself.g.add_dependency(dself.constraint_values)
        dself.Dg.add_dependency(dself.constraint_values)
         
    def gfunc(self):
        """
        Calculates the position of the mode of the KDE by iterative mode-seeking
        """

        ndof = len(self.q)
        q = dstrip(self.q).reshape((ndof,self.n_unique,3)).copy()        
        x = self.constraint_values.reshape((self.n_unique, 3))
        
        
        for i in xrange(4): # make it an option, maxiter
            dq = q-x
            print "KDE CONSTRAINTS _____ "
            print "target", x
            print "distance", q
            # distances of each atom to its target
            d2 = np.sum((dq)**2,axis=2)
            gw = np.exp(-d2*0.5/self.sigma**2) 
            gw /= gw.sum(axis=0) 
            qmode = np.einsum("bjk,bj->jk", q, gw)
            if np.linalg.norm(qmode-x)<1e-8: 
                break
            x = qmode
        
        print "qmode and x ", qmode, x
                
        return qmode.flatten()

    def Dgfunc(self, reduced=False):
        """
        Calculates the Jacobian of the constraint.
        """

        
        ndof = len(self.q)
        r = np.ones((self.ncons, ndof, self.n_unique*3))        
        return r

class ConstraintList(ConstraintBase):
    """ Class to hold a list of constraints to be treated 
    simultaneously by the solver."""

    def __init__(self, constraint_list):
        """ Initialize the constraint list. 
        Contains a list of constraint objects, that will
        be stored and used to compute the different blocks
        of the constraint values and Jacobian """
        
        
        self.constraint_list = constraint_list        
        self.ncons = sum([constr.ncons for constr in constraint_list])

        # determines the list of unique indices of atoms that are affected 
        # by this list of constraint
        self.i_unique = np.zeros(0)
        self.domains = []
        for c in self.constraint_list:
            self.i_unique = np.union1d(
                    c.constrained_indices.flatten(),self.i_unique
                    )
            self.domains.append(c.domain)
                    
        super(ConstraintList,self).__init__(self.i_unique, self.ncons, domain="beads")

        # must now find the mapping from the unique indices in each constraint 
        # ic_map[i] gives the position where the atoms involved 
        # in the i-th constraint are stored in the compact list
        self.ic_map = []
        self.ic3_map = []
        for c in self.constraint_list:
            i_map = np.array([self.i_reverse[i] for i in c.i_unique])
            self.ic_map.append(i_map)
            self.ic3_map.append(np.array([[i*3, i*3+1, i*3+2] for i in i_map]).flatten())

    def bind(self, beads):

        super(ConstraintList, self).bind(beads)
        
        # we link all of the sub-constraints to the lists of unique q and qprev,
        # so that each constraint gets automatically updated. This involves 
        # defining a function that transfer q and qprev to the individual
        # constraints, and setting dependencies appropriately
        dself = dd(self)
        def make_qgetter(k, domain):
            if domain == "centroid":
                return lambda: np.mean(self.q[:,None,self.ic3_map[k]], axis=0)
            else:
                return lambda: self.q[:,self.ic3_map[k]]
        def make_qprevgetter(k, domain):
            if domain == "centroid":
                return lambda: np.mean(self.qprev[:,None,self.ic3_map[k]], axis=0)
            else:
                return lambda: self.qprev[:,self.ic3_map[k]]
        for ic, c in enumerate(self.constraint_list):
            c.bind(beads)
            # deal with constraint functions
            dq = dd(c).q
            dq.add_dependency(dself.q)
            dq._func = make_qgetter(ic, c.domain)
            dself.g.add_dependency(dd(c).g)
            # ...and their gradients
            dqprev = dd(c).qprev
            dqprev.add_dependency(dself.qprev)
            dqprev._func = make_qprevgetter(ic, c.domain)
            dself.Dg.add_dependency(dd(c).Dg)

    def gfunc(self):
        """
        Compute the constraint function.
        """
        r = np.zeros(self.ncons)
        si = 0
        for constr in self.constraint_list:
            r[si:si+constr.ncons] = constr.g
            si += constr.ncons
        return r

    def Dgfunc(self):
        """
        Compute the Jacobian of the constraint function.
        """
        q = dstrip(self.qprev)
        ndof = len(q)
        r = np.zeros((self.ncons,)+q.shape)
        si = 0
        for ic, constr in enumerate(self.constraint_list):
            if constr.domain == "beads":
                r[si:si+constr.ncons,:,self.ic3_map[ic]] = constr.Dg
            else:
                r[si:si+constr.ncons,:,self.ic3_map[ic]] = constr.Dg/ndof
            si += constr.ncons
        return r

    def get_iai(self):
        iai = []
        for constr in self.constraint_list:
            iai += list(constr.get_iai())
        return np.unique(iai)

