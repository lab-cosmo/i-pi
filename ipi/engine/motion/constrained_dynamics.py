"""Contains the classes that deal with the different dynamics required in
different types of ensembles.

Holds the algorithms required for normal mode propagators, and the objects to
do the constant temperature and pressure algorithms. Also calculates the
appropriate conserved energy quantity for the ensemble of choice.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import numpy as np
import warnings
import sys

from ipi.utils.nmtransform import nm_fft
from ipi.engine.motion import Dynamics
from ipi.engine.motion.dynamics import DummyIntegrator, NVEIntegrator
from ipi.utils.depend import depend_value, depend_array, \
                             dd, dobject, dstrip, dpipe
from ipi.engine.thermostats import Thermostat
from ipi.engine.barostats import Barostat
from ipi.engine.quasicentroids import QuasiCentroids


class ConstrainedDynamics(Dynamics):

    """self (path integral) constrained molecular dynamics class.

    Gives the standard methods and attributes needed in all the
    constrained dynamics classes.

    Attributes:
        beads: A beads object giving the atoms positions.
        cell: A cell object giving the system box.
        forces: A forces object giving the virial and the forces acting on
            each bead.
        prng: A random number generator object.
        nm: An object which does the normal modes transformation.

    Depend objects:
        econs: The conserved energy quantity appropriate to the given
            ensemble. Depends on the various energy terms which make it up,
            which are different depending on the ensemble.he
        temp: The system temperature.
        dt: The timestep for the algorithms.
        ntemp: The simulation temperature. Will be nbeads times higher than
            the system temperature as PIMD calculations are done at this
            effective classical temperature.
    """

    def __init__(self, timestep, mode="nve", splitting="obabo",
                thermostat=None, barostat=None,
                quasicentroids=None, fixcom=False, fixatoms=None,
                nmts=None, nsteps_geo=1, constraint_groups=[]):

        """Initialises a "ConstrainedDynamics" motion object.

        Args:
            dt: The timestep of the simulation algorithms.
            fixcom: An optional boolean which decides whether the centre of mass
                motion will be constrained or not. Defaults to False.
        """

        super(Dynamics, self).__init__(fixcom=fixcom, fixatoms=fixatoms)
        dd(self).dt = depend_value(name='dt', value=timestep)
        if thermostat is None:
            self.thermostat = Thermostat()
        else:
            self.thermostat = thermostat
        if barostat is None:
            self.barostat = Barostat()
        else:
            self.barostat = barostat
        if quasicentroids is None:
            self.quasicentroids = QuasiCentroids()
        else:
            self.quasicentroids = quasicentroids
        self.enstype = mode
        if nmts is None or len(nmts) == 0:
            dd(self).nmts = depend_array(name="nmts", value=np.asarray([1], int))
        else:
            dd(self).nmts = depend_array(name="nmts", value=np.asarray(nmts, int))
        if self.enstype == "nve":
            self.integrator = NVEConstrainedIntegrator()
        elif self.enstype == "nvt":
            self.integrator = NVTConstrainedIntegrator()
        else:
            self.integrator = DummyIntegrator()
        # splitting mode for the integrators
        dd(self).splitting = depend_value(name='splitting', value=splitting)
        # constraints
        self.fixcom = fixcom
        if fixatoms is None:
            self.fixatoms = np.zeros(0, int)
        else:
            self.fixatoms = fixatoms
        self.constraint_groups = constraint_groups
        self.csolver = ConstraintSolver(self.constraint_groups)
        self.nsteps_geo = nsteps_geo

    def bind(self, ens, beads, nm, cell, bforce, prng, omaker):
        """Binds ensemble beads, cell, bforce, and prng to the dynamics.

        This takes a beads object, a cell object, a forcefield object and a
        random number generator object and makes them members of the ensemble.
        It also then creates the objects that will hold the data needed in the
        ensemble algorithms and the dependency network.

        Args:
            ens: The ensemble object specifying the thermodynamic state
                of the system.
            beads: The beads object from whcih the bead positions are taken.
            nm: A normal modes object used to do the normal-mode
                transformation.
            cell: The cell object from which the system box is taken.
            bforce: The forcefield object from which the force and virial are
                taken.
            prng: The random number generator object which controls random number
                generation.
            omaker: output maker
        """
        
        # Bind the constraints first
        for cgp in self.constraint_groups:
            cgp.bind(beads, nm)
        self.csolver.bind(nm)
        # Rest as in dynamics
        super(ConstrainedDynamics, self).bind(ens, beads, nm, cell, 
                                              bforce, prng, omaker)

        
class ConstraintBase(dobject):
    """Base constraint class; defines the constraint function and its Jacobian.
    """
    
    # Add constrained indices and values
    def __init__(self, index_list, targetvals=None,
                 tolerance=1.0e-4, domain="cartesian", ngp=0):
        """Initialise the constraint.

        Args:
            index_list: 1-d list of indices of the affected atoms
            targetvals: target values of the constraint function
            tolerance: the desired tolerance to which to converge the
            constraint
            domain: ['cartesian'/'normalmode'/'centroid'] - specifies whether 
            the constraint is expressed in terms of Cartesian, normalmode 
            or centroid coordinates.
            ngp: number of constraint groups; by default calculated internally
        """
        
        self.tol = tolerance
        dself = dd(self)
        dself.natoms = depend_value(name="natoms", func=self.get_natoms)
        if targetvals is None:
            dself.targetvals = None
        else:
            dself.targetvals = depend_array(
                    name="targetvals",
                    value=np.asarray(targetvals).flatten())
        dself.ilist = depend_array(
                name="ilist", 
                value=np.reshape(np.asarray(index_list), (-1, self.natoms)))
        counts = np.unique(dstrip(self.ilist), return_counts=True)[1]
        if np.any(counts != 1):
            raise ValueError(
"Constraint given overlapping groups of atoms.")
        dself.ngp = depend_value(name="ngp", value=len(self.ilist))
        dself.domain = domain.lower()
        if self.domain not in ["cartesian", "normalmode", "centroid"]:
            raise ValueError("Unknown constraint domain '{:s}'.".format(domain))
            
    def bind(self, beads, nm):
        """Bind the beads and the normal modes to the constraint.
        """
        
        self.beads = beads
        self.nm = nm
        dself = dd(self)
        arr_shape = (self.nm.nbeads, self.ngp, 3*self.natoms)
        i3list = np.asarray([i for j in dstrip(self.ilist).flatten()
                               for i in range(3*j, 3*j+3)])
        # Configurations of the affected beads (later to be made dependent
        # on sections of arrays in grouped constraints)
        
        dself.qnm = depend_array(name="qnm", value = 
                np.transpose(np.reshape(
                dstrip(self.nm.qnm)[:,i3list], 
                arr_shape), [1,2,0]))
        dself.q = depend_array(name="q", value = 
                np.transpose(np.reshape(
                dstrip(self.beads.q)[:,i3list], 
                arr_shape), [1,2,0]))
        dself.qc = depend_array(name="qc", value = 
                np.transpose(np.reshape(
                dstrip(self.beads.qc)[i3list], 
                (1,)+arr_shape[1:]), [1,2,0]))
        dself.qnmprev = depend_array(
                    name="qnmprev", value=dstrip(self.qnm).copy())
        dself.qprev = depend_array(
                    name="qprev", value=dstrip(self.q).copy())
        dself.qcprev = depend_array(
                    name="qcprev", value=dstrip(self.qc).copy())
        # Constraint functions and their derivatives
        if self.domain == "cartesian":
            deps = [dself.q, dself.qprev]
            fxns = [lambda: self.gfunc(dstrip(self.q)),
                    lambda: self.Dgfunc(dstrip(self.qprev))]
        elif self.domain == "normalmode":
            deps = [dself.qnm, dself.qnmprev]
            fxns = [lambda: self.gfunc(dstrip(self.qnm)),
                    lambda: self.Dgfunc(dstrip(self.qnmprev))]
        elif self.domain == "centroid":
            deps = [dself.qc, dself.qcprev]
            fxns = [lambda: self.gfunc(dstrip(self.qc)),
                    lambda: self.Dgfunc(dstrip(self.qcprev))]
        dself.g = depend_array(
                name="g", value=np.zeros(self.ngp),
                func=fxns[0], dependencies=[deps[0]])
        dself.Dg = depend_array(
                name="Dg", 
                value=np.zeros((self.ngp, 3*self.natoms, self.nm.nbeads)), 
                func=fxns[1], dependencies=[deps[1]])
        if self.targetvals is None:
            dself.targetvals = depend_array(name="targetvals",
                                            value=dstrip(self.g).copy())
            
    def get_natoms(self):
        """Return the number of atoms involved in the constraint
        """
        return -1

    def norm(self, x):
        """Defines the norm of the constraint function; typically just
        the absolute value.
        """
        return np.abs(x)

    def gfunc(self, q):
        if q.ndim != 3:
            raise ValueError(
                "Constraint.gfunc expects a three-dimensional input.")
        if self.domain == "centroid" and q.shape[-1] != 1:
            raise ValueError(
                "Constraint.gfunc given input with shape[-1] != 1 when "+
                "centroid domain was specified."
                )

    def Dgfunc(self, q):
        if q.ndim != 3:
            raise ValueError(
                    "Constraint.Dgfunc expects a three-dimensional input.")
        if self.domain == "centroid" and q.shape[-1] != 1:
            raise ValueError(
                "Constraint.gfunc given input with shape[-1] != 1 when "+
                "centroid domain was specified."
                )
            
class BondLengthConstraint(ConstraintBase):
    """Constrain the mean bond-length
    """
    def __init__(self, index_list, targetvals=None,
                 tolerance=1.0e-4, domain="cartesian", ngp=0):
        super(BondLengthConstraint, self).__init__(index_list, targetvals,
                                                   tolerance, domain, ngp)
        if self.domain == "normalmode":
            warnings.warn(
                "Using the 'BondLength' constraint in the 'normalmode' domain "+
                "may have unpredictable effects.")
            
    def get_natoms(self):
        """Return the number of atoms involved in the constraint
        """
        return 2

    def gfunc(self, q):
        """Calculate the bond-length, averaged over the beads. 
        """

        super(BondLengthConstraint, self).gfunc(q)
        ngp, ncart, nbeads = q.shape
        x = np.reshape(q, (ngp, 2, 3, nbeads))
        xij = x[:,1]-x[:,0]
        return np.sqrt(np.sum(xij**2, axis=1)).mean(axis=-1)

    def Dgfunc(self, q):
        """Calculate the Jacobian of the constraint function.
        """

        super(BondLengthConstraint, self).Dgfunc(q)
        ngp, ncart, nbeads = q.shape
        
        x = np.reshape(q, (ngp, 2, 3, nbeads))
        xij = x[:,1]-x[:,0] # (ngp, 3, nbeads)
        r = np.sqrt(np.sum(xij**2, axis=1)) # (ngp, nbeads)
        xij /= r[:,None,:]
        return np.concatenate((-xij,xij), axis=1)/nbeads
    
class BondAngleConstraint(ConstraintBase):
    """Constraint the mean bond-angle.
    """
    
    def __init__(self, index_list, targetvals=None,
                 tolerance=1.0e-4, domain="cartesian", ngp=0):
        super(BondAngleConstraint, self).__init__(index_list, targetvals,
                                                  tolerance, domain, ngp)
        if self.domain == "normalmode":
            warnings.warn(
                "Using the 'BondAngle' constraint in the 'normalmode' domain "+
                "may have unpredictable effects.")
            
    def get_natoms(self):
        """Return the number of atoms involved in the constraint
        """
        return 3

    def gfunc(self, q):
        """Calculate the bond-angle, averaged over the beads. 
        """
        super(BondAngleConstraint, self).gfunc(q)
        ngp, ncart, nbeads = q.shape
        x = np.reshape(q, (ngp, 3, 3, nbeads))
        x01 = x[:,1]-x[:,0]
        x01 /= np.sqrt(np.sum(x01**2, axis=1))[:,None,:]
        x02 = x[:,2]-x[:,0]
        x02 /= np.sqrt(np.sum(x02**2, axis=1))[:,None,:]
        
        return np.arccos(np.sum(x01*x02, axis=1)).mean(axis=-1)

    def Dgfunc(self, q):
        """Calculate the Jacobian of the constraint function.
        """
        super(BondAngleConstraint, self).Dgfunc(q)
        ngp, ncart, nbeads = q.shape
        x = np.reshape(q, (ngp, 3, 3, nbeads)).copy()
        # 0-1
        x01 = x[:,1]-x[:,0]
        r1 = np.expand_dims(np.sqrt(np.sum(x01**2, axis=1)), axis=1)
        x01 /= r1
        # 0-2
        x02 = x[:,2]-x[:,0]
        r2 = np.expand_dims(np.sqrt(np.sum(x02**2, axis=1)), axis=1)
        x02 /= r2
        # jacobian
        ct = np.expand_dims(np.sum(x01*x02, axis=1), axis=1)
        st = np.sqrt(1.0-ct**2)
        x[:,1] = (ct*x01-x02)/(r1*st)
        x[:,2] = (ct*x02-x01)/(r2*st)
        x[:,0] = -(x[:,1]+x[:,2])
        return np.reshape(x, (ngp, ncart, nbeads))/nbeads
    
class EckartConstraint(ConstraintBase):
    """ Constraint class for MD specialized for enforcing the Eckart conditions
        (see E. Bright Wilson et al. 'Molecular Vibrations') 
        This is special, and is implemented and read differently 
        to the other constraints
    """

    def __init__(self, index_list, targetvals=None,
                 tolerance=1.0e-4, domain="cartesian", ngp=0):
        """In this case, 'targetvals' are the reference geometries.
        Actual target values for the Eckart conditions are zero.
        """
        
        self.tol = tolerance
        if (ngp == 0):
            raise ValueError(
    "The number of constrained groups must be specified for " +
    EckartConstraint.__class__.__name__)
        dself = dd(self)
        dself.ngp = depend_value(name="ngp", value=ngp)
        if targetvals is None:
            dself.qref = None
        else:
            dself.qref = depend_array(
                    name="qref",
                    value=np.asarray(targetvals).reshape((self.ngp,-1,3)))
        dself.targetvals = depend_array(
                name="targetvals",
                value=np.zeros((6,self.ngp)))
        dself.ilist = depend_array(
                name="ilist", 
                value=np.reshape(np.asarray(index_list), (self.ngp,-1)))
        dself.natoms = depend_value(name="natoms", value=self.ilist.shape[-1])
        counts = np.unique(dstrip(self.ilist), return_counts=True)[1]
        if np.any(counts != 1):
            raise ValueError(
"Constraint given overlapping groups of atoms.")
        dself.domain = domain.lower()
        if self.domain != "centroid":
            raise ValueError(
                "Using the 'EckartConstraint' not in the 'centroid' domain "+
                "may have unpredictable effects.")
        
    def bind(self, beads, nm):
        self.beads = beads
        self.nm = nm
        dself = dd(self)
        arr_shape = (self.nm.nbeads, self.ngp, 3*self.natoms)
        i3list = np.asarray([i for j in dstrip(self.ilist).flatten()
                               for i in range(3*j, 3*j+3)])
        # Configurations of the affected beads (later to be made dependent
        # on sections of arrays in grouped constraints)
        dself.qnm = depend_array(name="qnm", value = 
                np.transpose(np.reshape(
                dstrip(self.nm.qnm)[:,i3list], 
                arr_shape), [1,2,0]))
        dself.qc = depend_array(name="qc", value = 
                np.transpose(np.reshape(
                dstrip(self.beads.qc)[i3list], 
                (1,)+arr_shape[1:]), [1,2,0]))
        dself.qnmprev = depend_array(
                    name="qnmprev", value=dstrip(self.qnm).copy())
        dself.qcprev = depend_array(
                    name="qcprev", value=dstrip(self.qc).copy())
        if self.qref is None:
            # Use qcprev
            dd(self).qref = depend_array(
                    name="qref",
                    value=dstrip(self.qcprev).copy().reshape((self.ngp,-1,3)))
        dself.m3 = depend_array(
                    name="m3",
                    value=np.zeros_like(dstrip(self.qref)),
                    func=(lambda: np.reshape(
                          self.nm.dynm3[0, i3list], (self.ngp,-1,3))))
        # Total mass of the group of atoms
        dself.mtot = depend_array(name="mtot", value=np.zeros(self.ngp),
            func=(lambda: dstrip(self.m3)[:,:,0].sum(axis=-1)),
            dependencies=[dself.m3]
            )
        # Coords of reference centre of mass
        dself.qref_com = depend_array(
                name="qref_com", value=np.zeros((self.ngp,3)),
                func=(lambda: np.sum(
                      dstrip(self.qref)*dstrip(self.m3),
                      axis=1)/dstrip(self.mtot)[:,None]),
                dependencies=[dself.m3, dself.qref]
                )
        # qref in its centre of mass frame
        dself.qref_rel = depend_array(
                name="qref_rel", value=np.zeros_like(dstrip(self.qref)),
                func=(lambda: dstrip(self.qref)-dstrip(self.qref_com)[:,None,:]),
                dependencies=[dself.qref_com]
                )
        # qref in the CoM frame, mass-weighted
        dself.mqref_rel = depend_array(
                name="mqref_rel", value=np.zeros_like(dstrip(self.qref)),
                func=(lambda: 
                    dstrip(self.qref_rel)*dstrip(self.m3)),
                dependencies=[dself.qref_rel]
                )
        # Set up the constraint functions and its derivatives;
        # NOTE that there are *six* Eckart conditions
        dself.g = depend_array(
                name="g", value=np.zeros((6,self.ngp)),
                func=(lambda: self.gfunc(dstrip(self.qc))), 
                dependencies=[dself.qc, dself.qref, dself.m3])
        dself.Dg = depend_array(
                name="Dg", 
                value=np.zeros((6, self.ngp, 3*self.natoms, self.nm.nbeads)), 
                func=(lambda: self.Dgfunc(dstrip(self.qcprev))),
                dependencies=[dself.qcprev, dself.qref, dself.m3])
        
    def gfunc(self, q):
        """
        Calculates the constraint.
        """

        q = np.reshape(q, (self.ngp,-1,3))
        qref = dstrip(self.qref)
        mqref_rel = dstrip(self.mqref_rel)
        m = dstrip(self.m3)
        M = dstrip(self.mtot).reshape((self.ngp,1))
        g = np.zeros((6, self.ngp))
        Delta = q-qref
        g[:3] = np.transpose(np.sum(m*Delta, axis=1)/M)
        g[3:] = np.transpose( np.sum ( np.cross(
                mqref_rel, Delta, axis=-1), axis=1)/M)
        return g

    def Dgfunc(self, q):
        """
        Calculates the Jacobian of the constraint.
        """
        
        q = np.reshape(q, (self.ngp,-1,3))
        Dg = np.zeros((6,)+q.shape) # Gradient w.r.t qc
        m = dstrip(self.m3)
        M = dstrip(self.mtot).reshape((self.ngp,1,1))
        mqref_rel = dstrip(self.mqref_rel)
        for i in range(3):
            Dg[i,:,:,i] = m[:,:,i]
        # Eckart rotation, x-component
        Dg[3,:,:,1] =-mqref_rel[:,:,2]
        Dg[3,:,:,2] = mqref_rel[:,:,1]
        # Eckart rotation, y-component
        Dg[4,:,:,0] = mqref_rel[:,:,2]
        Dg[4,:,:,2] =-mqref_rel[:,:,0]
        # Eckart rotation, z-component
        Dg[5,:,:,0] =-mqref_rel[:,:,1]
        Dg[5,:,:,1] = mqref_rel[:,:,0]
        Dg /= M
        ans = np.zeros((6, self.ngp, 3*self.natoms, self.nm.nbeads))
        Dg.shape = (6, self.ngp, 3*self.natoms)
        ans[...,0] = np.sqrt(1.0*self.nm.nbeads)*Dg
        return ans

class GroupedConstraints(dobject):
    """Describes a set of k constraint functions that are applied to 
    ngp non-overlapping groups of atoms.
    """
    
    def __init__(self, constraint_list, maxit=100, qnmprev=None,
                 eckart=False, tolerance=1.0e-6, qref=None):
        """Initialise the set of grouped constraints
        
        Args:
            constraint_list: list of objects derived from ConstraintBase
            maxit: maximum numbed of iterations to converge a single step
            qnmprev: normal-mode configuration at the end of the previous
                converged constrained propagation step, flattened from
                shape=(ngp, n3unique, nbeads)
            eckart: indicate whether the Eckart constraint is to be imposed
                   onto the individual groups
            tolerance: convergence criterion for Eckart conditions
            qref: "reference" configuration for Eckart conditions
        """

        self.clist = constraint_list
        self.ncons = len(self.clist)
        self.maxit = maxit
        self.tol = np.asarray([c.tol for c in self.clist])
        self.eckart = eckart
        self.ecktol = tolerance
        dself = dd(self)
        msg = "Non-conforming list of constraints given to GroupedConstraints."
        # Check that the number of atom groups agrees in each constraint
        ngps = []
        for c in self.clist:
            ngps.append(c.ngp)
        ngps = np.asarray(ngps)
        if np.all(ngps == ngps[0]):
            dself.ngp = depend_value(name="ngp", value=ngps[0])
            for c in self.clist:
                dpipe(dself.ngp, dd(c).ngp)
        else:
            raise ValueError(msg)
        # Collate the list of all constraint indices.
        self.mk_idmaps()
        self.qnmprev = qnmprev
        self.qref = qref
            
    def bind(self, beads, nm):
        self.beads = beads
        self.nm = nm
        for c in self.clist:
            c.bind(beads, nm)
        dself = dd(self)
        arr_shape = (self.nm.nbeads, self.ngp, self.n3unique)
        #-------- Set up copies of the affected phase-space -----------#
        #------------- coordinates and relevant masses ----------------#
        
        dself.dynm3 = depend_array(
                name="dynm3", 
                value=np.zeros((self.ngp, self.n3unique, self.nm.nbeads)),
                func=(lambda: np.transpose(np.reshape(
                        dstrip(self.nm.dynm3)[:,self.i3unique.flatten()],
                        arr_shape), [1,2,0])),
                dependencies=[dd(self.nm).dynm3])
        # Holds all of the atoms affected by this list of constraints
        dself.qnm = depend_array(
                name="qnm", 
                value = np.zeros((self.ngp, self.n3unique, self.nm.nbeads)), 
                func = (lambda: np.transpose(np.reshape(
                        dstrip(self.nm.qnm)[:,self.i3unique.flatten()],
                        arr_shape), [1,2,0])),
                dependencies = [dd(self.nm).qnm])
        dself.pnm = depend_array(
                name="pnm", 
                value = np.zeros((self.ngp, self.n3unique, self.nm.nbeads)), 
                func = (lambda: np.transpose(np.reshape(
                        dstrip(self.nm.pnm)[:,self.i3unique.flatten()],
                        arr_shape), [1,2,0])),
                dependencies = [dd(self.nm).pnm])
        if self.qnmprev is None:
            dself.qnmprev = depend_array(
                    name="qnmprev", value=dstrip(self.qnm).copy())
        else:
            try:
                dself.qnmprev = depend_array(
                    name="qnmprev", 
                    value=np.reshape(self.qnmprev.copy(), (self.ngp, self.n3unique, self.nbeads)))
            except:
                raise ValueError(
"Shape of previous converged configuration supplied at initialisation\n"+
"is inconsistent with the bound system: {:s} \= {:s}.".format(
                self.qnmprev.shape.__repr__(),
                self.qnm.shape.__repr__()))
        #--------- Set up Cartesian and centroid coordinates ----------#
        # TODO: in future check for open paths
        self.nmtrans = nm_fft(self.qnm.shape[2], np.prod(self.qnm.shape[:2])//3)
        dself.q = depend_array(
                name="q",
                value = np.zeros((self.ngp, self.n3unique, self.nm.nbeads)), 
                func = (lambda: np.transpose(np.reshape(
                        dstrip(self.beads.q)[:,self.i3unique.flatten()],
                        arr_shape), [1,2,0])),
                dependencies = [dd(self.beads).q])
        dself.qprev = depend_array(
                name="qprev", value=np.zeros_like(dstrip(self.qnmprev)),
                func=(lambda: self._to_beads(dstrip(self.qnmprev))), 
                dependencies=[dself.qnmprev])
        dself.qc = depend_array(
                name="qc", 
                value=np.transpose(np.reshape(
                    dstrip(self.beads.qc)[self.i3unique.flatten()], 
                    (1,)+arr_shape[1:]), [1,2,0]))
        dself.qcprev = depend_array(
                name="qcprev", value=np.zeros_like(dstrip(self.qc)),
                func=(lambda: np.expand_dims(
                        dstrip(self.qnmprev)[...,0] /
                        np.sqrt(1.0*self.beads.nbeads), axis=-1)), 
                dependencies=[dself.qnmprev])
        #------- Make the coordinates of the individual constraints ---------#
        #----------- depend on the coordinates in this object ---------------#
        def make_arrgetter(k, arr):
            return lambda: dstrip(arr)[:,self.i3list[k]]
        for k,c in enumerate(self.clist):
            if c.domain == "cartesian":
                dd(c).q.add_dependency(dself.q)
                dd(c).q._func = make_arrgetter(k, self.q)
                dd(c).qprev.add_dependency(dself.qprev)
                dd(c).qprev._func = make_arrgetter(k, self.qprev)
            elif c.domain == "normalmode":
                dd(c).qnm.add_dependency(dself.qnm)
                dd(c).qnm._func = make_arrgetter(k, self.qnm)
                dd(c).qnmprev.add_dependency(dself.qnmprev)
                dd(c).qnmprev._func = make_arrgetter(k, self.qnmprev)
            else:
                dd(c).qc.add_dependency(dself.qc)
                dd(c).qc._func = make_arrgetter(k, self.qc)
                dd(c).qcprev.add_dependency(dself.qcprev)
                dd(c).qcprev._func = make_arrgetter(k, self.qcprev)
        # Target values
        targetvals = []
        for c in self.clist:
            targetvals.append(dstrip(c.targetvals).copy())
        dself.targetvals = depend_array(
                name="targetvals", 
                value=np.column_stack(targetvals))
        def make_targetgetter(k):
            return lambda: dstrip(self.targetvals[:,k])
        for k, c in enumerate(self.clist):
            dd(c).targetvals.add_dependency(dself.targetvals)
            dd(c).targetvals._func = make_targetgetter(k)
        # Values of the constraint function (no Eckart)
        dself.g = depend_array(
                name="g", value=np.zeros((self.ngp, self.ncons)),
                func=self.gfunc, dependencies=[dd(c).g for c in self.clist])
        # Jacobian of the constraint function (with Eckart)
        dself.Dg = depend_array(
                name="Dg", 
                value=np.zeros((self.ngp, self.ncons, 
                                self.n3unique, self.nm.nbeads)), 
                func=self.Dgfunc, dependencies=[dd(c).Dg for c in self.clist])
        # The Cholesky decomposition of the Gramian matrix
        dself.GramChol = depend_array(
                name="GramChol", 
                value=np.zeros((self.ngp, self.ncons, self.ncons)),
                func=self.GCfunc, dependencies=[dself.Dg])
        
            
    def mk_idmaps(self):
        """Construct lookup dictionary and lists to quickly access the portions
        of arrays that are affected by the constraints
        """
        
        ilist = []
        for c in self.clist:
            ilist.append(dstrip(c.ilist))
        # Store a list of the unique atom indices for each group
        iunique = np.unique(np.hstack(ilist), axis=-1)
        counts = np.unique(iunique, return_counts=True)[1]
        if np.any(counts != 1):
            raise ValueError(
"GroupedConstraints given overlapping groups of atoms.")
        # List of unique indices
        self.i3unique = np.zeros((iunique.shape[0],
                                  iunique.shape[1]*3), dtype=int)
        self.n3unique = self.i3unique.shape[1]
        for i, i3 in zip(iunique, self.i3unique):
            i3[:] = np.asarray([ 3*k + j for k in i for j in range(3)])
        # List of constraint-specific indices
        i3list = []
        for lst in ilist:
            i3 = (3*np.ones(lst.shape+(3,),dtype=int) *
                  lst[:,:,None]) + np.arange(3)
            i3.shape = (len(lst),-1)
            i3list.append(i3)
        self.i3list = []
        for k in range(len(self.clist)):
            inv_idx_lst = []
            for ref,lst in zip(self.i3unique, i3list[k]):
                inv_idx = []
                for idx in lst:
                    inv_idx.append(np.argwhere(ref==idx).item())
                inv_idx_lst.append(inv_idx)
            inv_idx_lst = np.asarray(inv_idx_lst, dtype=int)
            if not np.all(inv_idx_lst == inv_idx_lst[0]):
                raise ValueError(
"Constrained atoms in GroupedConstraints are misaligned.")
            self.i3list.append(inv_idx_lst[0].flatten())

    def _to_beads(self, arr):
        """
        Convert the array contents to normal mode coordinates.
        """
        # (ngp, n3unique, nbeads) <-> (nbeads, ngp, n3unique)
        wkspace = np.reshape(np.transpose(
                arr, [2,0,1]), (self.nm.nbeads, -1))
        return np.transpose(np.reshape(
               self.nmtrans.nm2b(wkspace), 
               (self.nm.nbeads, self.ngp, self.n3unique)),
               [1,2,0])

    def _to_nm(self, arr):
        """
        Convert array to Cartesian coordinates.
        """
        
        wkspace = np.reshape(np.transpose(
                arr, [2,0,1]), (self.nm.nbeads, -1))
        return np.transpose(np.reshape(
               self.nmtrans.b2nm(wkspace), 
               (self.nm.nbeads, self.ngp, self.n3unique)),
               [1,2,0])

    def gfunc(self):
        """Return the value of each of the constraints for each of the
        atoms groups. The result has shape (ngp,ncons)
        """
        
        g = []
        for c in self.clist:
            if isinstance(c, EckartConstraint):
                g += list(dstrip(c.g))
            else:
                g.append(dstrip(c.g))
        return np.column_stack(g)

    def Dgfunc(self):
        """Return the Jacobian of each of the constraints for each of the
        atoms groups. The result has shape (ngp,ncons,ndim*natoms,nbeads)
        """

        ans = np.zeros((self.ncons,)+self.qnmprev.shape)
        arr = np.zeros(self.qnmprev.shape) # wkspace for nm conversion
        k = 0
        for c,i3 in zip(self.clist,self.i3list):
            if c.domain == "cartesian":
                arr[:,i3] = dstrip(c.Dg)
                ans[k,:] = self._to_nm(arr)
                k += 1
            elif isinstance(c, EckartConstraint):
                ans[k:k+6,:,i3] = dstrip(c.Dg)
                k += 6
            else:
                ans[k,:,i3] = dstrip(c.Dg)
                k += 1
        return np.transpose(ans, axes=[1,0,2,3]) 

    def GCfunc(self):
        """Return the Cholesky decomposition of the Gramian matrix
        for each of the groups of atoms.
        """

        Dg = dstrip(self.Dg)
        Dgm = Dg / dstrip(self.dynm3)[:,None,...]
        Dg = np.reshape(Dg, (self.ngp, self.ncons, -1))
        Dgm.shape = Dg.shape
        # (ngp, ncons, n)*(ngp, n, ncons) -> (ngp, ncons, ncons)
        gram = np.matmul(Dg, np.transpose(Dgm, [0, 2, 1]))
        return np.linalg.cholesky(gram)

    def norm(self, x):
        """Return the norm of the deviations from the targetvalues
        for an input of shape (ngp, k).
        """
        ans = np.empty_like(x)
        k = 0
        for c in self.clist:
            if isinstance(c, EckartConstraint):
                for i in range(6):
                    ans[:,k] = c.norm(x[:,k])
                    k += 1
            else:
                ans[:,k] = c.norm(x[:,k])
                k += 1
                
        return ans    

class ConstraintSolverBase(dobject):

    def __init__(self, constraint_groups, dt=1.0):
        self.constraint_groups = constraint_groups
        dd(self).dt = depend_value(name="dt", value=dt)

    def proj_cotangent(self):
        raise NotImplementedError()

    def proj_manifold(self):
        raise NotImplementedError()  
        
class ConstraintSolver(ConstraintSolverBase):

    def __init__(self, constraint_groups, dt=1.0):
        super(ConstraintSolver,self).__init__(constraint_groups, dt)
        
    def bind(self, nm, dt=1.0):
        self.nm = nm

    def proj_cotangent(self):
        """Projects onto the cotangent space of the constraint manifold.
        """
        pnm = dstrip(self.nm.pnm).copy()
        for cgp in self.constraint_groups:
            dynm3 = dstrip(cgp.dynm3)
            p = dstrip(cgp.pnm)
            v = np.reshape(p/dynm3, (cgp.ngp, -1, 1))
            Dg = np.reshape(dstrip(cgp.Dg), (cgp.ngp, cgp.ncons, -1))
            b = np.matmul(Dg, v)
            GramChol = dstrip(cgp.GramChol)
            x = np.linalg.solve(np.transpose(GramChol, [0,2,1]),
                                np.linalg.solve(GramChol, b))
            pnm[:,cgp.i3unique.flatten()] -= np.reshape(
                    np.matmul(np.transpose(Dg, [0,2,1]), x),
                    (cgp.ngp*cgp.n3unique, self.nm.nbeads)).T
        self.nm.pnm[:] = pnm

    def proj_manifold(self):
        """Projects onto the constraint manifold using the Gram matrix
        defined by self.Dg and self.Gram
        """
        
        pnm = dstrip(self.nm.pnm).copy()
        qnm = dstrip(self.nm.qnm).copy()
        for cgp in self.constraint_groups:
            icycle = 0
            active = np.ones(cgp.ngp, dtype=bool)
            g = np.empty((cgp.ngp, cgp.ncons, 1))
            Dg = np.transpose(np.reshape(dstrip(cgp.Dg), 
                                         (cgp.ngp, cgp.ncons,-1)), [0,2,1])
            GramChol = dstrip(cgp.GramChol)
            dynm3 = dstrip(cgp.dynm3)
            # Fetch current normal-mode coordinates and temporarily
            # suspend automatic updates 
            cgp.qnm.update_auto()
            qfunc, cgp.qnm._func = cgp.qnm._func, None
            cgp.pnm.update_auto()
            pfunc, cgp.pnm._func = cgp.pnm._func, None
            while (icycle < cgp.maxit):
                g[active,:,0] = (dstrip(cgp.g)[active] - 
                                 dstrip(cgp.targetvals)[active])
                active = np.any(cgp.norm(g[...,0]) > cgp.tol, axis=-1)
                if not np.any(active):
                    break
                gc = GramChol[active]
                dlambda = np.linalg.solve(
                        np.transpose(gc, [0,2,1]),
                        np.linalg.solve(gc, g[active]))
                delta = np.reshape(np.matmul(Dg[active], dlambda),
                                   (-1, cgp.n3unique, self.nm.nbeads))
                cgp.qnm[active] -= (delta / dynm3[active])
                cgp.pnm[active] -= delta/self.dt
                icycle += 1
                if (icycle == cgp.maxit):
                    raise ValueError('No convergence in Newton iteration '+
                                     'for positional component')
            cgp.qnmprev[:] = dstrip(cgp.qnm)
            qnm[:,cgp.i3unique.flatten()] = np.reshape(
                    dstrip(cgp.qnm), (-1, self.nm.nbeads)).T
            pnm[:,cgp.i3unique.flatten()] = np.reshape(
                    dstrip(cgp.pnm), (-1, self.nm.nbeads)).T
            # Restore automatic updates
            cgp.qnm._func = qfunc
            cgp.pnm._func = pfunc
        self.nm.pnm[:] = pnm
        self.nm.qnm[:] = qnm

class NVEConstrainedIntegrator(NVEIntegrator):
    """Integrator object for constant energy simulations of constrained
    systems.

    Has the relevant conserved quantity and normal mode propagator for the
    constant energy ensemble. Note that a temperature of some kind must be
    defined so that the spring potential can be calculated.

    Attributes:

    Depend objects:
        econs: Conserved energy quantity. Depends on the bead kinetic and
            potential energy, and the spring potential energy.
    """
    
    def get_gdt(self):
        """Geodesic flow timestep
        """
        return self.dt * 0.5 / self.inmts / self.nsteps_geo
    
    def pconstraints(self):
        """This removes the centre of mass contribution to the kinetic energy
        and projects the momenta onto the contangent space of the constraint
        manifold (implicitly assuming that the two operations commute)

        Calculates the centre of mass momenta, then removes the mass weighted
        contribution from each atom. If the ensemble defines a thermostat, then
        the contribution to the conserved quantity due to this subtraction is
        added to the thermostat heat energy, as it is assumed that the centre of
        mass motion is due to the thermostat.

        If there is a choice of thermostats, the thermostat
        connected to the centroid is chosen.
        """
        self.csolver.proj_cotangent()
        super(NVEConstrainedIntegrator, self).pconstraints()

    def bind(self, motion):
        """ Reference all the variables for simpler access."""

        dself = dd(self)
        dmotion = dd(motion)
        dself.nsteps_geo = dmotion.nsteps_geo
        
        super(NVEConstrainedIntegrator,self).bind(motion)
        self.csolver = motion.csolver
        dself.gdt = depend_value(name="gdt", func=self.get_gdt,
                                 dependencies=[dself.dt, dself.nmts])
        dpipe(dself.gdt, dd(self.csolver).dt)
        
    def free_p(self):
        """Velocity Verlet momentum propagator with ring-polymer spring forces,
           followed by projection onto the cotangent space of the constraint.
        """
        self.nm.pnm += dstrip(self.nm.fspringnm)*self.qdt
        self.pconstraints()
        
    def step_A(self):
        """Unconstrained A-step"""
        self.nm.qnm += dstrip(self.nm.pnm)/dstrip(self.nm.dynm3)*self.gdt
        
    def step_Ag(self):
        """Geodesic flow
        """
        for i in range(self.nsteps_geo):
            self.step_A()
            self.csolver.proj_manifold()
            self.csolver.proj_cotangent()
        
    def free_qstep_ba(self):
        """This overrides the exact free-ring-polymer propagator, performing 
        half of standard velocity Verlet with explicit spring forces. This is 
        done to retain the symplectic property of the constrained propagator
        """
        self.free_p()
        self.step_Ag()
        
    def free_qstep_ab(self):
        """This overrides the exact free-ring-polymer propagator, performing 
        half of standard velocity Verlet with explicit spring forces. This is 
        done to retain the symplectic property of the constrained propagator
        """
        self.step_Ag()
        self.free_p()

class NVTConstrainedIntegrator(NVEConstrainedIntegrator):

    """Integrator object for constant temperature simulations of constrained
    systems.

    Has the relevant conserved quantity and normal mode propagator for the
    constant temperature ensemble. Contains a thermostat object containing the
    algorithms to keep the temperature constant.

    Attributes:
        thermostat: A thermostat object to keep the temperature constant.
    """
    
    def tstep(self):
        """Geodesic integrator thermostat step.
        """
        
        self.thermostat.step()
        # Fix momenta and correct eens accordingly
        sm = np.sqrt(dstrip(self.nm.dynm3))
        p = (dstrip(self.nm.pnm)/sm).flatten()
        self.ensemble.eens += np.dot(p,p) * 0.5
        self.csolver.proj_cotangent()
        p = (dstrip(self.nm.pnm)/sm).flatten()
        self.ensemble.eens -= np.dot(p,p) * 0.5
        # CoM constraints include own correction to eens
        super(NVEConstrainedIntegrator, self).pconstraints()

    def step(self, step=None):
        """Does one simulation time step."""

        if self.splitting == "obabo":
            # thermostat is applied for dt/2
            self.tstep()
            # forces are integerated for dt with MTS.
            self.mtsprop(0)
            # thermostat is applied for dt/2
            self.tstep()

        elif self.splitting == "baoab":

            self.mtsprop_ba(0)
            # thermostat is applied for dt
            self.tstep()
            self.mtsprop_ab(0)

    
#class EckartConstraint(ConstraintBase):
#    """ Constraint class for MD specialized for enforcing the Eckart conditions
#        (see E. Bright Wilson et al. 'Molecular Vibrations') 
#        Unlike the constraints above, a single instance of this class can only
#        describe one set of Eckart condition.
#    """
#
#    def __init__(self,constrained_indices,constraint_values):
#
#        super(EckartConstraint,self).__init__(
#                constrained_indices, np.zeros(0,float), ncons=6)
#        self.constrained_indices.shape = -1
#        # Check that there are no repeats
#        if np.any(self.constrained_indices != self.i_unique):
#            raise ValueError("repeated atom indices in EckartConstraint")
#        self.i3_indirect.shape = (-1, 3)
#        if len(constraint_values) == 0:
#            self._calc_cons = True
#            dd(self).qref = depend_array(
#                    name="qref", value=np.zeros_like(self.i3_indirect, float)
#                    )
#        else:
#            self._calc_cons = False
#            dd(self).qref = depend_array(
#                    name="qref", 
#                    value=np.reshape(constraint_values, 
#                                     self.i3_indirect.shape).copy()
#                    )
#        
#    def bind(self, beads):
#        
#        super(EckartConstraint, self).bind(beads)
#        if self._calc_cons:
#            self.qref[:] = dstrip(beads.q[0])[self.i3_unique].reshape((-1,3))
#        dself = dd(self)
#        # Total mass of the group of atoms
#        dself.mtot = depend_value(name="mtot", value=1.0, 
#            func=(lambda: dstrip(self.m3)[::3].sum()),
#            dependencies=[dself.m3]
#            )
#        # Coords of reference centre of mass
#        dself.qref_com = depend_array(
#                name="qref_com", value=np.zeros(3, float),
#                func=(lambda: np.sum(
#                      dstrip(self.qref)*dstrip(self.m3).reshape((-1,3)),
#                      axis=0)/self.mtot),
#                dependencies=[dself.m3, dself.qref]
#                )
#        # qref in its centre of mass frame
#        dself.qref_rel = depend_array(
#                name="qref_rel", value=np.zeros_like(dstrip(self.qref)),
#                func=(lambda: dstrip(self.qref)-dstrip(self.qref_com)),
#                dependencies=[dself.qref, dself.qref_com]
#                )
#        # qref in the CoM frame, mass-weighted
#        dself.mqref_rel = depend_array(
#                name="mqref_rel", value=np.zeros_like(dstrip(self.qref)),
#                func=(lambda: 
#                    dstrip(self.qref_rel)*dstrip(self.m3).reshape((-1,3))),
#                dependencies=[dself.qref_rel, dself.m3]
#                )
#        # Make constraint function and gradient depend on the parameters
#        dself.g.add_dependency(dself.qref)
#        dself.g.add_dependency(dself.m3)
#        dself.Dg.add_dependency(dself.qref)
#        dself.Dg.add_dependency(dself.m3)
#        
#    def gfunc(self):
#        """
#        Calculates the constraint.
#        """
#
#        q = dstrip(self.q).reshape((-1,3))
#        m = dstrip(self.m3).reshape((-1,3))
#        qref = dstrip(self.qref)
#        r = np.zeros(self.ncons)
#        Delta = q-qref
#        r[:3] = np.sum(m*Delta, axis=0)/self.mtot
#        r[3:] = np.sum(np.cross(dstrip(self.mqref_rel), Delta), axis=0)/self.mtot
#        return r
#
#    def Dgfunc(self, reduced=False):
#        """
#        Calculates the Jacobian of the constraint.
#        """
#
#        q = dstrip(self.qprev)
#        r = np.zeros((self.ncons, self.n_unique, 3))
#        m = dstrip(self.m3).reshape((-1,3))
#        mqref_rel = dstrip(self.mqref_rel)
#        for i in range(3):
#            r[i,:,i] = m[:,i]
#        # Eckart rotation, x-component
#        r[3,:,1] =-mqref_rel[:,2]
#        r[3,:,2] = mqref_rel[:,1]
#        # Eckart rotation, y-component
#        r[4,:,0] = mqref_rel[:,2]
#        r[4,:,2] =-mqref_rel[:,0]
#        # Eckart rotation, z-component
#        r[5,:,0] =-mqref_rel[:,1]
#        r[5,:,1] = mqref_rel[:,0]
#        r /= self.mtot
#        r.shape = (self.ncons,-1)
#        return r