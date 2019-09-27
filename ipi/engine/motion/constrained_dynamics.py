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

from ipi.utils.nmtransform import nm_fft
from ipi.engine.motion import Dynamics
from ipi.engine.motion.dynamics import DummyIntegrator, NVEIntegrator
from ipi.utils.depend import depend_value, depend_array, \
                             dd, dobject, dstrip, dpipe
from ipi.engine.thermostats import Thermostat
from ipi.engine.barostats import Barostat


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
                thermostat=None, barostat=None, fixcom=False, fixatoms=None,
                nmts=None, nsteps_geo=1, constraint_list=[]):

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
        self.enstype = mode
        if nmts is None or len(nmts) == 0:
            dd(self).nmts = depend_array(name="nmts", value=np.asarray([1], int))
        else:
            dd(self).nmts = depend_array(name="nmts", value=np.asarray(nmts, int))
        if self.enstype == "nve":
            self.integrator = NVEConstrainedIntegrator()
#            elif self.enstype == "nvt":
#                self.integrator = NVTIntegrator_constraint()
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
        if constraint_list is not None:
            self.constraint_list = constraint_list
        self.csolver = ConstraintSolver(self.constraint_list)
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
        super(ConstrainedDynamics, self).bind(ens, beads, nm, cell, 
                                              bforce, prng, omaker)
        # now binds the constraints
        for cgp in self.constraint_list:
            cgp.bind(nm)
        self.csolver.bind(nm)
        
        
class ConstraintBase(object):
    """Base constraint class; defines the constraint function and its Jacobian.
    """
    
    def __init__(self, tolerance=1.0e-4, domain="cartesian"):
        """Initialise the constraint.

        Args:
            domain: ['cartesian'/'normalmode'/'centroid'] - specifies whether 
            the constraint is expressed in terms of Cartesian, normalmode 
            or centroid coordinates.
        """
        self.tol = tolerance
        self.domain = domain.lower()
        if self.domain not in ["cartesian", "normalmode", "centroid"]:
            raise ValueError("Unknown constraint domain '{:s}'.".format(domain))

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
    def __init__(self, tolerance=1.0e-4, domain="cartesian"):
        super(BondLengthConstraint, self).__init__(tolerance, domain)
        if self.domain == "normalmode":
            warnings.warn(
                "Using the 'BondLength' constraint in the 'normalmode' domain "+
                "may have unpredictable effects.")

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
    
    def __init__(self, tolerance=1.0e-4, domain="cartesian"):
        super(BondAngleConstraint, self).__init__(tolerance, domain)
        if self.domain == "normalmode":
            warnings.warn(
                "Using the 'BondAngle' constraint in the 'normalmode' domain "+
                "may have unpredictable effects.")

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

class GroupedConstraints(dobject):
    """Describes a set of k constraint functions that are applied to 
    ngp non-overlapping groups of atoms.
    """
    
    def __init__(self, constraint_list, index_list, maxit=100, 
                 targetvals=None, qnmprev=None):
        """Initialise the set of grouped constraints
        
        Args:
            constraint_list: list of objects derived from ConstraintBase
            index_list: list of indices of the atoms affected by the constraints
            maxit: maximum numbed of iterations to converge a single step
            targetvals: list of target values for the constraint functions
            qnmprev: normal-mode configuration at the end of the previous
                converged constrained propagation step, 
                shape=(ngp, n3unique, nbeads)
            
        index_list storage convention:
            [
              [                               \
                [ i(c1, n1), i(c1, n2), ...],  |
                [ i(c2, n1), i(c2, n2), ...],  |- constraints for group 1
                ...                            |
              ]                               /
              [
                ...
              ]
              ...
            ]
        where i(cj,nk) is the index of the k-th atom affected by the j-th 
        constraint function and len(index_list) is the number of atom groups.
        Similarly targetvals are to be supplied as
        [ [ c1, c2, ... ], [ c1, c2, ... ], ...]
          ^ group-1        ^group-2
          
        E.g. for a box of water molecules with OHH storage convention and
        OH1, OH2, HOH bond, bond, angle constraints the index list becomes
        [
          [[0, 1], [0, 2], [0, 1, 2]],
          [[3, 4], [3, 5], [3, 4, 5]],
          ...
        ]
        """

        self.clist = constraint_list
        self.ncons = len(self.clist) # number of constraint functions
        self.maxit = maxit
        self.tol = np.asarray([c.tol for c in self.clist])
        ilist = np.asarray(index_list)
        msg = "Non-conforming list of indices given to GroupedConstraints."
        self.ngp = len(ilist) # number of groups of atoms
        # Check that the number of constraint functions agrees
        try:
            k = ilist.shape[1]
        except IndexError:
            raise ValueError(msg)
        except:
            raise
        if k != self.ncons:
            raise ValueError(msg)
        self.ilist = []
        for i in range(self.ncons):
            # Check the number of atom indices agrees for each constraint
            try:
                self.ilist.append(np.vstack(ilist[:,i]))
            except ValueError:
                raise ValueError(msg)
            except:
                raise
        # Store a list of the unique atom indices for each group
        self.iunique = np.asarray([np.unique(np.hstack(lst)) for lst in ilist])
        # Check that the groups do not overlap
        counts = np.unique(self.iunique, return_counts=True)[1]
        if np.any(counts != 1):
            raise ValueError(
"GroupedConstraints given overlapping groups of atoms.")
        self.mk_idmaps()
        if targetvals is None:
            self.targetvals = None
        else:
            try:
                self.targetvals = np.reshape(
                        np.asarray(targetvals, dtype=float).flatten(),
                        (self.ngp, self.ncons), order="C")
            except:
                raise ValueError(
"Non-conforming list of target values given to GroupedConstraints.")
        self._fetch_cart = np.any(fxn.domain=="cartesian" for fxn in self.clist)
        if qnmprev is None:
            self.qnmprev = None
        else:
            dd(self).qnmprev = depend_array(
                name="qnmprev", value=np.asarray(qnmprev).copy())
            

    def bind(self, nm):
        self.nm = nm
        dself = dd(self)
        arr_shape = (self.nm.nbeads, self.ngp, self.n3unique)
        dself.dynm3 = depend_array(
                name="dynm3", value=np.zeros((self.ngp, self.n3unique, self.nm.nbeads)),
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
        # Holds the configuration obtained at the end of the previous converged
        # constrained propagation step
        if self.qnmprev is None:
            dself.qnmprev = depend_array(
                    name="qnmprev", value=dstrip(self.qnm).copy())
        else:
            if (self.qnmprev.shape != 
                    (self.ngp, self.n3unique, self.nbeads)):
                raise ValueError(
"Shape of previous converged configuration supplied at initialisation\n"+
"is inconsistent with the bound system: {:s} \= {:s}.".format(
    self.qnmprev.shape.__repr__(),
    self.qnm.shape.__repr__()))

        # Values of the constraint function
        dself.g = depend_array(
                name="g", value=np.zeros((self.ngp, self.ncons)),
                func=self.gfunc, dependencies=[dself.qnm])
        # Jacobian of the constraint function
        dself.Dg = depend_array(
                name="Dg", 
                value=np.zeros((self.ngp, self.ncons, 
                                self.n3unique, self.nm.nbeads)), 
                func=self.Dgfunc, dependencies=[dself.qnmprev])
        # The Cholesky decomposition of the Gramian matrix
        dself.GramChol = depend_array(
                name="GramChol", 
                value=np.zeros((self.ngp, self.ncons, self.ncons)),
                func=self.GCfunc, dependencies=[dself.Dg])
        # TODO: in future check for open paths
        self.nmtrans = nm_fft(self.qnm.shape[2], np.prod(self.qnm.shape[:2])//3)
        dself.q = depend_array(
                name="q", value=np.zeros_like(dstrip(self.qnm)),
                func=(lambda: self._to_beads(dstrip(self.qnm))), 
                dependencies=[dself.qnm])
        dself.qprev = depend_array(
                name="qprev", value=np.zeros_like(dstrip(self.qnmprev)),
                func=(lambda: self._to_beads(dstrip(self.qnmprev))), 
                dependencies=[dself.qnmprev])
        if self.targetvals is None:
            self.targetvals = self.gfunc()
            
    def mk_idmaps(self):
        """Construct lookup dictionary and lists to quickly access the portions
        of arrays that are affected by the constraints
        """
        
        # List of unique indices
        self.i3unique = np.zeros((self.iunique.shape[0],
                                  self.iunique.shape[1]*3), dtype=int)
        self.n3unique = self.i3unique.shape[1]
        for iunique, i3unique in zip(self.iunique, self.i3unique):
            i3unique[:] = np.asarray(
                    [ 3*i + j for i in iunique for j in range(3)])
        # List of constraint-specific indices
        i3list = []
        for ilist in self.ilist:
            i3 = (3*np.ones(ilist.shape+(3,),dtype=int) *
                  ilist[:,:,None]) + np.arange(3)
            i3.shape = (len(ilist),-1)
            i3list.append(i3)
        self.i3list = []
        # Construct the mapping between the array of all DoFs affected by the
        # set of constraints and the argument of each individual constraint 
        for k in range(self.ncons):
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
        atoms groups. The result has shape (ngp,k)
        """
        
        ans = np.zeros((self.ncons, self.ngp))
        qnm = dstrip(self.qnm)
        if self._fetch_cart:
            q = dstrip(self.q) 
        for fxn, i3, arr in zip(self.clist, self.i3list, ans):
            if fxn.domain=="normalmode":
                arr[:] = fxn.gfunc(qnm[:,i3])
            elif fxn.domain=="centroid":
                arr[:] = fxn.gfunc(qnm[:,i3,:1])
            else:
                arr[:] = fxn.gfunc(q[:,i3])
        return ans.T

    def Dgfunc(self):
        """Return the Jacobian of each of the constraints for each of the
        atoms groups. The result has shape (ngp,ncons,ndim*natoms,nbeads)
        """

        qnmprev = dstrip(self.qnmprev)
        ans = np.zeros((self.ncons,)+qnmprev.shape)
        if self._fetch_cart:
            qprev = dstrip(self.qprev) 
        for fxn, i3, arr in zip(self.clist, self.i3list, ans):
            if fxn.domain=="normalmode":
                arr[:,i3] = fxn.Dgfunc(qnmprev[:,i3])
            elif fxn.domain=="centroid":
                arr[:,i3,:1] = fxn.Dgfunc(qnmprev[:,i3,:1])
            else:
                arr[:,i3] = fxn.Dgfunc(qprev[:,i3])
                arr[:] = self._to_nm(arr)
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
        for k in range(self.ncons):
            ans[:,k] = self.clist[k].norm(x[:,k])
        return ans    

class ConstraintSolverBase(dobject):

    def __init__(self, constraint_groups):
        self.constraint_groups = constraint_groups

    def proj_cotangent(self):
        raise NotImplementedError()

    def proj_manifold(self):
        raise NotImplementedError()  
        
class ConstraintSolver(ConstraintSolverBase):

    def __init__(self, constraint_groups):
        super(ConstraintSolver,self).__init__(constraint_groups)
        
    def bind(self, nm, dt=1.0):
        self.nm = nm
        dd(self).dt = depend_value(name="dt", value=dt)

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
        super(NVEConstrainedIntegrator).pconstraints()

    def bind(self, motion):
        """ Reference all the variables for simpler access."""

        dself = dd(self)
        dmotion = dd(motion)
        dself.nsteps_geo = dmotion.nsteps_geo
        
        super(NVEConstrainedIntegrator,self).bind(motion)
        self.constraint_list = motion.constraint_list
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

#class NVTIntegrator_constraint(NVEIntegrator_constraint):
#
#    """Integrator object for constant temperature simulations.
#
#    Has the relevant conserved quantity and normal mode propagator for the
#    constant temperature ensemble. Contains a thermostat object containing the
#    algorithms to keep the temperature constant.
#
#    Attributes:
#        thermostat: A thermostat object to keep the temperature constant.
#    """
#    def __init__(self):
#        #print "**************** NVEIntegrator_constraint init **************"
#        super(NVTIntegrator_constraint,self).__init__()
#    #print("~~~~~~~~~~~~~~~~~~~~~~ tau = ", self.thermostat.tau)
#
#    def bind(self, motion):
#        """ Reference all the variables for simpler access."""
#
#        if len(motion.nmts) > 1 or motion.nmts[0] != 1 :
#            raise ValueError("NVTIntegrator_constraint does not support multiple time stepping. Use NVTIntegrator_constraintMTS instead")
#
#        ConstrainedIntegrator.bind(self, motion)
#
#
#    def step_Oc(self):
#
#        for i in xrange(self.nsteps_o):
#            self.thermostat.step()
#
#            p = dstrip(self.beads.p).copy()
#            sm = dstrip(self.thermostat.sm)
#            p /= sm
#            self.ensemble.eens += np.dot(p.flatten(), p.flatten()) * 0.5
#
#            self.proj_cotangent(self.beads)
#
#            p = dstrip(self.beads.p).copy()
#            p /= sm
#            self.ensemble.eens -= np.dot(p.flatten(), p.flatten()) * 0.5
#
#
#    def step(self, step=None):
#        """Does one simulation time step."""
#        #print("~~~~~~~~~~~~~~~~~~~~~~ tau = ", self.thermostat.tau)
#        #print("~~~~~~~~~~~~~~~~~~~~~~ dt = ", self.thermostat.dt)
#        if self.splitting == "obabo":
#            self.step_Oc()
#            self.step_Bc(.5 * self.dt)
#            self.step_Ag(self.dt,Nsteps=self.nsteps_geo)
#            self.step_Bc(.5 * self.dt)
#            self.step_Oc()
#
#
#        elif self.splitting == "baoab":
#            self.step_Bc(.5 * self.dt)
#            self.step_Ag(.5 * self.dt,Nsteps=self.nsteps_geo)
#            self.step_Oc()
#            self.step_Ag(.5 * self.dt,Nsteps=self.nsteps_geo)
#            self.step_Bc(.5 * self.dt)
#
#
#class ConstrainedIntegratorMTS(NVTIntegrator_constraint):
#
#    def bind(self, motion):
#        """ Reference all the variables for simpler access."""
#        ConstrainedIntegrator.bind(self, motion)
#
#
#    def step_A(self, stepsize=None, level=0):
#        if stepsize is None:
#            stepsize = self.qdt/np.prod(self.nmts[:(level+1)])
#
#        super(ConstrainedIntegratorMTS,self).step_A(stepsize)
#
#    def step_B(self, stepsize=None, level=0):
#        if stepsize is None:
#            stepsize = self.pdt[0]/np.prod(self.nmts[:(level+1)])
#        """Unconstrained B-step"""
#        self.beads.p[0] += self.forces.forces_mts(level)[0] * stepsize
#
#    def step_Bc(self, stepsize=None, level=0):
#        """Unconstrained B-step followed by a projection into the cotangent space"""
#        self.step_B(stepsize=stepsize, level=level)
#        self.proj_cotangent(self.beads)
#
#    def step_BAc(self, stepsize=None, level=0):
#        """
#            half unconstrained B step and full A step followed by a projection onto the manifold
#        """
#        if stepsize is None:
#            stepsize = self.dt/np.prod(self.nmts[:(level+1)])
#
#        if not self.ciu:
#            self.update_constraints(self.beads)
#
#        self.step_B(.5 * stepsize,level=level)
#        self.step_A(stepsize)
#        self.proj_manifold(self.beads, stepsize)
#
#
#    def step_Ag(self, stepsize=None, Nsteps=None, level=0):
#        '''
#        Geodesic flow
#
#        When called: -self.d_params is  assumed to satisfy contraint
#        '''
#        if stepsize is None:
#            stepsize = self.qdt/np.prod(self.nmts[:(level+1)])
#        if Nsteps is None:
#            Nsteps = self.n_geoflow
#
#        super(ConstrainedIntegratorMTS,self).step_Ag(stepsize=stepsize, Nsteps=Nsteps)
#
#    def step_rattle(self, stepsize=None, level=0):
#        if stepsize is None:
#            stepsize = self.dt/np.prod(self.nmts[:(level+1)])
#
#        self.step_BAc(stepsize=stepsize, level=level)
#        self.step_Bc(stepsize = .5 * stepsize, level=level)
#
#    def step_respa(self, level=0):
#
#        #print("level: ", level)
#        stepsize = self.dt/np.prod(self.nmts[:(level+1)])
#
#        #print("stepsize: ", stepsize)
#        self.step_Bc(stepsize = .5 * stepsize, level = level)
#        #self.step_B(stepsize = .5 * stepsize, level = level)
#        if level+1 < np.size(self.nmts):
#            for i in range(self.nmts[level+1]):
#                self.step_respa(level=level+1)
#        else:
#            self.step_A(stepsize)
#            self.proj_manifold(self.beads, stepsize=stepsize, proj_p=True)
#        #self.step_B(stepsize = .5 * stepsize, level = level)
#        self.step_Bc(stepsize = .5 * stepsize, level = level)
#
#    def step_grespa(self, level=0):
#
#        stepsize = self.dt/np.prod(self.nmts[:(level+1)])
#
#        self.step_Bc(stepsize = .5 * stepsize, level = level)
#        if level+1 < np.size(self.nmts):
#            for i in range(self.nmts[level+1]):
#                self.step_grespa(level=level+1)
#        else:
#            self.step_Ag(stepsize, Nsteps=self.nsteps_geo[level])
#        self.step_Bc(stepsize = .5 * stepsize, level = level)
#
#    def step(self, step=None):
#        raise NotImplementedError()
#
#class NVEIntegrator_constraintMTS(ConstrainedIntegratorMTS):
#
#    def step(self, step=None):
#        """Does one simulation time step."""
#        #print("~~~~~~~~~~~~~~~~~~~~~~ tau = ", self.thermostat.tau)
#        #print("~~~~~~~~~~~~~~~~~~~~~~ dt = ", self.thermostat.dt)
#        if self.splitting == "respa":
#            self.step_respa()
#
#
#        elif self.splitting == "grespa":
#            self.step_grespa()
#
#        else:
#            raise ValueError("No valid splitting method spefified for NVE integration. Choose \"respa\" or \"grespa\".")
#
#
#class NVTIntegrator_constraintMTS(ConstrainedIntegratorMTS):
#
#
#    def get_tdt(self):
#        if self.splitting in ["o-respa-o", "o-grespa-o"]:
#            return self.dt * 0.5
#        else:
#            raise ValueError("Invalid splitting requested. Only \"o-respa-o\" and \"o-grespa-o\" are supported.")
#
#    def step(self, step=None):
#        """Does one simulation time step."""
#        #print("~~~~~~~~~~~~~~~~~~~~~~ tau = ", self.thermostat.tau)
#        #print("~~~~~~~~~~~~~~~~~~~~~~ dt = ", self.thermostat.dt)
#        if self.splitting == "o-respa-o":
#            self.step_Oc()
#            self.step_respa()
#            self.step_Oc()
#
#        elif self.splitting == "o-grespa-o":
#            self.step_grespa()
#
#        else:
#            raise ValueError("No valid splitting method spefified for NVE integration. Choose \"o-respa-o\" or \"o-grespa-o\".")

#class ConstraintBase(dobject):
#    """ Constraint class for MD. Base class."""
#
#    def __init__(self, constrained_indices, constraint_values, ncons=0):
#        self.constrained_indices = constrained_indices
#        if len(constraint_values) != 0:
#            self.ncons = len(constraint_values)
#            dd(self).constraint_values = depend_array(name="constraint_values",
#                value=np.asarray(constraint_values).copy())
#        elif ncons != 0:
#            self.ncons = ncons
#            dd(self).constraint_values = depend_array(name="constraint_values",
#                value=np.zeros(ncons))
#        else:
#            raise ValueError("cannot determine the number of constraints in list")
#        # determines the list of unique indices of atoms that are affected by this constraint
#        self.i_unique = np.unique(self.constrained_indices.flatten())
#        self.mk_idmaps()
#
#    def mk_idmaps(self):
#
#        # makes lookup dictionary and lists to quickly access the portions of arrays that are affected by this constraint
#        self.i_reverse = { value : i for i,value in enumerate(self.i_unique)}
#        self.n_unique = len(self.i_unique)
#        self.i3_unique = np.zeros(self.n_unique*3, int)
#        for i in range(self.n_unique):
#            self.i3_unique[3*i:3*(i+1)] = [3*self.i_unique[i], 3*self.i_unique[i]+1, 3*self.i_unique[i]+2]
#        # this can be used to access the position array based on the constrained_indices list
#        self.i3_indirect = np.zeros((len(self.constrained_indices.flatten()),3), int)
#        for ri, i in enumerate(self.constrained_indices.flatten()):
#            rri = self.i_reverse[i]
#            self.i3_indirect[ri] = [3*rri, 3*rri+1, 3*rri+2]
#
#    def bind(self, beads):
#
#        self.beads = beads
#        dself = dd(self)
#        dself.m3 = depend_array(name="m3", value=np.zeros(self.n_unique*3),
#                    func=(lambda: self.beads.m3[0,self.i3_unique]),
#                    dependencies=[dd(self.beads).m3])
#        # The constraint function is computed at iteratively updated 
#        # coordinates during geodesic integration; this array holds such
#        # updates
#        dself.q = depend_array(name="q", value=np.zeros(self.n_unique*3))
#        dself.g = depend_array(name="g", value=np.zeros(self.ncons),
#                               func=self.gfunc, 
#                               dependencies=[dself.q, dself.constraint_values])
#        # The gradient of the constraint function is computed at the configuration
#        # obtained from the previous converged SHAKE step; this array holds
#        # a local copy of that configuration
#        dself.qprev = depend_array(name="qprev", 
#                                   value=np.zeros(self.n_unique*3))
#        dself.Dg = depend_array(name="Dg", 
#                                value=np.zeros((self.ncons, self.n_unique*3)),
#                                func=self.Dgfunc, 
#                                dependencies=[dself.qprev,
#                                              dself.constraint_values])
#        dself.GramChol = depend_array(name="GramChol",
#                                      value=np.zeros((self.ncons,self.ncons)),
#                                      func=self.GCfunc, 
#                                      dependencies=[dself.Dg] )
#
#    def gfunc(self):
#        raise NotImplementedError()
#
#    def Dgfunc(self):
#        """
#        Calculates the Jacobian of the constraint.
#        """
#        raise NotImplementedError()
#
#    def GCfunc(self):
#        dg = dstrip(self.Dg).copy()
#        dgm = dg/self.m3
#        return np.linalg.cholesky(np.dot(dg, dgm.T))


#class RigidBondConstraint(ConstraintBase):
#    """ Constraint class for MD.
#        Specialized for rigid bonds. This can actually hold a *list*
#        of rigid bonds, i.e. there will be a list of pairs of atoms and
#        a list of bond lengths. """
#
#    def __init__(self,constrained_indices,constraint_values):
#
#        if len(constraint_values) == 0:
#            ncons = -1
#            self._calc_cons = True
#        else:
#            ncons = len(constraint_values)
#            self._calc_cons = False
#        icons = np.reshape(constrained_indices,(ncons,2))
#        super(RigidBondConstraint,self).__init__(
#                constrained_indices, constraint_values, 
#                ncons=len(icons))
#        self.constrained_indices.shape = (self.ncons, 2)
#        self.i3_indirect.shape = (self.ncons, 2, 3)
#        
#    def bind(self, beads):
#        
#        super(RigidBondConstraint, self).bind(beads)
#        if self._calc_cons:
#            self.q = dstrip(beads.q[0])[self.i3_unique.flatten()]
#            self.constraint_values = np.sqrt(dstrip(self.g))
#
#    def gfunc(self):
#        """
#        Calculates the constraint.
#        """
#
#        q = dstrip(self.q)
#        r = np.zeros(self.ncons)
#        constraint_distances = dstrip(self.constraint_values)
#        for i in range(self.ncons):
#            c_atoms = self.i3_indirect[i]
#            c_dist = constraint_distances[i]
#            #print q[c_atoms[0]], q[c_atoms[1]], c_dist
#            r[i] = np.sum((q[c_atoms[0]] - q[c_atoms[1]])**2) - c_dist**2
#        if q[0] == float('inf'):
#            ValueError("fgfgf")
#            print("autsch")
#            exit()
#        #print("gfunc", r)
#        return r
#    
#    def Dgfunc(self, reduced=False):
#        """
#        Calculates the Jacobian of the constraint.
#        """
#
#        q = dstrip(self.qprev)
#        #constrained_indices = self.constrained_indices
#        r = np.zeros((self.ncons, self.n_unique*3))
#        for i in range(self.ncons):
#            c_atoms = self.i3_indirect[i]
#            inst_position_vector = q[c_atoms[0]] - q[c_atoms[1]]
#            r[i][c_atoms[0]] =   2.0 * inst_position_vector
#            r[i][c_atoms[1]] = - 2.0 * inst_position_vector
#        return r
#    
#class AngleConstraint(ConstraintBase):
#    """ Constraint class for MD specialized for angles. 
#        This can hold a list of angles, i.e. a list of triples of atoms
#        and the corresponding values. We adopt the convention that the 
#        middle atom appears first in the list.
#    """
#
#    def __init__(self,constrained_indices,constraint_values):
#
#        if len(constraint_values) == 0:
#            ncons = -1
#            self._calc_cons = True
#        else:
#            ncons = len(constraint_values)
#            self._calc_cons = False
#        icons = np.reshape(constrained_indices,(ncons,3))
#        super(AngleConstraint,self).__init__(
#                constrained_indices, constraint_values, 
#                ncons=len(icons))
#        self.constrained_indices.shape = (self.ncons, 3)
#        self.i3_indirect.shape = (self.ncons, 3, 3)
#        
#    def bind(self, beads):
#        
#        super(AngleConstraint, self).bind(beads)
#        if self._calc_cons:
#            self.constraint_values = np.pi/2 # so that cos(angle) = 0
#            self.q = dstrip(beads.q[0])[self.i3_unique.flatten()]
#            self.constraint_values = np.arccos(dstrip(self.g))
#
#    def gfunc(self):
#        """
#        Calculates the constraint.
#        """
#
#        q = dstrip(self.q)
#        r = np.zeros(self.ncons)
#        constraint_cosines = np.cos(dstrip(self.constraint_values))
#        for i in range(self.ncons):
#            c_atoms = self.i3_indirect[i]
#            c_cos = constraint_cosines[i]
#            q1 = q[c_atoms[1]] - q[c_atoms[0]]
#            r1 = np.sqrt(np.dot(q1,q1))
#            q2 = q[c_atoms[2]] - q[c_atoms[0]]
#            r2 = np.sqrt(np.dot(q2,q2))
#            r[i] = np.dot(q1,q2)/r1/r2
#            r[i] -= c_cos
#        return r
#
#    def Dgfunc(self, reduced=False):
#        """
#        Calculates the Jacobian of the constraint.
#        """
#
#        q = dstrip(self.qprev)
#        r = np.zeros((self.ncons, self.n_unique*3))
#        for i in range(self.ncons):
#            c_atoms = self.i3_indirect[i]
#            q1 = q[c_atoms[1]] - q[c_atoms[0]]
#            r1 = np.sqrt(np.dot(q1,q1))
#            q1 /= r1
#            q2 = q[c_atoms[2]] - q[c_atoms[0]]
#            r2 = np.sqrt(np.dot(q2,q2))
#            q2 /= r2
#            ct = np.dot(q1,q2)
#            r[i][c_atoms[1]] = (q2 - ct*q1)/r1
#            r[i][c_atoms[2]] = (q1 - ct*q2)/r2
#            r[i][c_atoms[0]] = -(r[i][c_atoms[1]] + r[i][c_atoms[2]])
#        return r
    
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
#
#class ConstraintList(ConstraintBase):
#    """ Constraint class for MD"""
#
#    def __init__(self, constraint_list):
#        self.constraint_list = constraint_list
#        self.ncons = sum([constr.ncons for constr in constraint_list])
#
#        # determines the list of unique indices of atoms that are affected 
#        # by this list of constraint
#        self.i_unique = np.zeros(0)
#        for c in self.constraint_list:
#            self.i_unique = np.union1d(
#                    c.constrained_indices.flatten(),self.i_unique
#                    )
#        self.constrained_indices = self.i_unique
#        self.mk_idmaps()
#
#        # must now find the mapping from the unique indices in each constraint 
#        # ic_map[i] gives the position where the atoms involved 
#        # in the i-th constraint are stored in the compact list
#        self.ic_map = []
#        self.ic3_map = []
#        for c in self.constraint_list:
#            i_map = np.array([self.i_reverse[i] for i in c.i_unique])
#            self.ic_map.append(i_map)
#            self.ic3_map.append(np.array([[i*3, i*3+1, i*3+2] for i in i_map]).flatten())
#
#    def bind(self, beads):
#
#        # this is special because it doesn't hold constraint_values so we have to really specialize
#        self.beads = beads
#        dself = dd(self)
#        dself.m3 = depend_array(name="m3", value=np.zeros(self.n_unique*3),
#            func=(lambda: self.beads.m3[0,self.i3_unique]),
#            dependencies=[dd(self.beads).m3])
#        # this holds all of the atoms in this list of constraints
#        dself.q = depend_array(name="q", value=np.zeros(self.n_unique*3))
#        # this holds the configurations of the listed atom obtained
#        # at the end of the previous step
#        dself.qprev = depend_array(name="qprev", value=np.zeros(self.n_unique*3))
#        dself.g = depend_array(name="g", value=np.zeros(self.ncons),
#                               func=self.gfunc)
#        dself.Dg = depend_array(name="Dg", 
#                                value=np.zeros((self.ncons, self.n_unique*3)), 
#                                func=self.Dgfunc)
#        dself.GramChol = depend_array(name="GramChol", 
#                                      value=np.zeros((self.ncons,self.ncons)),
#                                      func=self.GCfunc, dependencies=[dself.Dg])
#        # we link all of the sub-constraints to the lists of unique q and qprev,
#        # so that each constraint gets automatically updated
#        def make_qgetter(k):
#            return lambda: self.q[self.ic3_map[k]]
#        def make_qprevgetter(k):
#            return lambda: self.qprev[self.ic3_map[k]]
#        for ic, c in enumerate(self.constraint_list):
#            c.bind(beads)
#            # deal with constraint functions
#            dq = dd(c).q
#            dq.add_dependency(dself.q)
#            dq._func = make_qgetter(ic)
#            dself.g.add_dependency(dd(c).g)
#            # ...and their gradients
#            dqprev = dd(c).qprev
#            dqprev.add_dependency(dself.qprev)
#            dqprev._func = make_qprevgetter(ic)
#            dself.Dg.add_dependency(dd(c).Dg)
#
#    def gfunc(self):
#        """
#        Compute the constraint function.
#        """
#        r = np.zeros(self.ncons)
#        si = 0
#        for constr in self.constraint_list:
#            r[si:si+constr.ncons] = constr.g
#            si += constr.ncons
#        return r
#
#    def Dgfunc(self):
#        """
#        Compute the Jacobian of the constraint function.
#        """
#        q = dstrip(self.qprev)
#        r = np.zeros((self.ncons, np.size(q)))
#        si = 0
#        for ic, constr in enumerate(self.constraint_list):
#            r[si:si+constr.ncons, self.ic3_map[ic]] = constr.Dg
#            si += constr.ncons
#        return r
#
#    def get_iai(self):
#        iai = []
#        for constr in self.constraint_list:
#            iai += list(constr.get_iai())
#        return np.unique(iai)
#

#        
#
#class SparseConstraintSolver(ConstraintSolverBase):
#
#    def __init__(self, constraint_list, tolerance=0.001,maxit=1000,norm_order=2):
#        super(SparseConstraintSolver,self).__init__(constraint_list)
#
#        self.tolerance = tolerance
#        self.maxit = maxit
#        self.norm_order = norm_order
#        #self.ic_list = [constr.get_ic() for constr in self.constraint_list]
#
#    def bind(self, beads):
#
#        self.beads = beads
#
#        # sets the initial value of the constraint positions
#        q = dstrip(beads.q[0])
#        for constr in self.constraint_list:
#            constr.qprev = q[constr.i3_unique.flatten()]
#
#    def update_constraints(self, beads):
#
#        #m3 = dstrip(beads.m3[0])
#        #self.Dg_list = [constr.Dg[:,constr] for constr in self.constraint_list ]
#        #self.Gram_list = [np.dot(dg,(dg/m3[ic]).T) for dg, ic in zip(self.Dg_list,self.ic_list)]
#        #self.GramChol_list = [ np.linalg.cholesky(G) for G in self.Gram_list]
#        self.ciu = True
#
#
#    def proj_cotangent(self, beads):
#
#        m3 = dstrip(beads.m3[0])
#        p = dstrip(beads.p[0]).copy()
#
#        if len(self.constraint_list) > 0:
#            #print("number of constraints: ", len(self.constraint_list))
#            for constr in self.constraint_list : #zip(self.Dg_list, self.ic_list, self.GramChol_list):
#                dg = dstrip(constr.Dg)
#                ic = constr.i3_unique
#                gramchol = dstrip(constr.GramChol)
#
#                b = np.dot(dg, p[ic]/constr.m3)
#                x = np.linalg.solve(np.transpose(gramchol),np.linalg.solve(gramchol, b))
#                p[ic] += - np.dot(np.transpose(dg),x)
#        beads.p[0] = p
#
#    def proj_manifold(self, beads, stepsize=None, proj_p=True):
#        '''
#        projects onto Manifold using the Gram matrix defined by self.Dg and self.Gram
#        '''
#        m3 = dstrip(beads.m3[0])
#        p = dstrip(beads.p[0]).copy()
#        q = dstrip(beads.q[0]).copy()
#
#        #for constr in self.constraint_list:
#                #print "before", constr.qprev
#                #print "vs", q[constr.i3_unique.flatten()]
#
#        i = 0
#        if len(self.constraint_list) > 0:
#            for constr in self.constraint_list: # zip(self.Dg_list, self.ic_list, self.GramChol_list, self.constraint_list):
#                # these must only be computed on the manifold so we store them and don't update them
#                dg = dstrip(constr.Dg)#.copy()
#                gramchol = dstrip(constr.GramChol)#.copy()
#                ic = constr.i3_unique
#                constr.q = q[ic]
#                g = dstrip(constr.g)
#                #print "g vector", g
#                while (i < self.maxit and self.tolerance <= np.linalg.norm(g, ord=self.norm_order)):
#                    dlambda = np.linalg.solve(np.transpose(gramchol),np.linalg.solve(gramchol, g))
#                    delta = np.dot(np.transpose(dg),dlambda)
#                    q[ic] += - delta / m3[ic]
#                    constr.q = q[ic] # updates the constraint to recompute g
#                    g = dstrip(constr.g)
#                    if proj_p:
#                        update_diff = - delta / stepsize
#                        p[ic] += update_diff
#                        i += 1
#                    if (i == self.maxit):
#                        print('No convergence in Newton iteration for positional component');
#            # in theory each constraint update should be independent but we cannot be sure...
#            # after all constraints have been applied, q is on the manifold and we can update the constraint
#            # positions
#            for constr in self.constraint_list:
#                constr.qprev = q[constr.i3_unique.flatten()]
#                #print "after", constr.qprev
#
#        beads.p[0] = p
#        beads.q[0] = q
#
#
