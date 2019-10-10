"""Contains the classes that deal with the different dynamics required to
propagate mean-field quasicentroid dynamics under different types of ensembles.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np

from ipi.engine.atoms import Atoms
from ipi.engine.motion import Motion
from ipi.engine.motion.dynamics import Dynamics, DummyIntegrator, NVTIntegrator
from ipi.engine.motion.constrained_dynamics import EckartGroupedConstraints, \
    BondLengthConstraint, BondAngleConstraint
from ipi.engine.thermostats import Thermostat
from ipi.engine.barostats import Barostat
from ipi.utils.depend import depend_value, depend_array, \
                             dd, dobject, dstrip, dpipe
from ipi.utils.mathtools import to_com, mominertia, eckrot
from ipi.utils.messages import verbosity, info
from ipi.utils.units import unit_to_user, Constants

        
class QuasiCentroids(dobject):

    """Handles the ring-polymer quasicentroids.

    Attributes:
        beads: A beads object giving the atoms' configuration
        forces: A forces object giving the forces acting on each bead

    Depend objects:
        q: The quasi-centroid configuration
        p: The quasi-centroid momenta
        m: The atomic quasi-centroid masses (shape self.natoms)
        f: The total force acting on the quasi-centroid
        m3: Array of masses conforming with p and q (shape 3*self.natoms)
        bforce: The Forces object of the beads underlying the quasi-centroids.

    """

    def __init__(self, pqc=None, qqc=None):
        """Initialises a "QuasiCentroids" object --- simply store quasicentroid
        momenta whenever available.
        """
        
        if pqc is None:
            self.p = None
        elif len(pqc) == 0:
            self.p = None
        else:
            self.p = np.asarray(pqc).flatten()
            
        if qqc is None:
            self.q = None
        elif len(qqc) == 0:
            self.q = None
        else:
            self.q = np.asarray(qqc).flatten()
            
    def bind(self, ens, beads, nm, cell, bforce, prng, omaker, cdyn):
        self.beads = beads
        self.ensemble = ens
        self.forces = bforce
        self.bbias = ens.bias
        self.natoms = self.beads.natoms
        dself = dd(self)
        dforces = dd(self.forces)
        dself.names = depend_array(
                name="names", 
                value=np.zeros(self.natoms, np.dtype('|S6')))
        dpipe(dd(self.beads).names, dself.names)
        dself.m = depend_array(
                name="m", 
                value=dstrip(self.beads.m).copy(),
                func=(lambda: dstrip(self.beads.m).copy()),
                dependencies=[dd(self.beads).m]
                )
        dself.m3 = depend_array(
                name="m3", 
                value=dstrip(self.beads.m3[0]).copy(),
                func=(lambda: dstrip(self.beads.m3[0]).copy()),
                dependencies=[dd(self.beads).m3]
                )
        dself.sm3 = depend_array(
                name="sm3", 
                value=np.zeros(3*self.natoms, float),
                func=(lambda: np.sqrt(dstrip(self.m3))), 
                dependencies=[dself.m3])
        p = self.p
        dself.p = depend_array(name="p", value=np.zeros(3*self.natoms))
        if p is None:
            info(
" # Resampling quasicentroid velocities at {:g} K".format(
unit_to_user("energy", "kelvin", ens.temp)), verbosity.low)
            self.p = (prng.gvec(self.p.size) * np.sqrt(dstrip(self.m3)*ens.temp * Constants.kb))
        else:
            self.p = p
        q = self.q
        dself.q = depend_array(
                name="q", 
                value=np.nan*np.ones(3*self.natoms))
        # Determine the kinds of quasicentroids
        self.quasicentroid_groups = []
        for cgp in cdyn.constraint_groups:
            class_list = []
            for c in cgp.clist:
                class_list.append(c.__class__)
            if class_list == [
                    BondLengthConstraint,
                    BondLengthConstraint, 
                    BondAngleConstraint
                    ]:
                qgp = GroupedQuasiTriatomics()
            else:
                raise ValueError(
"Unknown constraint combination {:s}".format(class_list.__repr__()))
            qgp.bind(self.beads, cgp)
            if q is None:
                self.q[qgp.i3list] = dstrip(qgp.q).copy()
            else:
                self.q[qgp.i3list] = q[qgp.i3list]
            dd(qgp).q.add_dependency(dself.q)
            dd(qgp).q._func = lambda: dstrip(self.q)[qgp.i3list]
            self.quasicentroid_groups.append(qgp)
        if np.any(np.isnan(dstrip(self.q))):
            # TODO: apply centroid constraints to atoms that have not been hit
            raise ValueError("Quasicentriod constraints do not cover the entire system.")
        dself.f = depend_array(
            name="f", 
            value=np.zeros_like(dstrip(self.m)), 
            func=self.f_combine,
            dependencies=[dforces.f])
        # Access quasi-centroids as Atoms object
        self.atoms = Atoms(self.natoms, _prebind=(self.q, self.p, self.m, self.names))
        datoms = dd(self.atoms)
        # Kinetic energies of the quasi-centroids, and total kinetic stress tensor
        dself.kin = depend_value(name="kin", func=(lambda: self.atoms.kin),
                                 dependencies=[datoms.kin,])
        dself.kstress = depend_array(name="kstress", value=np.zeros((3, 3), float),
                                     func=(lambda: dstrip(self.atoms.kstress)),
                                     dependencies=[datoms.kstress,])
        # Add quasicentroid kinetic energy to ensemble
        self.ensemble.add_econs(dself.kin)
        self.ensemble.add_xlkin(dself.kin)
            
    def f_combine(self):
        """Obtains the total force vector."""
        
        f = np.zeros(3*self.natoms)
        fbead = dstrip(self.forces.f)
        for qgp in self.quasicentroid_groups:
            f[qgp.i3list] = qgp.get_forces(fbead)
        return f
    
    def forces_mts(self, level):
        """ Fetches the forces associated with a given MTS level."""
        
        f = np.zeros(3*self.natoms)
        fbead = self.forces.forces_mts(level)
        for qgp in self.quasicentroid_groups:
            f[qgp.i3list] = qgp.get_forces(fbead)
        return f
    
    def bias(self):
        """ Returns the bias force"""
        
        f = np.zeros(3*self.natoms)
        fbead = dstrip(self.bbias.f)
        for qgp in self.quasicentroid_groups:
            f[qgp.i3list] = qgp.get_forces(fbead)
        return f

class QuasiCentroidMotion(Motion):
    """A class to hold a QuasiCentroidDynamics and a ConstrainedDynamics
    object.
    """
    def __init__(self, constrained_motion, quasi_motion):
        """Initialises QuasiCentroidMotion.

        Args:
            constrained_motion: dynamics of the quasicentroid-constrained beads
            quasi_motion: dynamics of the quasicentroids
            timestep: the integration timestep of the two dynamics
            splitting: integrator splitting
        """
        
        self.cmotion = constrained_motion
        self.qmotion = quasi_motion
        self.fixcom = self.qmotion.fixcom
        self.fixatoms = self.qmotion.fixatoms
        dself = dd(self)
        dself.dt = depend_value(name="dt", value=1.0)
        dpipe(dd(self.cmotion).dt, dself.dt)
        
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
            constrained_dynamics: the integrator for the quasicentroid-constrained
                dynamics of the RP fluctuations.
        """
        
        # Bind constrained dynamics first
        self.cmotion.bind(ens, beads, nm, cell, bforce, prng, omaker)
        # Now the quasicentroids
        self.qmotion.bind(ens, beads, nm, cell, bforce, prng, omaker, self.cmotion)
        
    def step(self, step=None):
        """
        """
        # Quasicentroid motion also moves the fluctuations
        self.qmotion.step(step)
        

class QuasiCentroidDynamics(Dynamics):
    
    """A class for integrating quasi-centroid equations of motion.
    
    Attributes:
        tba
        
    Depend objects:
        tba
    """
    
    def __init__(self, timestep, mode="nve", splitting="obabo",
            thermostat=None, barostat=None, fixcom=False, 
            fixatoms=None, nmts=None, pqc=None, qqc=None):
        """Initialises a "quasicentroid dynamics" motion object.

        Args:
            tba
        """
        
        super(Dynamics, self).__init__(fixcom=fixcom, fixatoms=fixatoms)
        dd(self).dt = depend_value(name='dt', value=timestep)
        
        if thermostat is None:
            self.thermostat = Thermostat()
        else:
            self.thermostat = thermostat
            
        if nmts is None or len(nmts) == 0:
            dd(self).nmts = depend_array(name="nmts", value=np.asarray([1], int))
        else:
            dd(self).nmts = depend_array(name="nmts", value=np.asarray(nmts, int))
            
        if barostat is None:
            self.barostat = Barostat()
        else:
            self.barostat = barostat
        self.enstype = mode
        if self.enstype == "nve":
            self.integrator = NVEQuasiIntegrator()
        elif self.enstype == "nvt":
            self.integrator = NVTQuasiIntegrator()
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
        # quasicentroid object
        self.quasi = QuasiCentroids(pqc, qqc)
        
    
    def bind(self, ens, beads, nm, cell, bforce, prng, omaker,
             constrained_dynamics):
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
            constrained_dynamics: the integrator for the quasicentroid-constrained
                dynamics of the RP fluctuations.
        """
        
        super(Dynamics, self).bind(ens, beads, nm, cell, bforce, prng, omaker)
        self.cdyn = constrained_dynamics
        # Pipe integrator parameters from constrained dynamics
        dself = dd(self)
        dcdyn = dd(self.cdyn)
        dpipe(dcdyn.dt, dself.dt)
        dpipe(dcdyn.splitting, dself.splitting)
        dpipe(dcdyn.nmts, dself.nmts)
        if (len(self.nmts) != self.forces.nmtslevels):
            raise ValueError("The number of mts levels for the integrator does not agree with the mts_weights of the force components.")
        dthrm = dd(self.thermostat)
        dbaro = dd(self.barostat)
        dens = dd(ens)
        # temperature for quasicentroid partition function
        dself.ntemp = depend_value(
                name='ntemp', func=self.get_ntemp, dependencies=[dens.temp])
        # quasi-centroids
        self.quasi.bind(ens, beads, nm, cell, bforce, prng, omaker, constrained_dynamics)
        # fixed degrees of freedom count
        fixdof = 3*len(self.fixatoms)
        if self.fixcom:
            fixdof += 3
        dpipe(dself.ntemp, dthrm.temp)
        self.thermostat.bind(atoms=self.quasi.atoms, prng=prng, fixdof=fixdof)
        # NPT propagation not yet implemented for quasicentroids
        # self.barostat should not at any point be called.
        dpipe(dself.ntemp, dbaro.temp)
        dpipe(dens.pext, dbaro.pext)
        dpipe(dens.stressext, dbaro.stressext)
        self.barostat.bind(beads, nm, cell, bforce, prng=prng, fixdof=fixdof, nmts=len(self.nmts))
        # now that the timesteps are decided, we proceed to bind the integrator.
        self.integrator.bind(self)
        self.ensemble.add_econs(dthrm.ethermo)
        self.ensemble.add_econs(dbaro.ebaro)
        # adds the potential, kinetic enrgy and the cell jacobian to the ensemble
        self.ensemble.add_xlpot(dbaro.pot)
        self.ensemble.add_xlpot(dbaro.cell_jacobian)
        self.ensemble.add_xlkin(dbaro.kin)
        # applies constraints immediately after initialization.
        self.integrator.pconstraints()
                    
    def get_ntemp(self):
        """Returns the PI simulation temperature (just the physical T)."""

        return self.ensemble.temp
    
    
class GroupedQuasiBase(dobject):
    """Base class for grouped quasicentroids that converts between 
    quasicentroid coordinates and values of constraint functions,
    and from bead forces to quasicentroid forces.
    """
    
    def bind(self, beads, cgp):
        """
        Bind the beads, bead forces, and a GroupedConstraints
        object to the grouped quasicentroids.
        """
        # Create local references to useful objects
        self.beads = beads
        self.cgp = cgp
        self.i3list = cgp.i3unique.reshape((cgp.ngp, cgp.nunique, 3))
        self._eckart = isinstance(cgp, EckartGroupedConstraints)
        dself = dd(self)
        dself.qbead = depend_array(
            name="qbead",
            value=np.zeros((cgp.nbeads, cgp.ngp, cgp.nunique, 3)),
            func=(lambda: dstrip(self.beads.q)[:,self.i3list]),
            dependencies=[dd(self.beads).q]
            )
        
        dself.mbead = depend_array(
            name="mbead",
            value=np.zeros_like(dstrip(self.qbead)),
            func=(lambda: dstrip(self.beads.m3)[:,self.i3list]),
            dependencies=[dd(self.beads).m3]
            )
        dself.m = depend_array(
                name="m",
                value=np.zeros((cgp.ngp, cgp.nunique, 3)),
                func=(lambda: dstrip(self.beads.m3)[0,self.i3list]),
                dependencies=[dd(self.beads).m3]
                )
        dself.q = depend_array(
                name="q", 
                value=self.get_coords())  # will pipe from main array
        # Add dependency to cgp targetvals
        dcgp = dd(self.cgp)
        dcgp.targetvals.add_dependency(dself.q)
        dcgp.targetvals._func = lambda: self.get_targets()
        if isinstance(cgp, EckartGroupedConstraints):
            dcgp.qref.add_dependency(dself.q)
            dcgp.qref._func = lambda: dstrip(self.q).copy()
    
    def get_forces(self, f):
        raise NotImplementedError()
        
    def get_coords(self):
        raise NotImplementedError()
        
    def get_targets(self):
        raise NotImplementedError()


class GroupedQuasiTriatomics(GroupedQuasiBase):
    """Groups of triatomic quasicentroids (H2O-type).
    """
    
    @staticmethod
    def qcart2ba(q):
        """Convert Cartesian coordinates to r1, r2, cos(theta)
        
        Args:
            q (ndarray): Cartesian coordinates in an array of dimension
                (..., 3, 3), where the last dimension runs over
                x, y, z components and the penultimate dimension runs over
                the atoms (central atom comes first).
                
        Return:
            ba (ndarray): Bond-angle coordinates in an array of dimension
                (..., 3), where the last dimension runs over r1, r2, 
                cos(theta).
        """
        ba = np.zeros(q.shape[:-1])
        r1 = ba[...,0]
        r2 = ba[...,1]
        ct = ba[...,2]
        q1 = q[...,1,:] - q[...,0,:]
        r1[:] = np.sqrt(np.sum(q1**2, axis=-1))
        q1 /= r1[...,None]
        q2 = q[...,2,:] - q[...,0,:]
        r2[:] = np.sqrt(np.sum(q2**2, axis=-1))
        q2 /= r2[...,None]
        ct[:] = np.sum(q1*q2, axis=-1)
        return ba
    
    @staticmethod
    def fcart2ba(q, f):
        """Convert Cartesian forces to forces of r1, r2, and theta.
        NOTE: this assumes that external components have been subtracted
        from f.
        
        Args: 
            q (ndarray): Cartesian coordinates stored as described in qcart2ba
            f (ndarray): Cartesian forces with external components removed;
                same storage convention as for q.
                
        Return:
            fba (ndarray): Forces on the bond-angle coordinates returned as
                an array of shape (nbeads,...,3), where the last dimension
                runs over r1, r2, and cos(theta).
        """
        fba = np.zeros(f.shape[:-1])
        fr1 = fba[...,0]
        fr2 = fba[...,1]
        ft = fba[...,2]
        #
        q1 = q[...,1,:] - q[...,0,:]
        f1 = f[...,1,:]
        r1 = np.sqrt(np.sum(q1**2, axis=-1))
        q1 /= r1[...,None]
        fr1[:] = np.sum(q1*f1, axis=-1)
        #
        q2 = q[...,2,:] - q[...,0,:]
        f2 = f[...,2,:]
        r2 = np.sqrt(np.sum(q2**2, axis=-1))
        q2 /= r2[...,None]
        fr2[:] = np.sum(q2*f2, axis=-1)
        #
        ct = np.sum(q1*q2, axis=-1)
        st = np.sqrt(1.0-ct**2)
        r1 /= st
        ft[:] = r1*np.sum((q1*ct[...,None]-q2)*f1, axis=-1)
        
        return fba
        
    @staticmethod
    def fba2cart(q, fba):
        """Convert forces on r1, r2, and theta into Cartesian representation.
        
        Args:
            q (ndarray): Cartesian coordinates stored as described in qcart2ba
            fba (ndarray): Internal forces, as returned by fcart2ba
            
        Return:
            f (ndarray): Internal forces in Cartesian representation.
        """
        fr1 = fba[...,0]
        fr2 = fba[...,1]
        ft = fba[...,2]
        f = np.zeros_like(q)
        #
        f1 = f[...,1,:]
        q1 = q[...,1,:] - q[...,0,:]
        r1 = np.sqrt(np.sum(q1**2, axis=-1))
        q1 /= r1[...,None]
        # 
        f2 = f[...,2,:]
        q2 = q[...,2,:] - q[...,0,:]
        r2 = np.sqrt(np.sum(q2**2, axis=-1))
        q2 /= r2[...,None]
        #
        ct = np.sum(q1*q2, axis=-1)
        st = np.sqrt(1.0-ct**2)
        r1 *= st
        f1[:] = q1*fr1[...,None]-(q2-q1*ct[...,None])*(ft/r1)[...,None]
        r2 *= st
        f2[:] = q2*fr2[...,None]-(q1-q2*ct[...,None])*(ft/r2)[...,None]
        #
        f[...,0,:] = -(f1+f2)
        return f
    
    def get_forces(self, fbead):
        """Return the Cartesian forces on the quasicentroids derived from
        the mean forces on the bondlengths, mean force on the angle, and
        appropriately-defined translational and rotational components if
        self._eckart is True.
        """
        
        qbead = dstrip(self.qbead).copy() 
        fbead = fbead[:,self.i3list]
        mbead = dstrip(self.mbead)
        q = dstrip(self.q).copy()
        f = np.zeros_like(q)
        m = dstrip(self.m)
        q0 = dstrip(self.cgp.qc).copy().reshape(q.shape)
        f0 = np.mean(fbead, axis=0)
        # Shift each replica to its CoM frame
        qbead[:] = to_com(mbead, qbead)[0]
        # Remove external forces
        II = mominertia(mbead, qbead, shift=False)
        # Calculate the total torque about CoM for each replica
        tau_CoM = np.sum(np.cross(qbead, fbead,), axis=-2)
        # Convert to external forces and subtract from total
        alpha = np.linalg.solve(II, tau_CoM)
        f_CoM = np.sum(fbead, axis=-2)/np.sum(mbead[...,:1], axis=-2)
        fbead -= mbead*(f_CoM[...,None,:]+np.cross(alpha[...,None,:], qbead))
        # Extract the internal coordinates and mean forces
        fba = np.mean(self.fcart2ba(qbead, fbead), axis=0)
        # Shift quasicentroid to CoM
        q[:], CoM = to_com(m, q)
        # Convert bond-angle forces to Cartesians
        f[:] = self.fba2cart(q, fba)
        if (self._eckart):
            # Shift centroid to CoM
            q0 -= CoM
            # Extract the external forces on the quasicentroid
            rhs = np.sum(np.cross(f0, q)-np.cross(f, q0), axis=-2)
            lhs = -mominertia(m, q, q2=q0, shift=False)
            alpha = np.linalg.solve(lhs, rhs)
            f_CoM = np.sum(f0, axis=-2)/np.sum(m[...,0:1], axis=-2)
            f += m*(f_CoM[...,None,:] + np.cross(alpha[...,None,:], q))
        return f
    
    def get_coords(self):
        """Return the quasicentroid coordinates given the current system
        configuration.
        """
        
        qbead = dstrip(self.qbead)
        qbead_ba = self.qcart2ba(qbead)
        qbead_ba[...,2] = np.arccos(qbead_ba[...,2])
        q_ba = np.mean(qbead_ba, axis=0)
        # Triatomic molecule in X-Y plane with central atoms at (0,0),
        # first atom at (r1, 0) and second atom at r2*(cos(theta),sin(theta))
        q = np.zeros((qbead.shape[1:]))
        q[:,1,0] = q_ba[...,0]
        q[:,2,0] = q_ba[...,1]*np.cos(q_ba[...,2])
        q[:,2,1] = q_ba[...,1]*np.sin(q_ba[...,2])
        # Rotate to align with centroids as per Eckart conditions
        q0 = dstrip(self.cgp.qc).copy().reshape(q.shape)
        m = dstrip(self.m)
        # Shift centroids and quasicentroids to CoM
        q0[:], CoM = to_com(m, q0)
        q[:] = to_com(m, q)[0]
        # Rotate into the Eckart frame
        q[:] = eckrot(q, m, ref=q0)
        # Shift to centroid CoM
        q += CoM
        return q
    
    def get_targets(self):
        """Return the values of the target functions for the grouped
        constraints extracted from the quasicentroid configuration.
        """
        
        targetvals = np.zeros((self.cgp.ngp, self.cgp.ncons))
        q = dstrip(self.q)
        qba = self.qcart2ba(q)
        targetvals[:,:2] = qba[:,:2]
        targetvals[:,2] = np.arccos(qba[:,2])
        return targetvals
    
    
class NVEQuasiIntegrator(NVTIntegrator):
    
    """ Integrator object for constant-temperature simulation of quasicentroid
    dynamics.
    """
        
    def bind(self, motion):
        super(NVTIntegrator, self).bind(motion)
        self.quasi = motion.quasi
        self.cintegrator = motion.cdyn.integrator
    
    def pconstraints(self):
        """This removes the centre of mass contribution to the kinetic energy.

        Calculates the centre of mass momenta, then removes the mass weighted
        contribution from each atom. If the ensemble defines a thermostat, then
        the contribution to the conserved quantity due to this subtraction is
        added to the thermostat heat energy, as it is assumed that the centre of
        mass motion is due to the thermostat.

        If there is a choice of thermostats, the thermostat
        connected to the centroid is chosen.
        """

        if (self.fixcom):
            na3 = self.beads.natoms * 3
            p = dstrip(self.quasi.p)
            m = dstrip(self.quasi.m)
            M = self.beads[0].M
            dens = 0
            for i in range(3):
                pcom = p[i:na3:3].sum()
                dens += pcom**2
                pcom /= M
                self.quasi.p[i:na3:3] -= m * pcom
            self.ensemble.eens += dens * 0.5 / M
        if len(self.fixatoms) > 0:
            m = dstrip(self.quasi.m)
            bp = dstrip(self.quasi.p)
            for i in range(3):
                self.ensemble.eens += 0.5 * np.dot(
                    bp[3*self.fixatoms+i], 
                    bp[3*self.fixatoms+i]/ m[self.fixatoms])
            self.quasi.p[3*self.fixatoms+i] = 0.0
    
    def pstep(self, level=0):
        """Velocity Verlet momentum propagator."""
        self.quasi.p += self.quasi.forces_mts(level) * self.pdt[level]
        if level == 0:  # adds bias in the outer loop
            self.quasi.p += self.quasi.bias() * self.pdt[level]
        self.cintegrator.pstep(level)
        
    def tstep(self):
        """O-step in constrained dynamics only."""
        self.cintegrator.tstep()    
        
    def step_A(self):
        """Velocity Verlet position propagator with ring-polymer spring forces,
           followed by SHAKE.
        """
        self.quasi.q += ( dstrip(self.quasi.p) /
                          dstrip(self.quasi.m3) )*self.qdt
    
    def free_qstep_ba(self):
        """Override the exact normal mode propagator for the free ring-polymer
           with a sequence of RATTLE/SHAKE steps.
        """
        self.cintegrator.free_p()
        self.step_A()
        self.cintegrator.step_Ag()
            
    def free_qstep_ab(self):
        """Override the exact normal mode propagator for the free ring-polymer
           with a sequence of RATTLE/SHAKE steps.
        """    
        self.step_A()
        self.cintegrator.step_Ag()
        self.cintegrator.free_p()
        
class NVTQuasiIntegrator(NVEQuasiIntegrator):
    
    def tstep(self):
        super(NVTQuasiIntegrator, self).tstep()
        self.thermostat.step()