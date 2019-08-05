"""Contains the class that deals with quasi-centroids.

Deals with quasi-centroid dynamics, including the conversion between
quasi-centroid Cartesian and curviliner forces/coordinates, mass-scaling
of the underlying ring-polymer.

NOTE: this is currently written for the specific case of a bent triatomic,
with the storage convention (for e.g. water) O H H O H H ...
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2019 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np
from ipi.utils.depend import depend_value, depend_array, dd, dobject, dpipe, dstrip
from ipi.engine.thermostats import Thermostat
from ipi.engine.atoms import Atoms
from ipi.engine.constraints import ConstraintGroup
from ipi.utils.units import Constants

        
class QuasiCentroids(dobject):

    """Handles the ring-polymer quasi-centroids.

    Transformation to/from quasi-centroid coordinates,
    ring-polymer propagation under the quasi-centroid constraints, etc

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

    Methods:
#        scatter: scatter an array into disjoint sets grouped by
#                 quasi-centroid type
#        gather: gather the values from different quasi-centroid type

    """

    def __init__(self, nfree=1, dt=1.0, thermostat=None, qclist=[]):
        """Initialises a "QuasiCentroids" object. This contains a separate
           instance of a thermostat, to be applied to the quasi-centroids.

           Args:
               nfree (int): splitting of RP propagation under spring forces
               dt: the timestep of the simulation
               qclist: list of quasi-centroids grouped by type

        """

        dself = dd(self)
        if nfree < 1:
            raise ValueError("QuasiCentroids given negative free RP propagation splitting <nfree>")
        dself.nfree = depend_value(name="nfree", value=nfree)
        dself.dt = depend_value(name="dt", value=dt)
        dself.qdt = depend_value(name="qdt", func=(lambda: self.dt/self.nfree),
                                 dependencies=[dself.dt])
        if thermostat is None:
            self.thermostat = Thermostat()
        else:
            self.thermostat = thermostat
        self.qclist = qclist
        self._msg = ""
        natoms = 0
        indices = []
        # Cycle over the groups of constraints
        if len(self.qclist) != 0:
            for cgp in self.qclist:
                natoms += len(cgp.indices)
                indices += cgp.indices.tolist()
            # Check that all indices are hit
            indices.sort()
            if indices != list(range(natoms)):
                raise ValueError(
                        "ConstraintGroups failed to partition the quasi-centroids.")
        # Set up the arrays
        dself.natoms = depend_value(name="natoms", value=natoms)
        dself.names = depend_array(name="names",
                                   value=np.zeros(self.natoms, np.dtype('|S6')))
        # Atom masses
        dself.m = depend_array(name="m", value=np.zeros(self.natoms, float))
        dself.m3 = depend_array(name="m3", value=np.zeros((3 * self.natoms), float),
                                func=(lambda: np.ravel(np.ones((self.natoms,3)) *
                                                       dstrip(self.m)[:,None])),
                                dependencies=[dself.m])
        # Quasi-centroid positions and momenta
        dself.q = depend_array(name="q", value=np.nan*np.ones((3 * self.natoms), float))
        dself.p = depend_array(name="p", value=np.nan*np.ones((3 * self.natoms), float))
        # Access quasi-centroids as Atoms object
        self.atoms = Atoms(self.natoms, _prebind=(self.q, self.p, self.m, self.names))
        datoms = dd(self.atoms)
        # Kinetic energies of the quasi-centroids, and total kinetic stress tensor
        if len(self.qclist) != 0:
            dself.kin = depend_value(name="kin", func=(lambda: self.atoms.kin),
                                     dependencies=[datoms.kin,])
            dself.kstress = depend_array(name="kstress", value=np.zeros((3, 3), float),
                                         func=(lambda: dstrip(self.atoms.kstress)),
                                         dependencies=[datoms.kstress,])
        else:
            dself.kin = depend_value(name="kin", value=0.0)
            dself.kstress = depend_array(name="kstress", value=np.zeros((3, 3), float))


    def bind(self, ens, beads, nm, bforce, prng):
        """Binds ensemble, beads, normal modes, bead force, and
           quasi-centroids grouped by type to the main QuasiCentroids
           object.

           Args:
            ens: The ensemble object from which the temperature is taken.
            beads: The beads object from which the bead positions are taken.
            nm: A normal modes object used to do the normal modes transformation.
            bforce: The forcefield object from which the force and virial are
                taken.
            prng: The random number generator object which controls random number
                generation.

        """

        self.ensemble = ens
        self.beads = beads
        self.nm = nm
        self.bforce = bforce
        self.bbias = ens.bias
        self.prng = prng
        dself = dd(self)
        dthrm = dd(self.thermostat)
        dens = dd(self.ensemble)
        dbforce = dd(self.bforce)
        if self.natoms != beads.natoms:
            raise ValueError("Number of quasi-centroid and bead atoms is inconsistent.")
        # Make sure thermostat has correct temperature
        dpipe(dens.temp, dthrm.temp)
        self.thermostat.bind(beads=self, prng=prng)
        # Add quasicentroid thermostat conserved quantity to ensemble
        self.ensemble.add_econs(dthrm.ethermo)
        # Add quasicentroid kinetic energy to ensemble
        self.ensemble.add_econs(dself.kin)
        self.ensemble.add_xlkin(dself.kin)
        # Set up the constraint groups; unless already supplied, the quasi-centroid
        # positions are initialised here
        for cgp in self.qclist:
            dpipe(dself.qdt, dd(cgp).qdt)
            cgp.bind(self)
        # Initialise the quasicentroid momenta here, unless already supplied
        if np.any(np.isnan(dstrip(self.p))):
            self.p = self.prng.gvec(3*self.natoms) * np.sqrt(dstrip(self.m3) *
                                                             Constants.kb *
                                                             self.ensemble.temp)
        dself.f = depend_array(name="f", value=np.zeros(3*self.natoms), 
                               func=self.f_combine,
                               dependencies=[dbforce.f])
    
    def shake(self):
        """Cycle over constraint groups and enforce the holonomic constraints
           using SHAKE
        """
        
        for cgp in self.qclist:
            cgp.shake()
    
    def rattle(self):
        """Cycle over constraint groups and enforce the holonomic constraints
           onto the momenta using RATTLE
        """
        
        for cgp in self.qclist:
            cgp.rattle()
            
    def forces_mts(self, level):
        """ Fetches the forces associated with a given MTS level."""
        
        fk = np.zeros(3*self.natoms)
        bfk = self.bforce.forces_mts(level)
        for cgp in self.qclist:
            cgp.b2fc(bfk, fqc=fk)
        return fk
    
    def bias(self):
        """ Returns the bias force"""
        
        fk = np.zeros(3*self.natoms)
        bfk = dstrip(self.bbias.f)
        for cgp in self.qclist:
            cgp.b2fc(bfk, fqc=fk)
        return fk
    
    def f_combine(self):
        """Obtains the total force vector."""

        rf = np.zeros((3 * self.natoms), float)
        bf = dstrip(self.bforce.f)
        for cgp in self.qclist:
            cgp.b2fc(bf, fqc=rf)
        return rf