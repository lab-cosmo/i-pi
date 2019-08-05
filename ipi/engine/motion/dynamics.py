"""Contains the classes that deal with the different dynamics required in
different types of ensembles.

Holds the algorithms required for normal mode propagators, and the objects to
do the constant temperature and pressure algorithms. Also calculates the
appropriate conserved energy quantity for the ensemble of choice.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import time

import numpy as np

from ipi.engine.motion import Motion
from ipi.utils import mathtools
from ipi.utils import nmtransform
from ipi.utils.depend import *
from ipi.engine.thermostats import Thermostat
from ipi.engine.barostats import Barostat
from ipi.engine.quasicentroids import QuasiCentroids
from ipi.utils.softexit import softexit


#__all__ = ['Dynamics', 'NVEIntegrator', 'NVTIntegrator', 'NPTIntegrator', 'NSTIntegrator', 'SCIntegrator`']

class Dynamics(Motion):

    """self (path integral) molecular dynamics class.

    Gives the standard methods and attributes needed in all the
    dynamics classes.

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
                 quasicentroids=None, fixcom=False, 
                 fixatoms=None, nmts=None):
        """Initialises a "dynamics" motion object.

        Args:
            dt: The timestep of the simulation algorithms.
            fixcom: An optional boolean which decides whether the centre of mass
                motion will be constrained or not. Defaults to False.
        """

        super(Dynamics, self).__init__(fixcom=fixcom, fixatoms=fixatoms)
        dself = dd(self)

        # initialize time step. this is the master time step that covers a full time step
        dd(self).dt = depend_value(name='dt', value=timestep)

        if thermostat is None:
            self.thermostat = Thermostat()
        else:
            self.thermostat = thermostat

        if quasicentroids is None:
            self.quasicentroids = QuasiCentroids()
        else:
            self.quasicentroids = quasicentroids

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
            self.integrator = NVEIntegrator()
        elif self.enstype == "nvt":
            self.integrator = NVTIntegrator()
        elif self.enstype == "npt":
            self.integrator = NPTIntegrator()
        elif self.enstype == "nst":
            self.integrator = NSTIntegrator()
        elif self.enstype == "sc":
            self.integrator = SCIntegrator()
        elif self.enstype == "scnpt":
            self.integrator = SCNPTIntegrator()
        elif self.enstype == "qcmd":
            self.integrator = QCMDIntegrator()
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

    def bind(self, ens, beads, nm, cell, bforce, prng, omaker):
        """Binds ensemble beads, cell, bforce, and prng to the dynamics.

        This takes a beads object, a cell object, a forcefield object and a
        random number generator object and makes them members of the ensemble.
        It also then creates the objects that will hold the data needed in the
        ensemble algorithms and the dependency network. Note that the conserved
        quantity is defined in the init, but as each ensemble has a different
        conserved quantity the dependencies are defined in bind.

        Args:
            beads: The beads object from whcih the bead positions are taken.
            nm: A normal modes object used to do the normal modes transformation.
            cell: The cell object from which the system box is taken.
            bforce: The forcefield object from which the force and virial are
                taken.
            prng: The random number generator object which controls random number
                generation.
        """

        super(Dynamics, self).bind(ens, beads, nm, cell, bforce, prng, omaker)
        
        # Bind the quasicentroids to the beads
        if hasattr(self.beads, "quasicentroids"):
            raise AttributeError("Conflicting definitions of bead quasicentroids")
        self.beads.quasicentroids = self.quasicentroids

        # Checks if the number of mts levels is equal to the dimensionality of the mts weights.
        if (len(self.nmts) != self.forces.nmtslevels):
            raise ValueError("The number of mts levels for the integrator does not agree with the mts_weights of the force components.")

        # Strips off depend machinery for easier referencing.
        dself = dd(self)
        dthrm = dd(self.thermostat)
        dbaro = dd(self.barostat)
        dnm = dd(self.nm)
        dens = dd(self.ensemble)

        # n times the temperature (for path integral partition function)
        dself.ntemp = depend_value(name='ntemp', func=self.get_ntemp, dependencies=[dens.temp])

        # fixed degrees of freedom count
        fixdof = len(self.fixatoms) * 3 * self.beads.nbeads
        if self.fixcom:
            fixdof += 3

        # first makes sure that the thermostat has the correct temperature and timestep, then proceeds with binding it.
        dpipe(dself.ntemp, dthrm.temp)

        # depending on the kind, the thermostat might work in the normal mode or the bead representation.
        self.thermostat.bind(beads=self.beads, nm=self.nm, prng=prng, fixdof=fixdof)

        # first makes sure that the barostat has the correct stress andf timestep, then proceeds with binding it.
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

        #!TODO THOROUGH CLEAN-UP AND CHECK
        if (self.enstype == "nvt" or self.enstype == "npt" or
            self.enstype == "nst" or self.enstype == "qcmd"):
            if self.ensemble.temp < 0:
                raise ValueError("Negative or unspecified temperature for a constant-T integrator")
            if self.enstype == "npt":
                if type(self.barostat) is Barostat:
                    raise ValueError("The barostat and its mode have to be specified for constant-p integrators")
                if self.ensemble.pext < 0:
                    raise ValueError("Negative or unspecified pressure for a constant-p integrator")
            elif self.enstype == "nst":
                if np.trace(self.ensemble.stressext) < 0:
                    raise ValueError("Negative or unspecified stress for a constant-s integrator")

    def get_ntemp(self):
        """Returns the PI simulation temperature (P times the physical T)."""

        return self.ensemble.temp * self.beads.nbeads

    def step(self, step=None):
        """ Advances the dynamics by one time step """

        self.integrator.step(step)
        self.ensemble.time += self.dt # increments internal time


class DummyIntegrator(dobject):
    """ No-op integrator for (PI)MD """

    def __init__(self):
        pass

    def get_qdt(self):
        return self.dt * 0.5 / self.inmts

    def get_pdt(self):
        dtl = 1.0 / self.nmts
        for i in xrange(1, len(dtl)):
            dtl[i] *= dtl[i - 1]
        dtl *= self.dt * 0.5
        return dtl

    def get_tdt(self):
        if self.splitting == "obabo":
            return self.dt * 0.5
        elif self.splitting == "baoab":
            return self.dt
        else:
            raise ValueError("Invalid splitting requested. Only OBABO and BAOAB are supported.")

    def bind(self, motion):
        """ Reference all the variables for simpler access."""

        self.beads = motion.beads
        self.bias = motion.ensemble.bias
        self.ensemble = motion.ensemble
        self.forces = motion.forces
        self.prng = motion.prng
        self.nm = motion.nm
        self.thermostat = motion.thermostat
        self.barostat = motion.barostat
        self.quasicentroids = motion.quasicentroids
        self.fixcom = motion.fixcom
        self.fixatoms = motion.fixatoms
        self.enstype = motion.enstype

        dself = dd(self)
        dmotion = dd(motion)
        dbeads = dd(self.beads)
        dquasi = dd(self.quasicentroids)

        # no need to dpipe these are really just references
        dself.splitting = dmotion.splitting
        dself.dt = dmotion.dt
        dself.nmts = dmotion.nmts

        # total number of iteration in the inner-most MTS loop
        dself.inmts = depend_value(name="inmts", func=lambda: np.prod(self.nmts))
        dself.nmtslevels = depend_value(name="nmtslevels", func=lambda: len(self.nmts))
        # these are the time steps to be used for the different parts of the integrator
        dself.qdt = depend_value(name="qdt", func=self.get_qdt, dependencies=[dself.splitting, dself.dt, dself.inmts])  # positions
        dself.pdt = depend_array(name="pdt", func=self.get_pdt, value=np.zeros(len(self.nmts)), dependencies=[dself.splitting, dself.dt, dself.nmts])  # momenta
        dself.tdt = depend_value(name="tdt", func=self.get_tdt, dependencies=[dself.splitting, dself.dt, dself.nmts])  # thermostat

        dpipe(dself.qdt, dd(self.nm).dt)
        dpipe(dself.dt, dd(self.barostat).dt)
        dpipe(dself.qdt, dd(self.barostat).qdt)
        dpipe(dself.pdt, dd(self.barostat).pdt)
        dpipe(dself.tdt, dd(self.barostat).tdt)
        dpipe(dself.tdt, dd(self.thermostat).dt)

        if motion.enstype == "sc" or motion.enstype == "scnpt":
            # coefficients to get the (baseline) trotter to sc conversion
            self.coeffsc = np.ones((self.beads.nbeads, 3 * self.beads.natoms), float)
            self.coeffsc[::2] /= -3.
            self.coeffsc[1::2] /= 3.
        if motion.enstype == "qcmd":
            dpipe(dbeads.names, dquasi.names)
            dpipe(dbeads.m, dquasi.m)
            dpipe(dself.qdt, dquasi.dt)
            dpipe(dself.tdt, dd(self.quasicentroids.thermostat).dt)
            self.quasicentroids.bind(self.ensemble, self.beads,
                                     self.nm, self.forces,
                                     self.prng)

    def pstep(self):
        """Dummy momenta propagator which does nothing."""
        pass

    def qcstep(self):
        """Dummy centroid position propagator which does nothing."""
        pass

    def step(self, step=None):
        """Dummy simulation time step which does nothing."""
        pass

    def pconstraints(self):
        """Dummy centroid momentum step which does nothing."""
        pass


class NVEIntegrator(DummyIntegrator):

    """ Integrator object for constant energy simulations.

    Has the relevant conserved quantity and normal mode propagator for the
    constant energy ensemble. Note that a temperature of some kind must be
    defined so that the spring potential can be calculated.

    Attributes:
        ptime: The time taken in updating the velocities.
        qtime: The time taken in updating the positions.
        ttime: The time taken in applying the thermostat steps.

    Depend objects:
        econs: Conserved energy quantity. Depends on the bead kinetic and
            potential energy, and the spring potential energy.
    """

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
            nb = self.beads.nbeads
            p = dstrip(self.beads.p)
            m = dstrip(self.beads.m3)[:, 0:na3:3]
            M = self.beads[0].M
            Mnb = M*nb

            dens = 0
            for i in range(3):
                pcom = p[:, i:na3:3].sum()
                dens += pcom**2
                pcom /= Mnb
                self.beads.p[:, i:na3:3] -= m * pcom

            self.ensemble.eens += dens * 0.5 / Mnb

        if len(self.fixatoms) > 0:
            for bp in self.beads.p:
                m = dstrip(self.beads.m)
                self.ensemble.eens += 0.5 * np.dot(bp[self.fixatoms * 3], bp[self.fixatoms * 3] / m[self.fixatoms])
                self.ensemble.eens += 0.5 * np.dot(bp[self.fixatoms * 3 + 1], bp[self.fixatoms * 3 + 1] / m[self.fixatoms])
                self.ensemble.eens += 0.5 * np.dot(bp[self.fixatoms * 3 + 2], bp[self.fixatoms * 3 + 2] / m[self.fixatoms])
                bp[self.fixatoms * 3] = 0.0
                bp[self.fixatoms * 3 + 1] = 0.0
                bp[self.fixatoms * 3 + 2] = 0.0

    def pstep(self, level=0):
        """Velocity Verlet monemtum propagator."""

        # halfdt/alpha
        self.beads.p += self.forces.forces_mts(level) * self.pdt[level]
        if level == 0:  # adds bias in the outer loop
            self.beads.p += dstrip(self.bias.f) * self.pdt[level]

    def qcstep(self):
        """Velocity Verlet centroid position propagator."""
        # dt/inmts
        self.nm.qnm[0, :] += dstrip(self.nm.pnm)[0, :] / dstrip(self.beads.m3)[0] * self.qdt

    def free_qstep_ba(self):
        """Exact normal mode propagator for the free ring polymer, which combines
           propagation of the centroid using Velocity Verlet with a call to
           the exact propagator for non-centroid modes.
        """
        self.qcstep()
        self.nm.free_qstep()

    def free_qstep_ab(self):
        self.free_qstep_ba()

    # now the idea is that for BAOAB the MTS should work as follows:
    # take the BAB MTS, and insert the O in the very middle. This might imply breaking a A step in two, e.g. one could have
    # Bbabb(a/2) O (a/2)bbabB
    def mtsprop_ba(self, index):
        """ Recursive MTS step """

        mk = int(self.nmts[index] / 2)

        for i in range(mk):  # do nmts/2 full sub-steps

            self.pstep(index)
            self.pconstraints()
            if index == self.nmtslevels - 1:
                # call Q propagation for dt/alpha at the inner step
                self.free_qstep_ba()
                self.free_qstep_ab()
            else:
                self.mtsprop(index + 1)

            self.pstep(index)
            self.pconstraints()

        if self.nmts[index] % 2 == 1:
            # propagate p for dt/2alpha with force at level index
            self.pstep(index)
            self.pconstraints()
            if index == self.nmtslevels - 1:
                # call Q propagation for dt/alpha at the inner step
                self.free_qstep_ba()
            else:
                self.mtsprop_ba(index + 1)

    def mtsprop_ab(self, index):
        """ Recursive MTS step """

        if self.nmts[index] % 2 == 1:
            if index == self.nmtslevels - 1:
                # call Q propagation for dt/alpha at the inner step
                self.free_qstep_ab()
            else:
                self.mtsprop_ab(index + 1)

            # propagate p for dt/2alpha with force at level index
            self.pstep(index)
            self.pconstraints()

        for i in range(int(self.nmts[index] / 2)):  # do nmts/2 full sub-steps
            self.pstep(index)
            self.pconstraints()
            if index == self.nmtslevels - 1:
                # call Q propagation for dt/alpha at the inner step
                self.free_qstep_ba()
                self.free_qstep_ab()
            else:
                self.mtsprop(index + 1)

            self.pstep(index)
            self.pconstraints()

    def mtsprop(self, index):
        # just calls the two pieces together
        self.mtsprop_ba(index)
        self.mtsprop_ab(index)

    def step(self, step=None):
        """Does one simulation time step."""

        self.mtsprop(0)


class NVTIntegrator(NVEIntegrator):

    """Integrator object for constant temperature simulations.

    Has the relevant conserved quantity and normal mode propagator for the
    constant temperature ensemble. Contains a thermostat object containing the
    algorithms to keep the temperature constant.

    Attributes:
        thermostat: A thermostat object to keep the temperature constant.
    """

    def tstep(self):
        """Velocity Verlet thermostat step"""

        self.thermostat.step()

    def step(self, step=None):
        """Does one simulation time step."""

        if self.splitting == "obabo":
            # thermostat is applied for dt/2
            self.tstep()
            self.pconstraints()

            # forces are integerated for dt with MTS.
            self.mtsprop(0)

            # thermostat is applied for dt/2
            self.tstep()
            self.pconstraints()

        elif self.splitting == "baoab":

            self.mtsprop_ba(0)
            # thermostat is applied for dt
            self.tstep()
            self.pconstraints()
            self.mtsprop_ab(0)


class NPTIntegrator(NVTIntegrator):

    """Integrator object for constant pressure simulations.

    Has the relevant conserved quantity and normal mode propagator for the
    constant pressure ensemble. Contains a thermostat object containing the
    algorithms to keep the temperature constant, and a barostat to keep the
    pressure constant.
    """

    # should be enough to redefine these functions, and the step() from NVTIntegrator should do the trick
    def pstep(self, level=0):
        """Velocity Verlet monemtum propagator."""

        self.barostat.pstep(level)
        super(NPTIntegrator, self).pstep(level)
        #self.pconstraints()

    def qcstep(self):
        """Velocity Verlet centroid position propagator."""

        self.barostat.qcstep()

    def tstep(self):
        """Velocity Verlet thermostat step"""

        self.thermostat.step()
        self.barostat.thermostat.step()
        #self.pconstraints()


class NSTIntegrator(NPTIntegrator):

    """Ensemble object for constant pressure simulations.

    Has the relevant conserved quantity and normal mode propagator for the
    constant pressure ensemble. Contains a thermostat object containing the
    algorithms to keep the temperature constant, and a barostat to keep the
    pressure constant.

    Attributes:
    barostat: A barostat object to keep the pressure constant.

    Depend objects:
    econs: Conserved energy quantity. Depends on the bead and cell kinetic
    and potential energy, the spring potential energy, the heat
    transferred to the beads and cell thermostat, the temperature and
    the cell volume.
    pext: External pressure.
    """


class SCIntegrator(NVTIntegrator):
    """Integrator object for constant temperature simulations.

    Has the relevant conserved quantity and normal mode propagator for the
    constant temperature ensemble. Contains a thermostat object containing the
    algorithms to keep the temperature constant.

    Attributes:
        thermostat: A thermostat object to keep the temperature constant.

    Depend objects:
        econs: Conserved energy quantity. Depends on the bead kinetic and
            potential energy, the spring potential energy and the heat
            transferred to the thermostat.
    """

    def bind(self, mover):
        """Binds ensemble beads, cell, bforce, bbias and prng to the dynamics.

        This takes a beads object, a cell object, a forcefield object and a
        random number generator object and makes them members of the ensemble.
        It also then creates the objects that will hold the data needed in the
        ensemble algorithms and the dependency network. Note that the conserved
        quantity is defined in the init, but as each ensemble has a different
        conserved quantity the dependencies are defined in bind.

        Args:
        beads: The beads object from whcih the bead positions are taken.
        nm: A normal modes object used to do the normal modes transformation.
        cell: The cell object from which the system box is taken.
        bforce: The forcefield object from which the force and virial are
            taken.
        prng: The random number generator object which controls random number
            generation.
        """

        super(SCIntegrator, self).bind(mover)
        self.ensemble.add_econs(dd(self.forces).potsc)
        self.ensemble.add_xlpot(dd(self.forces).potsc)

    def pstep(self, level=0):
        """Velocity Verlet monemtum propagator."""

        if level == 0:
            # bias goes in the outer loop
            self.beads.p += dstrip(self.bias.f) * self.pdt[level]
        # just integrate the Trotter force scaled with the SC coefficients, which is a cheap approx to the SC force
        self.beads.p += self.forces.forces_mts(level) * (1.0 + self.forces.coeffsc_part_1) * self.pdt[level]

    def step(self, step=None):

        # the |f|^2 term is considered to be slowest (for large enough P) and is integrated outside everything.
        # if nmts is not specified, this is just the same as doing the full SC integration

        if self.splitting == "obabo":
            # thermostat is applied for dt/2
            self.tstep()
            self.pconstraints()

            # forces are integerated for dt with MTS.
            self.beads.p += dstrip(self.forces.fsc_part_2) * self.dt * 0.5
            self.mtsprop(0)
            self.beads.p += dstrip(self.forces.fsc_part_2) * self.dt * 0.5

            # thermostat is applied for dt/2
            self.tstep()
            self.pconstraints()

        elif self.splitting == "baoab":

            self.beads.p += dstrip(self.forces.fsc_part_2) * self.dt * 0.5
            self.mtsprop_ba(0)
            # thermostat is applied for dt
            self.tstep()
            self.pconstraints()
            self.mtsprop_ab(0)
            self.beads.p += dstrip(self.forces.fsc_part_2) * self.dt * 0.5


class SCNPTIntegrator(SCIntegrator):
    """Integrator object for constant pressure Suzuki-Chin simulations.

    Has the relevant conserved quantity and normal mode propagator for the
    constant pressure ensemble. Contains a thermostat object containing the
    algorithms to keep the temperature constant, and a barostat to keep the
    pressure constant.
    """

    # should be enough to redefine these functions, and the step() from NVTIntegrator should do the trick
    def pstep(self, level=0):
        """Velocity Verlet monemtum propagator."""

        self.barostat.pstep(level)
        super(SCNPTIntegrator, self).pstep(level)

    def qcstep(self):
        """Velocity Verlet centroid position propagator."""

        self.barostat.qcstep()

    def tstep(self):
        """Velocity Verlet thermostat step"""

        self.thermostat.step()
        self.barostat.thermostat.step()

    def step(self, step=None):

        # the |f|^2 term is considered to be slowest (for large enough P) and is integrated outside everything.
        # if nmts is not specified, this is just the same as doing the full SC integration

        if self.splitting == "obabo":
            # thermostat is applied for dt/2
            self.tstep()
            self.pconstraints()

            # forces are integerated for dt with MTS.
            self.barostat.pscstep()
            self.beads.p += dstrip(self.forces.fsc_part_2) * self.dt * 0.5
            self.mtsprop(0)
            self.barostat.pscstep()
            self.beads.p += dstrip(self.forces.fsc_part_2) * self.dt * 0.5

            # thermostat is applied for dt/2
            self.tstep()
            self.pconstraints()

        elif self.splitting == "baoab":

            self.barostat.pscstep()
            self.beads.p += dstrip(self.forces.fsc_part_2) * self.dt * 0.5
            self.mtsprop_ba(0)
            # thermostat is applied for dt
            self.tstep()
            self.pconstraints()
            self.mtsprop_ab(0)
            self.barostat.pscstep()
            self.beads.p += dstrip(self.forces.fsc_part_2) * self.dt * 0.5

class QCMDIntegrator(NVTIntegrator):
    """Integrator for a constant temperature simulation of water subject
       to quasi-centroid constraints.

       This is an early implementation that hard-codes a range of parameters.
       Eventually this is to be combined with a proper, general constrained
       propagator for ring-polymers and extended to systems beyond bent
       triatomics.

       Attributes:
           replica_list: list of Replicas objects
           clist: list of HolonomicConstraint objects

    """
    
    def get_qdt(self):
        return self.dt * 0.5 / (self.inmts * self.quasicentroids.nfree)

    def bind(self, motion):
        """ Reference all the variables for simpler access and initialise
            local storage of coordinates and associated constraints.

            Note: this assumes that atoms belonging to the same bent triatomic
                  are stored consecutively, and that the central atom is at the
                  beginning of each group of three.
        """
        super(QCMDIntegrator, self).bind(motion)
        self._ethermo = False
        self._msg = ""

    def qconstraints(self):
        """This applies SHAKE to the positions and momenta.

        Args:
           dt: integration time-step for SHAKE/RATTLE
        """    
        self.quasicentroids.shake()
    
    def pconstraints(self):
        """This applies RATTLE to the momenta and returns the change in the
        total kinetic energy that arises from applying the constraint. This
        is used to modify the conserved quantity externally, as it cannot
        always be assumed that the motion perpendicular to the constraint
        isosurface is due to the thermostat.

        The propagator raises an error if the centre-of-mass of the
        cell or any atoms are fixed.
        """
        
        # Apply CoM constraint to quasi-centroids
        if (self.fixcom):
            na3 = self.quasicentroids.natoms * 3
            p = dstrip(self.quasicentroids.p)
            m = dstrip(self.quasicentroids.m3)[0:na3:3]
            M = self.quasicentroids.atoms.M

            dens = 0
            for i in range(3):
                pcom = p[i:na3:3].sum()
                dens += pcom**2
                pcom /= M
                self.quasicentroids.p[i:na3:3] -= m * pcom

            self.ensemble.eens += dens * 0.5 / M
  
        # Fix the select quasi-centroids 
        if len(self.fixatoms) > 0:
            m = dstrip(self.quasicentroids.m)
            bp = self.quasicentroids.p
            self.ensemble.eens += 0.5 * np.dot(bp[self.fixatoms * 3], 
                                               bp[self.fixatoms * 3] / 
                                               m[self.fixatoms])
            self.ensemble.eens += 0.5 * np.dot(bp[self.fixatoms * 3 + 1], 
                                               bp[self.fixatoms * 3 + 1] / 
                                               m[self.fixatoms])
            self.ensemble.eens += 0.5 * np.dot(bp[self.fixatoms * 3 + 2], 
                                               bp[self.fixatoms * 3 + 2] / 
                                               m[self.fixatoms])
            bp[self.fixatoms * 3] = 0.0
            bp[self.fixatoms * 3 + 1] = 0.0
            bp[self.fixatoms * 3 + 2] = 0.0
        
        # Apply the momentum constraint to the underlying beads
        if self._ethermo:
            p = dstrip(self.nm.pnm)
            m = dstrip(self.nm.dynm3)
            self.ensemble.eens += 0.5*np.sum(p**2/m)
        self.quasicentroids.rattle()    
        if self._ethermo:
            p = dstrip(self.nm.pnm)
            self.ensemble.eens -= 0.5*np.sum(p**2/m)
            self._ethermo = False
        return
    
    def pstep(self, level=0):
        """Velocity Verlet monemtum propagator."""

        """
        # halfdt/alpha
        self.beads.p += self.forces.forces_mts(level) * self.pdt[level]
        if level == 0:  # adds bias in the outer loop
            self.beads.p += dstrip(self.bias.f) * self.pdt[level]
        """
        
        self.quasicentroids.p += self.quasicentroids.forces_mts(level) * self.pdt[level]
        if level == 0:  # adds bias in the outer loop
            self.quasicentroids.p += self.quasicentroids.bias() * self.pdt[level]
        super(QCMDIntegrator, self).pstep()

    def free_p(self):
        """Velocity Verlet momentum propagator with ring-polymer spring forces,
           followed by RATTLE.
        """
        
        # NOTE: no spring contributions for quasi-centroids
        self.nm.pnm += dstrip(self.nm.fspringnm)*self.qdt
        self.pconstraints() # RATTLE

    def free_q(self):
        """Velocity Verlet position propagator with ring-polymer spring forces,
           followed by SHAKE.
        """
        
        # Move quasi-centroids first
        self.quasicentroids.q += (dstrip(self.quasicentroids.p) /
                                  dstrip(self.quasicentroids.m3))*self.qdt
        # Then the beads
        self.nm.qnm += dstrip(self.nm.pnm)/dstrip(self.nm.dynm3)*self.qdt
        self.qconstraints()
        self.pconstraints()

    def tstep(self):
        """Velocity Verlet thermostat step, followed by RATTLE. This also
        calculates the kinetic energy removed after enforcing the constraint
        and modifies the contribution to the constrained quantity due to this
        subtraction
        """

        self._ethermo = True
        self.quasicentroids.thermostat.step()
        super(QCMDIntegrator,self).tstep()

    def free_qstep_ba(self):
        """Override the exact normal mode propagator for the free ring-polymer
           with a sequence of RATTLE/SHAKE steps.
        """
        
        for i in range(self.quasicentroids.nfree//2):
            
            self.free_p() # B
            self.free_q() # A
            self.free_q() # A
            self.free_p() # B

        if (self.quasicentroids.nfree%2 == 1):
            self.free_p() # B
            self.free_q() # A

    def free_qstep_ab(self):
        """Override the exact normal mode propagator for the free ring-polymer
           with a sequence of RATTLE/SHAKE steps.
        """
        
        if (self.quasicentroids.nfree%2 == 1):
            
            self.free_q() # A
            self.free_p() # B

        for i in range(self.quasicentroids.nfree//2):
            self.free_p() # B
            self.free_q() # A
            self.free_q() # A
            self.free_p() # B

    def step(self, step=None):
        """Does one simulation time step."""

        try:
            super(QCMDIntegrator, self).step(step)
        except:
            softexit.trigger(self._msg)
            raise