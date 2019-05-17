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
from ipi.engine.constraints import BondLength, BondAngle, Eckart, Constraints
from ipi.engine.thermostats import Thermostat
from ipi.engine.barostats import Barostat
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
                 thermostat=None, barostat=None, constraints=None,
                 fixcom=False, fixatoms=None, nmts=None):
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

        if constraints is None:
            self.constraints = Constraints()
        else:
            self.constraints = constraints

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
            self.integrator = QCMDWaterIntegrator()
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

            elif self.enstype == "qcmd":
                if self.constraints.maxcycle < 1:
                    raise ValueError("Negative or zero value of maximum number of iterations in RATTLE.")
                if self.constraints.tol < 0:
                    raise ValueError("Negative RATTLE convergence threshold.")

                if self.constraints.nfree < 1:
                    raise ValueError("Negative or zero value for splitting free constrained RP propagation")

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
        self.constraints = motion.constraints
        self.fixcom = motion.fixcom
        self.fixatoms = motion.fixatoms
        self.enstype = motion.enstype

        dself = dd(self)
        dmotion = dd(motion)

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

class QCMDWaterIntegrator(NVTIntegrator):
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

    def get_fpdt(self):
        return 0.5 * self.dt / (self.inmts * self.constraints.nfree)

    def get_fqdt(self):
        if self.splitting == "obabo":
            return self.dt / (self.inmts * self.constraints.nfree)
        elif self.splitting == "baoab":
            return 0.5 * self.dt / (self.inmts * self.constraints.nfree)
        else:
            raise ValueError("Invalid splitting requested. Only OBABO and BAOAB are supported.")

    def get_tdt(self):
        if self.splitting == "obabo":
            return self.dt * 0.5
        elif self.splitting == "baoab":
            return self.dt / (self.inmts * self.constraints.nfree)
        else:
            raise ValueError("Invalid splitting requested. Only OBABO and BAOAB are supported.")

    def bind(self, motion):
        """ Reference all the variables for simpler access and initialise
            local storage of coordinates and associated constraints.

            Note: this assumes that atoms belonging to the same bent triatomic
                  are stored consecutively, and that the central atom is at the
                  beginning of each group of three.
        """

        super(QCMDWaterIntegrator, self).bind(motion)
        dself = dd(self)
        dconst = dd(self.constraints)
        # Time-step for constrained free ring-polymer propagation (positions)
        dself.fqdt = depend_value(name="fqdt", func=self.get_fqdt,
                                  dependencies=[dself.splitting, dself.dt,
                                                dself.inmts, dconst.nfree])
        dself.fpdt = depend_value(name="fpdt", func=self.get_fpdt,
                                  dependencies=[dself.dt, dself.inmts,
                                                dconst.nfree])

        if (self.beads.natoms%3 != 0):
            raise ValueError("QCMDWaterIntegrator received a total "+
                             "of {:d} atoms. ".format(self.beads.natoms)+
                             "This is inconsistent with a set of triatomics.")
        # Create a workspace array with continuous storage of replicas corresponding
        # to the same degree of freedom
        self._temp = np.empty((self.beads.natoms//3,9,self.beads.nbeads))
        # Initialise the constraints
        self.clist = [BondLength(indices=[0,1]),
                      BondLength(indices=[0,2]),
                      BondAngle(indices=[0,1,2])]
        qc = dstrip(self.beads.qc[:]).reshape(self._temp.shape[:-1])
        mc = dstrip(self.beads.m3[0]).reshape(qc.shape)
        self.eckart = Eckart([], qref = qc, mref = mc)
        # Set up arrays for storing constraint targets and gradients
        self.targetvals = np.empty((len(self.clist),self.beads.natoms//3))
        self.grads = np.empty( (len(self.clist),)+self._temp.shape )
        self.mgrads = np.empty_like(self.grads)
        # Initialise the gradients
        self._temp[...] = np.reshape(dstrip(self.beads.q).T, self._temp.shape)
        for c, t, g, mg in zip(self.clist, self.targetvals,
                               self.grads, self.mgrads):
            t[:], g = c(self._temp, g)
            mg[...] = self._minv(g)
        self._ethermo = False
        self._msg = ""

    def _mtensor(self, arrin):
        """ Multiply an array by the mass tensor.

        Args:
            arrin .............. input 2d array, same size as beads.q
        """

        init_shape = arrin.shape
        wkspace = self.nm.transform.b2nm(
                arrin.reshape((-1, self.beads.nbeads)).T
                )
        wkspace *= dstrip(self.nm.dynm3)
        return np.reshape(self.nm.transform.nm2b(wkspace).T, init_shape)

    def _minv(self, arrin):
        """ Multiply an array by the inverse of the mass tensor.

        Args:
            arrin .............. input 2d array, same size as beads.q
        """

        init_shape = arrin.shape
        wkspace = self.nm.transform.b2nm(
                arrin.reshape((-1, self.beads.nbeads)).T
                )
        wkspace /= dstrip(self.nm.dynm3)
        return np.reshape(self.nm.transform.nm2b(wkspace).T, init_shape)

    def qconstraints(self):
        """This applies SHAKE to the positions and momenta.

        Args:
           dt: integration time-step for SHAKE/RATTLE
        """
        #!! TODO: remove this after debugging
        np.seterr(all='raise')
        # Copy the current ring-polymer configuration into workspace array
        self._temp[...] = np.reshape(dstrip(self.beads.q).T,self._temp.shape)
        q_init = self._temp.copy()
        # Initialise the Lagrange multipliers
        lambdas = np.zeros_like(self.targetvals)
        # Initialise arrays flagging convergences
        nc = np.ones(len(self._temp), dtype=np.bool) # "not converged"
        ns = np.ones((len(self.clist)+1, len(self._temp)), dtype=np.bool) # "not satisfied"
        # Initialise the current values of constraint fxns and grads
        sigmas = np.empty_like(ns, dtype=np.float)
        grads = np.empty_like(self.grads)
        qc_init = np.empty_like(self.eckart.qref)
        qc_fin = np.empty_like(qc_init)
        mtot = np.sum(self.eckart.mref[...,0:1], axis=-2)
        # Cycle over constraints until convergence
        ncycle = 1
        while True:
            if (ncycle > self.constraints.maxcycle):
                self._msg = "Maximum number of iterations exceeded in SHAKE."
                raise ValueError(self._msg)
            for i in range(len(self.clist)):
                try:
                    sigmas[i,nc], grads[i,nc,:,:] = \
                        self.clist[i](self._temp[nc,:,:], grads[i,nc,:,:])
                    sigmas[i,nc] -= self.targetvals[i,nc]
                    ns[i,:] = np.abs(sigmas[i]) > self.constraints.tol
                except:
                    igp = np.argwhere(np.isnan(sigmas[i]))[0,0]
                    self._msg = "SHAKE got invalid sigma at iter #{:d}".format(ncycle) +\
                          " for constraint #{:d}".format(i) + \
                          " in group #{:d}\n".format(igp) + \
                          " with target {:.16f}\n".format(self.targetvals[i,igp]) + \
                          " and configuration "+self._temp[igp].__repr__()
                    raise ValueError(self._msg)
                try:
                    dlambda = -sigmas[i,ns[i]] / np.sum(
                            grads[i,ns[i],:,:]*self.mgrads[i,ns[i],:,:],
                            axis=(-1,-2))
                    if np.any(np.isnan(dlambda)):
                        raise ValueError
                except:
                    igp = np.argwhere(np.isnan(dlambda))[0,0]
                    self._msg = "SHAKE got invalid dlambda at iter #{:d}".format(ncycle) +\
                          " for constraint #{:d}".format(i) + \
                          " with target {:.16f}\n".format(self.targetvals[i,igp]) + \
                          " and configuration "+self._temps[ns[i],...][igp].__repr__()
                    raise ValueError(self._msg)
                self._temp[ns[i],:,:] += dlambda[:,None,None] * \
                                         self.mgrads[i,ns[i],:,:]
            # Get the centoids
            qc_init[nc] = self._temp[nc].mean(axis=-1).reshape(qc_init[nc].shape)
            CoM = np.sum(
                    self.eckart.mref[nc] * qc_init[nc], axis=-2
                    )/mtot[nc]
            # Shift to CoM
            qc_fin[nc] = qc_init[nc]-CoM[:,None,:]
            # Calculate the Eckart product
            sigmas[-1,nc] = self.eckart(qc_fin, nc)
            ns[-1,:] = np.abs(sigmas[-1]) > self.constraints.tol
            # Rotate to Eckart frame
            qc_fin[ns[-1]] = mathtools.eckrot(
                    qc_fin[ns[-1]],
                    self.eckart.mref[ns[-1]],
                    self.eckart.qref_rel[ns[-1]]
                  )
            # Shift to CoM of reference
            qc_fin[nc] += self.eckart.qref_com[nc,None,:]
            # Calculate the change
            qc_fin[nc] -= qc_init[nc]
            self._temp[nc,:,:] += np.reshape(qc_fin,
                      (len(self._temp),-1))[nc,:,None]
            # Update the convergence status
            nc[:] = np.any(ns, axis=0) # not converged if any not satisfied
            if np.all(np.logical_not(nc)):
                # If all converged end cycle
                break
            ncycle += 1
        # Copy the coordinates back into beads
        self.beads.q[...] = self._temp.reshape((-1,self.beads.nbeads)).T
        # Update the momenta
        self._temp -= q_init # Change in positions
        self._temp[...] = self._mtensor(self._temp)/self.fqdt
        self.beads.p += self._temp.reshape((-1,self.beads.nbeads)).T
        # Update the constraint gradients
        self.grads[...] = grads
        for g, mg in zip(self.grads, self.mgrads):
            mg[...] = self._minv(g)

    def pconstraints(self):
        """This applies RATTLE to the momenta and returns the change in the
        total kinetic energy that arises from applying the constraint. This
        is used to modify the conserved quantity externally, as it cannot
        always be assumed that the motion perpendicular to the constraint
        isosurface is due to the thermostat.

        The propagator raises an error if the centre-of-mass of the
        cell or any atoms are fixed.
        """

        if len(self.fixatoms) > 0:
            raise ValueError("Cannot explicitly fix atoms in a constrained simulation")
        # Copy the current ring-polymer momenta into workspace array
        if self._ethermo:
            p = dstrip(self.nm.pnm)
            m = dstrip(self.nm.dynm3)
            kin_init = 0.5*np.sum(p**2/m, axis=-1)
        self._temp[...] = np.reshape(dstrip(self.beads.p).T,self._temp.shape)
        # Calculate the diagonal elements of the Jacobian matrix
        gmg = np.sum(self.grads*self.mgrads, axis=(-1,-2))
        # Initialise arrays flagging convergences
        nc = np.ones(len(self._temp), dtype=np.bool) # "not converged"
        ns = np.ones((len(self.clist)+1, len(self._temp)), dtype=np.bool) # "not satisfied"
        # Initialise the constraint time-derivatives
        sdots = np.empty_like(ns, dtype=np.float)
        pc_init = np.empty_like(self.eckart.qref)
        pc_fin = np.empty_like(pc_init)
        L = np.empty((len(pc_init),3))
        v = np.empty_like(L)
        mtot = np.sum(self.eckart.mref[...,0:1], axis=-2)
        qc = dstrip(self.beads.qc[:]).reshape(self.eckart.qref.shape)
        CoM = np.sum(self.eckart.mref * qc, axis=-2)/mtot
        qc = qc - CoM[:,None,:]
        # Cycle over constraints until convergence
        ncycle = 1
        while True:
            if (ncycle > self.constraints.maxcycle):
                #!! TODO: this currently quits mid-step, need to think of a better
                # way in future.
                raise ValueError("Maximum number of iterations exceeded in RATTLE.")
            for i in range(len(self.clist)):
                sdots[i,nc] = np.sum(self._temp[nc,:,:]*self.mgrads[i,nc,:,:],
                                      axis=(-1,-2))
                ns[i,:] = np.abs(sdots[i,:]) > self.constraints.tol
                dmu = -sdots[i,ns[i]] / gmg[i,ns[i]]
                self._temp[ns[i],:,:] += dmu[:,None,None] * \
                                         self.grads[i,ns[i],:,:]
            # Get centroid momenta
            pc_init[nc] = self._temp[nc].mean(axis=-1).reshape(pc_init[nc].shape)
            # Set CoM momentum to zero
            v[nc] = np.sum(pc_init[nc], axis=-2)/mtot[nc] # CoM velocity
            pc_fin[nc] = pc_init[nc] - self.eckart.mref[nc]*v[nc,None,:]
            # Calculate time-derivative or the rotational constraint
            L[nc,:] = np.cross(
                    self.eckart.qref_rel[nc],
                    pc_fin[nc],
                    axis=-1).sum(axis=-2)
            sdots[-1,nc] = np.sqrt(
                    np.sum(L[nc]**2, axis=-1)
                    )/mtot.reshape((-1,))[nc]
            ns[-1,:] = np.abs(sdots[-1,:]) > self.constraints.tol
            # Enforce the rotational constraint where not satisfied
            pc_fin[ns[-1]] = mathtools.eckspin(
                    pc_fin[ns[-1]], qc[ns[-1]],
                    self.eckart.mref[ns[-1]],
                    self.eckart.qref_rel[ns[-1]],
                    L[ns[-1]]
                    )
            # Calculate change in bead momenta
            pc_fin[nc] -= pc_init[nc]
            self._temp[nc,:,:] += np.reshape(pc_fin,
                      (len(self._temp),-1))[nc,:,None]
            # Update the convergence status
            nc[:] = np.any(ns, axis=0) # not converged if any not satisfied
            if np.all(np.logical_not(nc)):
                # If all converged end cycle
                break
            ncycle += 1
        # Copy the coordinates back into beads
        self.beads.p[...] = self._temp.reshape((-1,self.beads.nbeads)).T
        if self._ethermo:
            p = dstrip(self.nm.pnm)
            kin_fin = 0.5*np.sum(p**2/m, axis=-1)
            kin_fin -= kin_init
            for i,t in enumerate(self.thermostat._thermos):
                t.ethermo -= kin_fin[i]
            self._ethermo = False
        return

    def free_p(self):
        """Velocity Verlet momentum propagator with ring-polymer spring forces,
           followed by RATTLE.
        """
        # Note: m3*omegak**2 = dynm3*dynomegak**2
        self.nm.pnm[1:,:] -= (
                dstrip(self.nm.qnm)[1:,:] *
                dstrip(self.beads.m3)[1:,:] *
                dstrip(self.nm.omegak2)[1:,None])*self.fpdt
        self.pconstraints() # RATTLE

    def free_q(self):
        """Velocity Verlet position propagator with ring-polymer spring forces,
           followed by SHAKE.
        """
        self.nm.qnm += dstrip(self.nm.pnm)/dstrip(self.nm.dynm3)*self.fqdt
        self.qconstraints() # SHAKE
        self.pconstraints() #RATTLE

    def tstep(self):
        """Velocity Verlet thermostat step, followed by RATTLE. This also
        calculates the kinetic energy removed after enforcing the constraint
        and modifies the contribution to the constrained quantity due to this
        subtraction
        """

        self._ethermo = True
        super(QCMDWaterIntegrator,self).tstep()

    def free_qstep_ba(self):
        """Override the exact normal mode propagator for the free ring-polymer
           with a sequence of RATTLE/SHAKE steps.
        """
        for i in range(self.constraints.nfree//2):
            self.free_p() # B
            self.free_q() # A
            if self.splitting == "baoab":
                self.tstep() # O
                self.pconstraints()
                self.free_q() # A
            self.free_p() # B

        if (self.constraints.nfree%2 == 1):
            self.free_p()
            self.free_q()
            if self.splitting == "baoab":
                self.tstep()
                self.pconstraints()

    def free_qstep_ab(self):
        """Override the exact normal mode propagator for the free ring-polymer
           with a sequence of RATTLE/SHAKE steps.
        """

        if (self.constraints.nfree%2 == 1):
            if self.splitting == "baoab":
                self.free_q()
            self.free_p()

        for i in range(self.constraints.nfree//2):
            self.free_p()
            self.free_q()
            if self.splitting == "baoab":
                self.tstep()
                self.pconstraints()
                self.free_q()
            self.free_p()

    def step(self, step=None):
        """Does one simulation time step."""

        if self.splitting == "obabo":
            # thermostat is applied for dt/2
            self.tstep()
            self.pconstraints()

            # forces are integerated for dt with MTS.
            try:
                self.mtsprop(0)
            except:
                softexit.trigger(self._msg)
                raise

            # thermostat is applied for dt/2
            self.tstep()
            self.pconstraints()

        elif self.splitting == "baoab":

            try:
                self.mtsprop_ba(0)
            except:
                softexit.trigger(self._msg)
                raise
            try:
                self.mtsprop_ab(0)
            except:
                softexit.trigger(self._msg)
                raise

