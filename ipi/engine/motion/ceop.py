"""
Contains classes for different geometry optimization algorithms.

TODO

Algorithms implemented by Michele Ceriotti and Benjamin Helfrecht, 2015
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import numpy as np
import time

from ipi.engine.motion import Motion
from ipi.utils.depend import dstrip, dobject
from ipi.utils.softexit import softexit
from ipi.utils.mintools import min_brent, BFGS, BFGSTRM, L_BFGS
from ipi.utils.messages import verbosity, info


__all__ = ["CeopMotion"]


class CeopMotion(Motion):
    """Cell optimization class.

    Attributes:
        mode: minimization algorithm to use
        biggest_step: max allowed step size for BFGS/L-BFGS
        old_force: force on previous step
        old_direction: move direction on previous step
        invhessian_bfgs: stored inverse Hessian matrix for BFGS
        hessian_trm: stored  Hessian matrix for trm
        ls_options:
        {tolerance: energy tolerance for exiting minimization algorithm
        iter: maximum number of allowed iterations for minimization algorithm for each MD step
        step: initial step size for steepest descent and conjugate gradient
        adaptive: T/F adaptive step size for steepest descent and conjugate
                gradient}
        tolerances:
        {energy: change in energy tolerance for ending minimization
        force: force/change in force tolerance foe ending minimization
        position: change in position tolerance for ending minimization}
        corrections_lbfgs: number of corrections to be stored for L-BFGS
        scale_lbfgs: Scale choice for the initial hessian.
        qlist_lbfgs: list of previous positions (x_n+1 - x_n) for L-BFGS. Number of entries = corrections_lbfgs
        glist_lbfgs: list of previous gradients (g_n+1 - g_n) for L-BFGS. Number of entries = corrections_lbfgs
    """

    def __init__(
        self,
        fixcom=False,
        fixatoms=None,
        mode="bfgs",
        pressure=0.0,
        exit_on_convergence=True,
        biggest_step=100.0,
        old_pos=np.zeros(0, float),
        old_pot=np.zeros(0, float),
        old_force=np.zeros(0, float),
        old_direction=np.zeros(0, float),
        invhessian_bfgs=np.eye(0, 0, 0, float),
        hessian_trm=np.eye(0, 0, 0, float),
        tr_trm=np.zeros(0, float),
        ls_options={"tolerance": 1e-4, "iter": 100, "step": 1e-3, "adaptive": 1.0},
        tolerances={"energy": 1e-7, "force": 1e-4, "position": 1e-4},
        corrections_lbfgs=6,  # changed to 6 because it's 6 in inputs/motion/ceop.py, which overrides it anyways
        scale_lbfgs=1,
        qlist_lbfgs=np.zeros(0, float),
        glist_lbfgs=np.zeros(0, float),
    ):
        """Initialises CeopMotion.

        Args:
           fixcom: An optional boolean which decides whether the centre of mass
              motion will be constrained or not. Defaults to False.
        """
        # if len(fixatoms) > 0:
        #     raise ValueError("The optimization algorithm with fixatoms is not implemented. "
        #                      "We stop here. Comment this line and continue only if you know what you are doing")

        super(CeopMotion, self).__init__(fixcom=fixcom, fixatoms=fixatoms)

        # Optimization Options
        #print("CeopMotion.init.invhessian_bfgs",invhessian_bfgs)
        self.mode = mode
        self.pressure = pressure
        self.conv_exit = exit_on_convergence
        self.big_step = biggest_step
        self.tolerances = tolerances
        self.ls_options = ls_options

        #
        self.old_x = old_pos
        self.old_u = old_pot
        self.old_f = old_force
        self.d = old_direction
        #print("INIT.old_pos:", old_pos)

        # Classes for minimization routines and specific attributes
        if self.mode == "bfgs":
            self.invhessian = invhessian_bfgs
            self.optimizer = BFGSOptimizer()
        elif self.mode == "bfgstrm":
            self.tr = tr_trm
            self.hessian = hessian_trm
            self.optimizer = BFGSTRMOptimizer()
        elif self.mode == "lbfgs":
            self.corrections = corrections_lbfgs
            self.scale = scale_lbfgs
            self.qlist = qlist_lbfgs
            self.glist = glist_lbfgs
            self.optimizer = LBFGSOptimizer()
        elif self.mode == "sd":
            self.optimizer = SDOptimizer()
        elif self.mode == "cg":
            self.optimizer = CGOptimizer()
        else:
            self.optimizer = DummyOptimizer()

    def reset(self):  # necessary for Al6xxx-kmc
        # zeroes out all memory of previous steps
        self.old_x *= 0.0
        self.old_f *= 0.0
        self.old_u *= 0.0
        self.d *= 0.0

        if self.mode == "bfgs":
            self.invhessian[:] = np.eye(
                len(self.invhessian), len(self.invhessian), 0, float
            )
        # bfgstrm
        elif self.mode == "bfgstrm":
            self.hessian[:] = np.eye(len(self.hessian), len(self.hessian), 0, float)
            self.tr = self.initial_values["tr_trm"]
        # lbfgs
        elif self.mode == "lbfgst":
            self.corrections *= 0.0
            self.scale *= 0.0
            self.qlist *= 0.0
            self.glist *= 0.0

    def bind(self, ens, beads, nm, cell, bforce, prng, omaker):
        """Binds beads, cell, bforce and prng to CeopMotion

            Args:
            beads: The beads object from which the bead positions are taken.
            nm: A normal modes object used to do the normal modes transformation.
            cell: The cell object from which the system box is taken.
            bforce: The forcefield object from which the force and virial are taken.
            prng: The random number generator object which controls random number generation.
        """

        super(CeopMotion, self).bind(ens, beads, nm, cell, bforce, prng, omaker)
        # Binds optimizer
        self.optimizer.bind(self)

        if len(self.fixatoms) == len(self.beads[0]):
            softexit.trigger(
                "WARNING: all atoms are fixed, geometry won't change. Exiting simulation"
            )

    def step(self, step=None):
        if self.optimizer.converged:
            # if required, exit upon convergence. otherwise just return without action
            if self.conv_exit:
                softexit.trigger("Geometry optimization converged. Exiting simulation")
            else:
                info(
                    "Convergence threshold met. Will carry on but do nothing.",
                    verbosity.high,
                )
        else:
            self.optimizer.step(step)


class LineMapper(object):

    """Creation of the one-dimensional function that will be minimized.
    Used in steepest descent and conjugate gradient minimizers.

    Attributes:
        x0: initial position
        d: move direction
    """

    def __init__(self):
        self.x0 = self.d = None
        self.fcount = 0

    def bind(self, dumop):
        self.dbeads = dumop.beads.copy()
        self.dcell = dumop.cell.copy()
        self.dforces = dumop.forces.copy(self.dbeads, self.dcell)

        self.fixatoms_mask = np.ones(
            3 * dumop.beads.natoms, dtype=bool
        )  # Mask to exclude fixed atoms from 3N-arrays
        if len(dumop.fixatoms) > 0:
            self.fixatoms_mask[3 * dumop.fixatoms] = 0
            self.fixatoms_mask[3 * dumop.fixatoms + 1] = 0
            self.fixatoms_mask[3 * dumop.fixatoms + 2] = 0

    def set_dir(self, x0, mdir):
        self.x0 = x0.copy()

        # exclude fixed degrees of freedom and renormalize direction vector to unit length:
        tmp3 = mdir.copy()[:, self.fixatoms_mask]
        self.d = tmp3 / np.sqrt(np.dot(tmp3.flatten(), tmp3.flatten()))
        del tmp3
        if self.x0[:, self.fixatoms_mask].shape != self.d.shape:
            raise ValueError(
                "Incompatible shape of initial value and displacement direction"
            )

    def __call__(self, x):
        """ computes energy and gradient for optimization step
            determines new position (x0+d*x)"""

        self.fcount += 1
        self.dbeads.q[:, self.fixatoms_mask] = (
            self.x0[:, self.fixatoms_mask] + self.d * x
        )
        e = self.dforces.pot  # Energy
        g = -np.dot(
            dstrip(self.dforces.f[:, self.fixatoms_mask]).flatten(), self.d.flatten()
        )  # Gradient
        return e, g


class GradientMapper(object):

    """Creation of the multi-dimensional function that will be minimized.
    Used in the BFGS and L-BFGS minimizers.

    Attributes:
        dbeads:  copy of the bead object
        dcell:   copy of the cell object
        dforces: copy of the forces object
    """

    def __init__(self):
        self.fcount = 0
        pass

    def bind(self, dumop):
        self.dbeads = dumop.beads.copy()
        self.dcell = dumop.cell.copy()
        self.dforces = dumop.forces.copy(self.dbeads, self.dcell)
        self.pressure = dumop.pressure
        #print("GM.bind.cell:", dumop.cell)
        self.fixatoms_mask = np.ones(
            3 * dumop.beads.natoms, dtype=bool
        )  # Mask to exclude fixed atoms from 3N-arrays
        if len(dumop.fixatoms) > 0:
            self.fixatoms_mask[3 * dumop.fixatoms] = 0
            self.fixatoms_mask[3 * dumop.fixatoms + 1] = 0
            self.fixatoms_mask[3 * dumop.fixatoms + 2] = 0

    def __call__(self, x):
        """computes energy and gradient for optimization step"""
        self.fcount += 1
        self.dcell.h = np.dot((np.eye(3) + x[:, 0:9].reshape(3,3)), dstrip(self.dcell.h0))
        self.dbeads.q[:, self.fixatoms_mask] = self.dcell.get_absolute_positions(x[:, 9:])
        # Energy
        e = self.dforces.pot + self.pressure * self.dcell.V
        # Gradient
        g = np.zeros((self.dbeads.nbeads, 3 * self.dbeads.natoms + 9))
        g[:,0:9] = -np.dot((self.dforces.vir + np.eye(3) * self.pressure * self.dcell.V),  np.linalg.inv(np.eye(3) + dstrip(self.dcell.strain).T)).reshape((self.dbeads.nbeads, 9))
        g[:,9:] = -dstrip(self.dforces.f_scaled[:, self.fixatoms_mask])[0]
        #g[0:9] = -np.dot((self.dforces.vir + np.eye(3) * pV),invert_ut3x3( np.eye(3) + new_strain.reshape((3,3)).T)).flatten()
        #g[9:] = -self.dforces.forces_abs_to_scaled()
        #print("CALL.g:", g)
        return e, g


class DummyOptimizer(dobject):
    """ Dummy class for all optimization classes """

    def __init__(self):
        """initialises object for LineMapper (1-d function) and for GradientMapper (multi-dimensional function) """

        self.lm = LineMapper()
        self.gm = GradientMapper()
        self.converged = False

    def step(self, step=None):
        """Dummy simulation time step which does nothing."""
        pass

    def bind(self, ceop):
        """
        bind optimization options and call bind function of LineMapper and GradientMapper (get beads, cell,forces)
        check whether force size, direction size and inverse Hessian size from previous step match system size
        """
        #print("DUMMY.bind.cell:", ceop.cell)
        self.beads = ceop.beads
        self.cell = ceop.cell
        self.forces = ceop.forces
        self.fixcom = ceop.fixcom
        self.fixatoms = ceop.fixatoms
        self.pressure = ceop.pressure

        self.mode = ceop.mode
        self.tolerances = ceop.tolerances

        # Check for very tight tolerances

        if self.tolerances["position"] < 1e-7:
            raise ValueError(
                "The position tolerance is too small for any typical calculation. "
                "We stop here. Comment this line and continue only if you know what you are doing"
            )
        if self.tolerances["force"] < 1e-7:
            raise ValueError(
                "The force tolerance is too small for any typical calculation. "
                "We stop here. Comment this line and continue only if you know what you are doing"
            )
        if self.tolerances["energy"] < 1e-10:
            raise ValueError(
                "The energy tolerance is too small for any typical calculation. "
                "We stop here. Comment this line and continue only if you know what you are doing"
            )

        # The resize action must be done before the bind
        if ceop.old_x.size != self.beads.q.size + 9:
            if ceop.old_x.size == 0:
                ceop.old_x = np.zeros((self.beads.nbeads, 3 * self.beads.natoms + 9), float)
            else:
                raise ValueError(
                    "Conjugate gradient force size does not match system size"
                )
        if ceop.old_u.size != 1:
            if ceop.old_u.size == 0:
                ceop.old_u = np.zeros(1, float)
            else:
                raise ValueError(
                    "Conjugate gradient force size does not match system size"
                )
        if ceop.old_f.size != self.beads.q.size + 9:
            if ceop.old_f.size == 0:
                ceop.old_f = np.zeros((self.beads.nbeads, 3 * self.beads.natoms + 9), float)
            else:
                raise ValueError(
                    "Conjugate gradient force size does not match system size"
                )
        if ceop.d.size != self.beads.q.size + 9:
            if ceop.d.size == 0:
                ceop.d = np.zeros((self.beads.nbeads, 3 * self.beads.natoms + 9), float)
            else:
                raise ValueError(
                    "Conjugate gradient direction size does not match system size"
                )

        #print("BIND.old_x:", ceop.old_x)
        self.old_x = ceop.old_x
        self.old_u = ceop.old_u
        self.old_f = ceop.old_f
        self.d = ceop.d

    def exitstep(self, fx, u0, x):
        """ Exits the simulation step. Computes time, checks for convergence. """

        info(" @GEOP: Updating bead positions", verbosity.debug)
        self.qtime += time.time()

        if len(self.fixatoms) > 0:
            ftmp = self.forces.f.copy()
            for dqb in ftmp:
                dqb[self.fixatoms * 3] = 0.0
                dqb[self.fixatoms * 3 + 1] = 0.0
                dqb[self.fixatoms * 3 + 2] = 0.0
            fmax = np.amax(np.absolute(ftmp))
        else:
            fmax = np.amax(np.absolute(self.forces.f))

        e = np.absolute((fx - u0) / self.beads.natoms)
        info("@GEOP", verbosity.medium)
        self.tolerances["position"]
        info("   Current energy             %e" % (fx))
        info(
            "   Position displacement      %e  Tolerance %e"
            % (x, self.tolerances["position"]),
            verbosity.medium,
        )
        info(
            "   Max force component        %e  Tolerance %e"
            % (fmax, self.tolerances["force"]),
            verbosity.medium,
        )
        info(
            "   Energy difference per atom %e  Tolerance %e"
            % (e, self.tolerances["energy"]),
            verbosity.medium,
        )

        if np.linalg.norm(self.forces.f.flatten() - self.old_f[:, 9:].flatten()) <= 1e-20:
            info(
                "Something went wrong, the forces are not changing anymore."
                " This could be due to an overly small tolerance threshold "
                "that makes no physical sense. Please check if you are able "
                "to reach such accuracy with your force evaluation"
                " code (client)."
            )

        if (
            (np.absolute((fx - u0) / self.beads.natoms) <= self.tolerances["energy"])
            and (fmax <= self.tolerances["force"])
            and (x <= self.tolerances["position"])
        ):
            self.converged = True


class BFGSOptimizer(DummyOptimizer):
    """ BFGS Minimization """

    def bind(self, ceop):
        # call bind function from DummyOptimizer
        super(BFGSOptimizer, self).bind(ceop)

        if ceop.invhessian.size != (self.beads.q.size + 9)**2:
            if ceop.invhessian.size == 0:
                ceop.invhessian = np.eye(self.beads.q.size + 9, self.beads.q.size + 9, 0, float)
                ceop.invhessian[0:9,0:9] = np.eye(9) / self.cell.V**(1. / 3.) * 0.0
                #print("BFGSOptimizer.BIND.IF.invhessian:", ceop.invhessian)
                #ceop.invhessian[-3 * self.beads.natoms, -3 * self.beads.natoms] = np.kron(np.eye(self.beads.natoms), np.dot(self.cell.h0.T, self.cell.h0))
            else:
                raise ValueError("Inverse Hessian size does not match system size")

        self.invhessian = ceop.invhessian
        self.gm.bind(self)
        self.big_step = ceop.big_step
        self.ls_options = ceop.ls_options
        #print("BFGS.bind.inhessian",ceop.invhessian )

    def step(self, step=None):
        """ Does one simulation time step.
            Attributes:
            qtime: The time taken in updating the positions.
        """

        self.qtime = -time.time()
        info("\nMD STEP %d" % step, verbosity.debug)

        if step == 0:
            info(" @GEOP: Initializing BFGS", verbosity.debug)
            f = self.d * 0.0
            f[:,0:9] = -np.dot((self.forces.vir + np.eye(3) * self.pressure * self.cell.V),  np.linalg.inv(np.eye(3) + dstrip(self.cell.strain).T)).reshape((self.beads.nbeads, 9))
            #print("STEP=0.f[0:9]:", f[:,0:9])
            #print("STEP=0.forces.vir:", self.forces.vir)
            #print("STEP=0.pressure:", self.pressure)
            #print("STEP=0.cell.V:", self.cell.V)
            #print("STEP=0.cell.strain:", self.cell.strain)
            f[:,9:] = dstrip(self.forces.f_scaled)
            self.d += dstrip(f) / np.sqrt(
                np.dot(f.flatten(), f.flatten())
            )
            self.d = self.d * 0.001
            #print("STEP=0.invhessian:", self.invhessian)
            if len(self.fixatoms) > 0:
                for dqb in self.d:
                    dqb[self.fixatoms * 3] = 0.0
                    dqb[self.fixatoms * 3 + 1] = 0.0
                    dqb[self.fixatoms * 3 + 2] = 0.0
        #print("STEP.step:", step)
        #print("STEP.step.cell.strain:", self.cell.strain)
        #print("STEP.step.beads.q:",self.beads.q)
        self.old_x[:,0:9] = self.cell.strain.reshape((self.beads.nbeads, 9))
        self.old_x[:,9:] = self.cell.get_scaled_positions(self.beads.q)
        self.old_u[:] = self.forces.pot + self.pressure * self.cell.V
        self.old_f[:] = self.old_u * 0.0
        self.old_f[:,0:9] = -np.dot((self.forces.vir + np.eye(3) * self.pressure * self.cell.V),  np.linalg.inv(np.eye(3) + dstrip(self.cell.strain).T)).reshape((self.beads.nbeads, 9))
        self.old_f[:,9:] = dstrip(self.forces.f_scaled)

        fdf0 = (self.old_u, -self.old_f)

        # Do one iteration of BFGS
        # The invhessian and the directions are updated inside.
        BFGS(
            self.old_x,
            self.d,
            self.gm,
            fdf0,
            self.invhessian,
            self.big_step,
            self.ls_options["tolerance"] * self.tolerances["energy"],
            self.ls_options["iter"],
        )

        info("   Number of force calls: %d" % (self.gm.fcount))
        self.gm.fcount = 0
        # Update positions and forces
        self.beads.q = self.gm.dbeads.q
        self.cell.h = self.gm.dcell.h
        self.forces.transfer_forces(
            self.gm.dforces
        )  # This forces the update of the forces

        # Exit simulation step
        #print("STEP.old_x.shape:",self.old_x.shape)
        #print("STEP.old_x[:,:5]:",self.old_x[:,:5])
        #print("STEP.abs(old_x[:,9:]):",self.cell.get_absolute_positions(self.old_x[:,9:]))
        d_x_max = np.amax(np.absolute(np.subtract(self.beads.q, self.cell.get_absolute_positions(self.old_x[:,9:]))))
        self.exitstep(self.forces.pot, self.old_u, d_x_max)


class BFGSTRMOptimizer(DummyOptimizer):
    """ BFGSTRM Minimization with Trust Radius Method.  """

    def bind(self, ceop):
        # call bind function from DummyOptimizer
        super(BFGSTRMOptimizer, self).bind(ceop)

        if ceop.hessian.size != (self.beads.q.size * self.beads.q.size):
            if ceop.hessian.size == 0:
                ceop.hessian = np.eye(self.beads.q.size, self.beads.q.size, 0, float)
            else:
                raise ValueError("Hessian size does not match system size")

        self.hessian = ceop.hessian
        if ceop.tr.size == 0:
            ceop.tr = np.array([0.4])
        self.tr = ceop.tr
        self.gm.bind(self)
        self.big_step = ceop.big_step

    def step(self, step=None):
        """ Does one simulation time step.

            Attributes:
            qtime : The time taken in updating the real positions.
            tr    : current trust radius
        """

        self.qtime = -time.time()
        info("\nMD STEP %d" % step, verbosity.debug)

        if step == 0:
            info(" @GEOP: Initializing BFGSTRM", verbosity.debug)
        self.old_x[:] = self.beads.q
        self.old_u[:] = self.forces.pot
        self.old_f[:] = self.forces.f

        if len(self.fixatoms) > 0:
            for dqb in self.old_f:
                dqb[self.fixatoms * 3] = 0.0
                dqb[self.fixatoms * 3 + 1] = 0.0
                dqb[self.fixatoms * 3 + 2] = 0.0

            # Reduce dimensionality
            masked_old_x = self.old_x[:, self.gm.fixatoms_mask]
            masked_hessian = self.hessian[
                np.ix_(self.gm.fixatoms_mask, self.gm.fixatoms_mask)
            ]

            # Do one iteration of BFGSTRM.
            # The Hessian is updated inside. Everything is passed inside BFGSTRM() in masked form, including the Hessian
            BFGSTRM(
                masked_old_x,
                self.old_u,
                self.old_f[:, self.gm.fixatoms_mask],
                masked_hessian,
                self.tr,
                self.gm,
                self.big_step,
            )

            # Restore dimensionality of the hessian
            self.hessian[
                np.ix_(self.gm.fixatoms_mask, self.gm.fixatoms_mask)
            ] = masked_hessian
        else:
            # Make one step. ( A step is finished when a movement is accepted)
            BFGSTRM(
                self.old_x,
                self.old_u,
                self.old_f,
                self.hessian,
                self.tr,
                self.gm,
                self.big_step,
            )

        info("   Number of force calls: %d" % (self.gm.fcount))
        self.gm.fcount = 0
        # Update positions and forces
        self.beads.q = self.gm.dbeads.q
        self.forces.transfer_forces(
            self.gm.dforces
        )  # This forces the update of the forces

        # Exit simulation step
        d_x_max = np.amax(np.absolute(np.subtract(self.beads.q, self.old_x)))
        self.exitstep(self.forces.pot, self.old_u, d_x_max)


# ---------------------------------------------------------------------------------------


class LBFGSOptimizer(DummyOptimizer):
    """ L-BFGS Minimization: Note that the accuracy you can achieve with this method depends
        on how many ''corrections'' you store (default is 5). """

    def bind(self, ceop):
        # call bind function from DummyOptimizer
        super(LBFGSOptimizer, self).bind(ceop)

        self.corrections = ceop.corrections
        self.gm.bind(self)
        self.big_step = ceop.big_step
        self.ls_options = ceop.ls_options

        # if len(self.fixatoms) > 0:
        #     softexit.trigger("The L-BFGS optimization with fixatoms is implemented, but seems to be unstable. "
        #                      "We stop here. Comment this line and continue only if you know what you are doing.")

        if ceop.qlist.size != (self.corrections * self.beads.q.size):
            if ceop.qlist.size == 0:
                ceop.qlist = np.zeros((self.corrections, self.beads.q.size), float)
            else:
                raise ValueError("qlist size does not match system size")
        if ceop.glist.size != (self.corrections * self.beads.q.size):
            if ceop.glist.size == 0:
                ceop.glist = np.zeros((self.corrections, self.beads.q.size), float)
            else:
                raise ValueError("qlist size does not match system size")

        self.qlist = ceop.qlist
        self.glist = ceop.glist

        if ceop.scale not in [0, 1, 2]:
            raise ValueError("Scale option is not valid")

        self.scale = ceop.scale

    def step(self, step=None):
        """ Does one simulation time step
            Attributes:
            ttime: The time taken in applying the thermostat steps.
        """

        self.qtime = -time.time()

        info("\nMD STEP %d" % step, verbosity.debug)

        if step == 0:
            info(" @GEOP: Initializing L-BFGS", verbosity.debug)
            print(self.d)
            self.d += dstrip(self.forces.f) / np.sqrt(
                np.dot(self.forces.f.flatten(), self.forces.f.flatten())
            )

        self.old_x[:] = self.beads.q
        self.old_u[:] = self.forces.pot
        self.old_f[:] = self.forces.f

        if len(self.fixatoms) > 0:
            for dqb in self.old_f:
                dqb[self.fixatoms * 3] = 0.0
                dqb[self.fixatoms * 3 + 1] = 0.0
                dqb[self.fixatoms * 3 + 2] = 0.0

            # Reduce the dimensionality
            masked_old_x = self.old_x[:, self.gm.fixatoms_mask]
            masked_d = self.d[:, self.gm.fixatoms_mask]
            # self.gm is reduced inside its __init__() and __call__() functions
            masked_qlist = self.qlist[:, self.gm.fixatoms_mask]
            masked_glist = self.glist[:, self.gm.fixatoms_mask]
            fdf0 = (self.old_u, -self.old_f[:, self.gm.fixatoms_mask])

            # We update everything within L_BFGS (and all other calls).
            L_BFGS(
                masked_old_x,
                masked_d,
                self.gm,
                masked_qlist,
                masked_glist,
                fdf0,
                self.big_step,
                self.ls_options["tolerance"] * self.tolerances["energy"],
                self.ls_options["iter"],
                self.corrections,
                self.scale,
                step,
            )

            # Restore the dimensionality
            self.d[:, self.gm.fixatoms_mask] = masked_d
            self.qlist[:, self.gm.fixatoms_mask] = masked_qlist
            self.glist[:, self.gm.fixatoms_mask] = masked_glist

        else:
            fdf0 = (self.old_u, -self.old_f)

            # We update everything  within L_BFGS (and all other calls).
            L_BFGS(
                self.old_x,
                self.d,
                self.gm,
                self.qlist,
                self.glist,
                fdf0,
                self.big_step,
                self.ls_options["tolerance"] * self.tolerances["energy"],
                self.ls_options["iter"],
                self.corrections,
                self.scale,
                step,
            )

        info("   Number of force calls: %d" % (self.gm.fcount))
        self.gm.fcount = 0

        # Update positions and forces
        self.beads.q = self.gm.dbeads.q
        self.forces.transfer_forces(
            self.gm.dforces
        )  # This forces the update of the forces

        # Exit simulation step
        d_x_max = np.amax(np.absolute(np.subtract(self.beads.q, self.old_x)))
        self.exitstep(self.forces.pot, self.old_u, d_x_max)


class SDOptimizer(DummyOptimizer):
    """
    Steepest descent minimization
    dq1 = direction of steepest descent
    dq1_unit = unit vector of dq1
    """

    def bind(self, ceop):
        # call bind function from DummyOptimizer
        super(SDOptimizer, self).bind(ceop)
        self.lm.bind(self)
        self.ls_options = ceop.ls_options

    def step(self, step=None):
        """ Does one simulation time step
            Attributes:
            ttime: The time taken in applying the thermostat steps.
        """

        self.qtime = -time.time()
        info("\nMD STEP %d" % step, verbosity.debug)

        # Store previous forces for warning exit condition
        self.old_f[:] = self.forces.f

        # Check for fixatoms
        if len(self.fixatoms) > 0:
            for dqb in self.old_f:
                dqb[self.fixatoms * 3] = 0.0
                dqb[self.fixatoms * 3 + 1] = 0.0
                dqb[self.fixatoms * 3 + 2] = 0.0

        dq1 = dstrip(self.old_f)

        # Move direction for steepest descent
        dq1_unit = dq1 / np.sqrt(np.dot(dq1.flatten(), dq1.flatten()))
        info(" @GEOP: Determined SD direction", verbosity.debug)

        # Set position and direction inside the mapper
        self.lm.set_dir(dstrip(self.beads.q), dq1_unit)

        # Reuse initial value since we have energy and forces already
        u0, du0 = (
            self.forces.pot.copy(),
            np.dot(dstrip(self.forces.f.flatten()), dq1_unit.flatten()),
        )

        # Do one SD iteration; return positions and energy
        # (x, fx,dfx) = min_brent(self.lm, fdf0=(u0, du0), x0=0.0,  #DELETE
        min_brent(
            self.lm,
            fdf0=(u0, du0),
            x0=0.0,
            tol=self.ls_options["tolerance"] * self.tolerances["energy"],
            itmax=self.ls_options["iter"],
            init_step=self.ls_options["step"],
        )
        info("   Number of force calls: %d" % (self.lm.fcount))
        self.lm.fcount = 0

        # Update positions and forces
        self.beads.q = self.lm.dbeads.q
        self.forces.transfer_forces(
            self.lm.dforces
        )  # This forces the update of the forces

        d_x = np.absolute(np.subtract(self.beads.q, self.lm.x0))
        x = np.linalg.norm(d_x)
        # Automatically adapt the search step for the next iteration.
        # Relaxes better with very small step --> multiply by factor of 0.1 or 0.01

        self.ls_options["step"] = (
            0.1 * x * self.ls_options["adaptive"]
            + (1 - self.ls_options["adaptive"]) * self.ls_options["step"]
        )

        # Exit simulation step
        d_x_max = np.amax(np.absolute(d_x))
        self.exitstep(self.forces.pot, u0, d_x_max)


class CGOptimizer(DummyOptimizer):
    """
    Conjugate gradient, Polak-Ribiere
    gradf1: force at current atom position
    gradf0: force at previous atom position
    dq1 = direction to move
    dq0 = previous direction
    dq1_unit = unit vector of dq1
    """

    def bind(self, ceop):
        # call bind function from DummyOptimizer
        super(CGOptimizer, self).bind(ceop)
        self.lm.bind(self)
        self.ls_options = ceop.ls_options

    def step(self, step=None):
        """Does one simulation time step
           Attributes:
           ptime: The time taken in updating the velocities.
           qtime: The time taken in updating the positions.
           ttime: The time taken in applying the thermostat steps.
        """

        self.ptime = 0.0
        self.ttime = 0.0
        self.qtime = -time.time()

        info("\nMD STEP %d" % step, verbosity.debug)

        if step == 0:
            gradf1 = dq1 = dstrip(self.forces.f)

            # Move direction for 1st conjugate gradient step
            dq1_unit = dq1 / np.sqrt(np.dot(gradf1.flatten(), gradf1.flatten()))
            info(" @GEOP: Determined SD direction", verbosity.debug)

        else:

            gradf0 = self.old_f
            dq0 = self.d
            gradf1 = dstrip(self.forces.f)
            beta = np.dot((gradf1.flatten() - gradf0.flatten()), gradf1.flatten()) / (
                np.dot(gradf0.flatten(), gradf0.flatten())
            )
            dq1 = gradf1 + max(0.0, beta) * dq0
            dq1_unit = dq1 / np.sqrt(np.dot(dq1.flatten(), dq1.flatten()))
            info(" @GEOP: Determined CG direction", verbosity.debug)

        # Store force and direction for next CG step
        self.d[:] = dq1
        self.old_f[:] = gradf1

        if len(self.fixatoms) > 0:
            for dqb in dq1_unit:
                dqb[self.fixatoms * 3] = 0.0
                dqb[self.fixatoms * 3 + 1] = 0.0
                dqb[self.fixatoms * 3 + 2] = 0.0

        self.lm.set_dir(dstrip(self.beads.q), dq1_unit)

        # Reuse initial value since we have energy and forces already
        u0, du0 = (
            self.forces.pot.copy(),
            np.dot(dstrip(self.forces.f.flatten()), dq1_unit.flatten()),
        )

        # Do one CG iteration; return positions and energy
        min_brent(
            self.lm,
            fdf0=(u0, du0),
            x0=0.0,
            tol=self.ls_options["tolerance"] * self.tolerances["energy"],
            itmax=self.ls_options["iter"],
            init_step=self.ls_options["step"],
        )
        info("   Number of force calls: %d" % (self.lm.fcount))
        self.lm.fcount = 0

        # Update positions and forces
        self.beads.q = self.lm.dbeads.q
        self.forces.transfer_forces(
            self.lm.dforces
        )  # This forces the update of the forces

        d_x = np.absolute(np.subtract(self.beads.q, self.lm.x0))
        x = np.linalg.norm(d_x)
        # Automatically adapt the search step for the next iteration.
        # Relaxes better with very small step --> multiply by factor of 0.1 or 0.01

        self.ls_options["step"] = (
            0.1 * x * self.ls_options["adaptive"]
            + (1 - self.ls_options["adaptive"]) * self.ls_options["step"]
        )

        # Exit simulation step
        d_x_max = np.amax(np.absolute(d_x))
        self.exitstep(self.forces.pot, u0, d_x_max)
