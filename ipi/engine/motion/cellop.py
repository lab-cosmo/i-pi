""" hgjhgjhg
Contains classes for different geometry optimization algorithms.

TODO

Algorithms implemented by Michele Ceriotti and Benjamin Helfrecht, 2015
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import numpy as np
import time

#from ipi.utils.mathtools import *
#from ipi.utils.depend import *
from ipi.engine.motion import Motion
from ipi.utils.depend import dstrip, dobject
from ipi.utils.softexit import softexit
from ipi.utils.mintools import min_brent, BFGS, BFGSTRM, L_BFGS
from ipi.utils.messages import verbosity, info
#from ipi.engine.cell import scaled_pos
#from ipi.engine.cell import Cell


__all__ = ['CellopMotion']


class CellopMotion(Motion):
    """Geometry optimization class.

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

    def __init__(self, fixcom=False, fixatoms=None,
                 mode="lbfgs",
                 biggest_step=100.0,
                 old_pos=np.zeros(0, float),
                 old_pot=np.zeros(0, float),
                 old_force=np.zeros(0, float),
                 old_direction=np.zeros(0, float),
                 strain=np.zeros(0,float),
                 invhessian_bfgs=np.eye(0, 0, 0, float),
                 hessian_trm=np.eye(0, 0, 0, float),
                 tr_trm=np.zeros(0, float),
                 ls_options={"tolerance": 1, "iter": 100, "step": 1e-3, "adaptive": 1.0},
                 tolerances={"energy": 1e-7, "force": 1e-4, "position": 1e-4},
                 corrections_lbfgs=5,
                 scale_lbfgs=1,
                 qlist_lbfgs=np.zeros(0, float),
                 glist_lbfgs=np.zeros(0, float)):
        """Initialises CellopMotion.

        Args:
           fixcom: An optional boolean which decides whether the centre of mass
              motion will be constrained or not. Defaults to False.
        """
        if len(fixatoms) > 0:
            raise ValueError("The optimization algorithm with fixatoms is not implemented. "
                             "We stop here. Comment this line and continue only if you know what you are doing")

        super(CellopMotion, self).__init__(fixcom=fixcom, fixatoms=fixatoms)

        # Optimization Options

        self.mode = mode
        self.big_step = biggest_step
        self.tolerances = tolerances
        self.ls_options = ls_options

        #
        self.old_x = old_pos
        self.old_u = old_pot
        self.old_f = old_force
        self.strain = strain
        self.d = old_direction

        # Classes for minimization routines and specific attributes
        if self.mode == "bfgs":
            self.invhessian = invhessian_bfgs
            self.optimizer = BFGSOptimizer()
        else:
            self.optimizer = DummyOptimizer()

    def bind(self, ens, beads, nm, cell, bforce, prng):
        """Binds beads, cell, bforce and prng to CellopMotion

            Args:
            beads: The beads object from whcih the bead positions are taken.
            nm: A normal modes object used to do the normal modes transformation.
            cell: The cell object from which the system box is taken.
            bforce: The forcefield object from which the force and virial are taken.
            prng: The random number generator object which controls random number generation.
        """

        super(CellopMotion, self).bind(ens, beads, nm, cell, bforce, prng)
        # Binds optimizer
        self.optimizer.bind(self)

    def step(self, step=None):
        self.optimizer.step(step)
"""def dd(dobj):
    if not issubclass(dobj.__class__, dobject):
        raise ValueError("Cannot access a ddirect view of an object which is not a subclass of dobject")
    return dobj._direct
"""
class GradientMapper(object):


    """Creation of the multi-dimensional function that will be minimized.
    Used in the BFGS and L-BFGS minimizers.

    Attributes:
        dbeads:   copy of the bead object
        dcell:   copy of the cell object
        dforces: copy of the forces object
    """

    def __init__(self):
        self.fcount = 0
        """Initialises base cell class.

        Args:
           h: Optional array giving the initial lattice vector matrix. The
              reference cell matrix is set equal to this. Must be an upper
              triangular 3*3 matrix. Defaults to a 3*3 zeroes matrix.
        """
        print("INIT")
        pass

    def bind(self, dumop):
        print("BIND")
        self.dbeads = dumop.beads.copy()
        self.dcell = dumop.cell.copy()
        self.dforces = dumop.forces.copy(self.dbeads, self.dcell)
        self.h0 = dstrip(self.dcell.h).copy()
        self.ih0 = dstrip(self.dcell.ih).copy()
        self.strain = (np.dot(dstrip(self.dcell.h),self.ih0) - np.eye(3)).flatten()
        sp = self.dcell.positions_abs_to_scaled(self.dbeads.q)
        ap = self.dcell.positions_scaled_to_abs(sp)
        # stores a reference to the atoms and cell we are computing forces for


    def __call__(self, x):
        """computes energy and gradient for optimization step"""
        print("CALL")
        self.fcount += 1
        self.dbeads.q = x[:,9:]
        self.h0 ###should be passed from input.xml

        self.pext = 0.0 #  for zero external pressure
        self.strain = (np.dot(dstrip(self.dcell.h),self.ih0) - np.eye(3)).flatten() #epsilon
        self.metric = np.dot(self.dcell.h.T, self.dcell.h) #g = hTh(3,3)
        f  = self.dforces.f[0] #check if it refers to the 0 bead or centriod

        nat = len(f) / 3
        sf=self.dforces.forces_abs_to_scaled(self.dforces.f[0])
        sf= sf.reshape((nat,3))
        # Defines the effective energy
        e = self.dforces.pot   # Energy
        pV = self.pext * self.dcell.V
        p=0
        e = e + pV #assume p = 0
        self.strain= self.strain.reshape((3,3))

        # Defines the effective gradient
        g = np.zeros(nat*3 + 9)
          # Gradient contains 3N + 9 components
        g[0:9] = - np.dot((self.dforces.vir + np.eye(3) * pV), np.eye(3) + (self.strain).T).flatten()
        g[9:] = np.dot(self.metric, sf.T).T.flatten()

        #print(self.dforces.pot / self.dbeads.nbeads, np.trace((self.dforces.vir) / (3.0 * self.dcell.V)),self.tensor2vec((self.dforces.vir) / self.dcell.V))
        return e, g

    def tensor2vec(self, tensor):
        """Takes a 3*3 symmetric tensor and returns it as a 1D array,
        containing the elements [xx, yy, zz, xy, xz, yz].
        """
        return np.array([tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]])


class DummyOptimizer(dobject):
    """ Dummy class for all optimization classes """

    def __init__(self):
        """initialises object for LineMapper (1-d function) and for GradientMapper (multi-dimensional function) """


        self.gm = GradientMapper()

    def step(self, step=None):
        """Dummy simulation time step which does nothing."""
        pass


    def bind(self, geop):
        print("BIND_DUMMY_OPT")
        """
        bind optimization options and call bind function of LineMapper and GradientMapper (get beads, cell,forces)
        check whether force size, direction size and inverse Hessian size from previous step match system size
        """
        self.beads = geop.beads
        self.cell = geop.cell
        self.forces = geop.forces
        self.fixcom = geop.fixcom
        self.fixatoms = geop.fixatoms
        #self.p_ext = geop.p_ext # should come from the i-pi input

        self.mode = geop.mode
        self.tolerances = geop.tolerances

        # Check for very tight tolerances

        if self.tolerances["position"] < 1e-7:
            raise ValueError("The position tolerance is too small for any typical calculation. "
                             "We stop here. Comment this line and continue only if you know what you are doing")
        if self.tolerances["force"] < 1e-7:
            raise ValueError("The force tolerance is too small for any typical calculation. "
                             "We stop here. Comment this line and continue only if you know what you are doing")
        if self.tolerances["energy"] < 1e-10:
            raise ValueError("The energy tolerance is too small for any typical calculation. "
                             "We stop here. Comment this line and continue only if you know what you are doing")

        # The resize action must be done before the bind
        if geop.old_x.size != self.beads.q.size+9: #check
            if geop.old_x.size == 0:
                print("Here")
                geop.old_x = np.zeros((self.beads.nbeads, 3 * self.beads.natoms+9), float)
            else:
                raise ValueError("Conjugate gradient force size does not match system size")
        if geop.old_u.size != 1:
            if geop.old_u.size == 0:
                geop.old_u = np.zeros(1, float)
            else:
                raise ValueError("Conjugate gradient force size does not match system size")
        if geop.old_f.size != self.beads.q.size+9:
            if geop.old_f.size == 0:
                geop.old_f = np.zeros((self.beads.nbeads, 3 * self.beads.natoms+9), float)
            else:
                raise ValueError("Conjugate gradient force size does not match system size")
        if geop.d.size != self.beads.q.size+9:
            if geop.d.size == 0:
                geop.d = np.zeros((self.beads.nbeads, 3 * self.beads.natoms+9), float)
            else:
                raise ValueError("Conjugate gradient direction size does not match system size")
        print("DummyOptimizer bind shape",self.beads.q.size)
        self.old_x = geop.old_x
        self.old_u = geop.old_u
        self.old_f = geop.old_f
        self.strain = geop.strain
        self.d = geop.d

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
        info("   Position displacement      %e  Tolerance %e" % (x, self.tolerances["position"]), verbosity.medium)
        info("   Max force component        %e  Tolerance %e" % (fmax, self.tolerances["force"]), verbosity.medium)
        info("   Energy difference per atom %e  Tolerance %e" % (e, self.tolerances["energy"]), verbosity.medium)

        if (np.linalg.norm(self.forces.f.flatten() - self.old_f[:,9:].flatten()) <= 1e-20):
            softexit.trigger("Something went wrong, the forces are not changing anymore."
                             " This could be due to an overly small tolerance threshold "
                             "that makes no physical sense. Please check if you are able "
                             "to reach such accuracy with your force evaluation"
                             " code (client).")

        if (np.absolute((fx - u0) / self.beads.natoms) <= self.tolerances["energy"])   \
                and (fmax <= self.tolerances["force"])  \
                and (x <= self.tolerances["position"]):
            softexit.trigger("Geometry optimization converged. Exiting simulation")


class BFGSOptimizer(DummyOptimizer):

    """ BFGS Minimization """


    def bind(self, geop):
        # call bind function from DummyOptimizer
        super(BFGSOptimizer, self).bind(geop)

        if geop.invhessian.size != (self.beads.q.size+9 * self.beads.q.size+9):
            if geop.invhessian.size == 0:
                geop.invhessian = np.eye(self.beads.q.size+9, self.beads.q.size+9, 0, float) ###change the hessioan
            else:
                raise ValueError("Inverse Hessian size does not match system size")

        self.invhessian = geop.invhessian
        print("BIND_BFGS_OPT")
        self.gm.bind(self)
        self.big_step = geop.big_step
        self.ls_options = geop.ls_options

    def step(self, step=None):
        """ Does one simulation time step.
            Attributes:
            qtime: The time taken in updating the positions.
        """

        self.qtime = -time.time()
        info("\nMD STEP %d" % step, verbosity.debug)

        if step == 0:
            info(" @GEOP: Initializing BFGS", verbosity.debug)
            self.d += dstrip(self.forces.f) / np.sqrt(np.dot(self.forces.f.flatten(), self.forces.f.flatten()))

            if len(self.fixatoms) > 0:
                for dqb in self.d:
                    dqb[self.fixatoms * 3] = 0.0
                    dqb[self.fixatoms * 3 + 1] = 0.0
                    dqb[self.fixatoms * 3 + 2] = 0.0

        print(self.old_x[0,0:9])
        print(self.beads.q)
        print(self.gm.strain.flatten())

        print("1")
        print(self.gm.strain.flatten().shape)
        print("BFGSOptimizer step shape",self.old_x.shape)
        self.old_x[0,0:9] = self.gm.strain.flatten()
        self.old_x[:,9:] = self.beads.q #old_x[0:9] = strain.flatten()
        self.old_u[:] = self.forces.pot #+pV = 0
        self.old_f[:,0:9] = np.zeros(9)
        self.old_f[:,9:] = self.forces.f #as in _call_ of GradientMapper

        if len(self.fixatoms) > 0:
            for dqb in self.old_f:
                dqb[self.fixatoms * 3] = 0.0
                dqb[self.fixatoms * 3 + 1] = 0.0
                dqb[self.fixatoms * 3 + 2] = 0.0

        fdf0 = (self.old_u, -self.old_f)

        # Do one iteration of BFGS
        # The invhessian and the directions are updated inside.
        #####################e+pV, g (F on article)
        print("Bfgs")
        print()
        BFGS(self.old_x, self.d, self.gm, fdf0, self.invhessian, self.big_step,
             self.ls_options["tolerance"] * self.tolerances["energy"], self.ls_options["iter"])

        info("   Number of force calls: %d" % (self.gm.fcount)); self.gm.fcount = 0
        # Update positions and forces
        self.beads.q = self.gm.dbeads.q
        self.forces.transfer_forces(self.gm.dforces)  # This forces the update of the forces
        print("this is cellop")
        # Exit simulation step
        d_x_max = np.amax(np.absolute(np.subtract(self.beads.q, self.old_x[:,9:])))
        print("dxmax",d_x_max)
        self.exitstep(self.forces.pot, self.old_u, d_x_max)
