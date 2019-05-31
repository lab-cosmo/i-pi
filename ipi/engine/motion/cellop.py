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
from ipi.utils.mathtools import *


__all__ = ['CellopMotion']
pGPA = 5.

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

        pass

    def bind(self, dumop):
        print("BIND")
        self.dbeads = dumop.beads.copy()
        self.dcell = dumop.cell.copy()
        self.dforces = dumop.forces.copy(self.dbeads, self.dcell)
        self.h0 = dstrip(self.dcell.h).copy()
        self.ih0 = dstrip(self.dcell.ih).copy()
        self.strain = (np.dot(dstrip(self.dcell.h),self.ih0) - np.eye(3)).flatten()
        #sp = self.dcell.positions_abs_to_scaled(self.dbeads.q)
        #print(sp.shape)
        #ap = self.dcell.positions_scaled_to_abs(sp)
        # stores a reference to the atoms and cell we are computing forces for


    def __call__(self, x):
        """computes energy and gradient for optimization step"""

        self.fcount += 1
        #CHECK THE DIMENSIONALITY OF X
        new_strain = x[:,0:9].reshape((3,3))
        print("new_strain", new_strain)
        print("h0", self.h0)
        print("h", self.dcell.h)
        print("h computed",np.dot((np.eye(3)+new_strain), self.h0))
        self.dcell.h = np.dot((np.eye(3)+new_strain), self.h0)
        print("h updated", self.dcell.h)
        print("COORDS Before", self.dbeads.q)
        self.dbeads.q = self.dcell.positions_scaled_to_abs(x[:,9:]) #dbeads.q should be equal to scaled_to_abs positions because x[:,9:] contains scaled pos
        print("COORDS After", self.dbeads.q)
        #self.h0 ###should be passed from input.xml
        #self.pext = 0.0 #  for zero external pressure
        #self.strain = (np.dot(dstrip(self.dcell.h),self.ih0) - np.eye(3)).flatten() #epsilon
        print("Should be the same", new_strain, self.strain)
        print("Vir", self.dforces.vir)
        #metric = np.dot(self.dcell.h.T, self.dcell.h) #g = hTh(3,3)

        nat = len(self.dforces.f[0]) / 3
        #f_sc = self.dforces.forces_abs_to_scaled()
        #f_sc_reshaped = f_sc.reshape((nat,3))
        # Defines the effective energy
        #pGPA = 4
        p = pGPA*10**9/(2.9421912*10**13)
        pV = -p*self.dcell.V
        e = self.dforces.pot +pV  # Energy
        #pV = self.pext * self.dcell.V
        #pV=0
        print("vir+pV", self.dforces.vir+pV*np.eye(3))
        print("vir",self.dforces.vir )
        print("pV", pV*np.eye(3))
        e = e + pV #assume p = 0
        g = np.zeros(nat*3 + 9)
        g[0:9] = -np.dot((self.dforces.vir + np.eye(3) * pV),invert_ut3x3( np.eye(3) + new_strain.reshape((3,3)).T)).flatten()
        g[9:] = -self.dforces.forces_abs_to_scaled()
        #sf=self.dforces.forces_abs_to_scaled(self.dforces.f[0])
        #self.strain= np.zeros((3,3))#self.strain.reshape((3,3))
        # Defines the effective gradient
        #g_x_old_f = np.zeros((nat,3))
        # Gradient contains 3N + 9 components
        #g[0:9] = - np.dot((self.dforces.vir + np.eye(3) * pV),invert_ut3x3( np.eye(3) + (self.strain).T)).flatten()
        #g_x_old_f[:] = np.dot(metric, f_sc_reshaped.T).T
        #g[9:] = np.dot(self.metric, sf.T).T.flatten()
        #g[9:] = -g_x_old_f.flatten().reshape((1, nat*3))
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

    def exitstep(self, fx, u0, x, step):
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
        #if step==2:
        #    fmax = 0
        #    e = 0
        #    x = 0
        info("@GEOP", verbosity.medium)
        self.tolerances["position"]
        info("   Current energy             %e" % (fx))
        info("   Position displacement      %e  Tolerance %e" % (x, self.tolerances["position"]), verbosity.medium)
        info("   Max force component        %e  Tolerance %e" % (fmax, self.tolerances["force"]), verbosity.medium)
        info("   Energy difference per atom %e  Tolerance %e" % (e, self.tolerances["energy"]), verbosity.medium)
        print("shape",self.old_f.shape,self.forces.f.flatten().shape)
        if (np.linalg.norm(self.forces.f.flatten() - self.old_f[:,9:]) <= 1e-20):
            softexit.trigger("Something went wrong, the forces are not changing anymore."
                             " This could be due to an overly small tolerance threshold "
                             "that makes no physical sense. Please check if you are able "
                             "to reach such accuracy with your force evaluation"
                             " code (client).")

        if (e <= self.tolerances["energy"])   \
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
                nat = self.beads.q.size/3
                icell=np.dot(self.cell.ih,self.cell.ih)
                h_block =np.kron(np.eye(nat), icell)
                #h_block = np.zeros((nat*3, nat*3))
                #strain_matrix= np.diag(self.cell.)
                print("INIT_Of_Hessian", self.cell.ih.flatten())
                strain_matrix=np.eye(9)/self.cell.V**(1./3)
                print("strain_matrix", strain_matrix)
                invhess = np.zeros([9+3*nat,9+3*nat])
                invhess[0:9,0:9]= strain_matrix
                invhess[-3*nat:,-3*nat:]= h_block
                geop.invhessian = invhess ###change the hessioan
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


            ####should be rewritten properly
            print("ZERO STEP")
            print("step = 0", self.forces.f[0])
            self.ih0 = dstrip(self.cell.ih).copy()
            self.h0 = dstrip(self.cell.h).copy()
            #print("CHECK", self.ih0, invert_ut3x3(self.h0))
            #self.h0 = np.array([[9.82657000e+00, 4.62849404e-16, 4.62849404e-16],
            #  [0.00000000e+00, 7.55890440e+00, 4.62849404e-16],
            #  [0.00000000e+00, 0.00000000e+00, 7.55890440e+00]])
            #self.ih0 = invert_ut3x3(self.h0)
            nat = len(self.forces.f[0])/3
            strain = (np.dot(dstrip(self.cell.h),self.ih0) - np.eye(3)).flatten()
            ar_len = nat*3+9
            #pGPA = 4
            p = -pGPA*10**9/(2.9421912*10**13)
            pV = p*self.cell.V
            ff = np.zeros((1,ar_len))
            print("vir+pV", self.forces.vir+pV*np.eye(3))
            print("vir",self.forces.vir )
            print("pV", pV*np.eye(3))
            ff[:,0:9] = np.dot((self.forces.vir + np.eye(3) * pV),invert_ut3x3( np.eye(3) + strain.reshape((3,3)).T)).flatten()*0.1
            #print("strain", strain)
            #print("strain before flatten", (np.dot(dstrip(self.cell.h),self.ih0) - np.eye(3)))
            #print("strin reshaped", strain.reshape((3,3)))
            #print("cell", self.cell.h)
            #print("cell flatten", self.cell.h.flatten())
            #print("ff_0-9", ff[:,0:9])
            #print("vir", self.forces.vir)
            #print("energy", self.forces.pot)
            ff[:,9:] = dstrip(self.forces.f)
            d = np.zeros((1,nat*3))
            d += ff[:,9:] / np.sqrt(np.dot(ff[:,9:].flatten(), ff[:,9:].flatten()))
            d.shape = (nat, 3)
            d = np.dot(d, dstrip(self.cell.ih).T)
            d = d.reshape((1, nat * 3))
            self.d[:,0:9] = ff[:, 0:9]
            self.d[:,9:] = d
            #print("ff, d", ff, self.d)
            #print("self.d", self.d)

            #print("h0", self.h0)

            #g_x_old_f = np.zeros((nat,3))
            #f_sc = self.forces.forces_abs_to_scaled()
            #f_sc_reshaped = f_sc.reshape((nat,3))
            #metric = np.dot(self.cell.h.T, self.cell.h)
            #g_x_old_f[:] = np.dot(metric, f_sc_reshaped.T).T
            #f_sc =  g_x_old_f.flatten().reshape((1, nat*3))
            #ff[:,9:] = dstrip(self.forces. forces_abs_to_scaled())
            #ff[:,9:] = self.forces.forces_abs_to_scaled()
            #self.d += ff / np.sqrt(np.dot(ff.flatten(), ff.flatten()))
            #print("d", self.d)
            if len(self.fixatoms) > 0:
                for dqb in self.d:
                    dqb[self.fixatoms * 3] = 0.0
                    dqb[self.fixatoms * 3 + 1] = 0.0
                    dqb[self.fixatoms * 3 + 2] = 0.0

        print("step", step)
        #print(self.gm.strain.flatten().shape)
        #print("BFGSOptimizer step shape",self.old_x.shape)
        #!!!!!!##
        # dforces --> forces; dcell --> cell and so on
        # these 2 lines are the first 9 components of your X vector
        #self.ih0 = dstrip(self.cell.ih).copy()
        #print(self.forces.pot)

        #strain = strain.reshape((3,3))
        #nat = len(self.forces.f[0]) / 3
        #This is g in article
        #metric = np.dot(self.cell.h.T, self.cell.h) #g = hTh(3,3)
        # This is just to get n_of_atoms
        #f  = self.forces.f[0] #check if it refers to the 0 bead or centriod

        #print("f", f)
        #print("self.force.f[0]", self.forces.f[0])
        #print("self.beads.q[0]",self.beads.q[0])
        # This is for now to define pv

        #strain = np.zeros((3,3))
        #self.gm.strain.flatten() #self.gm.strain.flatten()
        #print("1 self.force.f[0]", self.forces.f[0])
        #print("2 self.force.f[0]", self.forces.f[0])
        # These 4 lines define the components of your self.old_f array
        # Replace g with old_f
        # dforces --> forces; dcell --> cell and so on
        #self.g = np.zeros(nat*3 + 9)
        #print("3 self.force.f[0]", self.forces.f[0])
        #sf=self.forces.forces_abs_to_scaled(self.forces.f[0])
        #sf=self.forces.forces_abs_to_scaled()
        #sf = sf.reshape((nat,3))
        #the expression for
        #self.g[0:9] = - np.dot((self.forces.vir + np.eye(3) * pV),invert_ut3x3( np.eye(3) + (strain).T)).flatten()
        #self.g[0:9] =
        #self.g[9:] = np.dot(metric,sf.T).T.flatten()
        #g_x_old_f = np.zeros((nat,3))
        #f_sc = self.forces.forces_abs_to_scaled()
        #f_sc_reshaped = f_sc.reshape((nat,3))
        #g_x_old_f[:] = np.dot(metric, f_sc_reshaped.T).T
        #self.g[9:] = self.forces.forces_abs_to_scaled()
        #self.old_f[0:9] = ..
        #np.dot((self.forces.vir + np.eye(3) * pV),invert_ut3x3( np.eye(3) + (strain).T)).flatten()

        nat = len(self.forces.f[0]) / 3
        #pGPA = 4
        p = pGPA*10**9/(2.9421912*10**13)
        pV = -p*self.cell.V
        #print("cell, cell0", self.cell.h, self.h0)
        strain = (np.dot(dstrip(self.cell.h),self.ih0) - np.eye(3)).flatten()
        self.old_x[:,0:9] = strain.flatten()
        print("x_0-9", (np.dot(dstrip(self.cell.h),self.ih0) - np.eye(3)).flatten())
        self.old_x[:,9:] = self.cell.positions_abs_to_scaled(self.beads.q) #scaled positions; you can use the function of the class cell
        self.old_u[:] = self.forces.pot +pV #  = 0 it's fine like that for now
        print("vir+pV", self.forces.vir+pV*np.eye(3))
        print("vir",self.forces.vir )
        print("pV", pV*np.eye(3))
        #print("strain, invert", strain, invert_ut3x3(strain.reshape((3,3))+np.eye(3)).T)
        #print("f_0-9",self.forces.vir + np.eye(3) * pV, invert_ut3x3( np.eye(3) + strain.reshape((3,3)).T) )#invert_ut3x3( np.eye(3) + (strain.reshape((3.3)).T)).flatten() ))
        self.old_f[:,0:9] = np.dot((self.forces.vir + np.eye(3) * pV),invert_ut3x3( np.eye(3) + strain.reshape((3,3)).T)).flatten()
        self.old_f[:,9:] = self.forces.forces_abs_to_scaled() #first 9 components are g[0:9], g[9:] is the rest

        if len(self.fixatoms) > 0:
            for dqb in self.old_f:
                dqb[self.fixatoms * 3] = 0.0
                dqb[self.fixatoms * 3 + 1] = 0.0
                dqb[self.fixatoms * 3 + 2] = 0.0

        fdf0 = (self.old_u, -self.old_f) #don't modify
        #print("old_x TEST", self.old_x[:,9:])
        #print("cell old TEST", self.cell.h)
        #print("fdf0", fdf0)

        # Do one iteration of BFGS
        # The invhessian and the directions are updated inside.
        #####################e+pV, g (F on article)

        BFGS(self.old_x, self.d, self.gm, fdf0, self.invhessian, self.big_step,
             self.ls_options["tolerance"] * self.tolerances["energy"], self.ls_options["iter"])

        info("   Number of force calls: %d" % (self.gm.fcount)); self.gm.fcount = 0
        # Update positions and forces
        #print("CHECK", self.d)
        #print("dcell.h updated", self.gm.dcell.h)
        #print("cell.h old", self.cell.h)
        #print("h0", self.h0)
        #print("old_x TEST the same?", self.old_x[:,9:])
        #print("cell old TEST the same?", self.cell.h, self.gm.dcell.h)
        print("vir+pV", self.forces.vir+pV*np.eye(3))
        print("vir",self.forces.vir )
        print("pV", pV*np.eye(3))
        old_x_abs = dstrip(self.cell.positions_scaled_to_abs(self.old_x[:,9:]))

        #V_old = np.linalg.det(np.dot(np.eye(3)+self.old_x[:,0:9].reshape((3,3)),self.h0))
        #print("V_old", V_old)

        self.cell.h = self.gm.dcell.h
        self.beads.q = self.gm.dbeads.q #don't touch !
                #print(self.beads.q)
        self.forces.transfer_forces(self.gm.dforces)  # This forces the update of the forces
        #print("this is cellop")
        # Exit simulation step
        #pGPA = 4
        p = pGPA*10**9/(2.9421912*10**13)
        pV = -p*self.cell.V
        d_x_max = np.amax(np.absolute(np.subtract(self.beads.q, old_x_abs)))
        print("final pos", self.beads.q, self.cell.h )
        self.exitstep(self.forces.pot+pV, self.old_u, d_x_max, step)
