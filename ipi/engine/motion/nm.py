"""Contains the classes that deal with the different dynamics required in
different types of ensembles.

Copyright (C) 2013, Joshua More and Michele Ceriotti

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http.//www.gnu.org/licenses/>.
"""

__all__ = ['NormalModeMover']

import numpy as np
import time
import itertools
import os

from ipi.engine.motion import Motion
from ipi.utils.depend import *
from ipi.utils import units
from ipi.utils.softexit import softexit
from ipi.utils.messages import verbosity, warning, info
from ipi.utils.io import print_file
from ipi.engine.atoms import Atoms
from scipy.special import hermite
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator
from scipy.misc import logsumexp
from itertools import combinations
import time


class NormalModeMover(Motion):
    """Normal Mode analysis.
    """

    def __init__(self, fixcom=False, fixatoms=None, mode="imf", dynmat=np.zeros(0, float),  prefix="", asr="none", nprim="1", fnmrms="1.0", nevib="25.0", nint="101", nbasis="10", athresh="1e-2", ethresh="1e-2", nkbt="4.0", nexc="5", solve=False, mptwo=False, print_mftpot=False, print_1b_map=False, print_2b_map=False, threebody=False, print_vib_density=False, nparallel=1, alpha=1.0, pair_range=np.zeros(0, int)):
        """Initialises NormalModeMover.
        Args:
        fixcom	: An optional boolean which decides whether the centre of mass
                  motion will be constrained or not. Defaults to False. 
        dynmatrix : A 3Nx3N array that stores the dynamic matrix.
        refdynmatrix : A 3Nx3N array that stores the refined dynamic matrix.
        """

        super(NormalModeMover, self).__init__(fixcom=fixcom, fixatoms=fixatoms)

        # Finite difference option.
        self.mode = mode
        if self.mode == "imf":
            self.calc = IMF()
        elif self.mode == "vscf":
            self.calc = VSCF()

        self.dynmatrix = dynmat
        self.mode = mode
        self.frefine = False
        self.U = None
        self.V = None
        self.prefix = prefix
        self.asr = asr
        self.nprim = nprim #1
        self.nz = 0 #1
        self.alpha = alpha #1
        self.fnmrms = fnmrms #1.0
        self.nevib = nevib #25.0
        self.nint = nint #101
        self.nbasis = nbasis #10
        self.athresh = athresh #1e-2
        self.ethresh = ethresh #1e-2
        self.nkbt = nkbt #4.0
        self.nexc = nexc #5
        self.solve = solve #5
        self.mptwo = mptwo #False
        self.print_mftpot = print_mftpot #False
        self.print_1b_map = print_1b_map #False
        self.print_2b_map = print_2b_map #False
        self.threebody = threebody #False
        self.print_vib_density = print_vib_density #False
        self.nparallel = nparallel #False
        self.pair_range = pair_range

        if self.prefix == "":
                self.prefix = self.mode

    def bind(self, ens, beads, nm, cell, bforce, prng, omaker):

        super(NormalModeMover, self).bind(ens, beads, nm, cell, bforce, prng, omaker)
        self.temp = self.ensemble.temp

        # Raises error for nbeads not equal to 1.
        if(self.beads.nbeads > 1):
            raise ValueError("Calculation not possible for number of beads greater than one.")

        self.ism = 1 / np.sqrt(dstrip(self.beads.m3[-1]))
        self.m = dstrip(self.beads.m)
        self.calc.bind(self)

        self.dbeads = self.beads.copy(nbeads=self.nparallel)
        self.dcell = self.cell.copy()
        self.dforces = self.forces.copy(self.dbeads, self.dcell)

    def step(self, step=None):
        """Executes one step of phonon computation. """
        self.calc.step(step)

    def apply_asr(self, dm):
        """
        Removes the translations and/or rotations depending on the asr mode.
        """
        if(self.asr == "none"):
            return dm

        if(self.asr == "crystal"):
            # Computes the centre of mass.
            com = np.dot(np.transpose(self.beads.q.reshape((self.beads.natoms, 3))), self.m) / self.m.sum()
            qminuscom = self.beads.q.reshape((self.beads.natoms, 3)) - com
            # Computes the moment of inertia tensor.
            moi = np.zeros((3, 3), float)
            for k in range(self.beads.natoms):
                moi -= np.dot(np.cross(qminuscom[k], np.identity(3)), np.cross(qminuscom[k], np.identity(3))) * self.m[k]

            U = (np.linalg.eig(moi))[1]
            R = np.dot(qminuscom, U)
            D = np.zeros((3, 3 * self.beads.natoms), float)

            # Computes the vectors along rotations.
            D[0] = np.tile([1, 0, 0], self.beads.natoms) / self.ism
            D[1] = np.tile([0, 1, 0], self.beads.natoms) / self.ism
            D[2] = np.tile([0, 0, 1], self.beads.natoms) / self.ism

            # Computes unit vecs.
            for k in range(3):
                D[k] = D[k] / np.linalg.norm(D[k])

            # Computes the transformation matrix.
            transfmatrix = np.eye(3 * self.beads.natoms) - np.dot(D.T, D)
            r = np.dot(transfmatrix.T, np.dot(dm, transfmatrix))

            # Sets the number of  zero modes.
            self.nz = 3
            return r

        elif(self.asr == "poly"):
            # Computes the centre of mass.
            com = np.dot(np.transpose(self.beads.q.reshape((self.beads.natoms, 3))), self.m) / self.m.sum()
            qminuscom = self.beads.q.reshape((self.beads.natoms, 3)) - com

            # Computes the moment of inertia tensor.
            moi = np.zeros((3, 3), float)
            for k in range(self.beads.natoms):
                moi -= np.dot(np.cross(qminuscom[k], np.identity(3)), np.cross(qminuscom[k], np.identity(3))) * self.m[k]

            U = (np.linalg.eig(moi))[1]
            R = np.dot(qminuscom, U)
            D = np.zeros((6, 3 * self.beads.natoms), float)

            # Computes the vectors along translations and rotations.
            D[0] = np.tile([1, 0, 0], self.beads.natoms) / self.ism
            D[1] = np.tile([0, 1, 0], self.beads.natoms) / self.ism
            D[2] = np.tile([0, 0, 1], self.beads.natoms) / self.ism
            for i in range(3 * self.beads.natoms):
                iatom = i / 3
                idof = np.mod(i, 3)
                D[3, i] = (R[iatom, 1] * U[idof, 2] - R[iatom, 2] * U[idof, 1]) / self.ism[i]
                D[4, i] = (R[iatom, 2] * U[idof, 0] - R[iatom, 0] * U[idof, 2]) / self.ism[i]
                D[5, i] = (R[iatom, 0] * U[idof, 1] - R[iatom, 1] * U[idof, 0]) / self.ism[i]

            # Computes unit vecs.
            for k in range(6):
                D[k] = D[k] / np.linalg.norm(D[k])

            # Computes the transformation matrix.
            transfmatrix = np.eye(3 * self.beads.natoms) - np.dot(D.T, D)
            r = np.dot(transfmatrix.T, np.dot(dm, transfmatrix))

            # Sets the number of zero modes. (Assumes poly-atomic molecules)
            self.nz = 6
            return r


class DummyCalculator(dobject):
    """ No-op Calculator """

    def __init__(self):
        pass

    def bind(self, imm):
        """ Reference all the variables for simpler access."""
        self.imm = imm

    def step(self, step=None):
        """Dummy simulation time step which does nothing."""
        pass


class IMF(DummyCalculator):
    """ Temperature scaled normal mode Born-Oppenheimer surface evaluator.
    """

    def bind(self, imm):
        """ Reference all the variables for simpler access."""
        super(IMF, self).bind(imm)

        # Initialises a 3*number of atoms X 3*number of atoms dynamic matrix.
        if(self.imm.dynmatrix.size != (self.imm.beads.q.size * self.imm.beads.q.size)):
            if(self.imm.dynmatrix.size == 0):
                self.imm.dynmatrix = np.zeros((self.imm.beads.q.size, self.imm.beads.q.size), float)
            else:
                raise ValueError("Force constant matrix size does not match system size")
        else:
            self.imm.dynmatrix = self.imm.dynmatrix.reshape(((self.imm.beads.q.size, self.imm.beads.q.size)))

        # Applies ASR
        self.imm.dynmatrix = self.imm.apply_asr(self.imm.dynmatrix)

        # Calculates normal mode directions.
        self.imm.w2, self.imm.U = np.linalg.eigh(self.imm.dynmatrix)

        # Calculates the normal mode frequencies.
        # TODO : Should also handle soft modes.
        self.imm.w = self.imm.w2 * 0
        self.imm.w[self.imm.nz:] = np.sqrt(dstrip(self.imm.w2[self.imm.nz:]))
        
        self.imm.V = self.imm.U.copy()
        for i in xrange(len(self.imm.V)):
            self.imm.V[:, i] *= self.imm.ism

        # Harm ZP RMS displacement along normal mode
        # Not temperature dependent so that sampled potentials can easily be 
        # reused to evaluate free energy at different temperature.
        self.imm.nmrms = np.zeros(len(self.imm.w)) 
        self.imm.nmrms[self.imm.nz:] = np.sqrt( 0.5 / self.imm.w[self.imm.nz:])
        self.nmrms = self.imm.nmrms

        # Harm vibr energy at finite temp
        # Similarly this is also not temperature dependent.
        self.imm.nmevib =  0.5 * self.imm.w 

        # Fraction of the harmonic RMS displacement 
        # used to sample along a normal mode
        self.fnmrms = self.imm.fnmrms

        # Multiple of harmonic vibrational energy up to which 
        # the BO surface is sampled 
        self.nevib = self.imm.nevib

        # Integration points for numerical integration of 
        # Hamiltonian matrix elements 
        self.nint = self.imm.nint

        # Number of SHO states used as basis for anharmonic wvfn
        self.nbasis = self.imm.nbasis

        # Convergence threshold for fractional error in vibrational free energy
        self.athresh = self.imm.athresh

        # Number of primitive unit (cells) per simulation (cell) 
        self.nprim = self.imm.nprim

        # Total number of vibrational modes
        self.dof = 3 * self.imm.beads.natoms 

        # Initializes the (an)har energy correction
        self.total_anhar_free_energy = 0.0
        self.total_har_free_energy = 0.0
        self.total_anhar_internal_energy = 0.0
        self.total_har_internal_energy = 0.0

        # Potential energy (offset) at equilibrium positions per primitve unit (cell)
        self.v0 = 0

        # Initializes the SHO wvfn basis for solving the 1D Schroedinger equation
        self.hermite_functions = [hermite(i) for i in xrange(max(20, 4 * self.nbasis))]

        # Sets the total number of steps for IMF.
        self.total_steps = 3 * self.imm.beads.natoms

    def psi(self, n, m, hw, q):
        """
        Returns the value of the n^th wavefunction of a 
        harmonic oscillator with mass m, frequency freq
        at position x.
        """

        # Defines variables for easier referencing
        alpha = m * hw
        try:
          herfun = self.hermite_functions[n]
        except:
          herfun = hermite(n)

        norm = (alpha / np.pi)**0.25 * np.sqrt( 1.0 / (2.0 ** (n) * np.math.factorial(n)))
        psival = norm * np.exp(-alpha * q**2 / 2.0) * herfun(np.sqrt(alpha) * q)

        return psival

    def solve_schroedingers_equation(self, hw, psigrid, vgrid, return_eigsys=False):
        """
        Given the frequency of the HO, the wavefunctions and
        the gridded potential, solves the schrodinger's
        equation.
        """

        # The number of basis set elements.
        nbasis = len(psigrid)

        # Constructs the Hamiltonian matrix.
        h = np.zeros((nbasis,nbasis))
        for i in xrange(nbasis):
            for j in xrange(i, nbasis, 1):
                h[i][j] = np.dot(psigrid[i] * psigrid[j], vgrid)
            h[i][i] *= 0.5
            h[i][i] += 0.5 * (i + 0.5) * hw
        h += h.T 

        # Diagonalise Hamiltonian matrix and evaluate anharmonic free energy and vibrational freq
        evals, evecs = np.linalg.eigh(h)
        
        # Calculates the free and internal energy
        A = -logsumexp(-1.0 * evals / dstrip(self.imm.temp)) * dstrip(self.imm.temp)
        E = np.sum(evals * np.exp(-1.0 * evals / dstrip(self.imm.temp))) / np.sum(np.exp(-1.0 * evals / dstrip(self.imm.temp)))

        if return_eigsys:
            return A, E, evals, evecs.T
        else:
            return A, E

    def step(self, step=None):
        """Computes the Born Oppenheimer curve along a normal mode."""

        if step == self.total_steps:
            self.terminate()

        # Ignores (near) zero modes.
        if step < self.imm.nz:
            info(" @NM : Ignoring the zero mode.", verbosity.medium)
            info(" ", verbosity.medium)
            return 
        elif self.imm.w[step] < 9.1126705e-06:
            info(" @NM : Ignoring normal mode no.  %8d with frequency %15.8f cm^-1." % (step, self.imm.w[step] * 219474) , verbosity.medium)
            info(" ", verbosity.medium)
            self.imm.nz += 1
            return
        else:
            info(" @NM : Treating normal mode no.  %8d with frequency %15.8f cm^-1." % (step, self.imm.w[step] * 219474) ,verbosity.medium) 

        self.v_indep_filename = self.imm.output_maker.prefix + '.' + self.imm.prefix + '.' + str(step) + '.qvf'
        if os.path.exists(self.v_indep_filename):

            # Loads the displacemnts, the potential energy and the forces. 
            qlist, vlist, flist = np.loadtxt(self.v_indep_filename).T

            # Fits cubic splines to data. 
            info("@NM : Fitting cubic splines.", verbosity.medium)
            vspline = interp1d(qlist, vlist, kind='cubic', bounds_error=False)
             
            # Converge wrt size of SHO basis
            bs_iter = 0
            bs_Aanh = [1e-20]
            bs_Eanh = [1e-20]

            # Calculates the 1D correction potential on a grid.
            qgrid = np.linspace(np.min(qlist), np.max(qlist), self.nint)
            vgrid = np.asarray([(np.nan_to_num(vspline(x))) for x in qgrid])

            while True:
                nnbasis = max(1, self.nbasis - 5) + 5 * bs_iter
                nnbasis = self.nbasis + 5 * bs_iter

                psigrid = np.zeros((nnbasis, self.nint))
                # Calculates the wavefunctions on the grid.
                for i in xrange(nnbasis):
                    psigrid[i] = self.psi(i, 1.0, self.imm.w[step], qgrid)
                    psigrid[i] = psigrid[i] / np.sqrt(np.dot(psigrid[i],psigrid[i]))

                # Solves the Schroedinger's Equation.
                bs_AEanh = self.solve_schroedingers_equation(self.imm.w[step], psigrid, vgrid)
                bs_Aanh.append(bs_AEanh[0])
                bs_Eanh.append(bs_AEanh[1])

                info(" @NM : CONVERGENCE : nbasis = %5d    A =  %10.8e   D(A) =  %10.8e /  %10.8e" % (nnbasis, bs_Aanh[-1]     , np.abs(bs_Aanh[-1] - bs_Aanh[-2]) / np.abs(bs_Aanh[-2]), self.athresh), verbosity.medium)
 
                # Check whether anharmonic frequency is converged
                if (np.abs(bs_Aanh[-1] - bs_Aanh[-2]) / np.abs(bs_Aanh[-2])) < self.athresh:
                    break
 
                bs_iter += 1
            Aanh = [bs_Aanh[-1]]
            Eanh = [bs_Eanh[-1]]

        else:
            # initializes the list containing sampled displacements,
            # forces and energies.
            vlist = []
            flist = []
            qlist = []

            # Adds to the list of "sampled configuration" the one that
            # corresponds to the minimum.
            q0 = 0.0
            v0 = dstrip(self.imm.forces.pots).copy()[0] / self.nprim
            f0 = np.dot(dstrip(self.imm.forces.f).copy()[0], np.real(self.imm.V.T[step])) / self.nprim

            self.v0 = v0 # TODO CHECK IF NECESSARY

            vlist.append(0.0)
            flist.append(f0)
            qlist.append(q0)

            # Converge anharmonic vibrational energies w.r.t. density of sampling points
            Aanh = []
            Eanh = []
            Aanh.append(1e-20)
            Athresh = self.athresh 

            sampling_density_iter = -1

            while True:
                
                sampling_density_iter += 1

                # Doubles the grid spacing, so that an estimate of the
                # anharmonic free energy convergence is 
                # possible at default/input grid spacing
                ffnmrms = self.fnmrms * 0.5**sampling_density_iter * 2.0

                # Calculates the displacement in Cartesian coordinates.
                nmd = ffnmrms * self.imm.nmrms[step]
                dev = np.real(self.imm.V.T[step]) * nmd * np.sqrt(self.nprim)
     
                # After the first iteration doubles the displacement to avoid
                # calculation of the potential at configurations already v_indep_listited
                # in the previous iteration.
                if (sampling_density_iter == 0):
                    delta_counter = 1
                else:
                    delta_counter = 2

                counter = 1

                # Explores configurations until the sampled energy exceeds
                # a user-defined threshold of the zero-point energy.
                while True:
        
                    # Displaces along the normal mode.
                    self.imm.dbeads.q = self.imm.beads.q + dev * counter

                    # Stores the "anharmonic" component of the potential
                    # and the force.
                    dv = dstrip(self.imm.dforces.pots).copy()[0] / self.nprim - 0.50 * self.imm.w2[step] * (nmd * counter)**2 - v0
                    df = np.dot(dstrip(self.imm.dforces.f).copy()[0], np.real(self.imm.V.T[step])) / self.nprim + self.imm.w2[step] * (nmd * counter)
     
                    # Adds to the list.
                    # Also stores the total energetics i.e. including 
                    # the harmonic component.
                    vlist.append(dv)
                    flist.append(df)
                    qlist.append(nmd * counter)  
        
                    # Bailout condition.
                    if self.nevib * self.imm.nmevib[step] < np.abs(0.50 * self.imm.w2[step] * (nmd * counter)**2 + dv):
                        break
                    
                    # Increases the displacement by 1 or 2 depending on the iteration.
                    counter += delta_counter
        
                info(" @NM : Using %8d configurations along the +ve direction." % (counter,), verbosity.medium)

                counter = -1

                # Similarly displaces in the "-ve direction"
                while True:

                    # Displaces along the normal mode.
                    self.imm.dbeads.q = self.imm.beads.q + dev * counter

                    # Stores the "anharmonic" component of the potential
                    # and the force.
                    dv = dstrip(self.imm.dforces.pots).copy()[0] / self.nprim - 0.50 * self.imm.w2[step] * (nmd * counter)**2 - v0
                    df = np.dot(dstrip(self.imm.dforces.f).copy()[0], np.real(self.imm.V.T[step])) / self.nprim + self.imm.w2[step] * (nmd * counter)

                    # Adds to the list.
                    # Also stores the total energetics i.e. including 
                    # the harmonic component.
                    vlist.append(dv)
                    flist.append(df)
                    qlist.append(nmd * counter)  
        
                    # Bailout condition.
                    if self.nevib * self.imm.nmevib[step] < np.abs(0.50 * self.imm.w2[step] * (nmd * counter)**2 + dv):
                        break

                    # Increases the displacement by 1 or 2 depending on the iteration.
                    counter -= delta_counter
        
                info(" @NM : Using %8d configurations along the -ve direction." % (-counter,), verbosity.medium)
       
                # Fits cubic splines to data. 
                info("@NM : Fitting cubic splines.", verbosity.medium)
                vspline = interp1d(qlist, vlist, kind='cubic', bounds_error=False)

                # Converge wrt size of SHO basis
                bs_iter = 0
                bs_Aanh = [1e-20]
                bs_Eanh = [1e-20]

                # Calculates the 1D correction potential on a grid.
                qgrid = np.linspace(np.min(qlist), np.max(qlist), self.nint)
                vgrid = np.asarray([(np.nan_to_num(vspline(x))) for x in qgrid])
                
                while True:
                    nnbasis = max(1, self.nbasis - 5) + 5 * bs_iter

                    psigrid = np.zeros((nnbasis, self.nint))
                    # Calculates the wavefunctions on the grid.
                    for i in xrange(nnbasis):
                        psigrid[i] = self.psi(i, 1.0, self.imm.w[step], qgrid)
                        psigrid[i] = psigrid[i] / np.sqrt(np.dot(psigrid[i],psigrid[i]))

                    # Solves the Schroedinger's Equation.
                    bs_AEanh = self.solve_schroedingers_equation(self.imm.w[step], psigrid, vgrid)
                    bs_Aanh.append(bs_AEanh[0])
                    bs_Eanh.append(bs_AEanh[1])

                    info(" @NM : CONVERGENCE : fnmrms = %10.8e   nbasis = %5d    A =  %10.8e   D(A) =  %10.8e /  %10.8e" % (ffnmrms, nnbasis, bs_Aanh[-1], np.abs(bs_Aanh[-1] - bs_Aanh[-2]) / np.abs(bs_Aanh[-2]), self.athresh), verbosity.medium)

                    # Check whether anharmonic frequency is converged
                    if (np.abs(bs_Aanh[-1] - bs_Aanh[-2]) / np.abs(bs_Aanh[-2])) < self.athresh:
                        break

                    bs_iter += 1

                Aanh.append(bs_Aanh[-1])
                Eanh.append(bs_Eanh[-1])

                # Check whether anharmonic frequency is converged
                if (np.abs(Aanh[-1] - Aanh[-2]) / np.abs(Aanh[-2])) < self.athresh:
                    break

            # Prints the normal mode displacement, the potential and the force.
            outfile = self.imm.output_maker.get_output(self.imm.prefix + '.' + str(step) + '.qvf')
            np.savetxt(outfile,  np.c_[qlist, vlist, flist], header="Frequency = %10.8f" % self.imm.w[step])
            outfile.close()

        # prints the mapped potential.
        if self.imm.print_1b_map == True:
          output_grid = np.linspace(np.min(qlist), np.max(qlist), 100)
          outfile = self.imm.output_maker.get_output(self.imm.prefix + '.' + str(step) + '.vfit')
          np.savetxt(outfile,  np.c_[output_grid, vspline(output_grid)], header="Frequency = %10.8f" % self.imm.w[step])
          info(" @NM : Prints the mapped potential energy to %s" % (self.imm.prefix + '.' + str(step) + '.vfit'), verbosity.medium)
          outfile.close()

        # Done converging wrt size of SHO basis.
        # Calculates the harmonic free and internal energy.
        Ahar = -logsumexp( [ -1.0 * np.sqrt(self.imm.w2[step]) * (0.5+i) / dstrip(self.imm.temp) for i in range(nnbasis) ] ) * dstrip(self.imm.temp)
        Zhar =  np.sum([np.exp(-1.0 * np.sqrt(self.imm.w2[step]) * (0.5+i) / dstrip(self.imm.temp)) for i in range(nnbasis)])
        Ehar =  np.sum([np.sqrt(self.imm.w2[step]) * (0.5+i) * np.exp(  -1.0 * np.sqrt(self.imm.w2[step]) * (0.5+i) / dstrip(self.imm.temp)) for i in range(nnbasis)]) / Zhar

        info(' @NM : HAR frequency     =  %10.8e' % (self.imm.w[step],), verbosity.medium)
        info(' @NM : HAR free energy   =  %10.8e' % (Ahar,), verbosity.medium)
        info(' @NM : IMF free energy   =  %10.8e' % (Aanh[-1],), verbosity.medium)
        info('\n', verbosity.medium )
        self.total_anhar_free_energy += Aanh[-1]
        self.total_har_free_energy += Ahar
        self.total_anhar_internal_energy += Eanh[-1]
        self.total_har_internal_energy += Ehar

    def terminate(self):
        """
        Prints out the free and internal energy
        for HAR and IMF, and triggers a soft exit.
        """

        info(' @NM : Potential offset               =  %10.8e' % (self.v0,), verbosity.low)
        info(' @NM : HAR free energy                =  %10.8e' % (np.sum((0.5 * np.sqrt(self.imm.w2[self.imm.nz:]) + self.imm.temp * np.log(1.0 - np.exp(-np.sqrt(self.imm.w2[self.imm.nz:]) / self.imm.temp)))) / self.nprim + self.v0,), verbosity.low)
        info(' @NM : IMF free energy correction     =  %10.8e' % ((self.total_anhar_free_energy - self.total_har_free_energy) / self.nprim,), verbosity.low)
        info(' @NM : HAR internal energy            =  %10.8e' % (np.sum(np.sqrt(self.imm.w2[self.imm.nz:]) * (0.5 + 1.0 / (np.exp(np.sqrt(self.imm.w2[self.imm.nz:]) / self.imm.temp) -1))) / self.nprim + self.v0,), verbosity.low)
        info(' @NM : IMF internal energy correction =  %10.8e' % ((self.total_anhar_internal_energy -self.total_har_internal_energy) / self.nprim,), verbosity.low)
        info(' @NM : ALL QUANTITIES PER PRIMITIVE UNIT CELL (WHERE APPLICABLE) \n', verbosity.low)
        softexit.trigger(" @NM : The IMF calculation has terminated.")


class VSCF(IMF):
    """ 
    """

    def bind(self, imm):
        """ 
        Reference all the variables for simpler access.
        """

        super(VSCF, self).bind(imm)
        self.nz = self.imm.nz
        #self.print_2b_map = self.imm.print_2b_map
        #self.threebody = self.imm.threebody
        self.solve = self.imm.solve
        self.alpha = self.imm.alpha
        self.pair_range = self.imm.pair_range


        # Filenames for storing the number of samples configurations
        # and their potential energy.
        self.modes_filename = 'modes.dat'
        self.v_offset_prefix = 'potoffset'
        self.npts_pos_prefix = 'npts_pos'
        self.npts_neg_prefix = 'npts_neg'
        self.v_indep_file_prefix = 'vindep'
        self.v_indep_grid_file_prefix = 'vindep_grid'
        self.v_coupled_file_prefix = 'vcoupled'
        self.v_coupled_grid_file_prefix = 'vcoupled_grid'

        # Creates a list of modes with frequencies greater than 2 cm^-1.

        if os.path.exists(self.imm.output_maker.prefix + '.' + self.modes_filename):
            self.inms = np.loadtxt(self.imm.output_maker.prefix + '.' + self.modes_filename, dtype=int).tolist()
            print self.inms
        else:
            info(" @NM : Identifying relevant frequency modes.", verbosity.medium) 
            self.inms = []
            for inm in range(self.dof):
                
                if self.imm.w[inm] < 9.1126705e-06:
                    info(" @NM : Ignoring normal mode no.  %8d with frequency %15.8f cm^-1." % (inm, self.imm.w[inm] * 219474,) , verbosity.medium)
                    continue
                else:
                    self.inms.append(inm)

            # Save for use in VSCFSolver.
            outfile = self.imm.output_maker.get_output(self.modes_filename)
            np.savetxt(outfile,  self.inms, fmt='%i', header="Indices of modes that are considered in the calculation.")
            outfile.close()

        # Saves the total number of steps for automatic termination.
        ndof = len(self.inms)
        self.total_steps = ndof + ndof * (ndof - 1) / 2

        # Saves the indices of pairs of modes
        self.pair_combinations = list(combinations(self.inms, 2))

        # Selects the range of pair of modes to be calculated.
        if self.pair_range.size == 0:
            self.pair_range = np.asarray([0, len(self.pair_combinations)])

        print self.pair_range

        # Variables for storing the number of sampled configurations 
        # along the +ve and -ve displacements along normal modes and 
        # the sampled potential energy.
        self.npts = np.zeros(self.dof, dtype=int)
        self.npts_neg = np.zeros(self.dof, dtype=int)
        self.npts_pos = np.zeros(self.dof, dtype=int)
        self.v_indep_list = []

        if self.solve:
            self.q_grids = np.zeros((self.dof, self.nint))
            self.v_indep_grids = np.zeros((self.dof, self.nint))
            self.v_mft_grids = np.zeros((self.dof, self.nint))
            self.v_coupled_grids = np.zeros((self.dof, self.dof, self.nint, self.nint))

            self.psi_i_grids = np.zeros((self.dof, self.nbasis, self.nint))
            self.rho_grids = np.zeros((self.dof, self.nint))

            self.evecs_imf = np.zeros((self.dof, self.nbasis, self.nbasis))
            self.evals_imf = np.zeros((self.dof, self.nbasis))

            self.evals_vscf = np.zeros((self.dof, self.nbasis))
            self.evecs_vscf = np.zeros((self.dof, self.nbasis, self.nbasis))

    def step(self, step=None):
        """Computes the Born Oppenheimer curve along a normal mode."""

        # Performs some basic initialization.
        if step == 0:
            # Initialize overall potential offset
            self.v_offset_filename = self.v_offset_prefix + '.dat'
            if os.path.exists(self.imm.output_maker.prefix + '.' + self.v_offset_filename):
                self.v0 = np.loadtxt(self.imm.output_maker.prefix + '.' + self.v_offset_filename)
            else:
                self.v0 = dstrip(self.imm.forces.pots).copy()[0] / self.nprim
                outfile = self.imm.output_maker.get_output(self.v_offset_filename)
                np.savetxt(outfile,  [self.v0])
                outfile.close()

        # Maps 1D curves.
        elif step <= len(self.inms):

            # Selects the normal mode to map out.
            self.inm = self.inms[step - 1]

            # Defines the names of the files that store the sampled 
            # and the interpolated potential energy.
            self.v_indep_filename = self.v_indep_file_prefix + "." + str(self.inm) + ".dat"
            self.v_indep_grid_filename = self.v_indep_grid_file_prefix + "." + str(self.inm) + ".dat"

            info("\n @NM : Treating normal mode no.  %8d with frequency %15.8f cm^-1." % (self.inm, self.imm.w[self.inm] * 219474) ,verbosity.medium)

            # If the indepent modes are already calculated, just loads from file.
            if os.path.exists(self.imm.output_maker.prefix + '.' + self.v_indep_filename):

                # Reads the number of sampled configurations from the header
                # and the sampeld potential energy from the body.
                with open(self.imm.output_maker.prefix + '.' + self.v_indep_filename) as f:
                    header = [line.split() for line in f if line.startswith('#')][0]
                    self.npts_neg[self.inm] =  int(header[2])
                    self.npts_pos[self.inm] =  int(header[4])
                    self.npts[self.inms] = self.npts_neg[self.inms] + self.npts_pos[self.inms] + 1
                self.v_indep_list.append(np.loadtxt(self.imm.output_maker.prefix + '.' + self.v_indep_filename).T)
                info(" @NM : Loading the sampled potential energy for mode  %8d" % (self.inm,), verbosity.medium)
                info(" @NM : Using %8d configurations along the +ve direction." % (self.npts_pos[self.inm],), verbosity.medium)
                info(" @NM : Using %8d configurations along the -ve direction." % (self.npts_neg[self.inm],), verbosity.medium)

            # If mapping has NOT been perfomed previously, maps the 1D curves.
            else:

                self.npts_neg[self.inm], self.npts_pos[self.inm], v_indeps = self.one_dimensional_mapper(step)
                self.npts[self.inm] = self.npts_neg[self.inm] + self.npts_pos[self.inm] + 1
                self.v_indep_list.append(v_indeps)

                info(" @NM : Using %8d configurations along the +ve direction." % (self.npts_pos[self.inm],), verbosity.medium)
                info(" @NM : Using %8d configurations along the -ve direction." % (self.npts_neg[self.inm],), verbosity.medium)
                info(" @NM : Saving the sampled potential energy for mode  %8d in %s" % (self.inm, self.v_indep_filename), verbosity.medium)
                outfile = self.imm.output_maker.get_output(self.v_indep_filename)
                np.savetxt(outfile, v_indeps, header=" npts_neg: %10d npts_pos: %10d" % (self.npts_neg[self.inm], self.npts_pos[self.inm]))
                outfile.close()


            # We need the independent mode correction on a grid.
            # Checks if the potential exists otherwise loads from file.
            if self.solve:

                if os.path.exists(self.imm.output_maker.prefix + '.' + self.v_indep_grid_filename):
                    igrid, vigrid = np.loadtxt(self.imm.output_maker.prefix + '.' + self.v_indep_grid_filename).T
                else:
                    info(" @NM : Interpolating the potential energy on a grid of %8d points." % (self.nint,), verbosity.medium)
                    displacements_nmi = [self.fnmrms * self.nmrms[self.inm] * (-self.npts_neg[self.inm] + i - 2.0) for i in xrange(self.npts[self.inm] + 4)]
                    vspline = interp1d(displacements_nmi, self.v_indep_list[-1], kind='cubic', bounds_error=False)
                    igrid = np.linspace(-self.npts_neg[self.inm] * self.fnmrms * self.nmrms[self.inm], self.npts_pos[self.inm] * self.fnmrms * self.nmrms[self.inm], self.nint)
                    vigrid = np.asarray([np.asscalar(vspline(igrid[iinm]) - 0.5 * self.imm.w2[self.inm] * igrid[iinm]**2 - self.v0) for iinm in range(self.nint)])

                    # Save coupling correction to file for vistualisation.
                    info(" @NM : Saving the interpolated potential energy to %s" % (self.v_indep_grid_filename,), verbosity.medium)
                    outfile = self.imm.output_maker.get_output(self.v_indep_grid_filename)
                    np.savetxt(outfile, np.c_[igrid, vigrid])
                    outfile.close()

                # Stores the interpolated potential in memory.
                self.q_grids[self.inm][:] = igrid
                self.v_indep_grids[self.inm][:] = vigrid

        # Maps 2D surfaces if the index of the pair lies in range.
        elif step - len(self.inms) - 1 < self.pair_range[1]:

            # Checks if the index lies in range.
            if not self.pair_range[0] <= step - len(self.inms) - 1:
                return
 
            # Selects the normal mode pair to map out.
            vijgrid = None
            self.inm, self.jnm = self.pair_combinations[step - len(self.inms) - 1]
            self.inm_index, self.jnm_index = self.inm - self.nz, self.jnm - self.nz

            # Defines the names of the files that store the sampled 
            # and the interpolated potential energy.
            self.v_coupled_filename = self.v_coupled_file_prefix + "." + str(self.inm) + "." + str(self.jnm) + ".dat"
            self.v_coupled_grid_filename = self.v_coupled_grid_file_prefix + "." + str(self.inm) + "." + str(self.jnm) + ".dat"

            info("\n @NM : Treating normal modes no.  %8d  and %8d  with frequencies %15.8f cm^-1 and %15.8f cm^-1, respectively." % (self.inm, self.jnm, self.imm.w[self.inm] * 219474,  self.imm.w[self.jnm] * 219474) ,verbosity.medium)

            if os.path.exists(self.imm.output_maker.prefix + '.' + self.v_coupled_filename) != True:

                # Initializes the grid for interpolating the potential when 
                # displacements are made along pairs of normal modes.
                self.v_coupled = np.zeros((self.npts[self.inm] + 4) * (self.npts[self.jnm] + 4))

                # Calculates the displacements as linear combinations of displacements along independent modes.
                displacements_nmi = [self.fnmrms * self.nmrms[self.inm] * (-self.npts_neg[self.inm] + i - 2.0) for i in xrange(self.npts[self.inm] + 4)]
                displacements_nmj = [self.fnmrms * self.nmrms[self.jnm] * (-self.npts_neg[self.jnm] + j - 2.0) for j in xrange(self.npts[self.jnm] + 4)]

                # Calculates the potential energy at the displaced positions.
                k = 0
                didjv = []
                unit_displacement_nmi = np.real(self.imm.V.T[self.inm]) * np.sqrt(self.nprim)
                unit_displacement_nmj = np.real(self.imm.V.T[self.jnm]) * np.sqrt(self.nprim)
                info(" @NM : Sampling a total of %8d configurations." % (len(displacements_nmi) * len(displacements_nmj),), verbosity.medium)

                for i in xrange(self.npts[self.inm] + 4):
                    for j in xrange(self.npts[self.jnm] + 4):

                        # Uses on-axis potentials are available from 1D maps.
                        if (-self.npts_neg[self.inm] + i - 2) == 0:
                            self.v_coupled[k] = self.v_indep_list[self.jnm_index][j]
                        elif (-self.npts_neg[self.jnm] + j -2) == 0:
                            self.v_coupled[k] = self.v_indep_list[self.inm_index][i]
                        else:
                            self.imm.dbeads.q = dstrip(self.imm.beads.q) + displacements_nmi[i] * unit_displacement_nmi + displacements_nmj[j] * unit_displacement_nmj
                            self.v_coupled[k] = dstrip(self.imm.dforces.pots)[0] / self.nprim
                        didjv.append([displacements_nmi[i], displacements_nmj[j], self.v_coupled[k] - self.v0])
                        k += 1

                # Saves the displacements and the sampled potential energy.
                info(" @NM : Saving the sampled potential energy to %s." % (self.v_coupled_filename,), verbosity.medium)
                outfile = self.imm.output_maker.get_output(self.v_coupled_filename)
                np.savetxt(outfile, didjv)
                outfile.close()

            else:
                info(" @NM : Skipping the mapping for modes %8d and %8d." % (self.inm, self.jnm), verbosity.medium)
                if self.solve:
                    displacements_nmi, displacements_nmj, self.v_coupled = np.loadtxt(self.imm.output_maker.prefix + '.' + self.v_coupled_filename).T
                    self.v_coupled += self.v0

            # We need the pair-wise coupling correction on a grid.
            # Checks if the correction exists otherwise loads from file.
            if self.solve:

                if os.path.exists(self.imm.output_maker.prefix + '.' + self.v_coupled_grid_filename):
                    vijgrid = np.loadtxt(self.imm.output_maker.prefix + '.' + self.v_coupled_grid_filename)

                else:

                    # Interpolates the displacements on a grid and saves for VSCFSOLVER.
                    info(" @NM : Interpolating the potential energy on a %8d x %8d grid." % (self.nint,self.nint), verbosity.medium)
                    vtspl = interp2d(displacements_nmi, displacements_nmj, self.v_coupled, kind='cubic', bounds_error=False)
                    igrid = np.linspace(-self.npts_neg[self.inm] * self.fnmrms * self.nmrms[self.inm], self.npts_pos[self.inm] * self.fnmrms * self.nmrms[self.inm], self.nint)
                    jgrid = np.linspace(-self.npts_neg[self.jnm] * self.fnmrms * self.nmrms[self.jnm], self.npts_pos[self.jnm] * self.fnmrms * self.nmrms[self.jnm], self.nint)
                    vijgrid = (np.asarray( [ np.asarray( [ (vtspl(igrid[iinm],jgrid[ijnm]) - vtspl(igrid[iinm],0.0) - vtspl(0.0,jgrid[ijnm]) + vtspl(0.0,0.0)) for iinm in range(self.nint) ] ) for ijnm in range(self.nint) ] )).reshape((self.nint,self.nint))

                    # Also saves the independent mode corrections along the modes.
                    vigrid = np.asarray([np.asscalar(vtspl(igrid[iinm], 0.0) - 0.5 * self.imm.w2[self.inm] * igrid[iinm]**2 - vtspl(0.0,0.0)) for iinm in range(self.nint)])
                    vjgrid = np.asarray([np.asscalar(vtspl(0.0, jgrid[ijnm]) - 0.5 * self.imm.w2[self.jnm] * jgrid[ijnm]**2 - vtspl(0.0,0.0)) for ijnm in range(self.nint)])

                    # Save coupling correction to file for vistualisation.
                    info(" @NM : Saving the interpolated potential energy to %s" % (self.v_coupled_grid_filename,), verbosity.medium)
                    outfile = self.imm.output_maker.get_output(self.v_coupled_grid_filename)
                    np.savetxt(outfile, vijgrid)
                    outfile.close()
                
                # Saves the interpolated potential in memory.
                self.v_coupled_grids[self.inm, self.jnm][:] = vijgrid
                self.v_coupled_grids[self.jnm, self.inm][:] = vijgrid.T

        # Solves the SE once the mapping is finished.
        elif self.solve == True:
            self.solver()

        else:
            self.terminate()

    def solver(self):
        """
        Solves the VSCF equations in a mean-field manner.
        """

        # Initializes the independent mode free and internal energy.
        ai, ei = np.zeros(self.dof), np.zeros(self.dof)
        self.v_mft_grids = self.v_indep_grids.copy()

        info('\n', verbosity.medium)
        # Initializes the wavefunctions for all the normal modes.
        for inm in self.inms:

            for ibasis in xrange(self.nbasis):
                self.psi_i_grids[inm, ibasis, :] = self.psi(ibasis, 1.0, self.imm.w[inm], self.q_grids[inm])
                self.psi_i_grids[inm, ibasis, :] /= np.sqrt(np.sum(self.psi_i_grids[inm, ibasis, :]**2))

            ai[inm], ei[inm], self.evals_imf[inm], self.evecs_imf[inm] = self.solve_schroedingers_equation(self.imm.w[inm], self.psi_i_grids[inm], self.v_indep_grids[inm], True)

            info(' @NM : The IMF free energy of mode %8d is %10.8e' % (inm, ai[inm]), verbosity.medium)

        vscf_iter = 0
        a_imf = self.v0 + ai.sum()
        a_imf = ai.sum()
        a_vscf = a_imf
        self.evals_vscf, self.evecs_vscf = self.evals_imf.copy(), self.evecs_imf.copy()
        info(' @NM : The total IMF free energy is %10.8e' % (a_imf), verbosity.medium)
        info('\n @NM : The SCF begins.', verbosity.medium)

        while True:

            # Calculates the VSCF free energy.
            a_vscf_old = a_vscf
            a_vscf = self.v0 + ai.sum()
            a_vscf = ai.sum()
            vscf_iter += 1
            info(' @NM : COMVERGENCE : iteration = %8d   A =  %10.8e    D(A) = %10.8e / %10.8e' % (vscf_iter, a_vscf, np.absolute(a_vscf - a_vscf_old), self.athresh), verbosity.medium)

            # Calculates the thermal density for each normal mode.
            # This is essentially the square of the wave function 
            # times the Boltzmann density of each state.
            #info(' @NM : Calculating the thermal density.', verbosity.medium)
            self.rho_grids *= 0.0
            for inm in self.inms:
                for ibasis in xrange(self.nbasis):
                    self.rho_grids[inm] += np.exp(-1.0 * (self.evals_vscf[inm, ibasis] - self.evals_vscf[inm, 0]) / self.imm.temp) * np.dot(self.psi_i_grids[inm].T, self.evecs_vscf[inm, ibasis])**2
                    self.rho_grids[inm] /= self.rho_grids[inm].sum()

            # Calculates the mean field potential for each normal
            # mode and solves the Schroedinger's equation.
            for inm in self.inms:

                self.v_mft_grids[inm] = self.v_indep_grids[inm] + (1 - self.alpha) * self.v_mft_grids[inm]
                for jnm in self.inms:
                    # Not sure if this represents the total correction or not.
                    self.v_mft_grids[inm] += self.alpha * np.dot(self.v_coupled_grids[inm][jnm].T, self.rho_grids[jnm]) / self.nprim

                ai[inm], ei[inm], self.evals_vscf[inm], self.evecs_vscf[inm] = self.solve_schroedingers_equation(self.imm.w[inm], self.psi_i_grids[inm], self.v_mft_grids[inm], True)


            # Checks the convergence of the SCF procedure.
            if np.absolute((a_vscf - a_vscf_old) / a_vscf) < self.athresh and vscf_iter > 4:
                info("\n @NM : Convergence reached.", verbosity.medium)
                info(" @NM : IMF free energy             = %10.8e" % (a_imf / self.nprim), verbosity.low) 
                info(" @NM : VSCF free energy correction = %10.8e" % ((a_vscf - a_imf) / self.nprim), verbosity.low)
                info(' @NM : ALL QUANTITIES PER PRIMITIVE UNIT CELL (WHERE APPLICABLE) \n', verbosity.low)
                self.terminate()

    def one_dimensional_mapper(self, step):
        """
        Maps the potential energy landscape along a mode and returns
        the number of sampled points and the sampled potential energy.
        """

        # Determines sampling range for given normal mode
        nmd = self.fnmrms * self.imm.nmrms[self.inm]

        # Determines the displacement vector in Cartesian space.
        dev = np.real(self.imm.V.T[self.inm]) * nmd * np.sqrt(self.nprim)

        # Adds the minimum configuration to the list of sampled potential.
        v_indeps = [self.v0]

        # Displaces along the negative direction.
        counter = -1
        while True:

            self.imm.dbeads.q = self.imm.beads.q + dev * counter
            v = dstrip(self.imm.dforces.pots).copy()[0] / self.nprim
            v_indeps.append(v)

            # Bails out if the sampled potenital energy exceeds a user defined threshold.
            if self.nevib * self.imm.nmevib[self.inm] < np.absolute(v - self.v0):

                # Adds two extra points required later for solid spline fitting at edges.
                self.imm.dbeads.q -= dev
                v_indeps.append(dstrip(self.imm.dforces.pots).copy()[0] / self.nprim)
                self.imm.dbeads.q -= dev
                v_indeps.append(dstrip(self.imm.dforces.pots).copy()[0] / self.nprim)
                break

            counter -= 1

        r_npts_neg = -counter

        # invert vi so it starts with the potential for the most negative displacement and ends on the equilibrium
        v_indeps = v_indeps[::-1]

        counter = 1
        while True:

            self.imm.dbeads.q = self.imm.beads.q + dev * counter
            v = dstrip(self.imm.dforces.pots).copy()[0] / self.nprim
            v_indeps.append(v)

            # Bails out if the sampled potenital energy exceeds a user defined threshold.
            if self.nevib * self.imm.nmevib[self.inm] < np.absolute(v - self.v0) :

                # add two extra points required later for solid spline fitting at edges
                self.imm.dbeads.q += dev
                v_indeps.append(dstrip(self.imm.dforces.pots).copy()[0] / self.nprim)
                self.imm.dbeads.q += dev
                v_indeps.append(dstrip(self.imm.dforces.pots).copy()[0] / self.nprim)
                break

            counter += 1

        r_npts_pos = counter

        return r_npts_neg, r_npts_pos, v_indeps

    def terminate(self):
        """
        Triggers a soft exit.
        """

        softexit.trigger(" @NM : The VSCF calculation has terminated.")

