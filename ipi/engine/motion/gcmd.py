"""Contains classes for planetary model calculations"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

import time
import numpy as np
from ipi.utils import sparse

from ipi.engine.motion import Motion, Dynamics
from ipi.utils.depend import *
from ipi.engine.thermostats import *
from ipi.utils.units import Constants
from ipi.utils.io import netstring_encoded_savez
from ipi.utils.messages import verbosity, info


class GCMD(Motion):
    """Gaussian-constraint centroid molecular dynamics class. 
    Uses a nested motion class to evaluate the bead distribution around the 
    centroid, and then re-evaluates the potential over this distribution to
    compute a potential of mean force that is constrained to a Gaussian distribution.
    It then uses the mean force to evolve the centroid position.

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

## TODO: everything needs to be done from scratch!!

    def __init__(
        self,
        timestep,
        mode="md",
        nsamples=0,
        stride=1,
        screen=0.0,
        nbeads=-1,
        thermostat=None,
        barostat=None,
        fixcom=False,
        fixatoms=None,
        nmts=None,
    ):
        """Initialises a "dynamics" motion object.

        Args:
            dt: The timestep of the simulation algorithms.
            fixcom: An optional boolean which decides whether the centre of mass
                motion will be constrained or not. Defaults to False.
        """

        self.mode = mode
        self.nsamples = nsamples
        self.stride = stride
        self.nbeads = nbeads
        self.screen = screen

        dself = dd(self)

        # the planetary step just computes constrained-centroid properties so it
        # should not advance the timer
        dself.dt = depend_value(name="dt", value=timestep)
        # dset(self, "dt", depend_value(name="dt", value = 0.0) )
        self.fixatoms = np.asarray([])
        self.fixcom = True

        # nvt-cc means contstant-temperature with constrained centroid
        # this basically is a motion class that will be used to make the
        # centroid propagation at each time step
        self.ccdyn = Dynamics(
            timestep,
            mode="nvt-cc",
            thermostat=thermostat,
            nmts=nmts,
            fixcom=fixcom,
            fixatoms=fixatoms,
        )

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

        if self.nbeads < 0:
            self.nbeads = beads.nbeads
        self.prng = prng
        self.basebeads = beads
        self.basenm = nm
        self.baseforce = bforce
        

        # copies of all of the helper classes that are needed to bind the ccdyn object
        self.dbeads = beads.copy(nbeads=self.nbeads)
        self.dcell = cell.copy()
        self.dforces = bforce.copy(self.dbeads, self.dcell)

        # options for NM propagation - hardcoded frequencies unless using a GLE thermo
        # for which frequencies matter
        if isinstance(self.ccdyn.thermostat, (ThermoGLE, ThermoNMGLE, ThermoNMGLEG)):
            self.dnm = nm.copy()
            self.dnm.mode = "rpmd"
        else:
            self.dnm = nm.copy(
                freqs=nm.omegak[1]
                * self.nbeads
                * np.sin(np.pi / self.nbeads)
                * np.ones(self.nbeads - 1)
                / (beads.nbeads * np.sin(np.pi / beads.nbeads))
            )
            self.dnm.mode = "manual"

        self.dnm.bind(ens, self, beads=self.dbeads, forces=self.dforces)
        self.dnm.qnm[:] = (
            nm.qnm[: self.nbeads] * np.sqrt(self.nbeads) / np.sqrt(beads.nbeads)
        )
        self.dens = ens.copy()
        self.dbias = ens.bias.copy(self.dbeads, self.dcell)
        self.dens.bind(self.dbeads, self.dnm, self.dcell, self.dforces, self.dbias, omaker)

        self.natoms = self.dbeads.natoms
        natoms3 = self.dbeads.natoms * 3
        self.cmdcov = np.zeros((natoms3, natoms3), float)

        # initializes counters
        self.tmc = 0
        self.tmtx = 0
        self.tsave = 0
        self.neval = 0

        # finally, binds the ccdyn object
        self.ccdyn.bind(
            self.dens, self.dbeads, self.dnm, self.dcell, self.dforces, prng, omaker
        )

        self.omaker = omaker
        self.fcmdcov = omaker.get_output("cmdcov")
        self.mean_pot = 0
        self.mean_f = np.zeros(natoms3)

    def increment(self, dnm):

        # accumulates an estimate of the covariance of the bead positions
        # around the centroid
        qc = dstrip(self.dbeads.qc)
        q = dstrip(self.dbeads.q)
        
        for b in range(self.dbeads.nbeads):
            dq = q[b]-qc
            self.cmdcov += np.tensordot(dq, dq, axes=0)
            
        self.mean_f += dstrip(self.dforces.f).sum(axis=0)
        self.mean_pot += dstrip(self.dforces.pots).sum(axis=0)

    def matrix_screen(self):
        """Computes a screening matrix to avoid the impact of
        noisy elements of the covariance and frequency matrices for
        far-away atoms"""

        q = np.array(dstrip(self.dbeads.qc)).reshape(self.natoms, 3)
        sij = q[:, np.newaxis, :] - q
        sij = sij.transpose().reshape(3, self.natoms ** 2)
        # find minimum distances between atoms (rigorous for cubic cell)
        sij = np.matmul(self.dcell.ih, sij)
        sij -= np.around(sij)
        sij = np.matmul(self.dcell.h, sij)
        sij = sij.reshape(3, self.natoms, self.natoms).transpose()
        # take square magnitudes of distances
        sij = np.sum(sij * sij, axis=2)
        # screen with Heaviside step function
        sij = (sij < self.screen ** 2).astype(float)
        # sij = np.exp(-sij / (self.screen**2))
        # acount for 3 dimensions
        sij = np.concatenate((sij, sij, sij), axis=0)
        sij = np.concatenate((sij, sij, sij), axis=1)
        sij = sij.reshape(-1).reshape(-1, self.natoms).transpose()
        sij = sij.reshape(-1).reshape(-1, 3 * self.natoms).transpose()
        return sij

    def save_matrix(self, matrix):
        """Writes the compressed, sparse frequency matrix to a netstring encoded file"""

        sparse.save_npz(
            self.fcmdcov, matrix, saver=netstring_encoded_savez, compressed=True
        )

    def step(self, step=None):

        if step is not None and step % self.stride != 0:
            return

        # Initialize positions to the actual positions, possibly with contraction

        self.dnm.qnm[:] = (
            self.basenm.qnm[: self.nbeads]
            * np.sqrt(self.nbeads)
            / np.sqrt(self.basebeads.nbeads)
        )

        # Randomized momenta
        self.dnm.pnm = (
            self.prng.gvec((self.dbeads.nbeads, 3 * self.dbeads.natoms))
            * np.sqrt(self.dnm.dynm3)
            * np.sqrt(self.dens.temp * self.dbeads.nbeads * Constants.kb)
        )
        self.dnm.pnm[0] = 0.0


        # stoopid velocity verlet part 1
        ### CHECK THIS IS THE RIGHT CONVERSION BETWEEN CENTROID FORCE AND NM[0] FORCE!!!!
        print("mean_force", self.mean_f)
        self.basenm.pnm[0] += self.mean_f * self.dt*0.5 *np.sqrt(self.nbeads)
        print("momentum", np.linalg.norm(self.basenm.pnm[0]))        
        self.basenm.qnm[0] += self.basenm.pnm[0]/ dstrip(self.basebeads.m3)[0] * self.dt
        print("centroid ", self.basebeads.qc[:3])
        print("bead ", self.basebeads.q[0,:3])
        print("centroid nm ", self.dnm.qnm[0,:3])
        
        # Resets the frequency matrix and the PMF
        self.cmdcov[:] = 0.0
        self.mean_pot = 0.0
        self.mean_f[:] = 0.0

        self.tmtx -= time.time()
        self.increment(self.dnm)
        self.tmtx += time.time()

        # sample by constrained-centroid dynamics
        for istep in range(self.nsamples):
            self.tmc -= time.time()
            self.ccdyn.step(step)
            self.tmc += time.time()
            self.tmtx -= time.time()
            self.increment(self.dnm)
            self.tmtx += time.time()

        self.neval += 1

        self.cmdcov /= (
            self.dbeads.nbeads
            * (self.nsamples + 1)
        )
        self.tsave -= time.time()

        if self.screen > 0.0:
            scr = self.matrix_screen()
            self.cmdcov *= scr

        # ensure perfect symmetry
        self.cmdcov[:] = 0.5 * (self.cmdcov + self.cmdcov.transpose())
        # only save lower triangular part
        self.cmdcov[:] = np.tril(self.cmdcov)

        # save as a sparse matrix in half precision
        save_cmdcov = sparse.csc_matrix(self.cmdcov.astype(np.float16))

        # save the frequency matrix to the covariance file
        self.save_matrix(save_cmdcov)

        # computes the mean force and potential
        self.mean_f /= self.dbeads.nbeads * (self.nsamples + 1)
        self.mean_pot /= self.dbeads.nbeads * (self.nsamples + 1)
        
        # stoopid velocity verlet, part 2
        print("mean f", self.mean_f[:3])
        print("centroid nm, at end ", self.dnm.qnm[0,:3])
        print("centroid, at end ", self.basebeads.qc[:3])
        self.basenm.pnm[0] += self.mean_f * self.dt*0.5 *np.sqrt(self.nbeads)
        
        # copy the higher normal modes from the CMD propagator to the physical system
        self.basenm.qnm[1:] = self.dnm.qnm[1:]
        self.basenm.pnm[1:] = self.dnm.pnm[1:]
        print("bead, at end ", self.basebeads.q[0,:3])
        print("fin qui tutto bene")
        print("forces? ", self.baseforce.pot)
        print("fin qui tutto bene")
        
        self.tsave += time.time()
        info(
            "@ GCMD Average timing: %f s, %f s, %f s\n"
            % (self.tmc / self.neval, self.tmtx / self.neval, self.tsave / self.neval),
            verbosity.high,
        )
