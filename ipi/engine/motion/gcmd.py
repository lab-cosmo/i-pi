"""Contains classes for planetary model calculations"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

from calendar import c
from pyexpat import native_encoding
import time
import numpy as np
from ipi.utils import sparse

from ipi.engine.motion import Motion, Dynamics
from ipi.utils.depend import *
from ipi.engine.thermostats import *
from ipi.utils.units import Constants
from ipi.utils.mathtools import root_herm
from ipi.utils.io import netstring_encoded_savez
from ipi.utils.messages import verbosity, info


def save_pmf_data(file, labels, cell, positions, potential, forces, virial):
    """Writes information about the PMF in a netstring-encoded file"""

    netstring_encoded_savez(
        file,
        compressed=True,
        labels=labels,
        cell=cell,
        positions=positions,
        potential=potential,
        forces=forces,
        virial=virial,
    )


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

    def __init__(
        self,
        timestep,
        mode="cmd",
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
        dself.dt = depend_value(name="dt", value=timestep)
        self.fixatoms = np.asarray([])
        self.fixcom = True

        # nvt-cc means contstant-temperature with constrained centroid
        # this basically is a motion class that will be used to make the
        # centroid propagation at each time step within the sampling loop
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

        # these are the "physical" beads and force objects. they should be left in a
        # "consistent" state so that the simulation samples a canonical ring polymer
        # distribution
        self.basebeads = beads
        self.basenm = nm
        self.baseforce = bforce
        self.basens = ens

        # we now create an "auxiliary system" which we can use to sample the constrained
        # centroid distribution. we can play fast and loose with these, because they do
        # not affect the distribution of the base classes
        # copies of all of the helper classes that are needed to bind the ccdyn object
        self.dbeads = beads.copy(nbeads=self.nbeads)
        self.dcell = cell.copy()
        self.dforces = bforce.copy(self.dbeads, self.dcell)
        self.dens = ens.copy()
        self.dbias = ens.bias.copy(self.dbeads, self.dcell)

        # TODO: give possibility of tuning the dynamical masses.
        self.dnm = nm.copy()
        """
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
        """

        self.dnm.bind(ens, self, beads=self.dbeads, forces=self.dforces)
        self.dnm.qnm[:] = (
            nm.qnm[: self.nbeads] * np.sqrt(self.nbeads) / np.sqrt(beads.nbeads)
        )
        self.dens.bind(
            self.dbeads, self.dnm, self.dcell, self.dforces, self.dbias, omaker
        )

        self.natoms = self.dbeads.natoms
        natoms3 = self.dbeads.natoms * 3

        # allocate space to save the bead fluctuations covariance matrix
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

        # creates files for outputting stuff
        self.omaker = omaker
        self.fcmdpmf = omaker.get_output("cmd_pmf")
        if self.mode == "gcmd":
            self.fgcmdpmf = omaker.get_output("gcmd_pmf")
        self.mean_pot = 0
        self.mean_f = np.zeros(natoms3)
        self.mean_vir = np.zeros((3, 3))
        self.g_mean_pot = 0
        self.g_mean_f = np.zeros(natoms3)
        self.g_mean_vir = np.zeros((3, 3))

    def increment(self):

        # accumulates an estimate of the covariance of the bead positions
        # around the centroid
        qc = dstrip(self.dbeads.qc)
        q = dstrip(self.dbeads.q)

        for b in range(self.dbeads.nbeads):
            dq = q[b] - qc
            self.cmdcov += np.tensordot(dq, dq, axes=0)

        self.mean_f += dstrip(self.dforces.f).sum(axis=0)
        self.mean_pot += dstrip(self.dforces.pots).sum(axis=0)
        self.mean_vir += dstrip(self.dforces.virs).sum(axis=0)

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
        # sij = (sij < self.screen ** 2).astype(float)
        sij = np.exp(-sij / (self.screen ** 2))
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

        # call only every stride steps, useful to save just the PMF without integrating,
        # in combination with a normal MD class
        if step is not None and step % self.stride != 0:
            return

        # Initialize positions to the actual positions, TODO: add possibly of contraction?
        if self.nbeads != self.basebeads.nbeads:
            raise ValueError("RPC not implemented for GCMD. Use same nbeads")

        self.dnm.qnm[:] = self.basenm.qnm[:]
        """(
            self.basenm.qnm[: self.nbeads]
            * np.sqrt(self.nbeads)
            / np.sqrt(self.basebeads.nbeads)
        )"""

        # Randomized momenta
        self.dnm.pnm = (
            self.prng.gvec((self.dbeads.nbeads, 3 * self.dbeads.natoms))
            * np.sqrt(self.dnm.dynm3)
            * np.sqrt(self.dens.temp * self.dbeads.nbeads * Constants.kb)
        )
        self.dnm.pnm[0] = 0.0

        # stoopid velocity verlet part 1
        self.basenm.pnm[0] += self.mean_f * self.dt * 0.5 * np.sqrt(self.nbeads)
        self.basenm.qnm[0] += (
            self.basenm.pnm[0] / dstrip(self.basebeads.m3)[0] * self.dt
        )
        self.dnm.qnm[0] = self.basenm.qnm[0]

        # Resets the frequency matrix and the PMF
        self.cmdcov[:] = 0.0
        self.mean_pot = 0.0
        self.mean_f[:] = 0.0
        self.mean_vir[:] = 0.0

        # sample by constrained-centroid dynamics
        for istep in range(self.nsamples):
            self.tmc -= time.time()
            self.ccdyn.step(istep)
            self.tmc += time.time()
            self.tmtx -= time.time()
            self.increment()
            self.tmtx += time.time()

        self.neval += 1

        # saves the position and momentum NM
        dqnm = dstrip(self.dnm.qnm).copy()
        dpnm = dstrip(self.dnm.pnm).copy()

        # computes the mean force and potential. these are computed as the raw
        # sum over beads and sampling steps, while we want to get the mean PMF
        self.mean_f /= self.dbeads.nbeads * self.nsamples
        self.mean_pot /= self.dbeads.nbeads * self.nsamples
        self.mean_vir /= self.dbeads.nbeads * self.nsamples

        self.cmdcov /= self.dbeads.nbeads * self.nsamples

        self.tsave -= time.time()

        if self.screen > 0.0:
            scr = self.matrix_screen()
            self.cmdcov *= scr

        # ensure perfect symmetry
        self.cmdcov[:] = 0.5 * (self.cmdcov + self.cmdcov.transpose())

        # save PMF info
        save_pmf_data(
            self.fcmdpmf,
            self.dbeads.names,
            self.dcell.h,
            self.dbeads.qc,
            self.mean_pot,
            self.mean_f,
            self.mean_vir,
        )

        if self.mode == "gcmd":
            # now we re-sample around the centroid based on the accumulated covariance
            sqrt_cov = root_herm(self.cmdcov)
            self.g_mean_pot = 0
            self.g_mean_f[:] = 0.0
            self.g_mean_vir[:] = 0.0

            for istep in range(self.nsamples):
                # random Gaussian displacement
                dq = (
                    sqrt_cov
                    @ np.random.uniform(
                        size=(3 * self.dbeads.natoms, self.dbeads.nbeads)
                    )
                ).T
                self.dbeads.q = self.basebeads.qc + dq
                self.g_mean_f += dstrip(self.dforces.f).sum(axis=0)
                self.g_mean_pot += dstrip(self.dforces.pots).sum(axis=0)
                self.g_mean_vir += dstrip(self.dforces.virs).sum(axis=0)
            self.g_mean_f /= self.dbeads.nbeads * self.nsamples
            self.g_mean_pot /= self.dbeads.nbeads * self.nsamples
            self.g_mean_vir /= self.dbeads.nbeads * self.nsamples
            print(f"CMD mean pot {self.mean_pot}, GCMD mean pot {self.g_mean_pot}")
            save_pmf_data(
                self.fgcmdpmf,
                self.dbeads.names,
                self.dcell.h,
                self.dbeads.qc,
                self.g_mean_pot,
                self.g_mean_f,
                self.g_mean_vir,
            )
            f_pmf = self.g_mean_f
        else:  # do conventional CMD
            f_pmf = self.mean_f

        # stoopid velocity verlet, part 2
        # since we evolve in the normal mode basis, where nm[0] is sqrt(nbeads)
        # times the centroid, we have to scale the PMF accordingly
        self.basenm.pnm[0] += f_pmf * self.dt * 0.5 * np.sqrt(self.nbeads)
        self.basens.time += self.dt  # increments time

        # copy the higher normal modes from the CMD propagator to the physical system.
        # note that we use the values saved after the constrained-centroid sampling,
        # so that the distribution of the ring polymer should be consistent with the
        # PIMD canonical distribution even if we messed around with Gaussian resampling
        self.basenm.qnm[1:] = dqnm[1:]
        self.basenm.pnm[1:] = dpnm[1:]

        self.tsave += time.time()
        info(
            "@ GCMD Average timing: %f s, %f s, %f s\n"
            % (self.tmc / self.neval, self.tmtx / self.neval, self.tsave / self.neval),
            verbosity.high,
        )
