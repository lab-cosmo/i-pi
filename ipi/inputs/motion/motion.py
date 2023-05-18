"""Deals with creating the ensembles class.

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


Classes:
   InputEnsemble: Deals with creating the Ensemble object from a file, and
      writing the checkpoints.
"""

import numpy as np
from copy import copy
import ipi.engine.initializer
from ipi.utils.softexit import softexit

from ipi.engine.motion import (
    Motion,
    Dynamics,
    ConstrainedDynamics,
    Replay,
    GeopMotion,
    NEBMover,
    StringMover,
    DynMatrixMover,
    MultiMotion,
    AlchemyMC,
    InstantonMotion,
    TemperatureRamp,
    PressureRamp,
    AtomSwap,
    Planetary,
    AlKMC,
    SCPhononsMover,
    NormalModeMover,
)
from ipi.utils.inputvalue import *
from ipi.inputs.thermostats import *
from ipi.inputs.initializer import *
from .geop import InputGeop
from .instanton import InputInst
from .neb import InputNEB
from .stringmep import InputStringMEP
from .dynamics import InputDynamics
from .vscf import InputNormalMode
from .constrained_dynamics import InputConstrainedDynamics
from .phonons import InputDynMatrix
from .scphonons import InputSCPhonons
from .alchemy import InputAlchemy
from .atomswap import InputAtomSwap
from .planetary import InputPlanetary
from .ramp import InputTemperatureRamp, InputPressureRamp
from .al6xxx_kmc import InputAlKMC
from ipi.utils.units import *
from ipi.utils.messages import verbosity, warning

__all__ = ["InputMotion"]


class InputMotionBase(Input):

    """Motion calculation input class.

    A class to encompass the different "motion" calculations.

    Attributes:
       mode: An optional string giving the kind of motion calculation to be performed.

    Fields:
       fixcom: An optional boolean which decides whether the centre of mass
          motion will be constrained or not.
       fixatoms: A list of the indices of atoms that should not be moved.

    """

    attribs = {
        "mode": (
            InputAttribute,
            {
                "dtype": str,
                "help": "How atoms should be moved at each step in the simulatio. 'replay' means that a simulation is replayed from trajectories provided to i-PI.",
                "options": [
                    "vibrations",
                    "minimize",
                    "replay",
                    "neb",
                    "string",
                    "dynamics",
                    "constrained_dynamics",
                    "t_ramp",
                    "p_ramp",
                    "alchemy",
                    "atomswap",
                    "planetary",
                    "instanton",
                    "al-kmc",
                    "dummy",
                    "scp",
                    "normalmodes",
                ],
            },
        )
    }

    fields = {
        "fixcom": (
            InputValue,
            {
                "dtype": bool,
                "default": True,
                "help": "This describes whether the centre of mass of the particles is fixed.",
            },
        ),
        "fixatoms": (
            InputArray,
            {
                "dtype": int,
                "default": np.zeros(0, int),
                "help": "Indices of the atmoms that should be held fixed.",
            },
        ),
        "optimizer": (
            InputGeop,
            {"default": {}, "help": "Option for geometry optimization"},
        ),
        "neb_optimizer": (
            InputNEB,
            {"default": {}, "help": "Option for NEB optimization"},
        ),
        "string_optimizer": (
            InputStringMEP,
            {
                "default": {},
                "help": "Option for String minimal-energy path optimization",
            },
        ),
        "dynamics": (
            InputDynamics,
            {"default": {}, "help": "Option for (path integral) molecular dynamics"},
        ),
        "constrained_dynamics": (
            InputConstrainedDynamics,
            {
                "default": {},
                "help": "Option for constrained classical molecular dynamics",
            },
        ),
        "file": (
            InputInitFile,
            {
                "default": input_default(
                    factory=ipi.engine.initializer.InitFile, kwargs={"mode": "xyz"}
                ),
                "help": "This describes the location to read a trajectory file from. "
                "Replay syntax allows using some POSIX wildcards in the filename "
                "of trajectory files. If symbols ?*[] are found in a filename, "
                "the code expects to find exactly Nbeads files that match "
                "the provided pattern. Bead indices will be read from the files, "
                "and the files will be ordered ascendingly by their bead indices. "
                "Wildcarded files are expected to be in the folder "
                "where the simulation runs.",
            },
        ),
        "vibrations": (
            InputDynMatrix,
            {"default": {}, "help": "Option for phonon computation"},
        ),
        "normalmodes": (
            InputNormalMode,
            {
                "default": {},
                "help": "Option for solving the vibrational Schroedinger's equations in normal mode coordinates.",
            },
        ),
        "scp": (
            InputSCPhonons,
            {"default": {}, "help": "Option for self consistent phonons computation"},
        ),
        "alchemy": (
            InputAlchemy,
            {"default": {}, "help": "Option for alchemical exchanges"},
        ),
        "atomswap": (
            InputAtomSwap,
            {"default": {}, "help": "Option for Monte Carlo atom swap"},
        ),
        "t_ramp": (
            InputTemperatureRamp,
            {"default": {}, "help": "Option for temperature ramp"},
        ),
        "p_ramp": (
            InputPressureRamp,
            {"default": {}, "help": "Option for pressure ramp"},
        ),
        "instanton": (
            InputInst,
            {"default": {}, "help": "Option for Instanton optimization"},
        ),
        "al6xxx_kmc": (InputAlKMC, {"default": {}, "help": "Option for Al-6xxx KMC"}),
        "planetary": (
            InputPlanetary,
            {"default": {}, "help": "Option for planetary model calculator"},
        ),
    }
    dynamic = {}

    default_help = "Allow chosing the type of calculation to be performed. Holds all the information that is calculation specific, such as geometry optimization parameters, etc."
    default_label = "MOTION"

    def store(self, sc):
        """Takes a motion calculation instance and stores a minimal representation of it.

        Args:
           sc: A motion calculation class.
        """

        super(InputMotionBase, self).store(sc)
        tsc = -1
        if type(sc) is Motion:
            self.mode.store("dummy")
        elif type(sc) is Replay:
            self.mode.store("replay")
            tsc = 0
        elif type(sc) is GeopMotion:
            self.mode.store("minimize")
            self.optimizer.store(sc)
            tsc = 1
        elif type(sc) is NEBMover:
            self.mode.store("neb")
            self.neb_optimizer.store(sc)
            tsc = 1
        elif type(sc) is StringMover:
            self.mode.store("string")
            self.string_optimizer.store(sc)
            tsc = 1
        elif type(sc) is Dynamics:
            self.mode.store("dynamics")
            self.dynamics.store(sc)
            tsc = 1
        elif type(sc) is ConstrainedDynamics:
            self.mode.store("constrained_dynamics")
            self.constrained_dynamics.store(sc)
            tsc = 1
        elif type(sc) is DynMatrixMover:
            self.mode.store("vibrations")
            self.vibrations.store(sc)
            tsc = 1
        elif type(sc) is SCPhononsMover:
            self.mode.store("scp")
            self.scp.store(sc)
            tsc = 1
        elif type(sc) is NormalModeMover:
            self.mode.store("normalmodes")
            self.normalmodes.store(sc)
            tsc = 1
        elif type(sc) is AlchemyMC:
            self.mode.store("alchemy")
            self.alchemy.store(sc)
            tsc = 1
        elif type(sc) is AtomSwap:
            self.mode.store("atomswap")
            self.atomswap.store(sc)
            tsc = 1
        elif type(sc) is InstantonMotion:
            self.mode.store("instanton")
            self.instanton.store(sc)
            tsc = 1
        elif type(sc) is Planetary:
            self.mode.store("planetary")
            self.planetary.store(sc)
            tsc = 1
        elif type(sc) is TemperatureRamp:
            self.mode.store("t_ramp")
            self.t_ramp.store(sc)
            tsc = 1
        elif type(sc) is PressureRamp:
            self.mode.store("p_ramp")
            self.p_ramp.store(sc)
        elif type(sc) is AlKMC:
            self.mode.store("al-kmc")
            self.al6xxx_kmc.store(sc)
            tsc = 1
        else:
            raise ValueError("Cannot store Mover calculator of type " + str(type(sc)))

        if (sc.fixcom is True) and (len(sc.fixatoms) > 0):
            warning(
                "Fixed atoms break translational invariance, and so should be used with <fixcom> False </fixcom>.",
                verbosity.low
            )

        if tsc == 0:
            self.file.store(sc.intraj)
        elif tsc > 0:
            self.fixcom.store(sc.fixcom)
            self.fixatoms.store(sc.fixatoms)

    def fetch(self):
        """Creates a motion calculator object.

        Returns:
           An ensemble object of the appropriate mode and with the appropriate
           objects given the attributes of the InputEnsemble object.
        """

        super(InputMotionBase, self).fetch()

        if self.mode.fetch() == "replay":
            sc = Replay(
                fixcom=self.fixcom.fetch(),
                fixatoms=self.fixatoms.fetch(),
                intraj=self.file.fetch(),
            )
        elif self.mode.fetch() == "minimize":
            sc = GeopMotion(
                fixcom=self.fixcom.fetch(),
                fixatoms=self.fixatoms.fetch(),
                **self.optimizer.fetch()
            )
        elif self.mode.fetch() == "neb":
            #            raise ValueError(
            #                "The nudged elastic band calculation has been "
            #                "temporarily disabled until further bug-fixes."
            #            )
            sc = NEBMover(
                fixcom=self.fixcom.fetch(),
                fixatoms=self.fixatoms.fetch(),
                **self.neb_optimizer.fetch()
            )
        elif self.mode.fetch() == "string":
            softexit.trigger(
                status="bad",
                message=(
                    "String method is experimental: not guaranteed to work correctly "
                    "and makes twice more force calls than it is expected to.\n"
                    "We stop here. If you want to proceed, comment out this trigger."
                ),
            )
            sc = StringMover(
                fixcom=self.fixcom.fetch(),
                fixatoms=self.fixatoms.fetch(),
                **self.string_optimizer.fetch()
            )
        elif self.mode.fetch() == "dynamics":
            sc = Dynamics(
                fixcom=self.fixcom.fetch(),
                fixatoms=self.fixatoms.fetch(),
                **self.dynamics.fetch()
            )
        elif self.mode.fetch() == "constrained_dynamics":
            sc = ConstrainedDynamics(
                fixcom=self.fixcom.fetch(),
                fixatoms=self.fixatoms.fetch(),
                **self.constrained_dynamics.fetch()
            )
        elif self.mode.fetch() == "vibrations":
            sc = DynMatrixMover(
                fixcom=self.fixcom.fetch(),
                fixatoms=self.fixatoms.fetch(),
                **self.vibrations.fetch()
            )
        elif self.mode.fetch() == "normalmodes":
            sc = NormalModeMover(
                fixcom=self.fixcom.fetch(),
                fixatoms=self.fixatoms.fetch(),
                **self.normalmodes.fetch()
            )
        elif self.mode.fetch() == "scp":
            sc = SCPhononsMover(
                fixcom=self.fixcom.fetch(),
                fixatoms=self.fixatoms.fetch(),
                **self.scp.fetch()
            )
        elif self.mode.fetch() == "alchemy":
            sc = AlchemyMC(
                fixcom=self.fixcom.fetch(),
                fixatoms=self.fixatoms.fetch(),
                **self.alchemy.fetch()
            )
        elif self.mode.fetch() == "atomswap":
            sc = AtomSwap(
                fixcom=self.fixcom.fetch(),
                fixatoms=self.fixatoms.fetch(),
                **self.atomswap.fetch()
            )
        elif self.mode.fetch() == "instanton":
            sc = InstantonMotion(
                fixcom=self.fixcom.fetch(),
                fixatoms=self.fixatoms.fetch(),
                **self.instanton.fetch()
            )
        elif self.mode.fetch() == "planetary":
            sc = Planetary(
                fixcom=self.fixcom.fetch(),
                fixatoms=self.fixatoms.fetch(),
                **self.planetary.fetch()
            )
        elif self.mode.fetch() == "t_ramp":
            sc = TemperatureRamp(**self.t_ramp.fetch())
        elif self.mode.fetch() == "p_ramp":
            sc = PressureRamp(**self.p_ramp.fetch())
        elif self.mode.fetch() == "al-kmc":
            sc = AlKMC(
                fixcom=self.fixcom.fetch(),
                fixatoms=self.fixatoms.fetch(),
                **self.al6xxx_kmc.fetch()
            )
        else:
            sc = Motion()
            # raise ValueError("'" + self.mode.fetch() + "' is not a supported motion calculation mode.")

        return sc


class InputMotion(InputMotionBase):

    """Extends InputMotionBase to allow the definition of a multimotion"""

    attribs = copy(InputMotionBase.attribs)

    attribs["mode"][1]["options"].append("multi")

    dynamic = {
        "motion": (
            InputMotionBase,
            {
                "default": input_default(factory=Motion),
                "help": "A motion class that can be included as a member of a 'multi' integrator.",
            },
        )
    }

    def store(self, motion):
        if type(motion) is MultiMotion:
            self.mode.store("multi")

            if len(self.extra) != len(motion.mlist):
                self.extra = [0] * len(motion.mlist)

            for ii, m in enumerate(motion.mlist):
                if self.extra[ii] == 0:
                    im = InputMotionBase()
                    im.store(m)
                    self.extra[ii] = ("motion", im)
                else:
                    self.extra[ii][1].store(m)
        else:
            super(InputMotion, self).store(motion)

    def fetch(self):
        if self.mode.fetch() == "multi":
            mlist = []
            for k, m in self.extra:
                mlist.append(m.fetch())
            motion = MultiMotion(motionlist=mlist)
        else:
            motion = super(InputMotion, self).fetch()

        return motion
