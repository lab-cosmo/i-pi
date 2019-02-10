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

Inputs created by Michele Ceriotti and Benjamin Helfrecht, 2015

Classes:
   InputGeop: Deals with creating the Geop object from a file, and
      writing the checkpoints.
"""

import numpy as np
import ipi.engine.initializer
from ipi.engine.motion import *
from ipi.utils.inputvalue import *
from ipi.inputs.thermostats import *
from ipi.inputs.initializer import *
from ipi.utils.units import *

__all__ = ['InputSCPhonons']


class InputSCPhonons(InputDictionary):
    """Dynamic matrix calculation options.

       Contains options related to self consistent phonons method. 

    """

    attribs = {"mode": (InputAttribute, {"dtype": str, "default": "qn",
                                         "help": "The algorithm to be used",
                                         "options": ["qn", "cl"]})}
    fields = {
        "prefix": (InputValue, {"dtype": str, "default": "phonons",
                                "help": "Prefix of the output files."
                                }),
                "asr": (InputValue, {"dtype": str, "default": "none",
                                     "options": ["none", "crystal", "molecule", "internal"],
                                     "help": "Shift by this much the dynamical matrix in the output."
                                     }),
                "random_type": (InputValue, {"dtype": str, "default": "pseudo",
                                             "options": ["sobol", "pseudo"],
                                             "help": "Chooses the type of random numbers."
                                             }),
                "displace_mode": (InputValue, {"dtype": str, "default": "rewt",
                                               "options": ["rewt", "hessian", "nmik", "sD"],
                                               "help": "Chooses the type of optimisation strategy for the centroid."
                                               }),
                "dynmat": (InputArray, {"dtype": float,
                                        "default": np.zeros(0, float),
                                        "help": "Portion of the dynamical matrix known up to now."}),
                "dynmat_r": (InputArray, {"dtype": float,
                                          "default": np.zeros(0, float),
                                          "help": "Portion of the dynamical matrix known up to now (refining)."}),
                "max_steps": (InputValue, {
                    "dtype": int,
                                        "default": None,
                                        "help": "maximum number of Monte carlo steps."}),
                "max_iter": (InputValue, {
                    "dtype": int,
                                        "default": 1,
                                        "help": "maximum number of self consistent iterations."}),
                "tau": (InputValue, {
                    "dtype": float,
                                        "default": 1.,
                                        "help": "displacement scaling along the gradient."}),
                "widening": (InputValue, {
                    "dtype": float,
                                        "default": 1.,
                                        "help": "ratio of width of sampled distribution wrt the target distribution."}),
                "wthreshold": (InputValue, {
                    "dtype": float,
                                        "default": 0.5,
                                        "help": "threshold on minimum Boltzmann weights before more statistics must be accumulated."}),
                "precheck": (InputValue, {
                    "dtype": bool,
                                        "default": True,
                                        "help": "flag for checking statistical significance of forces before optimisation of centroid."}),
                "checkweights": (InputValue, {
                    "dtype": bool,
                                        "default": False,
                                        "help": "flag for checking Boltzmann weights for whether more statistics are required."}),
                "chop": (InputArray, {"dtype": float,
                                      "default": np.asarray([1e-09, 100]),
                                      "help": ""}),
    }

    dynamic = {}

    default_help = "Fill in."
    default_label = "PHONONS"

    def store(self, phonons):
        if phonons == {}: return
        self.mode.store(phonons.mode)
        self.prefix.store(phonons.prefix)
        self.asr.store(phonons.asr)
        self.dynmat.store(phonons.dynmatrix)
        self.dynmat_r.store(phonons.dynmatrix_r)
        self.chop.store(phonons.chop)
        self.max_steps.store(phonons.max_steps)
        self.max_iter.store(phonons.max_iter)
        self.tau.store(phonons.tau)
        self.random_type.store(phonons.random_type)
        self.displace_mode.store(phonons.displace_mode)

    def fetch(self):
        rv = super(InputSCPhonons, self).fetch()
        rv["mode"] = self.mode.fetch()
        return rv
