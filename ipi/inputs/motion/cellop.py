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

__all__ = ['InputCellop']


class InputCellop(InputDictionary):

    """Geometry optimization options.

    Contains options related with geometry optimization, such as method,
    thresholds, linear search strategy, etc.

    """

    attribs = {"mode": (InputAttribute, {"dtype": str, "default": "bfgs",
                                         "help": "The geometry cell optimization algorithm to be used",
                                         "options": ['bfgs']})}

    # options of the method (mostly tolerances)
    fields = {"ls_options": (InputDictionary, {"dtype": [float, int, float, float],
                                               "help": """"Options for line search methods. Includes:
                              tolerance: stopping tolerance for the search (as a fraction of the overall energy tolerance),
                              iter: the maximum number of iterations,
                              step: initial step for bracketing,
                              adaptive: whether to update initial step.
                              """,
                                               "options": ["tolerance", "iter", "step", "adaptive"],
                                               "default": [1, 100, 1e-3, 1.0],
                                               "dimension": ["undefined", "undefined", "length", "undefined"]}),
              "tolerances": (InputDictionary, {"dtype": float,
                                               "options": ["energy", "force", "position"],
                                               "default": [1e-7, 1e-4, 1e-3],
                                               "help": "Convergence criteria for optimization. Default values are extremely conservative. Set them to appropriate values for production runs.",
                                               "dimension": ["energy", "force", "length"]}),
              "biggest_step": (InputValue, {"dtype": float,
                                            "default": 100.0,
                                            "help": "The maximum step size for (L)-BFGS line minimizations."}),
              "scale_lbfgs": (InputValue, {"dtype": int,
                                           "default": 2,
                                           "help": """Scale choice for the initial hessian.
                                            0 identity.
                                            1 Use first member of position/gradient list.
                                            2 Use last  member of position/gradient list."""}),

              # re-start parameters, estimate hessian, etc.
              "old_pos": (InputArray, {"dtype": float,
                                       "default": input_default(factory=np.zeros, args=(0,)),
                                       "help": "The previous positions in an optimization step.",
                                       "dimension": "length"}),
              "old_pot": (InputArray, {"dtype": float,
                                       "default": input_default(factory=np.zeros, args=(0,)),
                                       "help": "The previous potential energy in an optimization step.",
                                       "dimension": "energy"}),
              "old_force": (InputArray, {"dtype": float,
                                         "default": input_default(factory=np.zeros, args=(0,)),
                                         "help": "The previous force in an optimization step.",
                                         "dimension": "force"}),
              "old_direction": (InputArray, {"dtype": float,
                                             "default": input_default(factory=np.zeros, args=(0,)),
                                             "help": "The previous direction in a CG or SD optimization."}),
              "invhessian_bfgs": (InputArray, {"dtype": float,
                                               "default": input_default(factory=np.eye, args=(0,)),
                                               "help": "Approximate inverse Hessian for BFGS, if known."}),
              "hessian_trm": (InputArray, {"dtype": float,
                                           "default": input_default(factory=np.eye, args=(0,)),
                                           "help": "Approximate Hessian for trm, if known."}),
              "tr_trm": (InputArray, {"dtype": float,
                                      "default": input_default(factory=np.zeros, args=(0,)),
                                      "help": "The trust radius in trm.",
                                      "dimension": "length"}),
              "qlist_lbfgs": (InputArray, {"dtype": float,
                                           "default": input_default(factory=np.zeros, args=(0,)),
                                           "help": "List of previous position differences for L-BFGS, if known."}),
              "glist_lbfgs": (InputArray, {"dtype": float,
                                           "default": input_default(factory=np.zeros, args=(0,)),
                                           "help": "List of previous gradient differences for L-BFGS, if known."})
              }

    dynamic = {}

    default_help = "A Cell Optimization class implementing only BFGS"
    default_label = "CELLOP"

    def store(self, cellop):
        if cellop == {}:
            return

        self.mode.store(cell.mode)
        self.tolerances.store(cellop.tolerances)

        if cellop.mode == "bfgs":
            self.old_direction.store(cellop.d)
            self.invhessian_bfgs.store(cellop.invhessian)
            self.biggest_step.store(cellop.big_step)
        else:
            print("this mode doesn't exist")

    def fetch(self):
        rv = super(InputCellop, self).fetch()
        rv["mode"] = self.mode.fetch()
        return rv
