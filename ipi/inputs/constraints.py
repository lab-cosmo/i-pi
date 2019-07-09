"""Creates objects that deal with constrained prorpagation."""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import numpy as np

from ipi.utils.depend import dstrip
from ipi.utils.inputvalue import Input, InputValue, InputAttribute, InputArray, input_default
from ipi.engine.constraints import ConstraintGroup


__all__ = ['InputConstraintGroup']

class InputConstraintGroup(Input):

    """ConstraintGroup input class.

       Handles parameters used in managing quasi-centroids of the same type
       (e.g. monatomic, triatomic, etc.)

       Attributes:
           name: a string specifying the type of quasi-centroid (compulsory)

       Fields:
           tol: an optional float specifying the tolerance for imposing the
                constraint with SHAKE/RATTLE. Defaults to 1.0e-06
           maxcycle: an optional integer specifying the maximum number of
                iterations in a SHAKE/RATTLE cycle. Defaults to 100
           indices: An array of integer indices specifying the atoms that are
                    subject to the constraint (compulsory)
    """
    attribs = {"name": (InputAttribute, {"dtype": str,
                                     "options": ["triatomic"],
                                     "help": "The type of quasi-centroid. 'triatomic' means a molecular fragment with a bent triatomic geometry"
                                     })}
    fields = {"tol": (InputValue, {"dtype": float,
                                   "default": 1.0e-06,
                                   "help": "The tolerance threshold for enforcing the set of constraints with SHAKE/RATTLE",
                                   }),
              "maxcycle": (InputValue, {"dtype": int,
                                   "default": 100,
                                   "help": "The maximum number of iterations of SHAKE/RATTLE",
                                   }),
              "indices": (InputArray, {"dtype": int,
                                 "default": input_default(factory=np.zeros,
                                                          kwargs={'shape': (0,), 'dtype': int}),
                                 "help": "Specifies the indices of the atoms subject to the set of constraints that define the present quasi-centroid type."}),
               }
    default_help = "Deals with quasi-centroids of the same type"
    default_label = "CONSTRAINTGROUP"

    def store(self, cgp):
        """Takes a ConstraintGroup instance and stores a minimal representation of it.

        Args:
            cgp: A ConstraintGroup object from which to initialise
        """

        super(InputConstraintGroup, self).store()
        self.name.store(cgp.name)
        self.tol.store(cgp.tol)
        self.maxcycle.store(cgp.maxcycle)
        self.indices.store(dstrip(cgp.indices))

    def fetch(self):
        """Creates a ConstraintGroup object.

        Returns:
            A constraint-group object of the appropriate type and with the appropriate
            properties given the attributes of the InputConstraintGroup object.

        """
        super(InputConstraintGroup, self).fetch()
        cgp = ConstraintGroup(self.name.fetch(),
                              self.indices.fetch(),
                              self.tol.fetch(),
                              self.maxcycle.fetch())
        return cgp