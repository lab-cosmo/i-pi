"""Creates objects that deal with constrained prorpagation."""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import numpy as np

from ipi.utils.inputvalue import Input, InputValue
from ipi.engine.constraints import Constraints


__all__ = ['InputConst']


class InputConst(Input):

    """Constraints input class.

    Handles parameters used in constrained propagation

    Fields:
       tol: Tolerance threshold for RATTLE
       maxcycle: Maximum number of iterations in RATTLE
       nfree: Number of steps into which ring-polymer propagation
              under spring forces is split at the inner-most mts level.
    """


    fields = {"tol": (InputValue, {"default": 1.0e-06,
                                   "dtype": float,
                                   "help": "Constraint tolerance threshold for the RATTLE"}),
              "maxcycle": (InputValue, {
                                 "default": 100,
                                 "dtype": int,
                                 "help": "Maximum number of iterations in RATTLE"}),
              "nfree": (InputValue, {
                                 "default": 1,
                                 "dtype": int,
                                 "help": "Spltting of the free constrained RP propagation step."})
              }

    default_help = "Specifies constrained propagation parameters."
    default_label = "CONSTRAINTS"

    def store(self, const):
        """Takes a barostat instance and stores a minimal representation of it.

        Args:
           baro: A Constraints object.
        """

        super(InputConst, self).store(const)
        self.tol.store(const.tol)
        self.maxcycle.store(const.maxcycle)
        self.nfree.store(const.nfree)

    def fetch(self):
        """Creates a Constraints object.
        """

        super(InputConst, self).fetch()
        const = Constraints(
                tol=self.tol.fetch(),
                maxcycle=self.maxcycle.fetch(),
                nfree=self.nfree.fetch()
                            )
        return const
