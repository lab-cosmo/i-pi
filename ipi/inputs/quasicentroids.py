"""Creates objects that compose and apply forces."""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import ipi.engine.thermostats
from ipi.engine.quasicentroids import QuasiCentroids
from ipi.engine.constraints import ConstraintGroup
from ipi.utils.inputvalue import Input, InputValue, InputArray, input_default
from ipi.inputs.constraints import InputConstraintGroup
from ipi.inputs.thermostats import InputThermo
from ipi.utils.depend import dstrip
import numpy as np

__all__ = ['InputQuasiCentroids']

class InputQuasiCentroids(Input):

    """Deals with creating the quasi-centroids objects for a QCMD simulation

    Attributes:
        nfree: number of steps into which the propagation of the ring-polymers is split
        gamma: degree of adiabatic separation

    Fields:
        q: quasi-centroid positions
        p: quasi-centroid momenta
        thermostat: the thermostat applied to the quasi-centroids

    Dynamic fields:
       socket: Socket object to create the server socket.
    """
    attribs = {"nfree": (InputValue, {"dtype": int,
                                       "default": 1,
                                       "help": "The splitting of the ring-polymer propagation step"}),
              }
    fields = {"q": (InputArray, {"dtype": float,
                                 "default": input_default(factory=np.zeros, args=(0,)),
                                 "help": "The positions of the quasi-centroids. In an array of size [3*natoms].",
                                 "dimension": "length"}),
              "p": (InputArray, {"dtype": float,
                                 "default": input_default(factory=np.zeros, args=(0,)),
                                 "help": "The momenta of the quasi-centroids. In an array of size [3*natoms].",
                                 "dimension": "momentum"}),
              "thermostat": (InputThermo, {"default": input_default(factory=ipi.engine.thermostats.Thermostat),
                                     "help": "The thermostat for the atoms, keeps the atom velocity distribution at the correct temperature."})
               }
    dynamic = {"quasicentroid": (InputConstraintGroup,
                                 {"help": InputConstraintGroup.default_help})
               }
    default_help = "Deals with creating the quasi-centroids."
    default_label = "QUASICENTROIDS"

    def store(self, quasi):
        """Stores a minimal representation of the quasi-centroids object.

        Args:
           quasi: A quasi-centroids object from which to initialise
        """

        super(InputQuasiCentroids, self).store()
        self.nfree.store(quasi.nfree)
        self.q.store(dstrip(quasi.q))
        self.p.store(dstrip(quasi.p))
        self.thermostat.store(quasi.thermostat)
        self.extra = []
        for cgp in quasi.qclist:
            icgp = InputConstraintGroup()
            icgp.store(cgp)
            self.extra.append(("quasicentroid", icgp))

    def fetch(self):
        """Creates a QuasiCentroids object

        Returns:
           A QuasiCentroids object that contained individual ConstraintGroups
           specs, intended to partition the system byb quasi-centroid type.
        """

        super(InputQuasiCentroids, self).fetch()
        qclist = [q.fetch() for (n,q) in self.extra]
        quasi = QuasiCentroids(nfree=self.nfree.fetch(),
                               thermostat=self.thermostat.fetch(),
                               qclist=qclist)
        # initialise the quasi-centroid positions and momenta whenever possible
        q = self.q.fetch()
        if (q.shape == (3 * quasi.natoms)):
            quasi.q = q
        elif len(q) != 0:
            raise ValueError("Array shape mismatches for q in <quasicentroids> input.")
        p = self.p.fetch()
        if (p.shape == (3 * quasi.natoms)):
            quasi.p = p
        elif len(p) != 0:
            raise ValueError("Array shape mismatches for p in <quasicentroids> input.")

        return quasi