"""Creates objects that deal with the different ensembles."""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np
import ipi.engine.thermostats
import ipi.engine.barostats
from ipi.utils.inputvalue import InputDictionary, InputAttribute, InputValue, \
        InputArray, input_default
from ipi.inputs.barostats import InputBaro
from ipi.inputs.thermostats import InputThermo
from ipi.utils.depend import dstrip

__all__ = []
    
class InputQuasiCentroidDynamics(InputDictionary):

    """QuasiCentroidDynamics input class.

    Handles generating the appropriate ensemble class from the xml input file,
    and generating the xml checkpoint tags and data from an instance of the
    object.
    """

    attribs = {
        "mode": (InputAttribute, {"dtype": str,
                                  "default": 'nve',
                                  "help": "The ensemble that will be sampled during the simulation. ",
                                  "options": ['nve', 'nvt']}),
        "splitting": (InputAttribute, {"dtype": str,
                                       "default": 'baoab',
                                       "help": "The integrator used for sampling the target ensemble. ",
                      "options": ['obabo', 'baoab']})
    }

    fields = {
        "thermostat": (InputThermo, {"default": input_default(factory=ipi.engine.thermostats.Thermostat),
                                     "help": "The thermostat for the atoms, keeps the atom velocity distribution at the correct temperature."}),
        "barostat": (InputBaro, {"default": input_default(factory=ipi.engine.barostats.Barostat),
                                 "help": InputBaro.default_help}),
        "timestep": (InputValue, {"dtype": float,
                                  "default": 1.0,
                                  "help": "The time step.",
                                  "dimension": "time"}),
        "nmts": (InputArray, {"dtype": int,
                              "default": np.zeros(0, int),
                              "help": "Number of iterations for each MTS level (including the outer loop, that should in most cases have just one iteration)."}),
        "pqc": (InputArray, {"dtype": float,
                              "default": np.zeros(0, float),
                              "help": "Quasicentroid momenta."})
    }

    default_help = "Holds all the information for the MD integrator, such as timestep, the thermostats and barostats that control it."
    default_label = "QUASICENTROID_DYNAMICS"

    def store(self, dyn):
        """Takes an integrator instance and stores a minimal representation of it.

        Args:
            dyn: An integrator object.
        """

        if dyn == {}:
            return

        self.mode.store(dyn.enstype)
        self.splitting.store(dyn.splitting)
        self.timestep.store(dyn.dt)
        self.thermostat.store(dyn.thermostat)
        self.barostat.store(dyn.barostat)
        self.nmts.store(dyn.nmts)
        self.pqc.store(dstrip(dyn.quasi.p).flatten())

    def fetch(self):
        """Creates a ConstrainedDynamics object.

        Returns:
            An ensemble object of the appropriate mode and with the appropriate
            objects given the attributes of the InputEnsemble object.
        """

        rv = super(InputQuasiCentroidDynamics, self).fetch()
        rv["mode"] = self.mode.fetch()
        rv["splitting"] = self.splitting.fetch()

        return rv