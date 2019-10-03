"""Creates objects that deal with the different ensembles."""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np
#from copy import copy
import ipi.engine.thermostats
import ipi.engine.barostats
from ipi.engine.motion.constrained_dynamics import \
        BondLengthConstraint, BondAngleConstraint, \
        GroupedConstraints, EckartGroupedConstraints
from ipi.utils.inputvalue import InputDictionary, InputAttribute, InputValue, \
        InputArray, Input, input_default
from ipi.inputs.barostats import InputBaro
from ipi.inputs.thermostats import InputThermo
from ipi.utils.depend import dstrip

__all__ = ['InputConstrainedDynamics', 'InputConstraintBase', 'InputConstraintGroup']


class InputConstraintBase(Input):
    """An input class to define a constraint function. 
    """
    attribs = {
        "name": (InputAttribute, {
            "dtype": str,
            "help": "Type of constraint. ",
            "options": ["bondlength", "bondangle"]
            }),
        "tolerance": (InputAttribute, {
            "dtype": float,
            "default": 1.0e-04,
            "help": "The tolerance to which the constraint is converged. "
            }),
        "domain": (InputAttribute, {
            "dtype": str,
            "default": "cartesian",
            "help": "The coordinate domain for which the constraint is defined.",
            "options": ["cartesian", "normalmode", "centroid"]
            }),
              }
    fields = {
        "atoms": (InputArray, {
                "dtype": int,
                "default": np.zeros(0, int),
                "help": "List of atoms indices that are to be constrained."}),
        "values": (InputArray, {
                "dtype": float,
                "default": input_default(factory=np.zeros, args=(0,)),
                "dimension": "length",
                "help": "List of constraint lengths."})
              }
    def store(self, cnstr):
        super(InputConstraintBase, self).store()
        if type(cnstr) is BondLengthConstraint:
            self.name.store("bondlength")
        elif type(cnstr) is BondAngleConstraint:
            self.name.store("bondangle")
        self.atoms.store(dstrip(cnstr.ilist).flatten())
        self.tolerance.store(cnstr.tol)
        self.domain.store(cnstr.domain)
        self.values.store(dstrip(cnstr.targetvals).flatten())

    def fetch(self):
        name = self.name.fetch()
        tol = self.tolerance.fetch()
        domain = self.domain.fetch()
        alist = self.atoms.fetch()
        vlist = self.values.fetch()
        if name == "bondlength":
            cls = BondLengthConstraint
        elif name == "bondangle":
            cls = BondAngleConstraint
        if len(vlist) == 0:
            vlist = None
        return cls(alist, vlist, tol, domain)


class InputConstraintGroup(Input):
    """An input class to define a set of constraint function acting groups 
       of disjoint atoms.
    """

    attribs = {
        "eckart": (InputAttribute, {
            "dtype": bool,
            "default": False,
            "help": "Indicates whether the Eckart conditions are imposed onto the groups of constrained atoms."
            }),
        "ngroups": (InputAttribute, {
                "dtype": int,
                "help": "Number of groups of constrained atoms"}),
        "maxit": (InputAttribute, {
                "dtype": int,
                "default": 100,
                "help": "Maximum number of iterations to converge a constrained propagation step."})
            }
    fields = {
        "tolerance": (InputValue, {
                "dtype": float,
                "default": 1.0e-06,
                "help": "Converge criterion for the Eckart conditions"
                }),
        "qprev": (InputArray, {
                "dtype": float,
                "default":  np.zeros(0, float),
                "dimension": "length",
                "help": "Normal-mode coordinates from previous converged constrained propagation step."}),
        "qref": (InputArray, {
                "dtype": float,
                "default":  np.zeros(0, float),
                "dimension": "length",
                "help": "Reference configurations for the Eckart conditions"}),
        "atoms": (InputArray, {
                "dtype": int,
                "help": "Indices of the atoms affected by the constraints"})
              }
    dynamic = {"constraint" : (
            InputConstraintBase, 
            {"help" : "Define a constraint to be applied onto atoms"})
              }

    def store(self, cgp):
        self.ngroups.store(cgp.ngp)
        self.maxit.store(cgp.maxit)
        self.qprev.store(dstrip(cgp.qnmprev).flatten())
        self.atoms.store(cgp.iunique.flatten())
        if isinstance(cgp, EckartGroupedConstraints):
            self.qref.store(dstrip(cgp.qref).flatten())
            self.tolerance.store(cgp.tol[-1])
            self.eckart.store(True)
        else:
            self.eckart.store(False)
        self.extra = []
        for c in cgp.clist:
            iobj = InputConstraintBase()
            iobj.store(c)
            self.extra.append(("constraint", iobj))

    def fetch(self):
        """Creates a GroupedConstraints object.
        """
        ngp = self.ngroups.fetch()
        maxit = self.maxit.fetch()
        atoms = self.atoms.fetch()
        qnmprev = self.qprev.fetch()
        qref = self.qref.fetch()
        eckart = self.eckart.fetch()
        tol = self.tolerance.fetch()
        if (len(qnmprev)==0): 
            qnmprev = None
        if (len(qref)==0):
            qref = None
        clist = []
        for (k, c) in self.extra:
            if k == "constraint":
                clist.append(c.fetch())
            else:
                raise ValueError("Invalid entry "+k+" in constraint_group")
        if eckart is False:
            return GroupedConstraints(clist, atoms, ngp, maxit, qnmprev)
        else:
            return EckartGroupedConstraints(clist, atoms, ngp, maxit, 
                                            tol, qnmprev, qref)
    
class InputConstrainedDynamics(InputDictionary):

    """ConstrainedDynamics input class.

    Handles generating the appropriate ensemble class from the xml input file,
    and generating the xml checkpoint tags and data from an instance of the
    object.

    Attributes:
        mode: An optional string giving the mode (ensemble) to be simulated.
            Defaults to 'nve'.

    Fields:
        thermostat: The thermostat to be used for constant temperature dynamics.
        barostat: The barostat to be used for constant pressure or stress
            dynamics.
        timestep: An optional float giving the size of the timestep in atomic
            units. Defaults to 1.0.
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
        "nsteps_geo": (InputValue, {"dtype": int,
                                    "default": 1,
                                    "help": "The number of sub steps used in the evolution of the geodesic flow (used in function step_Ag)." })
    }

    dynamic = {"constraint_group" : 
        (InputConstraintGroup, 
         {"help" : "Define a set of constraints to be applied onto disjoint groups of atoms"})
              }

    default_help = "Holds all the information for the MD integrator, such as timestep, the thermostats and barostats that control it."
    default_label = "CONSTRAINED_DYNAMICS"

    def store(self, dyn):
        """Takes an ensemble instance and stores a minimal representation of it.

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
        self.nsteps_geo.store(dyn.nsteps_geo)

        self.extra = []

        for cgp in dyn.constraint_groups:
            iobj = InputConstraintGroup()
            iobj.store(cgp)
            self.extra.append( ("constraint_group", iobj) )

    def fetch(self):
        """Creates a ConstrainedDynamics object.

        Returns:
            An ensemble object of the appropriate mode and with the appropriate
            objects given the attributes of the InputEnsemble object.
        """

        rv = super(InputConstrainedDynamics, self).fetch()
        rv["mode"] = self.mode.fetch()
        rv["splitting"] = self.splitting.fetch()

        cnstr_list = []
        for (k, cgp) in self.extra:
            if k == "constraint_group":
                cnstr_list.append(cgp.fetch())
            else:
                raise ValueError("Invalid entry "+k+" in constrained_dynamics")

        rv["constraint_groups"] = cnstr_list

        return rv
