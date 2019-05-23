"""Contains the class that deals with quasi-centroids.

Deals with quasi-centroid dynamics, including the conversion between
quasi-centroid Cartesian and curviliner forces/coordinates, mass-scaling
of the underlying ring-polymer.

NOTE: this is currently written for the specific case of a bent triatomic,
with the storage convention (for e.g. water) O H H O H H ...
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2019 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np
from ipi.utils.depend import depend_value, dd, dobject
from constraints import ConstraintGroup

class QCForces(dobject):
    """
    A minimal quasi-centroid force object that deals with the conversion
    of bead forces.
    """
    pass

class QuasiCentroids(dobject):

    """Handles the ring-polymer quasi-centroids.

    Transformation to/from quasi-centroid coordinates, mass matrix scaling,
    ring-polymer propagation under the quasi-centroid constraints, etc

    Attributes:
        beads: A beads object giving the atoms' configuration
        forces: A forces object giving the forces acting on each bead
        nm: An object that performs normal-mode transformations

    Depend objects:
        qc: The quasi-centroid configuration
        pc: The quasi-centroid momenta
        mc: The quasi-centroid masses
        forces: QCForces object that emulates parts of Forces

    Methods:
        scatter: scatter an array into disjoint sets subject grouped by
                 quasi-centroid type
        gather: gather the values partitioned by quasi-centroid type

    """

    def __init__(self, dt=1.0, thermostat=None):
        pass

    def bind(self, ens, beads, nm, bforce, prng):
        pass