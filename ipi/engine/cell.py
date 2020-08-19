"""Classes which deal with the system box.

Used for implementing the minimum image convention.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import numpy as np

from ipi.utils.depend import *
from ipi.utils.mathtools import *


__all__ = ["Cell"]


class Cell(dobject):

    """Base class to represent the simulation cell in a periodic system.

    This class has the base attributes required for either flexible or
    isotropic cell dynamics. Uses an upper triangular lattice vector matrix to
    represent the cell.

    Depend objects:
       h: An array giving the lattice vector matrix.
       ih: An array giving the inverse of the lattice vector matrix.
       V: The volume of the cell.
    """

    def __init__(self, h=None):
        """Initialises base cell class.

        Args:
           h: Optional array giving the initial lattice vector matrix. The
              reference cell matrix is set equal to this. Must be an upper
              triangular 3*3 matrix. Defaults to a 3*3 zeroes matrix.
        """

        if h is None:
            h = np.zeros((3, 3), float)

        dself = dd(self)  # gets a direct-access view to self

        dself.h = depend_array(name="h", value=h)
        dself.ih = depend_array(
            name="ih",
            value=np.zeros((3, 3), float),
            func=self.get_ih,
            dependencies=[dself.h],
        )
        dself.h0 = dself.h.copy()
        dself.ih0 = dself.get_ih().copy()
        dself.strain = depend_array(
            name="strain",
            value=np.zeros((3, 3), float),
            func=self.get_strain,
            dependencies=[dself.h],
        )
        dself.h0 = dself.h.copy()
        dself.V = depend_value(name="V", func=self.get_volume, dependencies=[dself.h])

    def copy(self):
        return Cell(dstrip(self.h).copy())

    def get_ih(self):
        """Inverts the lattice vector matrix."""

        return invert_ut3x3(self.h)

    def get_strain(self):
        """Computes the strain using the initial cell as the reference."""

        return np.dot(self.h, self.ih0) - np.eye(3)

    def get_volume(self):
        """Calculates the volume of the system box."""

        return det_ut3x3(self.h)

    def apply_pbc(self, atom):
        """Uses the minimum image convention to return a particle to the
           unit cell.

        Args:
           atom: An Atom object.

        Returns:
           An array giving the position of the image that is inside the
           system box.
        """

        s = np.dot(self.ih, atom.q)

        for i in range(3):
            s[i] = s[i] - round(s[i])

        return np.dot(self.h, s)

    def array_pbc(self, pos):
        """Uses the minimum image convention to return a list of particles to the
           unit cell.

        Args:
           atom: An Atom object.

        Returns:
           An array giving the position of the image that is inside the
           system box.
        """

        s = dstrip(pos).copy()
        s.shape = (len(pos) // 3, 3)

        s = np.dot(dstrip(self.ih), s.T)
        s = s - np.round(s)

        s = np.dot(dstrip(self.h), s).T

        pos[:] = s.reshape((len(s) * 3))

    def get_scaled_positions(self, pos):
        """Returns scaled positions in internal coordinates."""

        s = pos.copy()
        s_return = dstrip(pos).copy()
        s.shape = (int(pos.shape[1] / 3), 3)
        s = np.dot(dstrip(self.ih), s.T).T
        s_return[:]= s.reshape((len(s) * 3)) 
        return s_return

    def get_absolute_positions(self, pos):
        """Returns scaled positions in internal coordinates."""

        s = pos.copy()
        s_return = dstrip(pos).copy()
        s.shape = (int(pos.shape[1] / 3), 3)
        s = np.dot(s, dstrip(self.h).T)
        s_return[:] = s.reshape((1, len(s) * 3)) 
        return s_return

    def minimum_distance(self, atom1, atom2):
        """Takes two atoms and tries to find the smallest vector between two
        images.

        This is only rigorously accurate in the case of a cubic cell,
        but gives the correct results as long as the cut-off radius is defined
        as smaller than the smallest width between parallel faces even for
        triclinic cells.

        Args:
           atom1: An Atom object.
           atom2: An Atom object.

        Returns:
           An array giving the minimum distance between the positions of atoms
           atom1 and atom2 in the minimum image convention.
        """

        s = np.dot(self.ih, atom1.q - atom2.q)
        for i in range(3):
            s[i] -= round(s[i])
        return np.dot(self.h, s)
    def positions_abs_to_scaled(self, pos, forces=None):
        """Uses the minimum image convention to return a list of particles to the
           unit cell.

        Args:
           atom: An Atom object.

        Returns:
           An array giving the position of the image that is inside the
           system box.
        """
        s = dstrip(pos).copy()
        s_return = dstrip(pos).copy()
        s.shape = (pos.shape[1] / 3, 3)
        s = np.dot(dstrip(self.ih), s.T).T
        s_return[:]= s.reshape((len(s) * 3))
        return s_return
    def positions_scaled_to_abs(self, pos):
        """Uses the minimum image convention to return a list of particles to the
           unit cell.

        Args:
           atom: An Atom object.

        Returns:
           An array giving the position of the image that is inside the
           system box.
        """

        s = dstrip(pos).copy()
        s_return = dstrip(pos).copy()
        s.shape = (pos.shape[1] / 3, 3)
        s = np.dot(s, dstrip(self.h).T)
        s_return[:] = s.reshape((len(s) * 3))
        return s_return
