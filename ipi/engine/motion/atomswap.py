"""Contains the classes that deal with the MC exchanges of isotopes.

Holds the algorithms required for alchemical exchanges. Also calculates the
appropriate conserved energy quantity.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import numpy as np

from ipi.engine.motion import Motion
from ipi.utils.depend import *
from ipi.utils.units import Constants
from ipi.utils.io import read_file

class AtomSwap(Motion):

    """Swap atom positions (typically useful to exchange species in
    a way that is compatible with the i-PI encapsulation paradigm.

    Attributes:
        names of the species for exchanges
    """

    def __init__(
        self, fixcom=False, fixatoms=None, mode=None, names=[], nxc=1, 
        ealc=None, reference_lattice=None, lattice_idx=[],
        region_lattice=None, region_radius=None, region_force_change=False
    ):
        """Initialises a "alchemical exchange" motion object.

        Args:
            names : A list of isotopes
            nmc : frequency of doing exchanges

        """

        super(AtomSwap, self).__init__(fixcom=fixcom, fixatoms=fixatoms)

        self.names = names
        self.nxc = nxc

        dself = dd(self)
        dself.ealc = depend_value(name="ealc")
        if ealc is not None:
            self.ealc = ealc
        else:
            self.ealc = 0.0
                
        self.lattice_file = reference_lattice
        self.lattice_idx = lattice_idx
        if self.lattice_file is not None and self.lattice_file.value != "":
            mode = self.lattice_file.mode
            units = self.lattice_file.units
            cell_units = self.lattice_file.cell_units
            rfile = open(self.lattice_file.value, "r")
            ret = read_file(
                mode, rfile, dimension="length", units=units, cell_units=cell_units
            )
            self.reference_lattice = ret["atoms"].q.reshape((-1,3))
            if len(self.lattice_idx) == 0 :
                self.lattice_idx = np.array(list(range(len(self.reference_lattice))))
        else:
            self.reference_lattice = None

        self.region_file = region_lattice
        self.region_radius = region_radius
        self.region_force_change = region_force_change
        if self.region_file is not None and self.region_file.value != "":
            mode = self.region_file.mode
            units = self.region_file.units
            cell_units = self.region_file.cell_units
            rfile = open(self.region_file.value, "r")
            ret = read_file(
                mode, rfile, dimension="length", units=units, cell_units=cell_units
            )
            self.region_lattice = ret["atoms"].q.reshape((-1,3))
        else:
            self.region_lattice = None

    def bind(self, ens, beads, cell, bforce, nm, prng, omaker):
        """Binds ensemble beads, cell, bforce, and prng to the dynamics.

        This takes a beads object, a cell object, a forcefield object and a
        random number generator object and makes them members of the ensemble.
        It also then creates the objects that will hold the data needed in the
        ensemble algorithms and the dependency network. Note that the conserved
        quantity is defined in the init, but as each ensemble has a different
        conserved quantity the dependencies are defined in bind.

        Args:
            beads: The beads object from whcih the bead positions are taken.
            prng: The random number generator object which controls random number
                generation.
        """

        super(AtomSwap, self).bind(ens, beads, cell, bforce, nm, prng, omaker)
        self.ensemble.add_econs(dd(self).ealc)
        self.dbeads = self.beads.copy()
        self.dcell = self.cell.copy()
        self.dforces = self.forces.copy(self.dbeads, self.dcell)

    def AXlist(self, atomtype):
        """This compile a list of atoms ready for exchanges."""

        # selects the types of atoms for exchange
        atomexchangelist = []
        for i in range(self.beads.natoms):
            if self.beads.names[i] in atomtype:
                atomexchangelist.append(i)

        return np.asarray(atomexchangelist)

    def RXList(self, i, axlist):
        """ This prepares a list of atoms belonging to each region, for swapping """

        if self.region_lattice is None:
            raise ValueError("Region centers are undefined")
        
        # only select among the required atom types
        qij = dstrip(self.dbeads.q).reshape(-1,3)[axlist] - self.region_lattice[i]
        self.dcell.array_pbc(qij.flatten())
        d2ij=(qij**2).sum(axis=1)
        region_idx = np.where(d2ij<self.region_radius**2)[0]
        return region_idx, qij[region_idx]
    
    def MinContact(self, i):
        q = dstrip(self.dbeads.q).reshape(-1,3)
        qij = q - q[i]
        self.dcell.array_pbc(qij.flatten())
        d2ij=(qij**2).sum(axis=1)
        d2ij[i] = 1e100
        return np.sqrt(d2ij.min())

        
        

    def step(self, step=None):
        """Does one round of alchemical exchanges."""
        # picks number of attempted exchanges
        ntries = self.prng.rng.poisson(self.nxc)
        if ntries == 0:
            return

        nb = self.beads.nbeads
        axlist = self.AXlist(self.names)
        lenlist = len(axlist)
        if lenlist == 0:
            raise ValueError("Atoms exchange list is empty in MC atom swapper.")

        ## does the exchange
        betaP = 1.0 / (Constants.kb * self.ensemble.temp * nb)
        nexch = 0

        self.dcell.h = (
            self.cell.h
        )  # just in case the cell gets updated in the other motion classes
        
        for x in range(ntries):
            old_energy = self.forces.pot
            # updates local bead copy
            self.dbeads.q[:] = self.beads.q[:]
            
            if self.region_lattice is None:
                i = self.prng.rng.randint(lenlist)
                j = self.prng.rng.randint(lenlist)
                while self.beads.names[axlist[i]] == self.beads.names[axlist[j]]:
                    j = self.prng.rng.randint(lenlist)  # makes sure we pick a real exchange
                
                # pick actual atom indices
                i = axlist[i]
                j = axlist[j]

                # swap the atom positions
                if self.reference_lattice is None:
                    # just swaps atom positions
                    self.dbeads.q[:, 3 * i : 3 * i + 3] = self.beads.q[:, 3 * j : 3 * j + 3]
                    self.dbeads.q[:, 3 * j : 3 * j + 3] = self.beads.q[:, 3 * i : 3 * i + 3]
                else:
                    # uses lattice swap - atoms are attached to lattice sites, and the swap 
                    # preserves the distance to the lattice sites
                    l_idx_i = self.lattice_idx[i]
                    l_idx_j = self.lattice_idx[j]
                    self.dbeads.q[:, 3 * i : 3 * i + 3] = (self.reference_lattice[l_idx_j]+
                                                        self.beads.q[:, 3 * i : 3 * i + 3]
                                                        -self.reference_lattice[l_idx_i])                                                      
                    self.dbeads.q[:, 3 * j : 3 * j + 3] = (self.reference_lattice[l_idx_i]+
                                                        self.beads.q[:, 3 * j : 3 * j + 3]
                                                        -self.reference_lattice[l_idx_j])
            else:
                # we have a region lattice so we need to determine which 
                # atoms belon to the regions we are swapping
                region_change = False
                while not region_change:                
                    i = self.prng.rng.randint(len(self.region_lattice))
                    region_i = self.RXList(i, axlist)
                    
                    j = self.prng.rng.randint(len(self.region_lattice))
                    while i==j:
                        j = self.prng.rng.randint(len(self.region_lattice))  # makes sure we pick a real exchange                
                    region_j = self.RXList(j, axlist)
                    if self.region_force_change:
                        # check in # atoms changed
                        region_change = len(region_i[0]) != len(region_j[0])                        
                        if not region_change:
                            # check if atom types changed
                            labels_i = np.sort(self.beads.names[axlist[region_i[0]]])
                            labels_j = np.sort(self.beads.names[axlist[region_j[0]]])

                            region_change = not np.all(np.char.equal(labels_i,labels_j))
                    else:
                        region_change = True # just move on
                print(len(region_i[0]),len(region_j[0]), 
                      np.sort(self.beads.names[axlist[region_i[0]]]), np.sort(self.beads.names[axlist[region_j[0]]]),
                      np.linalg.norm(region_i[1]),np.linalg.norm(region_j[1]),
                       end=" :: " )
                
                new_q = dstrip(self.dbeads.q).copy().reshape(-1,3)
                
                new_q[axlist[region_i[0]]] = self.region_lattice[j] + region_i[1]
                new_q[axlist[region_j[0]]] = self.region_lattice[i] + region_j[1]
                self.dbeads.q[:] = new_q.flatten()

            new_energy = self.dforces.pot            
            pexchange = np.exp(-betaP * (new_energy - old_energy))
            print(" ", pexchange)

            for i in axlist[region_i[0]]:
                mc = self.MinContact(i)
                if mc<2:
                    print("XXXXX", end=" ")
                print(mc, end=" ")

            for i in axlist[region_j[0]]:
                mc = self.MinContact(i)
                if mc<2:
                    print("XXXXX", end=" ")
                print(mc, end=" ")
            print(" << contacts")

            # attemps the exchange, and actually propagate the exchange if something has happened
            if pexchange > self.prng.u:
                nexch += 1

                # copy the exchanged beads position
                self.beads.q[:] = self.dbeads.q[:]
                # transfers the (already computed) status of the force, so we don't need to recompute
                self.forces.transfer_forces(self.dforces)

                if self.reference_lattice is not None:
                    # swaps lattice indices too
                    self.lattice_idx[i], self.lattice_idx[j] = self.lattice_idx[j], self.lattice_idx[i]
                    l_idx_i = self.lattice_idx[i]
                    l_idx_j = self.lattice_idx[j]
                    
                self.ealc += -(new_energy - old_energy)
<<<<<<< HEAD
                print("EXCHANGE!")
=======
>>>>>>> db008e1612f3cb407b44dd8dc169a249226cce3f
        print("attempts", ntries, nexch)          
