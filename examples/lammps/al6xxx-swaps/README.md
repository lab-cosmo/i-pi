Atom swap for Al-6XXX with vacancies
====================================

This example shows how to run a simulation with atomic swaps, including dummy 
atoms that allow to simulate vacancies.

The idea is that i-PI starts from an on-lattice configuration, in which the vacancies
are mapped to atoms (in this example V atoms). 
These atoms are mapped out before sending them to the calculator. This is achieved by
setting and `<activelist>`  array as part of the socket options. 
The calculator (in this case LAMMPS) has an input that contains only the physical atoms.
In this way, the V atoms do not really exist outside i-PI.
Given that in this way the dummy atoms feel no force, it is necessary to also fix their
coordinates, otherwise the'd perform a random walk under the action of the 
MD integrator. This is achieved using `<fixatoms>` as part of the `<motion>` class. 

Monte Carlo swaps
-----------------

The `<atomswap>` motion class performs swaps between atomic positions. This allows to
relax the ordering of atoms in the alloy, and also allows to move vacancies across the
system. This however means that vacancy atoms will take the position of physical atoms,
eventually making the system drift and leading to zero probability of vacancy motion.


Lattice swaps
-------------

By defining a `<reference_lattice>` section, that points to a structure file that gives
the ideal position of the lattice positions for the atoms, the `<atomswap>` class acts
by swapping the atomic displacements relative to these lattice positions, rather than
the absolute atomic positions. This increases the acceptance, and avoids drift, although
it still relies on the assumptions that atoms will not move from the neighborhood 
of their associated lattice site. If a physical atom jumps into a vacancy site, the 
acceptance probability will essentially go to zero, because that atom will have a very 
large displacement and won't swap into a lattice position. 