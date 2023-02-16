"""Interface with librascal to run machine learning potentials"""

import sys
from .dummy import Dummy_driver

from ipi.utils.mathtools import det_ut3x3
from ipi.utils.units import unit_to_internal, unit_to_user

sys.path.insert(0, "./alchemical-learning/")  # <-- git clone the repo in this folder
from driver import GenericMDCalculator as AlchemicalCalc

# print(AlchemicalCalc())

# except:
#    AlchemicalCalc = None


class Alchemical_driver(Dummy_driver):
    def __init__(self, args=None):
        self.error_msg = """Rascal driver requires specification of a .json model file fitted with librascal, 
                            and a template file that describes the chemical makeup of the structure. 
                            Example: python driver.py -m rascal -u -o model.json,template.xyz"""

        super().__init__(args)

        if AlchemicalCalc is None:
            raise ImportError("Couldn't load librascal bindings")

    def check_arguments(self):
        """Check the arguments required to run the driver

        This loads the potential and atoms template in librascal
        """
        try:
            arglist = self.args.split(",")
        except ValueError:
            sys.exit(self.error_msg)

        if len(arglist) == 3:
            self.model = arglist[0]
            self.hypers = arglist[1]
            # here params for soap
            self.template = arglist[2]
        else:
            sys.exit(self.error_msg)
        self.alchemical_calc = AlchemicalCalc(self.model, self.hypers, self.template)

    def __call__(self, cell, pos):
        """Get energies, forces, and stresses from the librascal model"""
        pos_calc = unit_to_user("length", "angstrom", pos)
        # librascal expects ASE-format, cell-vectors-as-rows
        cell_calc = unit_to_user("length", "angstrom", cell.T)
        # Do the actual calculation
        pot, force, stress = self.alchemical_calc.calculate(pos_calc, cell_calc)
        pot_ipi = unit_to_internal("energy", "electronvolt", pot)
        force_ipi = unit_to_internal("force", "ev/ang", force.reshape(-1, 3))

        print("pot, force, stress", pot, force_ipi.shape, stress.shape)
        # rascaline returns the stress in energy units already (i.e. as dV/deps)
        vir_calc = stress
        vir_ipi = unit_to_internal("energy", "electronvolt", vir_calc.T)
        extras = ""

        # print("pot_ipi, force_ipi, vir_ipi, extras" ,pot_ipi, force_ipi, vir_ipi, extras)
        return pot_ipi, force_ipi, vir_ipi, extras
