"""Interface with librascal to run machine learning potentials"""

import sys
from .dummy import Dummy_driver

from ipi.utils.mathtools import det_ut3x3
from ipi.utils.units import unit_to_internal, unit_to_user

#try:
    #from rascal.models.genericmd import GenericMDCalculator as RascalCalc
sys.path.insert(0,"/home/lopanits/chemlearning/alchemical-learning/")
from utils.driver import GenericMDCalculator as AlchemicalCalc
    #print(AlchemicalCalc())

#except:
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

        if len(arglist) == 2:
            self.model = 0 #arglist[0]
            #here params for soap
            self.template = arglist[1]
        else:
            sys.exit(self.error_msg)
        self.alchemical_calc = AlchemicalCalc(self.model, True, self.template)

    def __call__(self, cell, pos):
        """Get energies, forces, and stresses from the librascal model"""
        pos_calc = unit_to_user("length", "angstrom", pos)
        # librascal expects ASE-format, cell-vectors-as-rows
        cell_calc = unit_to_user("length", "angstrom", cell.T)
        # Do the actual calculation
        print("pos_calc",pos_calc.shape )
        pot, force, stress = self.alchemical_calc.calculate(pos_calc, cell_calc)
        print("hee")
        print("pot, force, stress" ,pot, force, stress)
        pot_ipi = unit_to_internal("energy", "electronvolt", pot)
        force_ipi = unit_to_internal("force", "ev/ang", force)
        # The rascal stress is normalized by the cell volume (in rascal units)
        vir_calc = -1 * stress * det_ut3x3(cell_calc)
        vir_ipi = unit_to_internal("energy", "electronvolt", vir_calc.T)
        extras = ""
        print("pot_ipi, force_ipi, vir_ipi, extras" ,pot_ipi, force_ipi, vir_ipi, extras)
        return pot_ipi, force_ipi, vir_ipi, extras
