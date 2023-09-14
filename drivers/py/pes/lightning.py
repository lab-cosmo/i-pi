import sys
import os
import json

# Get the driver path from an environment variable
LIGHTNING_CALCULATOR_PATH = os.environ.get("LIGHTNING_CALCULATOR_PATH")
if LIGHTNING_CALCULATOR_PATH is not None:
    sys.path.append(LIGHTNING_CALCULATOR_PATH)

from .dummy import Dummy_driver

from ipi.utils.mathtools import det_ut3x3
from ipi.utils.units import unit_to_internal, unit_to_user

from ipi_calculator import PytorchLightningCalculator


class Lightning_driver(Dummy_driver):
    def __init__(self, args=None):
        self.error_msg = """Lightning driver requires specification of a .json model file fitted with librascal, 
                            and a template file that describes the chemical makeup of the structure. 
                            Example: python driver.py -m rascal -u -o example.chk,template.xyz"""

        super().__init__(args)

        if PytorchLightningCalculator is None:
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
            self.model = arglist[0]
            self.template = arglist[1]
        else:
            sys.exit(self.error_msg)
        self.alchemical_calc = PytorchLightningCalculator(self.model, self.template)

    def __call__(self, cell, pos):
        """Get energies, forces, and stresses from the librascal model"""
        pos_calc = unit_to_user("length", "angstrom", pos)
        # librascal expects ASE-format, cell-vectors-as-rows
        cell_calc = unit_to_user("length", "angstrom", cell.T)
        # Do the actual calculation

        pot, force, stress, committee = self.alchemical_calc.calculate(
            pos_calc, cell_calc
        )

        pot_ipi = unit_to_internal("energy", "electronvolt", pot)
        force_ipi = unit_to_internal("force", "ev/ang", force.reshape(-1, 3))

        # rascaline returns the stress in energy units already (i.e. as dV/deps)
        # TODO: implement actual virial calculation
        vir_calc = stress
        vir_ipi = unit_to_internal("energy", "electronvolt", vir_calc.T)

        if len(committee) == 0:
            extras = ""
        else:
            extras = dict(committee_pot=[], committee_force=[], committee_virial=[])

            for p, f, v in committee:
                extras["committee_pot"].append(
                    unit_to_internal("energy", "electronvolt", p)
                )
                extras["committee_force"].append(
                    unit_to_internal("force", "ev/ang", f).tolist()
                )
                extras["committee_virial"].append(
                    unit_to_internal("energy", "electronvolt", v).tolist()
                )

            extras = json.dumps(extras)

        return pot_ipi, force_ipi, vir_ipi, extras
