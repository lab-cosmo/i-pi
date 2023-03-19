""" Small functions/classes providing access to driver PES to be called from driver.py """

from .dummy import Dummy_driver
from .harmonic import Harmonic_driver
from .rascal import Rascal_driver
from .equisolve import Equisolve_driver

__all__ = ["__drivers__", "Dummy_driver", "Harmonic_driver", "Rascal_driver", "Equisolve_driver"]

# dictionary linking strings
__drivers__ = {
    "dummy": Dummy_driver,
    "harmonic": Harmonic_driver,
    "rascal": Rascal_driver,
    "equisolve": Equisolve_driver,
}
