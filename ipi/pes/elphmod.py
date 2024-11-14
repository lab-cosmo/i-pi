"""Interface with [elphmod](https://github.com/janberges/elphmod) MD driver."""

import os
from .dummy import Dummy_driver

__DRIVER_NAME__ = "elphmod"
__DRIVER_CLASS__ = "ModelIIIDriver"


class ModelIIIDriver(Dummy_driver):
    """Wrapper around elphmod MD driver.
    A pickled driver instance is required (see elphmod.md.Driver.save).

    Example: python3 driver.py -u -m elphmod -o driver=driver.pickle
    """

    def __init__(self, *args, **kwargs):
        import elphmod

        filename = kwargs["driver"]
        if not os.path.exists(filename):
            raise ValueError(f"File '{filename}' does not exist.")
        self.driver = elphmod.md.Driver.load(filename)
        if self.driver is None:
            raise ValueError(
                f"Some error occured within `ModelIIIDriver.__init__` when trying to load the driver from file '{filename}'."
            )
        del kwargs["driver"]
        super().__init__(*args, **kwargs)
        pass

    def __call__(self, cell, pos):
        """Calculate energy and forces for given structure."""

        return self.driver(cell, pos)
