Installing SIRIUS
=================

A Python 2.7 compatible branch of SIRIUS can be found on github_.

.. code:: bash

          git clone https://github.com/simonpintarelli/SIRIUS.git
          mkdir -p SIRIUS/build
          cd SIRIUS/build
          cmake -DCMAKE_INSTALL_PREFIX=$HOME/local/sirius \
            -DCREATE_PYTHON_MODULE=On \
            -DPYTHON2=Yes \
            ../
          make install

The SIRIUS Python module will be installed under the path specified above. The
environment settings file `env.sh <../../env.sh>`_ will append the environment
variable ``SIRIUS_PYTHON2`` to the ``PYTHONPATH``. For the installation
path specified above, add the following to your ``~/.bashrc`` or equivalent:

.. code:: bash

   export SIRIUS_PYTHON2=$HOME/local/sirius/lib/python2.7/site-packages


More detailed installation instructions and the list of required dependencies can
be found in the `Sirius readme`_.

.. _github: https://github.com/simonpintarelli/SIRIUS/tree/python27_i-PI
.. _Sirius Readme: https://github.com/simonpintarelli/SIRIUS/tree/python27_i-PI
