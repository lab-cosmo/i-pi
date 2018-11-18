ENV_BASE_DIR="$(realpath $(dirname $0))"

export PATH=$ENV_BASE_DIR/bin:$PATH
export PYTHONPATH=$ENV_BASE_DIR:$PYTHONPATH
export IPI_ROOT=$ENV_BASE_DIR

# SIRIUS: append path of Python 2.7 module
export PYTHONPATH=${SIRIUS_PYTHON2}:${PYTHONPATH}

unset ENV_BASE_DIR
