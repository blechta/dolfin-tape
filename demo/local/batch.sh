#!/bin/bash
set -x -e

PREFIX="results"

# Prepare environment (FEniCS with DOLFIN fdc68f)
ml fenics/fenics-fdc68f
source ../../set_pythonpath.conf

mkdir -p $PREFIX

nohup $SHELL <<EOF > /dev/null
python test.py CarstensenKlose 4.0  5 2>&1 | tee $PREFIX/CK_4.0_05.log &
python test.py CarstensenKlose 4.0 10 2>&1 | tee $PREFIX/CK_4.0_10.log &
python test.py CarstensenKlose 4.0 15 2>&1 | tee $PREFIX/CK_4.0_15.log &
python test.py CarstensenKlose 4.0 20 2>&1 | tee $PREFIX/CK_4.0_20.log &
python test.py ChaillouSuri   10.0  5 2>&1 | tee $PREFIX/CS_10.0_05.log &
python test.py ChaillouSuri   10.0 10 2>&1 | tee $PREFIX/CS_10.0_10.log &
python test.py ChaillouSuri   10.0 15 2>&1 | tee $PREFIX/CS_10.0_15.log &
python test.py ChaillouSuri   10.0 20 2>&1 | tee $PREFIX/CS_10.0_20.log &
python test.py ChaillouSuri    1.5  5 2>&1 | tee $PREFIX/CS_1.5_05.log &
python test.py ChaillouSuri    1.5 10 2>&1 | tee $PREFIX/CS_1.5_10.log &
python test.py ChaillouSuri    1.5 15 2>&1 | tee $PREFIX/CS_1.5_15.log &
python test.py ChaillouSuri    1.5 20 2>&1 | tee $PREFIX/CS_1.5_20.log &
EOF
