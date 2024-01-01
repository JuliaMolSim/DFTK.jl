#!/bin/sh
export OMP_NUM_THREADS=2
export CC4S_PATH=/opt/cc4s/stable/Cc4s
BASE=cc4s
mpirun -np 1 $CC4S_PATH --in $BASE.in.yaml --out $BASE.out.yaml --log $BASE.log
