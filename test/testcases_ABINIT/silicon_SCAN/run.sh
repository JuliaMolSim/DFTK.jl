#!/bin/bash
cd "$(dirname "$0")"
rm -f silicon_SCAN.log silicon_SCAN.abo __ABI_MPIABORTFILE__ *.nc
mpirun -np 4 abinit  silicon_SCAN.abi &> silicon_SCAN.log
